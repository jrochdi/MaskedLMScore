import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW

from transformer import MaskedTransformerLM


class MaskedTransformerModule(pl.LightningModule):
    """
    Lightning wrapper around MaskedTransformerLM for masked unit prediction.

    Expects batches from UnitDataModule like:
        batch["units"]:   (B, T) LongTensor of unit IDs
        batch["lengths"]: (B,)
    """

    def __init__(
        self,
        vocab_size,
        pad_id=-100,
        mask_id=None,
        d_model=512,
        n_heads=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=2048,
        lr=1e-4,
        weight_decay=0.01,
        pll_max_utterances=None,   # <--- NEW: max utterances per epoch per stage for PLL
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MaskedTransformerLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            pad_id=pad_id,
            mask_id=mask_id,
        )

        self.pad_id = pad_id
        self.lr = lr
        self.weight_decay = weight_decay

        # PLL budget
        self.pll_max_utterances = pll_max_utterances  # None or int
        self._pll_counts = {"val": 0, "gib": 0}       # reset each val epoch

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)

    # ---------------------------
    # Core MLM step (unchanged)
    # ---------------------------
    def _step(self, batch, stage):
        """
        Common code for train/val/test step.
        """
        input_ids = batch["units"]          # (B, T)
        masked_input, labels = self.model.create_masked_input(input_ids)

        logits = self.model(masked_input)   # (B, T, V)
        loss = self.model.compute_loss(logits, labels)

        # masked positions
        with torch.no_grad():
            masked_positions = labels != -100
            num_masked = masked_positions.sum()

            if num_masked > 0:
                preds = logits.argmax(dim=-1)
                correct = (preds == labels) & masked_positions
                acc = correct.sum().float() / num_masked.float()
            else:
                acc = torch.tensor(0.0, device=logits.device)

            ppl = torch.exp(loss.detach())

        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
            batch_size=input_ids.size(0),
        )
        self.log(
            f"{stage}_acc",
            acc,
            prog_bar=(stage != "test"),
            on_step=False,
            on_epoch=True,
            batch_size=input_ids.size(0),
        )
        self.log(
            f"{stage}_ppl",
            ppl,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=input_ids.size(0),
        )

        return loss

    # ---------------------------
    # Pseudo log-likelihood (PLL)
    # ---------------------------
    @torch.no_grad()
    def compute_batch_pll(self, input_ids):
        """
        Compute pseudo-log-likelihood for a batch of sequences.

        For each non-pad token position i:
            - mask token i with [MASK]
            - run model
            - accumulate log p(x_i | x_{-i})

        Returns:
            total_logprob (scalar tensor, sum over batch)
            total_tokens  (scalar tensor, number of non-pad tokens)
        """
        device = input_ids.device
        B, T = input_ids.shape

        pad_mask = (input_ids == self.pad_id)  # True where pad
        total_logprob = torch.zeros((), device=device)
        total_tokens = torch.zeros((), device=device)

        for pos in range(T):
            # consider only sequences where this position is not pad
            valid_b = ~pad_mask[:, pos]
            if not valid_b.any():
                continue

            # mask position `pos` for those valid sequences
            masked = input_ids.clone()
            masked[valid_b, pos] = self.model.mask_id

            logits = self.model(masked)  # (B, T, V)
            log_probs = torch.log_softmax(logits[valid_b, pos, :], dim=-1)
            targets = input_ids[valid_b, pos]

            pos_logprob = log_probs[
                torch.arange(log_probs.size(0), device=device), targets
            ].sum()

            total_logprob += pos_logprob
            total_tokens += valid_b.sum()

        return total_logprob, total_tokens

    # ---------------------------
    # Lightning hooks
    # ---------------------------
    def on_validation_epoch_start(self):
        # reset PLL utterance counts per stage each val epoch
        self._pll_counts = {"val": 0, "gib": 0}

    def training_step(self, batch, batch_idx):
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # dataloader_idx 0 -> clean val, 1 -> gibberish val (if present)
        if dataloader_idx == 0:
            stage = "val"   # keep 'val_loss' for early stopping/checkpoints
        else:
            stage = "gib"   # logs 'gib_loss', 'gib_acc', 'gib_ppl'

        # 1) Standard MLM validation metrics
        self._step(batch, stage=stage)

        # 2) PLL-based pseudo-perplexity on a limited number of utterances
        if self.pll_max_utterances is None or self.pll_max_utterances <= 0:
            return  # PLL disabled

        # How many utterances can we still use for PLL this epoch, for this stage?
        used = self._pll_counts.get(stage, 0)
        remaining = self.pll_max_utterances - used
        if remaining <= 0:
            return  # budget exhausted

        input_ids = batch["units"]  # (B, T)
        batch_size = input_ids.size(0)

        if batch_size > remaining:
            # only use the first `remaining` utterances
            input_ids = input_ids[:remaining]
            batch_size = remaining

        total_logprob, total_tokens = self.compute_batch_pll(input_ids)

        if total_tokens > 0:
            avg_logprob = total_logprob / total_tokens
            pll_ppl = torch.exp(-avg_logprob)  # pseudo-perplexity
        else:
            avg_logprob = torch.tensor(float("nan"), device=input_ids.device)
            pll_ppl = torch.tensor(float("nan"), device=input_ids.device)

        # update budget
        self._pll_counts[stage] = used + batch_size

        # Log PLL metrics per stage (val vs gib)
        self.log(
            f"{stage}_pll_logprob",
            avg_logprob,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}_pll_ppl",
            pll_ppl,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

    def test_step(self, batch, batch_idx):
        self._step(batch, stage="test")

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer