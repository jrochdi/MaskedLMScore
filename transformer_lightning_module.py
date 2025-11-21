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

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)

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

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True, batch_size=input_ids.size(0))
        self.log(f"{stage}_acc", acc, prog_bar=(stage != "test"), on_step=False, on_epoch=True, batch_size=input_ids.size(0))
        self.log(f"{stage}_ppl", ppl, prog_bar=False, on_step=False, on_epoch=True, batch_size=input_ids.size(0))

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self._step(batch, stage="test")

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer