import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedTransformerLM(nn.Module):
    """
    Transformer encoder for masked unit prediction (BERT-style MLM).

    - inputs: integer unit IDs (HuBERT k-means tokens), shape (B, T)
    - outputs: logits over vocab, shape (B, T, vocab_size)

    You can:
        1) call `create_masked_input` to build (masked_input_ids, labels)
        2) pass masked_input_ids to forward() to get logits
        3) compute cross-entropy loss with labels (ignore_index for non-masked)
    """

    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=2048,
        pad_id=-100,
        mask_id=None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id
        self.mask_id = mask_id if mask_id is not None else vocab_size - 1

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, T, D)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: (B, T) LongTensor
        attention_mask: (B, T) with 1 for valid tokens, 0 for padding; if None, derived from pad_id.
        """
        bsz, seq_len = input_ids.shape

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).long()  # 1 for non-pad, 0 for pad

        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)

        # Transformer expects key_padding_mask: (B, T) with True for pads
        key_padding_mask = (attention_mask == 0)

        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        encoded = self.norm(encoded)

        logits = self.lm_head(encoded)  # (B, T, vocab_size)
        return logits

    def create_masked_input(
        self,
        input_ids,
        mask_prob=0.15,
        random_replace_prob=0.1,
        leave_unmasked_prob=0.1,
    ):
        """
        BERT-style masking:
            - choose ~mask_prob of non-pad tokens as "prediction targets"
            - of those:
                80% -> replaced with [MASK]
                10% -> replaced with random token
                10% -> left unchanged

        Returns:
            masked_input_ids: (B, T) LongTensor
            labels:          (B, T) LongTensor, with original token at masked positions,
                             and ignore_index elsewhere.
        """
        device = input_ids.device
        labels = input_ids.clone()

        # We do not predict pad positions
        is_pad = (input_ids == self.pad_id)
        probability_matrix = torch.full(input_ids.shape, mask_prob, device=device)
        probability_matrix.masked_fill_(is_pad, 0.0)

        # Sample mask positions
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # ignore_index for non-masked tokens

        # Now apply BERT masking strategy
        # 1) Replace with [MASK] 80% of the time
        masked_input_ids = input_ids.clone()

        indices_replaced = torch.bernoulli(
            torch.full(input_ids.shape, 1 - random_replace_prob - leave_unmasked_prob, device=device)
        ).bool() & masked_indices

        masked_input_ids[indices_replaced] = self.mask_id

        # 2) Replace with random token 10% of the time
        indices_random = torch.bernoulli(
            torch.full(input_ids.shape, random_replace_prob / (random_replace_prob + leave_unmasked_prob + 1e-8), device=device)
        ).bool() & masked_indices & ~indices_replaced

        random_tokens = torch.randint(low=0, high=self.vocab_size, size=input_ids.shape, device=device)
        masked_input_ids[indices_random] = random_tokens[indices_random]

        # 3) The rest (10%) are left unchanged → already unchanged in masked_input_ids

        return masked_input_ids, labels

    def compute_loss(self, logits, labels):
        """
        Convenience loss:
            logits: (B, T, V)
            labels: (B, T) with -100 as ignore_index
        """
        # flatten
        logits = logits.view(-1, self.vocab_size)
        labels = labels.view(-1)
        loss = F.cross_entropy(logits, labels, ignore_index=-100)
        return loss