import torch
import torch.nn as nn


class ImageConditionedTransformerDecoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
        max_seq_len: int,
        dropout: float,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        self.image_projection = nn.Linear(2048, self.d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_layers,
        )

    def forward(
        self,
        img_feats: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):

        batch_size, seq_len = input_ids.shape

        token_emb = self.token_embedding(input_ids) * (self.d_model ** 0.5)
        positions = torch.arange(0, seq_len, device=input_ids.device)

        position_emb = self.positional_embedding(positions)

        embedding = token_emb + position_emb

        img_proj = self.image_projection(img_feats)

        embedding[:, 0, :] = img_proj

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(
            device=embedding.device
        )

        if attention_mask is not None:

            tgt_key_padding_mask = attention_mask == 0

        else:
            tgt_key_padding_mask = None

        encoder_output = self.transformer(
            src=embedding, mask=tgt_mask, src_key_padding_mask=tgt_key_padding_mask
        )

        logits = self.lm_head(encoder_output)

        return logits
