from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Config:
    vocab_size: int
    context: int = 128
    width: int = 128
    layers: int = 4
    heads: int = 4
    dropout: float = 0.1


class CausalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.context, config.width)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                config.width, config.heads, 4 * config.width,
                config.dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(config.layers)
        ])
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab_size)
        self.register_buffer(
            "mask", torch.ones(config.context, config.context, dtype=torch.bool).triu(1),
            persistent=False,
        )

    def forward(self, tokens):
        length = tokens.size(1)
        x = self.token(tokens) + self.position(torch.arange(length, device=tokens.device))
        for block in self.blocks:
            x = block(x, src_mask=self.mask[:length, :length])
        return self.head(self.norm(x))
