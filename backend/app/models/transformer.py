from __future__ import annotations

import math

import torch
from torch import nn


class _SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequences up to max_len."""

    def __init__(self, hidden_size: int, max_len: int = 32) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size)
        )
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = self.pe
        assert isinstance(pe, torch.Tensor)
        return x + pe[:, : x.size(1), :]


class SmallTransformer(nn.Module):
    """Small Transformer encoder with sinusoidal positional encoding.

    Two layers, four heads, batch_first. Output shape matches LSTM/GRU/TCN:
    [batch, seq_len, hidden_size]. Returns (output, None) so the forward
    pass's `output, _ = self.lstm(x)` destructuring continues to work.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
            )
        self.input_proj = (
            nn.Linear(input_size, hidden_size)
            if input_size != hidden_size
            else nn.Identity()
        )
        self.pos = _SinusoidalPositionalEncoding(hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        h = self.input_proj(x)
        h = self.pos(h)
        out = self.encoder(h)
        return out, None
