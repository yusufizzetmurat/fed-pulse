from __future__ import annotations

import torch
from torch import nn


class TemporalConvNet(nn.Module):
    """Tiny TCN: two dilated 1D conv blocks with residual connections.

    Input shape: [batch, seq_len, input_size]. Output: [batch, seq_len, hidden_size].
    Causal padding via cropping so seq_len is preserved without leaking future
    information. Returns (output, None) so the forward pass's
    `output, _ = self.lstm(x)` destructuring works as for LSTM/GRU.
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        # Causal padding: pad left by (kernel - 1) * dilation, then crop to seq_len.
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=2,  # (3 - 1) * dilation=1
            dilation=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=4,  # (3 - 1) * dilation=2
            dilation=2,
        )
        self.residual: nn.Module = (
            nn.Conv1d(input_size, hidden_size, kernel_size=1)
            if input_size != hidden_size
            else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        # x: [batch, seq, feat] -> [batch, feat, seq] for Conv1d
        seq_len = x.shape[1]
        x_t = x.transpose(1, 2)
        h = self.conv1(x_t)[:, :, :seq_len]  # crop right (causal)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)[:, :, :seq_len]
        h = self.activation(h)
        h = self.dropout(h)
        if isinstance(self.residual, nn.Identity):
            residual = x_t
        else:
            residual = self.residual(x_t)[:, :, :seq_len]
        out = h + residual
        # Back to [batch, seq, feat]
        return out.transpose(1, 2), None
