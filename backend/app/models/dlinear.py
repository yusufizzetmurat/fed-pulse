"""DLinear baseline (Zeng et al. 2023, "Are Transformers Effective for Time
Series Forecasting?").

Decomposes the input into a moving-average trend and a residual seasonal
component, projects each through its own Linear layer, then sums. Despite
the simplicity, DLinear consistently matches or beats Transformer-style
architectures on financial time series in recent benchmarks; including it
gives the bake-off a strong non-recurrent control.

Input contract matches LSTM/TCN: ``(B, T, F)``. Output: ``(B, T, hidden)``
plus a ``None`` placeholder so the forecaster's ``output, _ = core(x)``
destructuring keeps working unchanged. Pooling to the head uses the same
``output[:, -1, :]`` step as the recurrent cores.
"""

from __future__ import annotations

import torch
from torch import nn


class _SeriesDecomposition(nn.Module):
    """Trend = moving average over a centred window; seasonal = input − trend."""

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        self.kernel_size = int(kernel_size)
        # AvgPool1d with stride=1 and replicate padding: trend has the same
        # length as input. Use F.pad with mode="replicate" for boundary
        # handling so DLinear matches the published recipe.

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F) — pool along T
        x_t = x.transpose(1, 2)  # (B, F, T)
        pad = (self.kernel_size - 1) // 2
        # Symmetric replicate padding so the trend keeps the original length.
        padded = nn.functional.pad(x_t, (pad, pad), mode="replicate")
        trend = nn.functional.avg_pool1d(
            padded, kernel_size=self.kernel_size, stride=1, padding=0
        )
        trend = trend.transpose(1, 2)  # (B, T, F)
        seasonal = x - trend
        return trend, seasonal


class DLinear(nn.Module):
    """DLinear core. Two per-feature linear maps, one for trend, one for seasonal.

    The published DLinear uses ``Linear(T, H)`` per channel (i.e., shared
    weights across features). We mirror that by stacking the input as
    ``(B*F, T)`` for each linear application, then reshaping back.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        sequence_length: int,
        decomposition_kernel: int = 5,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.sequence_length = int(sequence_length)
        self.decomposition = _SeriesDecomposition(kernel_size=decomposition_kernel)
        # DLinear's two heads operate on the time dimension. Output H lets us
        # plug into the existing forecaster head which expects (B, T, H).
        self.trend_linear = nn.Linear(self.sequence_length, self.sequence_length)
        self.seasonal_linear = nn.Linear(self.sequence_length, self.sequence_length)
        # Project the F input channels into hidden_size on the final pool step
        # so the downstream head sees the same dim as the recurrent variants.
        self.feature_proj = nn.Linear(self.input_size, self.hidden_size)
        # Zero-init like the chunk projection elsewhere — keeps DLinear close
        # to a no-op at step 0 so the cross-arch comparison starts from a
        # neutral baseline rather than a different random init.
        nn.init.zeros_(self.trend_linear.weight)
        nn.init.zeros_(self.trend_linear.bias)
        nn.init.zeros_(self.seasonal_linear.weight)
        nn.init.zeros_(self.seasonal_linear.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        if x.dim() != 3:
            raise ValueError(f"DLinear expects (B, T, F); got {tuple(x.shape)}")
        batch_size, seq_len, feat = x.shape
        if seq_len != self.sequence_length:
            raise ValueError(
                f"DLinear was built for seq_len={self.sequence_length}, got {seq_len}"
            )
        trend, seasonal = self.decomposition(x)  # both (B, T, F)
        # Per-feature linear over the time axis: reshape to (B*F, T), apply
        # linear, reshape back. The shared weights across features mirror the
        # published DLinear recipe.
        def _apply(linear: nn.Linear, tensor: torch.Tensor) -> torch.Tensor:
            return (
                linear(tensor.transpose(1, 2).reshape(batch_size * feat, seq_len))
                .reshape(batch_size, feat, seq_len)
                .transpose(1, 2)
            )

        trend_out = _apply(self.trend_linear, trend)
        seasonal_out = _apply(self.seasonal_linear, seasonal)
        summed = trend_out + seasonal_out  # (B, T, F)
        projected = self.feature_proj(summed)  # (B, T, hidden)
        return projected, None
