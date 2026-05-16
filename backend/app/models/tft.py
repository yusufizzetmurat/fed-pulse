"""Temporal Fusion Transformer encoder (Lim et al., 2021).

A lightweight, dependency-free implementation of the time-series-encoder
portion of TFT: per-timestep Variable-Selection Network (VSN) over the six
input features, GRN-gated residual blocks, and a multi-head self-attention
block over the selected representation. Mirrors the public TFT recipe in
spirit without pulling ``pytorch-forecasting`` as a dependency — pure
PyTorch only.

The published TFT has a richer structure (LSTM encoder/decoder, static
covariate enrichment, multi-horizon quantile heads). We keep only the
encoder-side pieces that match the project's single-horizon, no-static-
metadata input contract; the head, time-decay, and credibility paths live
upstream in ``ForecasterModel``.

Input shape: ``(batch, seq_len, input_size)``. Output: ``(batch, seq_len,
hidden_size)`` plus a ``None`` placeholder so the forecaster's
``output, _ = core(x)`` destructuring keeps working unchanged. Pooling to
the head uses the same ``output[:, -1, :]`` step as the recurrent cores.

Determinism contract: the module owns no global RNG state. Same input +
same upstream ``torch.manual_seed`` ⇒ bit-identical output.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _GatedLinearUnit(nn.Module):
    """GLU: ``sigmoid(gate(x)) * proj(x)`` — TFT's elementwise gating."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_size, hidden_size)
        self.gate = nn.Linear(input_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.proj(x) * torch.sigmoid(self.gate(x))
        return out


class _GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) from TFT Section 3.

    ``GRN(a) = LayerNorm(a' + GLU(dense2(ELU(dense1(a)))))`` where ``a'`` is
    the input mapped to the GRN's output width via a skip connection (an
    identity when widths match, a Linear when they do not). Dropout sits
    between ``dense2`` and the GLU.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        out_size = int(output_size) if output_size is not None else int(hidden_size)
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = out_size
        self.skip: nn.Module = (
            nn.Identity() if self.input_size == self.output_size else nn.Linear(self.input_size, self.output_size)
        )
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.glu = _GatedLinearUnit(self.hidden_size, self.output_size)
        self.norm = nn.LayerNorm(self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Order follows Lim et al. (2021), Section 3.3, Eq. 3:
        # ELU(fc1) -> fc2 -> GLU -> dropout -> residual add -> LayerNorm.
        # Dropout is applied AFTER the gating, not before — the GLU
        # weights the signal first, then dropout zeroes a fraction of
        # the gated activations.
        residual = self.skip(x)
        h = F.elu(self.fc1(x))
        h = self.fc2(h)
        h = self.glu(h)
        h = self.dropout(h)
        out: torch.Tensor = self.norm(residual + h)
        return out


class _VariableSelectionNetwork(nn.Module):
    """Per-timestep VSN over scalar features.

    For each of the ``num_inputs`` scalar input features we keep a small
    per-feature GRN that lifts the scalar into ``hidden_size``. A separate
    GRN over the concatenated raw inputs then produces softmax weights
    across the features. The output at every timestep is the weighted sum
    of the per-feature embeddings.
    """

    def __init__(
        self,
        num_inputs: int,
        hidden_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_inputs < 1:
            raise ValueError(f"num_inputs must be >= 1, got {num_inputs}")
        self.num_inputs = int(num_inputs)
        self.hidden_size = int(hidden_size)
        self.feature_grns = nn.ModuleList(
            [
                _GatedResidualNetwork(
                    input_size=1, hidden_size=hidden_size, output_size=hidden_size, dropout=dropout
                )
                for _ in range(self.num_inputs)
            ]
        )
        self.weight_grn = _GatedResidualNetwork(
            input_size=self.num_inputs,
            hidden_size=hidden_size,
            output_size=self.num_inputs,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F). Per-feature GRN lift -> (B, T, F, H).
        if x.shape[-1] != self.num_inputs:
            raise ValueError(
                f"VSN built for num_inputs={self.num_inputs}, got input feature dim {x.shape[-1]}"
            )
        embeds = []
        for idx, grn in enumerate(self.feature_grns):
            f_i = x[..., idx : idx + 1]  # (B, T, 1)
            embeds.append(grn(f_i))  # (B, T, H)
        stacked = torch.stack(embeds, dim=-2)  # (B, T, F, H)
        weights_raw = self.weight_grn(x)  # (B, T, F)
        weights = F.softmax(weights_raw, dim=-1).unsqueeze(-1)  # (B, T, F, 1)
        combined = (stacked * weights).sum(dim=-2)  # (B, T, H)
        return combined, weights.squeeze(-1)


class _InterpretableMultiHeadAttention(nn.Module):
    """Multi-head self-attention used by the TFT encoder block.

    Standard scaled dot-product attention with ``n_heads`` and per-head
    width ``hidden_size // n_heads``. Wrapped here so the encoder stays a
    single ``nn.Module`` without importing ``nn.MultiheadAttention`` (which
    is fine but has a different output shape convention).
    """

    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by n_heads={n_heads}"
            )
        self.hidden_size = int(hidden_size)
        self.n_heads = int(n_heads)
        self.d_head = self.hidden_size // self.n_heads
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.query_proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = self.key_proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = self.value_proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head**0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, v)  # (B, H, T, d_head)
        ctx = ctx.transpose(1, 2).contiguous().view(b, t, self.hidden_size)
        out: torch.Tensor = self.out_proj(ctx)
        return out


class TFTEncoder(nn.Module):
    """Lightweight TFT encoder matching the project's recurrent-core contract.

    Pipeline:
        VSN over the raw scalar features
        → GRN gated residual block (post-VSN enrichment)
        → multi-head self-attention with residual + LayerNorm
        → GRN gated feed-forward with residual + LayerNorm.

    Input ``(B, T, input_size)`` → output ``(B, T, hidden_size)``. Returns
    ``(output, None)`` so the forecaster's ``output, _ = self.lstm(x)``
    destructuring keeps working unchanged.

    Defaults follow Lim et al. 2021 for small-budget settings:
    ``hidden_size=64``, ``n_heads=4``, ``dropout=0.1``. ``hidden_size``
    must be divisible by ``n_heads``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.vsn = _VariableSelectionNetwork(
            num_inputs=self.input_size, hidden_size=hidden_size, dropout=dropout
        )
        self.post_vsn_grn = _GatedResidualNetwork(
            input_size=hidden_size, hidden_size=hidden_size, output_size=hidden_size, dropout=dropout
        )
        self.attn = _InterpretableMultiHeadAttention(
            hidden_size=hidden_size, n_heads=n_heads, dropout=dropout
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.ff_grn = _GatedResidualNetwork(
            input_size=hidden_size, hidden_size=hidden_size, output_size=hidden_size, dropout=dropout
        )
        self.ff_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        if x.dim() != 3:
            raise ValueError(f"TFTEncoder expects (B, T, F); got {tuple(x.shape)}")
        h, _weights = self.vsn(x)  # (B, T, H)
        h = self.post_vsn_grn(h)
        attn_out = self.attn(h)
        h = self.attn_norm(h + self.attn_dropout(attn_out))
        ff_out = self.ff_grn(h)
        h = self.ff_norm(h + ff_out)
        return h, None


__all__ = ["TFTEncoder"]
