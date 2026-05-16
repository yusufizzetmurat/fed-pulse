"""Informer encoder core (Zhou et al., AAAI 2021).

A compact, dependency-free implementation of the encoder side of Informer:
ProbSparse self-attention plus the standard transformer-style feed-forward
block, stacked across ``e_layers`` encoder layers and wrapped in a fixed
sinusoidal positional embedding. Mirrors the public-good Informer reference
(``zhouhaoyi/Informer2020``) without pulling that repository or
``pytorch-forecasting`` as a dependency — pure PyTorch only.

Input shape: ``(batch, seq_len, input_size)``. Output: ``(batch, seq_len,
hidden_size)`` plus a ``None`` placeholder so the forecaster's
``output, _ = core(x)`` destructuring keeps working unchanged. Pooling to
the head uses the same ``output[:, -1, :]`` step as the recurrent cores.

The ProbSparse attention reduces full self-attention's ``O(L^2)`` cost to
``O(L log L)`` by sampling a constant number of dot-product probes per
query and only routing the top-``u`` queries (by KL-vs-uniform score)
through the full softmax. For ``L=20`` the asymptotic win is small, but the
implementation is functionally identical to the published one and keeps
the encoder a drop-in for the longer-horizon sweeps the Phase-8 wiki
sketches.

Determinism contract: the module owns no global RNG state. Caller-side
``torch.manual_seed`` is honored because every stochastic choice (the
sampled probe indices) is derived from ``torch.randint`` against the
current default generator.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F


class _SinusoidalPositionalEmbedding(nn.Module):
    """Standard sinusoidal positional embedding shared by the encoder layers."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `register_buffer` returns None; mypy needs the explicit cast
        # because `nn.Module.__getattr__` is typed as `Tensor | Module`.
        pe = cast(torch.Tensor, self.pe)
        return x + pe[:, : x.size(1), :]


class _ProbSparseAttention(nn.Module):
    """ProbSparse self-attention from Informer (Zhou et al. 2021).

    For each query the sparsity score is approximated by sampling ``U_part``
    keys at random and computing the max-vs-mean gap of the partial dot
    products. The top ``u`` queries by that score are then routed through a
    real softmax over the full key set; the remaining queries are replaced
    by the mean of the value sequence, which is the standard "lazy"
    fallback from the paper.
    """

    def __init__(self, factor: int = 5, dropout: float = 0.1, sample_seed: int = 11) -> None:
        super().__init__()
        if factor < 1:
            raise ValueError(f"factor must be >= 1, got {factor}")
        self.factor = int(factor)
        self.dropout = nn.Dropout(dropout)
        # Per-instance sampling generator so the ProbSparse sampling
        # inside forward() does not advance the global RNG. The module
        # claims to own no global RNG state; the generator is what makes
        # that claim true and keeps callers' torch.manual_seed unpolluted
        # across forward passes.
        self._sample_generator = torch.Generator(device="cpu")
        self._sample_generator.manual_seed(int(sample_seed))

    def _prob_qk(
        self, queries: torch.Tensor, keys: torch.Tensor, sample_k: int, n_top: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # queries: (B, H, L_Q, D), keys: (B, H, L_K, D)
        b, h, l_k, d = keys.shape
        _, _, l_q, _ = queries.shape

        # Sample sample_k key positions per (B, H, L_Q) triple. The
        # generator is per-instance so the global RNG state of the
        # caller is preserved across forward passes.
        index_sample = torch.randint(
            l_k, (l_q, sample_k), generator=self._sample_generator
        ).to(keys.device)
        # Index into K: (B, H, L_Q, sample_k, D)
        k_expand = keys.unsqueeze(-3).expand(b, h, l_q, l_k, d)
        k_sample = k_expand[:, :, torch.arange(l_q).unsqueeze(1), index_sample, :]

        # Partial dot products against sampled keys: (B, H, L_Q, sample_k)
        q_k_sample = torch.matmul(queries.unsqueeze(-2), k_sample.transpose(-2, -1)).squeeze(-2)

        # Sparsity score: max - mean across sampled keys. Top n_top queries win.
        m = q_k_sample.max(dim=-1).values - q_k_sample.mean(dim=-1)
        m_top = m.topk(n_top, sorted=False).indices  # (B, H, n_top)

        # Real (full) softmax for the winning queries only.
        q_reduced = queries[
            torch.arange(b)[:, None, None],
            torch.arange(h)[None, :, None],
            m_top,
            :,
        ]
        q_k = torch.matmul(q_reduced, keys.transpose(-2, -1))
        return q_k, m_top

    def _get_initial_context(
        self, values: torch.Tensor, l_q: int
    ) -> torch.Tensor:
        # Average of V along the key axis, broadcast to query positions.
        v_mean = values.mean(dim=-2)  # (B, H, D)
        context = v_mean.unsqueeze(-2).expand(*values.shape[:-2], l_q, values.shape[-1])
        return context.clone()

    def _update_context(
        self,
        context_in: torch.Tensor,
        values: torch.Tensor,
        scores: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        b, h, _, d = values.shape
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context_top = torch.matmul(attn, values)  # (B, H, n_top, D)
        context_in[
            torch.arange(b)[:, None, None],
            torch.arange(h)[None, :, None],
            index,
            :,
        ] = context_top
        return context_in

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        # queries/keys/values: (B, L, H, D) — match the public reference layout.
        b, l_q, h, d = queries.shape
        _, l_k, _, _ = keys.shape

        # Move heads forward: (B, H, L, D)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Cap probe / top-k counts to the actual sequence length. For L=20
        # this typically yields U_part ≈ 15, u ≈ 15 — i.e., almost-dense
        # attention. The same code degrades gracefully to dense as L grows.
        u_part = min(self.factor * max(1, int(math.ceil(math.log(max(l_k, 2))))), l_k)
        u = min(self.factor * max(1, int(math.ceil(math.log(max(l_q, 2))))), l_q)

        scores_top, index = self._prob_qk(queries, keys, sample_k=u_part, n_top=u)
        scale = 1.0 / math.sqrt(d)
        scores_top = scores_top * scale

        context = self._get_initial_context(values, l_q)
        context = self._update_context(context, values, scores_top, index)

        # Restore (B, L, H, D)
        return context.transpose(1, 2).contiguous()


class _AttentionLayer(nn.Module):
    """Multi-head wrapper around ProbSparse attention."""

    def __init__(
        self, d_model: int, n_heads: int, factor: int = 5, dropout: float = 0.1
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        self.d_keys = d_model // n_heads
        self.n_heads = int(n_heads)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.inner = _ProbSparseAttention(factor=factor, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, _ = x.shape
        q = self.query_proj(x).view(b, seq_len, self.n_heads, self.d_keys)
        k = self.key_proj(x).view(b, seq_len, self.n_heads, self.d_keys)
        v = self.value_proj(x).view(b, seq_len, self.n_heads, self.d_keys)
        out = self.inner(q, k, v)  # (B, L, H, D)
        out = out.view(b, seq_len, self.n_heads * self.d_keys)
        projected: torch.Tensor = self.out_proj(out)
        return projected


class _EncoderLayer(nn.Module):
    """Attention + position-wise feed-forward, with residual + LayerNorm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        factor: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = _AttentionLayer(d_model, n_heads, factor=factor, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = x + self.dropout(self.attn(self.norm1(x)))
        h = h + self.dropout(self.ff(self.norm2(h)))
        return h


class InformerEncoder(nn.Module):
    """Informer encoder core matching the project's recurrent-core contract.

    Input ``(B, T, input_size)`` → output ``(B, T, hidden_size)``. Returns
    ``(output, None)`` so the forecaster's ``output, _ = self.lstm(x)``
    destructuring keeps working unchanged.

    Defaults follow the AAAI-2021 paper for short-horizon settings:
    ``d_model=64``, ``n_heads=4``, ``e_layers=2``, ``dropout=0.1``,
    ``factor=5``. ``d_model`` is taken from the forecaster's
    ``hidden_size`` so the head can pool the encoder output without an
    extra projection.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        dropout: float = 0.1,
        factor: int = 5,
        d_ff: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by n_heads={n_heads}"
            )
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.input_proj = (
            nn.Linear(input_size, hidden_size)
            if input_size != hidden_size
            else nn.Identity()
        )
        self.pos = _SinusoidalPositionalEmbedding(hidden_size)
        self.embed_dropout = nn.Dropout(dropout)
        d_ff_eff = int(d_ff) if d_ff is not None else hidden_size * 2
        self.layers = nn.ModuleList(
            [
                _EncoderLayer(
                    d_model=hidden_size,
                    n_heads=n_heads,
                    d_ff=d_ff_eff,
                    factor=factor,
                    dropout=dropout,
                )
                for _ in range(e_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        if x.dim() != 3:
            raise ValueError(f"InformerEncoder expects (B, T, F); got {tuple(x.shape)}")
        h = self.input_proj(x)
        h = self.pos(h)
        h = self.embed_dropout(h)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return h, None


__all__ = ["InformerEncoder"]
