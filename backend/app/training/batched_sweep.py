"""Bucketed hyperparameter sweep for the forecaster.

Forecasters in this project are TINY (hidden in {32, 64, 128}, layers in
{1, 2, 3}, sequence length 20). Per-cell forward + backward is a
handful of CUDA kernels that finishes before the next kernel is even
dispatched, so the GPU sits at ~25% TGP when each cell runs in its
own process. The bucketed sweep groups cells that share the same
model topology and data feed and runs them concurrently inside one
Python process, either by stacking the per-cell parameters along a
synthetic batch axis (``stacked`` mode) or by overlapping cell-level
kernel launches across CUDA streams (``streams`` mode).

The bucket key is::

    (architecture, hidden_size, num_layers, text_adapter_dim,
     text_encoder, fold_id, target_mode)

Cells inside a bucket differ only on ``(dropout, learning_rate,
weight_decay, seed)`` -- the four within-bucket axes the design treats
as stochastic / per-trial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn

# Per-architecture VRAM budget cap for the bucket size. Smaller
# architectures (dlinear, lstm, gru, tcn) absorb more cells per
# bucket; heavier architectures (transformer, informer, tft) stay
# small so the CUDA allocator has headroom on a 16 GB card.
DEFAULT_MAX_BUCKET_SIZE_BY_ARCH: dict[str, int] = {
    "lstm": 32,
    "lstm_attn": 16,
    "gru": 32,
    "tcn": 32,
    "transformer": 8,
    "dlinear": 64,
    "informer": 4,
    "tft": 4,
}


@dataclass(frozen=True)
class BucketKey:
    """Tuple of axes that fix a bucket's model topology + data feed.

    Cells with the same ``BucketKey`` share the same model
    architecture, the same parameter shapes, and the same training
    tensors; only the per-cell stochastic axes (dropout, learning
    rate, weight decay, seed) differ inside the bucket.
    """

    architecture: str
    hidden_size: int
    num_layers: int
    text_adapter_dim: int
    text_encoder: str
    fold_id: str | None
    target_mode: str


def bucket_key_for_candidate(
    candidate: dict[str, Any],
    *,
    text_encoder: str | None,
    target_mode: str,
) -> BucketKey:
    """Compute the ``BucketKey`` for a single sweep candidate.

    ``text_encoder`` and ``target_mode`` are sweep-wide knobs, not
    per-candidate axes -- they're supplied separately so the bucket
    key still distinguishes the no-text vs FinBERT vs voyage runs
    when those are split across separate sweep invocations.
    """

    model_config = candidate["model_config"]
    return BucketKey(
        architecture=str(model_config.architecture),
        hidden_size=int(model_config.hidden_size),
        num_layers=int(model_config.num_layers),
        text_adapter_dim=int(getattr(model_config, "text_adapter_dim", 0) or 0),
        text_encoder=str(text_encoder or "none"),
        fold_id=candidate.get("fold_id"),
        target_mode=str(target_mode),
    )


def resolve_max_bucket_size(
    architecture: str,
    *,
    override: int | None = None,
) -> int:
    """Pick the per-architecture bucket size cap.

    ``override`` (the CLI ``--max-bucket-size`` value) takes
    precedence; otherwise the per-arch default table is consulted with
    a conservative fallback of 4 cells for unknown architectures.
    """

    if override is not None and int(override) > 0:
        return int(override)
    return int(DEFAULT_MAX_BUCKET_SIZE_BY_ARCH.get(architecture, 4))


def group_candidates_into_buckets(
    candidates: Sequence[dict[str, Any]],
    *,
    text_encoder: str | None,
    target_mode: str,
    max_bucket_size: int | None = None,
) -> list[tuple[BucketKey, list[tuple[int, dict[str, Any]]]]]:
    """Partition sweep candidates into ``BucketKey``-keyed buckets.

    The returned list preserves the input candidate order across
    bucket boundaries: the first cell of bucket *i* comes before the
    first cell of bucket *i+1* by the original sweep candidate order.
    Each candidate is paired with its 1-based ``trial_index`` so the
    downstream trial-record emitter keeps the original numbering
    contract.

    When ``max_bucket_size`` (or the per-arch default) is exceeded by
    the cells assigned to a single ``BucketKey``, the bucket is split
    into chunks of at most ``max_bucket_size`` cells. The split is
    deterministic: cells stay in their original sweep order.
    """

    grouped: dict[BucketKey, list[tuple[int, dict[str, Any]]]] = {}
    first_seen: dict[BucketKey, int] = {}
    for index, candidate in enumerate(candidates, start=1):
        key = bucket_key_for_candidate(
            candidate, text_encoder=text_encoder, target_mode=target_mode
        )
        grouped.setdefault(key, []).append((index, candidate))
        if key not in first_seen:
            first_seen[key] = index

    ordered_keys = sorted(grouped.keys(), key=lambda k: first_seen[k])
    output: list[tuple[BucketKey, list[tuple[int, dict[str, Any]]]]] = []
    for key in ordered_keys:
        cells = grouped[key]
        cap = resolve_max_bucket_size(
            key.architecture, override=max_bucket_size
        )
        if cap <= 0 or len(cells) <= cap:
            output.append((key, cells))
            continue
        for start in range(0, len(cells), cap):
            output.append((key, cells[start : start + cap]))
    return output


class BatchedDropout(nn.Module):
    """Dropout layer with a per-cell ``p`` tensor.

    The forward expects an input of shape ``(N, ...)`` where ``N`` is
    the bucket size. ``p`` is a length-``N`` tensor of dropout
    probabilities (one per stacked cell), and the mask is drawn from a
    per-cell Bernoulli at training time. ``eval()`` makes the layer a
    no-op, mirroring ``nn.Dropout``'s training-vs-eval semantics.

    The mask is broadcast over every non-leading axis of the input so
    the same module can sit inside an LSTM head (input shape
    ``(N, batch, features)``) or after a Conv1d block
    (input shape ``(N, batch, channels, time)``) without per-call
    reshaping. ``generator`` is an optional ``torch.Generator`` so the
    bucket-level RNG produces reproducible masks at a fixed seed.

    Inverted dropout (the standard PyTorch convention) divides the
    kept activations by the per-cell keep probability so the expected
    activation at training time matches the eval-time pass-through.
    """

    def __init__(self, p: torch.Tensor):
        super().__init__()
        if p.ndim != 1:
            raise ValueError(
                f"BatchedDropout p must be 1-D (per-cell); got shape {tuple(p.shape)}"
            )
        if torch.any(p < 0) or torch.any(p > 1):
            raise ValueError("BatchedDropout p values must lie in [0, 1]")
        # Register as a non-trainable buffer so the module follows its
        # parent to the right device and dtype on .to(...).
        self.register_buffer("p", p.clone().detach(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if not self.training:
            return x
        if x.shape[0] != self.p.shape[0]:
            raise ValueError(
                f"BatchedDropout received input with leading dim {x.shape[0]} "
                f"but p has length {self.p.shape[0]}"
            )
        # Draw a per-cell, per-element Bernoulli with cell-specific keep
        # prob (1 - p_i). The scale factor 1 / (1 - p_i) compensates so
        # the expected activation is preserved (inverted dropout).
        keep_prob = (1.0 - self.p).clamp(min=1e-12, max=1.0).to(dtype=x.dtype, device=x.device)
        # Broadcast keep_prob to the input's full shape: (N, ...).
        broadcast_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        keep_view = keep_prob.view(broadcast_shape)
        # Uniform draw in [0, 1); keep where u < keep_prob.
        if generator is not None:
            u = torch.rand(
                x.shape, dtype=x.dtype, device=x.device, generator=generator
            )
        else:
            u = torch.rand(x.shape, dtype=x.dtype, device=x.device)
        mask = (u < keep_view).to(x.dtype)
        # Inverted dropout: divide by keep prob so the post-dropout mean
        # matches the input's mean per cell.
        return x * mask / keep_view


class StackedDLinear(nn.Module):
    """Vmap-friendly DLinear stack for stacked-mode bucket training.

    DLinear (Zeng et al., AAAI 2023) is a pure linear baseline that
    decomposes the input into a moving-average trend and a residual,
    then linearly projects each component along the time axis to the
    forecast horizon. The recurrent forecaster wraps it for the
    20-bar prior window; this stacked variant carries one set of
    weights per bucket cell along a synthetic leading axis so a
    single matmul handles the whole bucket.

    The forward expects ``x`` of shape ``(N, batch, seq_len, features)``
    where ``N`` is the bucket size; the output has shape
    ``(N, batch, 2)`` -- one (close, volatility) pair per cell per
    batch element. The head is a single Linear-GELU-Linear projection
    with a BatchedDropout in the middle so the per-cell dropout
    schedule is honoured.
    """

    def __init__(
        self,
        *,
        bucket_size: int,
        input_features: int,
        sequence_length: int,
        hidden_size: int,
        head_hidden_size: int,
        dropout_p: torch.Tensor,
    ):
        super().__init__()
        if dropout_p.shape[0] != bucket_size:
            raise ValueError(
                "StackedDLinear dropout_p length must equal bucket size; "
                f"got {dropout_p.shape[0]} vs {bucket_size}"
            )
        self.bucket_size = int(bucket_size)
        self.input_features = int(input_features)
        self.sequence_length = int(sequence_length)
        self.hidden_size = int(hidden_size)
        self.head_hidden_size = int(head_hidden_size)
        # Per-cell parameters as stacked tensors with shape
        # (N, *param_shape). The first three carry the DLinear
        # trend + residual mix; the last two are the (close,
        # volatility) projection head.
        scale = 1.0 / (input_features * sequence_length) ** 0.5
        self.trend_weight = nn.Parameter(
            (torch.rand(bucket_size, hidden_size, input_features * sequence_length) - 0.5)
            * 2.0
            * scale
        )
        self.trend_bias = nn.Parameter(
            torch.zeros(bucket_size, hidden_size)
        )
        self.residual_weight = nn.Parameter(
            (torch.rand(bucket_size, hidden_size, input_features * sequence_length) - 0.5)
            * 2.0
            * scale
        )
        self.residual_bias = nn.Parameter(
            torch.zeros(bucket_size, hidden_size)
        )
        head_scale = 1.0 / hidden_size**0.5
        self.head_weight = nn.Parameter(
            (torch.rand(bucket_size, head_hidden_size, hidden_size) - 0.5)
            * 2.0
            * head_scale
        )
        self.head_bias = nn.Parameter(torch.zeros(bucket_size, head_hidden_size))
        out_scale = 1.0 / head_hidden_size**0.5
        self.out_weight = nn.Parameter(
            (torch.rand(bucket_size, 2, head_hidden_size) - 0.5) * 2.0 * out_scale
        )
        self.out_bias = nn.Parameter(torch.zeros(bucket_size, 2))
        # Per-cell dropout between the head's GELU and the output
        # projection. The dropout schedule comes from the bucket's
        # per-cell HP table.
        self.dropout = BatchedDropout(dropout_p)

    @staticmethod
    def _decompose(x: torch.Tensor, *, kernel_size: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        """Trend + residual decomposition along the time axis.

        ``x`` has shape ``(N, batch, seq_len, features)``. The trend is
        a moving average over the time axis with reflective padding so
        the trend and residual share the same length; the residual is
        ``x - trend``.
        """

        # Move the time axis to the second-last position for the
        # avg_pool1d call. Shape: (N*batch*features, 1, seq_len).
        n, b, t, f = x.shape
        # Pool along T per (cell, batch, feature). Flatten leading
        # axes for the conv kernel.
        x_flat = x.permute(0, 1, 3, 2).reshape(n * b * f, 1, t)
        padding = kernel_size // 2
        # Reflective pad to keep the output length equal to T.
        x_padded = nn.functional.pad(x_flat, (padding, padding), mode="reflect")
        trend_flat = nn.functional.avg_pool1d(
            x_padded, kernel_size=kernel_size, stride=1, padding=0
        )
        trend = trend_flat.reshape(n, b, f, t).permute(0, 1, 3, 2)
        return trend, x - trend

    def forward(
        self,
        x: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"StackedDLinear expects (N, batch, seq, features); got {tuple(x.shape)}"
            )
        if x.shape[0] != self.bucket_size:
            raise ValueError(
                f"StackedDLinear input has leading dim {x.shape[0]}; "
                f"expected bucket_size={self.bucket_size}"
            )
        trend, residual = self._decompose(x)
        # Flatten (seq, features) so the matmul folds time-axis
        # projection and feature mixing into one weight matrix.
        n, b, t, f = x.shape
        trend_flat = trend.reshape(n, b, t * f)
        residual_flat = residual.reshape(n, b, t * f)
        # Per-cell matmul: (N, b, t*f) @ (N, t*f, hidden).T -> (N, b, hidden).
        # einsum lets the leading N axis stay outside the matmul.
        trend_h = torch.einsum("nbk,nhk->nbh", trend_flat, self.trend_weight) + self.trend_bias.unsqueeze(1)
        residual_h = torch.einsum("nbk,nhk->nbh", residual_flat, self.residual_weight) + self.residual_bias.unsqueeze(1)
        h = trend_h + residual_h
        h = nn.functional.gelu(h)
        h = self.dropout(h, generator=generator)
        head = torch.einsum("nbh,nph->nbp", h, self.head_weight) + self.head_bias.unsqueeze(1)
        head = nn.functional.gelu(head)
        raw = torch.einsum("nbp,nop->nbo", head, self.out_weight) + self.out_bias.unsqueeze(1)
        # Close stays unconstrained; volatility is non-negative via softplus.
        close = raw[..., 0:1]
        volatility = nn.functional.softplus(raw[..., 1:2])
        return torch.cat([close, volatility], dim=-1)
