"""Bucketed hyperparameter sweep for the forecaster.

Forecasters in this project are TINY (hidden in {32, 64, 128}, layers in
{1, 2, 3}, sequence length 20). Per-cell forward + backward is a
handful of CUDA kernels that finishes before the next kernel is even
dispatched, so the GPU sits at ~25% TGP when each cell runs in its
own process. The bucketed sweep groups cells that share the same
model topology and data feed and runs them concurrently inside one
Python process.

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
