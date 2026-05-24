from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from app.models.config import FeatureVector, SEQUENCE_LENGTH
from app.training.loaders import (
    _build_training_tensors,
    collect_forward_vols,
)
from app.training.loop import _build_partition_tensors


def _bar(idx: int, close: float, vol: float, *, forward_vol: float | None = None) -> FeatureVector:
    """Construct a minimal-but-valid 35-dim rich-feature bar."""

    return FeatureVector(
        date=f"2024-01-{(idx % 28) + 1:02d}",
        sentiment_score=0.0,
        market_close=close,
        market_volatility=vol,
        close_change_pct=0.0,
        volatility_change=0.0,
        elapsed_time=float(idx),
        rich_payload=True,
        forward_realized_vol_10d=forward_vol,
    )


def _group(forward_vol: float | None) -> list[FeatureVector]:
    """Build a single SEQUENCE_LENGTH+1 group whose target carries ``forward_vol``."""

    bars = [_bar(i, 100.0 + i, 0.01 + 0.001 * i) for i in range(SEQUENCE_LENGTH)]
    target = _bar(
        SEQUENCE_LENGTH,
        100.0 + SEQUENCE_LENGTH,
        0.02,
        forward_vol=forward_vol,
    )
    return [*bars, target]


# ---------------------------------------------------------------------------
# Regression contract (byte-identity)
# ---------------------------------------------------------------------------


def test_regression_mode_emits_float_target_tensor() -> None:
    groups = [_group(0.05), _group(0.10)]
    x, y, scale = _build_training_tensors(groups, output_mode="regression")
    assert y is not None
    assert y.dtype == torch.float32
    assert y.shape == (2, 2)  # (N, [close_norm, vol])
    assert x is not None
    assert x.shape[0] == 2


def test_regression_mode_ignores_missing_forward_vol() -> None:
    """Regression must not drop rows whose forward-vol is null."""

    groups = [_group(None), _group(None)]
    x, y, _ = _build_training_tensors(groups, output_mode="regression")
    assert y is not None and x is not None
    assert y.shape == (2, 2)


# ---------------------------------------------------------------------------
# Classification dispatch
# ---------------------------------------------------------------------------


def test_classification_mode_emits_long_class_index_tensor() -> None:
    groups = [_group(0.005), _group(0.015), _group(0.030)]
    cutoffs = (0.01, 0.02)
    x, y, _ = _build_training_tensors(
        groups, output_mode="classification", vol_regime_quantiles=cutoffs
    )
    assert y is not None
    assert y.dtype == torch.long
    assert y.shape == (3,)
    assert y.tolist() == [0, 1, 2]
    assert x is not None and x.shape[0] == 3


def test_classification_drops_rows_with_missing_forward_vol() -> None:
    groups = [_group(0.005), _group(None), _group(0.030)]
    x, y, _ = _build_training_tensors(
        groups, output_mode="classification", vol_regime_quantiles=(0.01, 0.02)
    )
    assert y is not None and x is not None
    assert y.shape == (2,)
    assert x.shape[0] == 2
    assert y.tolist() == [0, 2]


def test_classification_returns_none_when_all_rows_missing() -> None:
    groups = [_group(None), _group(None)]
    x, y, _ = _build_training_tensors(
        groups, output_mode="classification", vol_regime_quantiles=(0.01, 0.02)
    )
    assert x is None and y is None


# ---------------------------------------------------------------------------
# _build_partition_tensors row alignment
# ---------------------------------------------------------------------------


def test_partition_classification_keeps_text_rows_aligned_with_y() -> None:
    """The text-embedding tensor row count must equal the y row count
    even when some groups drop out of the classification path."""

    groups = [_group(0.005), _group(None), _group(0.030)]
    # Decorate each group's bars with a non-empty pooled embedding so
    # the text-embedding builder takes its non-None branch.
    for group in groups:
        for vec in group:
            vec.text_embedding_pooled = [0.1, 0.2, 0.3, 0.4]
            vec.text_embedding_missing = 0.0
    x, y, _, text_emb, text_missing = _build_partition_tensors(
        groups,
        fallback_text_in_dim=4,
        output_mode="classification",
        vol_regime_quantiles=(0.01, 0.02),
    )
    assert y is not None and text_emb is not None and text_missing is not None
    assert x is not None
    assert y.shape[0] == text_emb.shape[0] == text_missing.shape[0] == x.shape[0] == 2


def test_partition_regression_path_is_byte_identical_on_default_kwargs() -> None:
    """Default kwargs (output_mode='regression', empty quantiles) keep
    the byte-identity regression contract by not touching the x/y/text
    pipelines."""

    groups = [_group(0.005), _group(0.030)]
    out_default = _build_partition_tensors(groups, fallback_text_in_dim=0)
    out_explicit = _build_partition_tensors(
        groups, fallback_text_in_dim=0, output_mode="regression"
    )
    # x tensors equal element-wise
    assert torch.allclose(out_default[0], out_explicit[0])
    # y tensors equal element-wise
    assert torch.allclose(out_default[1], out_explicit[1])


# ---------------------------------------------------------------------------
# collect_forward_vols helper
# ---------------------------------------------------------------------------


def test_collect_forward_vols_returns_only_target_row_values() -> None:
    groups = [_group(0.005), _group(0.030), _group(None)]
    vols = collect_forward_vols(groups)
    assert sorted(vols) == [0.005, 0.030]


def test_collect_forward_vols_skips_short_groups() -> None:
    short = [_bar(i, 100.0, 0.01) for i in range(SEQUENCE_LENGTH)]  # no target row
    full = _group(0.020)
    vols = collect_forward_vols([short, full])
    assert vols == [0.020]
