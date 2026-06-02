"""Per-axis target tensor builder + FeatureVector population (#78).

The loader extracts axis labels off events.parquet, populates the
target fields on every FeatureVector, then materialises 3 target
tensors + 3 mask tensors aligned 1:1 with the classification rows
``_build_training_tensors`` emits. This covers both halves: the
FeatureVector population and the tensor-builder output.
"""

from __future__ import annotations

import torch

from app.models.config import (
    FeatureVector,
    MULTI_TASK_CERTAINTY_LABELS,
    MULTI_TASK_STANCE_LABELS,
    MULTI_TASK_TIME_LABELS,
    SEQUENCE_LENGTH,
)
from app.training.loaders import (
    _attach_rich_features,
    _build_multi_task_target_tensors,
)


def _base_vector(forward_vol: float | None = 0.1) -> FeatureVector:
    return FeatureVector(
        date="2024-01-01",
        sentiment_score=0.0,
        market_close=4000.0,
        market_volatility=0.01,
        forward_realized_vol_10d=forward_vol,
    )


def _make_group(forward_vol: float | None = 0.1) -> list[FeatureVector]:
    # SEQUENCE_LENGTH lookback bars + 1 target bar.
    return [_base_vector(forward_vol=None) for _ in range(SEQUENCE_LENGTH)] + [
        _base_vector(forward_vol=forward_vol)
    ]


def test_attach_rich_features_populates_target_fields_from_event_row() -> None:
    vectors = _make_group()
    event_row = {
        "axis_stance": "hawkish",
        "axis_time_label": "forward looking",
        "axis_certain_label": "uncertain",
    }
    _attach_rich_features(
        vectors,
        event_row=event_row,
        linguistic_lookup={},
        mp_surprise_lookup={},
        text_hash="xx",
        event_date_str="2024-01-01",
        use_credibility=False,
        use_linguistic=False,
        use_mp_surprise=False,
        use_multi_axis=True,
    )
    target = vectors[-1]
    assert target.target_stance_present is True
    assert target.target_stance_idx == MULTI_TASK_STANCE_LABELS.index("hawkish")
    assert target.target_time_present is True
    assert target.target_time_idx == MULTI_TASK_TIME_LABELS.index("forward looking")
    assert target.target_certainty_present is True
    assert (
        target.target_certainty_idx == MULTI_TASK_CERTAINTY_LABELS.index("uncertain")
    )


def test_attach_rich_features_leaves_masks_false_when_row_has_no_labels() -> None:
    vectors = _make_group()
    _attach_rich_features(
        vectors,
        event_row={},  # no axis_* keys at all
        linguistic_lookup={},
        mp_surprise_lookup={},
        text_hash="xx",
        event_date_str="2024-01-01",
        use_credibility=False,
        use_linguistic=False,
        use_mp_surprise=False,
        use_multi_axis=True,
    )
    target = vectors[-1]
    assert target.target_stance_present is False
    assert target.target_time_present is False
    assert target.target_certainty_present is False


def test_certainty_float_falls_back_to_binned_class_label() -> None:
    """When ``axis_certain_label`` is missing but ``axis_certainty``
    (float in [0, 1]) is populated, the loader bins the numeric value
    into the categorical class indices used by the multi-task head."""

    vectors = _make_group()
    _attach_rich_features(
        vectors,
        event_row={"axis_certainty": 0.85},
        linguistic_lookup={},
        mp_surprise_lookup={},
        text_hash="xx",
        event_date_str="2024-01-01",
        use_credibility=False,
        use_linguistic=False,
        use_mp_surprise=False,
        use_multi_axis=True,
    )
    target = vectors[-1]
    assert target.target_certainty_present is True
    # 0.85 > 0.66 -> certain
    assert target.target_certainty_idx == MULTI_TASK_CERTAINTY_LABELS.index("certain")


def test_target_tensor_builder_aligns_rows_with_classification_filter() -> None:
    """``_build_multi_task_target_tensors`` must apply the same row
    drop that ``_build_training_tensors`` does (rows whose
    ``forward_realized_vol_10d`` -> -1 under the fitted quantiles are
    dropped)."""

    keep_group = _make_group(forward_vol=0.05)
    drop_group = _make_group(forward_vol=None)  # target row has no forward vol
    # Populate stance on both target rows so we can verify the row
    # ordering survives the filter.
    keep_group[-1].target_stance_idx = MULTI_TASK_STANCE_LABELS.index("dovish")
    keep_group[-1].target_stance_present = True
    drop_group[-1].target_stance_idx = MULTI_TASK_STANCE_LABELS.index("hawkish")
    drop_group[-1].target_stance_present = True

    quantiles = (0.1, 0.2)
    out = _build_multi_task_target_tensors(
        [keep_group, drop_group], vol_regime_quantiles=quantiles
    )
    assert out is not None
    # Only the keep_group target survived (drop_group has no forward
    # vol so vol_regime_class_for returns -1 and the row is dropped).
    assert out["stance"].shape == (1,)
    assert int(out["stance"][0]) == MULTI_TASK_STANCE_LABELS.index("dovish")
    # Mask types: all axes are classification (CrossEntropy) heads, so
    # targets are long and masks are bool.
    assert out["stance"].dtype == torch.long
    assert out["stance_mask"].dtype == torch.bool
    assert out["time"].dtype == torch.long
    assert out["time_mask"].dtype == torch.bool


def test_target_tensor_builder_returns_none_on_empty_input() -> None:
    out = _build_multi_task_target_tensors([], vol_regime_quantiles=(0.1, 0.2))
    assert out is None
