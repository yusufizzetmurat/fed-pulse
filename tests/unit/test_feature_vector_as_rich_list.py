from __future__ import annotations

import pytest

from app.models.config import FeatureVector, RICH_FEATURE_SIZE


def _base_vector(**overrides) -> FeatureVector:
    base = dict(
        date="2024-09-18",
        sentiment_score=0.42,
        market_close=4321.0,
        market_volatility=0.01,
    )
    base.update(overrides)
    return FeatureVector(**base)


def test_as_rich_list_length_matches_rich_feature_size() -> None:
    """The per-bar feature vector emits exactly ``RICH_FEATURE_SIZE``
    floats. The numeric size changes as new families land (35 at the
    Phase 8 milestone; 37 after A2 (#207) introduced the realised-vol
    horizons slice); the contract this test enforces is that the
    function's output length tracks the module constant."""

    fv = _base_vector()
    assert len(fv.as_rich_list()) == RICH_FEATURE_SIZE


def test_option_a_slot_default_is_stance_missing() -> None:
    fv = _base_vector()
    slot = fv.as_rich_list()[29:35]
    assert slot == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def test_option_a_slot_emits_in_documented_order() -> None:
    fv = _base_vector(
        stance_hawk=1.0,
        stance_dove=0.0,
        stance_neutral=0.0,
        time_label_forward=1.0,
        certain_label_certain=1.0,
        stance_missing=0.0,
    )
    assert fv.as_rich_list()[29:35] == [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]


@pytest.mark.parametrize(
    "stance_fields,expected_one_hot",
    [
        ({"stance_hawk": 1.0}, [1.0, 0.0, 0.0]),
        ({"stance_dove": 1.0}, [0.0, 1.0, 0.0]),
        ({"stance_neutral": 1.0}, [0.0, 0.0, 1.0]),
    ],
)
def test_stance_one_hot_positions(
    stance_fields: dict[str, float],
    expected_one_hot: list[float],
) -> None:
    fv = _base_vector(stance_missing=0.0, **stance_fields)
    assert fv.as_rich_list()[29:32] == expected_one_hot


def test_market_block_unchanged_at_positions_zero_to_six() -> None:
    fv = _base_vector(stance_hawk=1.0, stance_missing=0.0)
    rich = fv.as_rich_list()
    legacy = fv.as_list()
    assert rich[:6] == legacy


def test_realized_vol_slice_emits_at_documented_positions() -> None:
    """A2 (#207) realised-vol horizons land at positions [35:37].

    The legacy fields default to 0.0 (no vol history attached) so a
    rich-feature vector built from the bar serdes carries the values
    the loader set, including 0.0 when the underlying parquet predated
    the A2 schema bump."""

    from app.models.config import RICH_REALIZED_VOL_SLICE

    fv = _base_vector(realized_vol_20d=0.0123, realized_vol_60d=0.0456)
    rich = fv.as_rich_list()
    assert rich[RICH_REALIZED_VOL_SLICE] == [0.0123, 0.0456]


def test_realized_vol_slice_default_is_zero() -> None:
    """A FeatureVector built without realised-vol payload (e.g. from
    legacy fixtures or pre-A2 events.parquet) leaves the slice at 0.0."""

    from app.models.config import RICH_REALIZED_VOL_SLICE

    fv = _base_vector()
    assert fv.as_rich_list()[RICH_REALIZED_VOL_SLICE] == [0.0, 0.0]


def test_cross_asset_slice_emits_at_documented_positions() -> None:
    """Cross-asset close levels + Chunk 1 (VIX term structure + 10Y-3M
    yield slope) land at positions [37:45] in the order VIX, DXY, TNX,
    gold, VIX3M, IRX, vix_term_slope, yield_curve_slope_10y_3m."""

    from app.models.config import RICH_CROSS_ASSET_SLICE

    fv = _base_vector(
        vix_close=22.3,
        dxy_close=104.5,
        tnx_close=4.21,
        gold_close=1942.0,
        vix3m_close=23.7,
        irx_close=5.15,
        vix_term_slope=0.061,
        yield_curve_slope_10y_3m=-0.94,
    )
    assert fv.as_rich_list()[RICH_CROSS_ASSET_SLICE] == [
        22.3,
        104.5,
        4.21,
        1942.0,
        23.7,
        5.15,
        0.061,
        -0.94,
    ]


def test_cross_asset_slice_default_is_zero() -> None:
    """Legacy / pre-Chunk-1 FeatureVectors carry 0.0 across all 8 axes."""

    from app.models.config import RICH_CROSS_ASSET_SLICE

    fv = _base_vector()
    assert fv.as_rich_list()[RICH_CROSS_ASSET_SLICE] == [0.0] * 8


def test_llm_features_slice_default_is_missing_flag_one() -> None:
    """B1 (#212): a FeatureVector built without LLM-feature payload
    emits an all-zeros 35-dim slot + a missing flag at 1.0."""

    from app.models.config import (
        RICH_LLM_FEATURE_DIM,
        RICH_LLM_FEATURE_SLICE,
        RICH_LLM_FEATURE_MISSING_SLICE,
    )

    fv = _base_vector()
    rich = fv.as_rich_list()
    assert rich[RICH_LLM_FEATURE_SLICE] == [0.0] * RICH_LLM_FEATURE_DIM
    assert rich[RICH_LLM_FEATURE_MISSING_SLICE] == [1.0]


def test_llm_features_slice_emits_attached_vector() -> None:
    """An attached one-hot vector lands at the documented offsets and
    the missing flag flips to 0."""

    from app.models.config import (
        RICH_LLM_FEATURE_DIM,
        RICH_LLM_FEATURE_SLICE,
        RICH_LLM_FEATURE_MISSING_SLICE,
    )

    fv = _base_vector()
    one_hot = [0.0] * RICH_LLM_FEATURE_DIM
    one_hot[0] = 1.0
    one_hot[10] = 1.0
    fv.llm_features = one_hot
    fv.llm_features_missing = 0.0
    rich = fv.as_rich_list()
    assert rich[RICH_LLM_FEATURE_SLICE] == one_hot
    assert rich[RICH_LLM_FEATURE_MISSING_SLICE] == [0.0]
