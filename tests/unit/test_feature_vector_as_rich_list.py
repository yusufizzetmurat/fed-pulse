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


def test_as_rich_list_length_is_thirty_five() -> None:
    fv = _base_vector()
    assert len(fv.as_rich_list()) == RICH_FEATURE_SIZE == 35


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
