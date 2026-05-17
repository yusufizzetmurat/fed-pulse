from __future__ import annotations

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("yaml")

from app.data.label_schemas import (  # noqa: E402
    MultiAxisLabel,
    Stance,
    Topic,
    load_schema,
    sample_weight_for,
)


def test_schema_yaml_loads_with_expected_keys() -> None:
    schema = load_schema()
    assert "axes" in schema
    assert set(schema["axes"].keys()) == {"stance", "factor", "certainty", "topic"}
    assert "provenance" in schema


def test_sample_weights_table_matches_provenance() -> None:
    # ``sample_weight`` is a binary inclusion gate, not a per-row loss
    # multiplier — see ``data/schema/labels.yaml`` for the contract.
    assert sample_weight_for("peer_reviewed") == 1.0
    assert sample_weight_for("kaggle") == 1.0
    assert sample_weight_for("peer_reviewed_cross_bank") == 0.0
    assert sample_weight_for("scraped") == 0.0
    assert sample_weight_for("unknown-bucket") == 0.0


def test_multi_axis_label_stance_only() -> None:
    label = MultiAxisLabel(stance=Stance.HAWKISH)
    assert label.stance is Stance.HAWKISH
    assert label.factor is None
    assert label.certainty is None
    assert label.topic is None


def test_multi_axis_label_full() -> None:
    label = MultiAxisLabel(
        stance=Stance.DOVISH,
        factor=-0.42,
        certainty=0.71,
        topic=Topic.INFLATION,
    )
    assert label.factor == pytest.approx(-0.42)
    assert label.certainty == pytest.approx(0.71)
    assert label.topic is Topic.INFLATION


def test_factor_out_of_range_rejected() -> None:
    with pytest.raises(Exception):
        MultiAxisLabel(stance=Stance.HAWKISH, factor=1.5)


def test_certainty_out_of_range_rejected() -> None:
    with pytest.raises(Exception):
        MultiAxisLabel(stance=Stance.HAWKISH, certainty=-0.1)


def test_nan_factor_is_coerced_to_none() -> None:
    label = MultiAxisLabel(stance=Stance.NEUTRAL, factor=float("nan"))
    assert label.factor is None
