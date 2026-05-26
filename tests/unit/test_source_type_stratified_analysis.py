from __future__ import annotations

import pytest

from app.data import source_type_stratified_analysis as analyzer


def _row(record_id, gold, pred, source_type):
    return {
        "record_id": record_id,
        "mapped_label": gold,
        "predicted_label": pred,
        "source_type": source_type,
    }


def test_compute_stratified_metrics_groups_by_source_type() -> None:
    rows = [
        _row("a", "hawkish", "hawkish", "fomc_minutes"),
        _row("b", "hawkish", "dovish", "fomc_minutes"),
        _row("c", "dovish", "dovish", "fomc_statement"),
        _row("d", "neutral", "neutral", "chair_speech"),
    ]

    out = analyzer.compute_stratified_metrics(rows)

    assert "fomc_minutes" in out
    assert "fomc_statement" in out
    assert "chair_speech" in out
    fm = out["fomc_minutes"]
    assert fm["support"] == 2
    assert fm["accuracy"] == pytest.approx(0.5)
    assert "macro_f1" in fm
    assert "per_class" in fm


def test_compute_stratified_metrics_handles_empty_input() -> None:
    out = analyzer.compute_stratified_metrics([])
    assert out == {}


def test_join_predictions_to_source_type_uses_record_id() -> None:
    predictions = [
        {"record_id": "a", "mapped_label": "hawkish", "predicted_label": "hawkish"},
        {"record_id": "b", "mapped_label": "dovish", "predicted_label": "neutral"},
    ]
    registry = [
        {"record_id": "a", "source_type": "fomc_minutes"},
        {"record_id": "b", "source_type": "chair_speech"},
        {"record_id": "c", "source_type": "fomc_statement"},
    ]

    joined = analyzer.join_predictions_to_source_type(predictions, registry)

    assert len(joined) == 2
    assert joined[0]["source_type"] == "fomc_minutes"
    assert joined[1]["source_type"] == "chair_speech"


def test_join_predictions_to_source_type_drops_predictions_without_registry_match() -> None:
    predictions = [
        {"record_id": "a", "mapped_label": "hawkish", "predicted_label": "hawkish"},
        {"record_id": "missing", "mapped_label": "dovish", "predicted_label": "dovish"},
    ]
    registry = [{"record_id": "a", "source_type": "fomc_minutes"}]
    joined = analyzer.join_predictions_to_source_type(predictions, registry)
    assert len(joined) == 1
    assert joined[0]["record_id"] == "a"
