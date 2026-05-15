"""Unit tests for app.evaluation.cross_bank_transfer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation import cross_bank_transfer as cbt


def _write_registry(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


@pytest.fixture
def package_with_two_banks(tmp_path: Path) -> Path:
    package_dir = tmp_path / "processed" / "tp_v1"
    _write_registry(
        package_dir / "registry_normalized.jsonl",
        [
            # FOMC rows — present in training (sample_weight=1), not cross-bank.
            {
                "record_id": "fomc_1",
                "text": "the committee judges policy is appropriate",
                "event_date": "2024-01-01",
                "mapped_label": "neutral",
                "source": "hf_fomc_communication",
                "provenance": "peer_reviewed",
                "sample_weight": 1.0,
            },
            # ECB row — labeled, weight=0 (held out).
            {
                "record_id": "ecb_1",
                "text": "inflation outlook firm; rate path on hold",
                "event_date": "2024-02-01",
                "mapped_label": "hawkish",
                "source": "gtfintechlab_european_central_bank",
                "provenance": "peer_reviewed_cross_bank",
                "sample_weight": 0.0,
                "multi_axis_extras": {
                    "gtfintechlab_time_label": "forward looking",
                    "gtfintechlab_certain_label": "certain",
                },
            },
            {
                "record_id": "ecb_2",
                "text": "easing financial conditions argue for cuts",
                "event_date": "2024-03-01",
                "mapped_label": "dovish",
                "source": "gtfintechlab_european_central_bank",
                "provenance": "peer_reviewed_cross_bank",
                "sample_weight": 0.0,
                "multi_axis_extras": {
                    "gtfintechlab_time_label": "not forward looking",
                    "gtfintechlab_certain_label": "uncertain",
                },
            },
            # BoJ row — labeled, weight=0.
            {
                "record_id": "boj_1",
                "text": "policy stance remains accommodative",
                "event_date": "2024-04-01",
                "mapped_label": "dovish",
                "source": "gtfintechlab_bank_of_japan",
                "provenance": "peer_reviewed_cross_bank",
                "sample_weight": 0.0,
            },
        ],
    )
    return package_dir


def test_load_cross_bank_rows_filters_to_target_bank(package_with_two_banks: Path) -> None:
    ecb = cbt.load_cross_bank_rows(package_with_two_banks, "gtfintechlab_european_central_bank")
    assert {r.record_id for r in ecb} == {"ecb_1", "ecb_2"}
    boj = cbt.load_cross_bank_rows(package_with_two_banks, "gtfintechlab_bank_of_japan")
    assert {r.record_id for r in boj} == {"boj_1"}


def test_load_cross_bank_rows_includes_weight_zero(package_with_two_banks: Path) -> None:
    """Cross-bank rows live in the registry with sample_weight=0 — must still surface in eval."""

    ecb = cbt.load_cross_bank_rows(package_with_two_banks, "gtfintechlab_european_central_bank")
    assert all(r.sample_weight == 0.0 for r in ecb)


def test_evaluate_cross_bank_with_perfect_predictor(package_with_two_banks: Path) -> None:
    """Inject an oracle predictor — macro-F1 should be 1.0."""

    def oracle(rows):
        labels = [r.label for r in rows]
        return list(labels), list(labels), [0.5] * len(labels)

    result = cbt.evaluate_cross_bank(
        package_dir=package_with_two_banks,
        bank_source="gtfintechlab_european_central_bank",
        checkpoint="oracle",
        predict_fn=oracle,
    )
    assert result.bank == "gtfintechlab_european_central_bank"
    assert result.support == 2
    assert result.accuracy == pytest.approx(1.0)
    # macro-F1 averages over all three official labels (dovish/hawkish/neutral);
    # the ECB fixture has no neutral row, so the neutral slot contributes F1=0
    # and the perfect-prediction ceiling is 2/3, not 1.
    assert result.macro_f1 == pytest.approx(2 / 3)
    assert result.per_class["hawkish"]["f1"] == pytest.approx(1.0)
    assert result.per_class["dovish"]["f1"] == pytest.approx(1.0)


def test_evaluate_cross_bank_with_constant_predictor(package_with_two_banks: Path) -> None:
    """A constant-prediction baseline lands far from perfect."""

    def constant_neutral(rows):
        return [r.label for r in rows], ["neutral"] * len(rows), [0.5] * len(rows)

    result = cbt.evaluate_cross_bank(
        package_dir=package_with_two_banks,
        bank_source="gtfintechlab_european_central_bank",
        checkpoint="constant",
        predict_fn=constant_neutral,
    )
    assert result.macro_f1 < 0.5
    assert "hawkish" in result.per_class
    assert "dovish" in result.per_class


def test_evaluate_cross_bank_emits_per_axis_slices(package_with_two_banks: Path) -> None:
    def oracle(rows):
        labels = [r.label for r in rows]
        return list(labels), list(labels), [0.5] * len(labels)

    result = cbt.evaluate_cross_bank(
        package_dir=package_with_two_banks,
        bank_source="gtfintechlab_european_central_bank",
        checkpoint="oracle",
        predict_fn=oracle,
    )
    assert set(result.per_axis.keys()) >= {"time", "certainty"}
    assert "forward looking" in result.per_axis["time"]
    assert result.per_axis["time"]["forward looking"]["support"] == 1


def test_evaluate_cross_bank_rejects_unknown_bank(package_with_two_banks: Path) -> None:
    with pytest.raises(ValueError, match="Unknown bank_source"):
        cbt.evaluate_cross_bank(
            package_dir=package_with_two_banks,
            bank_source="not_a_real_bank",
            checkpoint="x",
            predict_fn=lambda rows: ([], [], []),
        )


def test_evaluate_cross_bank_errors_on_empty_bank(tmp_path: Path) -> None:
    empty = tmp_path / "processed" / "tp_empty"
    _write_registry(empty / "registry_normalized.jsonl", [])
    with pytest.raises(ValueError, match="No labeled rows"):
        cbt.evaluate_cross_bank(
            package_dir=empty,
            bank_source="gtfintechlab_european_central_bank",
            checkpoint="x",
            predict_fn=lambda rows: ([], [], []),
        )
