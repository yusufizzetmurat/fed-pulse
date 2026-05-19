"""Unit tests for the Phase A ensemble aggregator (#226).

Locks the three aggregation strategies against synthetic inputs where
the analytical answer is known, and asserts the alignment + missing-arch
edge cases the production runner must handle.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.ensemble_aggregator import (
    _mean_logit,
    _mean_softmax,
    _plurality_vote,
    aggregate,
)


def _trial(
    *,
    architecture: str,
    fold_id: str,
    seed: int,
    predictions: list[int],
    targets: list[int],
    class_scores: list[list[float]] | None = None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "model_config": {"hidden_size": 256, "num_layers": 2, "dropout": 0.2},
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "test_metrics": {
            "predictions": predictions,
            "targets": targets,
            "classification_breakdown": {"macro_f1": 0.5},
        },
    }
    if class_scores is not None:
        summary["test_metrics"]["class_scores"] = class_scores  # type: ignore[index]
    return {
        "architecture": architecture,
        "fold_id": fold_id,
        "seed": seed,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Per-row strategy helpers
# ---------------------------------------------------------------------------


def test_mean_logit_picks_argmax_of_logsum() -> None:
    """Two architectures: scores [0.7, 0.2, 0.1] and [0.6, 0.3, 0.1]. The
    log-sum across class 0 dominates; class 0 wins."""
    assert _mean_logit([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]]) == 0


def test_mean_softmax_picks_argmax_of_sum() -> None:
    """Three architectures: each model's argmax is 0, 1, 2 respectively.
    Class 1's averaged softmax probability is highest -> mean_softmax
    picks 1, plurality picks the lowest tied class."""
    scores = [
        [0.5, 0.3, 0.2],
        [0.2, 0.7, 0.1],
        [0.2, 0.3, 0.5],
    ]
    assert _mean_softmax(scores) == 1


def test_plurality_breaks_ties_to_lowest_class() -> None:
    """Two architectures vote class 1 and class 2 — tied at 1 vote
    each. Tie-breaker is the lowest class index."""
    assert _plurality_vote([1, 2]) == 1


def test_plurality_consensus() -> None:
    assert _plurality_vote([2, 2, 0]) == 2


# ---------------------------------------------------------------------------
# Aggregator end-to-end
# ---------------------------------------------------------------------------


def test_ensemble_requires_two_architectures() -> None:
    """The aggregator raises ValueError on a single architecture."""
    trial = _trial(
        architecture="lstm",
        fold_id="wf_fold_1",
        seed=11,
        predictions=[0, 1, 2] * 5,
        targets=[0, 1, 2] * 5,
        class_scores=[[1.0, 0.0, 0.0]] * 15,
    )
    with pytest.raises(ValueError, match="at least 2 architectures"):
        aggregate({"lstm": [{"trials": [trial], "selection_metric": "macro_f1"}]})


def test_ensemble_aligns_per_fold_seed_cells() -> None:
    """Two architectures' (fold, seed)-aligned predictions are averaged
    into the ensemble; the cell breakdown matches the rule."""

    lstm_trial = _trial(
        architecture="lstm",
        fold_id="wf_fold_1",
        seed=11,
        predictions=[0, 0, 1],
        targets=[0, 1, 1],
        class_scores=[[0.7, 0.2, 0.1], [0.5, 0.3, 0.2], [0.2, 0.7, 0.1]],
    )
    tft_trial = _trial(
        architecture="tft",
        fold_id="wf_fold_1",
        seed=11,
        predictions=[0, 1, 1],
        targets=[0, 1, 1],
        class_scores=[[0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.3, 0.5, 0.2]],
    )
    payload = aggregate(
        {
            "lstm": [{"trials": [lstm_trial], "selection_metric": "macro_f1"}],
            "tft": [{"trials": [tft_trial], "selection_metric": "macro_f1"}],
        },
        strategies=("mean_softmax",),
        n_classes=3,
        n_resamples=20,
    )
    per_cell = payload["per_cell"]
    pooled = payload["pooled"]
    # One cell, one strategy -> one per_cell row + one pooled row.
    assert len(per_cell) == 1
    assert len(pooled) == 1
    cell = per_cell[0]
    assert cell.fold_id == "wf_fold_1"
    assert cell.seed == 11
    # Row 1: lstm scores [0.7,0.2,0.1], tft scores [0.6,0.3,0.1] -> mean
    # [0.65,0.25,0.10] -> argmax 0. Row 2: lstm [0.5,0.3,0.2] + tft
    # [0.3,0.6,0.1] -> mean [0.4,0.45,0.15] -> 1. Row 3: lstm
    # [0.2,0.7,0.1] + tft [0.3,0.5,0.2] -> mean [0.25,0.6,0.15] -> 1.
    # Targets are [0, 1, 1] -> all three correct -> macro-F1 = 1.0 over
    # classes 0 and 1 (class 2 absent in this minimal cell).
    assert cell.breakdown.macro_f1 == pytest.approx(1.0, rel=0, abs=1e-6)


def test_ensemble_skips_cells_with_missing_architectures() -> None:
    """If a cell has only one of the two architectures' predictions, the
    aggregator drops the cell rather than computing a biased ensemble."""

    lstm_a = _trial(
        architecture="lstm",
        fold_id="wf_fold_1",
        seed=11,
        predictions=[0, 1, 2] * 3,
        targets=[0, 1, 2] * 3,
        class_scores=[[1.0, 0.0, 0.0]] * 9,
    )
    lstm_b = _trial(
        architecture="lstm",
        fold_id="wf_fold_2",
        seed=11,
        predictions=[0, 1, 2] * 3,
        targets=[0, 1, 2] * 3,
        class_scores=[[1.0, 0.0, 0.0]] * 9,
    )
    tft_a = _trial(
        architecture="tft",
        fold_id="wf_fold_1",
        seed=11,
        predictions=[0, 1, 2] * 3,
        targets=[0, 1, 2] * 3,
        class_scores=[[1.0, 0.0, 0.0]] * 9,
    )
    payload = aggregate(
        {
            "lstm": [
                {"trials": [lstm_a, lstm_b], "selection_metric": "macro_f1"}
            ],
            "tft": [{"trials": [tft_a], "selection_metric": "macro_f1"}],
        },
        strategies=("plurality_vote",),
        n_classes=3,
        n_resamples=20,
    )
    per_cell = payload["per_cell"]
    # Only the (wf_fold_1, 11) cell is shared; wf_fold_2 has lstm only.
    assert len(per_cell) == 1
    assert per_cell[0].fold_id == "wf_fold_1"


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    from app.evaluation.ensemble_aggregator import main

    for arch in ("lstm", "tft"):
        sub = tmp_path / arch
        sub.mkdir()
        trial = _trial(
            architecture=arch,
            fold_id="wf_fold_1",
            seed=11,
            predictions=[0, 1, 2] * 5,
            targets=[0, 1, 2] * 5,
            class_scores=[[0.8, 0.1, 0.1]] * 15,
        )
        (sub / "forecaster_sweep_results.json").write_text(
            json.dumps({"trials": [trial], "selection_metric": "macro_f1"})
        )
    rc = main(
        [
            "--arch-sweep-dir",
            str(tmp_path),
            "--n-resamples",
            "10",
        ]
    )
    assert rc == 0
    assert (tmp_path / "ensemble_results.json").exists()
    assert (tmp_path / "ensemble_results.md").exists()
