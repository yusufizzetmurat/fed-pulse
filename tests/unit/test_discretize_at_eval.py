"""Unit tests for app.eval.discretize_at_eval (#498)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from app.eval.discretize_at_eval import (
    bucketize,
    compute_discretize_at_eval,
    discretized_macro_f1,
    extract_discretized_cells,
)


def test_bucketize_hand_computed_five_predictions() -> None:
    # Bin edges [-1.0, 0.5] split into classes:
    #   v < -1.0     -> 0
    #   -1.0 <= v <  0.5 -> 1
    #   v >= 0.5     -> 2
    preds = [-2.0, -1.0, 0.0, 0.49, 0.5]
    expected = [0, 1, 1, 1, 2]
    assert bucketize(preds, [-1.0, 0.5]) == expected


def test_bucketize_three_class_boundary_inclusive_above() -> None:
    # A value at the upper cutoff lands in the top class (strict-less semantic).
    assert bucketize([0.5], [-1.0, 0.5]) == [2]


def test_discretized_macro_f1_perfect_recovery() -> None:
    # Predictions inside each bin's interior -> argmax matches targets.
    preds = [-2.0, 0.0, 1.0, -1.5, 0.2, 0.9]
    targets = [0, 1, 2, 0, 1, 2]
    f1 = discretized_macro_f1(preds, targets, [-1.0, 0.5])
    assert f1 == pytest.approx(1.0)


def test_discretized_macro_f1_constant_prediction() -> None:
    # Predicting always class 1 -> only the middle class scores nonzero.
    preds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    targets = [0, 1, 2, 0, 1, 2]
    f1 = discretized_macro_f1(preds, targets, [-1.0, 0.5])
    # Class 1: TP=2, FP=4, FN=0 -> precision=1/3, recall=1.0, F1=0.5.
    # Class 0, Class 2: zero TP / FP / FN -> F1=0 each.
    # Macro = (0 + 0.5 + 0) / 3 = 0.5 / 3.
    assert f1 == pytest.approx(0.5 / 3)


def _write_no_per_event(tmp_path: Path) -> Path:
    p = tmp_path / "no_per_event.json"
    p.write_text(json.dumps({
        "trials": {
            "regression": [{
                "seed": 11,
                "folds": [{
                    "fold_id": "wf_fold_1",
                    "metrics": {
                        "regime_f1_macro": 0.14,
                        "regression_rmse_log_rv": 0.96,
                    },
                }],
            }],
        },
    }), encoding="utf-8")
    return p


def test_skip_missing_per_event_with_warning(tmp_path: Path) -> None:
    path = _write_no_per_event(tmp_path)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cells, n_skipped = extract_discretized_cells(path)
    assert cells == []
    assert n_skipped == 1
    assert any(
        "regression_predictions" in str(warning.message)
        or "bin_edges" in str(warning.message)
        for warning in w
    )


def _write_with_per_event(tmp_path: Path) -> Path:
    # Two seeds x one fold; bin edges [-1.0, 0.5]; predictions chosen so
    # discretized predictions exactly recover the targets -> F1 = 1.0.
    fold = {
        "fold_id": "wf_fold_1",
        "metrics": {
            "regime_f1_macro": 0.18,
            "regression_rmse_log_rv": 0.96,
            "regression_predictions": [-2.0, 0.0, 1.0, -1.5, 0.2, 0.9],
            "regression_targets": [0, 1, 2, 0, 1, 2],
            "bin_edges": [-1.0, 0.5],
        },
    }
    p = tmp_path / "with_per_event.json"
    p.write_text(json.dumps({
        "trials": {
            "classification": [
                {"seed": 11, "folds": [{
                    "fold_id": "wf_fold_1",
                    "metrics": {"regime_f1_macro": 0.40},
                }]},
                {"seed": 29, "folds": [{
                    "fold_id": "wf_fold_1",
                    "metrics": {"regime_f1_macro": 0.42},
                }]},
            ],
            "regression": [
                {"seed": 11, "folds": [fold]},
                {"seed": 29, "folds": [fold]},
            ],
            "dual": [
                {"seed": 11, "folds": [{
                    "fold_id": "wf_fold_1",
                    "metrics": {"regime_f1_macro": 0.44},
                }]},
                {"seed": 29, "folds": [{
                    "fold_id": "wf_fold_1",
                    "metrics": {"regime_f1_macro": 0.46},
                }]},
            ],
        },
    }), encoding="utf-8")
    return p


def test_end_to_end_aggregation_populated(tmp_path: Path) -> None:
    path = _write_with_per_event(tmp_path)
    result = compute_discretize_at_eval([path], n_resamples=50)
    assert result["n_discretized_cells"] == 2
    assert result["runner_extension_required"] is False
    arms = {s["arm"]: s for s in result["arm_summaries"]}
    assert arms["regression_discretized"]["f1_mean"] == pytest.approx(1.0)
    assert arms["classification"]["n_cells"] == 2
    assert arms["dual"]["n_cells"] == 2
    paired_labels = {(p["label_a"], p["label_b"]) for p in result["paired_tests"]}
    assert ("regression_discretized", "dual") in paired_labels
    assert ("regression_discretized", "classification") in paired_labels


def test_end_to_end_skips_when_no_per_event(tmp_path: Path) -> None:
    path = _write_no_per_event(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compute_discretize_at_eval([path], n_resamples=50)
    assert result["n_discretized_cells"] == 0
    assert result["runner_extension_required"] is True
    assert result["n_skipped_missing_predictions"] == 1
    assert result["paired_tests"] == []
