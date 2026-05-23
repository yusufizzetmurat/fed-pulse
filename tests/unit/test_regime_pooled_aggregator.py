"""Unit tests for the Phase A pooled-fold aggregator (#226).

The aggregator pools test-partition predictions across walk-forward
folds and reports a macro-F1 with a block-bootstrap CI. The key
property to lock down is that pooling N folds of n_per_fold rows
produces an n = N * n_per_fold evaluation surface, the bootstrap CI
brackets the point estimate, and the per-class breakdown is preserved
in the pooled output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.regime_pooled_aggregator import (
    aggregate,
    pool_cell,
)


def _trial(
    *,
    architecture: str,
    fold_id: str,
    seed: int,
    predictions: list[int],
    targets: list[int],
    hidden_size: int = 256,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> dict[str, object]:
    """Build a synthetic per-trial record matching the loop.py contract."""

    return {
        "architecture": architecture,
        "fold_id": fold_id,
        "seed": seed,
        "summary": {
            "model_config": {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
            },
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "test_metrics": {
                "predictions": predictions,
                "targets": targets,
                "classification_breakdown": {
                    "macro_f1": _compute_macro(predictions, targets),
                },
            },
        },
    }


def _compute_macro(preds: list[int], targets: list[int]) -> float:
    classes = sorted(set(targets))
    if not classes:
        return 0.0
    f1s: list[float] = []
    for c in classes:
        tp = sum(1 for p, t in zip(preds, targets) if p == c and t == c)
        fp = sum(1 for p, t in zip(preds, targets) if p == c and t != c)
        fn = sum(1 for p, t in zip(preds, targets) if p != c and t == c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def test_pool_cell_concatenates_predictions_and_targets() -> None:
    """Two 50-row folds pool into a single 100-row breakdown."""

    fold1 = _trial(
        architecture="lstm",
        fold_id="wf_fold_1",
        seed=11,
        predictions=[0, 1, 2] * 17 + [0],
        targets=[0, 1, 2] * 17 + [0],
    )
    fold2 = _trial(
        architecture="lstm",
        fold_id="wf_fold_2",
        seed=11,
        predictions=[2, 1, 0] * 17 + [2],
        targets=[0, 1, 2] * 17 + [0],
    )
    pooled = pool_cell([fold1, fold2], n_classes=3, n_resamples=50)
    assert pooled.n_pooled == 104
    assert pooled.architecture == "lstm"
    assert set(pooled.folds) == {"wf_fold_1", "wf_fold_2"}
    assert pooled.seeds == (11,)
    # The pooled macro-F1 is a function of the concatenated cells; it is
    # NOT in general equal to the mean of the per-fold macro-F1 values.
    per_fold = [
        _compute_macro(fold1["summary"]["test_metrics"]["predictions"], fold1["summary"]["test_metrics"]["targets"]),  # type: ignore[index]
        _compute_macro(fold2["summary"]["test_metrics"]["predictions"], fold2["summary"]["test_metrics"]["targets"]),  # type: ignore[index]
    ]
    mean_of_per_fold = sum(per_fold) / len(per_fold)
    # The distinction this aggregator exists to surface: pooled != mean.
    assert abs(pooled.breakdown.macro_f1 - mean_of_per_fold) > 1e-6


def test_bootstrap_ci_brackets_point_estimate() -> None:
    """The 95% CI band must contain the point estimate of macro-F1."""

    preds = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0] * 10
    targets = [0, 1, 2, 0, 1, 2, 1, 1, 2, 0] * 10
    trial = _trial(
        architecture="lstm",
        fold_id="wf_fold_1",
        seed=11,
        predictions=preds,
        targets=targets,
    )
    pooled = pool_cell([trial], n_classes=3, n_resamples=200, bootstrap_seed=11)
    ci = pooled.macro_f1_ci
    assert ci.lo <= ci.point <= ci.hi


def test_per_class_breakdown_preserved_in_pooled_output() -> None:
    """The pooled breakdown carries per-class metrics matching what a
    direct computation on the concatenated rows would produce."""

    preds = [0, 0, 0, 1, 1, 2]
    targets = [0, 0, 1, 1, 2, 2]
    pooled = pool_cell(
        [
            _trial(
                architecture="lstm",
                fold_id="wf_fold_1",
                seed=11,
                predictions=preds,
                targets=targets,
            )
        ],
        n_classes=3,
        n_resamples=20,
    )
    classes = pooled.breakdown.per_class
    assert len(classes) == 3
    # Class 0: TP=2, FP=1, FN=0 -> P=2/3, R=2/2=1.0, F1=0.8
    assert pytest.approx(classes[0].f1, rel=0, abs=1e-6) == 0.8
    # Class 1: TP=1, FP=1, FN=1 -> P=1/2, R=1/2, F1=0.5
    assert pytest.approx(classes[1].f1, rel=0, abs=1e-6) == 0.5
    # Class 2: TP=1, FP=0, FN=1 -> P=1.0, R=0.5, F1=2/3
    assert classes[2].f1 == pytest.approx(2.0 / 3.0, rel=1e-6)


def test_aggregate_selects_best_hp_per_architecture() -> None:
    """Two HP cells on the same architecture should collapse to the
    better one in the pooled output."""

    good_cell = [
        _trial(
            architecture="lstm",
            fold_id=f"wf_fold_{i}",
            seed=11,
            predictions=[0, 1, 2] * 5,
            targets=[0, 1, 2] * 5,
            hidden_size=256,
        )
        for i in range(1, 5)
    ]
    bad_cell = [
        _trial(
            architecture="lstm",
            fold_id=f"wf_fold_{i}",
            seed=11,
            predictions=[0] * 15,
            targets=[0, 1, 2] * 5,
            hidden_size=64,
        )
        for i in range(1, 5)
    ]
    blob = {"trials": good_cell + bad_cell, "selection_metric": "macro_f1"}
    rows = aggregate([blob], n_classes=3, n_resamples=20)
    assert len(rows) == 1
    assert rows[0].architecture == "lstm"
    # The good cell hits macro-F1 = 1.0; the bad cell collapses to a
    # single class. The aggregator must select the good cell.
    assert rows[0].breakdown.macro_f1 == pytest.approx(1.0, rel=0, abs=1e-6)


def test_aggregate_returns_one_row_per_architecture() -> None:
    blobs = [
        {
            "trials": [
                _trial(
                    architecture=arch,
                    fold_id="wf_fold_1",
                    seed=11,
                    predictions=[0, 1, 2] * 5,
                    targets=[0, 1, 2] * 5,
                )
            ],
            "selection_metric": "macro_f1",
        }
        for arch in ("lstm", "tft", "gru")
    ]
    rows = aggregate(blobs, n_classes=3, n_resamples=10)
    assert sorted(row.architecture for row in rows) == ["gru", "lstm", "tft"]


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    """End-to-end CLI run produces both output files."""

    from app.evaluation.regime_pooled_aggregator import main

    tier_dir = tmp_path / "tier_demo"
    tier_dir.mkdir()
    blob = {
        "trials": [
            _trial(
                architecture="lstm",
                fold_id="wf_fold_1",
                seed=11,
                predictions=[0, 1, 2] * 7,
                targets=[0, 1, 2] * 7,
            )
        ],
        "selection_metric": "macro_f1",
    }
    (tier_dir / "forecaster_sweep_results.json").write_text(json.dumps(blob))
    rc = main(
        [
            "--input-dir",
            str(tmp_path),
            "--n-resamples",
            "10",
        ]
    )
    assert rc == 0
    assert (tmp_path / "pooled_test_macro_f1.json").exists()
    assert (tmp_path / "pooled_test_macro_f1.md").exists()


def test_classification_sweep_defaults_to_macro_f1_selection() -> None:
    """When the sweep JSON's ``selection_metric`` is ``combined_rmse``
    but the trials carry ``output_mode == "classification"`` on their
    model_config, the aggregator must override to ``macro_f1`` for
    cell selection. The trainer writes ``combined_rmse`` as the
    selection metric regardless of output mode and every classification
    trial reports ``inf`` for combined_rmse, so honoring that field
    collapses the per-cell ranking to an arbitrary order."""

    def _classification_trial(*, fold_id: str, hp: int, macro: float) -> dict[str, object]:
        # 3 classes, 10 predictions, mostly correct so macro_f1 ~ ``macro``.
        n_right = int(round(macro * 10))
        preds = [0] * n_right + [1] * (10 - n_right)
        targets = [0] * 10
        return {
            "architecture": "gru",
            "fold_id": fold_id,
            "seed": 11,
            "hp_combo_id": hp,
            "summary": {
                "model_config": {
                    "hidden_size": 64,
                    "num_layers": 2,
                    "dropout": 0.2,
                    "output_mode": "classification",
                },
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "test_metrics": {
                    "predictions": preds,
                    "targets": targets,
                    "combined_rmse": float("inf"),
                    "classification_breakdown": {
                        "macro_f1": _compute_macro(preds, targets),
                    },
                },
            },
        }

    blob = {
        "trials": [
            _classification_trial(fold_id="wf_fold_1", hp=0, macro=0.3),
            _classification_trial(fold_id="wf_fold_2", hp=0, macro=0.3),
            _classification_trial(fold_id="wf_fold_1", hp=1, macro=0.8),
            _classification_trial(fold_id="wf_fold_2", hp=1, macro=0.8),
        ],
        "selection_metric": "combined_rmse",
    }
    rows = aggregate([blob], n_classes=3, n_resamples=20)
    assert len(rows) == 1
    row = rows[0]
    # Without the classification override the per-cell selector would
    # tie-break on inf combined_rmse and could pick HP 0 (macro_f1 ~
    # 0.3). With the override, HP 1 (macro_f1 ~ 0.8) wins.
    assert row.hp_combo_id == "1"
    assert row.breakdown.macro_f1 > 0.5
