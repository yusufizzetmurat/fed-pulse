"""Unit tests for per-fold class baselines (#500)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_per_fold_class_baselines import (
    _fit_quantile_cutoffs,
    _label_with_cutoffs,
    _macro_f1,
    compute_per_fold_baselines,
)


def test_fit_quantile_cutoffs_returns_empty_below_n_classes() -> None:
    arr = np.array([0.5, 0.6])
    assert _fit_quantile_cutoffs(arr, n_classes=3) == ()


def test_fit_quantile_cutoffs_matches_numpy_tertile() -> None:
    arr = np.arange(99.0)
    cutoffs = _fit_quantile_cutoffs(arr, n_classes=3)
    assert len(cutoffs) == 2
    expected = tuple(float(c) for c in np.quantile(arr, [1 / 3, 2 / 3]))
    assert cutoffs == pytest.approx(expected)


def test_label_with_cutoffs_maps_low_mid_high() -> None:
    cutoffs = (0.5, 1.0)
    vals = np.array([0.1, 0.6, 1.5, np.nan])
    labels = _label_with_cutoffs(vals, cutoffs)
    assert list(labels) == [0, 1, 2, -1]


def test_macro_f1_matches_sklearn_for_known_case() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    sk = pytest.importorskip("sklearn.metrics")
    expected = sk.f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2])
    assert _macro_f1(y_true, y_pred, n_classes=3) == pytest.approx(
        float(expected), abs=1e-9
    )


def test_macro_f1_all_correct_is_one() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = y_true.copy()
    assert _macro_f1(y_true, y_pred, n_classes=3) == 1.0


def test_macro_f1_majority_baseline_on_balanced_three_classes() -> None:
    """Predict class 0 on a balanced 3-class set: precision_0 = 1/3,
    recall_0 = 1, F1_0 = 0.5; F1_1 = F1_2 = 0; macro F1 = 1/6.
    """

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.zeros_like(y_true)
    assert _macro_f1(y_true, y_pred, n_classes=3) == pytest.approx(1 / 6)


def _write_synthetic_tp(
    tmp_path: Path, n: int = 60
) -> tuple[Path, Path]:
    """Two-fold synthetic TP with a clean three-cluster vol distribution
    so the tertile cutoffs hit at well-separated values.
    """

    rng = np.random.default_rng(42)
    low = rng.normal(0.005, 0.0005, n // 3)
    mid = rng.normal(0.010, 0.0005, n // 3)
    high = rng.normal(0.020, 0.0005, n - 2 * (n // 3))
    vols = np.concatenate([low, mid, high])
    rng.shuffle(vols)
    dates = pd.date_range("2020-01-01", periods=len(vols), freq="W-WED")
    df = pd.DataFrame(
        {
            "event_date": dates.strftime("%Y-%m-%d"),
            "forward_realized_vol_10d": vols.astype(float),
        }
    )
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    train = df["event_date"].iloc[: 2 * len(vols) // 3].tolist()
    test = df["event_date"].iloc[2 * len(vols) // 3 :].tolist()
    manifest_path = tmp_path / "fold_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "folds": [
                    {"fold_id": "wf_fold_1", "train": train, "test": test}
                ]
            }
        ),
        encoding="utf-8",
    )
    return parquet, manifest_path


def test_compute_per_fold_baselines_emits_one_row_per_fold(tmp_path: Path) -> None:
    parquet, manifest = _write_synthetic_tp(tmp_path)
    report = compute_per_fold_baselines(
        events_parquet=parquet, fold_manifest=manifest, seeds=(1, 2, 3)
    )
    assert len(report["per_fold"]) == 1
    row = report["per_fold"][0]
    assert row["fold_id"] == "wf_fold_1"
    assert row["n_test"] > 0
    assert row["majority_class_idx"] in (0, 1, 2)
    assert 0.0 <= row["majority_class_f1"] <= 1.0
    assert 0.0 <= row["stratified_random_f1"] <= 1.0


def test_compute_handles_empty_test_fold(tmp_path: Path) -> None:
    parquet, _ = _write_synthetic_tp(tmp_path)
    manifest_path = tmp_path / "manifest_empty_test.json"
    manifest_path.write_text(
        json.dumps(
            {
                "folds": [
                    {
                        "fold_id": "wf_fold_x",
                        "train": ["2020-01-01"],
                        "test": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = compute_per_fold_baselines(
        events_parquet=parquet, fold_manifest=manifest_path
    )
    row = report["per_fold"][0]
    assert row["n_test"] == 0
    assert row["majority_class_f1"] is None
    assert row["stratified_random_f1"] is None


def test_raises_when_events_missing_required_columns(tmp_path: Path) -> None:
    df = pd.DataFrame({"event_date": ["2020-01-01"]})
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"folds": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="forward_realized_vol_10d"):
        compute_per_fold_baselines(
            events_parquet=parquet, fold_manifest=manifest_path
        )
