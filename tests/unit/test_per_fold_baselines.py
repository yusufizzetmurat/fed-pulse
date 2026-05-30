"""Unit tests for app.eval.per_fold_baselines (#500)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.eval.per_fold_baselines import (
    CLASS_NAMES,
    compute_per_fold_baselines,
    majority_baseline_f1,
    stratified_random_f1,
)


def test_majority_baseline_returns_nonzero_f1() -> None:
    f1 = majority_baseline_f1([0, 0, 0, 1, 2], [0, 0, 1, 2])
    assert 0.0 < f1 <= 1.0


def test_majority_baseline_empty_returns_zero() -> None:
    assert majority_baseline_f1([], [0, 1]) == 0.0
    assert majority_baseline_f1([0, 1], []) == 0.0


def test_majority_baseline_all_mode_class_in_test() -> None:
    f1 = majority_baseline_f1([0, 0, 1, 2], [0, 0, 0])
    assert f1 == pytest.approx(1.0)


def test_stratified_random_f1_near_chance() -> None:
    train = [0, 0, 1, 1, 2, 2]
    test = list(range(3)) * 6
    f1 = stratified_random_f1(train, test, n_seeds=300, base_seed=0)
    assert 0.15 <= f1 <= 0.50


def test_stratified_random_f1_deterministic() -> None:
    train = [0, 1, 2] * 4
    test = [0, 1, 2] * 4
    a = stratified_random_f1(train, test, n_seeds=100, base_seed=7)
    b = stratified_random_f1(train, test, n_seeds=100, base_seed=7)
    assert a == b


def test_stratified_random_f1_empty_returns_zero() -> None:
    assert stratified_random_f1([], [0, 1]) == 0.0


def _write_fixture_package(tmp_path: Path) -> tuple[Path, str]:
    import pandas as pd

    pkg_id = "test_pkg"
    pkg_dir = tmp_path / pkg_id
    pkg_dir.mkdir(parents=True)

    rows = []
    for i in range(15):
        vol = 0.004 + i * 0.002
        if i < 9:
            date = f"2020-01-{i + 1:02d}"
        elif i < 12:
            date = f"2020-02-{i - 8:02d}"
        else:
            date = f"2020-03-{i - 11:02d}"
        rows.append({"event_date": date, "forward_realized_vol_10d": vol})

    pd.DataFrame(rows).to_parquet(pkg_dir / "events.parquet", index=False)
    manifest = {
        "folds": [{
            "fold_id": "wf_fold_1",
            "train_start": "2020-01-01",
            "train_end": "2020-01-09",
            "val_start": "2020-02-01",
            "val_end": "2020-02-03",
            "test_start": "2020-03-01",
            "test_end": "2020-03-03",
        }]
    }
    (pkg_dir / "fold_manifest_expanding_walk_forward.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return tmp_path, pkg_id


def test_end_to_end_fixture(tmp_path: Path) -> None:
    processed_root, pkg_id = _write_fixture_package(tmp_path)
    result = compute_per_fold_baselines(
        pkg_id,
        processed_root=processed_root,
        sweep_artefact=Path("/nonexistent.json"),
        n_stratified_seeds=50,
    )
    assert result["training_package_id"] == pkg_id
    folds = result["folds"]
    assert len(folds) == 1
    fold = folds[0]
    assert fold["fold_id"] == "wf_fold_1"
    assert 0.0 <= fold["majority_baseline_f1"] <= 1.0
    assert 0.0 <= fold["stratified_random_f1"] <= 1.0
    assert fold["encoder_f1"] is None
    for partition in ("train", "val", "test"):
        for cls in CLASS_NAMES:
            assert cls in fold["class_distribution"][partition]
