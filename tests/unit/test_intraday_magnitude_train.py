"""Tests for the intraday magnitude regression harness (pure units)."""

from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from app.data import intraday_magnitude_train as imt


def test_oos_r2_zero_for_mean_predictor() -> None:
    true = np.array([1.0, 2.0, 3.0, 4.0])
    train_mean = 2.5
    pred = np.full_like(true, train_mean)
    assert imt._oos_r2(pred, true, baseline_pred=train_mean) == pytest.approx(0.0)


def test_oos_r2_one_for_perfect() -> None:
    true = np.array([1.0, 2.0, 3.0])
    assert imt._oos_r2(true.copy(), true, baseline_pred=2.0) == pytest.approx(1.0)


def test_oos_r2_negative_when_worse_than_mean() -> None:
    true = np.array([1.0, 2.0, 3.0])
    pred = np.array([3.0, 2.0, 1.0])
    assert imt._oos_r2(pred, true, baseline_pred=2.0) < 0.0


def test_rmse() -> None:
    assert imt._rmse(np.array([1.0, 2.0]), np.array([1.0, 4.0])) == pytest.approx(np.sqrt(2.0))


def test_spearman_monotonic_is_one() -> None:
    pred = np.array([0.1, 0.2, 0.3, 0.4])
    true = np.array([10.0, 20.0, 30.0, 40.0])
    assert imt._spearman(pred, true) == pytest.approx(1.0)


def test_build_magnitude_arrays_targets_abs_return(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "event_date": ["2024-03-20", "2024-01-31"],
            "text": ["d", "h"],
            "pre_close": [[100.0, 101.0, 102.0], [50.0, 49.0, 50.0]],
            "pre_volume": [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
            "mag_immediate": [0.012, 0.004],
            "mag_delayed": [0.020, 0.006],
        }
    )
    x, text, y = imt.build_magnitude_arrays(
        df, "immediate", embed_fn=lambda s: np.zeros(768, dtype=np.float32)
    )
    assert x.shape == (2, 3, 3) and text.shape == (2, 768)
    # sorted by date: 2024-01-31 first -> mag 0.004
    assert y.tolist() == pytest.approx([0.004, 0.012])


def test_r2_bootstrap_ci_brackets_point() -> None:
    rng = np.random.default_rng(0)
    true = rng.random(40)
    pred = true + rng.normal(0, 0.1, 40)
    base = np.full(40, float(true.mean()))
    lo, point, hi = imt._r2_bootstrap_ci(pred, true, base, seed=11, n_boot=200)
    assert lo <= point <= hi
    assert point > 0.0  # pred tracks true, beats the mean
