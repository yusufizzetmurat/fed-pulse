"""Tests for the dense forecast baselines + metrics (pure units)."""

from __future__ import annotations

import numpy as np
import pytest

from app.data import dense_forecast_train as dft


def test_oos_r2_zero_for_mean_and_one_for_perfect() -> None:
    true = np.array([1.0, 2.0, 3.0, 4.0])
    base = np.full_like(true, true.mean())
    assert dft._oos_r2(base.copy(), true, base) == pytest.approx(0.0)
    assert dft._oos_r2(true.copy(), true, base) == pytest.approx(1.0)


def test_rmse() -> None:
    assert dft._rmse(np.array([1.0, 2.0]), np.array([1.0, 4.0])) == pytest.approx(np.sqrt(2.0))


def test_spearman_monotonic() -> None:
    assert dft._spearman(np.array([1.0, 2, 3, 4]), np.array([10.0, 20, 30, 40])) == pytest.approx(
        1.0
    )


def test_fit_predict_ols_recovers_linear_map() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 2))
    y = 3.0 + 2.0 * x[:, 0] - 1.5 * x[:, 1]  # exact linear, no noise
    pred = dft._fit_predict_ols(x[:150], y[:150], x[150:])
    assert np.allclose(pred, y[150:], atol=1e-6)


def test_har_baseline_predicts_persistent_vol() -> None:
    # rv that equals its own 1-day lag -> log-HAR should recover it well
    rng = np.random.default_rng(1)
    lag1 = np.abs(rng.normal(0.01, 0.003, 300)) + 0.005
    har = np.column_stack([lag1, lag1, lag1])  # all three lags identical
    target_log = np.log(lag1 + dft._EPS)
    pred = dft.har_baseline(har[:200], target_log[:200], har[200:])
    # near-perfect since target is a deterministic function of the (log) lag
    assert dft._oos_r2(pred, target_log[200:], np.full(100, target_log[:200].mean())) > 0.9


def test_bootstrap_r2_ci_orders() -> None:
    rng = np.random.default_rng(2)
    true = rng.random(80)
    pred = true + rng.normal(0, 0.05, 80)
    base = np.full(80, float(true.mean()))
    lo, hi = dft._bootstrap_r2_ci(pred, true, base, n_boot=200)
    assert lo <= hi
