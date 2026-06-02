"""Pure-function tests for the deployable banded RV forecaster."""

from __future__ import annotations

import numpy as np

from app.data import intraday_rv_production as p


def test_conformal_quantile_covers_nominal_on_synthetic() -> None:
    # iid calibration + test draws from the same dist: a 90% conformal band
    # should cover ~90% of test points (conformal validity).
    rng = np.random.default_rng(0)
    pred = np.zeros(4000)
    cal_scores = np.abs(rng.standard_normal(2000))  # |residual| nonconformity
    q = p._conformal_quantile(cal_scores, alpha=0.1)
    test_true = rng.standard_normal(4000)  # residual = true - 0 = true
    cov = p._coverage(test_true, pred, q)
    assert 0.86 <= cov <= 0.94  # ~90% nominal, tolerance for sampling noise


def test_walk_forward_conformal_covers_nominal_prior_folds_only() -> None:
    # Emulate the run() walk-forward band scheme on synthetic iid folds: fold k is
    # banded by the |residual| quantile of folds 1..k-1 (prior only, fold 1 skipped).
    # Predictions are 0, so residual = true; a 90% band should cover ~90% pooled.
    rng = np.random.default_rng(7)
    folds = [rng.standard_normal(500) for _ in range(6)]
    cal_resid: list[float] = []
    hits = 0.0
    n = 0
    for true in folds:
        pred = np.zeros_like(true)
        if cal_resid:
            q = p._conformal_quantile(np.asarray(cal_resid), alpha=0.1)
            hits += p._coverage(true, pred, q) * len(true)
            n += len(true)
        cal_resid.extend(np.abs(true - pred).tolist())
    cov = hits / n
    assert 0.86 <= cov <= 0.94  # ~90% nominal, prospective coverage, no leakage


def test_conformal_quantile_monotone_in_alpha() -> None:
    scores = np.abs(np.random.default_rng(1).standard_normal(500))
    q80 = p._conformal_quantile(scores, alpha=0.2)
    q90 = p._conformal_quantile(scores, alpha=0.1)
    assert q90 > q80  # tighter mis-coverage ⇒ wider band


def test_conformal_quantile_small_sample_falls_back_to_max() -> None:
    scores = np.array([0.5, 1.0, 2.0])  # n=3; ceil(4*0.9)=4 > n → widest score
    assert p._conformal_quantile(scores, alpha=0.1) == 2.0
    assert np.isnan(p._conformal_quantile(np.array([]), alpha=0.1))


def test_coverage_empty_is_nan() -> None:
    assert np.isnan(p._coverage(np.array([]), np.array([]), 1.0))


def test_ensemble_mean_shape_and_value() -> None:
    # The ensemble point forecast is the per-seed mean across axis 0.
    per_seed = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    mean = per_seed.mean(axis=0)
    assert mean.shape == (3,)
    assert np.allclose(mean, [2.0, 3.0, 4.0])


def test_build_full_column_layout() -> None:
    import pandas as pd

    n = 40
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "date": [f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n)],
            "rv": np.abs(rng.standard_normal(n)) * 1e-4 + 1e-5,
            "rs_pos": np.abs(rng.standard_normal(n)) * 1e-5,
            "rs_neg": np.abs(rng.standard_normal(n)) * 1e-5,
            "bv": np.abs(rng.standard_normal(n)) * 1e-4,
            "rq": np.abs(rng.standard_normal(n)) * 1e-8,
            "rskew": rng.standard_normal(n),
            "rkurt": np.abs(rng.standard_normal(n)) * 5,
            "parkinson": np.abs(rng.standard_normal(n)) * 1e-4,
            "rvol": np.abs(rng.standard_normal(n)) * 1e6,
        }
    )
    rv, log_rv, full = p._build_full(df)
    assert rv.shape == (n,)
    assert log_rv.shape == (n,)
    # 3 HAR cols + 7 realized measures + log(rvol) = 11.
    assert full.shape == (n, 11)
    # HAR daily occupies col 0 and equals log(rv).
    assert np.allclose(full[:, 0], log_rv)
