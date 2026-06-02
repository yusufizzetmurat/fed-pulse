"""Unit tests for the volume-head helpers (leak-safety + correctness)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.late_fusion_volume import (
    _forward_target,
    _har_matrix,
    _ols,
    _r2,
    fit_production_artifact,
)


def test_har_features_use_only_past() -> None:
    lv = np.arange(40, dtype=float)  # strictly increasing so lags are checkable
    feats = _har_matrix(lv)
    # before 22 lags exist -> NaN
    assert np.isnan(feats[:22]).all()
    t = 30
    assert feats[t, 0] == lv[t - 1]  # lag-1 strictly before t
    assert feats[t, 1] == pytest.approx(lv[t - 5 : t].mean())
    assert feats[t, 2] == pytest.approx(lv[t - 22 : t].mean())
    # no feature at t uses lv[t] or later
    assert feats[t, 0] < lv[t]


def test_forward_target_is_strictly_future() -> None:
    lv = np.arange(20, dtype=float)
    tgt = _forward_target(lv, h=3)
    t = 5
    # mean of t+1, t+2, t+3 — strictly after t
    assert tgt[t] == pytest.approx(np.mean([lv[6], lv[7], lv[8]]))
    # last h positions undefined
    assert np.isnan(tgt[-3:]).all()


def test_r2_known_values() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert _r2(y, y, baseline=2.0) == pytest.approx(1.0)
    assert _r2(y, np.full(3, 2.0), baseline=2.0) == pytest.approx(0.0)


def test_ols_recovers_linear_relation() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 2))
    y = 3.0 * x[:, 0] - 2.0 * x[:, 1] + 1.0
    pred = _ols(x, y, x)
    assert np.allclose(pred, y, atol=1e-6)


def test_fit_production_artifact_emits_serving_contract(tmp_path: Path) -> None:
    """The deployable artifact must carry every field the serving layer
    reads — har_coef (len 4), calendar_dummy_names/coef (matched length),
    conformal_quantiles at 0.10 + 0.20, and r2_har — for every horizon.
    """

    rng = np.random.default_rng(0)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    log_vol = 21.0 + 0.05 * np.cumsum(rng.standard_normal(n))
    frame = pd.DataFrame({"date": dates, "volume": np.exp(log_vol)})
    vol_path = tmp_path / "vol.parquet"
    frame.to_parquet(vol_path)

    out_path = tmp_path / "volume_har_artifact.json"
    spec = fit_production_artifact(
        vol_path, out_path, n_folds=3, alphas=(0.10, 0.20)
    )
    assert out_path.exists()
    by_h = spec["by_horizon"]
    assert set(by_h) == {"h1", "h5", "h22"}  # type: ignore[arg-type]
    for hk in ("h1", "h5", "h22"):
        row = by_h[hk]  # type: ignore[index]
        assert len(row["har_coef"]) == 4, "intercept + d + w + m"
        assert row["calendar_dummy_names"], "must carry seasonality names"
        assert len(row["calendar_dummy_coef"]) == len(row["calendar_dummy_names"])
        quants = row["conformal_quantiles"]
        assert "0.10" in quants and "0.20" in quants
        assert quants["0.10"] >= quants["0.20"] >= 0.0, (
            "90% half-width must be at least 80%"
        )
        assert "r2_har" in row
