"""Tests for the dense daily dataset builder (pure units)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from app.data import dense_daily_dataset as dds


def _series(dates, close, volume, symbol="^GSPC"):
    return pd.DataFrame({"symbol": symbol, "date": dates, "close": close, "volume": volume})


def test_load_and_align_inner_joins_on_date() -> None:
    d = pd.bdate_range("2020-01-01", periods=5).strftime("%Y-%m-%d").tolist()
    gspc = _series(d, [100, 101, 102, 101, 103], [10, 11, 12, 11, 13])
    vix = _series(d[:4], [20, 21, 19, 22], [0, 0, 0, 0], symbol="^VIX")
    frame = dds._align({"GSPC": gspc, "VIX": vix})
    assert "close_GSPC" in frame.columns and "close_VIX" in frame.columns
    assert len(frame) == 4  # inner join on the 4 shared dates


def test_realized_vol_targets() -> None:
    close = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111], dtype=float)
    out = dds._realized_vol_targets(close, horizons=(1, 3))
    r = np.log(close / close.shift(1))
    assert out["rv_1"].iloc[0] == pytest.approx(abs(r.iloc[1]))
    assert out["rv_3"].iloc[0] == pytest.approx(np.sqrt((r.iloc[1:4] ** 2).sum()))


def test_realized_vol_no_lookahead_tail_is_nan() -> None:
    close = pd.Series(np.linspace(100, 120, 12))
    out = dds._realized_vol_targets(close, horizons=(3,))
    # last 3 rows cannot have a complete forward 3-day window
    assert out["rv_3"].iloc[-3:].isna().all()


def test_abnormal_volume_target() -> None:
    vol = pd.Series([10.0] * 30 + [40.0, 40.0, 40.0, 10.0])
    av = dds._abnormal_volume_target(vol, post=3, lookback=30)
    assert av.iloc[29] == pytest.approx((120.0 / (3 * 10.0)) - 1.0)


def test_features_have_har_lags_and_calendar() -> None:
    d = pd.bdate_range("2019-01-01", periods=60).strftime("%Y-%m-%d").tolist()
    close = np.linspace(100, 130, 60)
    frame = pd.DataFrame(
        {
            "date": d,
            "close_GSPC": close,
            "volume_GSPC": np.full(60, 1e6),
            "close_VIX": np.full(60, 20.0),
            "volume_VIX": 0.0,
            "close_TNX": 2.0,
            "volume_TNX": 0.0,
            "close_IRX": 1.0,
            "volume_IRX": 0.0,
        }
    )
    feats = dds._build_features(frame)
    assert {"rv_lag_1", "rv_lag_5", "rv_lag_22", "logvol", "ret_5", "vix", "dow_0", "month"} <= set(
        feats.columns
    )
    assert feats.iloc[30][["rv_lag_22", "ret_22", "tnx_minus_irx"]].notna().all()


def test_walk_forward_embargo_enforced() -> None:
    folds = dds.walk_forward_splits(1000, n_folds=4, embargo=10)
    assert len(folds) == 4
    for tr, te in folds:
        assert max(tr) < min(te)
        assert min(te) - max(tr) > 10
    # train heads expand
    assert [len(t) for t, _ in folds] == sorted(len(t) for t, _ in folds)


def test_walk_forward_too_few_rows_raises() -> None:
    with pytest.raises(ValueError, match="too few"):
        dds.walk_forward_splits(20, n_folds=5, embargo=10)


def test_build_dataset_end_to_end_no_nan(tmp_path) -> None:
    d = pd.bdate_range("1990-01-01", periods=400).strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    vol = rng.uniform(1e6, 2e6, 400)
    for name in ("GSPC", "VIX", "TNX", "IRX"):
        c = close if name == "GSPC" else np.full(400, {"VIX": 20.0, "TNX": 2.0, "IRX": 1.0}[name])
        _series(d, c, vol if name == "GSPC" else 0.0, symbol=name).to_parquet(
            tmp_path / f"{name}.parquet", index=False
        )
    X, Y, dates = dds.build_dataset(tmp_path, start="1990-01-01")
    assert not X.isna().any().any()
    assert not Y.isna().any().any()
    assert list(Y.columns) == ["rv_1", "rv_3", "rv_5", "rv_10", "av"]
    assert len(X) == len(Y) == len(dates) and len(X) > 200
