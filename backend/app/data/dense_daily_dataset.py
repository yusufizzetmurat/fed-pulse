"""Dense daily volatility + volume forecasting dataset.

One row per trading day (no event sampling). Loads the daily _market_cache
series, builds HAR-style + cross-market features, realized-vol targets at
h=1/3/5/10 days and a 3-day abnormal-volume target, and emits embargoed
walk-forward splits. FOMC/text features are added in Phase 2.

All features use only information available by the close of day t
(backward rolling / positive shift); all targets are forward windows
(negative shift). The walk-forward embargo (> max horizon) is the
cross-boundary leakage guard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_HORIZONS = (1, 3, 5, 10)
_CACHE_FILES = {
    "GSPC": "GSPC.parquet",
    "VIX": "VIX.parquet",
    "VIX3M": "VIX3M.parquet",
    "TNX": "TNX.parquet",
    "IRX": "IRX.parquet",
}
# Headline feature set: the inner join is bounded by the *latest-starting*
# series, so VIX3M (2006+) is excluded by default — it would silently cut
# the span to 2006 for no feature benefit. Add it only for the 2006+
# ablation. VIX (1990+) is the binding series for the headline window.
DEFAULT_SYMBOLS = ("GSPC", "VIX", "TNX", "IRX")


def load_market_cache(
    cache_dir: Path | str, *, symbols: tuple[str, ...] = DEFAULT_SYMBOLS
) -> dict[str, pd.DataFrame]:
    """Load the requested per-symbol daily parquets that exist, by short name."""

    cache_dir = Path(cache_dir)
    out: dict[str, pd.DataFrame] = {}
    for name in symbols:
        p = cache_dir / _CACHE_FILES[name]
        if p.exists():
            out[name] = pd.read_parquet(p)
    return out


def _align(series_by_name: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Inner-join per-symbol (date, close, volume) on date → wide frame."""

    out: pd.DataFrame | None = None
    for name, df in series_by_name.items():
        cols = df[["date", "close", "volume"]].rename(
            columns={"close": f"close_{name}", "volume": f"volume_{name}"}
        )
        out = cols if out is None else out.merge(cols, on="date", how="inner")
    assert out is not None
    return out.sort_values("date").reset_index(drop=True)


def _realized_vol_targets(
    close: pd.Series, *, horizons: tuple[int, ...] = DEFAULT_HORIZONS
) -> pd.DataFrame:
    """rv_h(t) = sqrt(Σ_{i=1..h} r_{t+i}²) — forward realized vol, no peeking at t."""

    r = np.log(close / close.shift(1))
    r2 = r**2
    cols: dict[str, pd.Series] = {}
    for h in horizons:
        # rolling(h).sum() at index t+h = r2[t+1..t+h]; shift(-h) brings it to t.
        cols[f"rv_{h}"] = np.sqrt(r2.rolling(h).sum().shift(-h))
    return pd.DataFrame(cols)


def _abnormal_volume_target(volume: pd.Series, *, post: int = 3, lookback: int = 30) -> pd.Series:
    """(Σ vol_{t+1..t+post}) / (post · mean(vol_{t-lookback+1..t})) − 1."""

    fwd_sum = volume.rolling(post).sum().shift(-post)
    trailing_mean = volume.rolling(lookback).mean()
    return fwd_sum / (post * trailing_mean) - 1.0


def _build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Backward-only daily features from the aligned wide frame."""

    close = frame["close_GSPC"]
    volume = frame["volume_GSPC"]
    r = np.log(close / close.shift(1))
    r2 = r**2
    feats = pd.DataFrame(index=frame.index)
    # HAR-style trailing realized vol (backward windows)
    feats["rv_lag_1"] = np.sqrt(r2)
    feats["rv_lag_5"] = np.sqrt(r2.rolling(5).sum())
    feats["rv_lag_22"] = np.sqrt(r2.rolling(22).sum())
    # volume
    feats["logvol"] = np.log(volume.clip(lower=1.0))
    feats["vol_ratio_30"] = volume / volume.rolling(30).mean()
    feats["av_lag_1"] = volume.rolling(3).sum() / (3 * volume.rolling(30).mean().shift(3)) - 1.0
    # returns / momentum
    feats["ret_1"] = r
    feats["ret_5"] = np.log(close / close.shift(5))
    feats["ret_22"] = np.log(close / close.shift(22))
    # cross-market (levels + a change), all contemporaneous (known at close t)
    if "close_VIX" in frame:
        feats["vix"] = frame["close_VIX"]
        feats["vix_chg_5"] = frame["close_VIX"] - frame["close_VIX"].shift(5)
    if "close_TNX" in frame:
        feats["tnx"] = frame["close_TNX"]
    if "close_IRX" in frame:
        feats["irx"] = frame["close_IRX"]
    if "close_TNX" in frame and "close_IRX" in frame:
        feats["tnx_minus_irx"] = frame["close_TNX"] - frame["close_IRX"]
    # calendar
    dts = pd.to_datetime(frame["date"])
    dow = dts.dt.dayofweek
    for k in range(5):
        feats[f"dow_{k}"] = (dow == k).astype(float)
    feats["month"] = dts.dt.month.astype(float)
    return feats


def walk_forward_splits(
    n: int, *, n_folds: int = 5, embargo: int = 10
) -> list[tuple[list[int], list[int]]]:
    """Expanding train head + contiguous test block, with an embargo gap.

    The ``embargo`` rows immediately before each test block are dropped from
    nowhere (they simply are not in train, since train ends embargo rows
    before the block) — guaranteeing min(test) − max(train) > embargo so a
    forward target window (≤ embargo days) cannot straddle the split.
    """

    if n < n_folds * (embargo + 2):
        raise ValueError(f"too few rows ({n}) for {n_folds} folds with embargo {embargo}")
    test_size = n // (n_folds + 1)
    start = n - test_size * n_folds
    folds: list[tuple[list[int], list[int]]] = []
    cursor = start
    for _ in range(n_folds):
        train_idx = list(range(0, cursor - embargo))
        test_idx = list(range(cursor, cursor + test_size))
        folds.append((train_idx, test_idx))
        cursor += test_size
    return folds


def build_dataset(
    cache_dir: Path | str,
    *,
    start: str = "1990-01-01",
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return (X features, Y targets, dates), dense daily, NaN rows dropped."""

    series = load_market_cache(cache_dir, symbols=symbols)
    if "GSPC" not in series:
        raise FileNotFoundError("GSPC.parquet not found in market cache")
    frame = _align(series)
    feats = _build_features(frame)
    targets = _realized_vol_targets(frame["close_GSPC"], horizons=horizons)
    targets["av"] = _abnormal_volume_target(frame["volume_GSPC"])
    full = pd.concat([frame["date"], feats, targets], axis=1)
    full = full[full["date"] >= start]
    full = full.dropna().reset_index(drop=True)
    feat_cols = list(feats.columns)
    target_cols = [f"rv_{h}" for h in horizons] + ["av"]
    return full[feat_cols], full[target_cols], full["date"]
