"""Per-asset HAR-tercile baseline on ^NDX and ^DJI.

Resolves the SPX-only routing bug in `app/services/har_tercile.py` by giving
us forward-vol-regime macro-F1 numbers on the other two paper-track equity
indices the canonical TP carries. The protocol mirrors the canonical
`fed_comms_regime.run()` HAR-tercile baseline exactly:

  1. Daily realized variance proxy: r_t^2 (squared log-return). Canonical SPX
     uses 5-min intraday bars; yfinance only carries daily closes for NDX/DJI,
     so daily-r2 is the best proxy available. This is the same proxy the
     forecaster's checkpoint path uses on retail tickers.
  2. HAR lags in log-space: `_har_lags(log(rv + EPS))` -> [daily, weekly-mean,
     monthly-mean]. Same as `app.data.intraday_rv_forecast._har_lags`.
  3. Forward target: log of mean rv over the next h trading days (per
     `app.data.fed_comms_dataset._forward_target` with `is_log=True`).
  4. Walk-forward folds: dense `walk_forward_splits(n_valid, n_folds=5,
     embargo=23)` on the index of finite-target rows. Train slice fits OLS
     HAR; q33/q67 cutoffs derived from train-slice forward targets.
  5. Macro-F1 reported per fold (mean +- std across 5 folds) AND pooled,
     matching how the canonical 0.687/0.685/0.654 numbers are computed.
  6. KS-stat diagnostic on the train vs test forward-target distribution per
     fold (per the arm's failure-mode brief).

Seeds: OLS is deterministic, so the official seed set {11, 29, 47, 71, 97}
yields identical numbers. We document this and report n_seeds=1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp  # noqa: I001

# Match canonical constants from `app.data.intraday_rv_forecast`.
_EPS = 1e-12
_HORIZONS = (1, 5, 22)
_N_FOLDS = 5
_EMBARGO = max(_HORIZONS) + 1  # 23, matches `fed_comms_regime.run()`.

# Canonical SPX HAR-tercile baseline (wiki section 20; pooled-5-fold macro-F1).
_SPX_BASELINE = {1: 0.687, 5: 0.685, 22: 0.654}


def _har_lags(log_rv: np.ndarray) -> np.ndarray:
    """[logRV_t, mean last-5, mean last-22] -- mirrors app.data.intraday_rv_forecast."""
    n = len(log_rv)
    daily = log_rv.copy()
    weekly = np.array([log_rv[max(0, i - 4) : i + 1].mean() for i in range(n)])
    monthly = np.array([log_rv[max(0, i - 21) : i + 1].mean() for i in range(n)])
    return np.column_stack([daily, weekly, monthly])


def _forward_target(raw: np.ndarray, h: int) -> np.ndarray:
    """Forward log-mean over t+1..t+h; NaN past the end. is_log=True per canonical."""
    n = len(raw)
    out = np.full(n, np.nan)
    for t in range(n - h):
        m = float(raw[t + 1 : t + 1 + h].mean())
        out[t] = float(np.log(m + _EPS))
    return out


def _walk_forward_splits(
    n: int, *, n_folds: int = _N_FOLDS, embargo: int = _EMBARGO
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Mirror app.data.dense_daily_dataset.walk_forward_splits."""
    if n < n_folds * (embargo + 2):
        raise ValueError(f"too few rows ({n}) for {n_folds} folds with embargo {embargo}")
    test_size = n // (n_folds + 1)
    start = n - test_size * n_folds
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    cursor = start
    for _ in range(n_folds):
        train_idx = np.arange(0, cursor - embargo)
        test_idx = np.arange(cursor, cursor + test_size)
        folds.append((train_idx, test_idx))
        cursor += test_size
    return folds


def _fit_predict_ols(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray) -> np.ndarray:
    a_tr = np.column_stack([np.ones(len(x_tr)), x_tr])
    coef, *_ = np.linalg.lstsq(a_tr, y_tr, rcond=None)
    a_te = np.column_stack([np.ones(len(x_te)), x_te])
    return a_te @ coef


def _labels(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(values, thresholds).astype(np.int64)


def _macro_f1(true: np.ndarray, pred: np.ndarray, n_classes: int = 3) -> float:
    f1s = []
    for c in range(n_classes):
        tp = float(np.sum((pred == c) & (true == c)))
        fp = float(np.sum((pred == c) & (true != c)))
        fn = float(np.sum((pred != c) & (true == c)))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


@dataclass
class AssetFrame:
    symbol: str
    date: np.ndarray  # ISO strings, sorted ascending
    rv: np.ndarray   # daily realized variance proxy = log_return^2


def _load_asset(parquet_path: Path, symbol_label: str) -> AssetFrame:
    df = pd.read_parquet(parquet_path)
    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float).to_numpy()
    log_ret = np.diff(np.log(close), prepend=np.nan)
    r2 = log_ret**2
    # Drop the first NaN (no return for t=0); keep alignment with dates.
    valid = np.isfinite(r2)
    return AssetFrame(
        symbol=symbol_label,
        date=df["date"].astype(str).to_numpy()[valid],
        rv=r2[valid],
    )


def _per_horizon_eval(asset: AssetFrame, h: int) -> dict[str, object]:
    log_rv = np.log(asset.rv + _EPS)
    har = _har_lags(log_rv)
    fwd = _forward_target(asset.rv, h)

    # Match _assemble's `valid` mask: finite targets + finite HAR lags.
    valid = np.isfinite(fwd) & np.isfinite(har).all(axis=1)
    idx_all = np.where(valid)[0]
    folds = _walk_forward_splits(len(idx_all))

    har_v = har[idx_all]
    fwd_v = fwd[idx_all]
    date_v = asset.date[idx_all]

    per_fold: list[dict[str, float]] = []
    pooled_true: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    for fi, (tr, te) in enumerate(folds, start=1):
        y_tr, y_te = fwd_v[tr], fwd_v[te]
        x_tr, x_te = har_v[tr], har_v[te]
        thr = np.quantile(y_tr, [1.0 / 3.0, 2.0 / 3.0])
        y_te_pred = _fit_predict_ols(x_tr, y_tr, x_te)
        true_lab = _labels(y_te, thr)
        pred_lab = _labels(y_te_pred, thr)
        f1 = _macro_f1(true_lab, pred_lab)
        ks_stat, ks_p = ks_2samp(y_tr, y_te)
        per_fold.append(
            {
                "fold": fi,
                "train_start": str(date_v[int(tr[0])]),
                "train_end": str(date_v[int(tr[-1])]),
                "test_start": str(date_v[int(te[0])]),
                "test_end": str(date_v[int(te[-1])]),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "macro_f1": f1,
                "ks_train_vs_test_fwd": float(ks_stat),
                "ks_pvalue": float(ks_p),
                "q33": float(thr[0]),
                "q67": float(thr[1]),
            }
        )
        pooled_true.append(true_lab)
        pooled_pred.append(pred_lab)

    f1s = np.array([row["macro_f1"] for row in per_fold])
    pooled_t = np.concatenate(pooled_true)
    pooled_p = np.concatenate(pooled_pred)
    return {
        "horizon": h,
        "per_fold": per_fold,
        "fold_macro_f1_mean": float(f1s.mean()),
        "fold_macro_f1_std": float(f1s.std(ddof=0)),
        "pooled_macro_f1": _macro_f1(pooled_t, pooled_p),
        "n_pooled": int(len(pooled_t)),
        "ks_train_vs_test_fwd_mean": float(
            np.mean([row["ks_train_vs_test_fwd"] for row in per_fold])
        ),
    }


def main() -> int:
    repo = Path("/data/external/yfinance")
    out_dir = Path("/data/artifacts/har_tercile_per_asset_har")
    out_dir.mkdir(parents=True, exist_ok=True)

    asset_paths = {
        "^NDX": repo / "NDX.parquet",
        "^DJI": repo / "DJI.parquet",
        # ^GSPC daily-r^2 comparator (apples-to-apples vs NDX/DJI). The
        # canonical SPX baseline 0.687/0.685/0.654 uses 5-min intraday RV;
        # this daily-r^2 SPX number is what NDX/DJI should be compared
        # against to isolate the per-asset effect from the RV-proxy effect.
        "^GSPC_daily": Path("/data/processed/tp_v3_full_rebuild_2026_05_30/_market_cache/GSPC.parquet"),
    }

    result: dict[str, object] = {
        "arm_key": "per_asset_har",
        "protocol": {
            "rv_proxy": "daily log_return squared (yfinance daily closes)",
            "har_space": "log",
            "forward_target": "log-mean over t+1..t+h",
            "folds": "dense walk_forward_splits, n_folds=5, embargo=23",
            "macro_f1": "pooled across 5 folds (canonical); also report per-fold mean+/-std",
            "horizons": list(_HORIZONS),
            "seeds": "OLS HAR is deterministic; n_seeds=1",
        },
        "baselines_spx": _SPX_BASELINE,
        "by_asset": {},
    }

    for sym, path in asset_paths.items():
        asset = _load_asset(path, sym)
        per_h = {}
        for h in _HORIZONS:
            per_h[f"h{h}"] = _per_horizon_eval(asset, h)
        result["by_asset"][sym] = {
            "first_date": str(asset.date[0]),
            "last_date": str(asset.date[-1]),
            "n_days": int(len(asset.date)),
            "by_horizon": per_h,
            "delta_vs_spx_baseline_pooled": {
                f"h{h}": float(per_h[f"h{h}"]["pooled_macro_f1"] - _SPX_BASELINE[h])
                for h in _HORIZONS
            },
        }

    # Headline numbers for the synthesize step: pooled macro-F1 per asset per
    # horizon, plus a per-fold mean (for "mean+/-std across 5 folds").
    headline = {
        sym: {
            f"h{h}": {
                "pooled_macro_f1": result["by_asset"][sym]["by_horizon"][f"h{h}"][
                    "pooled_macro_f1"
                ],
                "fold_macro_f1_mean": result["by_asset"][sym]["by_horizon"][f"h{h}"][
                    "fold_macro_f1_mean"
                ],
                "fold_macro_f1_std": result["by_asset"][sym]["by_horizon"][f"h{h}"][
                    "fold_macro_f1_std"
                ],
            }
            for h in _HORIZONS
        }
        for sym in asset_paths
    }
    result["headline"] = headline

    out_path = out_dir / "result.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps(headline, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
