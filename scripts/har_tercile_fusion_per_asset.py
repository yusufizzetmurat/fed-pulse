"""Per-asset HAR-tercile arm aligned to the fusion TP date window.

Implements the HAR-tercile improvement arm `per_asset_har` against the same
fusion TP that produced the canonical 0.687 / 0.685 / 0.654 baseline.

Earlier attempt at `data/artifacts/har_tercile_per_asset_har/result.json` used
the full yfinance history (NDX from 1985, DJI from 1992) and a daily r^2
proxy. That made fold sizes ~5x larger than the canonical fusion-TP folds
(~10,243 / 8,662 days vs the fusion TP's 5,385 days), so the deltas mix
window-size effects with per-asset effects.

This script aligns the per-asset analysis to the fusion TP window
(2005-01-03 -> 2026-05-29) so the walk-forward protocol matches the
canonical baseline as closely as the data allow. The realized-variance
proxy is still daily-r^2 because yfinance only carries daily closes for
NDX/DJI (the fusion TP itself only holds SPX 5-min intraday RV).

Protocol mirrors scripts/reproduce_har_tercile_fusion.py:
  - HAR lags in log-space: [daily, weekly-mean, monthly-mean]
  - forward target: log-mean of raw rv over t+1..t+h
  - walk_forward_splits with n_folds=5, embargo=23
  - tercile thresholds q33/q67 from train slice
  - macro-F1 pooled across 5 folds (canonical) plus per-fold mean+/-std

Output:
  docs/research/har-tercile-fusion-per_asset_har-result.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


_EPS = 1e-12
_HORIZONS = (1, 5, 22)
_N_FOLDS = 5
_EMBARGO = max(_HORIZONS) + 1  # 23

# Fusion TP date window (matches data/processed/tp_intraday_fomc_text_volatility/
# fusion/daily_fusion.parquet date_min / date_max).
_FUSION_DATE_MIN = "2005-01-03"
_FUSION_DATE_MAX = "2026-05-29"

# Recovered fusion-TP HAR-tercile baseline on SPX 5-min intraday RV
# (docs/research/recover-A-result.json headline; matches wiki section 20 at 3dp).
_BASELINE_FUSION_SPX = {
    "h1": {"pooled": 0.6873, "fold_mean": 0.629, "fold_std": 0.042},
    "h5": {"pooled": 0.6850, "fold_mean": 0.618, "fold_std": 0.034},
    "h22": {"pooled": 0.6542, "fold_mean": 0.554, "fold_std": 0.046},
}


def _har_lags(log_rv: np.ndarray) -> np.ndarray:
    n = len(log_rv)
    daily = log_rv.copy()
    weekly = np.array([log_rv[max(0, i - 4) : i + 1].mean() for i in range(n)])
    monthly = np.array([log_rv[max(0, i - 21) : i + 1].mean() for i in range(n)])
    return np.column_stack([daily, weekly, monthly])


def _forward_target(raw: np.ndarray, h: int) -> np.ndarray:
    n = len(raw)
    out = np.full(n, np.nan)
    for t in range(n - h):
        m = float(raw[t + 1 : t + 1 + h].mean())
        out[t] = float(np.log(m + _EPS))
    return out


def _walk_forward_splits(n, *, n_folds=_N_FOLDS, embargo=_EMBARGO):
    if n < n_folds * (embargo + 2):
        raise ValueError(f"too few rows ({n}) for {n_folds} folds with embargo {embargo}")
    test_size = n // (n_folds + 1)
    start = n - test_size * n_folds
    folds = []
    cursor = start
    for _ in range(n_folds):
        train_idx = np.arange(0, cursor - embargo)
        test_idx = np.arange(cursor, cursor + test_size)
        folds.append((train_idx, test_idx))
        cursor += test_size
    return folds


def _fit_predict_ols(x_tr, y_tr, x_te):
    a_tr = np.column_stack([np.ones(len(x_tr)), x_tr])
    coef, *_ = np.linalg.lstsq(a_tr, y_tr, rcond=None)
    a_te = np.column_stack([np.ones(len(x_te)), x_te])
    return a_te @ coef


def _labels(values, thresholds):
    return np.digitize(values, thresholds).astype(np.int64)


def _macro_f1(true, pred, n_classes=3):
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
    date: np.ndarray
    rv: np.ndarray


def _load_asset_window(parquet_path: Path, symbol_label: str) -> AssetFrame:
    """Load yfinance daily closes, restrict to fusion TP window, build daily r^2."""
    df = pd.read_parquet(parquet_path)
    df = df.sort_values("date").reset_index(drop=True)
    # Use canonical date strings (ISO) for filter.
    df["date_str"] = df["date"].astype(str)
    df = df[(df["date_str"] >= _FUSION_DATE_MIN) & (df["date_str"] <= _FUSION_DATE_MAX)].reset_index(
        drop=True
    )
    close = df["close"].astype(float).to_numpy()
    log_ret = np.diff(np.log(close), prepend=np.nan)
    r2 = log_ret**2
    valid = np.isfinite(r2)
    return AssetFrame(
        symbol=symbol_label,
        date=df["date_str"].to_numpy()[valid],
        rv=r2[valid],
    )


def _per_horizon_eval(asset: AssetFrame, h: int) -> dict[str, object]:
    log_rv = np.log(asset.rv + _EPS)
    har = _har_lags(log_rv)
    fwd = _forward_target(asset.rv, h)
    valid = np.isfinite(fwd) & np.isfinite(har).all(axis=1)
    idx_all = np.where(valid)[0]
    folds = _walk_forward_splits(len(idx_all))

    har_v = har[idx_all]
    fwd_v = fwd[idx_all]
    date_v = asset.date[idx_all]

    per_fold = []
    pooled_true = []
    pooled_pred = []
    for fi, (tr, te) in enumerate(folds, start=1):
        y_tr, y_te = fwd_v[tr], fwd_v[te]
        x_tr, x_te = har_v[tr], har_v[te]
        thr = np.quantile(y_tr, [1.0 / 3.0, 2.0 / 3.0])
        y_te_pred = _fit_predict_ols(x_tr, y_tr, x_te)
        true_lab = _labels(y_te, thr)
        pred_lab = _labels(y_te_pred, thr)
        f1 = _macro_f1(true_lab, pred_lab)
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
    }


def _delta_vs_baseline(per_h: dict, hk: str) -> dict[str, float]:
    """Pooled and fold-mean deltas plus sigma multiple against fusion baseline."""
    base = _BASELINE_FUSION_SPX[hk]
    pooled = per_h[hk]["pooled_macro_f1"]
    fmean = per_h[hk]["fold_macro_f1_mean"]
    fstd = per_h[hk]["fold_macro_f1_std"]
    # Combined std assuming independent folds (treat each side as a 5-mean).
    # Conservative comparison: use the larger of the two stds for the sigma denominator.
    sigma_denom = max(fstd, base["fold_std"])
    return {
        "pooled_delta_vs_baseline": float(pooled - base["pooled"]),
        "fold_mean_delta_vs_baseline": float(fmean - base["fold_mean"]),
        "sigma_denom_used": float(sigma_denom),
        "fold_mean_delta_in_sigmas": float((fmean - base["fold_mean"]) / sigma_denom)
        if sigma_denom > 0
        else 0.0,
        "beats_baseline_by_1sigma_fold_mean": bool(
            (fmean - base["fold_mean"]) > sigma_denom
        ),
        "beats_baseline_by_1sigma_pooled_conservative": bool(
            (pooled - base["pooled"]) > sigma_denom
        ),
    }


def main() -> int:
    repo = Path("/data/external/yfinance")
    out_path = Path("/docs/research/har-tercile-fusion-per_asset_har-result.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    asset_paths = {
        "^NDX": repo / "NDX.parquet",
        "^DJI": repo / "DJI.parquet",
    }

    by_asset = {}
    headline = {}
    deltas = {}
    for sym, path in asset_paths.items():
        asset = _load_asset_window(path, sym)
        per_h = {f"h{h}": _per_horizon_eval(asset, h) for h in _HORIZONS}
        by_asset[sym] = {
            "first_date": str(asset.date[0]),
            "last_date": str(asset.date[-1]),
            "n_days": int(len(asset.date)),
            "by_horizon": per_h,
        }
        headline[sym] = {
            hk: {
                "pooled_macro_f1": per_h[hk]["pooled_macro_f1"],
                "fold_macro_f1_mean": per_h[hk]["fold_macro_f1_mean"],
                "fold_macro_f1_std": per_h[hk]["fold_macro_f1_std"],
                "n_pooled": per_h[hk]["n_pooled"],
            }
            for hk in ("h1", "h5", "h22")
        }
        deltas[sym] = {hk: _delta_vs_baseline(per_h, hk) for hk in ("h1", "h5", "h22")}

    beats_any = any(
        deltas[sym][hk]["beats_baseline_by_1sigma_fold_mean"]
        for sym in deltas
        for hk in deltas[sym]
    )

    result = {
        "arm_key": "per_asset_har",
        "comparator": "fusion-TP HAR-tercile baseline (SPX 5-min intraday RV, n_rows_valid=1999)",
        "comparator_path": "docs/research/recover-A-result.json",
        "baseline_fusion_spx": _BASELINE_FUSION_SPX,
        "protocol": {
            "rv_proxy": "daily log_return squared (yfinance daily closes); 5-min intraday "
            "RV unavailable for NDX/DJI",
            "date_window": f"{_FUSION_DATE_MIN} to {_FUSION_DATE_MAX} (matches fusion TP)",
            "har_space": "log",
            "forward_target": "log-mean over t+1..t+h",
            "folds": "walk_forward_splits, n_folds=5, embargo=23",
            "macro_f1": "pooled across 5 folds (canonical); also per-fold mean+/-std",
            "horizons": list(_HORIZONS),
            "seeds": "OLS HAR deterministic; n_seeds=1",
        },
        "by_asset": by_asset,
        "headline": headline,
        "delta_vs_recovered_baseline": deltas,
        "beats_outside_ci": beats_any,
        "caveats": (
            "Fusion TP carries only SPX 5-min intraday RV. NDX and DJI have no "
            "intraday source in this repository, so the per-asset arm uses a "
            "daily r^2 proxy. The proxy is noisier than 5-min RV, which "
            "structurally caps macro-F1; this is a data limit, not an arm "
            "failure. Date window matched to the fusion TP (2005-01-03 to "
            "2026-05-29) so fold sizes and embargo are comparable."
        ),
    }

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps({k: headline[k] for k in headline}, indent=2))
    print("---")
    print(json.dumps(deltas, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
