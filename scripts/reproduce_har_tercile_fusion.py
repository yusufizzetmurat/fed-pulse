"""Reproduce the canonical HAR-tercile pooled macro-F1 (0.687 / 0.685 / 0.654).

Mirrors :func:`app.data.fed_comms_regime.run` for the HAR-tercile baseline
arm only (no DL model needed). Reads the daily-fusion parquet directly,
applies the same OLS HAR fit, tercile bucketing, and walk-forward folds.

Inputs:
  /data/processed/tp_intraday_fomc_text_volatility/fusion/daily_fusion.parquet

Output:
  /data/artifacts/har_tercile_fusion_reproduction/result.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


_N_CLASSES = 3
_HORIZONS = (1, 5, 22)
_N_FOLDS = 5
_EMBARGO = max(_HORIZONS) + 1


def _walk_forward_splits(n: int, *, n_folds: int = _N_FOLDS, embargo: int = _EMBARGO):
    if n < n_folds * (embargo + 2):
        raise ValueError(f"too few rows ({n}) for {n_folds} folds with embargo {embargo}")
    test_size = n // (n_folds + 1)
    start = n - test_size * n_folds
    folds = []
    cursor = start
    for _ in range(n_folds):
        train_idx = list(range(0, cursor - embargo))
        test_idx = list(range(cursor, cursor + test_size))
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


def _macro_f1(true: np.ndarray, pred: np.ndarray, n_classes: int = _N_CLASSES) -> float:
    f1s = []
    for c in range(n_classes):
        tp = float(np.sum((pred == c) & (true == c)))
        fp = float(np.sum((pred == c) & (true != c)))
        fn = float(np.sum((pred != c) & (true == c)))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def main() -> int:
    fusion_path = Path(
        "/data/processed/tp_intraday_fomc_text_volatility/fusion/daily_fusion.parquet"
    )
    out_dir = Path("/data/artifacts/har_tercile_fusion_reproduction")
    out_dir.mkdir(parents=True, exist_ok=True)

    daily = pd.read_parquet(fusion_path).sort_values("date").reset_index(drop=True)

    har = daily[["rv_daily", "rv_weekly", "rv_monthly"]].to_numpy(dtype=np.float64)
    targets = np.column_stack(
        [daily[f"rv_fwd_{h}"].to_numpy(dtype=np.float64) for h in _HORIZONS]
    )

    # Match canonical valid mask: finite targets + finite HAR lags.
    # The canonical _assemble also requires finite market_feat (HAR + cross-market +
    # cal + surprise), but those add features beyond the HAR baseline itself.
    # For the HAR-only leg the relevant valid mask is finite-HAR + finite-target.
    valid = np.isfinite(targets).all(axis=1) & np.isfinite(har).all(axis=1)
    idx_all = np.where(valid)[0]
    folds = _walk_forward_splits(len(idx_all))

    per_horizon = {}
    for k, h in enumerate(_HORIZONS):
        pooled_true = []
        pooled_pred = []
        per_fold = []
        for fi, (tr_l, te_l) in enumerate(folds, start=1):
            tr = idx_all[np.array(tr_l)]
            te = idx_all[np.array(te_l)]
            y_tr = targets[tr, k]
            y_te = targets[te, k]
            thr = np.quantile(y_tr, [1.0 / 3.0, 2.0 / 3.0])
            har_pred_te = _fit_predict_ols(har[tr], y_tr, har[te])
            true_lab = _labels(y_te, thr)
            har_lab = _labels(har_pred_te, thr)
            f1 = _macro_f1(true_lab, har_lab)
            per_fold.append(
                {
                    "fold": fi,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "macro_f1": f1,
                    "q33": float(thr[0]),
                    "q67": float(thr[1]),
                }
            )
            pooled_true.append(true_lab)
            pooled_pred.append(har_lab)
        pooled_t = np.concatenate(pooled_true)
        pooled_p = np.concatenate(pooled_pred)
        f1s = np.array([row["macro_f1"] for row in per_fold])
        per_horizon[f"h{h}"] = {
            "per_fold": per_fold,
            "fold_macro_f1_mean": float(f1s.mean()),
            "fold_macro_f1_std": float(f1s.std(ddof=0)),
            "pooled_macro_f1": _macro_f1(pooled_t, pooled_p),
            "n_pooled": int(len(pooled_t)),
        }

    expected = {"h1": 0.687, "h5": 0.685, "h22": 0.654}
    headline = {
        hk: {
            "pooled_macro_f1": per_horizon[hk]["pooled_macro_f1"],
            "fold_macro_f1_mean": per_horizon[hk]["fold_macro_f1_mean"],
            "fold_macro_f1_std": per_horizon[hk]["fold_macro_f1_std"],
            "wiki_expected": expected[hk],
            "delta_vs_wiki_pooled": per_horizon[hk]["pooled_macro_f1"] - expected[hk],
        }
        for hk in ("h1", "h5", "h22")
    }

    result = {
        "arm_key": "har_tercile_fusion_reproduction",
        "protocol": {
            "source_parquet": str(fusion_path),
            "har_lags": "rv_daily / rv_weekly / rv_monthly (log-space, fusion TP)",
            "forward_targets": "rv_fwd_1 / rv_fwd_5 / rv_fwd_22 (log mean fwd RV)",
            "folds": "walk_forward_splits n_folds=5 embargo=23",
            "valid_mask": "finite HAR lags + finite forward targets",
            "tercile_thresholds": "q33/q67 of train-slice forward targets",
            "macro_f1": "pooled across 5 folds (canonical); also per-fold mean+/-std",
            "horizons": list(_HORIZONS),
            "seeds": "OLS HAR deterministic; n_seeds=1",
        },
        "n_rows_total": int(len(daily)),
        "n_rows_valid": int(len(idx_all)),
        "wiki_expected": expected,
        "by_horizon": per_horizon,
        "headline": headline,
    }

    out_path = out_dir / "result.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps(headline, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
