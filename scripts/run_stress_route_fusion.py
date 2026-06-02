"""VIX-stress routing on the fusion-TP daily HAR-tercile surface.

Re-runs the stress-route improvement arm against the daily-fusion parquet that
produced the recovered HAR-tercile baseline (h1=0.6873, h5=0.6850, h22=0.6542
pooled macro-F1). The earlier ``stress_route`` arm at
``docs/research/har-tercile-arm-stress_route-result.json`` evaluated a
statement-level canonical_vix surface against a dual-head DL classifier targeting
a single forward_realized_vol_10d horizon; that surface cannot reproduce the
0.687/0.685/0.654 wiki numbers (it returns 0.442 pooled at best).

This script keeps the protocol identical to ``reproduce_har_tercile_fusion.py``:
same parquet, same expanding walk-forward (5 folds, embargo=23), same
finite-target + finite-HAR-lag valid mask, same train-slice q33/q67 tercile
thresholds. The HAR leg is OLS on (rv_daily, rv_weekly, rv_monthly). The "DL"
leg is a small MLP regressor on the full cross-market + calendar + surprise
feature panel that fusion daily rows already carry (HAR lags plus
volume/downside/jump HARs plus corr_tnx/corr_dxy HARs plus
days_since_stmt/days_to_stmt plus surprise_{level,path,info}). The MLP is
trained per fold per horizon with deterministic seeds (5 seeds: 11/29/47/71/97).
After both per-row predictions are in hand, a per-row VIX gate selects DL when
the prior-day VIX close exceeds 22 and HAR otherwise.

Output: docs/research/har-tercile-fusion-stress_route_fusion-result.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

_N_CLASSES = 3
_HORIZONS = (1, 5, 22)
_N_FOLDS = 5
_EMBARGO = max(_HORIZONS) + 1
_SEEDS = (11, 29, 47, 71, 97)
_VIX_THRESHOLD = 22.0

_FUSION_PARQUET = Path(
    "/data/processed/tp_intraday_fomc_text_volatility/fusion/daily_fusion.parquet"
)
_VIX_PARQUET = Path("/data/external/yfinance/VIX.parquet")
_OUT_JSON = Path(
    "/data/artifacts/har_tercile_fusion_stress_route/result.json"
)

_HAR_COLS = ["rv_daily", "rv_weekly", "rv_monthly"]
_EXTRA_FEATURE_COLS = [
    "volume_daily",
    "volume_weekly",
    "volume_monthly",
    "downside_daily",
    "downside_weekly",
    "downside_monthly",
    "jump_daily",
    "jump_weekly",
    "jump_monthly",
    "corr_tnx_daily",
    "corr_tnx_weekly",
    "corr_tnx_monthly",
    "corr_dxy_daily",
    "corr_dxy_weekly",
    "corr_dxy_monthly",
    "days_since_stmt",
    "days_to_stmt",
    "surprise_level",
    "surprise_path",
    "surprise_info",
]
_ALL_FEATURE_COLS = _HAR_COLS + _EXTRA_FEATURE_COLS


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


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train_mlp(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    *,
    seed: int,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean = x_tr.mean(axis=0, keepdims=True)
    std = x_tr.std(axis=0, keepdims=True) + 1e-8
    x_tr_n = (x_tr - mean) / std
    x_te_n = (x_te - mean) / std
    y_mean = float(y_tr.mean())
    y_std = float(y_tr.std() + 1e-8)
    y_tr_n = (y_tr - y_mean) / y_std

    model = _MLP(in_dim=x_tr_n.shape[1], hidden=32)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    x_t = torch.from_numpy(x_tr_n.astype(np.float32))
    y_t = torch.from_numpy(y_tr_n.astype(np.float32))
    n = x_t.shape[0]
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            optim.zero_grad()
            pred = model(x_t[idx])
            loss = loss_fn(pred, y_t[idx])
            loss.backward()
            optim.step()
    model.eval()
    with torch.no_grad():
        pred_n = model(torch.from_numpy(x_te_n.astype(np.float32))).numpy()
    return pred_n * y_std + y_mean


def main() -> int:
    daily = pd.read_parquet(_FUSION_PARQUET).sort_values("date").reset_index(drop=True)
    vix = (
        pd.read_parquet(_VIX_PARQUET)[["date", "close"]]
        .rename(columns={"close": "vix_close"})
        .sort_values("date")
    )
    # Use prior-day VIX (lag 1) so the gate is causal w.r.t. the day being scored.
    vix["vix_lag1"] = vix["vix_close"].shift(1)
    daily = daily.merge(vix[["date", "vix_lag1"]], on="date", how="left")

    har = daily[_HAR_COLS].to_numpy(dtype=np.float64)
    feat = daily[_ALL_FEATURE_COLS].to_numpy(dtype=np.float64)
    targets = np.column_stack(
        [daily[f"rv_fwd_{h}"].to_numpy(dtype=np.float64) for h in _HORIZONS]
    )
    vix_lag1 = daily["vix_lag1"].to_numpy(dtype=np.float64)

    # Match baseline valid mask exactly: finite HAR lags + finite forward targets.
    valid = np.isfinite(targets).all(axis=1) & np.isfinite(har).all(axis=1)
    idx_all = np.where(valid)[0]
    folds = _walk_forward_splits(len(idx_all))

    # Extra-feature finite mask (used for the DL leg only; HAR leg uses the
    # baseline valid mask). We impute missing extras with the train-slice mean so
    # the per-row routing always has both predictions on the baseline rows.
    extra_finite_mask = np.isfinite(feat).all(axis=1)

    per_horizon = {}
    for k, h in enumerate(_HORIZONS):
        pooled_true = []
        pooled_har = []
        pooled_dl_by_seed = {s: [] for s in _SEEDS}
        pooled_routed_by_seed = {s: [] for s in _SEEDS}
        pooled_vix_lag1 = []
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
            har_f1 = _macro_f1(true_lab, har_lab)

            # Train DL leg using rows whose extras are finite. Impute missing
            # extras on the test slice with train-slice mean so DL predictions
            # cover every baseline-valid row in the fold.
            tr_extras_ok = tr[extra_finite_mask[tr]]
            x_tr = feat[tr_extras_ok]
            y_tr_dl = targets[tr_extras_ok, k]
            mean_extra = np.nanmean(feat[tr_extras_ok], axis=0)
            x_te = feat[te].copy()
            for j in range(x_te.shape[1]):
                col = x_te[:, j]
                col[~np.isfinite(col)] = mean_extra[j]
                x_te[:, j] = col

            dl_lab_per_seed = {}
            routed_lab_per_seed = {}
            dl_f1_per_seed = {}
            routed_f1_per_seed = {}
            for seed in _SEEDS:
                dl_pred_te = _train_mlp(x_tr, y_tr_dl, x_te, seed=seed)
                dl_lab = _labels(dl_pred_te, thr)
                vix_te = vix_lag1[te]
                stress = (vix_te > _VIX_THRESHOLD) & np.isfinite(vix_te)
                routed_lab = np.where(stress, dl_lab, har_lab)
                dl_f1 = _macro_f1(true_lab, dl_lab)
                routed_f1 = _macro_f1(true_lab, routed_lab)
                dl_lab_per_seed[seed] = dl_lab
                routed_lab_per_seed[seed] = routed_lab
                dl_f1_per_seed[seed] = dl_f1
                routed_f1_per_seed[seed] = routed_f1
                pooled_dl_by_seed[seed].append(dl_lab)
                pooled_routed_by_seed[seed].append(routed_lab)

            n_stress = int(((vix_lag1[te] > _VIX_THRESHOLD) & np.isfinite(vix_lag1[te])).sum())
            per_fold.append(
                {
                    "fold": fi,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "n_stress_rows_vix_lag1_gt_22": n_stress,
                    "har_macro_f1": har_f1,
                    "dl_macro_f1_per_seed": {str(s): dl_f1_per_seed[s] for s in _SEEDS},
                    "routed_macro_f1_per_seed": {str(s): routed_f1_per_seed[s] for s in _SEEDS},
                    "q33": float(thr[0]),
                    "q67": float(thr[1]),
                }
            )
            pooled_true.append(true_lab)
            pooled_har.append(har_lab)
            pooled_vix_lag1.append(vix_lag1[te])

        pooled_t = np.concatenate(pooled_true)
        pooled_h = np.concatenate(pooled_har)
        pooled_v = np.concatenate(pooled_vix_lag1)
        har_pooled_f1 = _macro_f1(pooled_t, pooled_h)
        dl_pooled_per_seed = {}
        routed_pooled_per_seed = {}
        for s in _SEEDS:
            dl_pooled_per_seed[str(s)] = _macro_f1(
                pooled_t, np.concatenate(pooled_dl_by_seed[s])
            )
            routed_pooled_per_seed[str(s)] = _macro_f1(
                pooled_t, np.concatenate(pooled_routed_by_seed[s])
            )
        dl_arr = np.array(list(dl_pooled_per_seed.values()))
        routed_arr = np.array(list(routed_pooled_per_seed.values()))
        fold_har = np.array([row["har_macro_f1"] for row in per_fold])
        fold_routed_means = np.array(
            [
                float(np.mean([row["routed_macro_f1_per_seed"][str(s)] for s in _SEEDS]))
                for row in per_fold
            ]
        )

        n_stress_total = int(((pooled_v > _VIX_THRESHOLD) & np.isfinite(pooled_v)).sum())
        per_horizon[f"h{h}"] = {
            "per_fold": per_fold,
            "n_pooled": int(len(pooled_t)),
            "n_stress_pooled_vix_lag1_gt_22": n_stress_total,
            "har_pooled_macro_f1": har_pooled_f1,
            "har_fold_mean": float(fold_har.mean()),
            "har_fold_std": float(fold_har.std(ddof=0)),
            "dl_pooled_macro_f1_per_seed": dl_pooled_per_seed,
            "dl_pooled_mean": float(dl_arr.mean()),
            "dl_pooled_std": float(dl_arr.std(ddof=0)),
            "routed_pooled_macro_f1_per_seed": routed_pooled_per_seed,
            "routed_pooled_mean": float(routed_arr.mean()),
            "routed_pooled_std": float(routed_arr.std(ddof=0)),
            "routed_fold_mean": float(fold_routed_means.mean()),
            "routed_fold_std": float(fold_routed_means.std(ddof=0)),
        }

    expected = {"h1": 0.687, "h5": 0.685, "h22": 0.654}
    baseline_fold_std = {"h1": 0.0425, "h5": 0.0342, "h22": 0.0465}
    headline = {}
    for hk in ("h1", "h5", "h22"):
        rec = per_horizon[hk]
        delta = rec["routed_pooled_mean"] - expected[hk]
        # 1-sigma test: routed must beat baseline by more than the baseline's
        # own fold std (matching the recovered-baseline std). Sigma here is the
        # recovered-baseline fold std at the same horizon.
        beat_by_1_sigma = delta > baseline_fold_std[hk]
        headline[hk] = {
            "routed_pooled_macro_f1_mean": rec["routed_pooled_mean"],
            "routed_pooled_macro_f1_std": rec["routed_pooled_std"],
            "har_pooled_macro_f1": rec["har_pooled_macro_f1"],
            "dl_pooled_macro_f1_mean": rec["dl_pooled_mean"],
            "wiki_expected": expected[hk],
            "delta_vs_wiki_pooled": delta,
            "baseline_fold_std_used": baseline_fold_std[hk],
            "beats_baseline_by_1_sigma": bool(beat_by_1_sigma),
        }

    result = {
        "arm_key": "stress_route_fusion",
        "arm_name": "VIX-stress routing on fusion TP",
        "protocol": {
            "source_parquet": str(_FUSION_PARQUET),
            "har_lags": "rv_daily / rv_weekly / rv_monthly (log-space, fusion TP)",
            "dl_features": _ALL_FEATURE_COLS,
            "dl_model": "MLP 32-32-1, Adam lr=1e-3 wd=1e-4, 200 epochs batch=256, standardized x and y",
            "forward_targets": "rv_fwd_1 / rv_fwd_5 / rv_fwd_22",
            "folds": "walk_forward_splits n_folds=5 embargo=23",
            "valid_mask": "finite HAR lags + finite forward targets (matches recovered baseline n=5363)",
            "tercile_thresholds": "q33/q67 of train-slice forward targets",
            "vix_source": str(_VIX_PARQUET),
            "vix_gate": "route to DL when vix_lag1>22 else HAR; vix_lag1 is prior-day VIX close",
            "macro_f1": "pooled across 5 folds (canonical); also per-fold mean+/-std",
            "horizons": list(_HORIZONS),
            "seeds": list(_SEEDS),
        },
        "n_rows_total": int(len(daily)),
        "n_rows_valid": int(len(idx_all)),
        "wiki_expected": expected,
        "by_horizon": per_horizon,
        "headline": headline,
    }

    _OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    _OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {_OUT_JSON}")
    print(json.dumps(headline, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
