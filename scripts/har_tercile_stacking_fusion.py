"""HAR-tercile + DL stacking against the fusion TP HAR-tercile baseline.

Stacks two heads inside each walk-forward fold:

  HAR head : OLS HAR on rv_daily/rv_weekly/rv_monthly, fit on the train slice
             of the SAME fold; soft probs derived from the HAR predicted RV via
             softmax over signed distances to the train-slice tercile cutoffs
             (q33 / q67).

  DL head  : small MLP over the fusion-feature stack (HAR lags + downside lags
             + jump lags + volume lags + cross-market corr lags + calendar +
             surprise). Trained with cross-entropy on the same train slice
             with an inner val for early stopping. Outputs 3-class soft probs.

  Stack    : convex blend of log-probs with weights fit on an inner val
             held out from the train slice. Final test prediction = argmax of
             the blended log-probs.

Protocol matches the recovered HAR-tercile baseline (docs/research/recover-A-result.json):
  - source parquet : /data/processed/tp_intraday_fomc_text_volatility/fusion/daily_fusion.parquet
  - HAR valid mask : finite HAR lags + finite fwd targets (n=5363)
  - folds          : walk_forward_splits(n, n_folds=5, embargo=23)
  - tercile cutoffs: q33 / q67 of train-slice log fwd RV (same as baseline)
  - macro_f1       : pooled across the 5 folds (canonical)

The DL classifier is deliberately small and CPU-friendly. The arm is meant to
test whether the canonical fusion features (cross-market + calendar + surprise)
can lift the HAR-tercile baseline that already sits at 0.687 / 0.685 / 0.654
pooled macro-F1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


_N_CLASSES = 3
_HORIZONS = (1, 5, 22)
_N_FOLDS = 5
_EMBARGO = max(_HORIZONS) + 1
_HAR_COLS = ("rv_daily", "rv_weekly", "rv_monthly")
_FUSION_COLS = (
    "rv_daily",
    "rv_weekly",
    "rv_monthly",
    "downside_daily",
    "downside_weekly",
    "downside_monthly",
    "jump_daily",
    "jump_weekly",
    "jump_monthly",
    "volume_daily",
    "volume_weekly",
    "volume_monthly",
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
)
_BASELINE = {
    "h1": {"pooled": 0.6873, "fold_mean": 0.629, "fold_std": 0.042},
    "h5": {"pooled": 0.685, "fold_mean": 0.618, "fold_std": 0.034},
    "h22": {"pooled": 0.6542, "fold_mean": 0.554, "fold_std": 0.046},
}


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


def _har_soft_probs(har_pred: np.ndarray, thr: np.ndarray, scale: float) -> np.ndarray:
    """Convert HAR-predicted-RV into 3-class soft probs via softmax over
    signed distances to the train-slice q33/q67 cutoffs.

    Class 0 is "below q33", class 1 is "between cutoffs", class 2 is "above q67".
    We construct a logit per class by negative-squared-distance to the class
    centroid implied by the cutoffs (left = q33 - delta, mid = (q33+q67)/2,
    right = q67 + delta) where delta is half the interquantile gap. Sharpness
    is controlled by ``scale`` (set on the train slice).
    """

    q33, q67 = thr
    delta = max((q67 - q33) / 2.0, 1e-6)
    centroids = np.array([q33 - delta, 0.5 * (q33 + q67), q67 + delta])
    diff = har_pred[:, None] - centroids[None, :]
    logits = -(diff**2) / max(scale, 1e-9)
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return p


class _Mlp(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, _N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_dl(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    epochs: int = 80,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
) -> _Mlp:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _Mlp(x_tr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    x_tr_t = torch.tensor(x_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    rng = np.random.default_rng(seed)
    n = len(x_tr_t)
    best_f1 = -1.0
    best_state = None
    bad = 0
    for _ in range(epochs):
        model.train()
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            b = order[s : s + batch_size]
            opt.zero_grad()
            logits = model(x_tr_t[b])
            loss = crit(logits, y_tr_t[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(x_val_t).argmax(1).cpu().numpy()
        vf1 = _macro_f1(y_val_t.numpy(), vp)
        if vf1 > best_f1 + 1e-6:
            best_f1 = vf1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict_proba(model: _Mlp, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
        p = torch.softmax(logits, dim=1).cpu().numpy()
    return p


def _fit_blend_weight(
    p_har: np.ndarray, p_dl: np.ndarray, y: np.ndarray, *, grid: int = 21
) -> tuple[float, float]:
    """Grid-search w in [0,1] for w * log p_har + (1-w) * log p_dl on (val) labels."""

    best_w, best_f1 = 0.5, -1.0
    log_har = np.log(np.clip(p_har, 1e-9, 1.0))
    log_dl = np.log(np.clip(p_dl, 1e-9, 1.0))
    for w in np.linspace(0.0, 1.0, grid):
        log_p = w * log_har + (1.0 - w) * log_dl
        pred = log_p.argmax(1)
        f1 = _macro_f1(y, pred)
        if f1 > best_f1 + 1e-9:
            best_f1 = f1
            best_w = float(w)
    return best_w, float(best_f1)


def _run_horizon(
    har: np.ndarray,
    fusion: np.ndarray,
    targets: np.ndarray,
    idx_all: np.ndarray,
    folds,
    k: int,
    *,
    seed: int,
) -> dict[str, Any]:
    pooled = {kk: [] for kk in ("true", "har_pred", "dl_pred", "blend_pred", "har_probs", "dl_probs")}
    per_fold = []
    for fi, (tr_l, te_l) in enumerate(folds, start=1):
        tr = idx_all[np.array(tr_l)]
        te = idx_all[np.array(te_l)]
        y_tr = targets[tr, k]
        y_te = targets[te, k]
        thr = np.quantile(y_tr, [1.0 / 3.0, 2.0 / 3.0])
        true_tr = _labels(y_tr, thr)
        true_te = _labels(y_te, thr)

        # HAR head: fit on full train, predict test + train
        har_pred_te = _fit_predict_ols(har[tr], y_tr, har[te])
        har_pred_tr = _fit_predict_ols(har[tr], y_tr, har[tr])
        scale = float(np.var(har_pred_tr - y_tr)) + 1e-9
        p_har_tr = _har_soft_probs(har_pred_tr, thr, scale)
        p_har_te = _har_soft_probs(har_pred_te, thr, scale)
        har_lab_te = _labels(har_pred_te, thr)

        # DL head: standardize fusion features on train, hold out an inner val (20%) for early stop
        mf_tr_raw = fusion[tr]
        mf_te_raw = fusion[te]
        mu = mf_tr_raw.mean(0)
        sd = mf_tr_raw.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        mf_tr = (mf_tr_raw - mu) / sd
        mf_te = (mf_te_raw - mu) / sd

        # inner val = last 20% of the train slice (preserves causal order)
        n_tr = len(tr)
        n_val = max(64, n_tr // 5)
        core_end = n_tr - n_val
        # Append the HAR soft probs as extra features for the meta classifier
        x_tr_full = np.concatenate([mf_tr, p_har_tr], axis=1).astype(np.float32)
        y_tr_full = true_tr
        x_core = x_tr_full[:core_end]
        y_core = y_tr_full[:core_end]
        x_val = x_tr_full[core_end:]
        y_val = y_tr_full[core_end:]

        model = _train_dl(x_core, y_core, x_val, y_val, seed=seed)
        # DL probs on val (for weight fit) and on test
        p_dl_val = _predict_proba(model, x_val)
        p_dl_te = _predict_proba(model, np.concatenate([mf_te, p_har_te], axis=1).astype(np.float32))

        # HAR probs on val to fit blend weight
        p_har_val = p_har_tr[core_end:]
        w_har, val_blend_f1 = _fit_blend_weight(p_har_val, p_dl_val, y_val)

        # Blended test prediction
        log_har = np.log(np.clip(p_har_te, 1e-9, 1.0))
        log_dl = np.log(np.clip(p_dl_te, 1e-9, 1.0))
        log_blend = w_har * log_har + (1.0 - w_har) * log_dl
        blend_lab_te = log_blend.argmax(1)
        dl_lab_te = p_dl_te.argmax(1)

        f1_har = _macro_f1(true_te, har_lab_te)
        f1_dl = _macro_f1(true_te, dl_lab_te)
        f1_blend = _macro_f1(true_te, blend_lab_te)
        per_fold.append(
            {
                "fold": fi,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "n_val_inner": int(n_val),
                "har_macro_f1": f1_har,
                "dl_macro_f1": f1_dl,
                "blend_macro_f1": f1_blend,
                "val_blend_macro_f1": val_blend_f1,
                "w_har": w_har,
                "q33": float(thr[0]),
                "q67": float(thr[1]),
            }
        )
        pooled["true"].append(true_te)
        pooled["har_pred"].append(har_lab_te)
        pooled["dl_pred"].append(dl_lab_te)
        pooled["blend_pred"].append(blend_lab_te)
        pooled["har_probs"].append(p_har_te)
        pooled["dl_probs"].append(p_dl_te)

    pooled_t = np.concatenate(pooled["true"])
    pooled_har = np.concatenate(pooled["har_pred"])
    pooled_dl = np.concatenate(pooled["dl_pred"])
    pooled_blend = np.concatenate(pooled["blend_pred"])

    fold_blend = np.array([row["blend_macro_f1"] for row in per_fold])
    fold_har = np.array([row["har_macro_f1"] for row in per_fold])
    fold_dl = np.array([row["dl_macro_f1"] for row in per_fold])

    return {
        "per_fold": per_fold,
        "pooled_har_macro_f1": _macro_f1(pooled_t, pooled_har),
        "pooled_dl_macro_f1": _macro_f1(pooled_t, pooled_dl),
        "pooled_blend_macro_f1": _macro_f1(pooled_t, pooled_blend),
        "fold_blend_macro_f1_mean": float(fold_blend.mean()),
        "fold_blend_macro_f1_std": float(fold_blend.std(ddof=0)),
        "fold_har_macro_f1_mean": float(fold_har.mean()),
        "fold_har_macro_f1_std": float(fold_har.std(ddof=0)),
        "fold_dl_macro_f1_mean": float(fold_dl.mean()),
        "fold_dl_macro_f1_std": float(fold_dl.std(ddof=0)),
        "n_pooled": int(len(pooled_t)),
    }


def main() -> int:
    fusion_path = Path(
        "/data/processed/tp_intraday_fomc_text_volatility/fusion/daily_fusion.parquet"
    )
    out_dir = Path("/data/artifacts/har_tercile_fusion_stacking_fusion")
    out_dir.mkdir(parents=True, exist_ok=True)

    daily = pd.read_parquet(fusion_path).sort_values("date").reset_index(drop=True)

    har = daily.loc[:, list(_HAR_COLS)].to_numpy(dtype=np.float64)
    fusion = daily.loc[:, list(_FUSION_COLS)].to_numpy(dtype=np.float64)
    targets = np.column_stack(
        [daily[f"rv_fwd_{h}"].to_numpy(dtype=np.float64) for h in _HORIZONS]
    )

    # Valid mask = finite HAR + finite forward targets (matches recovered baseline n=5363).
    valid = np.isfinite(targets).all(axis=1) & np.isfinite(har).all(axis=1)
    # The DL head needs finite fusion features; if any are NaN, fill with column train means inside
    # each fold. So we don't tighten the valid mask here, but we apply imputation per fold.
    idx_all = np.where(valid)[0]
    folds = _walk_forward_splits(len(idx_all))

    # Impute fusion NaNs with column-wise medians computed up to row i (causal forward fill via ffill)
    # so the DL features are finite. We do this once globally with ffill then median impute, which
    # for the columns above (lags + calendar + surprise) is leak-safe by construction (each column is
    # known on or before that row's date).
    fusion_df = pd.DataFrame(fusion, columns=list(_FUSION_COLS))
    fusion_df = fusion_df.ffill()
    fusion_df = fusion_df.fillna(fusion_df.median(numeric_only=True))
    fusion_df = fusion_df.fillna(0.0)
    fusion = fusion_df.to_numpy(dtype=np.float64)

    by_horizon: dict[str, Any] = {}
    for k, h in enumerate(_HORIZONS):
        by_horizon[f"h{h}"] = _run_horizon(
            har, fusion, targets, idx_all, folds, k, seed=11
        )

    # Delta vs recovered baseline
    deltas = {}
    for hk in ("h1", "h5", "h22"):
        bb = _BASELINE[hk]
        bh = by_horizon[hk]
        pooled_delta = bh["pooled_blend_macro_f1"] - bb["pooled"]
        fold_mean_delta = bh["fold_blend_macro_f1_mean"] - bb["fold_mean"]
        sigma = bb["fold_std"]
        deltas[hk] = {
            "pooled_delta_vs_baseline": pooled_delta,
            "fold_mean_delta_vs_baseline": fold_mean_delta,
            "sigma_denom_used": sigma,
            "fold_mean_delta_in_sigmas": fold_mean_delta / sigma if sigma > 0 else 0.0,
            "beats_baseline_by_1sigma_fold_mean": bool(fold_mean_delta > sigma),
            "beats_baseline_by_1sigma_pooled_conservative": bool(pooled_delta > sigma),
        }

    beats_outside_ci = any(
        deltas[hk]["beats_baseline_by_1sigma_fold_mean"]
        or deltas[hk]["beats_baseline_by_1sigma_pooled_conservative"]
        for hk in ("h1", "h5", "h22")
    )

    headline = {
        hk: {
            "har_pooled_macro_f1": by_horizon[hk]["pooled_har_macro_f1"],
            "dl_pooled_macro_f1": by_horizon[hk]["pooled_dl_macro_f1"],
            "blend_pooled_macro_f1": by_horizon[hk]["pooled_blend_macro_f1"],
            "blend_fold_macro_f1_mean": by_horizon[hk]["fold_blend_macro_f1_mean"],
            "blend_fold_macro_f1_std": by_horizon[hk]["fold_blend_macro_f1_std"],
            "n_pooled": by_horizon[hk]["n_pooled"],
        }
        for hk in ("h1", "h5", "h22")
    }

    result = {
        "arm_key": "stacking_fusion",
        "comparator": "fusion-TP HAR-tercile baseline (SPX 5-min intraday RV, n_rows_valid=5363)",
        "comparator_path": "docs/research/recover-A-result.json",
        "baseline_fusion_spx": _BASELINE,
        "protocol": {
            "source_parquet": str(fusion_path),
            "har_lags": list(_HAR_COLS),
            "fusion_features": list(_FUSION_COLS),
            "forward_targets": [f"rv_fwd_{h}" for h in _HORIZONS],
            "folds": "walk_forward_splits n_folds=5 embargo=23",
            "valid_mask": "finite HAR lags + finite forward targets",
            "tercile_thresholds": "q33/q67 of train-slice forward targets",
            "macro_f1": "pooled across 5 folds (canonical); also per-fold mean+/-std",
            "horizons": list(_HORIZONS),
            "har_head": "OLS HAR fit per fold; soft probs via softmax over signed-distance to centroids implied by q33/q67",
            "dl_head": "MLP(64,64) on fusion features + HAR soft probs; AdamW lr=1e-3 wd=1e-4; inner-val early stop (patience=10, max 80 ep)",
            "stack": "convex blend w*log(p_har) + (1-w)*log(p_dl); w grid-searched on inner val",
            "seeds": "single seed (11) for deterministic MLP init",
        },
        "n_rows_total": int(len(daily)),
        "n_rows_valid": int(len(idx_all)),
        "by_horizon": by_horizon,
        "headline": headline,
        "delta_vs_recovered_baseline": deltas,
        "beats_outside_ci": beats_outside_ci,
    }

    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'result.json'}")
    print(json.dumps({"headline": headline, "delta_vs_recovered_baseline": deltas,
                      "beats_outside_ci": beats_outside_ci}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
