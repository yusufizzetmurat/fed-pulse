"""Baselines + multi-head regressor for dense daily vol/volume forecasting.

Establishes the honest bar (HAR-RV for vol, AR + day-of-week for volume),
then trains a shared-trunk multi-head regressor and reports out-of-sample
R² / RMSE / Spearman vs those baselines under embargoed walk-forward.
Vol targets are evaluated in log space (HAR convention); abnormal volume
in raw space. No text — that is Phase 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from app.data.dense_daily_dataset import DEFAULT_HORIZONS, build_dataset, walk_forward_splits

_EPS = 1e-8


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def _oos_r2(pred: np.ndarray, true: np.ndarray, base: np.ndarray) -> float:
    """1 − SSE_model / SSE_baseline, baseline = per-point reference (fold train mean)."""

    sse = float(np.sum((true - pred) ** 2))
    sst = float(np.sum((true - base) ** 2))
    return 1.0 - sse / sst if sst > 0 else float("nan")


def _spearman(pred: np.ndarray, true: np.ndarray) -> float:
    if len(pred) < 2:
        return float("nan")
    pr = np.argsort(np.argsort(pred)).astype(float)
    tr = np.argsort(np.argsort(true)).astype(float)
    if pr.std() == 0 or tr.std() == 0:
        return float("nan")
    return float(np.corrcoef(pr, tr)[0, 1])


def _bootstrap_r2_ci(
    pred: np.ndarray, true: np.ndarray, base: np.ndarray, *, seed: int = 11, n_boot: int = 1000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(true)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = _oos_r2(pred[idx], true[idx], base[idx])
        if not np.isnan(v):
            boots.append(v)
    if not boots:
        return float("nan"), float("nan")
    return float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))


def _fit_predict_ols(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray) -> np.ndarray:
    """OLS with intercept via lstsq; returns test predictions."""

    a_tr = np.column_stack([np.ones(len(x_tr)), x_tr])
    coef, *_ = np.linalg.lstsq(a_tr, y_tr, rcond=None)
    a_te = np.column_stack([np.ones(len(x_te)), x_te])
    return cast(np.ndarray, a_te @ coef)


def har_baseline(
    har_lags_tr: np.ndarray, log_rv_tr: np.ndarray, har_lags_te: np.ndarray
) -> np.ndarray:
    """log-HAR: regress log realized-vol on log of the 1/5/22-day vol lags."""

    return _fit_predict_ols(np.log(har_lags_tr + _EPS), log_rv_tr, np.log(har_lags_te + _EPS))


def _baseline_matrix(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    *,
    target_cols: list[str],
    har_idx: list[int],
    av_idx: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-target baseline predictions on train and test (HAR for vol, AR for av).

    Returns train preds too so the model can learn the baseline *residual*
    (residual stacking), which floors the stacked forecast at the baseline.
    """

    base_tr = np.zeros_like(y_tr)
    base_te = np.zeros((x_te.shape[0], len(target_cols)))
    for j, col in enumerate(target_cols):
        if col == "av":
            base_tr[:, j] = _fit_predict_ols(x_tr[:, av_idx], y_tr[:, j], x_tr[:, av_idx])
            base_te[:, j] = _fit_predict_ols(x_tr[:, av_idx], y_tr[:, j], x_te[:, av_idx])
        else:
            base_tr[:, j] = har_baseline(x_tr[:, har_idx], y_tr[:, j], x_tr[:, har_idx])
            base_te[:, j] = har_baseline(x_tr[:, har_idx], y_tr[:, j], x_te[:, har_idx])
    return base_tr, base_te


# --------------------------------------------------------------------------
# Multi-head regressor (integration — exercised by the live run).
# --------------------------------------------------------------------------


def _build_model(n_features: int, n_heads: int) -> Any:
    import torch
    from torch import nn

    class MultiHeadRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.GELU(),
                nn.LayerNorm(64),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.GELU(),
            )
            self.heads = nn.ModuleList([nn.Linear(32, 1) for _ in range(n_heads)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.trunk(x)
            return torch.cat([head(h) for head in self.heads], dim=1)

    return MultiHeadRegressor()


def _train_fold(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    *,
    seed: int,
    epochs: int,
    device: str,
) -> np.ndarray:
    """Standardize on train, Huber-train the multi-head model, return de-standardized preds."""

    import torch
    from torch import nn

    from app.determinism import enable_deterministic_mode

    enable_deterministic_mode(seed)
    dev = torch.device(device)
    xm, xs = x_tr.mean(0), x_tr.std(0)
    xs = np.where(xs > 0, xs, 1.0)
    ym, ys = y_tr.mean(0), y_tr.std(0)
    ys = np.where(ys > 0, ys, 1.0)
    xtr = torch.tensor((x_tr - xm) / xs, dtype=torch.float32, device=dev)
    ytr = torch.tensor((y_tr - ym) / ys, dtype=torch.float32, device=dev)
    xte = torch.tensor((x_te - xm) / xs, dtype=torch.float32, device=dev)

    n_val = max(1, len(xtr) // 5)
    tr, val = slice(0, len(xtr) - n_val), slice(len(xtr) - n_val, len(xtr))
    model = _build_model(x_tr.shape[1], y_tr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.HuberLoss()
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(xtr[tr]), ytr[tr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(xtr[val]), ytr[val]))
        if vloss < best - 1e-6:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 40:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_std = model(xte).cpu().numpy()
    return cast(np.ndarray, pred_std * ys + ym)


def run(
    cache_dir: Path | str,
    *,
    seed: int = 11,
    n_folds: int = 5,
    embargo: int = 10,
    epochs: int = 300,
    device: str = "cpu",
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    X, Y, _dates = build_dataset(cache_dir, horizons=horizons)
    target_cols = [f"rv_{h}" for h in horizons] + ["av"]
    # vol targets evaluated in log space; av in raw space.
    Yt = Y.copy()
    for h in horizons:
        Yt[f"rv_{h}"] = np.log(Y[f"rv_{h}"] + _EPS)
    Xv = X.to_numpy(dtype=np.float64)
    Yv = Yt.to_numpy(dtype=np.float64)
    har_idx = [X.columns.get_loc(c) for c in ("rv_lag_1", "rv_lag_5", "rv_lag_22")]
    av_idx = [X.columns.get_loc(c) for c in ("av_lag_1", "vol_ratio_30", "dow_0", "dow_1")]
    folds = walk_forward_splits(len(Xv), n_folds=n_folds, embargo=embargo)

    pooled: dict[str, dict[str, list[float]]] = {
        c: {"model": [], "base_model": [], "true": [], "base": []} for c in target_cols
    }
    for tr, te in folds:
        # Residual stacking: baseline (HAR/AR) carries the persistent signal;
        # the model learns only the residual the baseline misses.
        base_tr, base_te = _baseline_matrix(
            Xv[tr], Yv[tr], Xv[te], target_cols=target_cols, har_idx=har_idx, av_idx=av_idx
        )
        resid_tr = Yv[tr] - base_tr
        model_resid = _train_fold(Xv[tr], resid_tr, Xv[te], seed=seed, epochs=epochs, device=device)
        final = base_te + model_resid
        for j, col in enumerate(target_cols):
            true = Yv[te, j]
            fold_mean = float(Yv[tr, j].mean())
            pooled[col]["model"].extend(final[:, j].tolist())
            pooled[col]["base_model"].extend(base_te[:, j].tolist())
            pooled[col]["true"].extend(true.tolist())
            pooled[col]["base"].extend([fold_mean] * len(te))

    results: dict[str, Any] = {"n_events": len(Xv), "n_folds": n_folds, "by_target": {}}
    for col in target_cols:
        p = {k: np.asarray(v) for k, v in pooled[col].items()}
        lo, hi = _bootstrap_r2_ci(p["model"], p["true"], p["base"], seed=seed)
        results["by_target"][col] = {
            "model_r2": _oos_r2(p["model"], p["true"], p["base"]),
            "model_r2_ci90": [lo, hi],
            "baseline_r2": _oos_r2(p["base_model"], p["true"], p["base"]),
            "model_rmse": _rmse(p["model"], p["true"]),
            "baseline_rmse": _rmse(p["base_model"], p["true"]),
            "model_spearman": _spearman(p["model"], p["true"]),
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dense daily vol/volume backbone: model vs baselines."
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    res = run(args.cache_dir, seed=args.seed, epochs=args.epochs, device=args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"n={res['n_events']} folds={res['n_folds']}")
    print(f"{'target':<8} {'model_R2':>9} {'CI90':>18} {'base_R2':>9} {'spearman':>9}")
    for col, r in res["by_target"].items():
        ci = f"[{r['model_r2_ci90'][0]:.3f},{r['model_r2_ci90'][1]:.3f}]"
        print(
            f"{col:<8} {r['model_r2']:>9.3f} {ci:>18} {r['baseline_r2']:>9.3f} "
            f"{r['model_spearman']:>9.3f}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
