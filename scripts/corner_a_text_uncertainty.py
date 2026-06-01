"""Corner A experiment — does textual policy uncertainty add to RV forecasting?

Pre-registered in docs/research/corner-a-text-uncertainty-rv-preregistration.md.
Paired walk-forward QLIKE comparison: the validated QLIKE-DLq ensemble with vs
without one leak-safe text-uncertainty column, on identical folds/seeds. Emits
per-horizon, per-cell (full / post-FOMC) QLIKE for HAR / ens / ens+text, the
incremental gain ΔQLIKE_text = QLIKE(ens) − QLIKE(ens+text) with the
Bonferroni-corrected bootstrap CI (6 tests, family α=0.10), and feature
diagnostics. Run inside the GPU container.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.data.dense_daily_dataset import walk_forward_splits
from app.data.dense_forecast_train import _fit_predict_ols
from app.data.intraday_rv_forecast import _forward_log_rv, _qlike, _qlike_pointwise
from app.data.intraday_rv_production import _build_full, _ensemble_predict

SEEDS = (11, 22, 33, 44, 55)
HORIZONS = (1, 5, 22)
N_FOLDS = 5
EPOCHS = int(os.environ.get("CORNER_A_EPOCHS", "300"))
FAMILY_ALPHA = 0.10
N_TESTS = 6  # 3 horizons × 2 cells, fixed by the pre-registration
RV_PATH = "data/external/alphavantage_bars/spx_5min_daily_rv.parquet"
FEAT_PATH = "data/artifacts/corner_a_text_uncertainty/text_uncertainty_daily.parquet"
OUT_PATH = "data/artifacts/corner_a_text_uncertainty/result.json"


def _boot_gain_ci(
    pred: np.ndarray,
    base: np.ndarray,
    true: np.ndarray,
    *,
    block: int,
    q_lo: float,
    q_hi: float,
    seed: int = 11,
    n_boot: int = 2000,
) -> list[float]:
    """Moving-block bootstrap CI of mean QLIKE gain (base − pred) at [q_lo, q_hi].

    >0 lower bound ⇒ pred (ens+text) beats base (ens). block=h so overlapping
    multi-day forward targets don't inflate significance.
    """

    gain = _qlike_pointwise(base, true) - _qlike_pointwise(pred, true)
    n = len(gain)
    if n <= block:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    boots = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        boots.append(float(gain[idx].mean()))
    return [float(np.quantile(boots, q_lo)), float(np.quantile(boots, q_hi))]


def run_experiment(device: str = "cpu") -> dict[str, Any]:
    df = pd.read_parquet(RV_PATH).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    feat = pd.read_parquet(FEAT_PATH)
    feat["date"] = pd.to_datetime(feat["date"])
    df = df.merge(feat[["date", "u", "post_fomc"]], on="date", how="left")
    df["u"] = df["u"].fillna(0.0)
    df["post_fomc"] = df["post_fomc"].fillna(False).astype(bool)

    rv, _log_rv, full_base = _build_full(df)
    u_col = df["u"].to_numpy(dtype=np.float64).reshape(-1, 1)
    full_text = np.column_stack([full_base, u_col])  # one extra column
    har = full_base[:, :3]
    post = df["post_fomc"].to_numpy(dtype=bool)

    # Bonferroni two-sided quantiles for the family of N_TESTS tests.
    per_test = FAMILY_ALPHA / N_TESTS
    q_lo, q_hi = per_test / 2.0, 1.0 - per_test / 2.0

    out: dict[str, Any] = {
        "preregistration": "docs/research/corner-a-text-uncertainty-rv-preregistration.md",
        "n_days": int(len(df)),
        "seeds": list(SEEDS),
        "n_folds": N_FOLDS,
        "epochs": EPOCHS,
        "family_alpha": FAMILY_ALPHA,
        "n_tests": N_TESTS,
        "bonferroni_ci_quantiles": [q_lo, q_hi],
        "feature": {
            "u_mean": float(df["u"].mean()),
            "u_std": float(df["u"].std()),
            "post_fomc_days": int(post.sum()),
        },
        "by_horizon": {},
        "hit_cells": [],
    }

    for h in HORIZONS:
        y = _forward_log_rv(rv, h)
        idx = np.where(~np.isnan(y))[0]
        folds = walk_forward_splits(len(idx), n_folds=N_FOLDS, embargo=max(h, 1) + 1)
        pools: dict[str, list[float]] = {"har": [], "ens": [], "ens_text": [], "true": []}
        post_pool: list[bool] = []

        for tr_l, te_l in folds:
            tr, te = idx[np.array(tr_l)], idx[np.array(te_l)]
            ytr, yte = y[tr], y[te]
            har_fit_tr = _fit_predict_ols(har[tr], ytr, har[tr])
            resid_tr = (ytr - har_fit_tr).reshape(-1, 1)
            har_pred_te = _fit_predict_ols(har[tr], ytr, har[te])

            _, ens_te = _ensemble_predict(
                har_fit_tr, full_base[tr], resid_tr, full_base[te], har_pred_te,
                seeds=SEEDS, epochs=EPOCHS, device=device,
            )
            _, ens_text_te = _ensemble_predict(
                har_fit_tr, full_text[tr], resid_tr, full_text[te], har_pred_te,
                seeds=SEEDS, epochs=EPOCHS, device=device,
            )
            pools["har"].extend(har_pred_te.tolist())
            pools["ens"].extend(ens_te.tolist())
            pools["ens_text"].extend(ens_text_te.tolist())
            pools["true"].extend(yte.tolist())
            post_pool.extend(post[te].tolist())

        arr = {k: np.asarray(v) for k, v in pools.items()}
        post_arr = np.asarray(post_pool, dtype=bool)
        row: dict[str, Any] = {}
        for cell, mask in (("full", np.ones(len(post_arr), bool)), ("post_fomc", post_arr)):
            if mask.sum() < 30:
                row[cell] = {"n": int(mask.sum()), "note": "too few obs"}
                continue
            har_c, ens_c, ext_c, true_c = (arr["har"][mask], arr["ens"][mask],
                                           arr["ens_text"][mask], arr["true"][mask])
            ci = _boot_gain_ci(ext_c, ens_c, true_c, block=max(h, 1), q_lo=q_lo, q_hi=q_hi)
            delta = _qlike(ens_c, true_c) - _qlike(ext_c, true_c)
            hit = bool(ci[0] > 0)
            row[cell] = {
                "n": int(mask.sum()),
                "qlike_har": _qlike(har_c, true_c),
                "qlike_ens": _qlike(ens_c, true_c),
                "qlike_ens_text": _qlike(ext_c, true_c),
                "delta_qlike_text": float(delta),
                "bonferroni_ci": ci,
                "hit": hit,
            }
            if hit:
                out["hit_cells"].append(f"h{h}:{cell}")
        out["by_horizon"][f"h{h}"] = row

    out["verdict"] = "CORNER_A_POSITIVE" if out["hit_cells"] else "NULL"
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_PATH).write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = run_experiment(device=dev)
    print(json.dumps(res, indent=2))
