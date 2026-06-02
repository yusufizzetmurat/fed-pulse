"""Corner C experiment — does textual uncertainty predict RV-forecast error magnitude?

Pre-registered in docs/research/corner-c-text-calibration-preregistration.md. The
QLIKE-DLq point forecast is unchanged; we test the *band*. Re-run the ensemble
walk-forward to pool OOS (forecast, truth, certainty u, constant-band coverage),
then ask whether u predicts |residual| incrementally (paired OLS + DM, Bonferroni
over 3 horizons) and whether the constant 90% band mis-covers conditional on u.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.data.dense_daily_dataset import walk_forward_splits
from app.data.dense_forecast_train import _fit_predict_ols
from app.data.intraday_rv_forecast import _forward_log_rv
from app.data.intraday_rv_production import _build_full, _conformal_quantile, _ensemble_predict
from app.determinism import enable_deterministic_mode

SEEDS = (11, 22, 33, 44, 55)
HORIZONS = (1, 5, 22)
N_FOLDS = 5
EPOCHS = int(os.environ.get("CORNER_C_EPOCHS", "300"))
BURN_IN = 100
ALPHA = 0.10  # 90% band
FAMILY_ALPHA = 0.10
N_TESTS = 3
BONF_P = FAMILY_ALPHA / N_TESTS  # 0.0333
RV_PATH = "data/external/alphavantage_bars/spx_5min_daily_rv.parquet"
FEAT_PATH = "data/artifacts/corner_a_text_uncertainty/text_uncertainty_daily.parquet"
OUT = "data/artifacts/corner_c_text_calibration/result.json"


def _dm_pvalue(e1: np.ndarray, e2: np.ndarray, lag: int) -> tuple[float, float]:
    d = e1**2 - e2**2
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    var = float((dc @ dc) / n)
    for ell in range(1, min(lag, n - 1) + 1):
        var += 2.0 * (1.0 - ell / (lag + 1.0)) * float((dc[ell:] @ dc[:-ell]) / n)
    if var <= 0 or n < 8:
        return float("nan"), float("nan")
    dm = dbar / math.sqrt(var / n)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(dm) / math.sqrt(2.0))))
    return float(dm), float(p)


def run_experiment(device: str = "cpu") -> dict[str, Any]:
    enable_deterministic_mode(SEEDS[0])
    if not Path(RV_PATH).exists():
        raise FileNotFoundError(f"Required RV data {RV_PATH} not found")
    if not Path(FEAT_PATH).exists():
        raise FileNotFoundError(f"Required feature data {FEAT_PATH} not found")
    df = pd.read_parquet(RV_PATH).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    feat = pd.read_parquet(FEAT_PATH)
    feat["date"] = pd.to_datetime(feat["date"])
    df = df.merge(feat[["date", "u"]], on="date", how="left")
    df["u"] = df["u"].fillna(0.0)

    rv, _log_rv, full = _build_full(df)
    har = full[:, :3]
    u_all = df["u"].to_numpy(dtype=float)

    out: dict[str, Any] = {
        "preregistration": "docs/research/corner-c-text-calibration-preregistration.md",
        "n_days": int(len(df)),
        "epochs": EPOCHS,
        "alpha": ALPHA,
        "bonferroni_p": BONF_P,
        "by_horizon": {},
        "hit_horizons": [],
    }

    for h in HORIZONS:
        y = _forward_log_rv(rv, h)
        idx = np.where(~np.isnan(y))[0]
        folds = walk_forward_splits(len(idx), n_folds=N_FOLDS, embargo=max(h, 1) + 1)
        ens_pool: list[float] = []
        true_pool: list[float] = []
        u_pool: list[float] = []
        cov_u: list[float] = []  # u where a band existed
        cov_hit: list[int] = []  # 1 if true within constant 90% band
        cal_resid: list[float] = []

        for tr_l, te_l in folds:
            tr, te = idx[np.array(tr_l)], idx[np.array(te_l)]
            ytr, yte = y[tr], y[te]
            har_fit_tr = _fit_predict_ols(har[tr], ytr, har[tr])
            resid_tr = (ytr - har_fit_tr).reshape(-1, 1)
            har_pred_te = _fit_predict_ols(har[tr], ytr, har[te])
            _, ens_te = _ensemble_predict(
                har_fit_tr, full[tr], resid_tr, full[te], har_pred_te,
                seeds=SEEDS, epochs=EPOCHS, device=device,
            )
            # constant walk-forward conformal band from prior folds only
            if cal_resid:
                q = _conformal_quantile(np.asarray(cal_resid), ALPHA)
                inside = (np.abs(yte - ens_te) <= q).astype(int)
                cov_u.extend(u_all[te].tolist())
                cov_hit.extend(inside.tolist())
            cal_resid.extend(np.abs(yte - ens_te).tolist())
            ens_pool.extend(ens_te.tolist())
            true_pool.extend(yte.tolist())
            u_pool.extend(u_all[te].tolist())

        ens = np.asarray(ens_pool)
        true = np.asarray(true_pool)
        u = np.asarray(u_pool)
        abs_r = np.abs(true - ens)
        abs_r_lag = np.concatenate([[abs_r[0]], abs_r[:-1]])

        # paired second stage: predict |residual| from base vs base+u
        base_x = np.column_stack([ens, abs_r_lag])
        treat_x = np.column_stack([ens, abs_r_lag, u])
        n = len(abs_r)
        pb, pt, truth = [], [], []
        for k in range(BURN_IN, n):
            a_b = np.column_stack([np.ones(k), base_x[:k]])
            beta_b, *_ = np.linalg.lstsq(a_b, abs_r[:k], rcond=None)
            pb.append(float(np.concatenate([[1.0], base_x[k]]) @ beta_b))
            a_t = np.column_stack([np.ones(k), treat_x[:k]])
            beta_t, *_ = np.linalg.lstsq(a_t, abs_r[:k], rcond=None)
            pt.append(float(np.concatenate([[1.0], treat_x[k]]) @ beta_t))
            truth.append(abs_r[k])
        pb_a, pt_a, tr_a = np.array(pb), np.array(pt), np.array(truth)
        e_b, e_t = tr_a - pb_a, tr_a - pt_a
        mse_b, mse_t = float((e_b**2).mean()), float((e_t**2).mean())
        dm, p = _dm_pvalue(e_b, e_t, max(h, 1))

        # conditional coverage of the constant 90% band across u-terciles
        cu, ch = np.asarray(cov_u), np.asarray(cov_hit, dtype=float)
        lo_cut, hi_cut = np.quantile(cu, [1 / 3, 2 / 3])
        cov_low = float(ch[cu <= lo_cut].mean())   # most "certain"
        cov_high = float(ch[cu >= hi_cut].mean())  # most "uncertain"
        gap_dir_ok = bool(cov_high < cov_low)  # high-u under-covered, as predicted

        hit = bool(np.isfinite(p) and p < BONF_P and (mse_b - mse_t) > 0 and gap_dir_ok)
        out["by_horizon"][f"h{h}"] = {
            "n_oos": int(len(tr_a)),
            "mse_abs_resid_base": mse_b,
            "mse_abs_resid_text": mse_t,
            "delta_mse": mse_b - mse_t,
            "dm_stat": dm,
            "dm_p": p,
            "band_coverage_low_u": cov_low,
            "band_coverage_high_u": cov_high,
            "coverage_gap_predicted_direction": gap_dir_ok,
            "hit": hit,
        }
        if hit:
            out["hit_horizons"].append(f"h{h}")

    out["verdict"] = "CORNER_C_POSITIVE" if out["hit_horizons"] else "NULL"
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(json.dumps(run_experiment(device=dev), indent=2))
