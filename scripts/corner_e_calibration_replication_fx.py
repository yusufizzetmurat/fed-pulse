"""Corner E — confirmatory replication of the Corner C h1 calibration finding on DXY.

Pre-registered in docs/research/corner-e-calibration-replication-fx-preregistration.md.
Single fixed test: does u = P(unc)-P(cert) predict the 1-day DXY RV-forecast error
magnitude |residual| incrementally over [forecast, lagged |residual|]? Same spec as
Corner C h1, transplanted to the independent DXY asset. DM at alpha=0.05; no search.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from app.data.dense_daily_dataset import walk_forward_splits  # noqa: E402
from app.data.dense_forecast_train import _fit_predict_ols  # noqa: E402
from app.data.intraday_rv_forecast import _har_lags  # noqa: E402
from app.data.intraday_rv_production import _conformal_quantile, _ensemble_predict  # noqa: E402
from app.models.config import MULTI_TASK_CERTAINTY_LABELS as CL  # noqa: E402
from app.services import multi_axis_classifier as mac  # noqa: E402
from app.determinism import enable_deterministic_mode  # noqa: E402

SEEDS = (11, 22, 33, 44, 55)
H = 1
N_FOLDS = 5
EPOCHS = int(os.environ.get("CORNER_E_EPOCHS", "300"))
BURN_IN = 100
ALPHA = 0.10
ALPHA_TEST = 0.05
_EPS = 1e-8
DXY = "data/processed/tp_v3_full_rebuild_2026_05_30/_market_cache/DX-Y.NYB.parquet"
EVENTS = "data/processed/tp_v3_full_rebuild_2026_05_30/events.parquet"
OUT = "data/artifacts/corner_e_calibration_replication_fx/result.json"


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


def _score_u() -> pd.DataFrame:
    ev = pd.read_parquet(EVENTS)
    st = ev[ev["event_kind"] == "statement"].copy()
    st["event_date"] = pd.to_datetime(st["event_date"])
    st = (
        st.sort_values(["event_date", "token_count"])
        .drop_duplicates("event_date", keep="last")
        .sort_values("event_date")
        .reset_index(drop=True)
    )
    import torch

    state = mac.get_classifier()
    if state is None:
        raise RuntimeError("classifier failed to load")
    ui, ci = CL.index("uncertain"), CL.index("certain")
    mdev = next(state.model.parameters()).device
    us = []
    for text in st["text"].tolist():
        enc = state.tokenizer(str(text), return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            lg = state.model(
                input_ids=enc["input_ids"].to(mdev),
                attention_mask=enc["attention_mask"].to(mdev),
            )
        p = torch.softmax(lg["certainty"], dim=-1)[0].cpu()
        us.append(float(p[ui] - p[ci]))
    st["u"] = us
    return st[["event_date", "u"]]


def run_experiment(device: str = "cpu") -> dict[str, Any]:
    enable_deterministic_mode(SEEDS[0])
    if not Path(DXY).exists():
        raise FileNotFoundError(f"Required DXY cache {DXY} not found")
    if not Path(EVENTS).exists():
        raise FileNotFoundError(f"Required events table {EVENTS} not found")
    px = pd.read_parquet(DXY)[["date", "close"]].copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values("date").reset_index(drop=True)
    r = np.concatenate([[np.nan], np.diff(np.log(px["close"].to_numpy(dtype=float)))])
    px["rv"] = r**2
    px = px.iloc[1:].reset_index(drop=True)
    txt = _score_u()
    px = pd.merge_asof(px, txt.rename(columns={"event_date": "date"}), on="date", direction="backward")
    px["u"] = px["u"].fillna(0.0)

    rv = px["rv"].to_numpy(dtype=float)
    log_rv = np.log(rv + _EPS)
    har = _har_lags(log_rv)
    u_all = px["u"].to_numpy(dtype=float)

    # h1 forward target
    y = np.full(len(rv), np.nan)
    y[:-1] = np.log(rv[1:] + _EPS)
    valid = ~np.isnan(y) & ~np.isnan(har).any(axis=1)
    idx = np.where(valid)[0]
    folds = walk_forward_splits(len(idx), n_folds=N_FOLDS, embargo=H + 1)

    ens_pool, true_pool, u_pool = [], [], []
    cov_u, cov_hit, cal_resid = [], [], []
    for tr_l, te_l in folds:
        tr, te = idx[np.array(tr_l)], idx[np.array(te_l)]
        ytr, yte = y[tr], y[te]
        har_fit_tr = _fit_predict_ols(har[tr], ytr, har[tr])
        resid_tr = (ytr - har_fit_tr).reshape(-1, 1)
        har_pred_te = _fit_predict_ols(har[tr], ytr, har[te])
        _, ens_te = _ensemble_predict(
            har_fit_tr, har[tr], resid_tr, har[te], har_pred_te,
            seeds=SEEDS, epochs=EPOCHS, device=device,
        )
        if cal_resid:
            q = _conformal_quantile(np.asarray(cal_resid), ALPHA)
            cov_u.extend(u_all[te].tolist())
            cov_hit.extend((np.abs(yte - ens_te) <= q).astype(int).tolist())
        cal_resid.extend(np.abs(yte - ens_te).tolist())
        ens_pool.extend(ens_te.tolist())
        true_pool.extend(yte.tolist())
        u_pool.extend(u_all[te].tolist())

    ens, true, u = np.asarray(ens_pool), np.asarray(true_pool), np.asarray(u_pool)
    abs_r = np.abs(true - ens)
    abs_r_lag = np.concatenate([[abs_r[0]], abs_r[:-1]])
    base_x = np.column_stack([ens, abs_r_lag])
    treat_x = np.column_stack([ens, abs_r_lag, u])
    n = len(abs_r)
    pb, pt, truth = [], [], []
    for k in range(BURN_IN, n):
        ab = np.column_stack([np.ones(k), base_x[:k]])
        beta_b, *_ = np.linalg.lstsq(ab, abs_r[:k], rcond=None)
        pb.append(float(np.concatenate([[1.0], base_x[k]]) @ beta_b))
        at = np.column_stack([np.ones(k), treat_x[:k]])
        beta_t, *_ = np.linalg.lstsq(at, abs_r[:k], rcond=None)
        pt.append(float(np.concatenate([[1.0], treat_x[k]]) @ beta_t))
        truth.append(abs_r[k])
    pb_a, pt_a, tr_a = np.array(pb), np.array(pt), np.array(truth)
    e_b, e_t = tr_a - pb_a, tr_a - pt_a
    mse_b, mse_t = float((e_b**2).mean()), float((e_t**2).mean())
    dm, p = _dm_pvalue(e_b, e_t, H)
    cu, ch = np.asarray(cov_u), np.asarray(cov_hit, dtype=float)
    lo_cut, hi_cut = np.quantile(cu, [1 / 3, 2 / 3])
    cov_low, cov_high = float(ch[cu <= lo_cut].mean()), float(ch[cu >= hi_cut].mean())
    gap_dir_ok = bool(cov_high < cov_low)
    replicated = bool(np.isfinite(p) and p < ALPHA_TEST and (mse_b - mse_t) > 0 and gap_dir_ok)

    out = {
        "preregistration": "docs/research/corner-e-calibration-replication-fx-preregistration.md",
        "asset": "DXY (independent of SPX)", "horizon": "h1", "n_oos": int(len(tr_a)),
        "alpha_test": ALPHA_TEST,
        "mse_abs_resid_base": mse_b, "mse_abs_resid_text": mse_t,
        "delta_mse": mse_b - mse_t, "dm_stat": dm, "dm_p": p,
        "band_coverage_low_u": cov_low, "band_coverage_high_u": cov_high,
        "coverage_gap_predicted_direction": gap_dir_ok,
        "replicated": replicated,
        "verdict": "C_H1_REPLICATES" if replicated else "C_H1_DOES_NOT_REPLICATE",
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(json.dumps(run_experiment(device=dev), indent=2))
