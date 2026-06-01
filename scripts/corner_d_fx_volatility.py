"""Corner D experiment — does FOMC text help dollar-index (FX) volatility forecasting?

Pre-registered in docs/research/corner-d-fx-volatility-preregistration.md. Closes
the project's original-but-untested FX target. DXY daily squared-return RV proxy;
QLIKE-DLq HAR ensemble with vs without text [certainty u, stance s]; paired
walk-forward; moving-block bootstrap CI with Bonferroni over 3 horizons.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from app.data.dense_daily_dataset import walk_forward_splits  # noqa: E402
from app.data.dense_forecast_train import _fit_predict_ols  # noqa: E402
from app.data.intraday_rv_forecast import _har_lags, _qlike, _qlike_pointwise  # noqa: E402
from app.data.intraday_rv_production import _ensemble_predict  # noqa: E402
from app.models.config import (  # noqa: E402
    MULTI_TASK_CERTAINTY_LABELS as CL,
)
from app.models.config import (
    MULTI_TASK_STANCE_LABELS as SL,
)
from app.services import multi_axis_classifier as mac  # noqa: E402

SEEDS = (11, 22, 33, 44, 55)
HORIZONS = (1, 5, 22)
N_FOLDS = 5
EPOCHS = int(os.environ.get("CORNER_D_EPOCHS", "300"))
FAMILY_ALPHA = 0.10
N_TESTS = 3
_EPS = 1e-8
DXY = "data/processed/tp_v3_full_rebuild_2026_05_30/_market_cache/DX-Y.NYB.parquet"
EVENTS = "data/processed/tp_v3_full_rebuild_2026_05_30/events.parquet"
OUT = "data/artifacts/corner_d_fx_volatility/result.json"


def _score_statements() -> pd.DataFrame:
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
    hi, di = SL.index("hawkish"), SL.index("dovish")
    mdev = next(state.model.parameters()).device
    us, ss = [], []
    for text in st["text"].tolist():
        enc = state.tokenizer(str(text), return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            lg = state.model(
                input_ids=enc["input_ids"].to(mdev),
                attention_mask=enc["attention_mask"].to(mdev),
            )
        pc = torch.softmax(lg["certainty"], dim=-1)[0].cpu()
        ps = torch.softmax(lg["stance"], dim=-1)[0].cpu()
        us.append(float(pc[ui] - pc[ci]))
        ss.append(float(ps[hi] - ps[di]))
    st["u"], st["s"] = us, ss
    return st[["event_date", "u", "s"]]


def _forward_log_mean_rv(rv: np.ndarray, h: int) -> np.ndarray:
    n = len(rv)
    y = np.full(n, np.nan)
    for t in range(n - h):
        y[t] = np.log(rv[t + 1 : t + 1 + h].mean() + _EPS)
    return y


def _boot_ci(pred: np.ndarray, base: np.ndarray, true: np.ndarray, *, block: int,
             q_lo: float, q_hi: float, seed: int = 11, n_boot: int = 2000) -> list[float]:
    gain = _qlike_pointwise(base, true) - _qlike_pointwise(pred, true)
    n = len(gain)
    if n <= block:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    boots = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        boots.append(float(gain[idx].mean()))
    return [float(np.quantile(boots, q_lo)), float(np.quantile(boots, q_hi))]


def run_experiment(device: str = "cpu") -> dict[str, Any]:
    px = pd.read_parquet(DXY)[["date", "close"]].copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values("date").reset_index(drop=True)
    r = np.concatenate([[np.nan], np.diff(np.log(px["close"].to_numpy(dtype=float)))])
    px["rv"] = r**2
    px = px.iloc[1:].reset_index(drop=True)  # drop first NaN return

    txt = _score_statements()
    px = pd.merge_asof(px, txt.rename(columns={"event_date": "date"}), on="date",
                       direction="backward")
    px["u"] = px["u"].fillna(0.0)
    px["s"] = px["s"].fillna(0.0)

    rv = px["rv"].to_numpy(dtype=float)
    log_rv = np.log(rv + _EPS)
    har = _har_lags(log_rv)
    full_base = har
    full_text = np.column_stack([har, px["u"].to_numpy(float), px["s"].to_numpy(float)])

    per_test = FAMILY_ALPHA / N_TESTS
    q_lo, q_hi = per_test / 2.0, 1.0 - per_test / 2.0
    out: dict[str, Any] = {
        "preregistration": "docs/research/corner-d-fx-volatility-preregistration.md",
        "target": "DXY daily r^2 realized-variance proxy",
        "n_days": int(len(px)), "epochs": EPOCHS,
        "bonferroni_ci_quantiles": [q_lo, q_hi],
        "n_statements": int((txt["event_date"] >= px["date"].min()).sum()),
        "by_horizon": {}, "hit_horizons": [],
    }

    for h in HORIZONS:
        y = _forward_log_mean_rv(rv, h)
        valid = ~np.isnan(y) & ~np.isnan(har).any(axis=1)
        idx = np.where(valid)[0]
        folds = walk_forward_splits(len(idx), n_folds=N_FOLDS, embargo=max(h, 1) + 1)
        pb, pt, tr_pool = [], [], []
        for tr_l, te_l in folds:
            tr, te = idx[np.array(tr_l)], idx[np.array(te_l)]
            ytr, yte = y[tr], y[te]
            har_fit_tr = _fit_predict_ols(full_base[tr], ytr, full_base[tr])
            resid_tr = (ytr - har_fit_tr).reshape(-1, 1)
            har_pred_te = _fit_predict_ols(full_base[tr], ytr, full_base[te])
            _, ens_b = _ensemble_predict(har_fit_tr, full_base[tr], resid_tr, full_base[te],
                                         har_pred_te, seeds=SEEDS, epochs=EPOCHS, device=device)
            _, ens_t = _ensemble_predict(har_fit_tr, full_text[tr], resid_tr, full_text[te],
                                         har_pred_te, seeds=SEEDS, epochs=EPOCHS, device=device)
            pb.extend(ens_b.tolist())
            pt.extend(ens_t.tolist())
            tr_pool.extend(yte.tolist())
        base, text, true = np.array(pb), np.array(pt), np.array(tr_pool)
        ci = _boot_ci(text, base, true, block=max(h, 1), q_lo=q_lo, q_hi=q_hi)
        delta = _qlike(base, true) - _qlike(text, true)
        hit = bool(ci[0] > 0)
        out["by_horizon"][f"h{h}"] = {
            "n_oos": int(len(true)),
            "qlike_base": _qlike(base, true),
            "qlike_text": _qlike(text, true),
            "delta_qlike": float(delta),
            "bonferroni_ci": ci,
            "hit": hit,
        }
        if hit:
            out["hit_horizons"].append(f"h{h}")

    out["verdict"] = "CORNER_D_POSITIVE" if out["hit_horizons"] else "NULL"
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(json.dumps(run_experiment(device=dev), indent=2))
