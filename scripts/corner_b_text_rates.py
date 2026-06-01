"""Corner B experiment — does FOMC stance predict short-end yield reactions?

Pre-registered in docs/research/corner-b-text-rates-preregistration.md. Event
study: for each FOMC statement, predict the 1-day / 5-day reaction of the 2Y and
1Y Treasury yield from a pre-meeting OLS baseline with vs without the stance
signal s = P(hawk) − P(dove). Walk-forward by event; Diebold-Mariano on the
squared-error differential with Bonferroni over the 4 cells. Pure CPU/numpy.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EVENTS = "data/processed/tp_v3_full_rebuild_2026_05_30/events.parquet"
STANCE = "data/artifacts/corner_b_text_rates/stance_daily.parquet"
OUT = "data/artifacts/corner_b_text_rates/result.json"
BURN_IN = 40
HAC_LAG = 5
FAMILY_ALPHA = 0.10
N_TESTS = 4
BONF_P = FAMILY_ALPHA / N_TESTS  # 0.025


def _load_yield(sym: str) -> pd.DataFrame:
    obs = json.load(open(f"data/external/fred/{sym}.json"))["observations"]
    df = pd.DataFrame(obs)[["date", "value"]]
    df = df[df["value"] != "."].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["y"] = df["value"].astype(float)
    return df[["date", "y"]].sort_values("date").reset_index(drop=True)


def _ols_fit_predict(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray) -> float:
    a_tr = np.column_stack([np.ones(len(xtr)), xtr])
    a_te = np.concatenate([[1.0], xte])
    beta, *_ = np.linalg.lstsq(a_tr, ytr, rcond=None)
    return float(a_te @ beta)


def _dm_pvalue(e1: np.ndarray, e2: np.ndarray, lag: int) -> tuple[float, float]:
    """Diebold-Mariano on squared-error loss diff d = e1^2 - e2^2, NW-HAC var."""
    d = e1**2 - e2**2
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    gamma0 = float((dc @ dc) / n)
    var = gamma0
    for ell in range(1, min(lag, n - 1) + 1):
        g = float((dc[ell:] @ dc[:-ell]) / n)
        var += 2.0 * (1.0 - ell / (lag + 1.0)) * g
    if var <= 0 or n < 8:
        return float("nan"), float("nan")
    dm = dbar / math.sqrt(var / n)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(dm) / math.sqrt(2.0))))
    return float(dm), float(p)


def run_experiment() -> dict[str, Any]:
    y2 = _load_yield("DGS2").rename(columns={"y": "y2"})
    y1 = _load_yield("DGS1").rename(columns={"y": "y1"})
    yld = y2.merge(y1, on="date", how="inner").sort_values("date").reset_index(drop=True)
    dates = yld["date"].to_numpy()
    stance = pd.read_parquet(STANCE)
    stance["date"] = pd.to_datetime(stance["date"])
    s_by_date = dict(zip(stance["date"], stance["s"], strict=False))

    ev = pd.read_parquet(EVENTS)
    st = ev[ev["event_kind"] == "statement"].copy()
    st["event_date"] = pd.to_datetime(st["event_date"])
    st = (
        st.sort_values(["event_date", "token_count"])
        .drop_duplicates("event_date", keep="last")
        .sort_values("event_date")
        .reset_index(drop=True)
    )

    # pre-compute Δy + rolling baseline features on the yield calendar
    feat: dict[str, dict[str, np.ndarray]] = {}
    for ten in ("y2", "y1"):
        y = yld[ten].to_numpy(dtype=float)
        dy = np.concatenate([[np.nan], np.diff(y)])
        mom5 = pd.Series(dy).rolling(5).mean().to_numpy()
        vol10 = pd.Series(dy).rolling(10).std().to_numpy()
        feat[ten] = {"y": y, "dy": dy, "mom5": mom5, "vol10": vol10}

    # build event rows: pre-meeting features + reactions
    rows: list[dict[str, Any]] = []
    for d_i in st["event_date"]:
        pos = int(np.searchsorted(dates, np.datetime64(d_i), side="left"))
        if pos < 11 or pos + 5 >= len(dates):  # need t-1 features + t+5 target
            continue
        t = pos  # first yield day >= statement date
        s_i = s_by_date.get(pd.Timestamp(dates[t]))
        if s_i is None or not np.isfinite(s_i):
            continue
        row: dict[str, Any] = {"date": pd.Timestamp(dates[t]), "s": float(s_i)}
        ok = True
        for ten in ("y2", "y1"):
            f = feat[ten]
            base = [f["mom5"][t - 1], f["y"][t - 1], f["vol10"][t - 1]]
            if not np.all(np.isfinite(base)):
                ok = False
                break
            row[f"{ten}_base"] = base
            row[f"{ten}_r1"] = float((f["y"][t] - f["y"][t - 1]) * 100.0)
            row[f"{ten}_r5"] = float((f["y"][t + 5] - f["y"][t - 1]) * 100.0)
        if ok:
            rows.append(row)
    ev_df = pd.DataFrame(rows).reset_index(drop=True)

    out: dict[str, Any] = {
        "preregistration": "docs/research/corner-b-text-rates-preregistration.md",
        "n_events": int(len(ev_df)),
        "burn_in": BURN_IN,
        "family_alpha": FAMILY_ALPHA,
        "n_tests": N_TESTS,
        "bonferroni_p": BONF_P,
        "stance_std": float(ev_df["s"].std()),
        "cells": {},
        "hit_cells": [],
    }

    for ten in ("y2", "y1"):
        for hz in ("r1", "r5"):
            tgt = ev_df[f"{ten}_r1" if hz == "r1" else f"{ten}_r5"].to_numpy(dtype=float)
            base_x = np.array(ev_df[f"{ten}_base"].tolist(), dtype=float)
            s_x = ev_df["s"].to_numpy(dtype=float).reshape(-1, 1)
            treat_x = np.column_stack([base_x, s_x])
            n = len(ev_df)
            pb, pt, truth = [], [], []
            for k in range(BURN_IN, n):
                tr = slice(0, k)
                pb.append(_ols_fit_predict(base_x[tr], tgt[tr], base_x[k]))
                pt.append(_ols_fit_predict(treat_x[tr], tgt[tr], treat_x[k]))
                truth.append(tgt[k])
            pb_a, pt_a, tr_a = np.array(pb), np.array(pt), np.array(truth)
            e_base, e_treat = tr_a - pb_a, tr_a - pt_a
            mse_b, mse_t = float((e_base**2).mean()), float((e_treat**2).mean())
            mse_zero = float((tr_a**2).mean())
            dm, p = _dm_pvalue(e_base, e_treat, HAC_LAG)
            diracc_b = float((np.sign(pb_a) == np.sign(tr_a)).mean())
            diracc_t = float((np.sign(pt_a) == np.sign(tr_a)).mean())
            hit = bool(np.isfinite(p) and p < BONF_P and mse_t < mse_b)
            key = f"{ten}_{hz}"
            out["cells"][key] = {
                "n_oos": int(len(tr_a)),
                "mse_baseline": mse_b,
                "mse_treatment": mse_t,
                "mse_predict_zero": mse_zero,
                "delta_mse": mse_b - mse_t,
                "dm_stat": dm,
                "dm_p": p,
                "diracc_baseline": diracc_b,
                "diracc_treatment": diracc_t,
                "hit": hit,
            }
            if hit:
                out["hit_cells"].append(key)

    out["verdict"] = "CORNER_B_POSITIVE" if out["hit_cells"] else "NULL"
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))
