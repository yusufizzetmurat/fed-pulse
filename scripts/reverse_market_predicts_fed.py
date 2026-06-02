"""Reverse direction: does pre-meeting market state predict FOMC statement stance?

Pre-registration: docs/research/reverse-market-predicts-fed-preregistration.md

M0 (baseline): OLS  s_t ~ s_{t-1}          (the Fed's own inertia)
M1 (market)  : Ridge s_t ~ s_{t-1} + 8 leak-safe pre-meeting market features

Walk-forward expanding (initial train 40 meetings); primary test = one-sided
Diebold-Mariano (Newey-West) on squared-error diffs M0-M1, plus OOS incremental
R^2 of M1 over the persistence baseline with a moving-block bootstrap CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

EVENTS = "data/processed/tp_v3_full_rebuild_2026_05_30/events.parquet"
STANCE = "data/artifacts/corner_b_text_rates/stance_daily.parquet"
MKT = "data/processed/tp_v3_full_rebuild_2026_05_30/_market_cache"
OUT = "data/artifacts/reverse_market_predicts_fed/result.json"

RATE_FEATS = [
    "pre_meeting_implied_next_move_bps",
    "pre_meeting_slope_10y_2y",
    "pre_meeting_trailing_2y_yield_change_5d_bps",
    "pre_meeting_yield_2y",
    "pre_meeting_days_since_last_rate_change",
]
INTERMEETING = {"2020-03-03", "2020-03-15"}  # emergency actions, excluded
K0 = 40
ALPHAS = [0.1, 1.0, 10.0, 100.0]


def _ols_fit(X: np.ndarray, y: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X])
    n_p = A.shape[1]
    reg = alpha * np.eye(n_p)
    reg[0, 0] = 0.0  # do not penalize intercept
    return np.linalg.solve(A.T @ A + reg, A.T @ y)


def _pred(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ w


def _pick_alpha(Xtr: np.ndarray, ytr: np.ndarray) -> float:
    """Inner split: last 10 train rows as val, pick alpha minimizing val MSE."""
    if len(Xtr) < 25:
        return 1.0
    nv = 10
    xi, yi = Xtr[:-nv], ytr[:-nv]
    xv, yv = Xtr[-nv:], ytr[-nv:]
    m, s = xi.mean(0), xi.std(0)
    s = np.where(s > 0, s, 1.0)
    best_a, best_e = 1.0, np.inf
    for a in ALPHAS:
        w = _ols_fit((xi - m) / s, yi, alpha=a)
        e = float(np.mean((yv - _pred(w, (xv - m) / s)) ** 2))
        if e < best_e:
            best_e, best_a = e, a
    return best_a


def _dm_pvalue(e0: np.ndarray, e1: np.ndarray) -> float:
    """One-sided DM (H1: M1 better, i.e. mean(e0-e1) > 0), Newey-West HAC."""
    d = e0 - e1
    n = len(d)
    dbar = d.mean()
    lag = max(1, int(n ** 0.25))
    g0 = np.mean((d - dbar) ** 2)
    var = g0
    for k in range(1, lag + 1):
        gk = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2 * (1 - k / (lag + 1)) * gk
    se = np.sqrt(max(var, 1e-12) / n)
    z = dbar / se
    # one-sided normal upper-tail
    from math import erfc, sqrt
    return float(0.5 * erfc(z / sqrt(2)))


def _boot_incr_r2(e0: np.ndarray, e1: np.ndarray, *, seed: int = 11, n_boot: int = 2000) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(e0)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s0, s1 = e0[idx].sum(), e1[idx].sum()
        vals.append(1.0 - s1 / s0 if s0 > 0 else 0.0)
    return [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))]


def _market_block(meeting_dates: list[Any]) -> np.ndarray:
    import pandas as pd

    gspc = pd.read_parquet(f"{MKT}/GSPC.parquet").sort_values("date").reset_index(drop=True)
    vix = pd.read_parquet(f"{MKT}/VIX.parquet").sort_values("date").reset_index(drop=True)
    gd = pd.to_datetime(gspc["date"]).to_numpy()
    vd = pd.to_datetime(vix["date"]).to_numpy()
    gc = gspc["close"].to_numpy(dtype=np.float64)
    vc = vix["close"].to_numpy(dtype=np.float64)
    out = []
    for d in meeting_dates:
        dd = np.datetime64(pd.Timestamp(d))
        gi = int(np.searchsorted(gd, dd) - 1)  # last trading day strictly before d
        vi = int(np.searchsorted(vd, dd) - 1)
        if gi < 22 or vi < 22:
            out.append([np.nan, np.nan, np.nan])
            continue
        spx_ret = float(np.log(gc[gi] / gc[gi - 22]))
        vix_lvl = float(vc[vi])
        vix_chg = float(np.log(vc[vi] / vc[vi - 22]))
        out.append([spx_ret, vix_lvl, vix_chg])
    return np.asarray(out, dtype=np.float64)


def _run_for_target(frame: dict[str, np.ndarray]) -> dict[str, Any]:
    y, sprev, X = frame["y"], frame["sprev"], frame["X"]
    n = len(y)
    p0, p1 = [], []
    for t in range(K0, n):
        ytr = y[:t]
        # M0: persistence OLS s ~ s_prev
        w0 = _ols_fit(sprev[:t].reshape(-1, 1), ytr)
        p0.append(float(_pred(w0, sprev[t : t + 1].reshape(-1, 1))[0]))
        # M1: Ridge s ~ s_prev + market block
        Xtr = np.column_stack([sprev[:t], X[:t]])
        a = _pick_alpha(Xtr, ytr)
        m, s = Xtr.mean(0), Xtr.std(0)
        s = np.where(s > 0, s, 1.0)
        w1 = _ols_fit((Xtr - m) / s, ytr, alpha=a)
        xt = np.column_stack([sprev[t : t + 1], X[t : t + 1]])
        p1.append(float(_pred(w1, (xt - m) / s)[0]))
    p0, p1 = np.asarray(p0), np.asarray(p1)
    yt = y[K0:]
    sp = sprev[K0:]
    e0 = (yt - p0) ** 2
    e1 = (yt - p1) ** 2
    dir_acc = float(np.mean(np.sign(p1 - sp) == np.sign(yt - sp)))
    from math import comb
    nC = len(yt)
    k = int(round(dir_acc * nC))
    binom_p = float(sum(comb(nC, i) for i in range(k, nC + 1)) / 2 ** nC)
    return {
        "n_oos": nC,
        "mse_M0": float(e0.mean()),
        "mse_M1": float(e1.mean()),
        "incr_r2_M1_over_M0": float(1.0 - e1.sum() / e0.sum()),
        "incr_r2_ci90": _boot_incr_r2(e0, e1),
        "dm_pvalue_onesided": _dm_pvalue(e0, e1),
        "dir_acc": dir_acc,
        "dir_binom_p_onesided": binom_p,
    }


def main() -> None:
    import pandas as pd

    ev = pd.read_parquet(EVENTS)
    ev = ev[ev["event_kind"] == "statement"].copy()
    ev["event_date"] = pd.to_datetime(ev["event_date"])
    ev = ev[ev["event_date"] >= "2010-01-01"]
    ev = ev.sort_values("event_date").drop_duplicates("event_date", keep="first")
    ev["dstr"] = ev["event_date"].dt.strftime("%Y-%m-%d")
    ev = ev[~ev["dstr"].isin(INTERMEETING)]

    sd = pd.read_parquet(STANCE)
    sd["date"] = pd.to_datetime(sd["date"]).dt.strftime("%Y-%m-%d")
    s_by_date = dict(zip(sd["date"], sd["s"]))
    cert_by_date = dict(zip(sd["date"], sd["c"])) if "c" in sd.columns else {}

    dates = ev["dstr"].tolist()
    s = np.array([s_by_date.get(d, np.nan) for d in dates], dtype=np.float64)
    rate = ev[RATE_FEATS].to_numpy(dtype=np.float64)
    mkt = _market_block(list(ev["event_date"]))
    Xfull = np.column_stack([rate, mkt])

    # chronological s_prev
    sprev = np.concatenate([[np.nan], s[:-1]])
    ok = np.isfinite(s) & np.isfinite(sprev) & np.all(np.isfinite(Xfull), axis=1)
    s, sprev, Xfull = s[ok], sprev[ok], Xfull[ok]
    kept_dates = [d for d, k in zip(dates, ok) if k]

    print(f"meetings usable: {len(s)} ({kept_dates[0]} -> {kept_dates[-1]}), OOS = {len(s) - K0}")
    res: dict[str, Any] = {
        "n_meetings": int(len(s)),
        "date_range": [kept_dates[0], kept_dates[-1]],
        "features": ["s_prev", *RATE_FEATS, "spx_ret_22d", "vix_level", "vix_chg_22d"],
        "stance": _run_for_target({"y": s, "sprev": sprev, "X": Xfull}),
    }
    r = res["stance"]
    print(
        f"STANCE: MSE M0 {r['mse_M0']:.4f} | M1 {r['mse_M1']:.4f} | "
        f"incr-R2 {r['incr_r2_M1_over_M0']:+.3f} CI90 {r['incr_r2_ci90']} | "
        f"DM p {r['dm_pvalue_onesided']:.3f} | dir {r['dir_acc']:.3f} (p {r['dir_binom_p_onesided']:.3f})"
    )
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=2))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
