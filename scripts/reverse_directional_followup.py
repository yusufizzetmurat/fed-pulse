"""Powered confirmation of the market->Fed directional lead (pre-registered).

Pre-registration: docs/research/reverse-directional-followup-preregistration.md

Rebuilds pre-meeting rate-expectation features from FRED/cache back to 2010,
target = stance shift Delta_s = s_t - s_{t-1}. M1 = Ridge(Delta_s ~ 8 features).
PRIMARY = directional accuracy sign(predDelta) == sign(Delta) on full OOS;
replication = the pre-2016 OOS slice (data the discovery never saw).
"""

from __future__ import annotations

import json
from math import comb, erfc, sqrt
from pathlib import Path
from typing import Any

import numpy as np

FRED = "data/external/fred"
MKT = "data/processed/tp_v3_full_rebuild_2026_05_30/_market_cache"
EVENTS = "data/processed/tp_v3_full_rebuild_2026_05_30/events.parquet"
STANCE = "data/artifacts/corner_b_text_rates/stance_daily.parquet"
OUT = "data/artifacts/reverse_directional_followup/result.json"
INTERMEETING = {"2020-03-03", "2020-03-15"}
K0 = 30
ALPHAS = [0.1, 1.0, 10.0, 100.0]


def _fred_series(name: str) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    j = json.load(open(f"{FRED}/{name}.json"))
    obs = j["observations"]
    d, v = [], []
    for o in obs:
        if o["value"] not in (".", "", None):
            d.append(np.datetime64(o["date"]))
            v.append(float(o["value"]))
    return np.array(d), np.array(v, dtype=np.float64)


def _cache_series(name: str) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_parquet(f"{MKT}/{name}.parquet").sort_values("date")
    return pd.to_datetime(df["date"]).to_numpy(), df["close"].to_numpy(dtype=np.float64)


def _asof(dates: np.ndarray, vals: np.ndarray, d: np.datetime64, back: int = 0) -> float:
    """Value at the (back)-th observation strictly before d. back=0 => last before d."""
    i = int(np.searchsorted(dates, d) - 1 - back)
    return float(vals[i]) if 0 <= i < len(vals) else float("nan")


def _ols_fit(X: np.ndarray, y: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X])
    reg = alpha * np.eye(A.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(A.T @ A + reg, A.T @ y)


def _pred(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ w


def _pick_alpha(X: np.ndarray, y: np.ndarray) -> float:
    if len(X) < 22:
        return 10.0
    nv = 8
    xi, yi, xv, yv = X[:-nv], y[:-nv], X[-nv:], y[-nv:]
    m, s = xi.mean(0), xi.std(0)
    s = np.where(s > 0, s, 1.0)
    best_a, best_e = 10.0, np.inf
    for a in ALPHAS:
        w = _ols_fit((xi - m) / s, yi, alpha=a)
        e = float(np.mean((yv - _pred(w, (xv - m) / s)) ** 2))
        if e < best_e:
            best_e, best_a = e, a
    return best_a


def _binom_p(k: int, n: int) -> float:
    return float(sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def _dm_p(e_base: np.ndarray, e_m1: np.ndarray) -> float:
    d = e_base - e_m1
    n = len(d)
    db = d.mean()
    lag = max(1, int(n ** 0.25))
    var = np.mean((d - db) ** 2)
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * np.mean((d[k:] - db) * (d[:-k] - db))
    z = db / np.sqrt(max(var, 1e-12) / n)
    return float(0.5 * erfc(z / sqrt(2)))


def main() -> None:
    import pandas as pd

    dgs1_d, dgs1 = _fred_series("DGS1")
    dgs2_d, dgs2 = _fred_series("DGS2")
    ffu_d, ffu = _fred_series("DFEDTARU")
    tnx_d, tnx = _cache_series("TNX")
    vix_d, vix = _cache_series("VIX")
    gspc_d, gspc = _cache_series("GSPC")
    tnx_pct = tnx / 10.0 if np.median(tnx) > 20 else tnx  # ^TNX may be yield x10

    # last DFEDTARU change date series (step changes)
    chg_dates = ffu_d[np.concatenate([[True], np.diff(ffu) != 0])]

    ev = pd.read_parquet(EVENTS)
    ev = ev[ev["event_kind"] == "statement"].copy()
    ev["event_date"] = pd.to_datetime(ev["event_date"])
    ev = ev[ev["event_date"] >= "2010-01-01"].sort_values("event_date")
    ev = ev.drop_duplicates("event_date", keep="first")
    ev["dstr"] = ev["event_date"].dt.strftime("%Y-%m-%d")
    ev = ev[~ev["dstr"].isin(INTERMEETING)]

    sd = pd.read_parquet(STANCE)
    sd["date"] = pd.to_datetime(sd["date"]).dt.strftime("%Y-%m-%d")
    s_by = dict(zip(sd["date"], sd["s"]))

    rows = []
    for d, dstr in zip(ev["event_date"].to_numpy(), ev["dstr"]):
        dd = np.datetime64(pd.Timestamp(d))
        f_imp = (_asof(dgs1_d, dgs1, dd) - _asof(ffu_d, ffu, dd)) * 100.0
        f_slope = _asof(tnx_d, tnx_pct, dd) - _asof(dgs2_d, dgs2, dd)
        f_chg = (_asof(dgs2_d, dgs2, dd) - _asof(dgs2_d, dgs2, dd, back=5)) * 100.0
        f_y2 = _asof(dgs2_d, dgs2, dd)
        last_chg = chg_dates[chg_dates < dd]
        f_dslc = float((dd - last_chg[-1]) / np.timedelta64(1, "D")) if len(last_chg) else np.nan
        gi = int(np.searchsorted(gspc_d, dd) - 1)
        vi = int(np.searchsorted(vix_d, dd) - 1)
        f_spx = float(np.log(gspc[gi] / gspc[gi - 22])) if gi >= 22 else np.nan
        f_vlvl = float(vix[vi]) if vi >= 0 else np.nan
        f_vchg = float(np.log(vix[vi] / vix[vi - 22])) if vi >= 22 else np.nan
        rows.append((dstr, s_by.get(dstr, np.nan),
                     [f_imp, f_slope, f_chg, f_y2, f_dslc, f_spx, f_vlvl, f_vchg]))

    dates = [r[0] for r in rows]
    s = np.array([r[1] for r in rows], dtype=np.float64)
    X = np.array([r[2] for r in rows], dtype=np.float64)
    sprev = np.concatenate([[np.nan], s[:-1]])
    ds = s - sprev
    ok = np.isfinite(s) & np.isfinite(sprev) & np.all(np.isfinite(X), axis=1)
    dates = [d for d, k in zip(dates, ok) if k]
    s, sprev, ds, X = s[ok], sprev[ok], ds[ok], X[ok]
    n = len(s)
    print(f"meetings usable: {n} ({dates[0]} -> {dates[-1]}), OOS = {n - K0}")

    pred_ds, base_ds = [], []
    for t in range(K0, n):
        Xtr, ytr = X[:t], ds[:t]
        a = _pick_alpha(Xtr, ytr)
        m, sd_ = Xtr.mean(0), Xtr.std(0)
        sd_ = np.where(sd_ > 0, sd_, 1.0)
        w = _ols_fit((Xtr - m) / sd_, ytr, alpha=a)
        pred_ds.append(float(_pred(w, ((X[t:t + 1] - m) / sd_))[0]))
        base_ds.append(float(ytr.mean()))  # mean-drift baseline
    pred_ds, base_ds = np.asarray(pred_ds), np.asarray(base_ds)
    oos_dates = dates[K0:]
    actual = ds[K0:]

    nz = np.sign(actual) != 0
    hit = (np.sign(pred_ds) == np.sign(actual)) & nz
    n_dir = int(nz.sum())
    k_dir = int(hit.sum())
    acc = k_dir / n_dir
    pre16 = np.array([d < "2016-01-01" for d in oos_dates]) & nz
    acc_pre16 = float((hit & pre16).sum() / pre16.sum()) if pre16.sum() else float("nan")

    e_base = (actual - base_ds) ** 2
    e_m1 = (actual - pred_ds) ** 2
    res: dict[str, Any] = {
        "n_meetings": n, "date_range": [dates[0], dates[-1]], "n_oos": len(actual),
        "PRIMARY_dir_acc": acc, "PRIMARY_dir_n": n_dir, "PRIMARY_binom_p_onesided": _binom_p(k_dir, n_dir),
        "REPLICATION_pre2016_acc": acc_pre16, "REPLICATION_pre2016_n": int(pre16.sum()),
        "secondary_ds_mse_M1": float(e_m1.mean()), "secondary_ds_mse_drift": float(e_base.mean()),
        "secondary_ds_dm_p_onesided": _dm_p(e_base, e_m1),
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=2))
    print(f"PRIMARY dir-acc {acc:.3f} (n={n_dir}, binom p {res['PRIMARY_binom_p_onesided']:.4f})")
    print(f"REPLICATION pre-2016 acc {acc_pre16:.3f} (n={res['REPLICATION_pre2016_n']})")
    print(f"secondary Δs MSE M1 {e_m1.mean():.4f} vs drift {e_base.mean():.4f} (DM p {res['secondary_ds_dm_p_onesided']:.3f})")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
