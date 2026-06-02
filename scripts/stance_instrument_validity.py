"""Validate the stance instrument against the Fed's actual policy action.

Pre-registration: docs/research/stance-instrument-validity-preregistration.md

Anchor = realised funds-rate move (FRED DFEDTARU step) per meeting, which the
classifier never trained on. Tests whether s = P(hawk)-P(dove) tracks Delta_ff
(concurrent + leading). numpy-only (Spearman = Pearson of ranks).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

FRED = "data/external/fred"
STANCE = "data/artifacts/corner_b_text_rates/stance_daily.parquet"
OUT = "data/artifacts/stance_instrument_validity/result.json"


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    # average ties
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, r)
    r = (sums / cnt)[inv]
    return r


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def _perm_p(a: np.ndarray, b: np.ndarray, rho: float, *, n: int = 10000, seed: int = 11) -> float:
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n):
        if _spearman(rng.permutation(a), b) >= rho:
            ge += 1
    return (ge + 1) / (n + 1)


def _boot_ci(a: np.ndarray, b: np.ndarray, *, n: int = 5000, seed: int = 11) -> list[float]:
    rng = np.random.default_rng(seed)
    vals = []
    m = len(a)
    for _ in range(n):
        idx = rng.integers(0, m, size=m)
        vals.append(_spearman(a[idx], b[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(s_pos > s_neg) via Mann-Whitney U / (n1 n2), tie=0.5."""
    c = 0.0
    for p in pos:
        c += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(c / (len(pos) * len(neg)))


def _auc_ci(pos: np.ndarray, neg: np.ndarray, *, n: int = 5000, seed: int = 11) -> list[float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        p = pos[rng.integers(0, len(pos), len(pos))]
        q = neg[rng.integers(0, len(neg), len(neg))]
        vals.append(_auc(p, q))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _dff_series() -> tuple[np.ndarray, np.ndarray]:
    j = json.load(open(f"{FRED}/DFEDTARU.json"))
    d, v = [], []
    for o in j["observations"]:
        if o["value"] not in (".", "", None):
            d.append(np.datetime64(o["date"]))
            v.append(float(o["value"]))
    return np.array(d), np.array(v, dtype=np.float64)


def main() -> None:
    import pandas as pd

    fd, fv = _dff_series()
    mp = pd.read_parquet(f"{FRED}/mp_surprises.parquet")
    mp["event_date"] = pd.to_datetime(mp["event_date"])
    mp = mp[(mp["event_date"] >= "2010-01-01") & (~mp["is_intermeeting"].astype(bool))]
    mp = mp.sort_values("event_date").reset_index(drop=True)

    sd = pd.read_parquet(STANCE)
    sd["date"] = pd.to_datetime(sd["date"]).dt.strftime("%Y-%m-%d")
    s_by = dict(zip(sd["date"], sd["s"]))

    s_list, dff_list, dates = [], [], []
    for m in mp["event_date"]:
        md = np.datetime64(pd.Timestamp(m))
        before = fv[fd <= md - np.timedelta64(1, "D")]
        after = fv[fd >= md + np.timedelta64(2, "D")]
        if not (len(before) and len(after)):
            continue
        dstr = pd.Timestamp(m).strftime("%Y-%m-%d")
        if dstr not in s_by or not np.isfinite(s_by[dstr]):
            continue
        s_list.append(float(s_by[dstr]))
        dff_list.append((after[0] - before[-1]) * 100.0)
        dates.append(dstr)

    s = np.array(s_list)
    dff = np.array(dff_list)
    cat = np.where(dff > 0, "hike", np.where(dff < 0, "cut", "hold"))
    n = len(s)

    # PRIMARY
    rho = _spearman(s, dff)
    p_perm = _perm_p(s, dff, rho)
    ci = _boot_ci(s, dff)

    # SECONDARY (a) hike vs cut AUC
    s_hike, s_cut, s_hold = s[cat == "hike"], s[cat == "cut"], s[cat == "hold"]
    auc = _auc(s_hike, s_cut)
    auc_ci = _auc_ci(s_hike, s_cut)
    # (b) ordinal trend
    ordmap = {"cut": 0, "hold": 1, "hike": 2}
    ordv = np.array([ordmap[c] for c in cat], dtype=np.float64)
    rho_ord = _spearman(s, ordv)
    # (c) leading: s_t vs dff_{t+1}
    rho_lead = _spearman(s[:-1], dff[1:])
    p_lead = _perm_p(s[:-1], dff[1:], rho_lead)

    # DIAGNOSTIC: within holds, s_t vs next move
    hold_mask = cat == "hold"
    s_next = np.concatenate([s[1:], [np.nan]])
    dff_next = np.concatenate([dff[1:], [np.nan]])
    hm = hold_mask & np.isfinite(dff_next)
    rho_hold_lead = _spearman(s[hm], dff_next[hm]) if hm.sum() > 5 else float("nan")

    res: dict[str, Any] = {
        "n_meetings": n, "date_range": [dates[0], dates[-1]],
        "action_counts": {k: int((cat == k).sum()) for k in ("cut", "hold", "hike")},
        "PRIMARY_spearman_s_vs_dff": rho, "PRIMARY_perm_p_onesided": p_perm, "PRIMARY_boot_ci95": ci,
        "mean_s_by_action": {k: float(s[cat == k].mean()) for k in ("cut", "hold", "hike")},
        "secondary_auc_hike_vs_cut": auc, "secondary_auc_ci95": auc_ci,
        "secondary_ordinal_spearman": rho_ord,
        "secondary_leading_spearman_st_vs_dff_next": rho_lead, "secondary_leading_perm_p": p_lead,
        "diagnostic_within_holds_lead_spearman": rho_hold_lead,
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=2))
    print(f"n={n} {res['action_counts']}")
    print(f"PRIMARY Spearman(s, dff) = {rho:+.3f}  perm-p {p_perm:.4f}  CI95 {ci}")
    print(f"mean s by action: {res['mean_s_by_action']}")
    print(f"AUC hike-vs-cut {auc:.3f} CI {auc_ci} | ordinal rho {rho_ord:+.3f}")
    print(f"leading s_t vs dff_t+1: rho {rho_lead:+.3f} (p {p_lead:.3f}) | within-holds lead rho {rho_hold_lead:+.3f}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
