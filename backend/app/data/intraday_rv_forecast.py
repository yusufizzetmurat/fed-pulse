"""Forecast intraday realized volatility — HAR vs HARQ/HAR⁺ vs DL bake-off.

On the daily realized-measure series (`intraday_realized`), predict log of
average realized variance over the next h ∈ {1,5,22} days, walk-forward.
Three contenders, each evaluated out-of-sample with a bootstrap CI:

  - HAR   : OLS of target on log-RV daily / weekly / monthly lags (Corsi).
  - HAR⁺  : HAR + realized-quarticity and downside-semivariance terms
            (HARQ / SHAR family — the strong intraday baselines).
  - DL    : MLP on the full realized-measure feature set, residual-stacked
            on HAR (floors at HAR, learns the remainder).

This is where beating HAR is both achievable and meaningful — intraday RV
carries the jump / semivariance / quarticity structure daily-close lacks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from app.data.dense_daily_dataset import walk_forward_splits
from app.data.dense_forecast_train import _bootstrap_r2_ci, _fit_predict_ols, _oos_r2, _train_fold

_EPS = 1e-12


def _har_lags(log_rv: np.ndarray) -> np.ndarray:
    """[logRV_t, mean last-5, mean last-22] — backward HAR regressors."""

    n = len(log_rv)
    daily = log_rv.copy()
    weekly = np.array([log_rv[max(0, i - 4) : i + 1].mean() for i in range(n)])
    monthly = np.array([log_rv[max(0, i - 21) : i + 1].mean() for i in range(n)])
    return np.column_stack([daily, weekly, monthly])


def _forward_log_rv(rv: np.ndarray, h: int) -> np.ndarray:
    """log of mean RV over t+1..t+h (NaN where the window runs off the end)."""

    n = len(rv)
    out = np.full(n, np.nan)
    for t in range(n - h):
        out[t] = np.log(rv[t + 1 : t + 1 + h].mean() + _EPS)
    return out


def run(
    rv_path: Path | str,
    *,
    seed: int = 11,
    n_folds: int = 5,
    horizons: tuple[int, ...] = (1, 5, 22),
) -> dict[str, Any]:
    import pandas as pd

    df = pd.read_parquet(rv_path).sort_values("date").reset_index(drop=True)
    rv = df["rv"].to_numpy(dtype=np.float64)
    log_rv = np.log(rv + _EPS)
    har = _har_lags(log_rv)
    # HAR⁺ extra columns: quarticity noise proxy + downside-semivariance share
    sqrt_rq = np.sqrt(df["rq"].to_numpy(dtype=np.float64) + _EPS)
    rq_term = (sqrt_rq / (rv + _EPS)) * har[:, 0]  # BPQ-style measurement-error correction on daily
    rs_neg_share = df["rs_neg"].to_numpy(dtype=np.float64) / (rv + _EPS)
    harplus = np.column_stack([har, rq_term, rs_neg_share])
    # full feature set for the DL contender
    feat_cols = ["rs_pos", "rs_neg", "bv", "rq", "rskew", "rkurt", "parkinson"]
    extra = np.column_stack([df[c].to_numpy(dtype=np.float64) for c in feat_cols])
    extra = np.column_stack([extra, np.log(df["rvol"].to_numpy(dtype=np.float64) + 1.0)])
    full = np.column_stack([har, extra])

    results: dict[str, Any] = {"n_days": len(df), "by_horizon": {}}
    for h in horizons:
        y = _forward_log_rv(rv, h)
        valid = ~np.isnan(y)
        idx = np.where(valid)[0]
        folds = walk_forward_splits(len(idx), n_folds=n_folds, embargo=h + 1)
        pools: dict[str, list[float]] = {k: [] for k in ("har", "harplus", "dl", "true", "base")}
        for tr_l, te_l in folds:
            tr, te = idx[np.array(tr_l)], idx[np.array(te_l)]
            ytr, yte = y[tr], y[te]
            fold_mean = float(ytr.mean())
            har_pred = _fit_predict_ols(har[tr], ytr, har[te])
            hp_pred = _fit_predict_ols(harplus[tr], ytr, harplus[te])
            # DL: residual-stack on HAR (learn what HAR misses, from the full set)
            har_tr_in = _fit_predict_ols(har[tr], ytr, har[tr])
            resid = (ytr - har_tr_in).reshape(-1, 1)
            dl_resid = _train_fold(full[tr], resid, full[te], seed=seed, epochs=300, device="cpu")
            dl_pred = har_pred + dl_resid[:, 0]
            for k, v in (("har", har_pred), ("harplus", hp_pred), ("dl", dl_pred)):
                pools[k].extend(np.asarray(v).tolist())
            pools["true"].extend(yte.tolist())
            pools["base"].extend([fold_mean] * len(te))
        p = {k: np.asarray(v) for k, v in pools.items()}
        row: dict[str, Any] = {}
        for k in ("har", "harplus", "dl"):
            row[k] = _oos_r2(p[k], p["true"], p["base"])
        lo, hi = _bootstrap_r2_ci(p["dl"], p["true"], p["base"], seed=seed)
        row["dl_ci90"] = [lo, hi]
        row["dl_minus_har"] = row["dl"] - row["har"]
        results["by_horizon"][f"h{h}"] = row
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Intraday RV forecast bake-off: HAR vs HAR+ vs DL."
    )
    parser.add_argument("--rv-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    res = run(args.rv_path, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "intraday_rv_bakeoff.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8"
    )
    print(f"n_days={res['n_days']}")
    print(f"{'horizon':<8}{'HAR':>8}{'HAR+':>8}{'DL':>8}{'DL-HAR':>9}{'DL_CI90':>18}")
    for hk, r in res["by_horizon"].items():
        ci = f"[{r['dl_ci90'][0]:.3f},{r['dl_ci90'][1]:.3f}]"
        print(
            f"{hk:<8}{r['har']:>8.3f}{r['harplus']:>8.3f}{r['dl']:>8.3f}{r['dl_minus_har']:>+9.3f}{ci:>18}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
