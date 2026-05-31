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


def _market_block(cache_dir: Path | str, dates: "Any", log_rv: np.ndarray) -> np.ndarray:
    """Forward-looking cross-market features aligned to the RV dates.

    Columns: [log implied-variance (VIX), VIX 5-day log-change, variance-risk
    premium (log-IV − log-RV), 10y yield level, term slope 10y−3m]. These carry
    information HAR's backward realized lags structurally cannot.
    """

    import pandas as pd

    from app.data.dense_daily_dataset import load_market_cache

    series = load_market_cache(cache_dir, symbols=("VIX", "TNX", "IRX"))
    base = pd.DataFrame({"date": pd.Series(dates).astype(str)})
    for name in ("VIX", "TNX", "IRX"):
        s = series[name][["date", "close"]].rename(columns={"close": name})
        s["date"] = s["date"].astype(str)
        base = base.merge(s, on="date", how="left")
    # ffill only — propagate the last *known* close into market holidays/gaps.
    # No bfill: it would pull a future value backward into a leading gap (leak).
    # VIX/TNX/IRX all predate the RV window, so no leading NaN remains here.
    base = base.ffill()
    vix = base["VIX"].to_numpy(dtype=np.float64)
    log_iv = np.log((vix / 100.0) ** 2 / 252.0 + _EPS)  # daily implied variance, log scale
    vix_chg5 = np.log(vix / np.concatenate([vix[:5], vix[:-5]]) + _EPS)
    vrp = log_iv - log_rv  # implied minus realized (variance-risk premium proxy)
    tnx = base["TNX"].to_numpy(dtype=np.float64)
    slope = tnx - base["IRX"].to_numpy(dtype=np.float64)
    return np.column_stack([log_iv, vix_chg5, vrp, tnx, slope])


def run(
    rv_path: Path | str,
    *,
    seed: int = 11,
    n_folds: int = 5,
    horizons: tuple[int, ...] = (1, 5, 22),
    market_cache_dir: Path | str | None = None,
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
    # full realized-measure feature set for the DL contender
    feat_cols = ["rs_pos", "rs_neg", "bv", "rq", "rskew", "rkurt", "parkinson"]
    extra = np.column_stack([df[c].to_numpy(dtype=np.float64) for c in feat_cols])
    extra = np.column_stack([extra, np.log(df["rvol"].to_numpy(dtype=np.float64) + 1.0)])
    full = np.column_stack([har, extra])
    # optional cross-market block (VIX / rates) → HAR+IV and DL+market contenders
    market = _market_block(market_cache_dir, df["date"], log_rv) if market_cache_dir else None
    har_iv = np.column_stack([har, market]) if market is not None else None
    full_mkt = np.column_stack([full, market]) if market is not None else None

    contenders = ["har", "harplus", "dl"] + (["har_iv", "dl_mkt"] if market is not None else [])
    results: dict[str, Any] = {
        "n_days": len(df),
        "has_market": market is not None,
        "by_horizon": {},
    }
    for h in horizons:
        y = _forward_log_rv(rv, h)
        idx = np.where(~np.isnan(y))[0]
        folds = walk_forward_splits(len(idx), n_folds=n_folds, embargo=h + 1)
        pools: dict[str, list[float]] = {k: [] for k in [*contenders, "true", "base"]}
        for tr_l, te_l in folds:
            tr, te = idx[np.array(tr_l)], idx[np.array(te_l)]
            ytr, yte = y[tr], y[te]
            har_pred = _fit_predict_ols(har[tr], ytr, har[te])
            har_fit_tr = _fit_predict_ols(har[tr], ytr, har[tr])  # in-sample HAR for residual
            resid = (ytr - har_fit_tr).reshape(-1, 1)
            pools["har"].extend(har_pred.tolist())
            pools["harplus"].extend(_fit_predict_ols(harplus[tr], ytr, harplus[te]).tolist())
            dl_resid = _train_fold(full[tr], resid, full[te], seed=seed, epochs=300, device="cpu")
            pools["dl"].extend((har_pred + dl_resid[:, 0]).tolist())
            if market is not None:
                assert har_iv is not None and full_mkt is not None
                pools["har_iv"].extend(_fit_predict_ols(har_iv[tr], ytr, har_iv[te]).tolist())
                dl_m = _train_fold(
                    full_mkt[tr], resid, full_mkt[te], seed=seed, epochs=300, device="cpu"
                )
                pools["dl_mkt"].extend((har_pred + dl_m[:, 0]).tolist())
            pools["true"].extend(yte.tolist())
            pools["base"].extend([float(ytr.mean())] * len(te))
        p = {k: np.asarray(v) for k, v in pools.items()}
        row: dict[str, Any] = {k: _oos_r2(p[k], p["true"], p["base"]) for k in contenders}
        best_mkt = "dl_mkt" if market is not None else "dl"
        lo, hi = _bootstrap_r2_ci(p[best_mkt], p["true"], p["base"], seed=seed)
        row["dl_ci90"] = [lo, hi]
        if "har_iv" in row:
            # Incremental skill of IV *over HAR*: HAR predictions are the baseline
            # (denominator), so this CI is the marginal-R² of adding IV, distinct
            # from the vs-mean R² in the table. <0 ⇒ IV does not help beyond HAR.
            lo2, hi2 = _bootstrap_r2_ci(p["har_iv"], p["true"], p["har"], seed=seed)
            row["har_iv_vs_har_ci90"] = [lo2, hi2]
        results["by_horizon"][f"h{h}"] = row
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Intraday RV forecast bake-off: HAR vs HAR+ vs DL (+market)."
    )
    parser.add_argument("--rv-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--market-cache-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    res = run(args.rv_path, seed=args.seed, market_cache_dir=args.market_cache_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "intraday_rv_bakeoff.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8"
    )
    print(f"n_days={res['n_days']}  market={res['has_market']}")
    mkt = res["has_market"]
    head = f"{'horizon':<8}{'HAR':>8}{'HAR+':>8}{'DL':>8}"
    head += f"{'HAR+IV':>8}{'DL+mkt':>8}{'IVvsHAR_CI90':>20}" if mkt else f"{'best_CI90':>18}"
    print(head)
    for hk, r in res["by_horizon"].items():
        line = f"{hk:<8}{r['har']:>8.3f}{r['harplus']:>8.3f}{r['dl']:>8.3f}"
        if mkt:
            c = r["har_iv_vs_har_ci90"]
            line += f"{r['har_iv']:>8.3f}{r['dl_mkt']:>8.3f}{f'[{c[0]:+.3f},{c[1]:+.3f}]':>20}"
        else:
            c = r["dl_ci90"]
            line += f"{f'[{c[0]:.3f},{c[1]:.3f}]':>18}"
        print(line)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
