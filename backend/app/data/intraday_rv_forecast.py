"""Forecast intraday realized volatility — HAR vs HARQ/HAR⁺ vs DL bake-off.

On the daily realized-measure series (`intraday_realized`), predict log of
average realized variance over the next h ∈ {1,5,22} days, walk-forward.
Three contenders, each evaluated out-of-sample with a bootstrap CI:

  - HAR   : OLS of target on log-RV daily / weekly / monthly lags (Corsi).
  - HAR⁺  : HAR + realized-quarticity and downside-semivariance terms
            (HARQ / SHAR family — the strong intraday baselines).
  - DL    : MLP on the full realized-measure feature set, residual-stacked
            on HAR (floors at HAR, learns the remainder), Huber-trained.
  - DLq   : same residual-stacked MLP, but trained to minimize QLIKE on the
            reconstructed variance exp(HAR + residual) — the econometric vol
            loss — rather than Huber on the log-residual.

Each contender is scored on OOS-R² (log-RV space) and QLIKE (variance space),
so the beat-HAR verdict is read under the correct vol metric, not just MSE.

This is where beating HAR is both achievable and meaningful — intraday RV
carries the jump / semivariance / quarticity structure daily-close lacks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from app.data.dense_daily_dataset import walk_forward_splits
from app.data.dense_forecast_train import _bootstrap_r2_ci, _fit_predict_ols, _oos_r2, _train_fold

_EPS = 1e-12
# Clamp on log-variance before exp() — keeps reconstructed σ² finite for the
# QLIKE loss (exp(80) ≈ 5e34, well inside float32) regardless of a stray
# residual blow-up during early training.
_LOGV_CLAMP = 80.0


def _qlike(pred_logrv: np.ndarray, true_logrv: np.ndarray) -> float:
    """QLIKE volatility loss on variance σ² = exp(log-RV), lower is better.

    QLIKE = mean( σ²_true/σ²_pred − log(σ²_true/σ²_pred) − 1 ). Zero at a
    perfect forecast and — unlike MSE on log-RV — asymmetric: it penalizes
    under-prediction of a variance spike far harder than over-prediction,
    which is the econometric-standard loss for ranking vol models.
    """

    var_pred = np.exp(np.clip(pred_logrv, -_LOGV_CLAMP, _LOGV_CLAMP)) + _EPS
    var_true = np.exp(np.clip(true_logrv, -_LOGV_CLAMP, _LOGV_CLAMP))
    ratio = var_true / var_pred
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def _qlike_pointwise(pred_logrv: np.ndarray, true_logrv: np.ndarray) -> np.ndarray:
    """Per-point QLIKE contributions (for bootstrapping the QLIKE difference)."""

    var_pred = np.exp(np.clip(pred_logrv, -_LOGV_CLAMP, _LOGV_CLAMP)) + _EPS
    var_true = np.exp(np.clip(true_logrv, -_LOGV_CLAMP, _LOGV_CLAMP))
    ratio = var_true / var_pred
    return cast(np.ndarray, ratio - np.log(ratio) - 1.0)


def _bootstrap_qlike_gain_ci(
    pred: np.ndarray,
    base_pred: np.ndarray,
    true: np.ndarray,
    *,
    seed: int = 11,
    n_boot: int = 1000,
) -> list[float]:
    """90% CI of mean QLIKE gain (base − pred); >0 throughout ⇒ pred beats base."""

    gain = _qlike_pointwise(base_pred, true) - _qlike_pointwise(pred, true)
    rng = np.random.default_rng(seed)
    n = len(gain)
    boots = [float(gain[rng.integers(0, n, n)].mean()) for _ in range(n_boot)]
    return [float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))]


def _train_fold_qlike(
    har_pred_tr: np.ndarray,
    full_tr: np.ndarray,
    resid_tr: np.ndarray,
    full_te: np.ndarray,
    har_pred_te: np.ndarray,
    *,
    seed: int,
    epochs: int,
    device: str,
) -> np.ndarray:
    """Residual-stacked DL trained to minimize QLIKE on reconstructed variance.

    Same architecture / standardization / early-stopping discipline as
    `_train_fold`, but the loss is QLIKE on σ² = exp(HAR_pred + DL_residual)
    vs true σ², not Huber on the standardized log-
    residual. Inputs are in log-RV space; HAR_pred floors the forecast and the
    net learns the remainder. Returns test predictions in log-RV space
    (HAR_pred + DL_residual), so the caller scores it identically to the others.
    """

    import torch

    from app.data.dense_forecast_train import _build_model
    from app.determinism import enable_deterministic_mode

    enable_deterministic_mode(seed)
    dev = torch.device(device)
    xm, xs = full_tr.mean(0), full_tr.std(0)
    xs = np.where(xs > 0, xs, 1.0)
    # standardize the residual target only to keep the head's raw output O(1);
    # the loss un-standardizes before reconstructing variance, so QLIKE is
    # always computed on the true σ² scale.
    rm, rs = resid_tr.mean(), resid_tr.std()
    rs = rs if rs > 0 else 1.0
    xtr = torch.tensor((full_tr - xm) / xs, dtype=torch.float32, device=dev)
    xte = torch.tensor((full_te - xm) / xs, dtype=torch.float32, device=dev)
    har_tr = torch.tensor(har_pred_tr, dtype=torch.float32, device=dev).reshape(-1, 1)
    log_true_tr = torch.tensor(
        (resid_tr.reshape(-1) + har_pred_tr).reshape(-1, 1), dtype=torch.float32, device=dev
    )

    def qlike_loss(resid_std: torch.Tensor, har: torch.Tensor, log_true: torch.Tensor) -> Any:
        log_pred = har + (resid_std * rs + rm)
        log_pred = torch.clamp(log_pred, -_LOGV_CLAMP, _LOGV_CLAMP)
        log_true_c = torch.clamp(log_true, -_LOGV_CLAMP, _LOGV_CLAMP)
        var_pred = torch.exp(log_pred) + _EPS
        var_true = torch.exp(log_true_c)
        ratio = var_true / var_pred
        return torch.mean(ratio - torch.log(ratio) - 1.0)

    n_val = max(1, len(xtr) // 5)
    tr, val = slice(0, len(xtr) - n_val), slice(len(xtr) - n_val, len(xtr))
    model = _build_model(full_tr.shape[1], 1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = qlike_loss(model(xtr[tr]), har_tr[tr], log_true_tr[tr])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # exp-loss → bound grads
        opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(qlike_loss(model(xtr[val]), har_tr[val], log_true_tr[val]))
        if vloss < best - 1e-6:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 40:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        resid_std = model(xte).cpu().numpy()
    resid_pred = resid_std[:, 0] * rs + rm
    return cast(np.ndarray, har_pred_te + resid_pred)


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

    contenders = ["har", "harplus", "dl", "dl_qlike"] + (
        ["har_iv", "dl_mkt"] if market is not None else []
    )
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
            dl_q = _train_fold_qlike(
                har_fit_tr, full[tr], resid, full[te], har_pred,
                seed=seed, epochs=300, device="cpu",
            )
            pools["dl_qlike"].extend(dl_q.tolist())
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
        # QLIKE alongside R² — the econometric vol metric for the beat-HAR verdict.
        row["qlike"] = {k: _qlike(p[k], p["true"]) for k in contenders}
        best_mkt = "dl_mkt" if market is not None else "dl"
        lo, hi = _bootstrap_r2_ci(p[best_mkt], p["true"], p["base"], seed=seed)
        row["dl_ci90"] = [lo, hi]
        # Does the QLIKE-trained DL beat HAR on QLIKE? Bootstrap the per-point
        # QLIKE difference (HAR − dl_qlike); >0 throughout ⇒ DL wins on QLIKE.
        row["dl_qlike_vs_har_qlike_ci90"] = _bootstrap_qlike_gain_ci(
            p["dl_qlike"], p["har"], p["true"], seed=seed
        )
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
    # OOS-R² table (vs-mean R², higher is better).
    head = f"{'horizon':<8}{'HAR':>8}{'HAR+':>8}{'DL':>8}{'DLq':>8}"
    head += f"{'HAR+IV':>8}{'DL+mkt':>8}{'IVvsHAR_CI90':>20}" if mkt else f"{'best_CI90':>18}"
    print("OOS-R2 (vs mean, higher better)")
    print(head)
    for hk, r in res["by_horizon"].items():
        line = f"{hk:<8}{r['har']:>8.3f}{r['harplus']:>8.3f}{r['dl']:>8.3f}{r['dl_qlike']:>8.3f}"
        if mkt:
            c = r["har_iv_vs_har_ci90"]
            line += f"{r['har_iv']:>8.3f}{r['dl_mkt']:>8.3f}{f'[{c[0]:+.3f},{c[1]:+.3f}]':>20}"
        else:
            c = r["dl_ci90"]
            line += f"{f'[{c[0]:.3f},{c[1]:.3f}]':>18}"
        print(line)
    # QLIKE table (econometric vol loss, lower better) + DLq-vs-HAR QLIKE gain CI.
    print("QLIKE (lower better)  +  DLq-vs-HAR QLIKE-gain CI90 (>0 ⇒ DLq beats HAR)")
    qhead = f"{'horizon':<8}{'HAR':>10}{'HAR+':>10}{'DL':>10}{'DLq':>10}{'DLqGain_CI90':>22}"
    print(qhead)
    for hk, r in res["by_horizon"].items():
        q = r["qlike"]
        g = r["dl_qlike_vs_har_qlike_ci90"]
        print(
            f"{hk:<8}{q['har']:>10.4f}{q['harplus']:>10.4f}{q['dl']:>10.4f}{q['dl_qlike']:>10.4f}"
            f"{f'[{g[0]:+.4f},{g[1]:+.4f}]':>22}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
