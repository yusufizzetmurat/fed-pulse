"""Forecasting matrix — every neural core on the same RV-regression / QLIKE task.

Runs HAR, GARCH(1,1) and the neural cores (MLP, LSTM, GRU, TCN, σLSTM,
Transformer, DLinear, Informer) on one realized-measure series, walk-forward,
across horizons {1,5,22} and two targets:

  - realized volatility (QLIKE-trained, variance-space QLIKE + OOS-R²)
  - abnormal volume    (MSE-trained on log-volume residual, OOS-R² only;
                         QLIKE is a variance loss and is not reported here)

Every model is scored on the identical set of (origin, true) test pairs per
fold, residual-stacked on HAR, with the official seed set {11,29,47,71,97}.
Neural rows report mean ± 90% CI over the five seeds; HAR and GARCH are
deterministic single values. Output: a JSON artefact and a printed table.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.data.dense_daily_dataset import walk_forward_splits
from app.data.dense_forecast_train import _fit_predict_ols, _oos_r2
from app.data.intraday_rv_arch import (
    _SEQ_LEN,
    _train_arch_fold,
    build_sequences,
)
from app.data.intraday_rv_forecast import _EPS, _forward_log_rv, _har_lags, _qlike

OFFICIAL_SEEDS = (11, 29, 47, 71, 97)
HORIZONS = (1, 5, 22)
NEURAL_ARCHS = ("mlp", "lstm", "gru", "tcn", "sigma_lstm", "transformer", "dlinear", "informer")
_T_95_DF4 = 2.131847  # two-sided 90% Student-t quantile, df = n_seeds - 1 = 4


def _ci90(values: list[float]) -> dict[str, float]:
    """Mean ± 90% Student-t CI over seeds (degenerate to the point if n<2)."""
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if len(arr) < 2:
        return {"mean": mean, "std": 0.0, "ci90_lo": mean, "ci90_hi": mean}
    std = float(arr.std(ddof=1))
    half = _T_95_DF4 * std / math.sqrt(len(arr)) if len(arr) == 5 else (
        1.645 * std / math.sqrt(len(arr))
    )
    return {
        "mean": mean,
        "std": std,
        "ci90_lo": mean - half,
        "ci90_hi": mean + half,
        # median + IQR — robust to a single diverged seed under the exp-QLIKE loss
        "median": float(np.median(arr)),
        "iqr": [float(np.quantile(arr, 0.25)), float(np.quantile(arr, 0.75))],
    }


def _series_matrix(rv_path: Path | str, target: str) -> dict[str, np.ndarray]:
    """Feature matrix + HAR floor + raw series for the chosen target.

    target='rv'     → series is realized variance, HAR on log-RV.
    target='volume' → series is daily volume, HAR on log-volume.
    Exogenous channels (rs_pos, rs_neg, bv, rq, rskew, rkurt, parkinson, log-vol)
    are identical for both, so the architecture comparison is feature-for-feature.
    """
    df = pd.read_parquet(rv_path).sort_values("date").reset_index(drop=True)
    series = (df["rv"] if target == "rv" else df["rvol"]).to_numpy(dtype=np.float64)
    log_s = np.log(series + _EPS)
    har = _har_lags(log_s)
    feat_cols = ["rs_pos", "rs_neg", "bv", "rq", "rskew", "rkurt", "parkinson"]
    extra = np.column_stack([df[c].to_numpy(dtype=np.float64) for c in feat_cols])
    extra = np.column_stack([extra, np.log(df["rvol"].to_numpy(dtype=np.float64) + 1.0)])
    full = np.column_stack([har, extra])
    return {
        "series": series,
        "har": har,
        "full": full,
        "dates": df["date"].astype(str).to_numpy(),
    }


def _daily_log_returns(close_path: Path | str, dates: np.ndarray) -> np.ndarray:
    """Percent daily log returns aligned to the RV dates (NaN before first obs)."""
    px = pd.read_parquet(close_path)[["date", "close"]].copy()
    px["date"] = px["date"].astype(str)
    base = pd.DataFrame({"date": pd.Series(dates).astype(str)})
    merged = base.merge(px, on="date", how="left").sort_values("date").reset_index(drop=True)
    close = merged["close"].to_numpy(dtype=np.float64)
    ret = np.full(len(close), np.nan)
    ret[1:] = 100.0 * np.log(close[1:] / close[:-1])
    return ret


def _neural_pools(
    arch: str,
    *,
    seq_all: np.ndarray,
    har: np.ndarray,
    origin: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    folds: list[tuple[list[int], list[int]]],
    seed: int,
    epochs: int,
    device: str,
    loss: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk-forward predictions for one arch/seed; returns (pred, true, base)."""
    pred_pool: list[float] = []
    true_pool: list[float] = []
    base_pool: list[float] = []
    for tr_l, te_l in folds:
        tr, te = idx[np.array(tr_l)], idx[np.array(te_l)]
        o_tr, o_te = origin[tr], origin[te]
        ytr, yte = y[o_tr], y[o_te]
        pred = _train_arch_fold(
            arch, seq_all[tr], seq_all[te], har[o_tr], har[o_te], ytr,
            seed=seed, epochs=epochs, device=device, loss=loss,
        )
        pred_pool.extend(pred.tolist())
        true_pool.extend(yte.tolist())
        base_pool.extend([float(ytr.mean())] * len(te))
    return np.asarray(pred_pool), np.asarray(true_pool), np.asarray(base_pool)


def _har_pools(
    *, har: np.ndarray, origin: np.ndarray, y: np.ndarray,
    idx: np.ndarray, folds: list[tuple[list[int], list[int]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_pool, true_pool, base_pool = [], [], []
    for tr_l, te_l in folds:
        tr, te = idx[np.array(tr_l)], idx[np.array(te_l)]
        o_tr, o_te = origin[tr], origin[te]
        ytr, yte = y[o_tr], y[o_te]
        pred = _fit_predict_ols(har[o_tr], ytr, har[o_te])
        pred_pool.extend(pred.tolist())
        true_pool.extend(yte.tolist())
        base_pool.extend([float(ytr.mean())] * len(te))
    return np.asarray(pred_pool), np.asarray(true_pool), np.asarray(base_pool)


def _garch_pools(
    *, ret: np.ndarray, origin: np.ndarray, y: np.ndarray, h: int,
    idx: np.ndarray, folds: list[tuple[list[int], list[int]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GARCH(1,1) on daily % log-returns: forecast mean variance over t+1..t+h.

    Fit constant-mean GARCH(1,1) on each fold's contiguous train returns, then
    roll the conditional-variance recursion forward over the test block with the
    train-fitted params (no refit, no leak). The h-step forecast uses the closed
    form E[σ²_{t+k}] = σ²_∞ + (α+β)^{k-1}·(σ²_{t+1} − σ²_∞), averaged over
    k=1..h. The percent-variance forecast is converted to decimal-variance
    (÷1e4) and compared in the same log-variance space as HAR/RV. GARCH on daily
    returns is a structurally weaker vol proxy than 5-min RV, so any level bias
    it carries is a fair part of the result.
    """
    from arch import arch_model

    pred_pool, true_pool, base_pool = [], [], []
    for tr_l, te_l in folds:
        tr, te = idx[np.array(tr_l)], idx[np.array(te_l)]
        o_tr, o_te = origin[tr], origin[te]
        ytr, yte = y[o_tr], y[o_te]
        # contiguous train returns by ROW index (origins are monotone)
        r_tr = ret[o_tr.min() : o_tr.max() + 1]
        r_tr = r_tr[~np.isnan(r_tr)]
        res = arch_model(r_tr, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(
            disp="off"
        )
        pr = res.params
        mu = float(pr.get("mu", 0.0))
        omega, alpha, beta = float(pr["omega"]), float(pr["alpha[1]"]), float(pr["beta[1]"])
        persist = alpha + beta
        var_uncond = omega / (1.0 - persist) if persist < 1.0 else float(np.var(r_tr))

        # Roll σ²_t over the full return path up to the last test origin (fixed
        # params, realized returns only — strictly leak-safe per origin).
        last = int(o_te.max())
        sigma2 = np.full(last + 2, var_uncond)  # sigma2[t] = Var(r_t | info up to t-1)
        for t in range(1, last + 1):
            eps_prev = (ret[t - 1] - mu) if not np.isnan(ret[t - 1]) else 0.0
            sigma2[t] = omega + alpha * eps_prev**2 + beta * sigma2[t - 1]

        preds = []
        for o in o_te.tolist():
            eps_o = (ret[o] - mu) if not np.isnan(ret[o]) else 0.0
            s2_next = omega + alpha * eps_o**2 + beta * sigma2[o]  # σ²_{o+1}
            fk = [var_uncond + (persist ** (k - 1)) * (s2_next - var_uncond) for k in range(1, h + 1)]
            mean_var_pct = float(np.mean(fk))
            preds.append(math.log(max(mean_var_pct, _EPS) / 1e4 + _EPS))
        pred_pool.extend(preds)
        true_pool.extend(yte.tolist())
        base_pool.extend([float(ytr.mean())] * len(te))
    return np.asarray(pred_pool), np.asarray(true_pool), np.asarray(base_pool)


def run(
    rv_path: Path | str,
    close_path: Path | str,
    *,
    seeds: tuple[int, ...] = OFFICIAL_SEEDS,
    horizons: tuple[int, ...] = HORIZONS,
    archs: tuple[str, ...] = NEURAL_ARCHS,
    epochs: int = 120,
    n_folds: int = 5,
    device: str = "cuda",
    with_garch: bool = True,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "seeds": list(seeds),
        "horizons": list(horizons),
        "n_folds": n_folds,
        "epochs": epochs,
        "seq_len": _SEQ_LEN,
        "targets": {},
    }
    for target in ("rv", "volume"):
        loss = "qlike" if target == "rv" else "mse"
        data = _series_matrix(rv_path, target)
        series, har, full = data["series"], data["har"], data["full"]
        ret = _daily_log_returns(close_path, data["dates"]) if target == "rv" else None
        seqs = build_sequences(full, np.ones(len(series), dtype=bool))
        origin, seq_all = seqs["origin"], seqs["seq"]
        out["targets"][target] = {"n_days": int(len(series)), "loss": loss, "by_horizon": {}}
        for h in horizons:
            y = _forward_log_rv(series, h)
            ok = ~np.isnan(y[origin])
            idx = np.where(ok)[0]
            folds = walk_forward_splits(len(idx), n_folds=n_folds, embargo=h + 1)
            row: dict[str, Any] = {"models": {}}

            hp, ht, hb = _har_pools(har=har, origin=origin, y=y, idx=idx, folds=folds)
            row["models"]["har"] = {
                "qlike": _qlike(hp, ht) if target == "rv" else None,
                "r2": _oos_r2(hp, ht, hb),
                "deterministic": True,
                "n_eval": int(len(ht)),
            }

            if with_garch and target == "rv" and ret is not None:
                try:
                    gp, gt, gb = _garch_pools(
                        ret=ret, origin=origin, y=y, h=h, idx=idx, folds=folds
                    )
                    row["models"]["garch"] = {
                        "qlike": _qlike(gp, gt),
                        "r2": _oos_r2(gp, gt, gb),
                        "deterministic": True,
                        "n_eval": int(len(gt)),
                    }
                except Exception as exc:  # noqa: BLE001 — record, don't abort the matrix
                    row["models"]["garch"] = {"failed": f"{type(exc).__name__}: {exc}"}

            for arch in archs:
                q_seeds, r_seeds = [], []
                for seed in seeds:
                    pp, pt, pb = _neural_pools(
                        arch, seq_all=seq_all, har=har, origin=origin, y=y, idx=idx,
                        folds=folds, seed=seed, epochs=epochs, device=device, loss=loss,
                    )
                    if target == "rv":
                        q_seeds.append(_qlike(pp, pt))
                    r_seeds.append(_oos_r2(pp, pt, pb))
                row["models"][arch] = {
                    "qlike": _ci90(q_seeds) if target == "rv" else None,
                    "r2": _ci90(r_seeds),
                    "deterministic": False,
                    "seed_qlike": q_seeds if target == "rv" else None,
                    "seed_r2": r_seeds,
                }
            out["targets"][target]["by_horizon"][f"h{h}"] = row
            print(f"[{target} h{h}] done — {len(idx)} scorable origins")
    return out


def _fmt(model: dict[str, Any], key: str) -> str:
    v = model.get(key)
    if v is None:
        return f"{'—':>10}"
    if isinstance(v, dict):
        return f"{v['mean']:>10.4f}"
    return f"{v:>10.4f}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rv-path", type=Path, default=Path("/data/external/alphavantage_bars/spx_5min_daily_rv.parquet"))
    p.add_argument(
        "--close-path",
        type=Path,
        # 1975→2026 daily close (covers the full 2005+ RV range; /data/raw/market
        # only starts 2010, which starves the early GARCH folds).
        default=Path("/data/processed/canonical_60bar/_market_cache/GSPC.parquet"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("/data/artifacts/forecasting_matrix"))
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seeds", type=int, nargs="+", default=list(OFFICIAL_SEEDS))
    p.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    p.add_argument("--archs", nargs="+", default=list(NEURAL_ARCHS))
    p.add_argument("--no-garch", action="store_true")
    args = p.parse_args()

    res = run(
        args.rv_path, args.close_path,
        seeds=tuple(args.seeds), horizons=tuple(args.horizons), archs=tuple(args.archs),
        epochs=args.epochs, device=args.device, with_garch=not args.no_garch,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "forecasting_matrix.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    for target, tdata in res["targets"].items():
        print(f"\n===== TARGET: {target} ({tdata['loss']}-trained) =====")
        for hk, row in tdata["by_horizon"].items():
            print(f"\n  {hk}:")
            print(f"    {'model':<14}{'QLIKE':>10}{'OOS-R2':>10}")
            for name, m in row["models"].items():
                if "failed" in m:
                    print(f"    {name:<14}  FAILED: {m['failed']}")
                    continue
                print(f"    {name:<14}{_fmt(m,'qlike')}{_fmt(m,'r2')}")
    print(f"\nwrote {args.out_dir / 'forecasting_matrix.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
