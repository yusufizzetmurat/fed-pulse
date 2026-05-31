"""Deployable banded RV forecaster — QLIKE-DLq ensemble + walk-forward conformal.

Productionizes the validated beat-HAR result from `intraday_rv_forecast`: the
residual-stacked DL trained on QLIKE (`_train_fold_qlike`) that beats HAR on the
econometric vol loss (block-bootstrap confirmed). Here we make it deployable:

  - Multi-seed ensemble — the QLIKE-DLq is trained across `N_SEEDS` seeds and the
    point forecast is the mean of the per-seed log-RV predictions. Averaging the
    seeds stabilizes the beat-HAR (kills single-seed init variance) and lets us
    report seed dispersion as a model-uncertainty diagnostic. Each fold's ensemble
    trains on the FULL train fold — no calibration carve-out — so it reproduces the
    bake-off DLq quality at the data-hungry h5/h22 horizons.

  - Walk-forward (rolling) conformal bands — we never sacrifice training data for
    calibration. Folds are processed in time order; the band for fold k uses the
    |true_logRV − ens_logRV| nonconformity residuals ACCUMULATED FROM PRIOR FOLDS
    (1..k-1), per horizon and per α. We take the (1−α) finite-sample quantile q and
    emit symmetric log-RV intervals [ens − q, ens + q] on fold k, then measure
    empirical coverage prospectively. Because the calibration residuals come from
    strictly earlier folds, there is no leakage and no training-data loss. Exponen-
    tiating the bounds maps them to variance/vol bands.

LEAK-SAFETY: HAR + ensemble fit on the full train fold; conformal calibration uses
only prior folds' OOS residuals (strictly precede the test fold); walk-forward
embargo = max horizon + 1; the forward target stays t+1..t+h.

The walk-forward eval (`run`) confirms the ensemble beat-HAR is seed-robust and the
prospective bands are calibrated. `fit_production` then fits HAR + the ensemble on
ALL history, derives the conformal quantiles from the pooled walk-forward OOS
residuals, and saves a serving artifact (per-seed state_dicts + JSON spec) a serving
layer can load to produce a banded multi-horizon forecast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from app.data.dense_daily_dataset import walk_forward_splits
from app.data.dense_forecast_train import _fit_predict_ols, _oos_r2
from app.data.intraday_rv_forecast import (
    _bootstrap_qlike_gain_ci,
    _forward_log_rv,
    _har_lags,
    _qlike,
    _train_fold_qlike,
)

if TYPE_CHECKING:
    import torch

_EPS = 1e-12

# Default ensemble seeds; the count (5) trades stability against compute.
N_SEEDS = 5
DEFAULT_SEEDS: tuple[int, ...] = (11, 22, 33, 44, 55)
DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 22)
# Nominal mis-coverage levels: α=0.2 → 80% band, α=0.1 → 90% band.
DEFAULT_ALPHAS: tuple[float, ...] = (0.2, 0.1)

# Realized-measure columns of the `full` feature matrix (must match the bake-off):
# HAR daily/weekly/monthly are prepended; these follow, with log(rvol) last.
_FEAT_COLS = ("rs_pos", "rs_neg", "bv", "rq", "rskew", "rkurt", "parkinson")


def _build_full(df: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rv, log_rv, full) — the realized-measure feature matrix.

    `full` = [HAR daily/weekly/monthly, rs_pos, rs_neg, bv, rq, rskew, rkurt,
    parkinson, log(rvol)] — identical column order to `intraday_rv_forecast.run`,
    so the ensemble sees the exact features the bake-off validated. HAR daily/
    weekly/monthly occupy `full[:, :3]`.
    """

    rv = df["rv"].to_numpy(dtype=np.float64)
    log_rv = np.log(rv + _EPS)
    har = _har_lags(log_rv)
    extra = np.column_stack([df[c].to_numpy(dtype=np.float64) for c in _FEAT_COLS])
    extra = np.column_stack([extra, np.log(df["rvol"].to_numpy(dtype=np.float64) + 1.0)])
    full = np.column_stack([har, extra])
    return rv, log_rv, full


def _ensemble_predict(
    har_fit_tr: np.ndarray,
    full_tr: np.ndarray,
    resid_tr: np.ndarray,
    full_te: np.ndarray,
    har_pred_te: np.ndarray,
    *,
    seeds: tuple[int, ...],
    epochs: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-seed QLIKE-DLq log-RV predictions on `full_te`, plus their mean.

    Returns (per_seed, mean) where per_seed has shape (len(seeds), n_test) and
    mean is the ensemble point forecast (mean across seeds in log-RV space).
    """

    per_seed = np.stack(
        [
            _train_fold_qlike(
                har_fit_tr,
                full_tr,
                resid_tr,
                full_te,
                har_pred_te,
                seed=s,
                epochs=epochs,
                device=device,
            )
            for s in seeds
        ]
    )
    return per_seed, per_seed.mean(axis=0)


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile of nonconformity scores.

    The (1−α) coverage guarantee needs the ceil((n+1)(1−α))/n empirical quantile
    of the |residual| calibration scores; the (n+1) correction is what makes the
    band valid in finite samples rather than only asymptotically.
    """

    n = len(scores)
    if n == 0:
        return float("nan")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank >= n:  # quantile level exceeds what n points can certify → widest score
        return float(np.max(scores))
    return float(np.sort(scores)[rank - 1])


def _coverage(true: np.ndarray, pred: np.ndarray, q: float) -> float:
    """Empirical fraction of test points inside the symmetric log-RV band ±q."""

    if len(true) == 0:
        return float("nan")
    return float(np.mean(np.abs(true - pred) <= q))


def run(
    rv_path: Path | str,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_folds: int = 5,
    epochs: int = 300,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    device: str = "cpu",
) -> dict[str, Any]:
    """Walk-forward eval: HAR vs ensemble-DLq QLIKE + conformal band coverage.

    Per horizon reports HAR QLIKE; per-seed and ensemble DLq QLIKE; the
    ensemble-vs-HAR QLIKE-gain block-bootstrap CI (block=h); OOS-R²; and the
    empirical coverage of the 80%/90% walk-forward conformal bands vs nominal.

    The ensemble trains on the FULL train fold (no calibration carve-out), so it
    reproduces the bake-off DLq quality. Bands come from prior-folds-only OOS
    residuals: fold k is banded by the quantile of the residuals pooled over folds
    1..k-1, so coverage is prospective and leakage-free (fold 1 has no prior
    residuals and is skipped for coverage).
    """

    import pandas as pd

    df = pd.read_parquet(rv_path).sort_values("date").reset_index(drop=True)
    rv, _log_rv, full = _build_full(df)
    har = full[:, :3]

    results: dict[str, Any] = {
        "n_days": int(len(df)),
        "seeds": list(seeds),
        "n_folds": n_folds,
        "epochs": epochs,
        "alphas": list(alphas),
        "by_horizon": {},
    }
    for h in horizons:
        y = _forward_log_rv(rv, h)
        idx = np.where(~np.isnan(y))[0]
        folds = walk_forward_splits(len(idx), n_folds=n_folds, embargo=max(h, 1) + 1)
        har_pool: list[float] = []
        ens_pool: list[float] = []
        true_pool: list[float] = []
        base_pool: list[float] = []
        seed_pools: list[list[float]] = [[] for _ in seeds]
        # Walk-forward conformal: residuals accumulate over folds; fold k is banded
        # by the (1−α) quantile of the residuals from folds 1..k-1 (prior only).
        cal_resid: list[float] = []
        band_hits: dict[float, float] = dict.fromkeys(alphas, 0.0)
        band_n: dict[float, int] = dict.fromkeys(alphas, 0)

        for tr_l, te_l in folds:
            tr, te = idx[np.array(tr_l)], idx[np.array(te_l)]
            ytr, yte = y[tr], y[te]

            # FULL-train HAR floors the residual stack and is the eval baseline.
            har_fit_tr = _fit_predict_ols(har[tr], ytr, har[tr])
            resid_tr = (ytr - har_fit_tr).reshape(-1, 1)
            har_pred_te = _fit_predict_ols(har[tr], ytr, har[te])

            # Ensemble trains on the FULL train fold (no calibration carve-out).
            per_seed_te, ens_te = _ensemble_predict(
                har_fit_tr, full[tr], resid_tr, full[te], har_pred_te,
                seeds=seeds, epochs=epochs, device=device,
            )

            # Band fold k from the prior folds' OOS residuals (skip fold 1: none yet).
            if cal_resid:
                scores = np.asarray(cal_resid)
                for a in alphas:
                    q = _conformal_quantile(scores, a)
                    band_hits[a] += _coverage(yte, ens_te, q) * len(te)
                    band_n[a] += len(te)
            # This fold's OOS residuals join the pool for later folds.
            cal_resid.extend(np.abs(yte - ens_te).tolist())

            har_pool.extend(har_pred_te.tolist())
            ens_pool.extend(ens_te.tolist())
            true_pool.extend(yte.tolist())
            base_pool.extend([float(ytr.mean())] * len(te))
            for j in range(len(seeds)):
                seed_pools[j].extend(per_seed_te[j].tolist())

        har_arr = np.asarray(har_pool)
        ens_arr = np.asarray(ens_pool)
        true_arr = np.asarray(true_pool)
        base_arr = np.asarray(base_pool)
        seed_arrs = [np.asarray(p) for p in seed_pools]

        per_seed_qlike = [_qlike(s, true_arr) for s in seed_arrs]
        row: dict[str, Any] = {
            "qlike_har": _qlike(har_arr, true_arr),
            "qlike_ens": _qlike(ens_arr, true_arr),
            "qlike_per_seed": per_seed_qlike,
            "qlike_per_seed_mean": float(np.mean(per_seed_qlike)),
            "qlike_per_seed_std": float(np.std(per_seed_qlike)),
            "r2_har": _oos_r2(har_arr, true_arr, base_arr),
            "r2_ens": _oos_r2(ens_arr, true_arr, base_arr),
            "ens_vs_har_qlike_gain_ci90": _bootstrap_qlike_gain_ci(
                ens_arr, har_arr, true_arr, seed=seeds[0], block=max(h, 1)
            ),
            "coverage": {},
        }
        for a in alphas:
            emp = band_hits[a] / band_n[a] if band_n[a] else float("nan")
            row["coverage"][f"{1.0 - a:.2f}"] = {
                "nominal": round(1.0 - a, 4),
                "empirical": round(emp, 4),
            }
        results["by_horizon"][f"h{h}"] = row
    return results


def _walk_forward_oos_resid(
    rv: np.ndarray,
    full: np.ndarray,
    har: np.ndarray,
    h: int,
    *,
    seeds: tuple[int, ...],
    n_folds: int,
    epochs: int,
    device: str,
) -> np.ndarray:
    """Pooled |true − ensemble| OOS residuals over a walk-forward at horizon h.

    Each fold's ensemble trains on the FULL train fold (matching `run`), so the
    residuals reflect the deployable full-train quality. The pool is the best
    prospective estimate of the serving model's nonconformity.
    """

    y = _forward_log_rv(rv, h)
    idx = np.where(~np.isnan(y))[0]
    folds = walk_forward_splits(len(idx), n_folds=n_folds, embargo=max(h, 1) + 1)
    resid: list[float] = []
    for tr_l, te_l in folds:
        tr, te = idx[np.array(tr_l)], idx[np.array(te_l)]
        ytr, yte = y[tr], y[te]
        har_fit_tr = _fit_predict_ols(har[tr], ytr, har[tr])
        resid_tr = (ytr - har_fit_tr).reshape(-1, 1)
        har_pred_te = _fit_predict_ols(har[tr], ytr, har[te])
        _per_seed, ens_te = _ensemble_predict(
            har_fit_tr, full[tr], resid_tr, full[te], har_pred_te,
            seeds=seeds, epochs=epochs, device=device,
        )
        resid.extend(np.abs(yte - ens_te).tolist())
    return np.asarray(resid)


def fit_production(
    rv_path: Path | str,
    out_dir: Path | str,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_folds: int = 5,
    epochs: int = 300,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    device: str = "cpu",
) -> dict[str, Any]:
    """Fit HAR + the QLIKE-DLq ensemble on ALL history; save a serving artifact.

    No test split — every row trains the deployable model on the FULL history
    (no calibration carve-out), so the served ensemble reproduces the beat-HAR
    quality. The conformal quantiles come from the pooled walk-forward OOS
    residuals (the prospective nonconformity estimate), NOT a sacrificed tail.
    Saves per-seed state_dicts (.pt) and a JSON spec (HAR coefficients, feature
    standardization stats, conformal quantiles per horizon/α, metadata) so a
    serving layer can reconstruct the banded forecast.
    """

    import pandas as pd
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(rv_path).sort_values("date").reset_index(drop=True)
    rv, _log_rv, full = _build_full(df)
    har = full[:, :3]
    n = len(df)

    spec: dict[str, Any] = {
        "model": "intraday_rv_production",
        "kind": "qlike_dlq_ensemble+walk_forward_conformal",
        "n_days": int(n),
        "seeds": list(seeds),
        "n_folds": n_folds,
        "epochs": epochs,
        "horizons": list(horizons),
        "alphas": list(alphas),
        "feature_order": ["har_daily", "har_weekly", "har_monthly", *_FEAT_COLS, "log_rvol"],
        "date_first": str(df["date"].iloc[0]),
        "date_last": str(df["date"].iloc[-1]),
        "by_horizon": {},
    }

    for h in horizons:
        y = _forward_log_rv(rv, h)
        idx = np.where(~np.isnan(y))[0]
        y_all = y[idx]

        # HAR coefficients on ALL valid-target history (intercept first), for serving.
        a_all = np.column_stack([np.ones(len(idx)), har[idx]])
        coef, *_ = np.linalg.lstsq(a_all, y_all, rcond=None)
        har_all_fit = a_all @ coef
        resid_all = (y_all - har_all_fit).reshape(-1, 1)

        # Standardization stats the QLIKE head used (all history), plus the residual
        # mean/std the head's output is un-standardized with — saved so serving can
        # reproduce exp(HAR + resid_std*rs + rm) without re-deriving anything.
        full_all = full[idx]
        xm, xs = full_all.mean(0), full_all.std(0)
        xs = np.where(xs > 0, xs, 1.0)
        rm = float(resid_all.mean())
        rs = float(resid_all.std()) if resid_all.std() > 0 else 1.0

        # Fit each seed's QLIKE-DLq on ALL history; serialize the all-history weights.
        seed_files = []
        for s in seeds:
            sd = _fit_seed_state(
                har_all_fit, full_all, resid_all, seed=s, epochs=epochs, device=device
            )
            fname = f"h{h}_seed{s}.pt"
            torch.save(sd, out / fname)
            seed_files.append(fname)

        # Conformal quantiles from the pooled walk-forward OOS residuals (prospective
        # nonconformity estimate), not from a held-out tail of the serving model.
        scores = _walk_forward_oos_resid(
            rv, full, har, h, seeds=seeds, n_folds=n_folds, epochs=epochs, device=device
        )
        quantiles = {f"{a:.2f}": _conformal_quantile(scores, a) for a in alphas}

        spec["by_horizon"][f"h{h}"] = {
            "har_coef": coef.tolist(),  # [intercept, daily, weekly, monthly]
            "feat_mean": xm.tolist(),
            "feat_std": xs.tolist(),
            "resid_mean": rm,
            "resid_std": rs,
            "conformal_quantiles": quantiles,  # α → log-RV half-width
            "seed_state_dicts": seed_files,
            "n_oos_resid": int(len(scores)),
        }

    (out / "production_artifact.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return spec


def _fit_seed_state(
    har_core_fit: np.ndarray,
    full_core: np.ndarray,
    resid_core: np.ndarray,
    *,
    seed: int,
    epochs: int,
    device: str,
) -> dict[str, "torch.Tensor"]:
    """Fit one QLIKE-DLq seed on the train-core; return its best state_dict.

    Mirrors `_train_fold_qlike`'s training discipline (standardization, QLIKE
    loss on reconstructed variance, grad-clip, early stop) but returns the
    serializable weights instead of test predictions, so the saved artifact
    reproduces the head a serving layer must run.
    """

    import torch

    from app.data.dense_forecast_train import _build_model
    from app.data.intraday_rv_forecast import _EPS as _RVEPS
    from app.data.intraday_rv_forecast import _LOGV_CLAMP
    from app.determinism import enable_deterministic_mode

    enable_deterministic_mode(seed)
    dev = torch.device(device)
    xm, xs = full_core.mean(0), full_core.std(0)
    xs = np.where(xs > 0, xs, 1.0)
    rm = resid_core.mean()
    rs = resid_core.std() if resid_core.std() > 0 else 1.0
    xtr = torch.tensor((full_core - xm) / xs, dtype=torch.float32, device=dev)
    har_tr = torch.tensor(har_core_fit, dtype=torch.float32, device=dev).reshape(-1, 1)
    log_true_tr = torch.tensor(
        (resid_core.reshape(-1) + har_core_fit).reshape(-1, 1), dtype=torch.float32, device=dev
    )

    def qlike_loss(resid_std: "torch.Tensor", har: "torch.Tensor", log_true: "torch.Tensor") -> Any:
        log_pred = har + (resid_std * rs + rm)
        log_pred = torch.clamp(log_pred, -_LOGV_CLAMP, _LOGV_CLAMP)
        log_true_c = torch.clamp(log_true, -_LOGV_CLAMP, _LOGV_CLAMP)
        var_pred = torch.exp(log_pred) + _RVEPS
        var_true = torch.exp(log_true_c)
        ratio = var_true / var_pred
        return torch.mean(ratio - torch.log(ratio) - 1.0)

    n_val = max(1, len(xtr) // 5)
    tr, val = slice(0, len(xtr) - n_val), slice(len(xtr) - n_val, len(xtr))
    model = _build_model(full_core.shape[1], 1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = qlike_loss(model(xtr[tr]), har_tr[tr], log_true_tr[tr])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    return cast("dict[str, torch.Tensor]", {k: v.cpu() for k, v in model.state_dict().items()})


def _print_table(res: dict[str, Any]) -> None:
    print(f"n_days={res['n_days']}  seeds={res['seeds']}  epochs={res['epochs']}")
    print("QLIKE (lower better) + ensemble-vs-HAR gain CI90 (>0 ⇒ ensemble beats HAR)")
    print(
        f"{'horizon':<8}{'HAR':>10}{'seedMean':>10}{'seedStd':>9}{'ENS':>10}"
        f"{'R2_HAR':>9}{'R2_ENS':>9}{'GainCI90':>22}"
    )
    for hk, r in res["by_horizon"].items():
        g = r["ens_vs_har_qlike_gain_ci90"]
        print(
            f"{hk:<8}{r['qlike_har']:>10.4f}{r['qlike_per_seed_mean']:>10.4f}"
            f"{r['qlike_per_seed_std']:>9.4f}{r['qlike_ens']:>10.4f}"
            f"{r['r2_har']:>9.3f}{r['r2_ens']:>9.3f}"
            f"{f'[{g[0]:+.4f},{g[1]:+.4f}]':>22}"
        )
    print("Conformal band coverage (empirical vs nominal)")
    print(f"{'horizon':<8}{'level':>8}{'nominal':>10}{'empirical':>12}")
    for hk, r in res["by_horizon"].items():
        for level, cov in r["coverage"].items():
            print(f"{hk:<8}{level:>8}{cov['nominal']:>10.2f}{cov['empirical']:>12.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deployable banded RV forecaster: QLIKE-DLq ensemble + conformal."
    )
    parser.add_argument("--rv-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    seeds = tuple(args.seeds)

    res = run(args.rv_path, seeds=seeds, epochs=args.epochs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "production_eval.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8"
    )
    _print_table(res)
    spec = fit_production(args.rv_path, args.out_dir, seeds=seeds, epochs=args.epochs)
    print(
        f"artifact: {len(spec['horizons'])} horizons × {len(seeds)} seeds "
        f"→ {args.out_dir}/production_artifact.json"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
