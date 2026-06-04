"""Phase 2: marginal value of FOMC statement text over the dense backbone.

The thesis question, asked correctly: on FOMC announcement days, does the
statement text explain realized-vol / abnormal-volume that the HAR/AR
baseline misses? Two-stage, leakage-safe test:

  1. Per walk-forward fold, the HAR/AR baseline (dense_forecast_train)
     predicts every row; its residual (true − baseline) is the signal the
     backbone leaves on the table.
  2. On FOMC rows only, fit a ridge of that residual on the statement-text
     features (per-fold PCA of the FinBERT embedding + a statement-delta
     scalar), fit on train-FOMC rows, evaluated on test-FOMC rows.

Marginal Δ = R²(baseline + text-residual) − R²(baseline) on FOMC-day test
rows. Δ CI clearing 0 ⇒ text adds value; otherwise an honest null inside a
model that demonstrably works. Restricting to FOMC rows holds calendar
proximity fixed, so the Δ isolates text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from app.data.dense_daily_dataset import DEFAULT_HORIZONS, build_dataset, walk_forward_splits
from app.data.dense_forecast_train import _baseline_matrix, _oos_r2

_EPS = 1e-8


def statement_dates_and_text(events_parquet: Path | str) -> dict[str, str]:
    """Map FOMC statement date → canonical text (one per date)."""

    import pandas as pd

    frame = pd.read_parquet(events_parquet)
    frame = frame[frame["event_kind"].astype(str).str.lower() == "statement"]
    frame = frame.assign(_d=frame["event_date"].astype(str).str[:10])
    deduped = frame.drop_duplicates(subset="_d", keep="first")
    return {str(r["_d"]): str(r["text"]) for _, r in deduped.iterrows()}


def build_fomc_embeddings(
    events_parquet: Path | str, out_path: Path | str, *, force: bool = False
) -> Path:
    """Cache date → 768 mean-pooled FinBERT embedding for FOMC statements."""

    import pandas as pd

    out_path = Path(out_path)
    if out_path.exists() and not force:
        return out_path
    from app.services.encoder_provenance import write_encoder_sidecar
    from app.services.text_encoder import (
        assert_primary_model_loaded,
        encode_chunks,
        loaded_encoder_provenance,
    )

    # Fail loudly rather than silently caching embeddings from a fallback model.
    assert_primary_model_loaded()
    provenance = loaded_encoder_provenance()
    texts = statement_dates_and_text(events_parquet)
    encoded: list[tuple[str, np.ndarray | None]] = []
    for date_iso in sorted(texts):
        encs = encode_chunks(texts[date_iso])
        vecs = [np.asarray(e.embedding, dtype=np.float64) for e in encs if e.embedding]
        encoded.append((date_iso, np.mean(vecs, axis=0) if vecs else None))
    # Size to the encoder's actual width, not a hardcoded 768 (FOMC-RoBERTa is
    # 1024); empty-text rows fall back to zeros of the SAME width.
    dim = next(
        (len(v) for _, v in encoded if v is not None),
        provenance.get("hidden_size") or 768,
    )
    rows = [
        {"date": date_iso, **{f"emb_{i}": float(emb[i]) for i in range(dim)}}
        for date_iso, v in encoded
        for emb in [v if v is not None else np.zeros(dim)]
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    # Stamp which encoder produced these vectors so provenance is auditable.
    write_encoder_sidecar(out_path, provenance)
    print(f"[dense_fomc_text] wrote {len(rows)} FOMC embeddings to {out_path}")
    return out_path


def _pca_fit_transform(train_emb: np.ndarray, all_emb: np.ndarray, k: int) -> np.ndarray:
    """Whitened PCA: fit on train_emb (mean-centre + top-k SVD), transform all_emb."""

    mean = train_emb.mean(axis=0)
    u, s, vt = np.linalg.svd(train_emb - mean, full_matrices=False)
    k = min(k, vt.shape[0])
    comps = vt[:k]
    scale = s[:k] / np.sqrt(max(len(train_emb) - 1, 1))
    scale = np.where(scale > _EPS, scale, 1.0)
    return cast(np.ndarray, ((all_emb - mean) @ comps.T) / scale)


def _statement_delta(emb_fomc_ordered: np.ndarray) -> np.ndarray:
    """1 − cosine(emb_t, emb_{prev}) over chronologically-ordered FOMC embeddings."""

    out = np.zeros(len(emb_fomc_ordered))
    for i in range(1, len(emb_fomc_ordered)):
        a, b = emb_fomc_ordered[i], emb_fomc_ordered[i - 1]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > _EPS and nb > _EPS:
            out[i] = 1.0 - float(a @ b / (na * nb))
    return out


def _ridge_fit_predict(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, *, alpha: float = 1.0
) -> np.ndarray:
    """Standardized ridge with intercept (closed form). Returns test predictions."""

    xm, xs = x_tr.mean(0), x_tr.std(0)
    xs = np.where(xs > _EPS, xs, 1.0)
    ym = y_tr.mean()
    xs_tr = (x_tr - xm) / xs
    xs_te = (x_te - xm) / xs
    d = xs_tr.shape[1]
    w = np.linalg.solve(xs_tr.T @ xs_tr + alpha * np.eye(d), xs_tr.T @ (y_tr - ym))
    return cast(np.ndarray, xs_te @ w + ym)


def _bootstrap_delta_ci(
    notext: np.ndarray, text: np.ndarray, true: np.ndarray, base: np.ndarray, *, seed: int = 11
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(true)
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        d = _oos_r2(text[idx], true[idx], base[idx]) - _oos_r2(notext[idx], true[idx], base[idx])
        if not np.isnan(d):
            boots.append(d)
    return float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))


def _diebold_mariano(
    true: np.ndarray, notext: np.ndarray, text: np.ndarray, *, h: int = 1
) -> dict[str, float]:
    """Diebold-Mariano test of equal predictive accuracy (squared-error loss).

    Loss differential d_t = e_notext² − e_text²; a positive mean means the text
    model has lower error (text helps). The variance of the mean uses a
    Newey-West HAC correction with lag h−1 (Bartlett kernel), the standard
    small-sample fix for h-step overlapping forecasts. Two-sided p-value from
    the standard normal. d̄>0 with p<0.05 ⇒ text significantly better; d̄<0 with
    p<0.05 ⇒ text significantly worse.
    """
    import math

    d = (true - notext) ** 2 - (true - text) ** 2
    n = len(d)
    if n < 8:
        return {
            "dm_stat": float("nan"),
            "dm_p_two_sided": float("nan"),
            "mean_loss_diff_notext_minus_text": float("nan"),
            "hac_lag": 0,
            "n": n,
        }
    dbar = float(d.mean())
    dc = d - dbar
    # Newey-West lag h-1, capped at n//4 so the HAC estimate stays determined on
    # small FOMC-only pools (a deep lag on a short series can drive var negative).
    lag = min(max(h - 1, 0), n // 4)
    var = float(np.mean(dc * dc))  # gamma_0
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1)
        var += 2.0 * w * float(np.mean(dc[k:] * dc[:-k]))
    se = math.sqrt(var / n) if var > 0 else float("nan")
    dm = dbar / se if se and not math.isnan(se) else float("nan")
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(dm) / math.sqrt(2.0)))) if dm == dm else float("nan")
    return {
        "dm_stat": float(dm),
        "dm_p_two_sided": float(p),
        "mean_loss_diff_notext_minus_text": dbar,
        "hac_lag": lag,
        "n": n,
    }


def run_text_marginal(
    cache_dir: Path | str,
    embeddings_parquet: Path | str,
    *,
    seed: int = 11,
    n_folds: int = 5,
    embargo: int = 10,
    pca_k: int = 16,
    weak_baseline: bool = False,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    import pandas as pd

    X, Y, dates = build_dataset(cache_dir, horizons=horizons)
    target_cols = [f"rv_{h}" for h in horizons] + ["av"]
    Yt = Y.copy()
    for h in horizons:
        Yt[f"rv_{h}"] = np.log(Y[f"rv_{h}"] + _EPS)
    Xv = X.to_numpy(dtype=np.float64)
    Yv = Yt.to_numpy(dtype=np.float64)
    har_idx = [X.columns.get_loc(c) for c in ("rv_lag_1", "rv_lag_5", "rv_lag_22")]
    av_idx = [X.columns.get_loc(c) for c in ("av_lag_1", "vol_ratio_30", "dow_0", "dow_1")]

    emb_df = pd.read_parquet(embeddings_parquet)
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    emb_map = {str(r["date"]): r[emb_cols].to_numpy(dtype=np.float64) for _, r in emb_df.iterrows()}
    date_str = dates.astype(str).str[:10].tolist()
    is_fomc = np.array([d in emb_map for d in date_str])
    emb_all = np.zeros((len(date_str), len(emb_cols)))
    for i, d in enumerate(date_str):
        if is_fomc[i]:
            emb_all[i] = emb_map[d]
    # statement-delta over the chronological FOMC rows, scattered back to all rows
    fomc_order = np.where(is_fomc)[0]
    delta_all = np.zeros(len(date_str))
    delta_all[fomc_order] = _statement_delta(emb_all[fomc_order])

    folds = walk_forward_splits(len(Xv), n_folds=n_folds, embargo=embargo)
    pooled: dict[str, dict[str, list[float]]] = {
        c: {"notext": [], "text": [], "true": [], "base": []} for c in target_cols
    }
    for tr, te in folds:
        base_tr, base_te = _baseline_matrix(
            Xv[tr], Yv[tr], Xv[te], target_cols=target_cols, har_idx=har_idx, av_idx=av_idx
        )
        if weak_baseline:
            # Replace the HAR/AR baseline with the per-fold train mean, so the
            # text model must predict the demeaned target on its own — measures
            # text's STANDALONE signal (does text help a weak baseline at all?).
            means = Yv[tr].mean(axis=0)
            base_tr = np.broadcast_to(means, base_tr.shape).copy()
            base_te = np.broadcast_to(means, base_te.shape).copy()
        tr_arr, te0 = np.array(tr), te[0]
        fomc_tr = tr_arr[is_fomc[tr_arr]]
        fomc_te = np.array([g for g in te if is_fomc[g]])
        if len(fomc_tr) < max(pca_k, 1) + 2 or len(fomc_te) == 0:
            continue
        if pca_k > 0:
            # full text block: per-fold PCA of the embedding + statement-delta
            pca_tr = _pca_fit_transform(emb_all[fomc_tr], emb_all[fomc_tr], pca_k)
            pca_te = _pca_fit_transform(emb_all[fomc_tr], emb_all[fomc_te], pca_k)
            text_tr = np.column_stack([pca_tr, delta_all[fomc_tr]])
            text_te = np.column_stack([pca_te, delta_all[fomc_te]])
        else:
            # delta-only: the single statement-change scalar (minimal overfit)
            text_tr = delta_all[fomc_tr].reshape(-1, 1)
            text_te = delta_all[fomc_te].reshape(-1, 1)
        for j, col in enumerate(target_cols):
            resid_tr = Yv[fomc_tr, j] - base_tr[is_fomc[tr_arr], j]
            text_resid = _ridge_fit_predict(text_tr, resid_tr, text_te)
            base_fomc_te = base_te[fomc_te - te0, j]
            true = Yv[fomc_te, j]
            fold_mean = float(Yv[tr, j].mean())
            pooled[col]["notext"].extend(base_fomc_te.tolist())
            pooled[col]["text"].extend((base_fomc_te + text_resid).tolist())
            pooled[col]["true"].extend(true.tolist())
            pooled[col]["base"].extend([fold_mean] * len(fomc_te))

    results: dict[str, Any] = {"n_fomc_test": len(pooled["av"]["true"]), "by_target": {}}
    for col in target_cols:
        p = {k: np.asarray(v) for k, v in pooled[col].items()}
        r2_no = _oos_r2(p["notext"], p["true"], p["base"])
        r2_tx = _oos_r2(p["text"], p["true"], p["base"])
        lo, hi = _bootstrap_delta_ci(p["notext"], p["text"], p["true"], p["base"], seed=seed)
        h_col = int(col.split("_")[1]) if col.startswith("rv_") else 1
        dm = _diebold_mariano(p["true"], p["notext"], p["text"], h=h_col)
        results["by_target"][col] = {
            "r2_notext": r2_no,
            "r2_text": r2_tx,
            "delta": r2_tx - r2_no,
            "delta_ci90": [lo, hi],
            "diebold_mariano": dm,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="FOMC text marginal test over the dense backbone.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--events-parquet", type=Path, required=True)
    parser.add_argument("--embeddings-parquet", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--pca-k", type=int, default=16, help="Embedding PCA dims; 0 = delta-only text feature."
    )
    parser.add_argument(
        "--weak-baseline",
        action="store_true",
        help="Use the train mean (not HAR/AR) as baseline — measures text's standalone signal.",
    )
    args = parser.parse_args()

    build_fomc_embeddings(args.events_parquet, args.embeddings_parquet)
    res = run_text_marginal(
        args.cache_dir,
        args.embeddings_parquet,
        seed=args.seed,
        pca_k=args.pca_k,
        weak_baseline=args.weak_baseline,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"FOMC test rows pooled: {res['n_fomc_test']}")
    print(f"{'target':<8} {'R2_notext':>10} {'R2_text':>10} {'delta':>8} {'delta_CI90':>18}")
    for col, r in res["by_target"].items():
        ci = f"[{r['delta_ci90'][0]:+.3f},{r['delta_ci90'][1]:+.3f}]"
        print(
            f"{col:<8} {r['r2_notext']:>10.3f} {r['r2_text']:>10.3f} {r['delta']:>+8.3f} {ci:>18}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
