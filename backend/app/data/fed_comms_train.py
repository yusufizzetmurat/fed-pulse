"""Train + evaluate the gated text↔market fusion forecaster, walk-forward.

Pipeline: cache a mean-pooled FinBERT embedding per communication → assemble the
daily fusion design (HAR/market features, the most-recent fresh communication's
embedding + mask, forward-RV targets) → walk-forward train `GatedFusionForecaster`
with the supervised + InfoNCE objective, residual-stacked on HAR so the floor is
HAR itself.

Reported per horizon: HAR R², market-only R² (gate forced off), fused R² and the
fused−HAR lift with a bootstrap CI — overall and on text-active days — plus the
mean gate by communication type (the interpretable "how much did text matter").
A positive, CI-clearing lift on text-active days is the result that would
overturn the text-is-null finding; anything else is a stronger, scaled null.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from app.config import DATA_DIR
from app.data.dense_daily_dataset import walk_forward_splits
from app.data.dense_forecast_train import _bootstrap_r2_ci, _fit_predict_ols, _oos_r2
from app.data.fed_comms_dataset import DEFAULT_HORIZONS, MEASURES
from app.data.intraday_rv_forecast import _market_block

DEFAULT_FUSION_DIR = DATA_DIR / "processed" / "fed_comms_fusion"
_FRESH_DAYS = 5  # a communication counts as "active" text within this many trading days
_DEFAULT_EMB_DIM = 768  # fallback only; the real dim is read from the encoder/parquet


def build_corpus_embeddings(
    corpus_path: Path | str, out_path: Path | str, *, force: bool = False
) -> Path:
    """Cache url → mean-pooled text embedding per communication (encoder-agnostic).

    The embedding dimension is taken from the encoder, not assumed — so swapping
    the encoder (e.g. 768-d FinBERT → 1024-d bge) needs no code change here.
    """

    import pandas as pd

    out_path = Path(out_path)
    if out_path.exists() and not force:
        return out_path
    from app.services.text_encoder import encode_chunks

    corpus = pd.read_parquet(corpus_path)
    embs: list[tuple[str, np.ndarray | None]] = []
    for _, doc in corpus.iterrows():
        encs = encode_chunks(str(doc["text"]))
        vecs = [np.asarray(e.embedding, dtype=np.float64) for e in encs if e.embedding]
        embs.append((str(doc["url"]), np.mean(vecs, axis=0) if vecs else None))
    dim = next((len(v) for _, v in embs if v is not None), _DEFAULT_EMB_DIM)
    rows = [
        {"url": url, **{f"emb_{i}": float(e[i]) for i in range(dim)}}
        for url, v in embs
        for e in [v if v is not None else np.zeros(dim)]
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"[fed_comms_train] wrote {len(rows)} doc embeddings (dim={dim}) to {out_path}")
    return out_path


def _assemble(
    daily: Any,
    corpus: Any,
    emb_df: Any,
    market_cache_dir: Path | str,
    horizons: tuple[int, ...],
    *,
    measure: str = "rv",
) -> dict[str, Any]:
    """Build per-day arrays for a target measure: market features, text, targets, HAR lags."""

    import pandas as pd

    daily = daily.sort_values("date").reset_index(drop=True)
    har = daily[[f"{measure}_daily", f"{measure}_weekly", f"{measure}_monthly"]].to_numpy(
        dtype=np.float64
    )
    # cross-market block keyed off the RV daily lag (a stable vol proxy) regardless of target
    rv_daily = daily["rv_daily"].to_numpy(dtype=np.float64)
    market = _market_block(market_cache_dir, daily["date"], rv_daily)
    # FOMC-calendar features: market baseline knows WHEN statements occur, so the
    # text contribution isolates content rather than FOMC-day detection.
    cal = daily[["days_since_stmt", "days_to_stmt"]].to_numpy(dtype=np.float64)
    # Market-derived MP surprise of the most-recent statement (as-of joined,
    # leak-safe). These are MARKET features → they feed both the gate-off market
    # path and the fused path, so we can test whether the surprise improves the
    # market forecast and whether text adds anything beyond market+surprise.
    surprise = daily[["surprise_level", "surprise_path", "surprise_info"]].to_numpy(
        dtype=np.float64
    )
    market_feat = np.column_stack([har, market, cal, surprise])

    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    dim = len(emb_cols)
    url_to_emb = {r["url"]: r[emb_cols].to_numpy(dtype=np.float64) for _, r in emb_df.iterrows()}
    corpus_urls = corpus.sort_values("date").reset_index(drop=True)["url"].tolist()

    n = len(daily)
    text_emb = np.zeros((n, dim))
    text_mask = np.zeros(n)
    doc_types: list[str | None] = [None] * n
    for i, row in daily.iterrows():
        di = int(row["doc_row"])
        fresh = bool(row["has_text"]) and 0 <= int(row["doc_age_days"]) <= _FRESH_DAYS
        if di >= 0 and fresh and corpus_urls[di] in url_to_emb:
            text_emb[i] = url_to_emb[corpus_urls[di]]
            text_mask[i] = 1.0
            doc_types[i] = str(row["doc_type"])

    targets = np.column_stack(
        [daily[f"{measure}_fwd_{h}"].to_numpy(dtype=np.float64) for h in horizons]
    )
    valid = np.isfinite(targets).all(axis=1) & np.isfinite(market_feat).all(axis=1)
    return {
        "market_feat": market_feat,
        "text_emb": text_emb,
        "text_mask": text_mask,
        "targets": targets,
        "har": har,
        "doc_types": doc_types,
        "valid": valid,
        "dates": daily["date"].astype(str).tolist(),
        "days_since_stmt": daily["days_since_stmt"].to_numpy(dtype=np.float64),
    }


def _standardize(x_tr: np.ndarray, x_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m, s = x_tr.mean(0), x_tr.std(0)
    s = np.where(s > 0, s, 1.0)
    return (x_tr - m) / s, (x_te - m) / s


def _block_bootstrap_r2_ci(
    pred: np.ndarray, true: np.ndarray, base: np.ndarray, *, block: int, seed: int, n_boot: int = 1000
) -> tuple[float, float]:
    """Relative OOS-R² CI via a moving-block bootstrap (block ≥ horizon).

    Overlapping multi-day forward targets are autocorrelated, so an iid point
    bootstrap is anti-conservative. Resampling contiguous blocks of length
    `block` preserves that dependence and gives an honest interval.
    """

    n = len(true)
    if n <= block:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    stats = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        stats.append(_oos_r2(pred[idx], true[idx], base[idx]))
    return (float(np.percentile(stats, 5)), float(np.percentile(stats, 95)))


def _train_fusion_fold(
    data: dict[str, np.ndarray],
    tr: np.ndarray,
    te: np.ndarray,
    *,
    seed: int,
    epochs: int,
    info_nce_weight: float = 0.1,
    patience: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train on a fold with early stopping; return (fused, market_only, gate) in RV space.

    Mirrors the bake-off DL discipline: a time-ordered validation split (last 20%
    of the train fold) drives early stopping, and the HAR-residual target is
    standardized — so when text carries no generalizable signal the model stops
    early and collapses toward the HAR floor instead of overfitting.
    """

    import torch

    from app.data.gated_fusion import build_model, fusion_loss
    from app.determinism import enable_deterministic_mode

    enable_deterministic_mode(seed)
    har, tgt = data["har"], data["targets"]
    nt = tgt.shape[1]
    har_te_pred = np.column_stack([_fit_predict_ols(har[tr], tgt[tr, k], har[te]) for k in range(nt)])
    har_tr_pred = np.column_stack([_fit_predict_ols(har[tr], tgt[tr, k], har[tr]) for k in range(nt)])
    resid = tgt[tr] - har_tr_pred

    n = len(tr)
    n_val = max(1, n // 5)
    core, val = slice(0, n - n_val), slice(n - n_val, n)

    # standardize on the training-core only (no val/test leakage)
    mf = data["market_feat"][tr]
    mfm, mfs = mf[core].mean(0), mf[core].std(0)
    mfs = np.where(mfs > 0, mfs, 1.0)
    mf_std = (mf - mfm) / mfs
    mf_te = (data["market_feat"][te] - mfm) / mfs
    emb, mask_tr = data["text_emb"][tr], data["text_mask"][tr]
    present = mask_tr[core] > 0
    ref = emb[core][present] if present.any() else emb[core]
    em, es = ref.mean(0), ref.std(0)
    es = np.where(es > 0, es, 1.0)
    emb_std = ((emb - em) / es) * mask_tr[:, None]
    emb_te = ((data["text_emb"][te] - em) / es) * data["text_mask"][te][:, None]
    ym, ys = resid[core].mean(0), resid[core].std(0)
    ys = np.where(ys > 0, ys, 1.0)
    resid_std = (resid - ym) / ys

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(emb.shape[1], mf.shape[1], nt).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def tt(a: np.ndarray) -> "torch.Tensor":
        return torch.tensor(a, dtype=torch.float32, device=dev)

    emb_t, mf_t, mask_t, y_t = tt(emb_std), tt(mf_std), tt(mask_tr), tt(resid_std)
    vb = {"text_emb": emb_t[val], "market_feat": mf_t[val], "text_mask": mask_t[val]}

    core_pos = np.arange(0, n - n_val)
    rng = np.random.default_rng(seed)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(core_pos))
        for s in range(0, len(order), 256):
            b = core_pos[order[s : s + 256]]
            opt.zero_grad()
            batch = {"text_emb": emb_t[b], "market_feat": mf_t[b], "text_mask": mask_t[b], "targets": y_t[b]}
            loss = fusion_loss(model, batch, info_nce_weight=info_nce_weight)["loss"]
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
        model.eval()
        with torch.no_grad():
            vpred = model(vb["text_emb"], vb["market_feat"], vb["text_mask"])["pred"]
            vloss = float(torch.nn.functional.huber_loss(vpred, y_t[val]))
        if vloss < best - 1e-6:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        tem, mfe, msk = tt(emb_te), tt(mf_te), tt(data["text_mask"][te])
        fused = model(tem, mfe, msk)
        mkt_only = model(tem, mfe, torch.zeros_like(msk))  # gate forced off
    fused_pred = har_te_pred + (fused["pred"].cpu().numpy() * ys + ym)
    mkt_pred = har_te_pred + (mkt_only["pred"].cpu().numpy() * ys + ym)
    return fused_pred, mkt_pred, fused["gate"].cpu().numpy()


def run(
    fusion_dir: Path | str = DEFAULT_FUSION_DIR,
    *,
    market_cache_dir: Path | str,
    corpus_path: Path | str,
    emb_path: Path | str,
    seed: int = 11,
    epochs: int = 60,
    n_folds: int = 5,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    measure: str = "rv",
) -> dict[str, Any]:
    import pandas as pd

    fusion_dir = Path(fusion_dir)
    daily = pd.read_parquet(fusion_dir / "daily_fusion.parquet")
    corpus = pd.read_parquet(corpus_path)
    emb_df = pd.read_parquet(emb_path)
    data = _assemble(daily, corpus, emb_df, market_cache_dir, horizons, measure=measure)

    idx_all = np.where(data["valid"])[0]
    embargo = max(horizons) + 1
    folds = walk_forward_splits(len(idx_all), n_folds=n_folds, embargo=embargo)
    pools: dict[str, list[Any]] = {
        k: []
        for k in ("har", "fused", "mkt", "true", "base", "gate", "mask", "type", "date", "dss")
    }
    for tr_l, te_l in folds:
        tr, te = idx_all[np.array(tr_l)], idx_all[np.array(te_l)]
        fused, mkt, gate = _train_fusion_fold(data, tr, te, seed=seed, epochs=epochs)
        har_pred = np.column_stack(
            [
                _fit_predict_ols(data["har"][tr], data["targets"][tr, k], data["har"][te])
                for k in range(len(horizons))
            ]
        )
        pools["har"].append(har_pred)
        pools["fused"].append(fused)
        pools["mkt"].append(mkt)
        pools["true"].append(data["targets"][te])
        pools["base"].append(np.tile(data["targets"][tr].mean(0), (len(te), 1)))
        pools["gate"].append(gate)
        pools["mask"].append(data["text_mask"][te])
        pools["type"].extend([data["doc_types"][i] for i in te])
        pools["date"].extend([data["dates"][i] for i in te])
        pools["dss"].extend([float(data["days_since_stmt"][i]) for i in te])

    har = np.vstack(pools["har"])
    fused = np.vstack(pools["fused"])
    mkt = np.vstack(pools["mkt"])
    true = np.vstack(pools["true"])
    base = np.vstack(pools["base"])
    gate = np.concatenate(pools["gate"])
    mask = np.concatenate(pools["mask"])
    types = np.array([t if t is not None else "none" for t in pools["type"]])
    dates = np.array(pools["date"])
    days_since_stmt = np.array(pools["dss"], dtype=np.float64)

    results: dict[str, Any] = {"n_eval": int(true.shape[0]), "measure": measure, "by_horizon": {}}
    active = mask > 0
    for k, h in enumerate(horizons):
        row: dict[str, Any] = {
            "har": _oos_r2(har[:, k], true[:, k], base[:, k]),
            "mkt_only": _oos_r2(mkt[:, k], true[:, k], base[:, k]),
            "fused": _oos_r2(fused[:, k], true[:, k], base[:, k]),
        }
        row["fused_vs_har_ci90"] = list(_bootstrap_r2_ci(fused[:, k], true[:, k], har[:, k], seed=seed))
        # text isolation: fused vs the SAME model with the gate forced off (market-only).
        # The difference is purely the text contribution; CI ≤ 0 ⇒ text adds nothing.
        row["text_vs_mkt_ci90"] = list(_bootstrap_r2_ci(fused[:, k], true[:, k], mkt[:, k], seed=seed))
        # block bootstrap (block = horizon) — honest CI under overlapping-window autocorrelation
        row["text_vs_mkt_block_ci90"] = list(
            _block_bootstrap_r2_ci(fused[:, k], true[:, k], mkt[:, k], block=max(h, 1), seed=seed)
        )
        if active.sum() > 5:
            row["fused_active"] = _oos_r2(fused[active, k], true[active, k], base[active, k])
            row["mkt_active"] = _oos_r2(mkt[active, k], true[active, k], base[active, k])
            row["text_vs_mkt_active_ci90"] = list(
                _bootstrap_r2_ci(fused[active, k], true[active, k], mkt[active, k], seed=seed)
            )
        results["by_horizon"][f"h{h}"] = row

    results["text_active_frac"] = float(active.mean())
    results["gate_by_type"] = {
        t: float(gate[(types == t) & active].mean())
        for t in sorted(set(types[active]))
        if ((types == t) & active).any()
    }
    # regime-stratified: text-vs-market R² within market eras (is the null an
    # average that hides a regime-specific effect?)
    eras = {
        "pre2020": dates < "2020-01-01",
        "covid2020": (dates >= "2020-01-01") & (dates < "2021-01-01"),
        "recent2021+": dates >= "2021-01-01",
        # FOMC-adjacent days (≤ 3 trading days since the last statement): the
        # text contribution over market+surprise read on the days the surprise
        # is freshest.
        "fomc_window": days_since_stmt <= 3,
    }
    results["by_era"] = {
        era: {
            "n": int(sel.sum()),
            **{
                f"h{h}_text_vs_mkt": _oos_r2(fused[sel, k], true[sel, k], mkt[sel, k])
                for k, h in enumerate(horizons)
            },
            # block-bootstrap CI of the text-vs-market gap within the era, so a
            # positive point estimate (e.g. the FOMC-window slice) is testable
            # rather than taken at face value on a small overlapping-window sample.
            **{
                f"h{h}_text_vs_mkt_ci": list(
                    _block_bootstrap_r2_ci(
                        fused[sel, k], true[sel, k], mkt[sel, k], block=max(h, 1), seed=seed
                    )
                )
                for k, h in enumerate(horizons)
            },
        }
        for era, sel in eras.items()
        if sel.sum() > 20
    }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Train+eval gated fusion forecaster.")
    parser.add_argument("--fusion-dir", type=Path, default=DEFAULT_FUSION_DIR)
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--emb-path", type=Path, required=True)
    parser.add_argument("--market-cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target", default="rv", choices=list(MEASURES))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    res = run(
        args.fusion_dir,
        market_cache_dir=args.market_cache_dir,
        corpus_path=args.corpus_path,
        emb_path=args.emb_path,
        seed=args.seed,
        epochs=args.epochs,
        measure=args.target,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "fusion_bakeoff.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"target={res['measure']}  n_eval={res['n_eval']}  text_active_frac={res['text_active_frac']:.3f}")
    print(f"gate_by_type={ {k: round(v, 3) for k, v in res['gate_by_type'].items()} }")
    hdr = f"{'horizon':<8}{'HAR':>8}{'mkt':>8}{'fused':>8}{'txt-vs-mkt(iid)':>20}{'txt-vs-mkt(block)':>22}"
    print(hdr)
    for hk, r in res["by_horizon"].items():
        c = r["text_vs_mkt_ci90"]
        cb = r["text_vs_mkt_block_ci90"]
        print(
            f"{hk:<8}{r['har']:>8.3f}{r['mkt_only']:>8.3f}{r['fused']:>8.3f}"
            f"{f'[{c[0]:+.3f},{c[1]:+.3f}]':>20}{f'[{cb[0]:+.3f},{cb[1]:+.3f}]':>22}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
