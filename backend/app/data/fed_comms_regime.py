"""Vol-regime classification: the fair-target test for Fed text.

Magnitude-RV is the worst target for text (HAR owns it). This reframes to a
3-class **regime** target — terciles of forward realized variance — scored by
**macro-F1**, where text's coarse hawkish/dovish/uncertainty signal has a fair
shot and a coarse metric can reveal value an R² washes out.

Same leak-safe gated-fusion machinery as the regression trainer, with a
cross-entropy head. The **canonical config** is the text-neutral one:
output-level **residual-logit fusion** (``pred = market_logits + gate·text``)
with the market head directly supervised and an L1 penalty + closed gate init,
so the gate collapses to ≈0 and the fused forecast cannot underperform its own
market-only path — the model is free to use text and learns to ignore it.
The earlier contrastive variant (``supcon_weight>0``, representation-space
fusion) is retained as an optional research arm: it made text *look*
label-useful on the train fold and opened the gate, which dragged the model OOS
(fused 0.592 < market 0.608). Tercile thresholds are fit on the train fold only.

Reported per horizon: macro-F1 for fused vs the gate-off market-only path, vs a
majority-class floor, vs a HAR-tercile baseline (classify by HAR's predicted-RV
bucket) — with a moving-block bootstrap CI on the fused−market F1 gap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from app.data.dense_daily_dataset import walk_forward_splits
from app.data.dense_forecast_train import _fit_predict_ols
from app.data.fed_comms_dataset import DEFAULT_HORIZONS, MEASURES
from app.data.fed_comms_train import DEFAULT_FUSION_DIR, _assemble

_N_CLASSES = 3


def _labels(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Digitize values into regime classes [0, n_classes) by tercile thresholds."""

    return cast(np.ndarray, np.digitize(values, thresholds).astype(np.int64))


def _macro_f1(true: np.ndarray, pred: np.ndarray, n_classes: int = _N_CLASSES) -> float:
    """Unweighted mean per-class F1 (no sklearn dependency at call sites)."""

    f1s = []
    for c in range(n_classes):
        tp = float(np.sum((pred == c) & (true == c)))
        fp = float(np.sum((pred == c) & (true != c)))
        fn = float(np.sum((pred != c) & (true == c)))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def _block_f1_gap_ci(
    true: np.ndarray, a: np.ndarray, b: np.ndarray, *, block: int, seed: int, n_boot: int = 1000
) -> tuple[float, float]:
    """Moving-block bootstrap CI for macro-F1(a) − macro-F1(b) (a,b = predictions)."""

    n = len(true)
    if n <= block:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    gaps = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        gaps.append(_macro_f1(true[idx], a[idx]) - _macro_f1(true[idx], b[idx]))
    return (float(np.percentile(gaps, 5)), float(np.percentile(gaps, 95)))


def _train_regime_fold(
    data: dict[str, np.ndarray],
    tr: np.ndarray,
    te: np.ndarray,
    k: int,
    *,
    seed: int,
    epochs: int,
    supcon_weight: float = 0.0,
    warmup: int = 8,
    patience: int = 12,
    gate_l1_weight: float = 0.1,
    gate_init_bias: float = -3.0,
    residual_logits: bool = True,
    market_aux_weight: float = 0.5,
) -> dict[str, np.ndarray]:
    """Train a regime classifier for horizon index k; return test predictions + labels."""

    import torch

    from app.data.gated_fusion import build_model, fusion_clf_loss
    from app.determinism import enable_deterministic_mode

    enable_deterministic_mode(seed)
    har = data["har"]
    vals = data["targets"][:, k]
    thr = np.quantile(vals[tr], [1 / 3, 2 / 3])  # tercile thresholds from TRAIN only
    y = _labels(vals, thr)
    # HAR-tercile baseline: bucket HAR's predicted RV by the same thresholds
    har_pred_te = _fit_predict_ols(har[tr], vals[tr], har[te])
    har_regime = _labels(har_pred_te, thr)

    n = len(tr)
    n_val = max(1, n // 5)
    core, val = slice(0, n - n_val), slice(n - n_val, n)
    mf = data["market_feat"][tr]
    mfm, mfs = mf[core].mean(0), mf[core].std(0)
    mfs = np.where(mfs > 0, mfs, 1.0)
    mf_std, mf_te = (mf - mfm) / mfs, (data["market_feat"][te] - mfm) / mfs
    emb, mask_tr = data["text_emb"][tr], data["text_mask"][tr]
    present = mask_tr[core] > 0
    ref = emb[core][present] if present.any() else emb[core]
    em, es = ref.mean(0), ref.std(0)
    es = np.where(es > 0, es, 1.0)
    emb_std = ((emb - em) / es) * mask_tr[:, None]
    emb_te = ((data["text_emb"][te] - em) / es) * data["text_mask"][te][:, None]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        emb.shape[1],
        mf.shape[1],
        _N_CLASSES,
        gate_init_bias=gate_init_bias,
        residual_logits=residual_logits,
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def tt(a: np.ndarray, long: bool = False) -> "torch.Tensor":
        return torch.tensor(a, dtype=torch.long if long else torch.float32, device=dev)

    emb_t, mf_t, mask_t, y_t = tt(emb_std), tt(mf_std), tt(mask_tr), tt(y[tr], long=True)
    core_pos = np.arange(0, n - n_val)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        lam = supcon_weight * min(1.0, ep / max(warmup, 1))  # SupCon λ warmup from 0
        model.train()
        order = rng.permutation(len(core_pos))
        for s in range(0, len(order), 256):
            b = core_pos[order[s : s + 256]]
            opt.zero_grad()
            batch = {
                "text_emb": emb_t[b],
                "market_feat": mf_t[b],
                "text_mask": mask_t[b],
                "labels": y_t[b],
            }
            loss = fusion_clf_loss(
                model,
                batch,
                supcon_weight=lam,
                gate_l1_weight=gate_l1_weight,
                market_aux_weight=market_aux_weight,
            )["loss"]
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(emb_t[val], mf_t[val], mask_t[val])["pred"].argmax(1).cpu().numpy()
        vf1 = _macro_f1(y[tr][val], vp)
        if vf1 > best + 1e-6:
            best, bad = vf1, 0
            best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
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
        mkt = model(tem, mfe, torch.zeros_like(msk))
    return {
        "fused": fused["pred"].argmax(1).cpu().numpy(),
        "mkt": mkt["pred"].argmax(1).cpu().numpy(),
        "har": har_regime,
        "true": y[te],
        "gate": fused["gate"].cpu().numpy(),
        "mask": data["text_mask"][te],
    }


def run(
    fusion_dir: Path | str = DEFAULT_FUSION_DIR,
    *,
    market_cache_dir: Path | str,
    corpus_path: Path | str,
    emb_path: Path | str,
    seed: int = 11,
    epochs: int = 100,
    n_folds: int = 5,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    measure: str = "rv",
    supcon_weight: float = 0.0,
    gate_l1_weight: float = 0.1,
    gate_init_bias: float = -3.0,
    residual_logits: bool = True,
    market_aux_weight: float = 0.5,
) -> dict[str, Any]:
    import pandas as pd

    daily = pd.read_parquet(Path(fusion_dir) / "daily_fusion.parquet")
    corpus = pd.read_parquet(corpus_path)
    emb_df = pd.read_parquet(emb_path)
    data = _assemble(daily, corpus, emb_df, market_cache_dir, horizons, measure=measure)
    idx_all = np.where(data["valid"])[0]
    folds = walk_forward_splits(len(idx_all), n_folds=n_folds, embargo=max(horizons) + 1)

    results: dict[str, Any] = {"n_eval": 0, "measure": measure, "by_horizon": {}}
    for k, h in enumerate(horizons):
        pools: dict[str, list[np.ndarray]] = {
            kk: [] for kk in ("fused", "mkt", "har", "true", "gate", "mask")
        }
        for tr_l, te_l in folds:
            tr, te = idx_all[np.array(tr_l)], idx_all[np.array(te_l)]
            out = _train_regime_fold(
                data,
                tr,
                te,
                k,
                seed=seed,
                epochs=epochs,
                supcon_weight=supcon_weight,
                gate_l1_weight=gate_l1_weight,
                gate_init_bias=gate_init_bias,
                residual_logits=residual_logits,
                market_aux_weight=market_aux_weight,
            )
            for kk in pools:
                pools[kk].append(out[kk])
        cat = {kk: np.concatenate(v) for kk, v in pools.items()}
        results["n_eval"] = int(len(cat["true"]))
        maj = int(np.bincount(cat["true"], minlength=_N_CLASSES).argmax())
        row = {
            "majority_f1": _macro_f1(cat["true"], np.full_like(cat["true"], maj)),
            "har_f1": _macro_f1(cat["true"], cat["har"]),
            "mkt_f1": _macro_f1(cat["true"], cat["mkt"]),
            "fused_f1": _macro_f1(cat["true"], cat["fused"]),
            "text_vs_mkt_f1_block_ci90": list(
                _block_f1_gap_ci(cat["true"], cat["fused"], cat["mkt"], block=max(h, 1), seed=seed)
            ),
        }
        active = cat["mask"] > 0
        if active.sum() > 5:
            row["fused_f1_active"] = _macro_f1(cat["true"][active], cat["fused"][active])
            row["mkt_f1_active"] = _macro_f1(cat["true"][active], cat["mkt"][active])
            row["gate_mean_active"] = float(cat["gate"][active].mean())
            row["gate_max_active"] = float(cat["gate"][active].max())
            row["gate_p95_active"] = float(np.quantile(cat["gate"][active], 0.95))
        results["by_horizon"][f"h{h}"] = row
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Vol-regime classification: text vs market.")
    parser.add_argument("--fusion-dir", type=Path, default=DEFAULT_FUSION_DIR)
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--emb-path", type=Path, required=True)
    parser.add_argument("--market-cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target", default="rv", choices=list(MEASURES))
    parser.add_argument("--epochs", type=int, default=100)
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
    (args.out_dir / "regime_bakeoff.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"target={res['measure']} regime  n_eval={res['n_eval']}  (macro-F1; floor ≈ 0.33)")
    print(f"{'horizon':<8}{'major':>8}{'HAR':>8}{'mkt':>8}{'fused':>8}{'txt-vs-mkt_F1_block':>24}")
    for hk, r in res["by_horizon"].items():
        c = r["text_vs_mkt_f1_block_ci90"]
        print(
            f"{hk:<8}{r['majority_f1']:>8.3f}{r['har_f1']:>8.3f}{r['mkt_f1']:>8.3f}"
            f"{r['fused_f1']:>8.3f}{f'[{c[0]:+.3f},{c[1]:+.3f}]':>24}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
