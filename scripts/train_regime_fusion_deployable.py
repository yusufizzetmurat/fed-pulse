"""Train ONE deployable text-neutral gated-fusion regime model and bundle it.

The walk-forward bake-off (`fed_comms_regime.run`) trains a fresh model per fold
and discards it — great for honest eval, but it leaves no artifact to load or
build on. This trains a single instance on ALL valid rows with the canonical
neutral-residual config, freezes everything inference needs (market + text
normalizers, per-horizon tercile thresholds, HAR-OLS baseline coeffs), and saves
a self-contained bundle per horizon.

NOTE on numbers: these weights are an all-data fit (no held-out test), so do NOT
read a test-F1 off them. The honest macro-F1 is the walk-forward eval in
`data/artifacts/late_fusion_gated_neutral/result.json` (neutral_residual:
0.629 / 0.634 / 0.496 at h1/h5/h22). A small chronological tail is held out only
for early stopping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.data.fed_comms_dataset import DEFAULT_HORIZONS
from app.data.fed_comms_regime import _labels, _macro_f1
from app.data.fed_comms_train import _assemble

FUSION_DIR = "data/processed/fed_comms_fusion"
MARKET_CACHE = "data/processed/tp_v3_full_rebuild_2026_05_30/_market_cache"
CORPUS = "data/external/fed_comms/fed_communications.parquet"
EMB = "data/processed/fed_comms_fusion/corpus_embeddings.parquet"
OUT_DIR = Path("data/artifacts/regime_fusion_deployable")

SEED = 11
EPOCHS = 100
WARMUP = 8
PATIENCE = 12
# canonical text-neutral config (matches the promoted run() defaults)
CFG = dict(
    supcon_weight=0.0,
    gate_l1_weight=0.1,
    gate_init_bias=-3.0,
    residual_logits=True,
    market_aux_weight=0.5,
)


def _train_one_horizon(data: dict[str, np.ndarray], k: int, h: int) -> dict[str, Any]:
    import torch

    from app.data.gated_fusion import build_model, fusion_clf_loss
    from app.determinism import enable_deterministic_mode

    enable_deterministic_mode(SEED)
    idx = np.where(data["valid"])[0]
    har = data["har"][idx]
    vals = data["targets"][idx, k]
    mf_all = data["market_feat"][idx]
    emb_all = data["text_emb"][idx]
    mask_all = data["text_mask"][idx]

    n = len(idx)
    n_val = max(1, n // 5)
    core, val = slice(0, n - n_val), slice(n - n_val, n)

    # frozen state — fit on the train-core only (everything inference needs)
    thr = np.quantile(vals[core], [1 / 3, 2 / 3])
    y = _labels(vals, thr)
    mfm, mfs = mf_all[core].mean(0), mf_all[core].std(0)
    mfs = np.where(mfs > 0, mfs, 1.0)
    present = mask_all[core] > 0
    ref = emb_all[core][present] if present.any() else emb_all[core]
    em, es = ref.mean(0), ref.std(0)
    es = np.where(es > 0, es, 1.0)
    _a = np.column_stack([np.ones(len(har[core])), har[core]])
    har_coef, *_ = np.linalg.lstsq(_a, vals[core], rcond=None)  # [intercept, w_daily, w_weekly, w_monthly]

    mf_std = (mf_all - mfm) / mfs
    emb_std = ((emb_all - em) / es) * mask_all[:, None]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        emb_all.shape[1], mf_all.shape[1], 3,
        gate_init_bias=CFG["gate_init_bias"], residual_logits=CFG["residual_logits"],
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def tt(a: np.ndarray, long: bool = False) -> "torch.Tensor":
        return torch.tensor(a, dtype=torch.long if long else torch.float32, device=dev)

    emb_t, mf_t, mask_t, y_t = tt(emb_std), tt(mf_std), tt(mask_all), tt(y, long=True)
    core_pos = np.arange(0, n - n_val)
    rng = np.random.default_rng(SEED)
    best, best_state, bad = -1.0, None, 0
    for ep in range(EPOCHS):
        lam = CFG["supcon_weight"] * min(1.0, ep / max(WARMUP, 1))
        model.train()
        order = rng.permutation(len(core_pos))
        for s in range(0, len(order), 256):
            b = core_pos[order[s : s + 256]]
            opt.zero_grad()
            batch = {
                "text_emb": emb_t[b], "market_feat": mf_t[b],
                "text_mask": mask_t[b], "labels": y_t[b],
            }
            loss = fusion_clf_loss(
                model, batch, supcon_weight=lam,
                gate_l1_weight=CFG["gate_l1_weight"], market_aux_weight=CFG["market_aux_weight"],
            )["loss"]
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(emb_t[val], mf_t[val], mask_t[val])["pred"].argmax(1).cpu().numpy()
        vf1 = _macro_f1(y[val], vp)
        if vf1 > best + 1e-6:
            best, bad = vf1, 0
            best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        gate_active = float(
            model(emb_t, mf_t, mask_t)["gate"][mask_all > 0].mean().cpu().item()
        )

    return {
        "horizon": h,
        "state_dict": {kk: v.cpu() for kk, v in model.state_dict().items()},
        "thresholds": thr.tolist(),
        "market_mean": mfm.tolist(),
        "market_std": mfs.tolist(),
        "text_emb_mean": em.tolist(),
        "text_emb_std": es.tolist(),
        "har_ols_coef": np.asarray(har_coef).tolist(),
        "val_macro_f1_early_stop": best,
        "gate_mean_active": gate_active,
        "n_train_core": int(n - n_val),
        "n_val_tail": int(n_val),
    }


def main() -> None:
    import pandas as pd
    import torch

    daily = pd.read_parquet(Path(FUSION_DIR) / "daily_fusion.parquet")
    corpus = pd.read_parquet(CORPUS)
    emb_df = pd.read_parquet(EMB)
    data = _assemble(daily, corpus, emb_df, MARKET_CACHE, DEFAULT_HORIZONS, measure="rv")

    horizons = list(DEFAULT_HORIZONS)
    per_h = {f"h{h}": _train_one_horizon(data, k, h) for k, h in enumerate(horizons)}

    d_text = int(data["text_emb"].shape[1])
    d_market = int(data["market_feat"].shape[1])
    bundle = {
        "schema_version": 1,
        "model": "GatedFusionForecaster (text-neutral residual fusion)",
        "config": {**CFG, "d_text": d_text, "d_market": d_market, "n_classes": 3,
                   "seed": SEED, "epochs": EPOCHS, "horizons": horizons},
        "encoder": "FOMC-RoBERTa CLS pool (corpus_embeddings.parquet)",
        "market_feature_order": [
            "rv_daily", "rv_weekly", "rv_monthly", "log_iv", "vix_chg5", "vrp",
            "tnx", "slope", "days_since_stmt", "days_to_stmt",
            "surprise_level", "surprise_path", "surprise_info",
        ][:d_market],
        "per_horizon": per_h,
        "eval_note": "All-data fit; honest macro-F1 is the walk-forward bake-off "
                     "(neutral_residual 0.629/0.634/0.496 h1/h5/h22). val_macro_f1_early_stop "
                     "is a tail split for early stopping only, NOT a test score.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, OUT_DIR / "regime_fusion_model.pt")
    meta = {k: v for k, v in bundle.items() if k != "per_horizon"}
    meta["per_horizon_summary"] = {
        hk: {"val_macro_f1_early_stop": round(r["val_macro_f1_early_stop"], 4),
             "gate_mean_active": round(r["gate_mean_active"], 4),
             "n_train_core": r["n_train_core"]}
        for hk, r in per_h.items()
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["per_horizon_summary"], indent=2))
    print(f"saved -> {OUT_DIR}/regime_fusion_model.pt")


if __name__ == "__main__":
    main()
