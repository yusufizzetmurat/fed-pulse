"""Text-neutral gated fusion: make the regime fusion stop losing to market-only.

The served regime fusion scores fused 0.592 < gate-off market 0.608 — the gate
opens on text-days and the text contribution hurts OOS (SupCon makes text look
label-useful on train; nothing penalises an open gate). This runs two arms on the
SAME dense daily frame + walk-forward protocol:

  anchor   : supcon_weight=0.1, no gate L1, gate open-able   (reproduces the drag)
  neutral  : supcon_weight=0.0, gate L1 + gate init ~closed  (text-neutral target)

Goal (pre-registered): neutral fused macro-F1 >= market-only, with the learned
gate ~0 (the model learns to ignore text). Beating HAR is NOT the goal.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app.data.fed_comms_regime import run

FUSION_DIR = "data/processed/fed_comms_fusion"
MARKET_CACHE = "data/processed/tp_v3_full_rebuild_2026_05_30/_market_cache"
CORPUS = "data/external/fed_comms/fed_communications.parquet"
EMB = "data/processed/fed_comms_fusion/corpus_embeddings.parquet"
OUT = "data/artifacts/late_fusion_gated_neutral/result.json"
EPOCHS = int(os.environ.get("LF_NEUTRAL_EPOCHS", "100"))


def _arm(label: str, **overrides: Any) -> dict[str, Any]:
    res = run(
        FUSION_DIR,
        market_cache_dir=MARKET_CACHE,
        corpus_path=CORPUS,
        emb_path=EMB,
        seed=11,
        epochs=EPOCHS,
        **overrides,
    )
    res["arm"] = label
    return res


def main() -> None:
    arms = {
        "anchor_supcon": _arm(
            "anchor_supcon", supcon_weight=0.1, gate_l1_weight=0.0, gate_init_bias=0.0
        ),
        "neutral_no_contrastive": _arm(
            "neutral_no_contrastive",
            supcon_weight=0.0,
            gate_l1_weight=0.1,
            gate_init_bias=-3.0,
        ),
        # Structural guarantee: output-level residual fusion + directly-supervised
        # market head → gate→0 collapses to the market head exactly.
        "neutral_residual": _arm(
            "neutral_residual",
            supcon_weight=0.0,
            gate_l1_weight=0.1,
            gate_init_bias=-3.0,
            residual_logits=True,
            market_aux_weight=0.5,
        ),
    }
    out: dict[str, Any] = {"epochs": EPOCHS, "arms": arms, "verdict": {}}
    print(f"\n{'arm':<24}{'h':>5}{'mkt_f1':>9}{'fused_f1':>10}{'Δ(fus-mkt)':>12}{'gate':>8}{'neutral?':>10}")
    for name, res in arms.items():
        for hk, r in res["by_horizon"].items():
            d = r["fused_f1"] - r["mkt_f1"]
            g = r.get("gate_mean_active")
            neutral = d >= -0.002  # fused not materially below market-only
            print(
                f"{name:<24}{hk:>5}{r['mkt_f1']:>9.3f}{r['fused_f1']:>10.3f}"
                f"{d:>+12.3f}{(g if g is not None else float('nan')):>8.3f}{str(neutral):>10}"
            )
            out["verdict"][f"{name}:{hk}"] = {
                "mkt_f1": r["mkt_f1"], "fused_f1": r["fused_f1"], "delta": d,
                "gate_mean_active": g, "text_neutral": neutral,
            }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
