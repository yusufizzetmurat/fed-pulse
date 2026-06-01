# Late-fusion gated model: fixing the text-negative regime fusion

**Goal (not text-alpha):** a good fusion should be able to *ignore* unhelpful text —
fused macro-F1 ≥ market-only, with the learned gate ≈ 0. The served regime fusion failed this:
fused **0.592 < gate-off market 0.608** (text-negative).

## Diagnosis

The served model (`fed_comms_regime` + `gated_fusion`) is already a dense 5385-bar daily panel
with a scalar gate (`mkt` = the same model with text masked off). The drag was **not** data
structure — it was the gate **opening on harmful text**: the SupCon contrastive term trains the
text rep to look label-useful on the train fold, and nothing penalised an open gate, so it
opened and overfit text noise → hurt OOS.

## Fixes (backward-compatible flags on `gated_fusion` / `fed_comms_regime`)

1. **no-contrastive + gate reg:** `supcon_weight=0`, L1 penalty on the gate, gate init ~closed.
2. **structural (residual-logit):** fuse at the output — `pred = market_logits + gate·text_head(zt)`
   — with the market head **directly supervised** (aux CE). gate→0 ⇒ `fused == market_logits`
   by construction.

## Result (100 epochs × 5 seeds-fold walk-forward, vs gate-off market-only)

| arm | gate | Δ(fused−mkt) h1 / h5 / h22 | fused−mkt CI90 |
|---|---:|---|---|
| anchor (current served) | 0.48–0.66 | −0.040 / −0.028 / −0.041 | all significantly < 0 |
| neutral_no_contrastive | 0.12–0.16 | −0.010 / −0.010 / +0.016 | h1/h5 still < 0 |
| **neutral_residual** | **0.01–0.04** | **−0.002 / −0.003 / −0.000** | touch / include 0 |

**Verdict:** the structural fix achieves text-neutrality — fused is statistically indistinguishable
from market-only (drag cut ~20×, from −0.04 to −0.002), and the gate collapses to ≈0.02, i.e. the
model *learns the text is uninformative and ignores it.* The residual −0.002 is negligible (CIs
touch 0) and reflects the gate not being pinned to exactly 0; a larger L1 would pin it.

**Promotion note:** to make the frontend "second opinion" honestly text-neutral (and lift its
displayed macro-F1 from 0.592 to ~market level), run the served regime trainer with
`residual_logits=True, supcon_weight=0, gate_l1_weight=0.1, gate_init_bias=-3, market_aux_weight=0.5`
and regenerate the artifact. The card's story becomes "the model learns to ignore text (gate≈0) and
matches the market baseline" — cleaner than "text drags it down."
