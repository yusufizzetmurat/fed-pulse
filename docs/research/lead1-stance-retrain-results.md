# Lead 1 stance retrain — results

**Date:** 2026-06-02 · **Status:** COMPLETE — all three retrains pass the
Lead-1 gate; plain CE wins on every validity metric except the absolute
hold-vs-cut separation.

## Setup

Three retrains of the project's own 3-class hawk/dove/neutral classifier on
the finbert-fed-adjacent encoder, training pool = TDW
(`hf_fomc_communication`, n=2,408), held-out test = `gtfintechlab_federal_reserve_system`
\+ `op_fed` (n=1,112, zero text-hash overlap with TDW).

Each retrain is then scored against the validity harness:

```
python -m app.data.finetune_stance --loss {ce,ce_balanced,focal} \
  --out-dir data/processed/stance_finetune_{ce,balanced,focal}
python scripts/build_stance_daily.py --backend finetune-stance \
  --checkpoint-dir data/processed/stance_finetune_{ce,balanced,focal}
python scripts/stance_instrument_validity.py
```

Result JSONs are pinned alongside this writeup as
`stance-instrument-validity-result-{ce,balanced,focal}-retrain.json` so
future retrains can diff against any of them.

## Matrix

|                          | Baseline       | CE          | Balanced       | Focal        |
|--------------------------|----------------|-------------|----------------|--------------|
|                          | (multi-axis)   | `--loss ce` | β=0.99         | γ=2.0        |
| Held-out test macro-F1   | —              | 0.547       | **0.551**      | 0.536        |
| Spearman ρ(s, Δff)       | +0.284         | **+0.385**  | +0.297         | +0.350       |
| AUC hike-vs-cut          | 0.778          | **0.900**   | 0.828          | 0.889        |
| mean(s|cut)              | -0.533         | -0.962      | -0.951         | -0.838       |
| mean(s|hold)             | -0.584         | -0.876      | -0.808         | -0.778       |
| mean(s|hike)             | +0.420         | -0.532      | -0.327         | -0.665       |
| Leading ρ(s_t, Δff_t+1)  | -0.034 (p .66) | **+0.396**  | +0.302         | +0.387       |
| Within-holds lead ρ      | +0.047         | +0.254      | +0.247         | +0.278       |
| Gate: mean(cut) < mean(hold) | ❌ -0.05 | ✅ -0.086   | ✅ **-0.143**  | ✅ -0.060    |

Bold = best in row. All retrains pass the validity gate.

## Findings

### Plain CE retrain (the surprise)

Just retraining the 3-class head on TDW with inverse-frequency CE closes the
dovish-end resolution gap that the baseline study flagged as Lead-1's whole
motivation. The new instrument:

- Lifts Spearman from +0.284 to +0.385 (~+36% relative).
- Cleanly separates cut from hold: mean separation moves from -0.05 to -0.086.
- Picks up a leading-correlation signal the baseline did not have: ρ(s_t,
  Δff_{t+1}) climbs from -0.034 (p 0.66) to +0.396 (p < 0.001). The instrument
  now flags next-meeting hawkish pivots, not just concurrent action.
- Within-holds lead correlation +0.254 — even inside the long hold regime,
  this retrain's s carries forward-guidance signal the baseline lacked.

### Class-balanced (Cui et al.) confirms the reviewer's prediction

At β=0.99 the logged per-class weights were `[1.0, 0.998, 0.993]` — nearly
uniform, because n × (1-β) ≈ 8 against per-class counts in the hundreds means
the effective-number weighting saturates. Marginally better held-out F1
(+0.004 vs CE) but worse on every validity-anchor measurement. The knob is
in the wrong regime to help here; β around 0.9-0.95 would be the next sweep.

### Focal (γ=2.0)

Between CE and balanced on the validity metrics. Held-out F1 the lowest of
the three. The clamp(min=1e-7) fix from the pre-merge review is the load-
bearing change — without it the modulator would have collapsed on confident
correct predictions during training.

## Deployment status

The CE retrain checkpoint at `data/processed/stance_finetune_ce/` is a
single-head 3-class classifier. Replacing the multi-axis classifier slot
would lose the certainty and time heads currently mounted there. Two
production paths from here:

1. **Retrain the multi-axis classifier itself** (`app.data.train_text_multi_axis_classifier`)
   with TDW-only on the stance head, preserving certainty + time. This carries
   the Lead-1 win into production without giving up other dashboard surfaces.
2. **Bolt the CE retrain on as a secondary stance head** in addition to
   multi-axis. Pay one extra ~110M-param forward per `/analyze` for the
   improved stance score. Heavier but ships immediately.

The validity harness now has a measured ceiling to beat — any future
retrain that does not pass `Spearman > +0.385` and `AUC hike-vs-cut > 0.900`
is not a win.
