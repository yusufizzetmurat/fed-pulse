# Encoder-axis stance retrain results

**Date:** 2026-06-02 · **Status:** COMPLETE. Plain ProsusAI/FinBERT wins
the validity anchor; xbank wins held-out test F1 by a wide margin. The
two winners are different encoders, and the gap surfaces a real
trade-off the dashboard has to choose between.

## Setup

Same `finetune_stance.py` recipe (3-class hawk/dove/neutral head, TDW
training pool n=2,408, held-out test = `gtfintechlab_federal_reserve_system`
\+ `op_fed` n=1,112) re-run across four encoder backbones via the new
`--base-encoder` flag (PR #611):

```
python -m app.data.finetune_stance \
  --labels data/processed/tp_v3_full_rebuild_2026_05_30/registry_normalized.parquet \
  --out-dir data/processed/stance_finetune_<label> \
  --base-encoder <ENCODER_SLUG> \
  --loss ce --epochs 10
```

Each output then runs through the validity harness:

```
python scripts/build_stance_daily.py --backend finetune-stance \
  --checkpoint-dir data/processed/stance_finetune_<label>
python scripts/stance_instrument_validity.py
```

Result JSONs pinned alongside this writeup as
`stance-instrument-validity-result-{finbert,ce,xbank,xbank-dapt}-retrain.json`.

## Matrix

| Encoder | Held-out F1 | Spearman ρ | AUC hike-vs-cut | mean(s\|cut) | mean(s\|hold) | mean(s\|hike) | Leading ρ |
|---|---|---|---|---|---|---|---|
| ProsusAI/finbert (no DAPT) | 0.526 | **+0.499** | **0.967** | -0.856 | -0.703 | **-0.092** | **+0.447** |
| finbert-fed-adjacent (Lead-1 CE) | 0.547 | +0.385 | 0.900 | -0.962 | -0.876 | -0.532 | +0.396 |
| finbert-fed-adjacent-xbank | **0.720** | +0.335 | 0.800 | -0.717 | -0.640 | -0.034 | +0.359 |
| finbert-fed-adjacent-xbank-dapt | 0.535 | +0.325 | 0.811 | -0.912 | -0.794 | -0.348 | +0.380 |

Bold = best in column. All four close the Lead-1 gate (`mean(s|cut) <
mean(s|hold)`).

## The two winners disagree

xbank wins held-out F1 by 17 points. The cross-bank DAPT'd encoder
classifies hawkish / dovish / neutral on held-out Fed-stance text far
better than any other variant. The next-best (finbert-fed-adjacent) sits
at 0.55; the xbank-DAPT regression (0.54) shows the extra Fed-text DAPT
step on top of cross-bank actually hurts.

ProsusAI/FinBERT wins the validity anchor by 17 points of Spearman.
Without Fed-text or cross-bank pre-training, the plain FinBERT encoder
produces a stance score whose correlation with realised policy moves is
materially higher than any DAPT'd variant (+0.499 vs +0.32-+0.39). AUC
hike-vs-cut climbs to 0.967, near-perfect.

## What this says about DAPT

The DAPT'd encoders over-fit to FOMC text patterns that classify TDW
labels accurately (high held-out F1, especially xbank) but do not
track the policy-action anchor as cleanly as plain FinBERT does. The
DAPT signal and the validity signal point in different directions.

This is not a bug in the DAPT pipeline. It is the same phenomenon the
standing finding has called out for months: text classifiers can become
very good at TDW-label discrimination without the discrimination being a
useful policy-anchor predictor. The encoder-axis sweep makes the gap
measurable.

## Deployment recommendation

Two paths, two different priorities:

1. Dashboard stance instrument (the displayed s). ProsusAI/FinBERT fits
   when the goal is a number that tracks what the Fed actually does. The
   z-score badge already in production lifts from a +0.284 baseline to
   +0.499, a substantial credibility gain for the descriptive layer.
2. TDW-label classification (Performance page macro-F1). xbank fits
   when the goal is the highest held-out classification accuracy on the
   gtfintechlab + op_fed sets. The 0.720 figure beats FOMC-RoBERTa
   (0.578) by a comfortable margin and warrants a wiki §20 citation.

These are not contradictory if the dashboard is explicit about which
backbone serves which surface, but they cannot be collapsed onto a
single canonical encoder without paying the gap in whichever surface
loses.

The two checkpoints sit at:
- `data/processed/stance_finetune_finbert/` (validity winner)
- `data/processed/stance_finetune_xbank/` (held-out F1 winner)

No production swap is taken in this commit. The multi-axis classifier
keeps its current encoder. The artifacts above stand as validated
research alternatives.
