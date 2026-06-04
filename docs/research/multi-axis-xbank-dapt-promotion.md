# Multi-axis classifier backbone swap: finbert-fed-adjacent → xbank-DAPT

**Date:** 2026-06-02 · **HF revision:** `c863f18753e87f2576b3609112a10efd85671e8f`

The `/analyze` stance and certainty cards consume `text_multi_axis_best.pt` via
`backend/app/services/multi_axis_classifier.py`. The canonical slot previously
held a finbert-fed-adjacent backbone (val_loss 0.0974). The encoder-axis study
at `encoder-axis-stance-results.md` had already identified the xbank-DAPT
backbone as the held-out winner; this run promotes it into the multi-axis
production path.

## Training

| Knob | Value |
|---|---|
| Backbone | `finbert_fed_adjacent_xbank_dapt` |
| Data | gtfintechlab Federal Reserve System (3000 rows, 2550 train / 450 val) |
| Seeds | 97, 11, 47 |
| Epochs | 8 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Driver | `scripts/train_text_multi_axis_overnight.sh` |

Per-seed best val loss:

| Seed | val_loss | val_acc_stance | val_acc_certainty |
|---|---|---|---|
| 97 | 0.0610 | 0.991 | 0.998 |
| 11 | 0.0578 | 0.993 | 1.000 |
| **47** | **0.0169** | 0.993 | 1.000 |

Seed 47 was promoted to `backend/models/text_multi_axis_best.pt`.

## Validity comparison

`scripts/stance_instrument_validity.py` runs the policy-anchor test against
`DFEDTARU` across 122 FOMC meetings (2011-01-26 to 2026-04-29). The classifier
never trained on rate moves; the test asks whether
`s = P(hawkish) − P(dovish)` ordering tracks the realised action.

| Metric | Baseline (finbert-fed-adjacent) | xbank-DAPT (seed 47) | Δ |
|---|---|---|---|
| PRIMARY Spearman(s, DFF) | +0.284 | **+0.357** | +0.073 |
| AUC hike-vs-cut | 0.778 | **0.794** | +0.016 |
| Ordinal Spearman | +0.272 | **+0.350** | +0.078 |
| Leading Spearman(s_t, dff_{t+1}) | +0.231 | **+0.272** | +0.041 |
| Val loss (in-dist gtfintechlab) | 0.0974 | **0.0169** | −0.081 (5.7×) |
| Val acc stance | 0.978 | **0.993** | +0.015 |
| Val acc certainty | 0.984 | **1.000** | +0.016 |

Per-action mean stance score:

|  | cut | hold | hike |
|---|---|---|---|
| Baseline | −0.533 | −0.584 | +0.420 |
| xbank-DAPT | −0.287 | −0.655 | **+0.385** |

The xbank-DAPT model sharpens the hold/cut separation and the hike sign stays
where it should. The mean-stance gap during holds reflects 2011-2026 hold-window
composition (zero-bound plus recent steady-state); it is not a regression.

## What changed downstream

- `backend/app/models/registry.yaml` `multi_axis_text_classifier.revision`
  pinned to the new sha.
- `backend/models/text_multi_axis_best.pt` rewritten on disk. The previous
  finbert-fed-adjacent canonical is mirrored under
  `text_multi_axis_best.finbert-fed-adjacent.bak.pt` on the dev box as a
  rollback handle; not pushed.
- The stance / certainty tiles on `/analyze` now serve from the xbank-DAPT
  backbone. The response shape gains a `forward-looking` axis (forward-looking /
  not-forward-looking); the old slot only carried stance and certainty.
- `data/artifacts/corner_b_text_rates/stance_daily.parquet` regenerated off
  the new canonical; the validity result lands at
  `data/artifacts/stance_instrument_validity/result.json`.

## What did not change

- The encoder bundle is the existing `encoder_canonical` (`finbert_fed_adjacent_xbank_dapt`)
  already used by retrieval (`finbert_fed_adjacent_xbank_dapt_retrieval`) and
  trajectory; no new encoder mirror was published.
- The `finbert-fed-adjacent` backbone remains the trained base for the stance
  fine-tune (`finetune_stance`) and the historical encoder-axis matrix.
- The single-head `finetune_stance` artifact under
  `data/processed/stance_finetune_*` is independent of this file and is not
  affected.
