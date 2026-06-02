# NLP baseline bake-off — 2026-06-02 re-run

**Run id:** `execution_20260602T091722Z` · **Mode:** full ·
**Training package:** `canonical` (5-fold expanding walk-forward, official seed set {11, 29, 47, 71, 97}).

Re-run of the Phase 3 NLP baseline matrix tracked in earlier session memory
as "bake-off rerun pending after PR #591/#594 stance label-map fix". Raw
summary pinned alongside this writeup at
[`nlp-baseline-bakeoff-2026-06-02-rerun.json`](./nlp-baseline-bakeoff-2026-06-02-rerun.json).

## Results

| Model | Checkpoint | Macro-F1 (mean ± std over 5 seeds) |
|---|---|---|
| `fomc_roberta` | `ZiweiChen/FinBERT-FOMC` | **0.5082 ± 0.0069** |
| `finbert`      | `ProsusAI/finbert`       | 0.4967 ± 0.0069 |
| `random_class` | sanity baseline          | 0.3331 ± 0.0063 |
| `majority`     | sanity baseline          | 0.1871 ± 0.0034 |
| `bert`         | `bert-base-uncased`      | 0.1839 ± 0.0229 |

## Headline

The post-label-map-fix re-run confirms the wiki §20 ordering: ZiweiChen
FinBERT-FOMC and ProsusAI FinBERT are the two viable text-only zero-shot
classifiers; vanilla BERT collapses to majority-class behaviour; both
sanity baselines (majority, random) land where expected, providing a clear
floor for the meaningful classifiers. Tie-break order from the
benchmark-policy doc (macro-F1 then worst-class F1 then p95 latency) puts
ZiweiChen at the top by ~0.011 macro-F1 over ProsusAI, just outside the
1σ noise band.

## What this resolves

- The wiki §20 "macro-F1 0.687" HAR-tercile headline is unaffected — that's
  a market-only classifier, not in this bake-off.
- The session memory item ("bake-off re-run pending against PR #591/#594
  stance label-map fix") is closed; the canonical baseline numbers are
  the ones above.
- Any downstream comparison (e.g. dashboard's late-fusion macro-F1 0.629
  surface) should reference these baselines, not pre-rerun snapshots.

## Caveat

Only the first checkpoint per `fomc_roberta` model_key (ZiweiChen) was
exercised; the script lists three alternative checkpoints as fallbacks but
does not iterate them separately. If a head-to-head between
`gtfintechlab/FOMC-RoBERTa` and ZiweiChen is needed it has to be a
separate pass with the script's MODEL_SPECS rotated.
