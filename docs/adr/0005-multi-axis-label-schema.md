# 0005 — Multi-axis label schema

**Status:** accepted
**Date:** 2026-05-13

## Context

Three-class stance (hawkish / dovish / neutral) collapses the economic signal in FOMC text into a single ordinal axis. Markets do not respond to "the message is hawkish" alone — they respond differently to a hawkish *forward-guidance* shock vs a hawkish *current-stance* shock (GSS), to an uncertain hawkish message vs a confident one, and to text whose topic emphasis is inflation vs employment vs financial stability. A model with only stance has no way to express these.

## Decision

Labels carry four axes:

1. `stance` — `hawkish | dovish | neutral`. Primary axis, sourced from the Trillion Dollar Words hand-labels and downstream encoder fine-tunes.
2. `factor` — regression target in `[-1, 1]`. Loads on the Gürkaynak-Sack-Swanson decomposition: positive values weight forward-guidance shocks, negative values weight target-rate shocks.
3. `certainty` — regression target in `[0, 1]`. Derived from modal-verb density and hedge-word counts (Lucca-Trebbi style).
4. `topic` — multiclass over `inflation | employment | financial_stability | growth`. Derived from Hansen-McMahon topic shares.

Axes 2–4 are nullable per row. The TDW labelled set carries axis 1 only; rows ingested with GSS-aligned dates fill axis 2; the rest are filled by feature-derived heuristics or left null. Pandera validates the schema on parquet write.

## Consequences

- The forecaster's multi-task head has four prediction heads with weighted loss; axes with nulls contribute zero loss for those rows. Multi-task acts as auxiliary regularisation on a tiny labelled n.
- The Phase 6 reframing of the corpus expansion drops LLM-as-judge pseudo-labelling (see [0006](0006-kill-llm-as-judge-pseudo-labeling.md)) in favour of external corpora that carry axes 2 or 4 directly.
- Source-type-stratified analysis ([0004](0004-source-type-schema.md)) reports per-axis macro-F1 per `source_type`, not just stance.
