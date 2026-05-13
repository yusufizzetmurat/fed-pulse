# 0002 — Elapsed-time decay attention

**Status:** accepted
**Date:** 2026-04-05

## Context

Each market row in the LSTM sequence has an `elapsed_time` value: days between the row's date and the most recent FOMC document. Information from FOMC text loses predictive value as elapsed time grows — forward guidance issued three weeks ago carries less weight on today's price than the same guidance issued yesterday. A naïve LSTM that consumes the sentiment scalar at every timestep treats far-from-FOMC and near-FOMC days equally.

## Decision

Apply `exp(-λ · |elapsed_time|)` decay to the text channel before the LSTM consumes it. `λ` is a learnable scalar parameterised through `softplus(raw_λ)` so it stays non-negative. The same machinery is later extended to `ChunkAttentionPooler` (Variant B) where the decay multiplies *values* before the softmax-weighted sum.

## Consequences

- Phase 4 ablations show Variant A (this decay) beats the no-decay baseline by ~8% on `wf_fold_3` combined RMSE.
- Phase 4 / v2 architecture promotes the same decay primitive to operate over real chunk embeddings instead of a pooled scalar — the math is unchanged, the input dimensionality grows.
- λ-sensitivity sweep against the official seed set is a Phase 5 deliverable; `λ_init = 1.5` is the current default.
- Attention weights and λ values are surfaced to the dashboard for the XAI panel.
