# ADR 0007 — Variant B (chunk-attention pooler) verdict

Status: deferred (does not ship with the v1 reporting pack).
Date: 2026-05-16.
Supersedes: nothing.
Superseded by: nothing.
References:
- `../../fed-pulse.wiki/06_Deep_Learning_Roadmap.md §1.0` — full outcome table.
- `backend/app/data/attention_ablation.py` — the ablation runner that produced the empirical verdict.
- `backend/app/data/chunk_embedding_store.py` / `chunk_embedding_retrieval.py` — the precomputed embedding pipeline Variant B relied on.

## Context

The forecaster reads a fixed-length window of market state plus a single
text-derived scalar feature. The Phase-4 attention-decay ablation evaluated
three architectures for fusing text into that window:

- **Variant A** — sentence-level decay applied to a per-document scalar at
  feature-assembly time. The text channel is one float; the LSTM never
  sees the raw embedding.
- **Variant B** — chunk-level FinBERT embeddings retrieved per anchor date,
  pooled with a scaled-dot-product attention head whose temperature is
  itself learnable, then projected down to the LSTM input width.
- **Variant C** — same architecture as Variant B but with a different
  encoder (for example, the FOMC-RoBERTa checkpoint) to disentangle "the
  pooler does not converge" from "FinBERT is the wrong encoder."

Variant A converges and ships in the v1 reporting pack. The open question
is whether Variant B is worth carrying through to the same reporting pack
alongside it.

## Decision

**Defer Variant B from the v1 reporting pack.**

The Plan-3 ablation grid ran Variant B on the official seed set
(`{11, 29, 47, 71, 97}`) under the standard expanding walk-forward folds.
The outcome was reproducible across folds and seeds: λ_chunk stays at its
zero-init value (drift on the order of 1e-4 over the entire training
schedule), the attention head's softmax collapses to near-uniform weights,
and combined RMSE / directional accuracy do not improve over the
text-free baseline. The diagnosis (data starvation at ~2k labelled
tuples) was confirmed fold-independent in Phase-7 (Plan-13 §1.0). With
a 5-7k labelled corpus the calculus changes; at the current data scale,
adding Variant B to the reporting pack would publish a no-op result that
is more confusing than informative.

## Consequences

- The reporting pack carries Variant A only as the text-fusion architecture.
- The chunk-embedding pipeline (`chunk_embedding_store.py`,
  `chunk_embedding_retrieval.py`) stays in the repository. Variant B
  remains runnable on demand and Variant C reuses the same retrieval seam.
- Phase-6 corpus expansion (pseudo-labelling, gtfintechlab 3k addition)
  is the prerequisite for revisiting Variant B. When the labelled corpus
  crosses ~5k tuples this ADR should be reopened with a fresh ablation.
- Variant C ships alongside Variant A under the same data scale because
  it answers a different question (encoder-bias) that is meaningful even
  at 2k tuples.
