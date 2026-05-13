# 0006 — Abandon LLM-as-judge pseudo-labelling

**Status:** accepted
**Date:** 2026-05-13

## Context

Phase 4 stood up an LLM-as-judge pipeline that scored every row of the 9.7k unlabelled scraped FOMC corpus with Gemini 2.5 Flash, then proposed gating rules so the highest-confidence rows would be promoted to training pseudo-labels. The intent was the *Trillion Dollar Words*-style pipeline cited in the project's literature anchor.

Two pieces of evidence settled against this:

1. The 10-row precision audit returned Cohen's κ = 0.00 versus the human labels. The teacher's confidence had no usable signal at the document-truncation length the corpus enforces (256 tokens of FOMC minutes is boilerplate). The required ≥0.90 precision-gate was nowhere near satisfied.
2. The user's advising-faculty position is that LLM-as-judge is acceptable for *metric labelling* (auto-evaluator scoring a model output) but not for *training labels* on supervised tasks — the gradient signal becomes self-referential and bakes the judge's biases into the student.

## Decision

Abandon LLM-as-judge for training-label assignment. Repurpose the 9.7k unlabelled Fed-adjacent corpus as **continued-pretraining material** for FinBERT under a masked-LM objective (Araci-style domain adaptation). Produces a `FinBERT-FedAdjacent` checkpoint pinned by SHA in `backend/app/models/registry.yaml`. The checkpoint is then fine-tuned on **external high-quality labelled corpora** that we ingest through the `BaseSourceScraper` interface.

Pseudo-labels are not used as fine-tune targets anywhere in the v2 architecture.

## Consequences

- `app/data/llm_judge.py`, `app/data/pseudo_labeling.py`, `app/data/llm_embedding_store.py` stay as code for the audit trail (so the κ=0.00 finding is reproducible) but are removed from the v2 fine-tuning path.
- Issues #37 (LLM-as-judge augmentation) and #41 (precision-gate audit) are closed as won't-fix.
- The corpus expansion battery now sources from peer-reviewed replication packages (TDW, GSS factor decomposition, Aruoba-Drechsel, Cieslak-Schrimpf, Hansen-McMahon, Lucca-Trebbi) catalogued by `scripts/inventory_external_corpora.py`. Sample-weight provenance routing gives peer-reviewed rows weight 1.0 and scraped-only rows weight 0.0 (excluded from fine-tune loss).

## Why this is recorded

This decision reverses a previously approved design that consumed two PRs and ~2k lines. Recording it as an ADR makes the reasoning durable for thesis defence and prevents a future contributor from re-introducing pseudo-labels under the same justification.
