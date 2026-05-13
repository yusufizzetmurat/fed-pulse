# 0004 — Source-type schema

**Status:** accepted
**Date:** 2026-05-05

## Context

The labelled corpus started as FOMC statements + minutes only. Corpus expansion brings in Fed-adjacent material: chair and governor speeches, Congressional testimony, FOMC press conference transcripts, Beige Book entries, NY Fed Liberty Street Economics posts, regional research. These differ in release cadence, audience, and discourse style, so models trained on FOMC text alone may not transfer cleanly. The schema needs an explicit field naming the source so cross-source stratified analysis is possible.

## Decision

Every normalized row carries `source_type` from a closed vocabulary: `fomc_statement`, `fomc_minutes`, `fomc_press_conference`, `chair_speech`, `governor_speech`, `congressional_testimony`, `beige_book`, `regional_research`, `ny_fed_liberty_street`. The vocabulary is owned by `docs/data-and-training-contracts.md`. Adding a new source requires both a code change (an adapter under `backend/app/data/sources/`) and a vocabulary update; out-of-vocabulary values are rejected at the quality-validation stage.

## Consequences

- `backend/app/data/source_type_stratified_analysis.py` joins predictions to `source_type` and emits per-source-type macro-F1 / accuracy / per-class scores.
- The Phase 5 cross-source transfer matrix trains on `fomc_statement` + `fomc_minutes` and evaluates on the other source types to measure transferability — a key piece of the thesis's external-validity claim.
- Sample weights (see [0007 follow-up] / sample-weight provenance routing) are computed per row independent of `source_type`, so source-type-stratified results remain comparable across provenance levels.
