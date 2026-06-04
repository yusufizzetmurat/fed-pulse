# `backend/app/data/` — offline data pipeline

Everything that runs outside the live API lives here: ingestion of raw text, label normalisation, quality validation, training-package assembly, baseline / fine-tune / ablation harnesses, and the (archived) pseudo-labelling pipeline. The live `/analyze` request path does not import from this directory; runtime code lives in `services/`.

Two layers coexist:

- Capability-first entry points (preferred for new code): `source_ingestion`, `label_normalization`, `quality_validation`, `training_package_builder`, `baseline_spec_generator`, `pipeline_data_prep`.
- Phase-named implementations under the same names with `phase3_*` / `phase4_*` prefixes. Makefile targets still call them, and they hold the real logic; the capability-first names are thin wrappers. The phase-named layer is slated for removal (`docs/REPO_TOUR.md §7`).

The Makefile is the authoritative entry point. `make data-prep`, `make train-smoke`, and `make train-batch` all route through this directory.

## File map

### Capability-first wrappers (use these)

| File | What it does | Wraps |
| --- | --- | --- |
| `source_ingestion.py` | Pull rows from approved sources (TDW, Kaggle, scraped Fed JSON, Op-Fed, GSS), emit `data/raw/phase2/source_registry.jsonl`. | `ingest_sources.py` |
| `label_normalization.py` | Map raw label strings to `{hawkish, dovish, neutral}`. Drop unmappable rows and log exceptions. Sample-weight by provenance. | `normalize_labels.py` |
| `quality_validation.py` | Near-duplicate filter (0.97), text-hash collisions, leakage checks. | `quality_checks.py` |
| `training_package_builder.py` | Build the canonical training package directory (`registry_normalized.parquet`, splits, fold manifest, `dataset_metadata.json`, `quality_reports/`). | `build_training_package.py` |
| `baseline_spec_generator.py` | Pre-run-configuration markdown per planned run. | `generate_baseline_run_specs.py` |
| `pipeline_data_prep.py` | Orchestrator. `make data-prep` calls this. | — (top-level) |

### Implementation modules (wrapped by the capability-first entries above)

| File | What it does |
| --- | --- |
| `ingest_sources.py` | The actual ingestion logic. CLI flags `--include-hf/--include-kaggle/--include-scraped/--include-op-fed/--include-gss-factors`. |
| `normalize_labels.py` | Label-mapping logic. `_PEER_REVIEWED_SOURCES` whitelist is here. |
| `quality_checks.py` | Quality-validation logic. |
| `build_training_package.py` | Training-package assembly logic. |
| `generate_baseline_run_specs.py` | Pre-run-config markdown. |

### Training / evaluation harnesses

| File | What it does | Status |
| --- | --- | --- |
| `nlp_baseline_batch.py` | Official NLP zero-shot batch (BERT / FinBERT / FOMC-RoBERTa × 5 seeds). | alive — `make train-batch` calls this |
| `finetune_batch.py` | Fine-tune full batch (6 encoders × 5 seeds). | alive |
| `finetune_pilot.py` | Single-seed fine-tune. Writes `predictions.jsonl` for the cross-source analyzer. | alive |
| `attention_ablation.py` | Variant A / B / C ablation sweep (6-cell grid × seeds). | alive |
| `pseudo_labeling.py` | Chunk-aggregated teacher for the 9,696-row unlabelled scraped pool. Strategies: `chunk_max_pool` (default), `chunk_mean_pool`, `chunk_vote`, `doc_truncated` (legacy). | kept for ADR-0006 audit trail; not used as training labels |
| `llm_judge.py` | Gemini judge plus three gating policies (`confidence_only`, `confidence_and_judge`, `judge_only`), a 100-row stratified audit sampler, and Cohen's κ. | alive |
| `continued_pretraining.py` | MLM continued pretraining of FinBERT-FedAdjacent on the unlabelled scraped rows. Pipeline ready; the checkpoint has not yet been trained on a GPU. | code ready, not run |

### Embedding stores (used by the chunk-attention pooler and the chunk-aware teacher)

| File | What it does |
| --- | --- |
| `chunk_embedding_store.py` | Persist per-document CLS embeddings to `data/processed/<package_id>/chunk_embeddings.parquet`. Uses `app.services.text_encoder.split_into_chunks` (480-token windows, 400 stride). |
| `chunk_embedding_retrieval.py` | Look up chunk embeddings for a given anchor date within a lookback window. Used by the Phase-4 chunk-attention pooler. |
| `llm_embedding_store.py` | Per-document Gemini embedding precompute (Variant C ablation cell). |

### Historical / archived runs

These produced one-shot artefacts already published under `data/artifacts/`. CLI access remains, but reruns are rare.

| File | Produced |
| --- | --- |
| `embedding_comparator.py` | MiniLM frozen-head baseline (Phase 4 FR-35). macro-F1 0.543. |
| `llm_zero_shot_execution.py` | Qwen-3B zero-shot baseline (Phase 4 FR-26). macro-F1 0.229. Phase-5 rerun targets a 30B+ model. |
| `source_type_stratified_analysis.py` | Joiner from per-row predictions to per-source-type macro-F1 / accuracy / per-class P/R/F1. Run after each fine-tune batch. |

### Schema definitions

| File | What it does |
| --- | --- |
| `label_schemas.py` | Pydantic models for `MultiAxisLabeledRow` (stance / certainty / forward-looking axes; the factor axis was retired in PR #597 and the topic axis in ADR 0044). Sample-weight by provenance. |
| `source_type.py` | The 10-value `SOURCE_TYPE_VALUES` whitelist and `infer_source_type()` mapping from `(document_type, title)`. |

### Per-source scrapers

Located under `sources/`. See `sources/base.py` for the `BaseSourceScraper` protocol and the parallel `_VALID_SOURCE_TYPES` whitelist.

## See also

- `../../README.md` — quick start and tour pointer.
- `../../docs/REPO_TOUR.md` — single-page walkthrough of the whole codebase. §7 lists which files in this directory are alive vs historical.
- `../../docs/benchmark-policy.md` — version IDs, splits, seeds, leakage rules every official run must satisfy.
- `../../docs/data-and-training-contracts.md` — approved sources, ingestion contract fields, required training-package artefacts.
