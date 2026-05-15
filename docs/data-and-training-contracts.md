# Data and Training Contracts

## Purpose
Define a single contract from ingestion to training package export.

## Approved Sources

### Supervised training pool (provenance ∈ {peer_reviewed, kaggle, scraped})
- `hf_fomc_communication` (research-only, citation required) — Trillion Dollar Words sentence-level stance labels.
- `kaggle_fed_statements_minutes` (license/terms apply) — Kaggle FOMC statement/minutes mirror.
- `scraped_fed` (internal scraper output; FOMC minutes/statements/Chair speeches/governor speeches/testimonies/press conferences/Beige Book/regional research).
- `op_fed` (MIT, Keith et al. 2025) — FOMC meeting-transcript sentence-level stance + opinion + monetary-policy annotations.
- `gss_factor` (research-only, Gürkaynak-Sack-Swanson 2005 IJCB) — per-FOMC target/path factor decomposition; populates the factor axis, no stance label.
- `gtfintechlab_federal_reserve_system` (research-only, Shah et al. 2024 gtfintechlab) — FOMC sentence-level multi-axis labels (stance + time + certainty), 3,000 rows, complements TDW.

### Cross-bank generalization pool (provenance = peer_reviewed_cross_bank, sample_weight = 0.0)
These sources enter the unified registry but are excluded from the supervised training loss. They drive the cross-CB generalization evaluation harness.
- `gtfintechlab_european_central_bank`
- `gtfintechlab_bank_of_japan`
- `gtfintechlab_bank_of_england`
- `gtfintechlab_bank_of_canada`
- `gtfintechlab_reserve_bank_of_australia`

### Credibility-only pool (provenance = scraped, sample_weight = 0.0)
Unlabelled corpora that feed the credibility module (drift, realized-vs-stated gap) and serve as auxiliary continued-pretraining substrate. Not in the supervised training pool.
- `vtasca_fomc_archive` (vtasca/fomc-statements-minutes) — 463 whole-document FOMC statements + minutes.

When adding a new source, append it here AND to `_PEER_REVIEWED_SOURCES` / `_KAGGLE_SOURCES` in `backend/app/data/normalize_labels.py`. HF-hosted datasets must additionally appear in `_DATASET_REVISIONS` in `backend/app/data/ingest_sources.py` with a pinned commit SHA so `record_id` does not rotate on upstream pushes.

## Ingestion Contract
Each row must contain:
- `record_id`, `source`, `source_record_id`
- `document_type`, `source_type`, `event_date`, `text`
- `label` (optional), `label_origin`
- `license_scope`, `citation_ref`
- `ingested_at_utc`, `text_hash`

`source_type` values are the closed set in `backend/app/data/source_type.py`. They are finer-grained than `source` (which scopes to a provider) and finer-grained than `document_type` (which scopes to a high-level kind). Stratified analyses filter on `source_type`.

Rules:
1. Normalize text before hashing
2. Build deterministic fallback IDs when source ID is missing
3. Reject rows with missing `event_date` or empty `text`
4. Log rejects with reason codes

## Label Contract
Target label set:
- `hawkish`
- `dovish`
- `neutral`

Unmappable labels are excluded and logged.

## Quality and Leakage Controls
- Exact dedup key: `text_hash`
- Near-duplicate checks on normalized text
- No train/test near-duplicate leakage inside the same fold
- Chronological splits only
- Pseudo-labeling (`backend/app/data/pseudo_labeling.py`) excludes the reporting holdout. Pseudo rows carry `label_origin = "pseudo"`, `teacher_model_id`, `teacher_model_version`, `teacher_max_score`, and `teacher_scores`.
- Scalers/statistics fit on train only

## Training Package Contract
Required metadata:
- `dataset_version`
- `feature_version`
- `evaluation_protocol`
- `generated_at_utc`

Required artifacts:
1. `registry_normalized.parquet`
2. `splits_train_val_test.parquet`
3. `fold_manifest_expanding_walk_forward.json`
4. `dataset_metadata.json`
5. `quality_reports/`

## Canonical Entities
- `raw_documents`
- `nlp_inference`
- `market_timeseries`
- `event_aligned_features`
- `forecast_targets`
- `training_packages`
- `model_registry`
- `experiment_runs`
- `online_predictions`

## Minimum Validation
- No duplicate aligned feature rows
- Targets built strictly from future timestamps
- Every run references valid model/data versions
- Every online prediction stores `model_id` and `runtime_mode`
