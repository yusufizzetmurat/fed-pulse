# Data and Training Contracts

## Purpose
Define a single contract from ingestion to training package export.

## Approved Sources
- `hf_fomc_communication` (research-only, citation required)
- `kaggle_fed_statements_minutes` (license/terms apply)
- `scraped_fed` (internal scraper output)

## Ingestion Contract
Each row must contain:
- `record_id`, `source`, `source_record_id`
- `document_type`, `event_date`, `text`
- `label` (optional), `label_origin`
- `license_scope`, `citation_ref`
- `ingested_at_utc`, `text_hash`

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
- Pseudo-labeling excludes reporting holdout
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
