# Data and Training Contracts

## Goal
Define one clear contract from raw source ingestion to training-ready package export.

## Approved Sources
- `hf_fomc_communication` (research-only; citation required)
- `kaggle_fed_statements_minutes` (source terms apply)
- `scraped_fed` (internal scraper output)

## Source Ingestion Contract
Each ingested row must include:
- `record_id`
- `source`
- `source_record_id`
- `document_type`
- `event_date` (`YYYY-MM-DD`)
- `text`
- `label` (if available)
- `label_origin` (`human` or `pseudo`)
- `license_scope`
- `citation_ref`
- `ingested_at_utc`
- `text_hash`

Deterministic rules:
1. Normalize text before hashing.
2. Build stable fallback ids when source id is missing.
3. Reject rows with missing `event_date` or empty `text`.
4. Log all rejected rows with reason codes.

## Dataset Integration Strategy
Use labeled external data as supervision anchor, and scraped data for coverage:
1. Train initial model on human-labeled rows.
2. Generate pseudo-labels on unlabeled scraped rows.
3. Keep high-confidence pseudo-labels only.
4. Merge with source-aware sample weighting.
5. Retrain and export final sentiment features.

Official labels must map to:
- `hawkish`
- `dovish`
- `neutral`

Unmappable labels are excluded and logged.

## Quality and Leakage Controls

### Dedup
- Exact dedup key: `text_hash`.
- Near-duplicate checks on normalized text.
- Near-duplicates cannot cross train/test boundaries in same fold.
- Keep provenance when collapsing duplicates.

### Leakage
- Chronological splits only.
- Reporting holdout excluded from pseudo-label generation.
- No future-aware features.
- Scalers/statistics fit on train only.

### Label Integrity
- `label_origin` required for all rows.
- Pseudo labels require confidence and threshold policy.

Pre-export checks:
- no missing critical fields
- dedup completed
- cross-boundary near-duplicate violations = 0
- fold-level leakage checks passed
- mapping exceptions reviewed
- source/class distribution report generated

## Training Package Spec
Each package must include:
- `dataset_version`
- `feature_version`
- `evaluation_protocol` (`evaluation_protocol_v1`)
- `generated_at_utc`

Required files:
1. `registry_normalized.parquet`
2. `splits_train_val_test.parquet`
3. `fold_manifest_expanding_walk_forward.json`
4. `dataset_metadata.json`
5. `quality_reports/` (dedup, leakage, label exceptions)

Valid split tags:
- `train`, `val`, `test`
- `wf_fold_<k>_train`, `wf_fold_<k>_val`, `wf_fold_<k>_test`

## Canonical Schema (Entity-Level)
- `raw_documents`: source text records and provenance.
- `nlp_inference`: sentiment outputs and confidence.
- `market_timeseries`: symbol/date market features.
- `event_aligned_features`: event + market aligned training rows.
- `forecast_targets`: horizon-specific future targets.
- `training_packages`: metadata for package exports.
- `model_registry`: model metadata and artifact pointer.
- `experiment_runs`: offline run metadata and metrics.
- `online_predictions`: serving-time prediction logs.

## Minimum Validation Baseline
- no duplicate (`doc_id`, `symbol`, `feature_version`) aligned rows
- targets created strictly from future timestamps
- every run references existing model/data versions
- every online prediction includes `model_id` and `runtime_mode`
# Data and Training Contracts

## Goal
Define one clear contract from raw source ingestion to training-ready package export.

## Approved Sources
- `hf_fomc_communication` (research-only; citation required)
- `kaggle_fed_statements_minutes` (source terms apply)
- `scraped_fed` (internal scraper output)

## Source Ingestion Contract
Each ingested row must include:
- `record_id`
- `source`
- `source_record_id`
- `document_type`
- `event_date` (`YYYY-MM-DD`)
- `text`
- `label` (if available)
- `label_origin` (`human` or `pseudo`)
- `license_scope`
- `citation_ref`
- `ingested_at_utc`
- `text_hash`

Deterministic rules:
1. Normalize text before hashing.
2. Build stable fallback ids when source id is missing.
3. Reject rows with missing `event_date` or empty `text`.
4. Log all rejected rows with reason codes.

## Dataset Integration Strategy
Use labeled external data as supervision anchor, and scraped data for coverage:
1. Train initial model on human-labeled rows.
2. Generate pseudo-labels on unlabeled scraped rows.
3. Keep high-confidence pseudo-labels only.
4. Merge with source-aware sample weighting.
5. Retrain and export final sentiment features.

Official labels must map to:
- `hawkish`
- `dovish`
- `neutral`

Unmappable labels are excluded and logged.

## Quality and Leakage Controls

### Dedup
- Exact dedup key: `text_hash`.
- Near-duplicate checks on normalized text.
- Near-duplicates cannot cross train/test boundaries in same fold.
- Keep provenance when collapsing duplicates.

### Leakage
- Chronological splits only.
- Reporting holdout excluded from pseudo-label generation.
- No future-aware features.
- Scalers/statistics fit on train only.

### Label Integrity
- `label_origin` required for all rows.
- Pseudo labels require confidence and threshold policy.

Pre-export checks:
- no missing critical fields
- dedup completed
- cross-boundary near-duplicate violations = 0
- fold-level leakage checks passed
- mapping exceptions reviewed
- source/class distribution report generated

## Training Package Spec
Each package must include:
- `dataset_version`
- `feature_version`
- `evaluation_protocol` (`evaluation_protocol_v1`)
- `generated_at_utc`

Required files:
1. `registry_normalized.parquet`
2. `splits_train_val_test.parquet`
3. `fold_manifest_expanding_walk_forward.json`
4. `dataset_metadata.json`
5. `quality_reports/` (dedup, leakage, label exceptions)

Valid split tags:
- `train`, `val`, `test`
- `wf_fold_<k>_train`, `wf_fold_<k>_val`, `wf_fold_<k>_test`

## Canonical Schema (Entity-Level)
- `raw_documents`: source text records and provenance.
- `nlp_inference`: sentiment outputs and confidence.
- `market_timeseries`: symbol/date market features.
- `event_aligned_features`: event + market aligned training rows.
- `forecast_targets`: horizon-specific future targets.
- `training_packages`: metadata for package exports.
- `model_registry`: model metadata and artifact pointer.
- `experiment_runs`: offline run metadata and metrics.
- `online_predictions`: serving-time prediction logs.

## Minimum Validation Baseline
- no duplicate (`doc_id`, `symbol`, `feature_version`) aligned rows
- targets created strictly from future timestamps
- every run references existing model/data versions
- every online prediction includes `model_id` and `runtime_mode`
