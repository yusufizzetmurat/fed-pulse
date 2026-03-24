# Fed Pulse Documentation

This directory is intentionally compact for public GitHub readability.

## Main Docs
- `project-guide.md`  
  Project objective, key decisions, scope, and phase-by-phase plan.
- `data-and-training-contracts.md`  
  Source ingestion rules, dataset integration strategy, quality/leakage controls, training package contract, and schema summary.
- `benchmark-policy.md`  
  Evaluation protocol, versioning policy, and official NLP baseline run policy.
- `architecture-overview.md`  
  API flow, backend pipeline, feature set, runtime modes, frontend contract, and main code modules.
- `run-templates.md`  
  Pre-run checklist, post-run report, and artifact manifest template.

## Suggested Reading Order
1. `project-guide.md`
2. `benchmark-policy.md`
3. `data-and-training-contracts.md`
4. `architecture-overview.md`
5. `run-templates.md`

## Engineering Entry Points

### Recommended: `make`
- Local stack:
  - `make dev-cpu`
  - `make dev-gpu` (requires NVIDIA runtime)
- Data preparation:
  - `make data-prep DATASET_VERSION=<dataset_version> FEATURE_VERSION=<feature_version> OWNER=<owner>`
- Training execution:
  - `make train-smoke TRAINING_PACKAGE_ID=<training_package_id> SEED=11 OWNER=<owner>`
  - `make train-batch TRAINING_PACKAGE_ID=<training_package_id> OWNER=<owner>`

### Canonical Capability-First Python Commands
- End-to-end data preparation:
  - `python -m app.data.pipeline_data_prep --dataset-version <dataset_version> --feature-version <feature_version> --owner <owner>`
- Step-by-step data preparation:
  - `python -m app.data.source_ingestion --all-sources --data-dir ../data`
  - `python -m app.data.label_normalization --input ../data/raw/phase2/source_registry.jsonl --output ../data/interim/phase2/registry_labeled.jsonl`
  - `python -m app.data.quality_validation --input ../data/interim/phase2/registry_labeled.jsonl --output ../data/interim/phase2/registry_quality_passed.jsonl`
  - `python -m app.data.training_package_builder --input ../data/interim/phase2/registry_quality_passed.jsonl --dataset-version <dataset_version> --feature-version <feature_version>`
  - `python -m app.data.baseline_spec_generator --dataset-version <dataset_version> --feature-version <feature_version> --training-package-id <training_package_id> --owner <owner>`
- Training execution:
  - `python -m app.data.phase3_training_execution --training-package-id <training_package_id> --mode smoke --model bert --seed 11 --owner <owner>`
  - `python -m app.data.phase3_training_execution --training-package-id <training_package_id> --mode full --owner <owner>`

## Legacy Command Mapping (Temporary Compatibility)
- `app.data.run_phase2_pipeline` -> `app.data.pipeline_data_prep`
- `app.data.ingest_sources` -> `app.data.source_ingestion`
- `app.data.normalize_labels` -> `app.data.label_normalization`
- `app.data.quality_checks` -> `app.data.quality_validation`
- `app.data.build_training_package` -> `app.data.training_package_builder`
- `app.data.generate_baseline_run_specs` -> `app.data.baseline_spec_generator`
