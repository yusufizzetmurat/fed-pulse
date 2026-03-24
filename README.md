# Fed Pulse

Fed Pulse is a thesis-oriented research project on market forecasting from central bank communication.  
The system combines NLP signals from FOMC text with market time-series features, then evaluates prediction quality and runtime behavior under a reproducible workflow.

## What This Project Is Trying to Answer

This project focuses on three practical research questions:

1. Do text-derived signals improve short-horizon market forecasts?
2. How much does on-demand adaptation help compared to static inference?
3. Can we keep the workflow reproducible enough for academic reporting?

## Repository Layout

- `backend/`: API, data engineering modules, forecasting services, and training utilities.
- `frontend/`: single-page interface for analysis requests, charting, diagnostics, and overlay checks.
- `data/`: raw, interim, processed, and artifact outputs used by the pipeline.

## Running the System

### Recommended

Use `make` commands from the repository root:

- `make dev-cpu`: run frontend and backend in CPU-safe mode.
- `make dev-gpu`: run frontend and GPU backend profile (requires NVIDIA runtime).
- `make down`: stop running containers.
- `make logs`: tail service logs.

### Direct Docker Compose

- CPU: `docker compose up -d --build backend frontend`
- GPU: `docker compose --profile gpu up -d --build backend-gpu frontend`

Frontend URL: `http://localhost:3000`  
Backend docs: `http://localhost:8000/docs`

## Data Preparation Workflow (Capability-First)

The data pipeline is intentionally named by capability, not by abstract phase labels:

- `app.data.source_ingestion`
- `app.data.label_normalization`
- `app.data.quality_validation`
- `app.data.training_package_builder`
- `app.data.baseline_spec_generator`
- `app.data.pipeline_data_prep` (orchestrator)

### End-to-End Data Prep

```
make data-prep DATASET_VERSION=<dataset_version> FEATURE_VERSION=<feature_version> OWNER=<owner>
```

or

```
python -m app.data.pipeline_data_prep --dataset-version <dataset_version> --feature-version <feature_version> --owner <owner>
```

## Model Execution (Smoke First, Then Full)

### Smoke Run

Run a single-seed sanity pass before large runs:

```
make train-smoke TRAINING_PACKAGE_ID=<training_package_id> SEED=11 OWNER=<owner>
```

or

```
python -m app.data.phase3_training_execution --training-package-id <training_package_id> --mode smoke --model bert --seed 11 --owner <owner>
```

### Full Batch

Run all configured candidates over official seeds:

```
make train-batch TRAINING_PACKAGE_ID=<training_package_id> OWNER=<owner>
```

or

```
python -m app.data.phase3_training_execution --training-package-id <training_package_id> --mode full --owner <owner>
```

## Forecast and Overlay Behavior

The UI supports:

- sentiment output from statement text,
- close/volatility forecasts with confidence bands,
- realized overlay for historical dates,
- diagnostics (RMSE/MAPE-style error views, runtime metadata, and band checks).

Important detail: overlay and evaluation metrics are now timestamp-aligned (not index-aligned), so realized comparisons are based on actual overlapping dates.

## Legacy Command Compatibility

Older module names are still accepted for compatibility and now emit deprecation warnings:

- `app.data.run_phase2_pipeline` -> `app.data.pipeline_data_prep`
- `app.data.ingest_sources` -> `app.data.source_ingestion`
- `app.data.normalize_labels` -> `app.data.label_normalization`
- `app.data.quality_checks` -> `app.data.quality_validation`
- `app.data.build_training_package` -> `app.data.training_package_builder`
- `app.data.generate_baseline_run_specs` -> `app.data.baseline_spec_generator`

## Current Limitations and Notes

- Some external checkpoints can require authentication or can be unavailable depending on environment/network policy.
- GPU mode is optional by design; CPU mode remains the default path for wider portability.
- This repository is still under active research development and should be treated as experimental software.

## License

No public license has been attached to this repository at this time.  
All rights are reserved unless a license is explicitly provided.
