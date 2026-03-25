# Fed Pulse

Fed Pulse is a research project for short-horizon market forecasting from FOMC text + market data.

## Quick Start

- Start (CPU): `make dev-cpu`
- Start (GPU): `make dev-gpu`
- Stop: `make down`
- Logs: `make logs`

Frontend: `http://localhost:3000`  
Backend API docs: `http://localhost:8000/docs`

## Core Workflow

- Data prep: `make data-prep DATASET_VERSION=<dataset_version> FEATURE_VERSION=<feature_version> OWNER=<owner>`
- Smoke run: `make train-smoke TRAINING_PACKAGE_ID=<training_package_id> SEED=11 OWNER=<owner>`
- Full batch: `make train-batch TRAINING_PACKAGE_ID=<training_package_id> OWNER=<owner>`

## Main Directories

- `backend/`: API, forecasting services, and data pipeline modules
- `frontend/`: dashboard and charts
- `data/`: datasets and generated artifacts
- `docs/`: project policies and contracts

## Documentation

- `docs/project-guide.md`
- `docs/benchmark-policy.md`
- `docs/data-and-training-contracts.md`
- `docs/architecture-overview.md`
- `docs/run-templates.md`

## Notes

- Forecast modes: `fast`, `quick_train`, `real_train`
- Overlay/error comparisons are timestamp-aligned
- Legacy phase-named data module entrypoints still work as wrappers
