# Fed Pulse

Short-horizon market forecasting from FOMC text + market data.

A FastAPI + PyTorch backend ingests an FOMC document, scores its monetary-policy stance along three axes, fuses the result with market history through an ensemble of sequence models, and returns close + volatility forecasts with confidence bands. A Next.js 14 dashboard renders the output as multi-axis cards, per-sentence XAI highlights, a credibility panel, and persisted analysis history. SWE 599 research project at Boğaziçi University. Live deployment: <https://fedpulse.yusufizzetmurat.com/>.

## Start here

New readers should open [`docs/REPO_TOUR.md`](docs/REPO_TOUR.md), a single-page walkthrough of each directory and where to start reading for any given change.

## Quick start

```bash
make dev-cpu           # backend + frontend on CPU
make dev-gpu           # same, requires NVIDIA runtime
make down              # stop the stack
make logs              # tail container logs
```

- Frontend: <http://localhost:3000>
- Backend OpenAPI: <http://localhost:8000/docs>

## Core workflows

```bash
make data-prep   DATASET_VERSION=<v> FEATURE_VERSION=<v> OWNER=<who>
make train-smoke TRAINING_PACKAGE_ID=<id> SEED=11 OWNER=<who>
make train-batch TRAINING_PACKAGE_ID=<id> OWNER=<who>
make verify      # toy snapshot + unit/property/contract/regression tests + import sanity
```

`/analyze` ships a single inference path: `fast` mode runs checkpoint inference, with a cold start triggering a 252-day bootstrap. Replay mode loads pinned per-fold checkpoints. The earlier `quick_train` / `real_train` runtime modes and the `/train-jobs` queue were retired in PR #265.

## Layout

```
backend/          FastAPI + PyTorch backend (app/, models, services, data pipeline)
frontend/         Next.js 14 dashboard (analyze, history, calendar, /legacy)
data/             runtime data volume (raw scrapes, processed packages, run artefacts, SQLite history)
docs/             executable contracts that code and CI gate on (read REPO_TOUR.md first)
scripts/          one-shot utilities (toy snapshot, OpenAPI regen, GSS factor extraction, …)
tests/            pytest suite (~1102+ unit / property / contract / regression / e2e tests)
configs/          ablation configs (YAML)
```

Long-form design, requirements, and ADRs live in the companion wiki at `../fed-pulse.wiki/`, checked out separately from the same GitHub remote.

## Forecasters

Eight architectures are registered in `app.models.factory`. The headline forecasters are the QLIKE-DLq ensemble and HAR-tercile. The multi-asset QLIKE-DLq variant now serves `^GSPC`, `^NDX`, and `^DJI` (PR #660).

## Text classification

The multi-axis classifier scores stance, certainty, and forward-looking on each input document. The factor axis was retired in PR #597; the topic axis was retired in ADR 0044.

## Local setup

1. `cp .env.example .env` and fill in the credentials needed (`HF_TOKEN` covers most gated checkpoints).
2. Install pre-commit (`pip install --user pre-commit`) and run `pre-commit install` once in this clone. Every commit then runs ruff, ruff-format, gitleaks, and the whitespace/EOF hooks.

## Deploy

The production stack runs on a single DigitalOcean droplet (8 GB / 4 vCPU, ~$48/month). See [`docs/deploy.md`](docs/deploy.md) for the provisioning runbook and [`docs/reproduce.md`](docs/reproduce.md) for the `make reproduce-all` walkthrough.

- Hostname: `fedpulse.yusufizzetmurat.com`
- Stack: Caddy reverse proxy + FastAPI backend + Next.js standalone frontend (all in `compose.prod.yml`)
- HF Hub stores every model artefact. The droplet eager-pulls the hot path at boot and lazy-fetches the alternatives on first use.
- Deploy automation: `.github/workflows/deploy.yml` triggers on push to `main` and blocks until `ci.yml` succeeds on the same sha. Required secrets: `DROPLET_SSH_KEY`, `HF_TOKEN`, `FRED_API_KEY`.
- HF artefact pushes run server-side via the manual `.github/workflows/hf-push.yml` workflow (`workflow_dispatch`) using `HF_TOKEN_WRITE`. The write-scoped token never leaves the GH secret store.

## Executable contracts

The four files under `docs/` pin down what every official run must do. Code and CI gate on them directly:

- [`docs/benchmark-policy.md`](docs/benchmark-policy.md) — required IDs, splits + seeds, leakage rules, NLP-baseline winner criteria.
- [`docs/data-and-training-contracts.md`](docs/data-and-training-contracts.md) — approved sources, ingestion contract fields, training-package artifacts.
- [`docs/run-templates.md`](docs/run-templates.md) — naming conventions for `run_id` and artifact directories.
- [`docs/security-acceptance.md`](docs/security-acceptance.md) — known-accepted advisories with mitigation owners.

Narrative design and research material (project guide, system architecture, requirements, design document, deep learning material, risk register, ADRs, external corpora inventory) lives in the [wiki](https://github.com/yusufizzetmurat/fed-pulse/wiki).

## Notes

- Realised overlay + error metrics are timestamp-aligned when the analysed document date is historical.
- Data pipeline entry points are capability-first: `source_ingestion`, `label_normalization`, `quality_validation`, `training_package_builder`, `baseline_spec_generator`, `pipeline_data_prep`, `nlp_baseline_batch`, `finetune_pilot`, `finetune_batch`, `attention_ablation`. The earlier `phase3_*` / `phase4_*` / `run_phase2_pipeline` aliases were retired in Phase 7.2.
