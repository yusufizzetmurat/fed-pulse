# Fed Pulse

Short-horizon market forecasting from FOMC text + market data.

A FastAPI + PyTorch backend ingests an FOMC document, scores its monetary-policy stance, fuses the score with market history through a small sequence model, and returns a close + volatility forecast with confidence bands. A Next.js 14 dashboard renders the result with multi-axis cards, per-sentence XAI highlights, a credibility panel, and a persisted analysis history. Research project for SWE 599 at Boğaziçi University.

## Start here

**New to the repo?** Open [`docs/REPO_TOUR.md`](docs/REPO_TOUR.md). It's a single-page walkthrough of what each directory holds, what's been built so far, and where to start reading for any given change.

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

Three runtime modes on `/analyze`: `fast` (checkpoint inference, cold start triggers a 252-day bootstrap), `quick_train` (short bounded adaptation, no persistence), `real_train` (async — returns `{job_id, status: queued}`, client polls `GET /train-jobs/{job_id}`).

## Layout

```
backend/          FastAPI + PyTorch backend (app/, models, services, data pipeline)
frontend/         Next.js 14 dashboard (analyze, history, calendar, /legacy)
data/             runtime data volume (raw scrapes, processed packages, run artefacts, SQLite history)
docs/             executable contracts that code and CI gate on (read REPO_TOUR.md first)
scripts/          one-shot utilities (toy snapshot, OpenAPI regen, GSS factor extraction, …)
tests/            pytest suite (unit / property / contract / regression / e2e)
configs/          ablation configs (YAML)
```

Long-form design / requirements / roadmap / ADRs live in the companion wiki at `../fed-pulse.wiki/`, separately checked out from the same GitHub remote.

## Local setup

1. `cp .env.example .env` and fill in the credentials you need (`HF_TOKEN` covers most gated checkpoints).
2. Install pre-commit (`pip install --user pre-commit`) and run `pre-commit install` once in this clone — every commit then runs ruff, ruff-format, gitleaks, and the whitespace/EOF hooks.

## Deploy

The production stack runs on a single DigitalOcean droplet (8 GB / 4 vCPU, ~$48/month). See [`docs/deploy.md`](docs/deploy.md) for the provisioning runbook and [`docs/reproduce.md`](docs/reproduce.md) for the `make reproduce-all` walkthrough.

- Hostname: `fedpulse.yusufizzetmurat.com`
- Stack: Caddy reverse proxy + FastAPI backend + Next.js standalone frontend (all in `compose.prod.yml`)
- HF Hub stores every model artefact; the droplet eager-pulls the hot path at boot and lazy-fetches the alternatives on first use
- Deploy automation: `.github/workflows/deploy.yml` triggers on push to `main`. Required secrets: `DROPLET_SSH_KEY`, `HF_TOKEN`, `FRED_API_KEY`

## Executable contracts

The four files under `docs/` pin down what every official run must do. Code and CI gate on these directly:

- [`docs/benchmark-policy.md`](docs/benchmark-policy.md) — required IDs, splits + seeds, leakage rules, NLP-baseline winner criteria.
- [`docs/data-and-training-contracts.md`](docs/data-and-training-contracts.md) — approved sources, ingestion contract fields, training-package artifacts.
- [`docs/run-templates.md`](docs/run-templates.md) — naming conventions for `run_id` and artifact directories.
- [`docs/security-acceptance.md`](docs/security-acceptance.md) — known-accepted advisories with mitigation owners.

Narrative + design + research docs (project guide, system architecture, requirements, design document, deep learning roadmap, risk register, ADRs, external corpora inventory) live in the [wiki](https://github.com/yusufizzetmurat/fed-pulse/wiki).

## Notes

- Forecast modes: `fast`, `quick_train`, `real_train`.
- Realised overlay + error metrics are timestamp-aligned when the analysed document date is historical.
- Data pipeline entry points are capability-first: `source_ingestion`, `label_normalization`, `quality_validation`, `training_package_builder`, `baseline_spec_generator`, `pipeline_data_prep`, `nlp_baseline_batch`, `finetune_pilot`, `finetune_batch`, `attention_ablation`. The earlier `phase3_*` / `phase4_*` / `run_phase2_pipeline` names were retired in Phase 7.2.
