# Repo tour

Single-page walkthrough of the Fed Pulse codebase, written so a reader who has never opened the repo can pick it up cold. The companion wiki (`../../fed-pulse.wiki/`) carries the long-form design, requirements, ADRs, and roadmap; this document is the **fast index** — what each directory holds, which files are alive vs historical, and where to start reading for any given change.

Read top to bottom the first time. After that, jump to §6 ("I want to …") for task-oriented entry points.

---

## 1. What Fed Pulse actually does

Fed Pulse is a research project (SWE 599 thesis at Boğaziçi University) that takes a piece of FOMC text — a statement, minutes excerpt, speech, or pasted snippet — plus market history, and produces:

1. A short-horizon close-price + volatility forecast (next 1 / 3 / 5 / 10 trading days) with confidence bands.
2. A stance classification (hawkish / dovish / neutral) plus three secondary axes (factor / certainty / topic) — the **multi-axis** schema.
3. Per-sentence XAI highlights showing which tokens drive the stance call.
4. A credibility panel showing how the document compares to the prior four FOMC communications.
5. A persisted history so each analysis is searchable / comparable across runs.

The thesis question is whether fusing FOMC text with market history through a small sequence model produces better short-horizon forecasts than a market-only baseline, and when runtime adaptation is worth its compute cost. Done criteria live in `../../fed-pulse.wiki/05_Project_Plan.md §1`.

---

## 2. The 60-second tour: what happens when a user clicks "Analyze"

```
Browser  ─POST /analyze─▶  FastAPI (backend/app/main.py)
                              │
                              ├─ sentiment           services/sentiment.py     ─▶ HF classifier (services/text_encoder.py)
                              ├─ market history     services/market_data.py   ─▶ yfinance
                              ├─ feature vectors    services/forecaster.py    ─▶ 6-feature sequence
                              ├─ forecast           services/forecaster.py    ─▶ checkpoint inference / quick / real train
                              ├─ confidence bands   evaluation/conformal.py   ─▶ Lei–Wasserman split-conformal (else Gaussian z fallback)
                              ├─ XAI                evaluation/xai.py         ─▶ keyword-salience per sentence/token
                              ├─ history write     db.py                      ─▶ SQLite at data/db/fed_pulse.db
                              ├─ audit              audit.py                  ─▶ JSONL at data/artifacts/audit.log
                              └─ AnalyzeResponse (schemas.py)  ────────────────▶  Next.js dashboard
```

The dashboard then renders:

- Multi-axis cards (`frontend/components/analyze/MultiAxisCards.tsx`)
- Forecast charts with confidence bands (`ForecastChart.tsx`)
- XAI highlights (`XaiPanel.tsx` via lazy-loaded `PreviewPanels.tsx`)
- Credibility panel (`CredibilityPanel.tsx`, currently fixture)
- Market context, error metrics if the date is historical

Three runtime modes on `/analyze`:

- `fast` — checkpoint inference. Cold start triggers a one-shot bootstrap train against 252-day history.
- `quick_train` — short bounded adaptation (history 30) before inference. Doesn't persist.
- `real_train` — async: enqueue a job, return `{job_id, status: queued}`, client polls `GET /train-jobs/{job_id}`.

The async job state is in-process only (`_train_jobs` dict + lock in `main.py`). Restarts lose queued/running jobs. Phase 7.12 (arq worker) is the fix; until then, the limitation is documented.

---

## 3. What's been built so far (project history in plain English)

The strategic roadmap lives in `../../fed-pulse.wiki/05_Project_Plan.md`. This section translates the phase IDs into what actually changed on disk.

### Phase 0 (closed 2026-03-15) — Frame the project

Locked the research question + evaluation protocol. Output: `docs/benchmark-policy.md` and the `docs/project-guide.md` referenced from the wiki. No code shipped.

### Phase 1 (closed 2026-03-26) — Data contract + pipeline stabilisation

Ad-hoc CSV ingestion → versioned pipeline. Six capability-first entry points landed under `backend/app/data/`: `source_ingestion`, `label_normalization`, `quality_validation`, `training_package_builder`, `baseline_spec_generator`, `pipeline_data_prep`. Each capability-first name is a thin wrapper around a phase-named module (`ingest_sources.py`, `normalize_labels.py`, etc.). New code should call the capability-first names; the phase-named modules remain because Make targets call them and Phase 7.2 hasn't retired them yet.

### Phase 2 (closed 2026-03-26) — Baseline LSTM, no adaptation

`services/forecaster.py` + `FeatureVector` + `bootstrap_checkpoint` + `forecast_quantitative_series`. The `fast` mode returns a forecast end-to-end. Frontend on Next.js 14 Pages Router. Docker Compose for `make dev-cpu` / `make dev-gpu`.

### Phase 3 (closed 2026-05-04) — Baseline NLP evaluation

Three encoders × five seeds: BERT-base, FinBERT (ProsusAI), FinBERT-FOMC (ZiweiChen). Plus sanity baselines (majority-class, random-class). Headline: zero-shot does not clear the random-class floor on this corpus. Fine-tuned FinBERT-FOMC hits macro-F1 0.6192 ± 0.0192 on the fold-2 test slice. Artifacts under `data/artifacts/phase3/`.

### Phase 4 (closed 2026-05-04) — Attention + decay + NLP expansion

Variant A (time-decay on scalar sentiment): **wins on combined-RMSE by 32% over baseline** on fold-2-val (seed 11, 200 epochs). λ_time learns from 1.49 → 1.62. Wins again on the wf_fold_3 holdout (5 seeds, 8% margin over baseline). This is the cleanest signal in the project.

Variant B (FinBERT chunk-attention pooler) and Variant C (Gemini LLM embeddings): both fail to converge at 2.4k labelled tuples. The diagnosis is data-starvation (not encoder choice or architecture). Cross-encoder comparison holds the pooling architecture fixed; both encoders fail similarly.

Additional Phase 4 deliverables: LLM zero-shot Qwen-3B (macro-F1 0.229, below random floor — flagged as conservative lower-bound), MiniLM embedding comparator (0.543), encoder battery extension to 6 encoders (BERT-base, FinBERT, FinBERT-FOMC, FOMC-RoBERTa, distilbert, deberta-v3-base).

### Phase 6 (in flight) — Frontend product + corpus expansion

Phase 6 came **before** Phase 5 in the execution order (the 2026-05-05 reorder). The frontend track is foundation-complete after the v2-foundation batches landed:

- **PR #67** dashboard cutover: `/legacy` keeps the v1 page; new `/` redirects to `/analyze` on the shadcn shell. Multi-axis cards, XAI panel, credibility panel scaffolds.
- **PR #68** history persistence (SQLite + `AnalysisRun` / `AnalysisResult` models), FOMC calendar widget, watchlist localStorage helper.
- **PR #69** embedding adapter `768→128` + LayerNorm + GELU (Phase 4.1) and `text_channel: "scalar" | "embeddings"` flag.
- **PR #70** split-conformal bands with the Lei–Wasserman correction; manifest persisted alongside the checkpoint. Block-bootstrap CIs (6-month blocks, 1000 resamples) in `regime_aggregator`.
- **PR #71** SE hardening: structlog with `run_id` ContextVar, typed error middleware, audit log JSONL, RNG state in checkpoint, strict Pydantic on `AnalyzeRequest`.
- **PR #106** CI hygiene: pip-audit, npm audit, Dependabot, gitlint, git-cliff CHANGELOG.
- **PR #107** lifespan classifier warmup + threadpool offload on `/analyze`.
- **PR #108** `POST /documents/parse` for paste / PDF / DOCX / URL ingestion.
- **PR #109** credibility feature module (drift, realized-vs-stated, market-implied-gap, months-since-reversal).
- **PR #110** keyword-salience XAI for `/analyze` (real attribution, drops fixture mode).
- **PR #120** Op-Fed ingestion (+159 stance rows + multi-axis annotations), GSS IJCB 2005 factor adapter (+138 factor-axis rows), chunk-aggregated teacher for pseudo-labels (resolves §2.5.8 of `06_Deep_Learning_Roadmap.md`).

Corpus expansion is the remaining Phase 6 work. The 92-row scraped pool is on disk but unlabelled. The earlier teacher-only pseudo-labelling failed audit at precision 0.30 (vs ≥0.90 gate); the chunk-aggregated teacher landed in PR #120 is the structural fix. Re-running the pipeline + audit + Phase 4 ablation against the expanded corpus is the next workflow operation.

### Phase 5 (in flight) — v2 model expansion

The v2 architecture's foundation pieces (embedding adapter, conformal bands, credibility module, bootstrap CIs) all landed in the Phase 6 batches above. What still needs to run on a GPU: forecaster-alternatives production benchmark (LSTM / GRU / TCN / small Transformer), Variant C 6-cell × 200-epoch ablation, regime-stratified eval slicer over the wf_fold_3 holdout, stronger LLM zero-shot rerun (Gemini 2.5 Pro), λ sensitivity sweep, full calibration plot.

### Phase 7 — SE hardening lockdown

In flight. Items landed so far: structlog + run_id middleware (7.7), Pydantic strict on `AnalyzeRequest` + frozen everywhere (7.8), error middleware + audit log (7.13), RNG state in checkpoint (7.4 partial), HF cache lifespan (7.11 partial), first CI hygiene pass (7.14 partial). Deferred: forecaster decomposition (7.1), phase-named module deletion (7.2), mypy strict on services (7.3), per-fold scaler + 2 pm ET cutoff (7.4 remainder), Pandera at stage boundaries (7.9), Schemathesis + perf regression (7.10), arq worker for real_train (7.12).

---

## 4. Where everything lives

### Top-level layout

```
fed-pulse/
├── backend/                 FastAPI + PyTorch backend
├── frontend/                Next.js 14 dashboard
├── data/                    runtime data volume (mounted into containers as /data)
├── docs/                    executable contracts + this tour
├── scripts/                 one-shot utility scripts
├── tests/                   pytest suite (unit / property / contract / regression / e2e)
├── configs/                 ablation configs (YAML)
├── CLAUDE.md                project conventions
├── Makefile                 the entry-point for every workflow
├── docker-compose.yml       dev stack (backend + frontend + GPU profile)
├── pyproject.toml           backend Python deps + ruff/mypy config
└── README.md                quick start; points here for deeper context
```

The wiki lives in a sibling directory `../fed-pulse.wiki/` — same git remote, separate checkout. Wiki pages are the design / requirements / roadmap layer; this code repo is the implementation layer.

### `backend/app/` — the API surface

| File | Lines | Purpose | Notes |
| --- | --- | --- | --- |
| `main.py` | ~483 | FastAPI router + lifespan. Nine endpoints. The only file that knows about FastAPI. | Async `/analyze` runs the synchronous forecaster via `run_in_threadpool`. |
| `schemas.py` | ~216 | Pydantic request/response models. `AnalyzeRequest` is `extra="forbid", strict=True, frozen=True`. Response models stay `frozen=True` only to keep the OpenAPI snapshot stable. | Edit here when API contract changes. |
| `config.py` | small | `BaseSettings` reading `FED_PULSE_DATA_DIR`. Used by every module to resolve `data/` paths. | |
| `determinism.py` | small | PyTorch seed logic + `worker_init_fn`. | |
| `db.py` | ~235 | SQLAlchemy 2.0 declarative. `AnalysisRun` + `AnalysisResult`. SQLite at `data/db/fed_pulse.db`. | Hooked from `/analyze` success path. |
| `audit.py` | ~82 | Append-only JSONL at `data/artifacts/audit.log`. | Hooks on checkpoint write, training-job finalization, benchmark publish. |
| `logging.py` | ~75 | structlog JSON config. `bind_run_id()` context manager. | |

### `backend/app/services/` — runtime services

| File | Purpose | Used by |
| --- | --- | --- |
| `forecaster.py` | LSTM / GRU / TCN / Transformer cores, attention pooling, conformal bands, checkpoint I/O, RNG state. **~1,635 lines — Phase 7.1 decomposition target.** | `main.py` `/analyze` |
| `sentiment.py` | Thin wrapper around `text_encoder.aggregate_label()` | `main.py` |
| `text_encoder.py` | HF classifier load + chunked embedding aggregation. `warmup_classifier()` called in lifespan. `split_into_chunks` is the canonical 480-token windower. | `sentiment.py`, `chunk_embedding_store.py`, `pseudo_labeling.py` (chunk-aware path) |
| `market_data.py` | yfinance client with 7-day holiday fallback + 5-day rolling volatility. Forward trading dates. | `main.py`, `forecaster.py` |
| `document_parser.py` | Paste / PDF / DOCX / URL ingestion behind `POST /documents/parse`. Lazy pdfplumber + python-docx. Async httpx for URL fetching. | `main.py` |
| `fomc_calendar.py` | Hardcoded Federal Reserve schedule 2023–2026. | `main.py` `GET /fomc/calendar` |
| `gemini_client.py` | Google Gemini 2.5 Pro / Flash calls for the LLM judge. | `pseudo_labeling.py` follow-up + `llm_judge.py` |
| `langsmith_client.py` | LangSmith tracing client. | `gemini_client.py` |
| `scraper*.py` | Six Fed-adjacent scrapers (FOMC base, beige book, chair/governor speeches, press conferences, congressional testimony, NY Fed Liberty Street). | Manual workflow ops |

### `backend/app/data/` — offline data pipeline

Capability-first entry points (use these in new code):

| File | Wraps | Purpose |
| --- | --- | --- |
| `source_ingestion.py` | `ingest_sources.py` | Pull rows from HF (`gtfintechlab/fomc_communication`), Kaggle (`drlexus/fed-statements-and-minutes`), scraped JSON, Op-Fed (`--include-op-fed`), GSS (`--include-gss-factors`). Writes `data/raw/phase2/source_registry.jsonl`. |
| `label_normalization.py` | `normalize_labels.py` | Map raw label strings to `{hawkish, dovish, neutral}`. Drop unmappable rows + log exceptions. Sample-weight by provenance. |
| `quality_validation.py` | `quality_checks.py` | Near-duplicate filter (0.97), text-hash collisions, leakage checks. |
| `training_package_builder.py` | `build_training_package.py` | Emit `registry_normalized.parquet`, `splits_train_val_test.parquet`, `fold_manifest_expanding_walk_forward.json`, `dataset_metadata.json`, `quality_reports/`. |
| `baseline_spec_generator.py` | `generate_baseline_run_specs.py` | Pre-run-configuration markdown per planned run. |
| `pipeline_data_prep.py` | — | Orchestrator for all of the above. `make data-prep` calls this. |

Training / evaluation harnesses:

| File | Purpose |
| --- | --- |
| `nlp_baseline_batch.py` | Official NLP baseline batch (BERT / FinBERT / FOMC-RoBERTa × 5 seeds). Zero-shot harness. |
| `finetune_batch.py` | Fine-tune full batch (6 encoders × 5 seeds). |
| `finetune_pilot.py` | Single-seed fine-tune. Writes `predictions.jsonl` for the cross-source analyzer. |
| `attention_ablation.py` | Variant A / B / C ablation sweep (6 cells × N seeds). |
| `pseudo_labeling.py` | Chunk-aggregated teacher for the unlabelled scraped pool. Strategies: `chunk_max_pool` (default), `chunk_mean_pool`, `chunk_vote`, `doc_truncated` (legacy). |
| `llm_judge.py` | Gemini judge + three gating policies + 100-row audit sampler + Cohen's κ. |
| `continued_pretraining.py` | MLM continued pretraining of FinBERT-FedAdjacent on the 9.7k unlabelled scraped rows. Checkpoint not yet trained. |

Per-source scrapers (`backend/app/data/sources/`):

```
sources/
├── base.py                 BaseSourceScraper protocol + SourceMetadata + 10-value source_type whitelist
├── registry.py             SOURCES dict, register() decorator
├── beige_book.py           Beige Book regional summaries
├── governor_speeches.py    Federal Reserve governor speeches (excludes chair via _CHAIR_PATTERN)
├── press_conference.py     FOMC press-conference transcripts (PDF-extracted via pypdf)
├── regional_research.py    NY Fed Liberty Street Economics
└── testimony.py            Congressional testimony
```

### `backend/app/evaluation/` — analytical outputs

| File | Purpose |
| --- | --- |
| `bootstrap.py` | `block_bootstrap_ci()` with 6-month blocks, 1000 resamples. |
| `calibration.py` | `coverage_curve()` for the 80% conformal bands. |
| `conformal.py` | Split-conformal calibration with Lei–Wasserman finite-sample correction. Manifest persisted as `forecaster_best.conformal.json` next to the checkpoint. |
| `regime_aggregator.py` | Aggregate runs across pre-2020 / 2020-shock / 2022-hike windows. Emits `ci_lo` / `ci_hi` columns. |
| `xai.py` | `attribute_text()` keyword-salience over hawkish/dovish weight dictionaries. Per-sentence + per-token attributions. |

### `backend/app/features/` — feature engineering modules

| File | Purpose |
| --- | --- |
| `credibility.py` | `CredibilityVector` dataclass + helpers (drift vs prior 4 docs, realized-vs-stated 90d Pearson, SEP-vs-OIS terminal gap, months-since-reversal). Backend wiring into the forecaster input pipeline is the Phase 4.4 follow-up bundle. |

### `backend/app/models/` — model definitions

| File | Purpose |
| --- | --- |
| `embedding_adapter.py` | `768→128` projection + LayerNorm + GELU. Replaces the zero-init `768→8` bottleneck. |
| `registry.py` | YAML loader for `registry.yaml`. |
| `registry.yaml` | HF checkpoint pinning (model id + revision SHA + task). |

### `backend/app/training/` — training-side utilities

| File | Purpose |
| --- | --- |
| `manifest.py` | Run-manifest writer (dataset version + feature version + seed + training config + metrics). |
| `config_loader.py` | YAML loader for ablation configs under `configs/`. |

### `backend/app/middleware/` — request middleware

| File | Purpose |
| --- | --- |
| `errors.py` | `RunIdMiddleware` binds a fresh `run_id` per request to the structlog ContextVar. Maps `RequestValidationError` → 422, `ValueError` → 422, bare `Exception` → 500 with `{code, detail, run_id}`. |

### `frontend/`

```
frontend/
├── pages/
│   ├── index.tsx              Server-side redirect to /analyze
│   ├── analyze.tsx            Main dashboard (multi-axis cards + forecast charts + XAI + credibility + history hook)
│   ├── history.tsx            Persisted analyses with filters
│   ├── calendar.tsx           FOMC calendar widget
│   ├── preview.tsx            Component showcase (internal)
│   └── legacy.jsx             The v1 dashboard kept for A/B during defense rehearsals
├── components/
│   ├── analyze/
│   │   ├── AnalyzeForm.tsx           input form
│   │   ├── DocumentIngestionTabs.tsx paste / PDF·DOCX / URL tabs (calls /documents/parse)
│   │   ├── MultiAxisCards.tsx        stance / factor / certainty / topic
│   │   ├── ForecastChart.tsx         close + volatility charts with confidence bands
│   │   ├── XaiPanel.tsx              per-sentence highlights + token tooltips
│   │   ├── CredibilityPanel.tsx      drift badge + 4 sub-stats (currently fixture-driven)
│   │   ├── MarketContext.tsx, MarketSnapshot.tsx, RealTrainStatus.tsx, PredictionCards.tsx, SentimentCard.tsx, ErrorBadges.tsx, WatchlistChips.tsx, PreviewPanels.tsx
│   ├── shell/                  AppHeader, AppNav, AppFooter, Header
│   ├── ui/                     shadcn primitives (button, card, dialog, tabs, tooltip, select, skeleton, progress, …)
│   └── theme-toggle.tsx
└── lib/
    ├── analyze/
    │   ├── api.ts          axios wrappers
    │   ├── types.ts        TypeScript interfaces mirroring the Pydantic models
    │   ├── constants.ts    UI strings, horizon options
    │   ├── fixtures.ts     mock data for components
    │   ├── derive.ts       chart-series builders + error-metric helpers
    │   └── format.ts       number/currency/percent formatters
    ├── watchlist.ts        localStorage CRUD
    └── utils.ts            classname helpers
```

### `data/` — runtime volume

```
data/
├── *.json                                  Scraped raw JSON per source (8 files; 92 rows total)
├── raw/phase2/source_registry.jsonl        Unified pre-normalisation registry
├── interim/phase2/                         Mid-pipeline parquet caches
├── processed/<training_package_id>/        One directory per published training package
│   ├── registry_normalized.parquet
│   ├── splits_train_val_test.parquet
│   ├── fold_manifest_expanding_walk_forward.json
│   ├── dataset_metadata.json
│   ├── chunk_embeddings.parquet            (live HF embeddings, per chunk)
│   └── quality_reports/
├── artifacts/                              Run outputs
│   ├── phase3/                             NLP baseline batch artefacts
│   ├── phase4_attention/                   Variant ablation artefacts + holdout_summary
│   ├── phase4_llm_zero_shot/, phase4_embedding_comparator/
│   ├── pseudo_label_audits/                audit_set.csv, audit_metrics.json, policy_sweep.json
│   ├── audit.log                           Append-only JSONL hook log
│   └── experiments/<run_id>/               Per-run artefacts
├── external/                               Manually-staged external corpora
│   ├── op_fed/                             Keith et al. 2025 (1,044 sentences, MIT)
│   ├── financial_phrasebank/               Malo et al. 2014 (reserved for continued-pretraining)
│   ├── gss/                                Gürkaynak-Sack-Swanson 2005 IJCB data appendix + parsed CSVs
│   └── README.md                           Per-source status table
├── db/fed_pulse.db                         SQLite history database
└── schema/labels.yaml                      Multi-axis label schema
```

### `tests/`

```
tests/
├── unit/                  ~54 files, per-module coverage
├── contract/              OpenAPI snapshot diff
├── properties/            Hypothesis property tests (chronological event_date, no shared text_hash, no target leakage)
├── regression/            test_plan13_variant_a.py locks seed-11 wf_fold_3 numbers ±1e-4
├── e2e/                   test_api_e2e.py (health + single /analyze)
└── snapshots/             OpenAPI snapshot
```

### `scripts/`

| File | Purpose |
| --- | --- |
| `build_toy_snapshot.py` | 50-event toy snapshot for `make verify`. |
| `inventory_corpora.py` | Probes external corpora URLs + license terms. |
| `regen_openapi_snapshot.py` | Regenerate `tests/snapshots/openapi.json` after API changes. |
| `snapshot_market_data.py` | One-shot yfinance pull. |
| `verify_smoke.py` | End-to-end smoke for `make verify`. |
| `extract_gss_factors.py` | Parse the GSS 2005 IJCB data appendix PDF into `gss_factors.csv` + `gss_surprises.csv`. Run via the backend container so pdfplumber is available. |

### `docs/` — executable contracts

These four short markdown files are what code and CI actually gate on:

| File | What it pins down |
| --- | --- |
| `benchmark-policy.md` | Required IDs on every official run, splits + seeds, leakage rules, NLP-baseline winner criteria. |
| `data-and-training-contracts.md` | Approved sources, ingestion contract fields, training package artifacts. |
| `run-templates.md` | Naming conventions for `run_id` + artifact directories. |
| `security-acceptance.md` | Known-accepted Next.js DoS advisories + pip-audit suppressions with mitigation owners. |

The wiki (sibling directory, separate checkout) carries the long-form material: 01 Progress Snapshot, 02 SRS, 03 System Architecture, 04 SDD, 05 Project Plan, 06 Deep Learning Roadmap, 07 Data Schema, 08 Test Plan, 09 Risk Register, 10 References, 11 ML Lifecycle, 12 ADRs, 13 External Corpora Inventory.

---

## 5. Reading order if you're new

1. **README.md** — 30-second elevator pitch + how to start the dev stack.
2. **This file** — orient yourself on what lives where.
3. **`../../fed-pulse.wiki/01_Progress_Snapshot.md`** — what's done as of today, what's in flight, active risks.
4. **`../../fed-pulse.wiki/05_Project_Plan.md`** — phase roadmap with hard deadlines and the Gantt chart.
5. **`backend/app/main.py`** — every endpoint, in one file. Read top to bottom.
6. **`backend/app/services/forecaster.py`** — the longest file; skim once to know it exists. Decomposition is Phase 7.1.
7. **`docs/benchmark-policy.md`** — the rules every official run obeys.

---

## 6. "I want to …" — task-oriented entry points

### Add a new Fed-adjacent source

1. Write a scraper under `backend/app/data/sources/<source>.py` implementing `BaseSourceScraper` (see `governor_speeches.py` as the template).
2. Add the `source_type` value to `backend/app/data/source_type.py::SOURCE_TYPE_VALUES` AND the parallel `_VALID_SOURCE_TYPES` set in `backend/app/data/sources/base.py`.
3. Wire the scraper into the ingestion CLI in `backend/app/data/ingest_sources.py` (new `--include-<source>` flag + dispatch branch).
4. If the source is peer-reviewed, add it to `_PEER_REVIEWED_SOURCES` in `backend/app/data/normalize_labels.py`.
5. Add a `SOURCES.lock` JSON under `data/external/<source>/` with download URLs + SHA-256s.
6. Add unit tests under `tests/unit/test_<source>_*.py` (see `test_external_corpora_ingestion.py` for the pattern).
7. Update the wiki inventory at `../fed-pulse.wiki/13_External_Corpora_Inventory.md`.

### Add a new API endpoint

1. Define request/response Pydantic models in `backend/app/schemas.py` (use `_FORBID_FROZEN_CONFIG` for responses, `_STRICT_REQUEST_CONFIG` for requests).
2. Add the route in `backend/app/main.py`. If it does sync work, wrap in `await run_in_threadpool(...)`.
3. Regenerate the OpenAPI snapshot: `python scripts/regen_openapi_snapshot.py`. Commit `tests/snapshots/openapi.json`.
4. Add unit tests under `tests/unit/test_main_api*.py`.

### Change the forecaster

1. Edit `backend/app/services/forecaster.py`. Keep the `forecast_quantitative_series` signature stable — `/analyze` depends on it.
2. If you touch the confidence-band path, also touch `backend/app/evaluation/conformal.py`.
3. Run the regression test: `pytest tests/regression/test_plan13_variant_a.py` — this locks seed-11 wf_fold_3 numbers within ±1e-4.

### Score the unlabelled pseudo set

```
python -m app.data.pseudo_labeling \
  --teacher-checkpoint <FinBERT-FOMC seed 71 dir> \
  --strategy chunk_max_pool --tau-chunk 0.50 --threshold 0.85 \
  --input data/raw/phase2/source_registry.jsonl \
  --output data/interim/phase2/registry_pseudo.jsonl
```

Then sample a 100-row stratified audit via `llm_judge.sample_audit_set(rows, n=100)`, hand-label, re-run with the judge confirmation filter, check teacher precision ≥ 0.90 (audit gate from `docs/benchmark-policy.md`).

### Tweak the frontend dashboard

1. The page lives at `frontend/pages/analyze.tsx`. The component tree is mostly under `frontend/components/analyze/`.
2. Fixtures for fixture-driven cards live in `frontend/lib/analyze/fixtures.ts`.
3. Run `npm test` (vitest) + `npm run type-check` + `npm run build` before pushing.

### Add a new training-package source

1. Drop the source files under `data/external/<source>/` with a `SOURCES.lock`.
2. Add a loader in `backend/app/data/ingest_sources.py` mirroring `_iter_op_fed_records` (CSV) or `_iter_hf_records` (HF datasets).
3. Add a `--include-<source>` CLI flag + an entry in the `--all-sources` fan-out branch.
4. If peer-reviewed, register in `_PEER_REVIEWED_SOURCES` in `normalize_labels.py`.

### Update the wiki

The wiki lives at `../fed-pulse.wiki/`. It is a separate git remote; `cd` over and `git commit && git push origin master` like any repo. Code repo + wiki should stay in sync — when a code change drifts from a wiki page, the PR description should call out the page that needs updating.

### Run the official benchmark

```
make data-prep DATASET_VERSION=v1 FEATURE_VERSION=v1 OWNER=<who>
# data-prep prints the training_package_id; pass it in:
make train-batch TRAINING_PACKAGE_ID=<id> OWNER=<who>
```

---

## 7. What's currently dead vs alive

The audit ran on `dev` HEAD before PR #120. After the merge, the status of the modules under `backend/app/data/` is:

| File | Status | Notes |
| --- | --- | --- |
| `pseudo_labeling.py` | **alive** | Chunk-aggregated teacher landed in PR #120. The primary path for pseudo-labelling the 9,696-row unlabelled pool. |
| `llm_judge.py` | **alive** | Judge confirmation filter on top of pseudo-labelling. Three gating policies + audit sampler. |
| `chunk_embedding_store.py` | **alive** | Persists per-document CLS embeddings used by both the Phase-4 chunk-attention pooler AND the new chunk-aggregated teacher. |
| `chunk_embedding_retrieval.py` | **alive** | Lookback windowing for the chunk pooler. |
| `llm_embedding_store.py` | partially alive | Variant C (Gemini embeddings) cell of the ablation grid. May be retired if Variant C is dropped from the final benchmark. |
| `embedding_comparator.py` | historical | MiniLM frozen-head baseline (Phase 4 #35). One run already published. |
| `llm_zero_shot_execution.py` | historical | Qwen-3B zero-shot baseline (Phase 4 #26). One run already published; stronger model rerun is a Phase-5 workflow operation. |
| `source_type_stratified_analysis.py` | historical | Joiner utility for per-source-type tables. Run after each fine-tune batch. |
| `nlp_baseline_batch.py` / `finetune_batch.py` / `finetune_pilot.py` / `attention_ablation.py` | **alive** | Make targets call these. Renamed from `phase3_*` / `phase4_attention_ablation` in Phase 7.2. |
| `continued_pretraining.py` | alive but unused | MLM pipeline ready; checkpoint not yet trained. |

---

## 8. Wiki ↔ code mapping

| Wiki page | Code touchpoint |
| --- | --- |
| `02_SRS.md` (FR-01 … FR-43) | `main.py` endpoints, `services/`, `models/`, `evaluation/` |
| `03_System_Architecture.md` (C4 diagrams) | `main.py` + `services/` directory + `db.py` + `audit.py` |
| `04_SDD.md` (service-level design) | Per-service docstrings in `services/` |
| `06_Deep_Learning_Roadmap.md` (Variants A/B/C, pseudo-labelling) | `services/forecaster.py`, `data/attention_ablation.py`, `data/pseudo_labeling.py` |
| `07_Data_Schema.md` (ER + table layouts) | `data/schema/labels.yaml`, `db.py` models, the parquet outputs under `data/processed/<package_id>/` |
| `08_Test_and_Verification_Plan.md` | `tests/properties/test_no_leakage.py`, `tests/regression/test_plan13_variant_a.py`, `tests/contract/test_openapi_snapshot.py` |
| `09_Risk_Register.md` | The known threats in `01_Progress_Snapshot.md §Risks` — see `docs/security-acceptance.md` for the security slice |
| `12_ADRs.md` | `docs/adr/` (when Phase 2.1 ships the directory split) |
| `13_External_Corpora_Inventory.md` | `data/external/`, `backend/app/data/ingest_sources.py` |

If you change architecture, schema, API surface, or evaluation protocol, name the wiki page in your PR description.

---

## 9. Conventions

- **Branches:** `feat/<short-desc>`, never push to `main`, always PR into `dev`.
- **Commits:** present-tense imperative ("add the X", not "added the X"). No AI co-author trailers.
- **PRs:** terse, in the user's voice. ≤8 summary bullets, ≤15 words each, ≤3 test-plan commands. The PR description is the durable accounting; don't repeat it in commit bodies.
- **Tests:** lint + typecheck + pytest + frontend + vuln-python + vuln-npm jobs must all be green before merge.
- **Data conventions:** `BACKEND_ROOT` resolves via `Path(__file__).resolve().parents[2]`; `DEFAULT_DATA_DIR` falls back to `BACKEND_ROOT.parent / "data"` when `/data` is absent. `make` targets fail fast on missing `DATASET_VERSION` / `FEATURE_VERSION` / `TRAINING_PACKAGE_ID` — don't paper over with defaults.

---

## 10. When in doubt

- Read the docstring at the top of the file. Most modules have a 5-line summary.
- Grep for the function name across `tests/unit/` — the test usually shows how the function is meant to be used.
- The wiki's `01_Progress_Snapshot.md` carries the freshest "what's done, what's in flight, what's at risk" snapshot.
- The Makefile is the contract for every workflow — if it isn't in the Makefile, it isn't a supported flow.
