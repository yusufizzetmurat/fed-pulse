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

## Event-row dataset (Phase 8)

`backend/app/data/event_dataset_builder.py` produces two Phase-8 artifacts
in a single CLI run, both under `data/processed/<training_package_id>/`:

- `events.parquet` — **collapsed view**. One row per
  `(event_date, event_kind, asset_symbol, horizon)`. Multi-source
  duplicates are pinned to one preferred source via
  `_SOURCE_PREFERENCE`. Use this for "one event one row" stats.
- `events_full.parquet` — **full view**. One row per
  `(event_date, event_kind, source, asset_symbol, horizon)`. Every
  source survives so sentence-level / source-stratified analyses can
  read the raw shards directly. Same column schema as the collapsed
  view; the extra column `source_record_id` is populated in both.

Both parquets are byte-identical on identical inputs (deterministic +
idempotent). The same training package can be rebuilt repeatedly.

Event kinds: `{statement, minutes, speech, testimony, press_conference}`.
`document_type` values from the registry are normalized via a fixed map
in the builder; speeches (chair + governor), congressional testimonies,
press conferences and FOMC meeting transcripts are accepted alongside
statements/minutes.

Required columns (full schema in module docstring):

- `event_date`, `event_kind`, `document_id`, `text_hash`, `source`
- `as_of_ts` — placeholder announcement time. FOMC kinds use
  `<event_date>T19:00:00Z` (2pm ET), speeches use `T14:00:00Z`. A future
  PR (OIS surprise, #146) can replace these with real timestamps without
  changing column semantics.
- `text`, `token_count`
- Multi-axis labels: `axis_stance`, `axis_time`, `axis_certainty`,
  `axis_factor`, `axis_topic`. Pulled from `mapped_label` plus the
  per-source `axes` payload in the registry; None where unavailable.
- Credibility 4-vector: `credibility_drift_score`,
  `credibility_realized_vs_stated_gap`, `credibility_market_implied_gap`,
  `credibility_months_since_reversal`. Reuses
  `app.services.credibility_loader.load_credibility_for_run`; missing
  inputs degrade to zeros (semantics: "credibility unknown").
- 20 trading-day prior market window: `prior_window_sha256` plus
  `prior_bars_json` (JSON-encoded list of bars with `date`, `close`,
  `volume`, `vol_5d`, `cum_return_20d`). Window ends strictly before
  `as_of_ts.date()` — the no-look-ahead contract is enforced by an
  assertion in `_build_prior_window` and `_assert_no_lookahead`.
- Per-horizon targets: `horizon ∈ {1, 5, 10, 30}` (trading days).
  Base close = last trading day strictly before `as_of_ts`. Target close
  = h-th trading day on-or-after `event_date`. Returns are simple
  close-to-close.
  - `realized_return` — raw target return
  - `abnormal_return` — `realized_return - (alpha + beta * benchmark_return)`.
    When `asset_symbol == benchmark` (default `^GSPC` vs `^GSPC`) this is
    just the raw return (alpha=0, beta=1 by contract).
  - `alpha`, `beta` — OLS on the trailing 252 trading-day window ending
    strictly before `as_of_ts`.
  - `direction_t1d` — sign of the t+1d realized return (-1, 0, +1).
- `volatility_shift` — post-event 10d realized vol minus pre-event 10d
  realized vol (log returns; sample std).
- `concurrent_macro_release` — boolean. True when a major US macro
  release (CPI, NFP, ISM) falls within ±2 trading days. Flagged only;
  no event is dropped on this basis. The calendar is loaded from
  `data/external/macro_releases.csv` (bundled in the repo; CPI dates
  2008-2026 hand-encoded against BLS published Schedule of Releases
  archives, NFP/ISM 1977-2026 rule-generated with federal-holiday
  forward shifts, CPI 1977-2007 second-Wednesday fallback; refreshable
  from FRED via `app.data.macro_releases.refresh_from_fred`). When the
  CSV is absent the builder falls back to a rule-based heuristic
  (first Friday / second Wednesday / first business day) so the smoke
  run still works on a fresh checkout. On the Sprint 1 package the
  swap from heuristic to real BLS/ISM dates lifts the flagged rate
  from 43.46 % to 49.31 % — real CPI releases tend to land closer to
  FOMC meeting days (mid-month) than the second-Wednesday rule
  estimates, so the real calendar is methodologically more accurate
  even though it flags more events. Tightening the ±2-trading-day
  radius is an option if the higher rate proves too noisy for the
  downstream confounder analysis.
- `asset_symbol` — default `^GSPC`. Per-asset rows are supported: a
  future sweep can rebuild with `--asset NDX` etc. without touching the
  schema.

Multi-source dedup: when several registry sources carry the same
`(event_date, event_kind)` (common for FOMC statements: scraped_fed,
vtasca, hf, kaggle, gtfintechlab), the builder selects exactly one
source per event via the preference order
`scraped_fed > vtasca > op_fed > gtfintechlab > hf > kaggle > gss_factor`.
Sentence-level shards inside the chosen source are concatenated in
`source_record_id` order.

Hard guarantees:

1. No look-ahead. Last prior bar's date is strictly < `as_of_ts.date()`.
   The market-model regression window ends strictly before that bar.
2. No survivorship filter. Every FOMC event with text + event_date + a
   usable prior window emits a row, regardless of post-event move.
3. Deterministic. Same training package → same parquet bytes.
4. Idempotent. Re-running overwrites with identical content.

CLI:
```
python -m app.data.event_dataset_builder \
    --training-package-id <id> --asset ^GSPC \
    --output events.parquet --full-output events_full.parquet
```
Both parquets are always written; pass `--full-output ''` to skip the
full view. The yfinance fetch is cached at
`<package_dir>/_market_cache/<symbol>.parquet` with a `SOURCES.lock`
entry. Re-runs use the cache; pass `--market-cache-dir` to relocate it.
Override the macro release calendar with `--macro-release-csv`.

Sprint 1 reference counts (training package
`tp_v2_sprint1_2026_05_15_sentiment_market_core_v1.0_epv1_v1.0`,
`^GSPC` asset):

| Output                | Rows  | Unique events                          | `concurrent_macro_release` |
| --------------------- | ----- | -------------------------------------- | -------------------------- |
| `events.parquet`      | 4 103 | 1 026 (date × kind × preferred source) | 49.31 % (real BLS/ISM)     |
| `events_full.parquet` | 5 339 | 1 335 (date × kind × source)           | 49.31 % (same calendar)    |

Heuristic baseline on the same package was 43.46 %; the swap lifts the
rate because real CPI releases tend to land closer to FOMC days than
the second-Wednesday rule places them. The list of flagged dates is
deterministic — same input ⇒ bit-identical parquet bytes.
