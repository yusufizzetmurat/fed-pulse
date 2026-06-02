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
  realized vol (log returns; sample std). Both windows are strict-flank:
  the pre-window covers returns on days `t-10..t-1` and the post-window
  covers returns on days `t+1..t+10`, where `t` is the first trading bar
  on or after the event. `close[t]` participates only as the denominator
  of the post-window's first return, never as a numerator, so the
  announcement-day close-to-close move does not enter either flank.
- `forward_realized_vol_10d` — strict-forward 10-day realised vol over
  `t+1..t+10`, sample std of log returns (ddof=1). This is the target
  column for the Phase 9 V2 vol-regime classifier (#195). Per-fold
  quantile cutoffs map this continuous value to a 3-class label at
  training time (train-slice fit only; never on val / test).
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
- `intra_meeting_stance_shift`,
  `intra_meeting_certainty_shift`,
  `intra_meeting_factor_shift` — signed within-meeting tone shift
  between the press-conference and statement rows that share an
  `event_date`. The shift is computed as `press_conference_axis -
  statement_axis` after encoding each side through
  `_INTRA_MEETING_AXIS_ENCODING`. Categorical stance values are
  encoded `hawkish=+1, dovish=-1, neutral=0`; categorical certainty
  values, when present, are encoded `certain=+1, neutral=0,
  uncertain=-1`. Numeric values (regression-typed `axis_certainty` /
  `axis_factor` per `data/schema/labels.yaml`) pass through and are
  subtracted directly. When either kind is missing for the date, all
  three shifts are `NaN` — never coerced to zero. Multi-source
  duplicates are collapsed via `_SOURCE_PREFERENCE` so the shift uses
  the preferred statement / press-conference pair only; the same
  per-date shift value is replicated to every row sharing that date
  on both the collapsed and full views.
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
## Phase 8 feature sources

Approved external feature parquets that downstream Phase-8 models
(#147 next-FOMC prediction, #148 cross-asset response) read directly.
Each parquet ships with a SOURCES.lock entry recording its sha256,
methodology label, and row count so reproductions are auditable.

### `mp_surprises.parquet` — monetary-policy surprise time-series

Path: `data/external/fred/mp_surprises.parquet`. Built by
`backend/app/data/mp_surprise.py`; closes #146.

One row per FOMC meeting from 2010-01-01 to today, with:

- `event_date`, `meeting_id` (sequential)
- `ff_target_prior`, `ff_target_after` — fed-funds target rate
  reconstructed from `DFEDTAR` (1982-2008-12-15) joined to the
  `DFEDTARU` / `DFEDTARL` band midpoint (2008-12-16 onward)
- `mp_surprise_level` — change in the 1-month-ahead policy-rate proxy
  from t-1 EOD to t+1 EOD, in basis points
- `mp_surprise_path_factor` — first principal component of the
  level-residualised changes at {3, 6, 12}-month tenors. PCA is fit
  once on the full historical sample and the eigenvector is persisted
  in `SOURCES.lock[mp_surprises].path_factor_model` so re-builds are
  byte-identical.
- `pre_event_curve`, `post_event_curve` — JSON lists of
  `(months_ahead, implied_rate)` at {1, 3, 6, 12, 24}-month points
- `fed_info_factor` — residual of `mp_surprise_level` regressed on the
  SPX return around each announcement (Cieslak-Vissing-Jorgensen
  2021-style decomposition). The `fed_info_factor_source` column
  records which SPX series the row used:
  - `alphavantage_intraday_30min` — ±30 min SPY return from
    `app.data.alphavantage_spx`, cache at
    `data/external/alphavantage/spx_intraday_fomc_days.parquet`. This
    is the CVJ-faithful measurement when available.
  - `daily_window_proxy` — close-to-close return over the
    `[t-1, t+1]` window from the FRED-licensed daily SPX. Used as a
    fallback when the intraday cache does not cover the event date.
  - `unavailable` — neither source had coverage; the row stamps
    `fed_info_factor = None` so it is distinguishable from a real-
    but-tiny residual.
- `is_intermeeting` — true for unscheduled / emergency actions
  (2020-03-03 and 2020-03-15 in the bundled calendar)
- `methodology` — `ois_proxy` (Treasury-yield proxy via DGS1MO /
  DGS3MO / DGS6MO / DGS1 / DGS2; the **honest default**) or
  `ff_futures` (reserved for a future CME-settlement source)
- `data_version` — short sha capturing FRED series IDs + observation
  ends + calendar signature

Hard guarantees:

1. **No look-ahead.** `pre_event_curve` reads the last published yield
   strictly before `event_date`; `post_event_curve` reads the first
   strictly after. Enforced by an assertion in `_pre_post_yields`.
2. **Deterministic.** Same FRED inputs imply byte-identical parquet
   (sign-normalised eigenvector, snappy compression, sorted rows).
3. **Honest methodology label.** The freely-available FRED data does
   not include CME fed-funds-futures settlements; we proxy the
   surprise curve from Treasury constant-maturity yields. The
   `methodology` column records this on every row so downstream
   models can stratify by source quality.

CLI:

```
python -m app.data.mp_surprise \
    --start 2010-01-01 --end today \
    --output mp_surprises.parquet
```

The output parquet lands under `data/external/fred/`. Pass
`--methodology ff_futures` only when a real CME settlement source has
been wired (out of scope for #146).

### `macro_state.parquet` — FOMC decision-eve macro snapshot

Path: `data/external/fred/macro_state.parquet`. Built by
`backend/app/data/macro_state.py`; closes #147. One row per business
day in `[start, end]` carrying the last published value strictly
before that day for every FRED indicator listed below. The as-of join
applies `publication_delay_days` (default 30) to monthly observations
to mirror the conservative BLS / BEA release lag.

Columns (twelve numeric series plus the four bookkeeping columns):

| column            | FRED series      | transform                                    |
| ----------------- | ---------------- | -------------------------------------------- |
| `unrate`          | `UNRATE`         | level (% civilian unemployment)              |
| `cpi_yoy`         | `CPIAUCSL`       | 12-month log-change × 100 (% YoY)            |
| `core_pce_yoy`    | `PCEPILFE`       | 12-month log-change × 100 (% YoY)            |
| `ism_proxy`       | `MANEMP`         | 3-month % change (documented NAPM proxy)     |
| `payems_mom`      | `PAYEMS`         | MoM change in thousands of jobs              |
| `rsafs_mom`       | `RSAFS`          | MoM % change in retail sales                 |
| `treas_10y`       | `DGS10`          | level (% 10y Treasury constant maturity)     |
| `slope_10y_2y`    | `T10Y2Y`         | level (% 10y - 2y slope)                     |
| `slope_10y_3m`    | `T10Y3M`         | level (% 10y - 3m slope, recession signal)   |
| `hy_oas`          | `BAMLH0A0HYM2`   | level (% ICE BofA HY OAS)                    |
| `nfci`            | `NFCI`           | level (Chicago Fed NFCI; 5-day pub delay)    |
| `tips_10y_real`   | `DFII10`         | level (% 10y TIPS real yield)                |

Plus `as_of_date` (ISO date string), `ism_proxy_source`
(`MANEMP_3m_pct`), `publication_delay_days` (monthly-panel delay), and
`data_version` (short sha over FRED inputs + rates-panel delay map).
Daily Treasury / spread / OAS / TIPS series ship with a zero-day
publication delay; NFCI carries a 5-day delay so a Friday-dated print
is visible the following Wednesday. The SOURCES.lock entry persists
the per-series delay map and the column mapping so the join contract
round-trips through the artefact.

CLI:

```
python -m app.data.macro_state \
    --start 2010-01-01 --end today \
    --output data/external/fred/macro_state.parquet
```

## Encoder embedding cache

Sentence-embedding and chunk-pool encoders cache one parquet per
`(encoder_alias, training_package_id)` under
`data/raw/embeddings/<encoder_alias>_<rev_slug>.parquet`. Each row has
`record_id`, `doc_id`, `event_date`, `chunk_index`, `chunk_preview`,
and a `embedding` list. The per-encoder SOURCES.lock at
`data/raw/embeddings/SOURCES.lock` is JSON Lines (one entry per
encoder revision) and carries the encoder alias, model repo /
revision, registry sha256, parquet sha256, row count, and retrieval
timestamp.

The cache set covers the bake-off encoders shipped through
`backend/app/data/embedding_cache.py` (FinBERT, FinBERT-FOMC,
BGE-large-en-v1.5, Nomic-embed-text-v1.5, FinBERT-fed-adjacent,
BERT-base-fed-adjacent) plus voyage-finance-2 (1024-dim,
finance-tuned) served by the hosted Voyage REST API. Voyage entries
land via `scripts/cache_voyage_embeddings.py --allow-network`; the
SOURCES.lock format is identical to the Hugging Face encoders so a
downstream consumer reads voyage rows with no shape change.

## LLM-features cache immutability

The B1 (#212) LLM-as-features cache at
`data/raw/llm_features/claude-sonnet-4-6_2026-05-19.v1/tp_v2_sprint1_2026_05_15_sentiment_market_core_v1.0_epv1_v1.0.parquet`
is the authoritative artefact for the §6.6 Tier 4 / Tier 5 results.
The catalogue was extracted with a frontier instruction-following LLM
at temperature 0; reproducibility binds to that specific model snapshot
(the specific model and version are pinned in the code). The provider
will eventually retire that snapshot, after which the extraction script
cannot reproduce the cache byte-for-byte from a future model — a
successor model would shift category-level agreement on the 10-feature
catalogue even at the same temperature.

Rule:

1. **Do not delete or regenerate the cache file.** It is the only
   reproducible artefact for the §6.6 LLM-features experiment once
   `claude-sonnet-4-6` is deprecated. If the cache is lost, the
   experiment is irreplicable and the §6.6 Tier 4 / Tier 5 rows
   cannot be re-derived from a future model snapshot.
2. **Integrity is enforced by the registry SHA pin.** The cache
   `sha256` + `size_bytes` are pinned in
   `backend/app/models/registry.yaml` under the `llm_features:`
   top-level block; a tampered cache file fails the
   `tests/unit/test_llm_features_pin.py` integrity check.
3. **A regeneration requires an ADR.** Any change to the catalogue,
   the model snapshot, or the cache content must land as an
   `Architecture Decision Record` in
   `fed-pulse.wiki/12_Architecture_Decision_Records.md` together
   with a registry pin bump; silent regenerations are forbidden.

## Structured linguistic features (Phase 8)

`backend/app/features/linguistic.py` emits a 15-dim interpretable
linguistic feature vector per document, keyed by `text_hash` so it joins
directly onto event rows or any other registry-derived table.

Output artifacts under `data/processed/<training_package_id>/`:

- `linguistic_features.parquet` — one row per unique `text_hash`. 15
  numeric columns: 5 named LDA topic shares (`inflation`, `employment`,
  `financial_stability`, `growth`, `balance_sheet`), 3 misc topic shares
  (`misc_1..3`), `hedge_density`, `comparison_density`,
  `forward_density`, `concrete_ratio`, `hawk_dove_asymmetry`,
  `log_token_count`, `pivot_distance`.
- `linguistic_lda_model.pkl` — pickled `(CountVectorizer, LatentDirichletAllocation)`
  bundle plus the slot→topic-index assignment map. Sufficient to score
  any new document without re-fitting.
- `linguistic_lda_topics.json` — top-15 vocabulary words per topic plus
  the human label, coherence notes, and configuration constants
  (`random_state=11`, `num_topics=8`, `max_iter=50`). The wiki entry
  reads this file directly.

LDA fit is deterministic: `random_state=11`, batch learning,
`max_iter=50`, fixed `CountVectorizer` cutoffs. The hand-crafted
densities are pure functions of the document text — scrambling the
order of other documents in the corpus does not change any single
document's feature row beyond the LDA fit dependency, which is itself
permutation-invariant under sklearn's batch LDA with a fixed seed.

CLI:
```
python -m app.features.linguistic \
    --training-package-id <id> \
    --output linguistic_features.parquet
```

Sprint 1 reference counts (training package
`tp_v2_sprint1_2026_05_15_sentiment_market_core_v1.0_epv1_v1.0`):

| Output                          | Rows   | Notes                                  |
| ------------------------------- | ------ | -------------------------------------- |
| `linguistic_features.parquet`   | 16 721 | one row per unique `text_hash`         |
| `linguistic_lda_model.pkl`      | n/a    | `CountVectorizer` + LDA, seed=11       |
| `linguistic_lda_topics.json`    | n/a    | 8 topics × 15 words + coherence audit  |

Coherence on the Sprint 1 fit (see `linguistic_lda_topics.json`):

Seed-overlap floor (`MIN_SEED_OVERLAP=2`, top-10): every named slot
emitted in `linguistic_features.parquet` is guaranteed to have at
least two of its seed words inside the winning topic's top-10
vocabulary. Slots that fail the floor are emitted as `0.0` and their
candidate topics fall to `misc_*`. This blocks the prior failure mode
where `topic_share_employment` was silently measuring QE language
(topic 5: `committee, federal, policy, securities, rate, ..., agency,
..., purchases`).

Three named slots clear the floor on the Sprint 1 fit:

- `financial_stability` (topic 3) — overlap: `{credit, financial}`.
- `balance_sheet` (topic 5) — overlap: `{agency, securities}`. This
  is the QE / asset-purchases topic; pre-fix the seed-assignment
  race had this topic mislabeled as `employment`.
- `growth` (topic 6) — overlap: `{growth, spending}`.

Two named slots fall to misc (emitted as `0.0`):

- `inflation` — top-10 of topic 0 contains only `{inflation}` from
  the inflation seed list (count = 1, below floor). The topic is
  inflation-heavy in posterior mass, but the seed list as currently
  written does not have a second high-frequency seed in the corpus
  top-10. Honest miss; reviewable in `linguistic_lda_topics.json`.
- `employment` — best candidate topic has zero labor seeds in its
  top-10. The floor blocks the assignment; pre-fix this was the
  silent-mislabel bug flagged by reviewer audit of PR #155.

Misc slots: five LDA topics fall to misc after the floor; only the
first three populate `topic_share_misc_1..3`. The 14-column schema
is preserved.

Open follow-up: raising `num_topics` to 10-12 and widening the
inflation seed list are out of scope for this correctness fix and
will be separate PRs after the bake-off / forecaster sweep produce
results.

### `pivot_distance` — token diff vs prior FOMC statement

The 15th column captures how much a given FOMC statement deviates in
vocabulary from the previous statement. It is the token-set Jaccard
distance `1 - |A ∩ B| / |A ∪ B|` between the normalised token sets of
the current document and the latest preceding row whose `event_kind`
is `statement` and whose `event_date` is strictly earlier. The
tokeniser is the same `_TOKEN_RE` that backs the hand-crafted
densities (case-folded alphanumeric runs).

NaN semantics:

- `pivot_distance = NaN` when `event_kind != "statement"` — minutes,
  press conferences, speeches and testimonies follow different
  stylistic conventions, so the diff is undefined.
- `pivot_distance = NaN` for the first statement in the corpus (no
  strictly-earlier peer).
- Same-date duplicates share the same earlier prior; none of them
  becomes the prior for any other same-date peer because the
  contract requires `as_of_ts < current.as_of_ts` strictly.

On the Sprint 1 fit the distribution audit is reproduced by re-running
`make data-prep`. Placeholder ranges (fill in after the next pipeline
run): `pivot_distance` ranges roughly `[<min>, <max>]` with mean
`<mean>` across the statement rows of
`tp_v2_sprint1_2026_05_15_sentiment_market_core_v1.0_epv1_v1.0`. The
non-statement rows are NaN by construction and excluded from the
summary.

## Next-FOMC decision dataset (Phase 8)

The next-FOMC decision forecaster
(`backend/app/forecasting/next_fomc_decision.py`, closes #147) reframes
the project from price-forecasting to central-bank-forecasting. It
predicts the rate decision at meeting `N+1` given features known
strictly before meeting `N+1`'s `as_of_ts`.

### Target

Reconstructed from `mp_surprises.parquet`:

    delta_bp = (ff_target_after_N1 - ff_target_prior_N1) * 100

Mapped to the ordinal class set
`{cut_50, cut_25, hold, hike_25, hike_50, hike_75}` with a 12.5 bp
slack (half a 25 bp step). Deltas outside the set (e.g. the March
2020 75 bp emergency cuts, the October 2008 emergency 50 bp cut
sequence) emit a `UserWarning` and the row is dropped from the
supervised set so jumbo intermeeting moves surface explicitly.
Intermeeting meetings are excluded from the *target* role but still
contribute as a feature meeting when the next scheduled meeting is
the supervisor.

### Feature matrix join

Per-meeting feature row at meeting `N`:

- `events.parquet` (`data/processed/<pkg>/`) provides multi-axis
  stance / time / certainty / factor / topic and the 4-vector
  `credibility_*`. Multi-source duplicates collapse to one row per
  `event_date` via the same preference order
  (statement -> press conference -> minutes -> first available).
- `mp_surprises.parquet` (`data/external/fred/`) provides the 5-tenor
  `pre_event_curve`, `mp_surprise_level`, `mp_surprise_path_factor`,
  `fed_info_factor`, and the `ff_target_prior` / `ff_target_after`
  used to reconstruct the target.
- `linguistic_features.parquet` (`data/processed/<pkg>/`) joins on
  `text_hash` and contributes the 15 structured features
  (14 from #149 plus `pivot_distance` from #166).
- `macro_state.parquet` (`data/external/fred/`) provides per-as-of-date
  snapshots of UNRATE, CPI YoY, core PCE YoY, ISM proxy
  (`MANEMP_3m_pct`, documented substitute for the paywalled NAPM
  series), nonfarm-payroll MoM change, retail-sales MoM, and the
  rates + financial-conditions panel: 10-year Treasury yield
  (`treas_10y`), 10y-2y slope (`slope_10y_2y`), 10y-3m slope
  (`slope_10y_3m`), ICE BofA High-Yield OAS (`hy_oas`), Chicago Fed
  National Financial Conditions Index (`nfci`, 5-day publication
  delay), and 10y TIPS real yield (`tips_10y_real`). Twelve numeric
  columns in total alongside `as_of_date`, `ism_proxy_source`,
  `publication_delay_days`, and `data_version`.

### OIS-implied baseline (sigma = 12.5 bp)

For every held-out meeting `M`, the baseline reads
`mp_surprises.parquet`'s `pre_event_curve` at the 3-month tenor
*for meeting M itself* (which is published the trading day before
`M.as_of_ts` -- strictly before, so no look-ahead). The OIS-implied
next-meeting rate change in basis points is
`(pre_curve_3m - ff_target_prior) * 100`, smoothed with a Gaussian
of sigma 12.5 bp over the 6-class set. 12.5 bp is half the smallest
non-zero class step (25 bp) so the kernel partitions the bp axis at
class midpoints without aliasing one class onto its neighbour. The
choice is pinned by `next_fomc_decision.OIS_BASELINE_SIGMA_BP` and
asserted in `tests/unit/test_next_fomc_decision.py`.

### Walk-forward CV

Leave-one-meeting-out: at meeting `M+1`, the train set is every
supervised row whose `target_event_date < M+1.target_event_date`.
The constructor asserts this strict inequality on every fold. Train
folds with fewer than 6 rows (one per class) fall back to baselines
only.

### Artifact layout

Outputs land under `data/artifacts/next_fomc/`:

- `results.json` -- per-meeting predictions for every model.
- `metrics.json` -- Brier (multi-class), multi-class log-loss, top-1
  accuracy, macro-F1, confusion matrix. Reports both the full window
  and the pandemic-excluded window (`2020-04-01..2021-06-30`).
- `feature_attribution.md` -- ablation table:
  `ois_only`, `ois_text`, `ois_text_linguistic`, `ois_text_credibility`,
  `ois_text_macro`, `full`, plus the model-free `ois_baseline_only`
  and `naive_carry_only` rows for reference.

## Cross-asset response (Phase 8)

The cross-asset response head
(`backend/app/forecasting/cross_asset_response.py`, closes #148) reuses
the same per-meeting feature joins as the next-FOMC head and predicts
the cross-section of asset abnormal returns rather than a single Fed
decision class. Where the next-FOMC head asks "what will the Fed
do next?", this head asks "how does the basket move when the Fed
speaks?".

### Target

Per-row regression target is the `abnormal_return` column on
`events.parquet`. The event-row dataset builder (#145) already
produces one row per `(event_date, event_kind, asset_symbol, horizon)`
with the market-model residual at horizon `h` computed against a
trailing 252-day window strictly before `as_of_ts`. The cross-asset
head reshapes that to one supervised row per
`(meeting, asset, horizon)`.

### Asset universe

Read from `events.parquet`'s `asset_symbol` column. The canonical
issue-#148 universe is:

    ^GSPC ^IXIC ^DJI ^TNX DX-Y.NYB GC=F CL=F XLF XLK XLE

Whatever subset is actually present in the parquet is what we model;
the realised list is recorded under `metrics.json[asset_universe]`.
`--asset` flags on the CLI can restrict the universe further.

### Horizons

`1, 5, 10, 30` trading days (the canonical set the event-row builder
emits). `--horizon` flags restrict the modelled subset.

### Feature families

Same five families as the next-FOMC head (`ois`, `text`,
`linguistic`, `credibility`, `macro`); the helpers are imported
directly from `next_fomc_decision` so the feature-name contracts stay
in lock-step.

### Models

Per-cell (one regression per `(asset, horizon)`):

- **ridge** -- `sklearn.linear_model.Ridge(alpha=1.0)`. Headline
  model; L2-regularised linear fit on the joint feature matrix.
- **hist_gbt** -- `sklearn.ensemble.HistGradientBoostingRegressor`
  seeded with `random_state=11`. Non-linear comparator.

Optional pooled-panel exploration:

- **pooled_ridge** -- single `Ridge` on the stacked frame with
  per-asset and per-horizon one-hot dummies. Marked exploratory;
  documented in the module docstring.

### Baselines

- **zero_baseline** -- predicts `0` abnormal return. The strict
  null for a mean-zero residual.
- **ois_bp_baseline** -- OIS-implied basis-point signal from
  meeting `N`'s `post_event_curve` at the 1-month tenor minus
  `ff_target_after`, divided by 100 so the units roughly align with
  percentage abnormal returns. Same information cutoff as the
  model. Rate-sensitive cells (e.g. ^TNX) make this baseline
  competitive; sector-equity cells make it inflated. Documented in
  the module docstring caveat -- read it before drawing inference.

### Walk-forward CV

Leave-one-meeting-out *per cell*. For held-out meeting `M` in cell
`(asset, horizon)`, the train set is every supervised row in that
cell whose `feature_event_date < M`. The fitter asserts the strict
inequality on every fold. Train folds with fewer rows than the
feature dimension fall back to the baselines.

The pooled-panel variant walks the time boundary over the whole
panel: equal-date rows in other cells are excluded so same-event
information cannot leak across assets.

### Artifact layout

Outputs land under `data/artifacts/cross_asset/`:

- `predictions.json` -- per `(meeting, asset, horizon, model)`
  prediction with the realised target and train-row count.
- `metrics.json` -- per-cell RMSE, MAE, R^2, directional hit rate
  for every model. Reports both the full window and the
  pandemic-excluded window (`2020-04-01..2021-06-30`).
- `feature_attribution.md` -- ablation table for the headline cells
  `^GSPC|h1` and `^GSPC|h5`. Same subset list as the next-FOMC
  attribution table, plus the model-free `zero_baseline` and
  `ois_bp_baseline` reference rows.

## Forecaster architecture sweep (Phase 8)

The quantitative-forecaster CLI under `backend/app/train_forecaster.py`
ships a six-architecture sweep harness with an optional credibility
features flag. All six architectures share the same input contract
(`(batch, 20, 6)`) and the same output contract (`(batch, 2)` for
close/volatility) so the sweep harness, evaluation loop, and downstream
inference path treat them interchangeably.

### Architecture roster

| Arch         | Core                                             | Notes |
|--------------|--------------------------------------------------|-------|
| `lstm`       | `nn.LSTM` (default)                              | The v2 default; byte-identical to pre-#70 behaviour when `credibility_features=False`. |
| `lstm_attn`  | `nn.LSTM` + `RecurrentSequenceAttention` pool    | Additive-attention pool over LSTM outputs replaces `output[:, -1, :]`. |
| `gru`        | `nn.GRU`                                         | Same hyperparameter shape as the LSTM core. |
| `tcn`        | Two dilated-conv `TemporalConvNet` blocks        | Causal padding; residual identity. |
| `transformer`| `SmallTransformer` (2 layers, 4 heads)           | `hidden_size` must be divisible by 4 (default 64 satisfies). |
| `dlinear`    | DLinear (trend + seasonal decomposition)         | Pinned to `SEQUENCE_LENGTH=20`. |
| `informer`   | Informer encoder (ProbSparse self-attention)     | 2 encoder layers, 4 heads, `factor=5`. Same `(B,T,H)` core output as the recurrent variants. |
| `tft`        | Temporal Fusion Transformer encoder              | VSN over 6 features + GRN gating + 4-head self-attention. `hidden_size` must be divisible by 4. |

The official registry constant lives at `app.models.FORECASTER_ARCHITECTURES`.

#### Informer

`backend/app/models/informer.py` implements the encoder side of Informer
(Zhou et al., AAAI 2021) in pure PyTorch — no `pytorch-forecasting` or
upstream Informer repo dependency. ProbSparse self-attention reduces full
self-attention's `O(L^2)` cost to `O(L log L)` by sampling probe keys per
query and routing only the top-`u` queries through a real softmax;
remaining queries fall back to the mean of the value sequence. Defaults
match the AAAI-2021 short-horizon recipe: `d_model = hidden_size = 64`,
`n_heads = 4`, `e_layers = 2`, `dropout = 0.1`, `factor = 5`. Input
contract `(batch, 20, 6)`; encoder-output contract `(batch, 20,
hidden_size)` + `None` so the wrapper's `output, _ = core(x)`
destructuring keeps working unchanged.

#### TFT

`backend/app/models/tft.py` implements a lightweight Temporal Fusion
Transformer encoder (Lim et al., 2021) in pure PyTorch — no
`pytorch-forecasting` dependency. Per-timestep Variable Selection Network
over the six scalar features, GRN-gated residual blocks, and a 4-head
self-attention block sit between two LayerNorms. Defaults follow the
small-budget setup from the paper: `hidden_size = 64`, `n_heads = 4`,
`dropout = 0.1`. Same input/output contract as the rest of the registry.
The published TFT's LSTM encoder/decoder, static-covariate enrichment, and
multi-horizon quantile head are intentionally out of scope — the project's
single-horizon head and time-decay/credibility paths live in
`ForecasterModel` above the encoder.

### Credibility-features flag

`--credibility-features` activates the four-axis credibility vector
(`drift_score`, `realized_vs_stated_gap`, `market_implied_gap`,
`months_since_reversal`) on the forecaster input. Default off preserves
the byte-identical training contract — the determinism regression at
`tests/regression/test_forecaster_determinism.py` plus the lock test at
`tests/unit/test_forecaster_credibility_flag.py` enforce that
`architecture="lstm"` + `credibility_features=False` is bit-identical
across runs at the same seed (within `1e-7` for the in-test contract;
the published `1e-4` contract covers cross-platform drift).

### Sweep output schema

`forecaster_sweep_results.json` (JSON, sorted keys) carries:

```
{
  "mode": "sweep",
  "selection_metric": "combined_rmse",
  "architectures": ["dlinear", "gru", "informer", "lstm", "lstm_attn", "tcn", "tft", "transformer"],
  "seeds": [11, 29, 47, 71, 97],
  "credibility_features": false,
  "trial_count": 30,
  "best_trial_index": <int>,
  "best_trial": {...},
  "selected_checkpoint": {...},
  "trials": [
    {
      "trial_index": <int>,
      "architecture": "<arch>",
      "seed": <int>,
      "selected": <bool>,
      "summary": <TrainingRunSummary.to_dict()>
    },
    ...
  ]
}
```

A sibling `.csv` with one row per trial is written next to the JSON.
The companion aggregator `app.evaluation.forecaster_sweep_aggregator`
reads one or more sweep result files and emits a markdown headline
table plus per-architecture block-bootstrap CIs (95% by default,
`block_size=1`, `n_resamples=1000`, deterministic at `seed=11`). The
aggregator output schema is:

```
{
  "generated_at_utc": "<ISO 8601>",
  "block_size": 1,
  "n_resamples": 1000,
  "coverage": 0.95,
  "bootstrap_seed": 11,
  "architectures": [
    {
      "architecture": "<arch>",
      "seeds": [...],
      "credibility_features": <bool>,
      "combined_rmse": {"values": [...], "ci": {...}},
      "close_rmse":    {"values": [...], "ci": {...}},
      "volatility_rmse": {"values": [...], "ci": {...}}
    },
    ...
  ]
}
```

### Make targets

- `make forecaster-sweep TRAINING_PACKAGE_ID=<id>` — the full
  8-arch x 5-seed sweep. Pass `ARCHITECTURES=<csv>` (or
  `--architectures …`) to restrict. Defaults to the bucketed runner
  (`BATCHING_MODE=auto`); override `BATCHING_MODE=off` to fall back
  to the legacy `ProcessPoolExecutor` path.
- `make forecaster-sweep-exhaustive TRAINING_PACKAGE_ID=<id>` —
  every cell in the HP cross-product on a single worker. Pinned to
  `--batching-mode=off --parallel-workers=1` so the
  byte-identity regression contract on the sweep-report JSON stays
  green.
- `make forecaster-sweep-aggregate TRAINING_PACKAGE_ID=<id>` —
  per-architecture headline (block-bootstrap CIs).
- `make forecaster-credibility-train TRAINING_PACKAGE_ID=<id> ARCHITECTURE=lstm SEED=11`
  — single-architecture run with `--credibility-features` on.

### Bucketed-HP sweep runner

The default `make forecaster-sweep` path groups hyperparameter cells
into model-topology buckets and dispatches each bucket as one
concurrent unit, instead of one cell per spawn-mode subprocess. The
goal is GPU saturation on the project's small-model regime (hidden in
{32, 64, 128}, 1-3 layers, sequence length 20): each cell's
forward + backward is a handful of CUDA kernels that finishes before
the next is dispatched, so the GPU sits at ~25% TGP when each cell
runs in its own process.

A bucket is the maximal set of cells that share the same
`(architecture, hidden_size, num_layers, text_adapter_dim, text_encoder,
fold_id, target_mode)` tuple. Cells inside a bucket differ only on
`(dropout, learning_rate, weight_decay, seed)`.

`--batching-mode={auto, stacked, streams, off}` selects the routing:

- `auto` (default) consults the per-architecture table. Architectures
  whose forward is vmap-friendly route to `stacked`; the rest route
  to `streams`. The current capability table is:

  | Architecture | Routed mode | Reason                       |
  |--------------|-------------|------------------------------|
  | `dlinear`    | `stacked`   | Pure Linear stack            |
  | `lstm`       | `streams`   | cuDNN RNN -- not vmap-able   |
  | `lstm_attn`  | `streams`   | cuDNN RNN -- not vmap-able   |
  | `gru`        | `streams`   | cuDNN RNN -- not vmap-able   |
  | `tcn`        | `streams`   | Conv1d -- partial vmap       |
  | `transformer`| `streams`   | Fused MHA -- not vmap-able   |
  | `informer`   | `streams`   | Fused MHA -- not vmap-able   |
  | `tft`        | `streams`   | Fused MHA + gating           |

- `stacked` forces stacked-mode; an explicit `stacked` against an
  architecture not flagged stacked-capable warns and falls back to
  `streams` so the run still completes.
- `streams` forces the CUDA-streams scheduler: one
  `torch.cuda.Stream` per cell inside one Python process and one
  CUDA context, so the GPU pipelines kernel launches across cells.
  On the CPU device path the streams scheduler collapses to a
  sequential loop because threads on CPU share a single global RNG
  and concurrent training calls would trample each other's seed setup
  -- the CPU path is not the saturation target anyway.
- `off` preserves the legacy `ProcessPoolExecutor` path verbatim:
  each cell runs in its own spawn-mode subprocess with its own
  CUDA context. This is the byte-identity regression contract for
  the pre-bucketed sweep output.

`--max-bucket-size INT` overrides the per-architecture VRAM-budget
cap. Default unset picks 64 for `dlinear`, 32 for `lstm`/`gru`/`tcn`,
16 for `lstm_attn`, 8 for `transformer`, 4 for `informer`/`tft`.

The grouped bucket emits one log line per bucket so the runner's
routing decision is auditable from the run log:

```
[bucket] arch=lstm hidden=32 layers=1 text_adapter=0 encoder=none
  fold=wf_fold_1 target_mode=event_study bucket_size=8 mode=streams
```

`bucket_size > 1` confirms the cells actually grouped rather than
collapsing to one-cell-per-bucket. The `mode` field reflects the
final routing decision after the per-arch table consultation.

### Training-package data flow

`train_forecaster.py` accepts `--training-package-id <id>` and consumes
`data/processed/<id>/events.parquet` directly. The legacy `--data-dir`
scan (raw market-record JSON / JSONL / CSV under `/data`) stays
available; when both flags are set `--training-package-id` wins and
the data-dir override is logged.

Per-event sequence construction:

- Reads the collapsed `events.parquet` view (one row per
  `event_date × event_kind × asset_symbol × horizon`). Rows are
  deduplicated to one per `text_hash`, preferring the `horizon=1` row
  so the appended event-day target close is the next trading day.
- Parses each row's `prior_bars_json` (a JSON list of 20 trading-day
  prior bars). Each bar becomes one `FeatureVector` with
  `sentiment_score` derived from `axis_stance`
  (`hawkish=+1, dovish=-1, neutral/None=0`), `market_close` from
  `bar.close`, `market_volatility` from `bar.vol_5d`,
  `close_change_pct` and `volatility_change` computed bar-to-bar, and
  `elapsed_time` set to the signed day count between the bar date and
  the event date.
- Appends one event-day target frame per event. The target's close
  and volatility derive from one of two modes (selected via
  `--target-mode` on `train_forecaster.py` and the `target_mode`
  kwarg on `load_training_sequences_from_package`); both produce a
  `SEQUENCE_LENGTH + 1`-row group so the downstream window slicer
  materialises one supervised `(window, target)` pair per event.
- Sorts the resulting sequences by `event_date` (then `text_hash` as
  a deterministic tiebreaker) so two runs on the same package emit
  the same sequence ordering.

#### Walk-forward CV protocol

The forecaster sweep partitions training-package events into three
sequence lists (train, val, test) per fold and the training loop
consumes them independently. Two protocols are supported through the
loader's `load_walk_forward_split(training_package_id, fold_id=...)`
entrypoint and the corresponding `train_forecaster.py` flags
(`--folds`, `--protocol`).

**OLD pre-PR semantics (removed).** The loader read
`splits_train_val_test.parquet` and KEPT ONLY rows tagged `train`,
discarding `val` and `test` outright. The training loop then ran an
internal 80/20 random split on the already-train-only sequences and
reported the resulting "validation" RMSE as the headline number. Net
effect: no real held-out test partition existed, and the per-trial
`combined_rmse` was the internal 80/20-val RMSE on the train
partition. The wiki and the contracts documented the result as
"walk-forward CV" but the implementation never honoured that.

**NEW single-fold semantics (`--protocol single-fold`, default when
`--folds` is unset).** Reads `splits_train_val_test.parquet` and
partitions events by the `split_tag` column into three lists. All
three lists feed the loop: `train` drives the optimiser, `val` drives
early stopping, `test` is the held-out evaluation set whose RMSE is
the reported `test_rmse`. The package's `splits_train_val_test.parquet`
already encodes a chronological partition (a single fold), so no
internal random split happens.

**NEW walk-forward multi-fold semantics (`--protocol walk-forward`,
selected when `--folds wf_fold_1 wf_fold_2 ...` is set).** Reads
`fold_manifest_expanding_walk_forward.json` and, for each named
fold, partitions events into three lists by `event_date` falling in
the manifest's date ranges:

- Train list: every event with `event_date < val_start`. Expanding
  window — fold k's train set strictly contains fold (k-1)'s.
- Val list: `[val_start, val_end]`.
- Test list: `[test_start, test_end]`.

Each (architecture, seed, hp_combo, fold) becomes one trial. The
aggregator emits one row per (architecture, fold) plus an all-folds
aggregate row per architecture, with the all-folds CI bootstrapped
across every (seed, fold) cell.

**Why the refactor was necessary (ADR).** The pre-PR loader silently
discarded the val + test partitions and reported the internal 80/20
random split as the headline number; under that protocol no
held-out event ever entered the loss or the reported RMSE, which
breaks the walk-forward CV contract the project documents. The
refactor introduces three changes that restore the documented
protocol: (a) the loader returns a `WalkForwardSplit` dataclass with
explicit train, val, test sequence lists, (b) the training loop
honours those partitions separately (no internal 80/20 random split
on the walk-forward path), and (c) the aggregator's headline column
renames from `combined-RMSE` to `test-RMSE` so the published number
is the real held-out RMSE rather than an internal-val artefact. The
back-compat wrapper `load_training_sequences_from_package` stays
callable on the legacy `--data-dir` path so the byte-identity
regression contract on that path stays green.

**Aggregator output schema changes.** The per-trial summary now
carries explicit `train_metrics`, `val_metrics`, and `test_metrics`
blocks; the headline `metrics` slot maps to `test_metrics` on the
walk-forward path. The aggregator emits a per-fold row plus an
all-folds aggregate row per architecture and the `test_train_gap =
(test_rmse - train_rmse) / train_rmse` column. The pre-PR
`holdout_train_gap` column stays for back-compat readers.

#### Forecaster training-package target modes

`load_training_sequences_from_package(training_package_id, target_mode=...)`
exposes two target-frame derivations. The default is `event_study`;
`realized_return` is preserved for back-compat smoke tests against
pre-event-study sweep numbers.

| Mode              | Target close                                       | Target volatility                            |
| ----------------- | -------------------------------------------------- | -------------------------------------------- |
| `event_study`     | `prior_bars[-1].close * (1 + abnormal_return)`     | `prior_bars[-1].vol_5d + volatility_shift`   |
| `realized_return` | `prior_bars[-1].close * (1 + realized_return)`     | `prior_bars[-1].vol_5d` (literal identity)   |

`abnormal_return` is the market-model residual `realized_return -
(alpha + beta * benchmark_return)` against the trailing 252-day window
that `app.data.event_dataset_builder` fits per event; the target is the
component of the realised move not explained by the broad market.
`volatility_shift` is the post-event minus pre-event 10d realised vol
(log-return std) shipped on the same parquet row, so reconstructing the
target from `prior_vol + shift` yields the actual post-event vol rather
than a copy of the input. The `realized_return` mode is mathematically
identical to the pre-event-study target; under it the volatility column
collapses to a literal identity over the last input row, which gives
linear-decomposition models an artefactual edge on the volatility-RMSE
column.

`NaN` values in `abnormal_return` or `volatility_shift` fall back to
the realized-return formula for that row and emit a `UserWarning`, so
a downstream sweep against a package with missing target columns
surfaces the gap immediately rather than silently training on the
legacy target.

Filtering against the splits parquet is opt-in: when
`splits_train_val_test.parquet` is present and carries a
`partition` (or legacy `split_tag`) column joinable on `text_hash`,
rows tagged `excluded_from_training` are dropped before sequence
construction. The current Phase 8 builder emits `{train, val, test}`
only, so the filter is a no-op on existing packages.

Regression contract: `architecture="lstm"` +
`credibility_features=False` invoked through the legacy `--data-dir`
path is bit-identical to prior versions. The
`tests/regression/test_forecaster_determinism.py` suite drives the
trainer directly with synthesised vectors and never crosses the
training-package code path.

### Forecaster rich-feature input space

The training-package loader joins four feature families onto every
event and broadcasts the event-level signal onto every bar of the
20-day prior window plus the appended event-day target frame. Per-bar
feature size grows from `FEATURE_SIZE = 6` to `RICH_FEATURE_SIZE = 35`.

Per-bar slice layout (positions inside `FeatureVector.as_rich_list()`):

| Slice    | Width | Source / fields |
| -------- | ----- | --------------- |
| `[0:6]`  | 6     | Existing market features (`sentiment_score`, `market_close`, `market_volatility`, `close_change_pct`, `volatility_change`, `elapsed_time`). Byte-identical to `as_list()`. |
| `[6:10]` | 4     | Credibility — `credibility_drift_score`, `credibility_realized_vs_stated_gap`, `credibility_market_implied_gap`, `credibility_months_since_reversal`. Off the event row directly. |
| `[10:25]` | 15   | Linguistic — full `LinguisticVector` (8 LDA topic shares + `hedge_density` / `comparison_density` / `forward_density` / `concrete_ratio` / `hawk_dove_asymmetry` / `log_token_count` + `pivot_distance`). Joined on `text_hash` from `linguistic_features.parquet`. |
| `[25:29]` | 4    | MP-surprise — `mp_surprise_level`, `mp_surprise_path_factor`, `fed_info_factor`, `mp_is_intermeeting` (boolean encoded as 0.0 / 1.0). Joined on `event_date` from `mp_surprises.parquet`. |
| `[29:35]` | 6    | Multi-axis — `axis_factor` / `axis_factor_missing`, `axis_certainty` / `axis_certainty_missing`, `axis_time` / `axis_time_missing`. NaN inputs collapse to `0.0` and flip the paired missing flag to `1.0`. |

The 6-dim `as_list()` output stays unchanged. The legacy
`--data-dir` JSON / JSONL / CSV path emits `FeatureVector` rows whose
`rich_payload` flag is `False`, so `_build_training_tensors`
auto-routes them through `as_list()` and the pre-PR-#173 6-feature
input contract is preserved.

#### Loader flag and per-family ablation

`load_training_sequences_from_package` accepts a `rich_features: bool
= True` kwarg plus four per-family ablation flags
(`use_credibility`, `use_linguistic`, `use_mp_surprise`,
`use_multi_axis`, all `True` by default). When `rich_features=False`
the loader bypasses the side-table joins entirely and the resulting
sequences emit 6-dim feature rows. When a per-family flag is `False`
the relevant slice is zeroed on every bar but the per-bar feature
size stays at 35, so a single sweep can measure per-family lift
without changing the model input shape.

The `train_forecaster.py` CLI exposes the same flags:

```
--rich-features / --no-rich-features
--no-credibility
--no-linguistic
--no-mp-surprise
--no-multi-axis
```

`--rich-features` is the default; `--no-rich-features` reproduces the
pre-PR-#173 6-feature input. The per-family flags are no-ops when
`--no-rich-features` is set.

#### Missing side-table semantics

- `linguistic_features.parquet` absent or unjoined on `text_hash` →
  the linguistic slice is all zeros for that event.
- `mp_surprises.parquet` absent or unjoined on `event_date` → the
  MP-surprise slice is all zeros for that event.
- `axis_factor` / `axis_certainty` / `axis_time` NaN → the value
  collapses to `0.0` AND the paired `*_missing` flag flips to `1.0`,
  so the model can tell "no signal" apart from "neutral signal".
- Credibility fields are required on `events.parquet` (the
  event-row builder guarantees them) and so do not carry a missing
  flag; absent fields are coerced to `0.0` defensively.

#### Make targets

- `make forecaster-sweep TRAINING_PACKAGE_ID=<id> TEXT_ENCODER=<alias>`
  — 8-architecture rich-feature sweep (lstm, lstm_attn, gru, tcn,
  transformer, dlinear, informer, tft) across the official
  five-seed set. The grid sweeps hidden_size {32, 64, 128} x
  num_layers {1, 2, 3} x dropout {0.1, 0.2, 0.3, 0.4} x
  learning_rate {1e-3, 3e-4} x weight_decay {0, 1e-4, 1e-3} x
  text_adapter_dim {32, 64, 128}. The target picks the speedup
  defaults: `--random-search --random-search-samples=50
  --random-search-seed=42 --parallel-workers=8`, so the headline run
  draws 50 HP combos uniformly from the full 216-cell cross-product
  and trains eight cells concurrently on the RTX 4080. Pass
  `TEXT_ENCODER=none` for the text-off baseline row. Override
  `RANDOM_SEARCH_SAMPLES=216` to fall back to exhaustive enumeration
  inside the random-search wrapper, or `PARALLEL_WORKERS=1` for
  sequential timing.
- `make forecaster-sweep-exhaustive TRAINING_PACKAGE_ID=<id>
  TEXT_ENCODER=<alias>` — same architecture roster and HP grid, but
  enumerates every cell sequentially. Reproduces the pre-speedup
  byte-identical sweep output and is the back-compat path the
  regression test at `tests/regression/test_forecaster_sweep_back_compat.py`
  pins.
- `make forecaster-sweep-shuffled-control TRAINING_PACKAGE_ID=<id>
  TEXT_ENCODER=<alias>` — same architecture roster and seed set,
  median HP combo, `--shuffle-targets-control` on. The
  memorisation-control row lands in the aggregator's
  "Shuffled-targets control" section.
- `make forecaster-sweep-baseline TRAINING_PACKAGE_ID=<id>` — the
  pre-PR-#173 6-feature path against the original six architectures,
  preserved for back-compat smoke checks against earlier sweep numbers.

#### Random-search HP sampler

`--random-search` draws `M` HP combos uniformly without replacement
from the full HP cross-product (the seven axes above except
architecture and seed). The architecture roster and seed set
enumerate exhaustively on top of the sampled subset, so an
M-sample run produces `len(architectures) * len(seeds) * M` trial
cells instead of the full `len(architectures) * len(seeds) *
|HP grid|`.[^bb2012]

[^bb2012]: Bergstra & Bengio, "Random Search for Hyper-Parameter
Optimization," JMLR 13 (2012). The paper shows that on
high-dimensional HP grids where only a small subset of axes
dominate the loss surface, random search recovers grid-search
performance at a fraction of the trial budget.

Knobs:

- `--random-search` — default off; the legacy exhaustive enumeration
  is the back-compat path and stays byte-identical at the same
  package + seed set.
- `--random-search-samples M` — default 50; clamps to the grid size
  when `M` exceeds the HP cross-product (asking for 500 against a
  216-cell grid returns all 216 combos rather than erroring).
- `--random-search-seed N` — default 42; controls the sampling RNG
  separately from per-cell training seeds. Re-running with the same
  `N` samples the same HP subset regardless of which architectures
  or seeds the outer enumeration uses.

Each sampled cell carries an `hp_combo_id` field (the index into the
sampled subset, `0..M-1`) on its trial record so the aggregator can
group by-combo for ablation. The exhaustive path omits the field
entirely, which keeps the legacy CSV column set unchanged.

#### Parallel-worker pool

`--parallel-workers N` runs `N` cells concurrently on the same GPU
via `concurrent.futures.ProcessPoolExecutor` with the spawn
multiprocessing context. Each worker is a fresh subprocess that
re-imports torch and acquires its own CUDA context; the parent
collects per-cell results in completion order and sorts the trial
records by `(architecture, seed, hp_combo_id, trial_index)` before
writing the report, so the JSON / CSV output is independent of
worker scheduling.

The default is `1` (sequential, current behaviour). The
`make forecaster-sweep` target picks `8`, which matches the RTX
4080's 16 GB VRAM budget: the largest registered architecture
(transformer, hidden=128, layers=3) holds roughly 1 GB per cell,
leaving headroom for the CUDA allocator's fragmentation pool. The
CLI logs a `WARNING` when `N > 8`; higher values risk an OOM on
the larger architectures.

#### Deterministic-result contract

The same `--random-search-seed` plus the same per-cell `seed`
reproduce both the sampled HP set and the per-cell model weights.
The sampler RNG is isolated inside `numpy.random.RandomState(N)`
so it is independent of torch / numpy / random module state in the
parent process; per-cell determinism is the existing
`enable_deterministic_mode(seed)` call at the top of
`train_model`, which runs once per worker subprocess and re-seeds
torch, numpy, and the standard `random` module. Two cells with
the same `seed` produce bit-identical weights regardless of which
worker process they ran in (modulo cuDNN nondeterminism, which is
the existing default).

### Forecaster text-embedding input space

The training-package loader optionally pulls per-statement
embeddings from `data/raw/embeddings/<encoder>_<rev>.parquet` (built
by `app.data.embedding_cache`) and feeds them to the forecaster as
a fifth feature family on top of the 35-dim rich-feature scalar
input. The encoder pick is exposed through the
`--text-encoder {finbert, finbert_fomc, finbert_fed_adjacent,
bert_base_fed_adjacent, bge_large_en_v15, nomic_embed_text_v15,
voyage_finance_2, none}` CLI flag; the default `none` keeps the
rich-features-only path byte-identical.

#### Pooling formula

For each event the loader picks the four most recent FOMC
statements strictly before the event date and pools them with
softmax-weighted means:

```
w_i = softmax_i ( - Delta t_i / lambda_inv_days )
pooled = sum_i ( w_i * embedding_i )
```

where `Delta t_i` is the day count between the prior statement and
the current event. `lambda_inv_days` defaults to 30.0 (override
through `--text-pool-lambda-inv-days`). The closer the prior
statement, the larger its weight; at `lambda_inv = 30` the most
recent statement carries roughly half the mass when the prior was
30 days back.

#### Encoder list and pooled dim

| Encoder alias              | Pooled `in_dim` | Source           |
| -------------------------- | ---:           | ---------------- |
| `finbert`                  | 768            | ProsusAI/finbert (Araci 2019) |
| `finbert_fomc`             | 768            | ZiweiChen/FinBERT-FOMC |
| `finbert_fed_adjacent`     | 768            | local continued-pretrain on BIS + Fed adjacent |
| `bert_base_fed_adjacent`   | 768            | local continued-pretrain control |
| `bge_large_en_v15`         | 1024           | BAAI/bge-large-en-v1.5 |
| `nomic_embed_text_v15`     | 768            | nomic-ai/nomic-embed-text-v1.5 |
| `voyage_finance_2`         | 1024           | voyageai/voyage-finance-2 |

The encoder-native dim flows onto the `FeatureVector` as
`text_embedding_pooled: list[float]`; the adapter projection from
`in_dim` to `text_adapter_dim` runs inside `ForecasterModel.forward`
so the recurrent core sees a fixed per-bar feature size regardless
of which encoder is active. Adapter shape is
`Linear(in_dim, out_dim) -> LayerNorm(out_dim) -> GELU`, zero-init
so a freshly enabled text path forwards to the same point in
feature space as the rich-features-only baseline.

#### Per-bar input size when the text path is on

| Slice                                    | Width  | Notes |
| ---------------------------------------- | ------ | ----- |
| `[0:35]`                                 | 35     | Existing scalar rich-feature slice (`as_rich_list`). |
| `[35:35 + text_adapter_dim]`             | 32 / 64 / 128 | Adapter output broadcast onto every bar. |
| `[35 + text_adapter_dim]`                | 1      | `text_embedding_missing` flag (1.0 when fewer than one prior statement is available). |

#### Missing semantics

When the four-statement pool can't materialise (e.g. the
chronologically-earliest event in the corpus has no prior), the
loader emits a zero `in_dim`-vector and flips
`text_embedding_missing` to 1.0. The model multiplies the adapter
output by `(1 - missing)` so the recurrent core sees an
unambiguous zero slot rather than an interpolated mean. When the
encoder parquet is missing from `data/raw/embeddings/`, the loader
emits a single `WARNING` log line and the entire run degrades to
the missing-flag path.

#### Shuffled-targets memorisation control

`--shuffle-targets-control` permutes the target column per fold
(seed-fixed for reproducibility) before training. macro-RMSE on
the shuffled-targets run should sit near the constant-mean
predictor; a real-targets run whose RMSE is close to its shuffled
counterpart is memorising rather than learning the input-target
mapping. The aggregator reports the shuffled run in a separate
"Shuffled-targets control" section so the headline real-target
table is not contaminated.

The `forecaster_sweep_aggregator` markdown table grows
`train-RMSE`, `holdout-RMSE`, and `holdout/train gap` columns
alongside the existing close / volatility / combined CIs. Rows
whose mean gap crosses 0.5 get a trailing `!` on the gap cell.

## Pipeline schema validation

`backend/app/data/schemas.py` defines one `pandera.DataFrameSchema` per
row contract the pipeline writes to disk. Each emitter calls
`<schema>.validate(frame)` at its write seam so a row that violates the
contract raises at the write site rather than three stages downstream.

Schemas:

| Schema                       | Emitter                              | Output                                                      |
| ---------------------------- | ------------------------------------ | ----------------------------------------------------------- |
| `IngestedDocSchema`          | `app.data.ingest_sources`            | `source_registry.jsonl`                                     |
| `NormalizedDocSchema`        | `app.data.normalize_labels`          | `registry_labeled.jsonl` + `registry_normalized.parquet`    |
| `QualityPassedRowSchema`     | `app.data.quality_checks`            | `registry_quality_passed.jsonl`                             |
| `FoldRowSchema`              | `app.data.build_training_package`    | `splits_train_val_test.parquet`                             |
| `EventRowSchema`             | `app.data.event_dataset_builder`     | `events.parquet`, `events_full.parquet`                     |
| `LinguisticFeatureRowSchema` | `app.features.linguistic`            | `linguistic_features.parquet`                               |
| `MpSurpriseRowSchema`        | `app.data.mp_surprise`               | `mp_surprises.parquet`                                      |
| `MacroStateRowSchema`        | `app.data.macro_state`               | `macro_state.parquet`                                       |

Each schema runs in lazy mode (`lazy=True`) so a single failed write
reports every offending row / column in one `pandera.errors.SchemaErrors`
exception rather than aborting on the first violation. The shared helper
`app.data.schemas.validate_frame(schema, frame)` is the canonical entry
point for emitters.

Set `FED_PULSE_SKIP_SCHEMA_VALIDATION=1` to bypass validation. The env
var exists for diagnostic re-runs against intentionally malformed inputs
(reproducing a known bad-row scenario without unblocking it). Default
behaviour is validation on; opt-in only.

Schema notes:

- `NormalizedDocSchema` accepts both the nested `axes` dict (the form
  written by `build_training_package` into `registry_normalized.parquet`)
  and the flat `axis_*` columns (the form written by
  `event_dataset_builder` into `events.parquet`). The flat columns are
  `required=False` on the normalized schema and `required=True` on
  `EventRowSchema`.
- `QualityPassedRowSchema` asserts `text_hash` uniqueness — the exact-
  dedup pass must run before the schema gate.
- `EventRowSchema` asserts `event_kind ∈ {statement, minutes, speech,
  testimony, press_conference}`, `horizon ∈ {1, 5, 10, 30}`,
  `direction_t1d ∈ {-1, 0, 1}`, and that `prior_window_sha256` is a
  64-char lower-hex string. The no-look-ahead contract is enforced by
  the builder's `_assert_no_lookahead`, not by the schema.
- `LinguisticFeatureRowSchema` requires every named topic share and
  hand-crafted density to be finite. `pivot_distance` is allowed to be
  `NaN` (non-statement rows and the first statement in the corpus emit
  `NaN` by design).
- `MpSurpriseRowSchema` constrains `methodology ∈ {ois_proxy,
  ff_futures}` so a future CME-settlement source is the only way to
  emit a non-`ois_proxy` row.
