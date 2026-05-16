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
  same-day SPX return (Cieslak-Vissing-Jorgensen 2021-style
  decomposition; documented `daily_window_proxy` flag because intraday
  ±30 min SPX data is out of scope)
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
  `text_hash` and contributes the 14 structured features from #149.
- `macro_state.parquet` (`data/external/fred/`) provides per-as-of-date
  snapshots of UNRATE, CPI YoY, core PCE YoY, ISM proxy
  (`MANEMP_3m_pct`, documented substitute for the paywalled NAPM
  series), nonfarm-payroll MoM change, and retail-sales MoM.

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
  `--architectures …`) to restrict.
- `make forecaster-sweep-aggregate TRAINING_PACKAGE_ID=<id>` —
  per-architecture headline (block-bootstrap CIs).
- `make forecaster-credibility-train TRAINING_PACKAGE_ID=<id> ARCHITECTURE=lstm SEED=11`
  — single-architecture run with `--credibility-features` on.

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
