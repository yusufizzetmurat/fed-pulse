# Feature Provenance Audit (FeatureVector, issue #324)

Per-column time-of-availability audit of `FeatureVector` (`backend/app/models/config.py:516`)
against the supervised row's `event_date`. Motivated by the ADR 0014
strict-forward target fix, which dropped the headline ~10pp on a
mechanical 1-of-10-return overlap and suggested other leak paths may
exist alongside the target-window correction.

The contract under audit is: every per-bar feature on a supervised
sequence must read from a source dated strictly before `event_date`
(i.e. `as_of_offset <= T-0` and computed from data observable at `T-0`),
unless the row is the appended event-day target frame or the column is
itself a documented training target.

## Notation

- `T` = `row.event_date` (the supervised event for the sequence).
- `T-Δ` = data observable strictly before `T` (e.g. `T-1 close`).
- `T (snapshot)` = a quantity defined on `T` itself but observable from the
  document released on `T` (e.g. the FOMC statement text scraped from
  the Fed website on `T`).
- `T+Δ` = data observable strictly after `T` (post-event window).
- `future-derived` = a column that is by design a function of post-`T`
  market data; treated as a training target, not an input feature.
- `train-fit, applied per-row` = a quantity derived from a per-fold or
  per-package fit that has seen rows from later dates than `T`.

## Per-column table

| field | source | as_of_offset | leak_risk | notes |
| --- | --- | --- | --- | --- |
| `date` | loader (`_bars_to_feature_vectors`, `_append_event_day_target`) | `T-Δ` (prior bars) / `T+h` (target frame) | none | metadata, not consumed by the model. |
| `sentiment_score` | `loaders._stance_to_sentiment(event_row.axis_stance)` | `T (snapshot)` | none | broadcast off the event's own stance label. Document-level signal observable from the released FOMC text. |
| `market_close` | `event_dataset_builder._build_prior_window` → `_bars_to_feature_vectors` | `T-Δ` (prior 20 bars) / `T+h` (target frame) | none | asserted `bars[-1].date < as_of` in `_assert_no_lookahead`; target-frame `market_close` is the supervised target, not an input feature. |
| `market_volatility` (`vol_5d`) | `_build_prior_window` (5d rolling std of log returns) | `T-Δ` | none | rolling std anchored on each prior bar, strict left-window. |
| `close_change_pct` | `FeatureVector.from_market_state` | `T-Δ` (computed from consecutive prior closes) | none | difference of two prior closes; no `T+` data. |
| `volatility_change` | `FeatureVector.from_market_state` | `T-Δ` | none | difference of two prior `vol_5d` values. |
| `elapsed_time` | loader (`bar_date − event_date`) | `T-Δ` (negative offset for lookback bars) / `T+h` (target frame) | none | metadata-derived; magnitude only, no market data. |
| `text_embedding` | unused on the package-loader path; legacy `from_market_state` slot | n/a | none | left as `None` on every supervised path; pooled text uses `text_embedding_pooled` instead. |
| `credibility_drift_score` | `services.credibility_loader._select_document_embedding` (on-or-before `as_of`) + `_select_prior_embeddings` (strict-before `as_of`) | `T (snapshot)` | none | document-side embedding is the as-of snapshot's own embedding; prior side strictly precedes `T`. |
| `credibility_realized_vs_stated_gap` | `services.credibility_loader._realized_series_from_fred` (`parsed <= as_of`) | `T (snapshot)` | low | FRED realized series is bounded by `<= T`; same-day publication risk for the `T-0` observation is minor and bounded to a single bar. |
| `credibility_market_implied_gap` | `features.credibility.market_implied_gap(sep_terminal, ois_terminal)` | `T (snapshot)` | none | both inputs are scalars defined on the FOMC release day; SEP terminal is part of the released package, OIS terminal is the day's close. Placeholder `0.0` on most rows today. |
| `credibility_months_since_reversal` | `features.credibility.months_since_last_reversal` over `_stance_history_series` (`parsed <= as_of`) | `T-Δ`+`T (snapshot)` | none | counts months on the stance sequence up to and including `T`. |
| `linguistic_features` (15-dim LDA + densities + pivot_distance) | `features.linguistic` LDA fit, keyed by `text_hash` | `T (snapshot)` per-document, **package-wide LDA fit** | low | per-row topic distribution is the as-of document's own signal. The LDA model itself is fit on the full package corpus (cross-split), so the fitted topic basis is informed by future text. Acceptable under standard topic-model practice but flagged. |
| `mp_surprise_level` | `data.mp_surprise.build_mp_surprises` (actual target change minus pre-implied next move at T-1) | `T (snapshot)` for `ff_target_after`; `T-Δ` for the pre-implied leg | none | Strict-prior reformulation under #350 (ADR-0024). `level = (ff_target_after − ff_target_prior) * 100 − (pre_yield_1m_T-1 − ff_target_prior) * 100`. The only input observable at `T` (not `T-Δ`) is the announcement itself (`ff_target_after`), which is the realised decision the surprise is defined against. No `T+Δ` reads. Methodology footnote: this is no longer the literal Cieslak-Vissing-Jorgensen post-event quantity. |
| `mp_surprise_path_factor` | `data.mp_surprise.build_mp_surprises` (PCA on strict-prior trailing curve drift at PATH_TENORS_MONTHS) | `T-Δ` | none | Strict-prior reformulation under #350. PCA is fit on `(pre_yield − pre_yield_trail_5td) * 100` per tenor, residualised against the 1m trailing drift. Inputs are all yields published strictly before `event_date`. Methodology footnote: replaces the literature `post_curve − pre_curve` PC1 with a strict-prior trailing-curve PC1. |
| `fed_info_factor` | `data.mp_surprise._spx_return_on` (strict-prior trailing SPX return through T-1) | `T-Δ` | none | Strict-prior reformulation under #350. Residual of `mp_surprise_level` against `alpha + beta * pre_trailing_spx_return` where `pre_trailing_spx_return` is the close-to-close return over `[T - ~7 calendar days, T - 1]`. The leaky `[T-1, T+1]` daily-window proxy and the Alpha Vantage ±30 min intraday route (rejected at runtime) are both removed. Methodology footnote: no longer the CVJ ±30 min equity-information channel. |
| `mp_is_intermeeting` | `data.mp_surprise` calendar lookup on `T` | `T (snapshot)` | none | boolean flag derived from the FOMC calendar entry for `T`. |
| `stance_hawk` / `stance_dove` / `stance_neutral` / `stance_missing` | `loaders._attach_rich_features` from `event_row.axis_stance` | `T (snapshot)` | none | one-hot of the document's own stance label. |
| `time_label_forward` | `event_row.axis_time_label` (gtfintechlab cross-bank rows only) | `T (snapshot)` | none | document-level indicator from the released text. |
| `certain_label_certain` | `event_row.axis_certain_label` (gtfintechlab cross-bank rows only) | `T (snapshot)` | none | document-level indicator from the released text. |
| `realized_vol_20d` | `_build_prior_window` (20d rolling std of log returns, anchored per bar) | `T-Δ` | none | computed from the prior log-return slice; never reads `T+Δ`. |
| `realized_vol_60d` | `_build_prior_window` (60d rolling std of log returns, anchored per bar) | `T-Δ` | none | same construction as `realized_vol_20d`. |
| `vix_close` | `_build_prior_window` cross-asset join on bar date | `T-Δ` (each lookback bar's date) | none | per-bar VIX close from the yfinance cache; bar dates are strictly `< T`. Daily-bar resolution, not intraday — no `T-0` close. |
| `dxy_close` | same as `vix_close` | `T-Δ` | none | DXY daily close per bar. |
| `tnx_close` | same as `vix_close` | `T-Δ` | none | ^TNX daily close per bar. |
| `gold_close` | same as `vix_close` | `T-Δ` | none | gold front-month daily close per bar. |
| `vix3m_close` | same as `vix_close` | `T-Δ` | none | VIX3M daily close per bar. |
| `irx_close` | same as `vix_close` | `T-Δ` | none | ^IRX 13-week T-bill yield per bar. |
| `vix_term_slope` | `_build_prior_window` derived from `vix3m_close` / `vix_close` per bar | `T-Δ` | none | per-bar log slope; both inputs already `T-Δ`. |
| `yield_curve_slope_10y_3m` | `_build_prior_window` derived from `tnx_close - irx_close` per bar | `T-Δ` | none | per-bar level difference; both inputs already `T-Δ`. |
| `llm_features` (35-dim one-hot) | `_load_llm_feature_lookup` keyed by `text_hash` | `T (snapshot)` | none | LLM extraction over the released document; one-hot of categorical levels per feature. The extractor model itself is a pretrained LLM not fit on this corpus. |
| `llm_features_missing` | loader flag on the LLM cache lookup | `T (snapshot)` | none | per-row 0/1 mask. |
| `analog_features` (5-dim contextual summary) | `loaders._compute_analog_features_for_event` over the top-K analog hits from the on-disk retrieval index, queried with strict-backward `as_of_date < event_date` filter | `T-Δ` (analog event dates are strictly before `T`); current-event stance read is `T (snapshot)` | none | #306. Five scalars: max / mean / dispersion of cosine similarities, count above the 0.40 floor, fraction of analogs whose stance matches the current event's stance. The analog's own post-event observed move (`forward_realized_vol_10d` or `subsequent_vol_regime`) is **not** in this block — admitting it would be a label leak via similarity. Strict-backward analog filter is enforced at the retrieval-query level (`app.retrieval.index.query(..., as_of_date=event_date)`). See ADR 0028. |
| `analog_features_missing` | loader flag on the retrieval lookup | `T (snapshot)` | none | per-row 0/1 mask. `1.0` when the retrieval bundle is absent on disk (graceful degrade — the model sees the all-zeros block + this flag) or when `--use-retrieval-analogs` is off. |
| `rich_payload` | loader flag, set after rich-feature attachment | n/a | none | structural flag, not consumed as a numeric feature. |
| `forward_realized_vol_10d` | `event_dataset_builder._forward_realized_vol` (closes `[T..T+10]`) | **`T+Δ`, future-derived** | none (target) | declared training target under the strict-forward ADR. The loader broadcasts the target-row value onto every bar of the sequence for convenience, but it is **not** emitted by `FeatureVector.as_rich_list` and therefore never enters the model input tensor. Downstream consumers (`collect_forward_vols`, `_build_training_tensors`) only read it off the target row at index `>= SEQUENCE_LENGTH`. Per-fold quantile cutoffs are fit on the train slice only. |
| `target_yield_2y_change_5d` | `data.rates_event_features.forward_yield_change_bps` (t → t+5) | **`T+Δ`, future-derived** | none (target) | rates-head training target. Same broadcast-but-not-emitted pattern as `forward_realized_vol_10d`; not in `as_rich_list` output, only read off the target row by `app.training.rates_targets.build_partition_rates_targets`. |
| `target_yield_5y_change_5d` | same as above (5y tenor) | **`T+Δ`, future-derived** | none (target) | rates-head training target; same storage / emission contract. |
| `target_terminal_rate_change_5d` | same as above (terminal-rate proxy) | **`T+Δ`, future-derived** | none (target) | rates-head training target; same storage / emission contract. |
| `target_yield_2y_change_5d_fomc_attributable` | 1-D projection of the observed 2y bps move onto `sign(mp_surprise_level)` (strict-prior, post-#350) | **`T+Δ`, future-derived** | none (target) | #305 surprise-decomposition target. Computed in `app.training.loaders` from the observed move scaled by the strict-prior surprise direction; `None` when `|mp_surprise_level| < 1.0 bp` (no-change meetings; direction ill-defined). Same broadcast-but-not-emitted contract as the raw siblings; only the target row is read by `build_partition_rates_targets`. See ADR 0027. |
| `target_yield_5y_change_5d_fomc_attributable` | same as above (5y tenor) | **`T+Δ`, future-derived** | none (target) | #305 surprise-decomposition target; same storage / emission contract. |
| `target_terminal_rate_change_5d_fomc_attributable` | same as above (terminal-rate proxy) | **`T+Δ`, future-derived** | none (target) | #305 surprise-decomposition target; same storage / emission contract. |
| `text_embedding_pooled` | `loaders._compute_prior4_pooled_embedding` (softmax-weighted mean of the four most recent statements with date strictly `< T`) | `T-Δ` | none | the pool is the four prior statements only; the as-of document itself is excluded from the pool (`prior_text_hashes` are statement dates `< event_date`). |
| `text_embedding_missing` | loader flag on the pooled-embedding lookup | n/a | none | structural mask, no leakage surface. |
| `text_per_bar` | `loaders.build_per_bar_text_tensor` (per-bar pooled-text payload aligned to each lookback bar's date; falls back to tile-replicating `text_embedding_pooled` when no per-bar pool is attached) | `T-Δ` (prior FOMC docs only) | none | issue #327 Arm A. Each bar's row reads from the prior-N FOMC documents aligned to that bar's calendar date; bars are emitted on calendar dates strictly `< T` and the fallback reuses the pooled vector that is itself a function of prior statements only. Default `None` collapses the per-bar slot to the broadcast-zero path. |
| `raw_text` | loader, target-row only when `encoder_lora=True` | `T (snapshot)` | none | the as-of document's own released text; LoRA path tokenises this for the encoder. |
| `target_stance_idx` / `target_stance_present` | `event_row.axis_stance` mapped to canonical index | `T (snapshot)` | none (target) | multi-task head training target; the document's released stance label. |
| `target_factor` / `target_factor_present` | `event_row.axis_factor`, clipped to `[-1, 1]` | `T (snapshot)` | none (target) | multi-task head training target. |
| `target_certainty_idx` / `target_certainty_present` | `event_row.axis_certain_label` (or `axis_certainty` float, tertile-binned) | `T (snapshot)` | none (target) | multi-task head training target. |
| `target_topic_idx` / `target_topic_present` | `event_row.axis_topic` mapped to canonical index | `T (snapshot)` | none (target) | multi-task head training target. |

## Per-fold transforms (outside FeatureVector but on the same input tensor)

These quantities are applied to the per-bar tensor at training time and
do not live on `FeatureVector` itself, but are listed here for
completeness because they are part of the train→test feature-construction
boundary.

| transform | source | as_of_offset | leak_risk | notes |
| --- | --- | --- | --- | --- |
| `RichFeatureScalerParams` (median / IQR) | `training.loaders.fit_rich_feature_scaler_tensor` over the rich-feature block | **train-fit, applied per-row** | none | fit on the train slice only and persisted into the checkpoint; the inference and val/test paths read the fitted parameters off the checkpoint. Verified by `tests/unit/test_scaler_train_only_fit.py` (per docstring). |
| `vol_regime_quantiles` cutoffs | per-fold quantile fit on `forward_realized_vol_10d` over the train slice | **train-fit, applied per-row** | none | quantile cutoffs from the train fold only; persisted in `fold_manifest_expanding_walk_forward.json` and pinned by `docs/benchmark-policy.md §Canonical Training Objective`. |
| `close_scale` | constant `DEFAULT_CLOSE_SCALE = 10000.0` | constant | none | not data-fit. |

## Leaks found

Originally three columns read from post-event data by construction:
`mp_surprise_level`, `mp_surprise_path_factor`, and `fed_info_factor`.
All three were built from a `[T-1, T+1]` window centred on the
announcement (`backend/app/data/mp_surprise.py` `_pre_post_yields` and
`_spx_return_on`). They were the canonical Cieslak-Vissing-Jorgensen
2021 / Kuttner 2001 surprise quantities used as **features** in the
FeatureVector layout.

For a forecaster predicting `forward_realized_vol_10d` over `[T, T+10]`,
treating these `T+1`-derived quantities as known-at-`T` features is the
same class of leak the strict-forward target fix addressed on the
output side: a small mechanical overlap that the model can latch onto.

**Resolved under #350 (ADR-0024).** All three columns now read from
strictly-prior inputs:

- `mp_surprise_level` = `actual_target_change_bps - pre_implied_next_move_bps`
  where the pre-implied leg is `(pre_yield_1m_T-1 - ff_target_prior) * 100`.
  Only the announced policy decision (`ff_target_after`) is observable
  at `T`; every other input is `T-Δ`.
- `mp_surprise_path_factor` is the PCA-residualised trailing curve drift
  (`pre_yield − pre_yield_trail_5td`) at PATH_TENORS_MONTHS, fit and
  applied on strict-prior anchors only.
- `fed_info_factor` is the residual of the strict-prior `mp_surprise_level`
  against a strict-prior trailing SPX return (close-to-close over
  `[T - ~7d, T - 1]`).

The Alpha Vantage ±30 min intraday route is rejected at runtime under
#350 because the 14:00-14:30 ET half is post-announcement. The
`spx_intraday_returns` argument remains on the builder signature for
backwards compatibility but is ignored.

Methodology footnote: the literature mapping shifts. The original
construction was the canonical CVJ post-event surprise; the strict-prior
reformulation is a *different* quantity (actual-vs-pre-implied at T-1
rather than post-pre across the announcement window). ADR-0024 records
this decision and the rationale for choosing drop-T+1 over the
keep-with-caveat alternative.

No other `FeatureVector` column reads from a source post-dating
`row.event_date` beyond the documented training-target columns
(`forward_realized_vol_10d`, `target_yield_2y_change_5d`,
`target_yield_5y_change_5d`, `target_terminal_rate_change_5d`, and the
three `_fomc_attributable` projections added under #305).

## Regression test

`tests/regression/test_feature_provenance_as_of.py` materialises a
synthetic training package, loads the supervised sequences via
`load_walk_forward_split`, and asserts the contract for every per-bar
column on every lookback row of every sequence: no scalar feature reads
from a source post-dating `event_date`. The MP-surprise / fed-info
columns are zeroed on the fixture (no `mp_surprises.parquet` shipped)
so the per-bar contract holds package-wide.

The source-data construction for `mp_surprise_level`,
`mp_surprise_path_factor`, and `fed_info_factor` is exercised separately
by `test_mp_surprise_columns_read_strictly_before_event_date` (added
under #350) which builds a synthetic `mp_surprises` row through
`app.data.mp_surprise.build_mp_surprises` and asserts the strict-prior
contract on the pre/trail yield helper and the SPX return helper. The
test also locks the rejection of the leaky Alpha Vantage intraday
route. Per-unit-test coverage on the strict-prior construction lives in
`tests/unit/test_mp_surprise.py` (cases 11+).
