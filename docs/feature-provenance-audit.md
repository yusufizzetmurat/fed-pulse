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
| `macro_regime_features` (3-dim signed scalars) | `loaders._compute_macro_regime_features_for_event` -> `app.training.regime_features` composer; reads `ff_target_prior` from strictly-prior MP-surprise rows over the trailing 12 months, the supervised event's own prior-bar `vix_close` series, and the last prior bar's `tnx_close - irx_close` | `T-Δ` (every input dated strictly before `T`) | none | #307. Three scalars in `{-1, 0, +1}`: `policy_cycle_phase_score` (rolling 12-month band-midpoint move with ±25 bp thresholds), `vix_level_regime_score` (T-1 VIX vs the trailing prior-bar tertile cutoffs), `term_spread_sign` (sign of 10y-3m slope at T-1). Block is appended past `RICH_FEATURE_SIZE` by `FeatureVector.as_rich_list` only when the loader populates the slot under `--use-regime-conditioning`. See ADR 0029. |
| `macro_regime_features_missing` | loader flag on the regime composer | `T (snapshot)` | none | per-row 0/1 mask. `1.0` when `--use-regime-conditioning` is off (default). The conditional-emit contract on `as_rich_list` keeps the per-bar feature size byte-identical to pre-#307 in that case (the block + flag are not emitted at all). |
| `sep_features` (5-dim scalar block) | `loaders._compute_sep_features_for_event` -> `app.training.sep_features.compute_sep_features_for_event`; reads ``meeting_date / ffr_median_* / ffr_range_*`` from the SEP-projections lookup, forward-filling the most recent prior release on non-SEP meetings | `T (snapshot)` on SEP-release meetings (March / June / September / December); `T-Δ` on forward-filled rows | none | #215, next-year slot restored in #415. Five scalars: current-year median FFR (`FEDTARMD`), next-year median FFR pulled per release off the year-specific `FEDTARMD<year+1>` series (e.g. a 2024-09-18 release reads `FEDTARMD2025`), longer-run median FFR (`FEDTARMDLR`), current-year all-participants range (`FEDTARRH` − `FEDTARRL`), and a release flag (`1.0` when the meeting itself refreshed the SEP, `0.0` when the slot carries forward-filled values from a strictly-prior release). The SEP is released simultaneously with the FOMC statement, so the release-day values sit in the same snapshot band as the `stance_*` text features. Forward-fill is strict-prior by construction: a non-SEP meeting reads from a prior meeting's SEP whose own `meeting_date < event_date`. Pre-2014 vintages lack a `FEDTARMD<YYYY>` series and the next-year slot collapses to `0.0` for those rows; the rest of the block stays populated. Block appended past the regime tail by `FeatureVector.as_rich_list` only when the loader populates the slot under `--use-sep`. See ADR 0030. |
| `sep_features_missing` | loader flag on the SEP composer | `T (snapshot)` | none | per-row 0/1 mask. `1.0` when `--use-sep` is off (default) or when the SEP-projections parquet is absent on disk (graceful degrade — the code path stays live without the parquet). The conditional-emit contract on `as_rich_list` keeps the per-bar feature size byte-identical to pre-#215 in that case (the block + flag are not emitted at all). |
| `statement_delta_embedding` (768-dim mean-pooled redline) | `loaders._read_statement_delta_embedding` reads the `statement_delta_embedding` column off events.parquet; the column itself is populated by `app.data.statement_delta.compute_delta_for_event` against the strict-prior statement (`prior.event_date < this.event_date`, asserted in `select_prior_statement_text`) | `T-Δ` (prior statement) + `T (snapshot)` (current statement); the diff is a function of both | none | #443 statement-delta redline. The diff is computed at events.parquet build time using `difflib.SequenceMatcher` on the lowercase-normalised token streams; the three text spans (inserted / deleted / substituted) are encoded once each by the FinBERT-Fed-Adjacent classifier-head encoder and mean-pooled to a single 768-dim vector. Strict-prior is enforced by `select_prior_statement_text` (the helper raises if a same-date prior shows up in the index); cold-start events (the first statement on the panel) carry `None` and the missing flag fires. Block is appended past the SEP tail by `FeatureVector.as_rich_list` only when the loader populates the slot under `--use-statement-delta`. See ADR 0038. |
| `statement_delta_embedding_missing` | loader flag on the delta embedding | `T (snapshot)` | none | per-row 0/1 mask. `1.0` when `--use-statement-delta` is off (default), the supervised row's event_kind is not `statement`, or the cold-start path fired (no strict-prior statement available). The conditional-emit contract on `as_rich_list` keeps the per-bar feature size byte-identical to pre-#443 in those cases (the block + flag are not emitted at all). |
| `vote_features` (4-dim signed block) | `loaders._compute_vote_features_for_event` reads the `votes_for` / `votes_against` / `is_unanimous` / `dissent_direction` columns off events.parquet; the columns themselves are populated by `app.data.vote_tally.parse_vote_tally` against the supervised event's own statement text | `T (snapshot)` | none | #444 vote tally + dissent. The vote IS the event — the FOMC statement document carries the vote block, so every input the parser reads is observable on `T` from the released document. No leak surface. The 4-vector is `[votes_for_norm, votes_against_norm, is_unanimous_float, dissent_direction_signed]` where the count axes are divided by 12 (the canonical FOMC voting-member cap) and `dissent_direction_signed` maps `hawkish_dissent` → `+1.0`, `dovish_dissent` → `-1.0`, unanimous / unparseable / mixed → `0.0`. Block is appended past the statement-delta tail by `FeatureVector.as_rich_list` only when the loader populates the slot under `--use-vote-features`. See ADR 0038. |
| `vote_features_missing` | loader flag on the vote composer | `T (snapshot)` | none | per-row 0/1 mask. `1.0` when `--use-vote-features` is off (default), the supervised row's event_kind is not `statement` (the vote lives only in the statement document), or the parser failed to find a vote block (older pre-1990s archived rows where the template was less standardised). The conditional-emit contract on `as_rich_list` keeps the per-bar feature size byte-identical to pre-#444 in those cases. |
| `rich_payload` | loader flag, set after rich-feature attachment | n/a | none | structural flag, not consumed as a numeric feature. |
| `forward_realized_vol_10d` | `event_dataset_builder._forward_realized_vol` (closes `[T..T+10]`) | **`T+Δ`, future-derived** | none (target) | declared training target under the strict-forward ADR. The loader broadcasts the target-row value onto every bar of the sequence for convenience, but it is **not** emitted by `FeatureVector.as_rich_list` and therefore never enters the model input tensor. Downstream consumers (`collect_forward_vols`, `_build_training_tensors`) only read it off the target row at index `>= SEQUENCE_LENGTH`. Per-fold quantile cutoffs are fit on the train slice only. |
| `forward_realized_vol_10d_garch_baseline` | `app.data.garch_residual.compute_for_event`: GARCH(1,1) fitted on strict-prior log returns of the asset's close series (closes dated `< event_date`), forecast `horizon=10` days ahead, mean per-step variance square-rooted to a 1-day-equivalent vol. | `T-Δ` (fit window) → 10-step-ahead forecast | none (target-side) | #236 GARCH baseline. The fit consumes only closes strictly before `event_date`; the forecast is conditional-on-fit and reads no close at or after `event_date`. The leak surface is identical to `_volatility_shift`'s pre-event leg. `None` when the strict-prior window is shorter than `MIN_FIT_RETURNS` (~252 td) or the QMLE step does not converge. Same broadcast-but-not-emitted contract as `forward_realized_vol_10d`; only the target row is read by the residual target consumer. See ADR 0034. |
| `forward_realized_vol_10d_garch_residual` | `app.data.garch_residual.compute_for_event`: `forward_realized_vol_10d − forward_realized_vol_10d_garch_baseline` | **`T+Δ`, future-derived** | none (target) | #236 GARCH-residual variant of the forward-vol target. The residual isolates the unanticipated component of realised vol given the GARCH(1,1) conditional-variance model — the part a hybrid GARCH-NN forecaster predicts after the classical baseline is stripped off. `None` whenever either the raw target or the baseline is `None`. Stored on the events.parquet at build time so the target column is frozen in the training package; same broadcast-but-not-emitted contract as the raw sibling. See ADR 0034. |
| Per-asset realized volatility targets (`forward_realized_vol_10d_{gspc,ndx,dji,dxy,vix,eurusd,usdjpy,gbpusd}`) | `event_dataset_builder._forward_realized_vol` per symbol (closes `[T..T+10]` on each symbol's own series; price series fetched once per symbol via `_fetch_close_series` and cached under `<DATA_DIR>/external/yfinance/<symbol>.parquet`) | **`T+Δ`, future-derived** | none (target) | #481 per-asset 10d forward realised-vol training targets — data foundation for per-asset regime prediction (workspace asset picker set + VIX, 8 symbols total). Slug rule: lowercase, strip `^` / `=X` / `.NYB`, drop remaining `-` (e.g. `DX-Y.NYB` -> `dxy`). The `_gspc` column is the canonical alias of `forward_realized_vol_10d`; the remaining seven add the other indices, dollar index, VIX, and the three major FX pairs. Same strict-forward convention as the canonical sibling: returns enter the window as `log(close[t+1]/close[t]) ... log(close[t+window]/close[t+window-1])` so `close[t]` participates only as the first denominator and the announcement-day close-to-close move never enters the target. `None` whenever the symbol's series is absent (cache fetch failed, pre-listing event date, holiday) or the event sits within the forward window of the series tail; the downstream classifier learns to skip rather than treat absent as zero. v1 is 10d-only per asset; per-asset heads, multi-horizon × multi-asset, and the inference path remain out of scope and live under follow-up issues. |
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
`target_yield_5y_change_5d`, `target_terminal_rate_change_5d`, the
three `_fomc_attributable` projections added under #305, and the
`forward_realized_vol_10d_garch_baseline` / `forward_realized_vol_10d_garch_residual`
decomposition added under #236).

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
