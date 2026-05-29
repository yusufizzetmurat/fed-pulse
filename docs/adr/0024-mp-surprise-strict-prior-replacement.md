# ADR 0024 — MP-surprise / fed-info strict-prior reformulation

The per-feature provenance audit (#324, `docs/feature-provenance-audit.md`) flagged three `FeatureVector` columns as `T+Δ`-derived: `mp_surprise_level`, `mp_surprise_path_factor`, and `fed_info_factor`. All three were built in `backend/app/data/mp_surprise.py` from a `[T-1, T+1]` window centred on the announcement (`_pre_post_yields` for the curve legs, `_spx_return_on` for the equity leg). For a forecaster predicting `forward_realized_vol_10d` over `[T, T+10]`, the `T+1` post-event yield close and the `T+1` SPX close are inputs the model sees at `T` but that mechanically embed one day of post-announcement information. The forward-target window carries `T+1` as one of its ten realised returns, so the leak is small but real — the same class of mechanical overlap ADR 0014's strict-forward target fix addressed on the output side, mirrored on the input side.

This ADR records the construction change: drop the `T+1` leg from every input that feeds the three columns, and replace the leaky `post − pre` reads with strict-prior reformulations.

## The three reformulations

`mp_surprise_level` becomes `actual_target_change_bps − pre_implied_next_move_bps`:

- `actual_target_change_bps = (ff_target_after − ff_target_prior) × 100`
- `pre_implied_next_move_bps = (pre_yield_1m_T-1 − ff_target_prior) × 100`
- Both pre-side anchors observable strictly before `event_date`. The only `T`-snapshot input is `ff_target_after` — the announced policy decision the surprise is defined *against*, not a feature read out of post-event market data.

This matches `implied_next_move_bps` in `rates_event_features.py` line for line on the expected leg, reusing the strict-prior implied-move proxy already shipped on the rates panel (DGS1 − DFEDTARU).

`mp_surprise_path_factor` becomes the PCA-residualised trailing curve drift at `PATH_TENORS_MONTHS = (3, 6, 12)`. Per tenor, `trail_drift_bps = (pre_yield − pre_yield_trail_5td) × 100` where the trailing yield is five trading days earlier than `pre_yield` (still strictly before `event_date`). The PCA fit and persisted eigenvector compute on the trailing drifts at the path tenors residualised against the 1m trailing drift (the level proxy for the fit), and the persisted eigenvector projects onto each meeting's trailing-drift residual.

`fed_info_factor` stays the residual of `mp_surprise_level` against `alpha + beta × spx_return`, but the SPX leg becomes a strict-prior trailing close-to-close return over `[T − ~7 calendar days, T − 1]`. The Alpha Vantage ±30 min intraday route is rejected at runtime — the 14:00-14:30 ET half is post-announcement. The `spx_intraday_returns` argument stays on the builder signature for backwards compat but is ignored; the diagnostic log line stays so an operator who runs the AV backfill is not silently surprised.

`pre_event_curve` (already strict-prior) is unchanged. `post_event_curve` stays on the parquet as a diagnostic-only column for `cross_asset_response.py` and `next_fomc_decision.py`, both of which evaluate the post-event response at meeting N to project onto meeting N+1 — a different time-axis from the volatility-forecaster leak this ADR closes. The `_pre_post_yields` helper stays for those diagnostic reads.

## Why not the alternatives

Keeping the leaky construction and footnoting it in the audit was the obvious low-effort path; rejected because the audit is the merge gate, and downstream readers should not have to cross-reference a methodology caveat to know whether a feature is honest. Dropping the three columns entirely was the other end; rejected because the canonical-cell ablations show a non-trivial lift (~1-2 pp) from the MP-surprise block, and the literature is clear that the *expected* leg of the policy decision is a real predictor. Clipping the post leg to `T-0` instead of `T+1` doesn't help either — daily Treasury yields close `T-0` after the announcement and are functionally indistinguishable from `T+1` for the decomposition. The gate has to be `< T`, not `≤ T`.

## Methodology footnote

The literature mapping shifts. The previous construction was the Cieslak-Vissing-Jorgensen 2021 / Kuttner 2001 post-event surprise — `(post_yield − pre_yield)` at 1m, projected onto a path PC1, residualised against an SPX equity leg. The strict-prior reformulation is a different quantity: actual-vs-pre-implied at `T-1` for the level, trailing-curve PC1 for the path, trailing close-to-close for the equity leg. Neither construction is wrong; the strict-prior version fits a forecaster scoring at `T-0`, while the post-event version is what a researcher reading the announcement at `T+1` would compute. The audit row in `docs/feature-provenance-audit.md` and the module docstring document the change per column.

## Downstream effects

The three columns continue to land on `FeatureVector` via the existing `mp_surprises.parquet` join in `backend/app/training/loaders.py`. Column names, dtypes, schema, and join key are unchanged; only the values shift. The canonical 5-seed × 4-fold sweep needs a re-baseline against the cleaned features. The post-#350 sweep artefact lives at `backend/artifacts/experiments/canonical_comparison_post_350.json`; §6.10 row `10b'` annotates the new headline against pre-fix `10b`. If the sweep hasn't run yet when this ADR lands, the PR description marks the GPU step BLOCKED and the artefact follows in a follow-up commit on the same branch.

Expected delta on the canonical metric is bounded but uncertain. The leak was a one-day overlap on a ten-day forward window, so the upper bound on headline drop is small (≤ 2 pp); the lower bound is zero if the model was ignoring the leak. Either outcome is acceptable — this is correctness, not headline-chasing.

The PCA eigenvector persisted in `SOURCES.lock` drifts (fit input changed from `(post − pre)` at PATH_TENORS_MONTHS to trailing 5td drift at the same tenors). Determinism within the new construction is preserved by the eigh-based fit + sign-normalisation; the cross-build determinism test still passes. Pre-#350 `SOURCES.lock` entries are not forward-compatible — operators must rebuild `mp_surprises.parquet` from FRED once before re-running the canonical sweep.

## References

- `backend/app/data/mp_surprise.py`, `backend/app/data/rates_event_features.py`
- `docs/feature-provenance-audit.md`
- `tests/unit/test_mp_surprise.py` cases 11+, `tests/regression/test_feature_provenance_as_of.py::test_mp_surprise_columns_read_strictly_before_event_date`
- Kuttner (2001); Cieslak-Vissing-Jorgensen (2021)
