# ADR 0024 — MP-surprise / fed-info strict-prior reformulation

Status: accepted, in production (as of merge).
Date: 2026-05-27.
References:
- Issue #350 (closes).
- Issue #324 (per-feature provenance audit; the source flag).
- `docs/feature-provenance-audit.md` — column-level provenance table; rows for `mp_surprise_level`, `mp_surprise_path_factor`, `fed_info_factor` moved from `T+Δ` to strict-prior.
- `backend/app/data/mp_surprise.py` — `_strictly_prior_pre_and_trailing_yield`, `_spx_return_on`, `build_mp_surprises` Pass 1b + Pass 2 + Pass 3.
- `backend/app/data/rates_event_features.py` — `implied_next_move_bps`, the strict-prior FRED-only proxy that anchors the level reformulation's "expected" leg.
- `tests/unit/test_mp_surprise.py` cases 11+ — strict-prior contract on the helpers and on the build path.
- `tests/regression/test_feature_provenance_as_of.py::test_mp_surprise_columns_read_strictly_before_event_date` — source-data audit gate.

## Context

The per-feature provenance audit (#324, `docs/feature-provenance-audit.md`) flagged three `FeatureVector` columns as `T+Δ`-derived by construction:

- `mp_surprise_level`
- `mp_surprise_path_factor`
- `fed_info_factor`

All three were built in `backend/app/data/mp_surprise.py` from a `[T-1, T+1]` window centred on the FOMC announcement (`_pre_post_yields` for the curve legs, `_spx_return_on` for the equity leg). For a forecaster predicting `forward_realized_vol_10d` over `[T, T+10]`, the `T+1` post-event yield close and the `T+1` SPX close are both inputs the model sees at `T` but that mechanically embed one day of post-announcement information. The forward-target window `[T, T+10]` carries `T+1` as one of its ten realised returns, so the leak is small but real — the same class of mechanical overlap ADR-0014's strict-forward target fix addressed on the output side, mirrored on the input side.

The audit's #324 PR shipped the regression test for the rest of the column surface and filed #350 to scope this ADR and the canonical re-baseline. The fix is the construction change in `mp_surprise.py`; this ADR is the methodology footnote.

## Decision

Drop the `T+1` leg from every input that feeds `mp_surprise_level`, `mp_surprise_path_factor`, and `fed_info_factor`. Replace the leaky `post − pre` constructions with strict-prior reformulations:

- **`mp_surprise_level`** becomes `actual_target_change_bps − pre_implied_next_move_bps` where
  - `actual_target_change_bps = (ff_target_after − ff_target_prior) × 100`,
  - `pre_implied_next_move_bps = (pre_yield_1m_T-1 − ff_target_prior) × 100`,
  - both pre-side anchors observable strictly before `event_date`,
  - the only `T`-snapshot input is `ff_target_after` — the announced policy decision the surprise is defined *against*, not a feature read out of post-event market data.

  This matches `implied_next_move_bps` in `backend/app/data/rates_event_features.py` line for line on the "expected" leg, so the construction reuses the strict-prior implied-move proxy already shipped on the rates panel (DGS1 − DFEDTARU).

- **`mp_surprise_path_factor`** becomes the PCA-residualised trailing curve drift at PATH_TENORS_MONTHS = (3, 6, 12). For each meeting:
  - per tenor: `trail_drift_bps = (pre_yield − pre_yield_trail_5td) × 100`, where `pre_yield_trail_5td` is the yield five trading days earlier than `pre_yield` (still strictly before `event_date`).
  - The PCA fit and persisted eigenvector are computed on the trailing drifts at the path tenors residualised against the 1m trailing drift (the level proxy for the fit).
  - The persisted eigenvector is then projected onto each meeting's trailing drift residual.

- **`fed_info_factor`** stays the residual of `mp_surprise_level` against `alpha + beta × spx_return`, but the SPX leg becomes a strict-prior trailing close-to-close return over `[T − ~7 calendar days, T − 1]` (both anchors strictly before `event_date`). The Alpha Vantage ±30 min intraday route is rejected at runtime — the 14:00-14:30 ET half is post-announcement. The `spx_intraday_returns` argument stays on the builder signature for backwards compatibility but is ignored; the diagnostic log line stays in place so an operator who runs the AV backfill is not silently surprised by the rejection.

`pre_event_curve` (already strict-prior) stays on the parquet unchanged. `post_event_curve` stays on the parquet as a diagnostic-only column for `backend/app/forecasting/cross_asset_response.py` and `backend/app/forecasting/next_fomc_decision.py`, both of which evaluate the post-event response at meeting `N` to project onto meeting `N+1` — a different time-axis than the volatility-forecaster leak this ADR addresses. The `_pre_post_yields` helper stays in the codebase for the diagnostic column reads.

## Alternatives considered

**Keep + caveat.** The simplest path was to keep the leaky `[T-1, T+1]` construction and footnote it in the audit. Rejected because the audit is the merge gate; downstream readers should not have to cross-reference a methodology caveat to know whether the feature is honest. The point of #324 was to remove the asterisks, not document them.

**Drop entirely (zero out the three columns).** The lowest-effort fix was to replace the surprise columns with zeros and rely on the residual market-data block (cross-asset closes, realized vol) to carry the FOMC signal. Rejected because the literature is unambiguous that the *expected* leg of the policy decision is a real predictor; deleting it would also delete signal we already verified the model uses (canonical-cell ablations on the MP-surprise block in earlier wiki cells showed a non-trivial lift, ~1-2 pp). The strict-prior reformulation preserves the predictor while closing the leak.

**Keep `_pre_post_yields` but clip the post leg to `T-0` instead of `T+1`.** Rejected because `T-0` is the announcement day itself; for daily-bar Treasury yields the close on `T-0` reflects the announcement and is functionally indistinguishable from `T+1` for the surprise-decomposition use case. The strict inequality has to be `< T`, not `≤ T`.

## Consequences

### Methodology

The literature mapping shifts. The previously-shipped construction was the Cieslak-Vissing-Jorgensen 2021 / Kuttner 2001 post-event surprise quantity — `(post_yield − pre_yield)` at the 1m tenor, projected onto a path PC1, residualised against an SPX equity leg. The strict-prior reformulation is a *different* quantity: actual-vs-pre-implied at `T-1` for the level, trailing-curve PC1 for the path, trailing close-to-close for the equity leg. Neither construction is wrong; the strict-prior version is the one that fits a forecaster scoring at `T-0`, while the post-event version is the one a researcher reading the announcement at `T+1` would compute.

This is honest under the audit framing (no `T+Δ` reads in the input tensor) and is documented per column in `docs/feature-provenance-audit.md`. The methodology change is footnoted there and in the module docstring of `backend/app/data/mp_surprise.py`.

### Model + sweep

The three columns continue to land on `FeatureVector` via the existing `mp_surprises.parquet` join in `backend/app/training/loaders.py`. The column names, dtypes, schema, and loader join key are unchanged; only the values shift. The canonical 5-seed × 4-fold sweep needs a re-baseline against the cleaned features. The post-#350 sweep artefact lives at `backend/artifacts/experiments/canonical_comparison_post_350.json`; the §6.10 wiki row `10b'` annotates the new headline against pre-fix row `10b`. **If the sweep has not yet been run when this ADR lands, the PR description marks the GPU step BLOCKED and the artefact is added in a follow-up commit on the same branch.**

The expected delta on the canonical metric is bounded but uncertain. The leak was a one-day overlap on a ten-day forward window, so the upper bound on the headline drop is small (≤ 2 pp); the lower bound is zero if the model was already ignoring the leak. Either outcome is acceptable — the construction change is correctness, not headline-chasing.

### Diagnostic columns

`pre_event_curve` and `post_event_curve` stay on the parquet. Downstream `next_fomc_decision.py` and `cross_asset_response.py` continue to read `post_event_curve` for their meeting-N+1 projection logic; that path is a different time-axis (post-T evaluation, not a forecaster input at T) and is not under audit here. The strict-prior contract for the **surprise feature columns** is what this ADR locks; the diagnostic column reads stay in the codebase unchanged.

### Reproducibility

The PCA eigenvector persisted in the SOURCES.lock JSON drifts (the fit input changed from `(post − pre)` at PATH_TENORS_MONTHS to trailing 5td drift at the same tenors). Determinism within the new construction is preserved by the existing eigh-based fit and sign-normalisation; the cross-build determinism test in `test_mp_surprise.py` still passes against rebuilds of the new path. Pre-#350 SOURCES.lock entries are not forward-compatible — operators must rebuild `mp_surprises.parquet` from FRED once before re-running the canonical sweep.

### Reviewer focus

- Is the strict-prior contract enforced on the helper boundary? Yes — `_strictly_prior_pre_and_trailing_yield` asserts `trail < pre < on_date` and `_spx_return_on` asserts `trail < pre < event_date`. The unit tests in `test_mp_surprise.py` lock both contracts.
- Does the literature mapping survive? Partially. The strict-prior level surprise is still an actual-vs-expected policy decision (Kuttner-style, with the expectation pulled from the strict-prior implied next-move proxy rather than a post-event futures reading). The path and fed-info legs shift more — they are no longer the CVJ post-event PC1 / ±30 min equity-information channel, only their strict-prior analogues. ADR footnotes this honestly.
- Are there other call sites of `_pre_post_yields` / `_spx_return_on` that need the same treatment? `_pre_post_yields` is still called inside the build to populate `pre_event_curve` and `post_event_curve` (diagnostic-only). `_spx_return_on` is only called from the fed-info pass and is the strict-prior path now. No additional call sites need migrating.
