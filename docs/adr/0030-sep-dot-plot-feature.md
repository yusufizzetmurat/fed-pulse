# ADR 0030 — Summary of Economic Projections (SEP) dot-plot ingestion

The Summary of Economic Projections is the FOMC's own explicit forward-guidance instrument. Four times a year (March, June, September, December) the Committee releases a structured table of individual projections for the federal funds rate, GDP growth, unemployment, and PCE inflation across multiple horizons: current calendar year, next year, year after, longer run. The headline output is the dot plot, where each participant's preferred FFR path is plotted as a dot per year. Markets read the SEP the way they read the policy statement itself; dispersion across dots and median drift between releases are first-order signals for the rate path.

Until now the model has consumed every other input channel except the FOMC's most explicit forward-guidance instrument. The MP-surprise block carries realised-vs-implied action, the macro-regime block carries trailing policy direction, and the cross-asset block carries the OIS-implied path, but the SEP itself sits outside the input surface. This ADR records the decision to add it as a small strict-prior feature family.

The methodology angle is the structural addition: the input surface should not silently omit the policy committee's own quantified projections. Whether it lifts the headline regime-classification cell is an empirical question for the canonical sweep.

## What lands

An opt-in `--use-sep` flag on `app.train_forecaster`. Default OFF: the new fields stay `None` and `FeatureVector.as_rich_list()` does not append the SEP block; the per-bar feature size on the legacy / opt-out path is byte-identical to pre-#215. When ON, the loader writes a 4-scalar SEP block (plus a paired missing flag) onto every bar of every supervised sequence, and the model factory widens the recurrent core's input projection by `RICH_SEP_DIM + RICH_SEP_MISSING_DIM` in lockstep.

Five scalars + a release flag:

| Feature | Strict-prior construction |
| --- | --- |
| `sep_ffr_median_current_year` | FOMC's median projection for the current calendar-year-end FFR, as published in the meeting's SEP table. FRED series `FEDTARMD`. Forward-filled on non-SEP meetings from the most recent prior release. |
| `sep_ffr_median_next_year` | FOMC's median projection for the next calendar-year-end FFR. FRED publishes this as a year-specific series per vintage (`FEDTARMD<YYYY>`) rather than a single rolling line (`FEDTARMD2024`, `FEDTARMD2025`, and so on), so the loader pivots per release date: at each SEP meeting it pulls `FEDTARMD<year(release)+1>` and reads the value stamped on the release. A 2024-09-18 release reads its next-year median off `FEDTARMD2025`; a 2025-03-19 release reads off `FEDTARMD2026`. Same forward-fill rule. Pre-2014 vintages lack the year-specific series and the slot collapses to `None`; the per-row missing flag carries the signal. Restored via #415 after the original #215 ship dropped the slot when the rolling-line series `FEDTARMDLM` turned out not to exist on FRED. |
| `sep_ffr_median_longer_run` | Median longer-run FFR projection, the Committee's neutral-rate estimate. FRED series `FEDTARMDLR`. Same forward-fill. |
| `sep_ffr_range_current` | Upper minus lower of the all-participants full range for the current-year projection (`FEDTARRH` − `FEDTARRL`). Dispersion measure: small range means views cluster tightly; wide range means substantial disagreement. These are full range bounds, not central-tendency bounds; central tendency trims three high and three low and would need the separate `FEDTARCT*` series this loader does not currently pull. |
| `sep_release_flag` | `1.0` on the SEP-release meeting itself; `0.0` on forward-fill meetings. Lets the model learn the interaction between a fresh SEP and the reaction to the released document. |

Plus a paired `sep_features_missing` flag (`1.0` when `--use-sep` is off or when the SEP parquet is absent on disk; `0.0` when populated).

## Source policy — FRED with CSV fixture fallback

Two paths feed `data/external/fred/sep_projections.parquet`. The production default is the FRED median series: the Fed republishes the SEP medians as projection series on FRED. The builder reads the configured series IDs at quarterly SEP meeting dates pulled off the bundled FOMC calendar (filtered to March / June / September / December by `filter_sep_meeting_dates`) and writes one row per release. The fallback is `data/external/sep_projections.csv`, one row per release with the same columns as the parquet schema. It is used when FRED is unreachable (no API key, network failure, manually-pinned source) and is what the test suite drives so unit tests carry no network dependency.

The parquet schema is the single contract the training-package loader joins against on `meeting_date`. Source path is stamped per row (`source` column = `"fred"` or `"fixture_csv"`) so a downstream consumer can audit which path each row came from.

## Forward-fill

Non-SEP meetings (the eight per year outside the quarterly cadence) carry the most recent prior SEP's values with `sep_release_flag = 0.0`. The two alternative conventions:

- Emit `None` on non-SEP meetings. Would leave eight of every twelve meetings with the SEP block missing. The model would learn to ignore the slot rather than reading the most-recently-stated FOMC view. Forward-fill preserves the contextual signal that the Committee's most recent quantified guidance is still in effect.
- Emit zeros on non-SEP meetings. Zero is not a defensible "no signal" value for an FFR projection; it would imply the FOMC projects a zero rate, which is meaningfully different from "no fresh projection." The missing flag is the correct signal channel.

The release flag distinguishes the two cases. The model can learn three regimes: fresh-SEP meeting (flag = 1.0, values current), non-SEP meeting with a recent-prior SEP (flag = 0.0, values carry forward), and the cold-start corner where no prior SEP exists (block stays `None` and the missing flag fires).

## Strict-prior contract

The SEP is released simultaneously with the FOMC statement at the meeting. Release-day projections sit in the same `T (snapshot)` band as the existing `stance_*` text features and the `sentiment_score` axis: quantities defined on `T` itself but observable from the document released on `T`. No `T+Δ` reads.

On non-SEP meetings the forward-fill walks the SEP-projections lookup for the most recent release whose `meeting_date <= event_date`. When that match is the supervised event itself, the release flag is `1.0` and the band classification stays `T (snapshot)`. When the match is an earlier meeting, the band drops to `T-Δ` and the release flag stamps the row as forward-filled.

The audit row in `docs/feature-provenance-audit.md` records both branches; the per-feature provenance regression at `tests/regression/test_feature_provenance_as_of.py` is extended to declare `sep_features` (list payload) and `sep_features_missing` (snapshot scalar) in its inventory.

## Conditional emission

The schema mirrors the #307 pattern, not #305 / #306: `RICH_FEATURE_SIZE` does not widen when the flag is on. Instead, the SEP block appends past the regime block by `FeatureVector.as_rich_list` only when `sep_features is not None`. The opt-out default keeps the per-bar feature size byte-identical to pre-#215 for every downstream caller that iterates slices inside `[0:RICH_FEATURE_SIZE]` (or inside the regime-widened width when `--use-regime-conditioning` is on).

The two opt-in blocks compose. The new helper `rich_feature_size_with_blocks(use_regime, use_sep)` returns the combined width so model factories widen the input projection in one place. The fixed append order (regime first, then SEP) means a checkpoint trained with one flag on still loads cleanly when the other is off; the block widths past `RICH_FEATURE_SIZE` are additive and the inactive block contributes nothing to the tensor.

The per-fold `RobustScaler` slice `[FEATURE_SIZE:RICH_FEATURE_SIZE]` is unchanged: the SEP block (past the regime tail, past `RICH_FEATURE_SIZE`) is not scaled. The SEP scalars live in percentage points (typical range 0%-6% for FFR medians, 0-1.5 pp for the central-tendency range, exactly `{0.0, 1.0}` for the release flag), already in a tighter range than the rich-feature block after IQR. Running a per-fold scaler on a release-flag column of exactly `{0.0, 1.0}` would also trigger the constant-column guard (IQR < epsilon → IQR floored to 1.0), reducing the transform to a centering step, a no-op on the {0, 1} support after centering at the train median.

## Why not the alternatives

Parse the SEP PDF directly from the Fed's release archive. Rejected as the primary path: the SEP comes as a structured PDF, and pulling median values requires a PDF-parsing layer with its own brittleness and a per-meeting layout-fingerprint test. The FRED republished series cover the same numbers with a stable schema and the existing FRED / SOURCES.lock plumbing the MP-surprise builder already uses. The CSV fixture is the operator's escape hatch when FRED's coverage lags or a manually-vetted source needs pinning.

Emit the full dot-plot as a histogram per horizon. Each SEP releases ~19 dots per horizon × 4 horizons = ~76 numbers per release. With ~250 supervised events and ~50 SEP releases over the training window, fitting per-dot features would burn most of the available degrees of freedom on a single family. The three headline medians plus the current-year range capture the first two moments (location + dispersion) per horizon at a fraction of the dimension cost.

Add SEP projections for GDP / unemployment / PCE inflation. Rejected as initial scope: the FFR median is the most-directly-relevant signal for a rate-and-vol forecaster, the other three series are downstream of the FFR via the Committee's model of the economy, and adding them would multiply the feature surface threefold for marginal lift on a small corpus. A follow-up can extend the block once the per-family ablation on the FFR-only version is in.

Add SEP as a `T+Δ` post-event signal. Sometimes treated this way in the literature (what the dots said determines the next-meeting reaction). Rejected: the SEP is released at `T`, so it is part of the meeting outcome the model is supposed to be reading at `T`, not a post-event read of the market reaction. Treating it as `T+Δ` would mis-classify the strict-prior contract and break the audit.

A separate "SEP fresh" architectural gate. Rather than a release-flag indicator scalar, mount a gate on the SEP block that activates only on SEP-release meetings. Rejected: the model can learn the interaction between the release flag and the SEP scalars through the recurrent core's hidden state. A dedicated gate doubles the post-#215 model surface for the same interpretability the in-bar scalar provides for free.

## Framing per #334

The #334 substitution finding showed that stacking small feature families on top of the existing rich-feature input can give negative interaction lift. The SEP block is small (5 scalars + 1 flag) so the dimensional tax is bounded, but the contribution is the addition of the FOMC's own forward-guidance instrument, not the headline cell against the flag. The canonical sweep against `--use-sep` may surface positive, negative, or null lift on the dual-head regime-classification headline.

The code path ships because: treating the FOMC's explicit forward-guidance instrument as a model input, with the same strict-prior + conditional-emission contract every other rich-feature family rides, is defensible on its own; the block is small and the per-event computation is cheap (one dict lookup against the SEP parquet); the flag-off default keeps every existing sweep byte-identical; and the forward-fill + release-flag design lets the model learn the interaction between fresh projections and the released document without burning a separate gate on the surface.

## Downstream effects

The recurrent core's `lstm_input_size` widens by `RICH_SEP_DIM + RICH_SEP_MISSING_DIM = 6` when the flag is on, mirroring the regime-tail widening from #307. The state_dict gains no new keys (the widening sits on the input projection of the recurrent core, which already exists); existing checkpoints with the legacy width deserialise unchanged on `--no-sep`. The opt-in path needs a fresh sweep because the input projection has a different number of input units; the per-fold rich-feature scaler stays anchored on `[FEATURE_SIZE:RICH_FEATURE_SIZE]` (unchanged slice).

Compute: pure-Python composition over a tiny parquet (~50 rows for the full training window). No new I/O beyond the one-shot parquet read at loader startup; per-event cost is dominated by a single dict scan over eligible releases. Full canonical training package stays well under the +10 s budget. CI smoke stays under 60 s on CPU.

The headline cell is a Runpod follow-up; §16 populates with both modes side-by-side as the numbers arrive.

## References

- `backend/app/data/sep_projections.py`, `backend/app/training/sep_features.py`
- `backend/app/training/loaders.py` (`_read_sep_projections_lookup` + per-event call sites)
- `backend/app/models/config.py` (`FeatureVector.sep_features`; `rich_feature_size_with_sep` + `rich_feature_size_with_blocks`)
- `backend/app/models/forecaster_base.py` (`sep_tail_dim` extension to `lstm_input_size`)
- Issues #215, #334; ADR 0024, ADR 0028, ADR 0029
