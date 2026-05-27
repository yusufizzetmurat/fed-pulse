# ADR 0030 — Summary of Economic Projections (SEP) dot-plot ingestion

Status: accepted, code path live; canonical sweep deferred to operator.
Date: 2026-05-27.
References:
- Issue #215 (closes).
- Issue #350 — strict-prior MP-surprise reformulation (precedent for treating the meeting-day announcement value as `T (snapshot)` while every other input is `T-Δ`).
- ADR 0024 — strict-prior MP-surprise reformulation.
- ADR 0028 — retrieval-augmented input features (the same opt-in flag + sweep-hand-off framing applied to a different feature family).
- ADR 0029 — macro-regime conditioning (the conditional-emission contract on `as_rich_list` that this ADR mirrors).
- `backend/app/data/sep_projections.py` — parquet builder (FRED median series + CSV fixture fallback).
- `backend/app/training/sep_features.py` — pure-Python composer (`compute_sep_features_for_event`).
- `backend/app/training/loaders.py` — `_read_sep_projections_lookup` + `_compute_sep_features_for_event` + per-event call sites on both loader entry points.
- `backend/app/models/config.py` — `FeatureVector.sep_features` schema extension; slice constants; `rich_feature_size_with_sep`; combined helper `rich_feature_size_with_blocks`; `ModelConfig.use_sep`.
- `backend/app/models/forecaster_base.py` — `sep_tail_dim` extension to `lstm_input_size` so the recurrent core absorbs the widened per-bar tensor.
- `backend/app/train_forecaster.py` — `--use-sep` / `--no-sep` CLI flag, threaded through both loader call sites + `ModelConfig` assembly + the run-report payload.

## Context

The Summary of Economic Projections (SEP) is the FOMC's own explicit forward-guidance instrument. Four times a year — at the March, June, September, and December meetings — the Committee releases a structured table of individual projections for the federal funds rate, GDP growth, unemployment, and PCE inflation across multiple horizons (current calendar year, next year, year after, longer run). The headline output is the "dot plot": each participant's preferred FFR path plotted as a dot per year. Markets read the SEP the way they read the policy statement itself; the dispersion across dots and the drift in the median between releases are first-order signals for the rate path.

The model up to now has had every other input channel except the FOMC's most explicit forward-guidance instrument. The MP-surprise block carries the realised-vs-implied action, the macro-regime block carries trailing policy direction, and the cross-asset block carries the OIS-implied path, but the SEP itself sits outside the input surface. This ADR records the decision to add it as a small strict-prior feature family.

The methodology contribution is the structural addition: we add the SEP — the FOMC's own explicit forward-guidance instrument — as a model input. Whether it lifts the headline regime-classification cell is an empirical question for the canonical sweep; the case for the ADR is that the input surface should not silently omit the policy committee's own quantified projections.

## Decision

Add an opt-in `--use-sep` flag on `app.train_forecaster`. Default OFF — when off the new fields stay `None` and `FeatureVector.as_rich_list()` does NOT append the SEP block; the per-bar feature size on the legacy / opt-out path is byte-identical to pre-#215. When ON, the loader writes a 4-scalar SEP block (plus a paired missing flag) onto every bar of every supervised sequence, and the model factory widens the recurrent core's input projection by `RICH_SEP_DIM + RICH_SEP_MISSING_DIM` in lockstep.

### Feature block — four scalars + a release flag

| Feature | Strict-prior construction |
| --- | --- |
| `sep_ffr_median_current_year` | The FOMC's median projection for the current calendar-year-end fed funds rate, as published in the SEP table released at the meeting. FRED series `FEDTARMD`. Forward-filled on non-SEP meetings from the most recent prior release. |
| `sep_ffr_median_longer_run` | Median longer-run FFR projection — the Committee's neutral-rate estimate. FRED series `FEDTARMDLR`. Same forward-fill rule. |
| `sep_ffr_range_current` | Upper minus lower of the all-participants full range for the current-year projection (FRED series `FEDTARRH` − `FEDTARRL`). A dispersion measure: small range means views cluster tightly; wide range means substantial disagreement. These are full range bounds, not central-tendency bounds — central tendency trims three high and three low and would require the separate `FEDTARCT*` series this loader does not currently pull. |
| `sep_release_flag` | `1.0` on the March / June / September / December meeting that released the SEP itself; `0.0` on non-SEP meetings where the values are forward-filled. Lets the model learn the interaction between a fresh SEP and the reaction to the released document. |

Median for the next calendar-year-end is intentionally NOT in the block. FRED does not publish a single multi-vintage "next-year median" series; that slot would require pulling year-specific `FEDTARMD<YY>` series and pivoting per event date. Tracked as a follow-up to #215.

Plus a paired `sep_features_missing` flag (`1.0` when `--use-sep` is off or when the SEP parquet is absent on disk; `0.0` when the slot is populated).

### Source policy — FRED median series with a CSV fixture fallback

Two ingestion paths feed `data/external/fred/sep_projections.parquet`:

- **FRED median series.** The Fed republishes the SEP medians as fed-funds-rate projection series on FRED. The builder (`backend/app/data/sep_projections.py`) reads the configured series IDs at the quarterly SEP meeting dates pulled off the bundled FOMC calendar (filtered to the March / June / September / December months by `filter_sep_meeting_dates`) and writes one row per release. Path (a) is the production default.

- **CSV fixture.** When FRED is unreachable (no API key, network failure, or a manually-pinned source preference) the loader reads `data/external/sep_projections.csv` with one row per release and the same columns as the parquet schema. Path (b) is what the test suite drives so the unit tests carry no network dependency.

The parquet schema is the single contract the training-package loader joins against on `meeting_date`. Source path is stamped per row (`source` column = `"fred"` or `"fixture_csv"`) so a downstream consumer can audit which path each row came from.

### Forward-fill convention — every meeting carries SEP context

Non-SEP meetings (the eight meetings per year outside the quarterly SEP cadence) carry the most recent prior SEP's values, with `sep_release_flag = 0.0`. The two alternative conventions were considered and rejected:

- **Emit `None` on non-SEP meetings.** Would leave eight of every twelve meetings with the SEP block missing. The model would learn to ignore the slot on those rows rather than reading the most-recently-stated FOMC view. Forward-fill preserves the contextual signal that the Committee's most recent quantified guidance is still in effect.
- **Emit zeros on non-SEP meetings.** Zero is not a defensible "no signal" value for an FFR projection (it would imply the FOMC projects a zero rate, which is meaningfully different from "no fresh projection"). The missing flag is the correct signal channel.

The release flag distinguishes the two cases explicitly. The model can therefore learn three regimes: fresh-SEP meeting (flag = 1.0, values up-to-date), non-SEP meeting with a recent-prior SEP (flag = 0.0, values carry forward from a recent release), and the cold-start corner where no prior SEP exists (block stays `None` and the missing flag fires).

### Strict-prior contract — `T (snapshot)` on SEP meetings, `T-Δ` on forward-filled rows

The SEP is released simultaneously with the FOMC statement at the meeting. The release-day projections sit in the same `T (snapshot)` band as the existing `stance_*` text features and the `sentiment_score` axis — quantities defined on `T` itself but observable from the document released on `T`. No `T+Δ` reads.

On non-SEP meetings the forward-fill walks the SEP-projections lookup for the most recent release whose `meeting_date <= event_date`. When that match is the supervised event itself, the release flag is `1.0` and the band classification stays `T (snapshot)`. When the match is an earlier meeting, the band drops to `T-Δ` and the release flag stamps the row as forward-filled.

The audit row in `docs/feature-provenance-audit.md` records both branches; the existing per-feature provenance regression at `tests/regression/test_feature_provenance_as_of.py` is extended to declare `sep_features` (list payload) and `sep_features_missing` (snapshot scalar) in its inventory.

### Conditional emission contract — preserves byte-identity for legacy callers

The schema deliberately mirrors the #307 pattern rather than the #305 / #306 pattern: `RICH_FEATURE_SIZE` (the module-level constant) does NOT widen when the flag is on. Instead, the SEP block is appended past the regime block by `FeatureVector.as_rich_list` only when `sep_features is not None`. The opt-out default keeps the per-bar feature size byte-identical to pre-#215 for every downstream caller that iterates slices inside `[0:RICH_FEATURE_SIZE]` (or inside the regime-widened width when `--use-regime-conditioning` is on).

The two opt-in blocks compose: the new helper `rich_feature_size_with_blocks(use_regime, use_sep)` returns the combined width so model factories widen the input projection in one place. The fixed append order (regime first, then SEP) means a checkpoint trained with one flag on still loads cleanly when the other is off — the block widths past `RICH_FEATURE_SIZE` are additive and the inactive block contributes nothing to the tensor.

The per-fold `RobustScaler` slice `[FEATURE_SIZE:RICH_FEATURE_SIZE]` is unchanged: the SEP block (which sits past the regime tail, past `RICH_FEATURE_SIZE`) is not scaled. Reason: the SEP scalars live in percentage points (typical range 0%-6% for FFR medians, 0-1.5 pp for the central-tendency range, exactly `{0.0, 1.0}` for the release flag), already in a tighter range than the rich-feature block after the IQR transform. Running a per-fold scaler on a release-flag column whose values are exactly `{0.0, 1.0}` would also trigger the constant-column guard (IQR < epsilon → IQR floored to 1.0), reducing the transform to a pure centering step — a no-op on the {0, 1} support after centering at the train median.

### CLI surface

`--use-sep` / `--no-sep` on `app.train_forecaster`. Threaded through both `load_walk_forward_split` call sites (walk-forward + single-fold) and into the `ModelConfig.use_sep` field so the factory widens the input projection at build time. The report-payload JSON gains a `rich_feature_families.sep` boolean so the sweep aggregator can group cells by the flag's state.

## Alternatives considered

**Parse the SEP PDF directly from the Fed's release archive.** Rejected as the primary path. The Fed publishes the SEP as a structured PDF, and pulling the median values requires a PDF-parsing layer with its own brittleness and a per-meeting layout-fingerprint test. The FRED republished series cover the same numbers with a stable schema and the existing FRED client / SOURCES.lock plumbing the MP-surprise builder already uses. The CSV fixture path is the operator's escape hatch when FRED's coverage lags or when the operator wants to pin a manually-vetted source.

**Emit the full dot-plot as a histogram per horizon.** Rejected on the small-corpus tax. Each SEP releases ~19 dots per horizon × 4 horizons = ~76 numbers per release. With ~250 supervised events and ~50 SEP releases over the training window, fitting per-dot features would burn most of the available degrees of freedom on a single feature family. The headline three numbers (current-year / next-year / longer-run median) plus the current-year central-tendency range capture the dot-plot's first two moments (location + dispersion) per horizon at a fraction of the dimension cost.

**Add SEP projections for GDP / unemployment / PCE inflation.** The SEP table publishes these alongside the FFR projections. Rejected as the initial scope: the FFR median is the most-directly-relevant signal for a rate-and-vol forecaster, the other three series are downstream of the FFR via the Committee's model of the economy, and adding them would multiply the feature surface threefold for a small marginal lift on a small corpus. A follow-up issue can extend the block once the per-family ablation on the FFR-only version is in.

**Add SEP as a `T+Δ` post-event signal.** The SEP is sometimes treated in the literature as the market's read on the meeting outcome (i.e. what the dots said determines the next-meeting reaction). Rejected because the SEP is itself released at `T` — it IS part of the meeting outcome the model is supposed to be reading at `T`, not a post-event read of the market reaction. Treating it as `T+Δ` would mis-classify the strict-prior contract and break the audit invariant.

**Add a separate "SEP fresh" architectural gate.** Rather than a release-flag indicator scalar, mount a gate on the SEP block that activates only on SEP-release meetings. Rejected on the SNR ceiling argument: the model can learn the interaction between the release flag and the SEP scalars through the existing recurrent core's hidden state. Adding a dedicated gate doubles the post-#215 model surface for a marginal interpretability gain that the in-bar scalar provides for free.

## Consequences

### Honest framing per #334

The #334 substitution finding showed that stacking small feature families on top of the existing rich-feature input can give negative interaction lift. The SEP block is small (5 scalars + 1 flag) so the dimensional tax is bounded, but the methodology contribution is the addition of the FOMC's own forward-guidance instrument to the input surface — not the headline cell against the flag. The canonical sweep against `--use-sep` may surface a positive, negative, or null lift on the dual-head regime-classification headline; the per-family ablation row populates with whatever number the operator measures.

The case for shipping the code path anyway:

- The architectural decision (treat the FOMC's explicit forward-guidance instrument as a model input, with the same strict-prior + conditional-emission contract every other rich-feature family rides) is a defensible methodology contribution on its own.
- The block is small (5 scalars + 1 flag) and the per-event computation is cheap (one dict lookup against the SEP-projections parquet), so the comparison sweep is cheap.
- The flag-off default keeps every existing sweep byte-identical, so the change is non-disruptive on the canonical sweep grid.
- The forward-fill + release-flag design lets the model learn the interaction between "fresh projections" and the reaction to the released document without burning a separate architectural gate on the surface.

### Methodology

The strict-prior contract is the load-bearing claim. SEP release-day values are `T (snapshot)` — observable from the SEP document released simultaneously with the FOMC statement, same band as the `stance_*` text features. Forward-filled rows are `T-Δ` — read from a prior meeting's SEP whose own `meeting_date < event_date`. The audit row in `docs/feature-provenance-audit.md` classifies both branches; the regression test in `tests/regression/test_feature_provenance_as_of.py` declares the new fields.

The forward-fill convention is documented as the chosen path; the two alternatives (emit `None`, emit zeros) are recorded in the "Alternatives considered" section so a reviewer can audit the design choice without re-deriving the trade-off.

### Model + checkpoint

The recurrent core's `lstm_input_size` widens by `RICH_SEP_DIM + RICH_SEP_MISSING_DIM = 6` when the flag is on, mirroring the regime-tail widening from #307. The state_dict gains no new keys (the widening sits on the input projection of the recurrent core, which already exists); existing checkpoints with the legacy width deserialise unchanged on `--no-sep`. The opt-in path requires a fresh sweep because the recurrent core's input projection has a different number of input units; the per-fold rich-feature scaler stays anchored on `[FEATURE_SIZE:RICH_FEATURE_SIZE]` (unchanged slice).

### Compute

The SEP computation is a pure-Python composition over a tiny parquet (~50 rows for the full training window). No new I/O beyond the one-shot parquet read at loader startup; the per-event cost is dominated by a single dict scan over the eligible SEP releases. Full canonical training package stays well under the +10s budget mentioned in the spec. The CI smoke test stays under 60s on CPU.

### Sweep hand-off

The headline cell against `--use-sep` is a Runpod follow-up. The §16 comparison table populates with both modes side-by-side as the numbers arrive. Default `--no-sep` runs against the canonical training package remain byte-identical so the existing sweep numbers stay reproducible.
