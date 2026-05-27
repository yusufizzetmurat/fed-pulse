# ADR 0028 — Retrieval-augmented input features for the forecaster

Status: accepted, code path live; canonical sweep deferred to operator.
Date: 2026-05-27.
References:
- Issue #306 (closes).
- Issue #294 — historical-analog retrieval encoder + on-disk index + `/analyze/analogs` endpoint.
- Issue #295 — historical-analog panel (frontend, top-3 cards, 0.40 similarity floor).
- Issue #334 — text-feature substitution / interaction finding: stacking multiple small text-derived feature families on top of the rich-feature input may give negative interaction lift.
- ADR 0021 — retrieval supervision pair policy.
- `backend/app/services/analogs.py` — runtime singleton wrapping the retrieval index, the `find_analogs(text, k, as_of_date)` entry point.
- `backend/app/retrieval/index.py` — index query with strict-backward `as_of_date < event_date` filter and self-match-by-text_hash suppression.
- `backend/app/training/retrieval_features.py` — pure-Python derived-feature helper.
- `backend/app/training/loaders.py` — per-event call sites on both loader entry points.
- `backend/app/models/config.py` — `FeatureVector.analog_features` schema extension; slice constants.

## Context

The on-disk retrieval index built under #294 already serves the historical-analog panel (#295): given an event's text, it returns the top-K past FOMC statements ranked by cosine similarity, each carrying its stance and a coarse post-event vol-regime bucket. The bundle is built once per training package, queried at inference time with a strict-backward `as_of_date < event_date` filter, and exposed via `/analyze/analogs`.

#306 takes the same retrieval pipeline and wires it into the forecaster's training-input pipeline. For each event during loader assembly, query the index for the top-K analogs at training time and append a small derived feature block onto every bar of the supervised sequence. The block summarises the retrieval result — similarity moments, count above the panel's floor, stance-agreement against the current event — and rides the same per-fold `RobustScaler` slot the other rich-feature families use.

The motivation is architectural symmetry. The retrieval system has been an engineering investment of its own (encoder fine-tune, on-disk index, panel-side rendering); reusing the same `find_analogs` entry point as an input feature family on the regime / rates heads earns the system double duty without forking a parallel pipeline. The methodology contribution is precisely that symmetry: retrieval is used as a panel surface (Panel 3) and as an input feature family on the heads, both reading off the same cached cosine-similarity computation, under the same `as_of_date` walk-forward filter.

## Decision

Add an opt-in `--use-retrieval-analogs` flag on `app.train_forecaster`. Default OFF — when off the new fields stay `None` and `FeatureVector.as_rich_list()` emits zeros into the 6 new slots, byte-identical to pre-#306. When ON, the loader queries the retrieval index per event and writes a 5-scalar contextual summary block + 1-dim missing flag onto every bar of the supervised sequence.

### Feature block

Five scalars, in the slice order documented on `RICH_RETRIEVAL_ANALOG_SLICE`:

| Feature | Definition |
| --- | --- |
| `analog_max_similarity` | max cosine similarity in the top-K result set |
| `analog_mean_similarity` | mean cosine similarity over the top-K |
| `analog_similarity_dispersion` | population std of the top-K similarities (clamped `>= 0`; population not sample so the value is defined at `n=1`) |
| `analog_count_above_floor` | count of hits with similarity `>= 0.40`, normalised against `top_k` so the value sits in `[0, 1]` |
| `analog_max_stance_score` | fraction of analogs whose canonical `axis_stance` matches the current event's stance |

Plus a paired `analog_features_missing` flag (1.0 when the retrieval bundle is absent on disk, the flag is off, or the lookup returns an empty hit list; otherwise 0.0).

### K and similarity floor

`K = 3`, matching the #295 panel's display top-K. The trade-off: small K keeps the summary stats stable across runs and matches the set of analogs a human reviewer sees on the UI. Larger K (5, 10) would smooth the similarity moments but dilute the stance-agreement signal because rank-5 hits are typically well below the 0.40 floor on the production index. The panel-side default is the natural anchor; the trainer reuses it.

Similarity floor `= 0.40`, matching the #295 panel's `MIN_SIMILARITY` constant. Below this, hits are considered too weak to count as a contextual signal — both for the panel's "no analogs" empty state and for the trainer's `analog_count_above_floor` numerator.

### Strict-backward filter

The retrieval call enforces `analog_event_date < event_date` at the index-query level via the existing `app.retrieval.index.query(..., as_of_date=event_date)` cutoff. The gate is strict `<` (not `<=`), the same gate the panel uses at request time. Self-match suppression by `text_hash` runs unconditionally inside the runtime singleton's `find_analogs` helper.

The loader threads each event's own `event_date` (not a fold boundary, not a relaxed cutoff) into the retrieval call. The `tests/unit/test_retrieval_augmented_features.py::test_loader_retrieval_query_uses_strict_backward_filter` regression pins the contract.

### Leak audit

The analog's post-event observed move (`forward_realized_vol_10d` or the `subsequent_vol_regime` bucket the index exposes) is **not** in the feature block. Admitting it would be a label leak via similarity: two near-identical past statements share most of the surprise direction and a non-trivial fraction of the post-event vol response, so a feature that admitted the analog's outcome would let the forecaster read a lossy copy of its own target through the cosine-similarity gate. The block is therefore restricted to contextual stats over the retrieval result (similarity moments + stance-agreement count); the analog's outcome is excluded by construction.

The per-feature row in `docs/feature-provenance-audit.md` classifies the block as strict-prior. The analog event dates are strict-prior by the index-query cutoff above; the current-event stance read driving the `analog_max_stance_score` numerator is `T (snapshot)` — observable from the released FOMC text on `T` itself, identical to how the existing `stance_hawk` / `stance_dove` / `stance_neutral` block is classified.

### Per-fold standardisation

The new 5-scalar slot rides the same `RobustScaler` slot the other rich-feature families use (median / IQR fitted on the train slice only via `fit_rich_feature_scaler_tensor`, val / test reuse the train scaler). No bespoke per-family standardiser; the existing `apply_rich_feature_scaler_tensor` slices over `[FEATURE_SIZE:RICH_FEATURE_SIZE]` and picks up the new positions automatically since the constants widen in lockstep.

### Graceful degrade

When the retrieval bundle is absent on disk (ops deployments that do not ship the bundle alongside the training package), `app.services.analogs.find_analogs` returns `None` and the loader collapses every event to the all-zeros + missing-flag-1.0 state. Same contract when `--use-retrieval-analogs=False` (default). The training run completes without crashing in either case; the model just sees the missing block and treats the slot as "no retrieval signal."

### Per-family ablation impact prediction — honest framing

The #334 substitution finding (text-feature substitution / interaction sweep) showed that stacking multiple small text-derived feature families on top of the rich-feature input may give negative interaction lift: every new text-derived family has to fight the others for variance because the underlying signal is shared. The block here is itself text-derived (the retrieval encoder is a fine-tuned sentence transformer over FOMC + cross-bank text), so the headline lift may be negative or near-zero rather than positive.

We ship the code path anyway because:

- The architectural symmetry (retrieval used as panel surface AND as input feature family, off the same encoder bundle and the same `as_of_date` filter) is a defensible methodology contribution on its own.
- The block is small (5 scalars + 1 flag). It does not blow up the input dimension and the comparison sweep against `--use-retrieval-analogs` is cheap.
- The canonical sweep can report the honest delta and the report frames the result accordingly — positive, negative, or null, the methodology claim stands.

The headline cell against this flag is a Runpod follow-up; the operator runs the canonical sweep against `--use-retrieval-analogs` once the code path lands.

## Alternatives considered

**Include the analog's post-event observed move as an input feature.** Most direct interpretation of "retrieval-augmented forecasting": copy the analog's `forward_realized_vol_10d` (or the `subsequent_vol_regime` bucket) onto the supervised event's input vector. Rejected: label leak via similarity. The retrieval gate is cosine similarity in encoder space; the encoder is fine-tuned for the same supervised signal the forecaster reads, so the top-K analogs by construction concentrate similarity in regions of the encoder's input space that correlate with the post-event outcome. Admitting the analog's outcome therefore admits a lossy copy of the supervised target. The contextual-only block sidesteps the leak without losing the methodology claim.

**Train a parallel retrieval-augmented head with the analog's outcome as a target.** Rather than feature, treat the analog's outcome as a separate auxiliary supervisory signal — train a head that predicts the next event's outcome from the top-K analog outcomes. Rejected: the canonical regime head is already supervised on the strict-forward observed outcome (#322 / ADR 0015), and an auxiliary head off the analog's outcome would either duplicate the main supervisory signal (no methodological lift) or fight it through the joint loss (interaction tax under the #334 finding). The contextual feature path is a smaller surface and a cleaner methodology claim.

**Use K=10 (or larger) to smooth the similarity moments.** Larger K would reduce the variance of the similarity-mean and similarity-dispersion stats. Rejected: on the production index, hits beyond rank 5-7 typically score below the 0.40 floor, so they dilute the `analog_count_above_floor` numerator without adding signal to the moments. K=3 also matches the panel's display top-K, which keeps the two surfaces honest (the trainer summarises the same set of analogs the human reviewer sees).

**Compute the block at inference time and persist it on the events.parquet.** Move the retrieval call from the loader into the events-package builder so the block is materialised on disk. Rejected: the events package is built once and re-read across folds / seeds; the retrieval bundle's `as_of_date` cutoff is per-event, so a one-time bake at events-builder time would have to pre-emptively materialise the right cutoff for every event, blowing up the events parquet's row count. The loader-time call reuses the same `find_analogs` entry point the panel uses at request time and stays in the per-fold scaler's purview — cleaner and consistent with how other rich-feature families are wired.

**Add the block to `_DERIVED_TEXT_SLICES` so the #309 derived-text-features toggle covers it.** The block is text-derived (the retrieval encoder runs over text), so a `--no-derived-text-features` ablation could plausibly include it. Rejected: the block is opt-in via its own flag (`--use-retrieval-analogs`) and the derived-text-features ablation predates it; bundling them under the same toggle would make the per-family ablation matrix less interpretable. Operators who want to zero out every text-derived feature family flip both flags off explicitly.

## Consequences

- New `--use-retrieval-analogs` CLI flag on `app.train_forecaster`. Default off; existing sweeps and the canonical determinism regression stay byte-identical.
- `RICH_FEATURE_SIZE` widens by 6 (5 scalars + 1 missing flag). Old rich-feature checkpoints carrying the pre-#306 width become incompatible at load time; the next sweep refits both the model and the per-fold `RobustScaler`. Pre-#306 6-feature checkpoints (the legacy `as_list` path) are unaffected — those models do not see the rich-feature input.
- The per-fold `RobustScaler` slot picks up the new 5 positions automatically since it slices over `[FEATURE_SIZE:RICH_FEATURE_SIZE]`. Missing-flag column is constant (always 0.0 or 1.0); the scaler's constant-column guard reduces the transform on that slot to a centering step.
- The `FeatureVector.analog_features` slot is `list[float] | None`. The dataclass field-name machinery in `_coerce_model_config` already round-trips arbitrary new fields off the dict payload without bespoke wiring (the surface added under #305 for the `_fomc_attributable` fields).
- Compute: ~50-100 ms per event on the retrieval call (CPU-bound cosine similarity on a ~250-row index, K=3). Full canonical training package (~250 events) adds well under +30s end-to-end. The CI smoke test stays under 60s on CPU.
- HF Hub interaction: none. The retrieval bundle is already loaded locally by the runtime singleton under #294; the loader reuses the same singleton and does not push or pull.

## Code-path PR, not a headline cell

This PR ships the code path. The canonical sweep against `--use-retrieval-analogs` is the operator's Runpod hand-off. The §16 comparison table populates with both modes side-by-side as the numbers arrive.
