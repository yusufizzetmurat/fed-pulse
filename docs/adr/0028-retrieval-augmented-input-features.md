# ADR 0028 — Retrieval-augmented input features for the forecaster

The on-disk retrieval index built under #294 already serves the historical-analog panel (#295): given an event's text, it returns the top-K past FOMC statements ranked by cosine similarity, each carrying its stance and a coarse post-event vol-regime bucket. The bundle is built once per training package, queried at inference time with a strict-backward `as_of_date < event_date` filter, and exposed via `/analyze/analogs`.

#306 takes the same retrieval pipeline and wires it into the forecaster's training-input pipeline. For each event during loader assembly, the loader queries the index for the top-K analogs and appends a small derived feature block onto every bar of the supervised sequence. The block summarises the retrieval result — similarity moments, count above the panel's floor, stance-agreement against the current event — and rides the same per-fold `RobustScaler` slot the other rich-feature families use.

The motivation is reuse. The retrieval system has been an engineering investment of its own (encoder fine-tune, on-disk index, panel-side rendering); reusing the same `find_analogs` entry point as an input feature family on the regime / rates heads earns the system double duty without forking a parallel pipeline. The methodology angle is the symmetry: retrieval used as a panel surface and as an input feature family on the heads, both reading off the same cached cosine-similarity computation under the same `as_of_date` walk-forward filter.

## What lands

An opt-in `--use-retrieval-analogs` flag on `app.train_forecaster`. Default OFF — the new fields stay `None` and `FeatureVector.as_rich_list()` emits zeros into the 6 new slots, byte-identical to pre-#306. When ON, the loader queries the retrieval index per event and writes a 5-scalar contextual summary block plus a 1-dim missing flag onto every bar.

Five scalars in `RICH_RETRIEVAL_ANALOG_SLICE` order:

| Feature | Definition |
| --- | --- |
| `analog_max_similarity` | max cosine similarity in the top-K result set |
| `analog_mean_similarity` | mean cosine similarity over the top-K |
| `analog_similarity_dispersion` | population std of the top-K similarities (clamped `>= 0`; population not sample so the value is defined at `n=1`) |
| `analog_count_above_floor` | count of hits with similarity `>= 0.40`, normalised against `top_k` so the value sits in `[0, 1]` |
| `analog_stance_agreement_fraction` | fraction of analogs whose canonical `axis_stance` matches the current event's stance |

Plus a paired `analog_features_missing` flag (1.0 when the retrieval bundle is absent on disk, the flag is off, or the lookup returns an empty hit list; 0.0 otherwise).

`K = 3` matches #295's display top-K. Small K keeps summary stats stable across runs and matches what a human reviewer sees on the UI; larger K (5, 10) smooths the similarity moments but dilutes stance-agreement because rank-5 hits typically sit well below the 0.40 floor on the production index. Similarity floor `= 0.40` matches the panel's `MIN_SIMILARITY` — below this, hits are too weak to count as a contextual signal, for both the panel's empty state and the `analog_count_above_floor` numerator.

## Strict-backward and leak audit

The retrieval call enforces `analog_event_date < event_date` at the index-query level via the existing `app.retrieval.index.query(..., as_of_date=event_date)` cutoff. Strict `<`, the same gate the panel uses at request time. Self-match suppression by `text_hash` runs unconditionally inside the runtime singleton's `find_analogs`. The loader threads each event's own `event_date` into the retrieval call; `tests/unit/test_retrieval_augmented_features.py::test_loader_retrieval_query_uses_strict_backward_filter` pins the contract.

The analog's post-event observed move (`forward_realized_vol_10d` or the `subsequent_vol_regime` bucket the index exposes) is not in the feature block. Admitting it would be a label leak via similarity: two near-identical past statements share most of the surprise direction and a non-trivial fraction of the post-event vol response, so admitting the analog's outcome lets the forecaster read a lossy copy of its own target through the cosine-similarity gate. The block is restricted to contextual stats; the analog's outcome is excluded.

The per-feature row in `docs/feature-provenance-audit.md` classifies the block as strict-prior. Analog event dates are strict-prior by the index-query cutoff; the current-event stance read driving `analog_stance_agreement_fraction` is `T (snapshot)` — observable from the released FOMC text on `T` itself, identical to how `stance_hawk` / `stance_dove` / `stance_neutral` is classified.

## Standardisation and degrade

The new 5-scalar slot rides the same `RobustScaler` slot the other rich-feature families use (median / IQR fitted on the train slice via `fit_rich_feature_scaler_tensor`, val/test reuse the train scaler). No bespoke per-family standardiser; the existing `apply_rich_feature_scaler_tensor` slices over `[FEATURE_SIZE:RICH_FEATURE_SIZE]` and picks up the new positions automatically as the constants widen in lockstep.

When the retrieval bundle is absent on disk (ops deployments that don't ship it alongside the training package), `app.services.analogs.find_analogs` returns `None` and the loader collapses every event to all-zeros + missing-flag-1.0. Same contract when `--use-retrieval-analogs=False`. The run completes either way; the model sees the missing block and treats the slot as "no retrieval signal."

## Per-family expectation

The #334 substitution finding (text-feature substitution / interaction sweep) showed that stacking multiple small text-derived feature families on top of the rich-feature input can give negative interaction lift: every new text-derived family fights the others for variance because the underlying signal is shared. This block is text-derived (the retrieval encoder is a fine-tuned sentence transformer over FOMC + cross-bank text), so the headline lift may be negative or near-zero rather than positive.

The code path ships anyway. The architectural reuse (retrieval used as panel surface AND as input feature family, off the same encoder bundle and the same `as_of_date` filter) is defensible methodology on its own. The block is small (5 scalars + 1 flag), so it doesn't blow up the input dimension and the comparison sweep against `--use-retrieval-analogs` is cheap. The canonical sweep reports the honest delta and §16 frames the result accordingly — positive, negative, or null, the methodology claim stands.

## Why not the alternatives

Include the analog's post-event observed move as an input feature. Most direct interpretation of "retrieval-augmented forecasting": copy the analog's `forward_realized_vol_10d` (or `subsequent_vol_regime`) onto the supervised event. Rejected: label leak via similarity, as above.

Train a parallel retrieval-augmented head with the analog's outcome as a target. Rather than feature, treat the analog's outcome as a separate auxiliary supervisory signal. Rejected: the canonical regime head is already supervised on the strict-forward observed outcome (#322 / ADR 0015); an auxiliary head off the analog's outcome would either duplicate the main signal (no lift) or fight it through the joint loss (interaction tax under #334).

K=10 (or larger) to smooth the similarity moments. Rejected: hits beyond rank 5-7 typically score below 0.40 on the production index, so they dilute `analog_count_above_floor` without adding signal to the moments. K=3 also matches the panel's display top-K, keeping the trainer and human reviewer on the same set of analogs.

Compute the block at events-package build time and persist it on `events.parquet`. Rejected: the events package is built once and re-read across folds / seeds; the bundle's `as_of_date` cutoff is per-event, so a one-time bake would have to pre-emptively materialise the right cutoff for every event, blowing up the parquet's row count. The loader-time call reuses the same `find_analogs` entry point the panel uses at request time and stays in the per-fold scaler's purview.

Add the block to `_DERIVED_TEXT_SLICES` so the #309 derived-text-features toggle covers it. The block is text-derived, but it's opt-in via its own flag, and the derived-text-features ablation predates it; bundling them under the same toggle makes the per-family ablation matrix less interpretable. Operators who want every text-derived family zeroed flip both flags off explicitly.

## Downstream effects

`RICH_FEATURE_SIZE` widens by 6 (5 scalars + 1 missing flag). Old rich-feature checkpoints carrying the pre-#306 width become incompatible at load time; the next sweep refits both the model and the per-fold `RobustScaler`. Pre-#306 6-feature legacy checkpoints (the `as_list` path) are unaffected — those models don't see the rich-feature input. The missing-flag column is constant (0.0 or 1.0); the scaler's constant-column guard reduces the transform on that slot to a centering step.

`FeatureVector.analog_features` is `list[float] | None`. The dataclass field-name machinery in `_coerce_model_config` already round-trips arbitrary new fields off the dict payload without bespoke wiring (the surface added under #305 for the `_fomc_attributable` fields).

Compute: ~50-100 ms per event on the retrieval call (CPU-bound cosine similarity on a ~250-row index, K=3). Full canonical training package (~250 events) adds well under +30 s end-to-end. The CI smoke stays under 60 s on CPU. No HF Hub interaction — the runtime singleton already loads the retrieval bundle locally under #294.

The headline cell against `--use-retrieval-analogs` is a Runpod follow-up; §16 populates with both modes side-by-side as the numbers arrive.

## References

- `backend/app/services/analogs.py`, `backend/app/retrieval/index.py`
- `backend/app/training/retrieval_features.py`, `backend/app/training/loaders.py`
- `backend/app/models/config.py` (FeatureVector + slice constants)
- ADR 0021 (retrieval supervision pair policy); Issues #294, #295, #306, #334
