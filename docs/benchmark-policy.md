# Benchmark Policy

## Status
Policy is fixed for the current benchmark cycle. Method changes require a new version.

## Evaluation Scope
- Targets: close, volatility
- Horizons: `1d`, `3d`, `5d`, `10d`
- Modes: `fast`, `quick_train`, `real_train` (when applicable)

## Split and Seed Rules
- Split: expanding walk-forward only
- Constraint: `train_end < val_start < test_start`
- Use identical folds across model comparisons
- Official seed set: `{11, 29, 47, 71, 97}`
- Report mean and standard deviation across seeds

## Metrics
- Forecast quality: RMSE, MAE, MAPE
- Reliability: coverage (and calibration error if available)
- Runtime: latency `p50`/`p95`, adaptation time, peak memory (if available)

## Canonical Training Objective
As of ADR 0015 (`docs/adr/0015-regression-canonical-objective.md`, issue #322), the canonical training objective for the vol-regime head is the regression head on `log(forward_realized_vol_10d)`, optimised with MSE on per-fold standardised log-RV. The 3-class calm / normal / high label is a UI-side bucketing of the regression output against the per-fold `vol_regime_quantiles` cutoffs persisted in `fold_manifest_expanding_walk_forward.json`; it is no longer a training target under the canonical setting. The classification head stays mounted on every checkpoint for shape stability and to back the aux conformal calibration surface (`rates_softmax_quantiles`), but contributes zero gradient when `head_mode="regression"`. The dual-head mixing surface (`head_mode={classification,regression,dual}` plus `regression_alpha`) introduced in #304 remains available for the methodology comparison in §6.15 of the deep-learning roadmap. Headline vol-regime rows in published reports must cite RMSE-log_rv and R² alongside any UI-derived classification metrics.

## Aggregation Rule
The canonical pooling rule for macro-F1 across walk-forward folds is **mean-of-fold-means**: compute per-fold macro-F1, then average unweighted across folds. The secondary rule is **row-pooled**: concatenate every per-row prediction across all folds into one confusion matrix, then compute macro-F1 once. Both numbers are published on every honest macro-F1 cell so the per-fold class-balance variance is visible — the row-pooled number weights folds by support, the mean-of-fold-means number weights folds equally and is the conservative read against the `wf_fold_4` zero-`calm` slice (R-17).

Honest macro-F1 reports MUST include:
1. **Per-class P/R/F1 footnote** on every published macro-F1 cell — precision, recall, F1, and support per class. Pooled headlines that suppress this footnote can mask a degenerate per-class distribution and are not eligible for publication.
2. **Fold-4 with-and-without rows** on every headline cell — publish both `with all folds` and `without wf_fold_4` so the reader can see the magnitude of the zero-`calm`-class fold's contribution. Flag if the two readings diverge by more than the bootstrap CI half-width.
3. **Macro-release with-and-without rows** on every headline cell — publish both `with macro-release-augmented rows` and `FOMC-only` so the §6.7 Chunk-3 lift attribution stays auditable. The FOMC-only cell is the primary thesis number; the mixed-pool cell is the secondary comparator.

The four reporting variants (mixed-pool, FOMC-only, fold-4 with/without, macro-release with/without) are computed by `backend/app/evaluation/reporting.py` (issue #323). Block-bootstrap CIs use 1000 resamples at `block_size=20` by default, matching the convention in `backend/app/evaluation/regime_pooled_aggregator.py`.

## Leakage Rules
1. No future-derived features
2. No future target leakage in feature creation
3. No near-duplicate leakage across train/test in the same fold
4. Pseudo-labeling excludes final reporting holdout
5. Scaling/statistics fit on train only

## Per-Feature Provenance
Every `FeatureVector` column carries a declared `as_of` offset relative to `row.event_date`. The declarations live in `docs/feature-provenance-audit.md` and cover four bands: `T-Δ` (data observable strictly before the event), `T (snapshot)` (a quantity defined on the event itself and observable from the released document or its calendar entry), `T+Δ` (post-event data), and `future-derived` (training targets only).

Contract:
1. No scalar feature column on a lookback bar may read from a source post-dating `row.event_date`. Lookback bars consume only `T-Δ` data; `T (snapshot)` columns are document-level signals on `T` itself.
2. `T+Δ` columns are training targets only and are mounted on the appended event-day target frame, not on lookback bars.
3. New `FeatureVector` columns require a row in the audit table before merge; the regression test enforces the column inventory.
4. `T (snapshot)` columns whose sources are post-event by construction (e.g. monetary-policy surprise quantities defined on a `[T-1, T+1]` window) are flagged in the audit's "Leaks found" section and tracked through an ADR + canonical-cell re-baseline.

Regression coverage: `tests/regression/test_feature_provenance_as_of.py` walks a canonical training package and asserts the contract per row. The test is the gate; the audit is the human-readable inventory.

## Versioning
Required IDs:
- `dataset_version`
- `feature_version`
- `model_version`
- `run_id`
- `training_package_id`

Immutability rules:
- Published `run_id` is never reused
- Checkpoints behind a version are not replaced silently
- Protocol/split/feature changes require version bump

## NLP Baseline Selection
Candidates run on the same folds/seeds. Winner order:
1. Macro-F1
2. Worst-class F1
3. Latency `p95` (tie-break)

## Report Requirements
Official reports must include protocol ID, all version IDs, runtime mode, fold/seed info, metrics, and known deviations. Reports that include source-type-stratified breakdowns must report per-`source_type` mean and standard deviation alongside the headline aggregates so cross-source generalisation is visible.

## Source-mix Stratification
Labelled sources have different stance priors (TDW skews hawkish 49/26/24, Op-Fed skews dovish 59/30/11). To prevent a model from learning the source instead of the stance, every published training package must record:
- `source_drift_max` per fold in `fold_manifest_*.json` — the largest absolute share-point gap between the train slice's source distribution and either the val or test slice (e.g. 0.50 = train is 100% TDW but val is 50/50 TDW/Op-Fed).
- `source_drift_per_fold` and `source_drift_max_across_folds` in `dataset_metadata.json` for a one-glance package-wide check.

`build_training_package.py --source-drift-tolerance T` (default `0.0`, report only) aborts the build with a non-zero exit code if any fold's drift exceeds `T`. The convention for published packages is `T = 0.15` (15 percentage points); packages built without the gate must record `source_drift_tolerance: 0.0` in the manifest so reviewers can see the check was disabled.

## Contamination Handling
Encoders that may have seen the project's training rows during their own pretraining must be excluded from the primary NLP comparison table. The current case: `gtfintechlab/FOMC-RoBERTa` was trained on the same Trillion Dollar Words corpus that this project ingests as `hf_fomc_communication`. The 1-epoch smoke macro-F1 of 0.8409 on fold-2-test is almost certainly a test-leak signal.

Policy:
1. Exclude `gtfintechlab/FOMC-RoBERTa` from the headline comparison row.
2. Cite the contaminated number as a **ceiling reference** in the discussion chapter only — labelled as "upper-bound for what's achievable on this corpus given likely contamination".
3. The encoder may stay in `finetune_batch.ENCODERS` for completeness, but the published table must use only independent encoders (BERT-base, FinBERT, FinBERT-FOMC, DistilBERT, DeBERTa-v3).
4. Any future encoder whose pretraining mixture overlaps with this project's training rows falls under the same exclusion rule. Add a one-line entry to the deny-list in `finetune_batch.py` and document the rationale in `06_Deep_Learning_Roadmap.md`.
