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

## Leakage Rules
1. No future-derived features
2. No future target leakage in feature creation
3. No near-duplicate leakage across train/test in the same fold
4. Pseudo-labeling excludes final reporting holdout
5. Scaling/statistics fit on train only

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
3. The encoder may stay in `phase3_finetune_batch.ENCODERS` for completeness, but the published table must use only independent encoders (BERT-base, FinBERT, FinBERT-FOMC, DistilBERT, DeBERTa-v3).
4. Any future encoder whose pretraining mixture overlaps with this project's training rows falls under the same exclusion rule. Add a one-line entry to the deny-list in `phase3_finetune_batch.py` and document the rationale in `06_Deep_Learning_Roadmap.md`.
