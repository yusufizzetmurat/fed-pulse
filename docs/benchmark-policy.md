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
Official reports must include protocol ID, all version IDs, runtime mode, fold/seed info, metrics, and known deviations.
