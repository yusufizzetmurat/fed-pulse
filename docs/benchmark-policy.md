# Benchmark Policy

## Status
This policy is frozen for the current benchmark cycle.  
Any methodological change requires a new version.

## Evaluation Rules

### Scope
- Targets: close and volatility forecasts
- Horizons: `1d`, `3d`, `5d`, `10d`
- Modes: static and adaptive

### Split Policy
- Expanding walk-forward only.
- No random global split for official reporting.
- For each fold: `train_end < val_start < test_start`.
- Use identical fold boundaries across compared models.

### Seed Policy
- Official seed set: `{11, 29, 47, 71, 97}`.
- Report mean and standard deviation across seeds.
- Non-approved seeds are exploratory only.

### Metrics
- Forecast quality: RMSE, MAE, MAPE
- Reliability: coverage (and calibration error if implemented)
- Runtime: latency `p50`/`p95`, adaptation time, peak memory (if available)

### Leakage Rules
1. No future timestamp information in features.
2. No future target values in feature generation.
3. No near-duplicate source rows crossing train/test in same fold.
4. Pseudo-label generation must exclude final reporting holdout.
5. Scaling/statistics must be fit on train only.

## Versioning Rules

Required identifiers:
- `dataset_version`
- `feature_version`
- `model_version`
- `run_id`
- `training_package_id`

Naming formats:
- `dataset_version`: `ds_<source>_<yyyy-mm-dd>_v<major>.<minor>`
- `feature_version`: `fv_<featurepack>_v<major>.<minor>`
- `model_version`: `mv_<family>_<mode>_v<major>.<minor>.<patch>`
- `run_id`: `run_<yyyymmdd>_<protocol>_<model_version>_<seed>`
- `training_package_id`: `tp_<dataset_version>_<feature_version>_<protocol>_v<major>.<minor>`

Immutability:
- published `run_id` cannot be reused
- checkpoints behind a `model_version` cannot be silently replaced
- protocol/split/feature changes require version bumps

## NLP Baseline Run Policy

Baseline model candidates:
- `bert-base-uncased`
- `ProsusAI/finbert` (or equivalent)
- `gtfintechlab/fomc-roberta-any-exp` (or equivalent)

All candidates must run on the same folds and seeds.

Winner rule:
1. highest macro-F1
2. best worst-class F1
3. lowest latency `p95` tie-breaker

## Official Report Must Include
- protocol id (`evaluation_protocol_v1`)
- all required version identifiers
- runtime mode, seed list, fold count
- per-horizon metrics + runtime metrics
- known deviations and failures

If any required rule fails, tag run as exploratory and exclude from final benchmark tables.
# Benchmark Policy

## Status
This policy is frozen for the current benchmark cycle.  
Any methodological change requires a new version.

## Evaluation Rules

### Scope
- Targets: close and volatility forecasts
- Horizons: `1d`, `3d`, `5d`, `10d`
- Modes: static and adaptive

### Split Policy
- Expanding walk-forward only.
- No random global split for official reporting.
- For each fold: `train_end < val_start < test_start`.
- Use identical fold boundaries across compared models.

### Seed Policy
- Official seed set: `{11, 29, 47, 71, 97}`.
- Report mean and standard deviation across seeds.
- Non-approved seeds are exploratory only.

### Metrics
- Forecast quality: RMSE, MAE, MAPE
- Reliability: coverage (and calibration error if implemented)
- Runtime: latency `p50`/`p95`, adaptation time, peak memory (if available)

### Leakage Rules
1. No future timestamp information in features.
2. No future target values in feature generation.
3. No near-duplicate source rows crossing train/test in same fold.
4. Pseudo-label generation must exclude final reporting holdout.
5. Scaling/statistics must be fit on train only.

## Versioning Rules

Required identifiers:
- `dataset_version`
- `feature_version`
- `model_version`
- `run_id`
- `training_package_id`

Naming formats:
- `dataset_version`: `ds_<source>_<yyyy-mm-dd>_v<major>.<minor>`
- `feature_version`: `fv_<featurepack>_v<major>.<minor>`
- `model_version`: `mv_<family>_<mode>_v<major>.<minor>.<patch>`
- `run_id`: `run_<yyyymmdd>_<protocol>_<model_version>_<seed>`
- `training_package_id`: `tp_<dataset_version>_<feature_version>_<protocol>_v<major>.<minor>`

Immutability:
- published `run_id` cannot be reused
- checkpoints behind a `model_version` cannot be silently replaced
- protocol/split/feature changes require version bumps

## NLP Baseline Run Policy

Baseline model candidates:
- `bert-base-uncased`
- `ProsusAI/finbert` (or equivalent)
- `gtfintechlab/fomc-roberta-any-exp` (or equivalent)

All candidates must run on the same folds and seeds.

Winner rule:
1. highest macro-F1
2. best worst-class F1
3. lowest latency `p95` tie-breaker

## Official Report Must Include
- protocol id (`evaluation_protocol_v1`)
- all required version identifiers
- runtime mode, seed list, fold count
- per-horizon metrics + runtime metrics
- known deviations/failures

If any required rule fails, tag run as exploratory and exclude from final benchmark tables.
