# Project Guide

## What This Project Is
`fed-pulse` studies event-conditioned market forecasting with deep learning.  
The core question is not only forecast accuracy, but whether bounded runtime adaptation improves results enough to justify added latency and compute.

## Thesis-Facing Goal
Build a reproducible benchmark that answers:
1. Does adaptation help vs static inference on the same data?
2. How much latency and compute overhead does it add?
3. Is behavior stable across different market conditions?

## Scope
- In scope: model families, adaptation policy, evaluation rigor, reproducibility.
- In scope: text + market features in a realistic forecasting workflow.
- Out of scope: macroeconomic theory as the primary contribution.

## Key Decisions
- Use both static and adaptive runtime modes:
  - `fast`: checkpoint inference only.
  - `quick_train`: bounded adaptation before inference.
- Compare at least three deep model families:
  - LSTM
  - GRU
  - one non-recurrent deep model (for example TCN/transformer-like forecaster)
- Keep one simple sanity baseline (naive or lightweight tabular).

## Execution Plan (Compact)

### Phase 1 - Governance Lock
- Freeze evaluation protocol and versioning policy.
- Use run templates for all official experiments.

### Phase 2 - Data Integration
- Ingest approved external sources and scraped corpus.
- Normalize labels, run quality/leakage checks.
- Export versioned training package + fold manifest.

### Phase 3 - NLP + Feature Pipeline
- Compare NLP candidates under fixed folds and seeds.
- Select winner and generate sentiment features.
- Align event features with market data.

### Phase 4 - Forecasting Baselines
- Train LSTM and GRU baselines.
- Add one non-recurrent deep baseline.

### Phase 5 - Adaptation Experiments
- Run adaptation grid under strict runtime budgets.
- Select default runtime policy and fallback behavior.

### Phase 6 - Robustness
- Run feature ablations (text-only, market-only, combined).
- Report regime-wise behavior and uncertainty calibration.

### Phase 7 - Integration
- Log prediction metadata for traceability.
- Validate dashboard diagnostics and overlays.

### Phase 8 - Thesis Packaging
- Produce reproducible final tables and figures from artifacts.

## Completion Criteria
- Static and adaptive models are compared under one fixed protocol.
- At least three model families are benchmarked.
- Ablation and robustness analyses are complete.
- Reported outputs are reproducible from versioned artifacts.
# Project Guide

## What This Project Is
`fed-pulse` studies event-conditioned market forecasting with deep learning.  
The core question is not only forecast accuracy, but whether bounded runtime adaptation improves results enough to justify added latency and compute.

## Thesis-Facing Goal
Build a reproducible benchmark that answers:
1. Does adaptation help vs static inference on the same data?
2. How much latency/compute overhead does it add?
3. Is the behavior stable across different market conditions?

## Scope
- In scope: model families, adaptation policy, evaluation rigor, reproducibility.
- In scope: text + market features in a realistic forecasting workflow.
- Out of scope: macroeconomic theory as the main contribution.

## Key Decisions
- Use both static and adaptive runtime modes:
  - `fast`: checkpoint inference only.
  - `quick_train`: bounded adaptation before inference.
- Compare at least three deep model families:
  - LSTM
  - GRU
  - one non-recurrent deep model (for example TCN/transformer-like forecaster)
- Keep one simple sanity baseline (naive or lightweight tabular).

## Execution Plan (Compact)

### Phase 1 - Governance Lock
- Freeze evaluation protocol and versioning policy.
- Use run templates for all official experiments.

### Phase 2 - Data Integration
- Ingest approved external sources and scraped corpus.
- Normalize labels, run quality/leakage checks.
- Export versioned training package + fold manifest.

### Phase 3 - NLP + Feature Pipeline
- Compare NLP candidates under fixed folds/seeds.
- Select winner and generate sentiment features.
- Align event features with market data.

### Phase 4 - Forecasting Baselines
- Train LSTM and GRU baselines.
- Add one non-recurrent deep baseline.

### Phase 5 - Adaptation Experiments
- Run adaptation grid under strict runtime budgets.
- Select default runtime policy and fallback behavior.

### Phase 6 - Robustness
- Run feature ablations (text-only, market-only, combined).
- Report regime-wise behavior and uncertainty calibration.

### Phase 7 - Integration
- Log prediction metadata for traceability.
- Validate dashboard diagnostics and overlays.

### Phase 8 - Thesis Packaging
- Produce reproducible final tables and figures from artifacts.

## Completion Criteria
- Static and adaptive models are compared under one fixed protocol.
- At least three model families are benchmarked.
- Ablation and robustness analyses are complete.
- Reported outputs are reproducible from versioned artifacts.
