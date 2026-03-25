# Project Guide

## Objective
`fed-pulse` evaluates whether FOMC-text signals improve short-horizon market forecasts, and whether runtime adaptation is worth its latency/compute cost.

## Research Questions
1. Does adaptation outperform static inference under the same split protocol?
2. What is the runtime cost (`p50`, `p95`, adaptation time)?
3. Is performance stable across different market regimes?

## Scope
- In scope: forecasting pipeline, adaptation policy, evaluation, reproducibility
- In scope: text + market feature integration
- Out of scope: macroeconomic theory as primary contribution

## Runtime Modes
- `fast`: checkpoint inference
- `quick_train`: short bounded adaptation
- `real_train`: async training with longer history (252 trading days)

## Work Plan
1. Lock protocol/versioning rules
2. Ingest and normalize sources
3. Build NLP features and align with market data
4. Train forecasting baselines
5. Run adaptation experiments
6. Perform ablations/robustness checks
7. Finalize integration and reporting artifacts

## Done Criteria
- All compared models use the same official split/seed policy
- Report includes quality + latency metrics
- Artifacts are versioned and reproducible
