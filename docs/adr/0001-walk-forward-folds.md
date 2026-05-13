# 0001 — Walk-forward folds

**Status:** accepted
**Date:** 2026-03-15

## Context

FOMC events are sparse (~8 per year) and the publishable claim is about *forecast* quality — what the model knows about future returns given only information available at decision time. K-fold cross-validation on a time-ordered dataset leaks future information into training and produces optimistic metrics. The benchmark policy needs a split protocol that mirrors how the model would be used in production: train on the past, evaluate on the future.

## Decision

Use expanding walk-forward folds with strict `train_end < val_start < test_start` ordering. Five folds total. The same fold manifest is used across every model in a comparison so RMSE differences are attributable to the model, not the split.

## Consequences

- Fold 2 and fold 4 are the only folds with labelled test rows in the current corpus; folds 1, 3, 5 cover periods where the labelled set is empty on the test surface. Closing the coverage gap is a Phase 2/6 deliverable.
- Reporting holdout for v1 architecture closure is `wf_fold_3`. v2 evaluation uses the same fold so the headline numbers compare cleanly.
- Block-bootstrap CIs on the test fold replace 5-seed mean±SD wherever the thesis quotes a comparison; the seed standard deviation captures head-init noise, not generalisation noise.
