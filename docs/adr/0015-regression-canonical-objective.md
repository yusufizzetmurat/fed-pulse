# ADR 0015 — Regression head as canonical vol-regime training objective

Status: accepted, in production (as of merge).
Date: 2026-05-26.
Supersedes: the classification-canonical default established pre-#304, under which the vol-regime head was trained directly on the 3-class calm / normal / high label.
References:
- Issue #322, depends on #304 (merged PR #314, squash `e88dbfa`).
- `backend/app/models/config.py` — `head_mode` default flip from `"classification"` to `"regression"`.
- `backend/app/services/regime_bucketing.py` — new UI-side bucketing helpers that read the per-fold `vol_regime_quantiles` cutoffs.
- `scripts/run_dual_head_comparison.py` — three-way comparison runner across `classification`, `regression`, and `dual` head modes.
- `artifacts/experiments/dual_head_comparison_canonical.json` — comparison artefact committed in this PR.
- `fed-pulse.wiki/06_Deep_Learning_Roadmap.md §6.15` — methodology paragraph and populated three-way comparison table.
- `fed-pulse.wiki/16_Finalization_Roadmap.md §7` — MUST-tier acceptance bullet for #322.

## Context

The vol-regime classifier hit a documented macro-F1 ceiling of 0.4538 with a block-bootstrap 95% interval of [0.434, 0.469] under the strict-forward target convention (`06_Deep_Learning_Roadmap §6.7` row 10). The 5-seed × 4-fold Bundle A.2 sweep diagnosed the ceiling as structural rather than as a gradient-flow or capacity issue.

Two observations drove the diagnosis. First, class-1 (`normal`) recall collapsed to roughly 0.07 across both cross-bank arms, and the `class_weight_power=2.0` null probe did not move it; head-side reweighting cannot recover a class whose decision boundary is itself unstable across folds. Second, the per-fold tertile edges (33rd and 67th percentile of the continuous target) sit in the meat of the `log_rv_10d` distribution where data density is highest, so small shifts in train composition move both edges meaningfully — the middle class is a moving target by construction.

Quantile classification on a continuous target therefore has a structural macro-F1 ceiling that no head-side trick — class weights, focal loss, label smoothing — can break. The issue is target construction, not training dynamics. PR #314 (issue #304) shipped the dual-head retrofit and the regression branch over `log(forward_realized_vol_10d)`, validated checkpoint-shape stability across head modes, and showed that the regression branch carries the directional signal the classification branch was approximating. #322 promotes the regression head to canonical.

## Decision

The canonical training objective for the vol-regime head is the regression head on per-fold standardised `log(forward_realized_vol_10d)`, optimised with MSE. The 3-class calm / normal / high label is retained only as a UI-side bucketing of the regression output, scored against the existing per-fold `vol_regime_quantiles` cutoffs persisted in `fold_manifest_expanding_walk_forward.json`.

The classification head stays mounted on every checkpoint for two reasons: shape stability (existing serving paths and saved checkpoints continue to load unchanged), and the aux conformal calibration surface (`rates_softmax_quantiles`) that the conformal layer reads. Under `head_mode="regression"` the classification head contributes zero gradient.

`head_mode` default flips from `"classification"` to `"regression"` in `backend/app/models/config.py`. Checkpoints with an explicit `head_mode="classification"` on disk continue to load and serve unchanged — the flip is a default, not a forced migration. The dual-head mixing surface (`head_mode="dual"` with `regression_alpha`) stays available for the §6.15 methodology comparison and for ablation work.

## Consequences

- The headline vol-regime row in `06_Deep_Learning_Roadmap §6.10` row 10 is annotated as deprecated-as-canonical. A new row reports regression metrics — RMSE on standardised log-RV, R², and directional accuracy on the UI-derived bucket — each with block-bootstrap 95% intervals across the official seed set.
- R-18 in `09_Risk_Register` reframes: the previously cited macro-F1 ceiling was a target-construction artefact, not a signal-quality verdict on the FOMC-conditioned vol forecaster.
- Class-conditional conformal coverage (#326) and the UI alignment work that consumes `log_rv_point` plus the conformal band (#338) become the natural follow-ups; both were blocked on the canonical-objective decision landing.
- `/analyze` `regime_classification` now carries `log_rv_point` and band fields. The `bucket_source` field flags whether `argmax_class` came from the classification head or from the UI-side bucketing of the regression output, so the legacy dashboard renders correctly during the #338 transition.
- The comparison artefact's headline numbers — macro-F1 for the classification arm, RMSE-log_rv and R² for the regression arm, and the dual-head crossover — are cited at <from artifacts/experiments/dual_head_comparison_canonical.json — populated in Wave 3>. The Wave 4 orchestrator replaces this stub with concrete values from the comparison run.
