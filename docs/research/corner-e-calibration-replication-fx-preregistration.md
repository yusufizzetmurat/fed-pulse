# Corner E — Pre-registration: confirmatory replication of the Corner C h1 calibration finding

**Status:** pre-registered 2026-06-01, *before any result was inspected*. This is a **confirmatory**
test of a hypothesis **already fixed by discovery** (Corner C, h1). No horizon search, no feature
search, no transform search — a single pre-specified test on an independent asset.

## What is being confirmed (fixed, not chosen here)

Corner C found, on SPX RV, that the certainty signal `u = P(uncertain) − P(certain)` predicts the
**1-day** RV-forecast error magnitude `|residual|` incrementally over `[forecast level, lagged
|residual|]` (DM p=0.003, ΔMSE +0.26%, constant band under-covers more on high-`u` days). That
result was at h1 only, tiny, and h5/h22 were null — i.e. fragile. The honest test of whether it is
real is **replication on a sample C never touched**, not tuning on the same SPX residuals (which
C's own pre-registration forbade).

## Independent sample

**DXY (dollar index)** RV-forecast residuals — a different asset, an independent draw. We already
have the DXY QLIKE-DLq baseline from Corner D. SPX informed the *hypothesis*; DXY informs the *test*.

## Hypothesis (single, fixed)

- **H1:** on DXY h1 residuals, adding `u` to the `|residual|` predictor lowers OOS MSE (ΔMSE > 0),
  and the constant 90% band under-covers more on high-`u` days — the same effect and direction as C-h1.
- **H0:** no effect (ΔMSE = 0) and no conditional-coverage gap.

## Design (identical to Corner C's h1 cell, transplanted to DXY)

- Point forecast: the **unchanged** DXY QLIKE-DLq baseline ensemble (HAR lags on `r²` RV proxy,
  5 seeds × 5 folds, 300 epochs), forecast at **h1 only**.
- Target: `|residual_t| = |y_t − ŷ_t|`; predictors known as-of `t`: `[ŷ_t, |residual|_{t-1}]`
  (baseline) vs `⊕ [u_t]` (treatment). Expanding walk-forward over the pooled OOS, burn-in 100.
- `u`: leak-safe, as-of forward-filled certainty signal on the DXY calendar.

## Metric, significance, decision

- **Metric:** OOS MSE of `|residual|`; `ΔMSE = MSE(base) − MSE(base+u)`; Diebold–Mariano (NW-HAC lag 1).
- **Single confirmatory test → no multiplicity correction; α = 0.05 two-sided.**
- **Pre-registered confirmation:** DM `p < 0.05` **and** `ΔMSE > 0` **and** the conditional-coverage
  gap runs in the predicted direction (high-`u` under-covered). Corroboration is reported regardless.
- **Replicates →** C-h1 is real (text → forecast-uncertainty calibration); *then* a method upgrade
  is warranted (conformalized quantile regression with `u`, and/or classifier-entropy as the
  uncertainty feature) as a separate pre-registered step.
- **Fails →** C-h1 was the flicker the h5/h22 pattern hinted at; text calibration is closed.

## Caveat (stated in advance)

DXY's `r²` RV proxy is noisier than SPX 5-min RV, so power is lower; a null here is "no effect
strong enough to survive the noise," not a perfectly clean rejection. A replication *despite* the
noise would be the more striking outcome.

## Artifacts

- `data/artifacts/corner_e_calibration_replication_fx/result.json`. Committed before it exists.
