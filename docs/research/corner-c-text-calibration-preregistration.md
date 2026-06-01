# Corner C — Pre-registration: does textual uncertainty predict *when* the RV forecast is unreliable?

**Status:** pre-registered 2026-06-01, *before any result was inspected*. Committed before the run.

## The question (and why it differs from everything we've killed)

Corners A & B and the late-fusion rebuild all asked the same shape of question — does text
predict the **level** of a market quantity (return, vol, yield)? All null, for two reasons:
the statement is public and instantly priced (efficient information), and the level is
persistence-dominated (HAR/realized measures own it).

This corner asks a **different, second-order** question that neither killer directly rules out:
does textual uncertainty predict **the magnitude of the forecast's own error** — i.e., *when*
the model should be less confident? Persistence gives you the expected vol level; it does not
tell you which days the forecast-error variance blows up. If "the Fed hedged / sounded
uncertain" flags days where the market reaction is genuinely less predictable, then the
certainty axis carries **calibration** information even though it carries no level information.
This is the one economically-plausible channel left.

## Hypotheses

- **H1:** the certainty signal `u = P(uncertain) − P(certain)` predicts the absolute OOS residual
  `|y − ŷ|` of the QLIKE-DLq RV forecast, incrementally over a baseline that already uses the
  forecast level and recent error size → enabling a better-calibrated (conditional) band.
- **H0:** `u` carries no information about residual magnitude (ΔMSE = 0) and the constant
  conformal band's coverage does not vary with `u`.

## Data and fixed point forecast

- SPX 5-min daily RV (`spx_5min_daily_rv.parquet`); horizons 1/5/22; the **point forecast is the
  unchanged QLIKE-DLq ensemble** (5 seeds × 5 folds, 300 epochs) — we are testing the *band*, not
  re-fitting the point model.
- `u`: the same leak-safe, as-of forward-filled certainty feature built for Corner A
  (`data/artifacts/corner_a_text_uncertainty/text_uncertainty_daily.parquet`).

## Models (paired, second stage on the pooled walk-forward OOS)

Target = `|residual_t| = |y_t − ŷ_t|` on the pooled OOS predictions. All predictors are known
as-of `t` (the forecast `ŷ_t`, the lagged error size, and `u_t` from statements ≤ t), so the
error-magnitude prediction is leak-free.

- **Baseline:** expanding-window OLS of `|residual|` on `[ŷ_t (forecast level), |residual|_{t-1}]`.
- **Treatment:** baseline `⊕ [u_t]`.
- Burn-in 100 pooled OOS days; expanding walk-forward; per horizon (1/5/22).

## Metric, significance, multiplicity

- **Primary:** OOS MSE of the `|residual|` prediction; `ΔMSE = MSE(base) − MSE(base+u)`;
  Diebold–Mariano on the squared-error differential (Newey–West HAC, lag = h).
- **Multiplicity:** 3 horizons → Bonferroni at family α = 0.10 → per-test **p < 0.0333**.
- **Secondary (corroborating, descriptive):** empirical coverage of the constant 90% conformal
  band within `u`-terciles — if `u` has calibration value, the constant band should **under-cover**
  on high-`u` days and **over-cover** on low-`u` days (a conditional-coverage gap).
- **Pre-registered hit:** DM `p < 0.0333` **and** `ΔMSE > 0` in **≥ 1** horizon, **and** the
  conditional-coverage gap runs in the predicted direction (high-`u` worse-covered) at that horizon.

## Decision rule

- **Hit →** textual uncertainty has genuine **calibration** value — a real, novel use of the
  certainty axis (band-widening on hedged statements). Report it; build the `u`-conditioned band.
- **No hit →** text is closed even for second-order/calibration use. Report and stop. No iterating
  on the residual transform (|r| vs r²), tercile cut, or feature.

## Honest ceiling

A hit is a *calibration* result — sharper risk bands on FOMC days, not a new point forecast or a
trading edge. Still a clean, publishable contribution and the first non-null text result in the
project if it lands.

## Artifacts

- `data/artifacts/corner_c_text_calibration/result.json`. Committed before it exists.

---

## Result (run 2026-06-01, 300 epochs × 5 seeds × 5 folds) — **technical hit at h1, but negligible**

| horizon | MSE\|resid\| base | +u | ΔMSE | DM p | band cov low-u | high-u | hit |
|---|---:|---:|---:|---:|---:|---:|---|
| **h1**  | 0.13937 | 0.13900 | **+0.000365** | **0.003** | 0.861 | 0.855 | **yes** |
| h5      | 0.08674 | 0.08673 | +0.00001 | 0.80 | 0.863 | 0.911 | no (coverage backwards) |
| h22     | 0.04116 | 0.04121 | −0.00006 | 0.10 | 0.903 | 0.919 | no |

**Verdict per the pre-registered rule: hit at h1** — DM p=0.003 (< Bonferroni 0.0333), ΔMSE>0,
and the constant 90% band under-covers slightly more on high-`u` days (0.855 vs 0.861), the
predicted direction. This is the **first formally-passing pre-registered text result** in the
whole project.

**But it is tiny and fragile, and I will not oversell it:**
- Effect size ≈ **0.26%** MSE reduction in predicting `|residual|`. Statistically real at
  n≈4400, economically negligible.
- **Only h1.** h5 is flat *and* its conditional-coverage gap runs the **wrong way** (high-`u`
  better covered); h22 is null. No consistency across horizons.
- The u-conditional coverage spread at h1 (0.861 vs 0.855) is trivial next to the band's overall
  ~86% vs nominal 90% under-coverage — `u` barely moves the calibration.

**Honest interpretation:** the *one* place text shows any incremental signal is exactly the
second-order channel we hypothesized — calibration, not level — which is theoretically tidy. But
the magnitude is too small and too horizon-fragile to build a production `u`-conditioned band on.
It refines, rather than overturns, the headline: text is essentially closed for forecasting; the
faintest residual signal lives in 1-day error-magnitude calibration, and even there it is a flicker.

**Decision:** report as-is (a hit by the pre-committed rule, flagged as negligible). Do **not**
p-hack a robustness search to inflate or kill it. Move to Corner D (the FX target) as planned.
