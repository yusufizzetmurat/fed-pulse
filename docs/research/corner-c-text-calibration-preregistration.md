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
