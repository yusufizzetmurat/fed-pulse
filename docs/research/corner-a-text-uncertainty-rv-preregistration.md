# Corner A — Pre-registration: does textual policy uncertainty add to RV forecasting?

**Status:** pre-registered 2026-06-01, *before any result was inspected*. This document
fixes the hypothesis, data, models, metric, significance rule, and decision rule. It is
committed prior to running so the test cannot be retro-fit to the outcome.

## Background and motivation

FOMC text is a confirmed **null for directional/level forecasting** across this project
(clean-room late-fusion rebuild: event n=167, daily n=1999, LoRA p=0.92, magnitude null;
drift study null in equities; intraday pivot full negative). Directional bets on a public,
instantly-priced statement run into the efficient-market objection.

Two facts survive that objection and motivate this test:
1. The single positive deep-learning result is on **second moments**: the QLIKE-DLq
   ensemble beats HAR on realized volatility (RV). From `backend/models/rv_qlike/production_eval.json`:
   h1 QLIKE 0.197 (ens) vs 0.223 (HAR), gain CI90 [0.019, 0.033]; n=5385 days, 5 seeds × 5 folds.
2. The served multi-axis classifier exposes a **certainty axis** (textual hedging/uncertainty).
   "Fed sounded uncertain → wider realized vol" is a *volatility* claim, not a return claim,
   and a volatility-risk premium does not arbitrage it away.

This corner tests **only** whether the certainty axis adds *incremental* RV-forecast skill on
top of the model that already wins. It does **not** re-test the null equity-direction cell.

## Hypotheses

- **H1 (alternative):** adding a leak-safe textual-uncertainty feature to the QLIKE-DLq
  ensemble strictly lowers out-of-sample QLIKE relative to the same ensemble without it.
- **H0 (null):** ΔQLIKE = 0. The text feature adds nothing.

## Data

- **Target / market features:** `data/external/alphavantage_bars/spx_5min_daily_rv.parquet`
  — SPX 5-min daily RV, ~5385 trading days, columns `date, rv, rvol, rs_pos, rs_neg, bv, rq,
  rskew, rkurt, parkinson` (the exact bake-off feature set).
- **Text feature:** certainty-axis output of the served multi-axis classifier
  (`app.services.multi_axis_classifier.score_text`), scored on each FOMC **statement** in the
  event corpus. Signal = `P(uncertain) − P(certain)` (continuous, in [−1, 1]); higher = more
  hedged/uncertain language.

### Feature construction (fixed, leak-safe)

1. Score every FOMC statement; attach its **release date** `d_i` and signal `u_i`.
2. For each market day `t`, `text_uncertainty[t] = u_i` for the most recent statement with
   `d_i ≤ t` (strict as-of; a statement released on day `t` is allowed since RV[t] is realized
   *after* the announcement). No look-ahead.
3. Forward-fill between meetings (the most recent signal persists). Before the first scored
   statement, the feature is 0.
4. **No** post-hoc transforms beyond the fixed `P(uncertain) − P(certain)` and the harness's
   train-fold standardizer. The feature definition is frozen by this document.

## Models (paired)

- **Baseline:** QLIKE-DLq ensemble on `full = [HAR_d, HAR_w, HAR_m, rs_pos, rs_neg, bv, rq,
  rskew, rkurt, parkinson, log_rvol]` — the validated beat-HAR model.
- **Treatment:** identical ensemble on `full ⊕ [text_uncertainty]` (one extra column).
- **Pairing:** same walk-forward splits, same seeds (11,22,33,44,55), same epochs (300),
  same horizons (1, 5, 22). The only difference is the single text column. HAR is reported as
  the floor in both.

## Protocol

- Walk-forward OOS via `intraday_rv_production.run` machinery (`walk_forward_splits`, 5 folds,
  time-ordered). HAR coefficients, the DL ensemble, and the feature standardizer are fit on the
  **train fold only**; the text feature uses only statements with `d_i ≤ t`. No leakage.

## Primary metric and comparison

- **Metric:** out-of-sample QLIKE (lower is better).
- **Primary comparison:** `ΔQLIKE_text = QLIKE(ens) − QLIKE(ens+text)` at each horizon, asking
  whether text improves the *already-winning* ensemble. (Secondary: each vs HAR.)
- **Evaluation cells (pre-registered):**
  - (a) **full sample** — all OOS days.
  - (b) **post-FOMC window** — OOS days in `[d_i+1, d_i+5]` after each statement, where text
    should bite. The feature is near-constant elsewhere, which dilutes a full-sample test.

## Significance and multiplicity (fixed)

- Bootstrap CI on the paired QLIKE differential (the harness's `_bootstrap_qlike_gain_ci`,
  moving-block to respect serial dependence).
- **Multiplicity:** 3 horizons × 2 cells = **6 tests** → Bonferroni. Each CI computed at
  `1 − 0.10/6` (≈ CI98.3) instead of CI90.
- **Pre-registered hit:** the Bonferroni-corrected CI lower bound on `ΔQLIKE_text` is **> 0**
  (text strictly helps) in **≥ 1** cell, **and** the point estimate is **≥ 0** in all other cells.

## Decision rule (fixed)

- **Hit → Corner A positive →** proceed to Corner B (forward-guidance/time axis → 2Y/5Y curve).
- **No hit → null →** report the null and **stop**. Do not iterate on the feature
  definition, horizons, window, or signal transform to manufacture significance.

## Ceiling (acknowledged in advance)

Even a clean hit means a marginally better FOMC-window volatility forecast, useful for
risk/options sizing and as a research result, not a tradeable equity edge. FOMC is ~8
scheduled meetings/year; the signal is low-frequency by construction.

## Artifacts

- Result JSON: `data/artifacts/corner_a_text_uncertainty/result.json` (per-horizon, per-cell
  QLIKE for HAR / ens / ens+text, ΔQLIKE point + Bonferroni CI, seed dispersion, feature
  coverage/variation diagnostics).
- This pre-registration is committed before that file exists.

---

## Result (run 2026-06-01, 300 epochs × 5 seeds × 5 folds, GPU) — **NULL**

**Harness validated:** the baseline ensemble reproduces the known beat-HAR result —
QLIKE(ens) < QLIKE(HAR) at every horizon (h1 0.197 vs 0.223; h5 0.198 vs 0.219;
h22 0.327 vs 0.360). The pipeline is correct; the null below is about the *text* increment.

**Incremental text effect ΔQLIKE = QLIKE(ens) − QLIKE(ens+text):**

| cell | QLIKE_HAR | QLIKE_ens | ens+text | ΔQLIKE | Bonferroni CI | hit |
|------|----------:|----------:|---------:|-------:|---------------|-----|
| h1 / full       | 0.2229 | 0.1974 | 0.1984 | −0.0010 | [−0.0027, 0.0006] | no |
| h1 / post-FOMC  | 0.1926 | 0.1905 | 0.1901 | +0.0004 | [−0.0037, 0.0042] | no |
| h5 / full       | 0.2194 | 0.1975 | 0.1977 | −0.0002 | [−0.0033, 0.0027] | no |
| h5 / post-FOMC  | 0.2374 | 0.2155 | 0.2203 | −0.0048 | [−0.0184, 0.0034] | no |
| h22 / full      | 0.3597 | 0.3268 | 0.3359 | −0.0092 | [−0.0308, 0.0071] | no |
| h22 / post-FOMC | 0.3102 | 0.2938 | 0.2928 | +0.0011 | [−0.0222, 0.0214] | no |

0 / 6 cells hit. Every CI straddles zero; point estimates are small and mostly negative
(text slightly hurts at h5/h22 full). Verdict: null.

**Why (the pre-flagged confound won):** the certainty signal varies (std 0.51) and spikes
in crises, but textual uncertainty is collinear with the high-vol regime HAR already reads
through its own lags + the realized measures (rs±, bv, rq, parkinson). Once the model knows
realized vol is elevated, "the statement sounded uncertain" carries no *incremental*
information. The same efficient-information story behind the directional nulls now
extends to the second moment.

**Decision (per the pre-registered rule):** null → stop. Corner B is not run by
default (the rule was "Corner B only if A shows anything"). Corner B (forward-guidance/time
axis → 2Y/5Y curve) remains the only economically-motivated avenue left, and would require
a fresh, separately-pre-registered test.
