# Corner A — Pre-registration: does textual policy uncertainty add to RV forecasting?

**Status:** pre-registered 2026-06-01, *before any result was inspected*. This document
fixes the hypothesis, data, models, metric, significance rule, and decision rule. It is
committed prior to running so the test cannot be retro-fit to the outcome.

## Background and motivation

Across this project, FOMC text is a confirmed **null for directional/level forecasting**
(clean-room late-fusion rebuild: event n=167, daily n=1999, LoRA p=0.92, magnitude null;
drift study null in equities; intraday pivot full negative). Directional bets on a public,
instantly-priced statement are dead on arrival (efficient-market objection).

Two facts survive that objection and motivate this test:
1. Our **one** positive deep-learning result is on **second moments**: the QLIKE-DLq
   ensemble beats HAR on realized volatility (RV). From `backend/models/rv_qlike/production_eval.json`:
   h1 QLIKE 0.197 (ens) vs 0.223 (HAR), gain CI90 [0.019, 0.033]; n=5385 days, 5 seeds × 5 folds.
2. We now serve a **certainty axis** (textual hedging/uncertainty) on the multi-axis classifier.
   "The Fed sounded uncertain → wider realized vol" is a *volatility* claim, not a return claim —
   the kind a volatility-risk premium does not arbitrage away.

This corner tests **only** whether the certainty axis adds *incremental* RV-forecast skill on
top of the model that already wins. It does **not** re-test the dead equity-direction cell.

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
- **Primary comparison:** `ΔQLIKE_text = QLIKE(ens) − QLIKE(ens+text)` at each horizon —
  does text improve the *already-winning* ensemble. (Secondary: each vs HAR.)
- **Evaluation cells (pre-registered):**
  - (a) **full sample** — all OOS days.
  - (b) **post-FOMC window** — OOS days in `[d_i+1, d_i+5]` after each statement, where text
    should bite. (The feature is near-constant elsewhere, diluting a full-sample test.)

## Significance and multiplicity (fixed)

- Bootstrap CI on the paired QLIKE differential (the harness's `_bootstrap_qlike_gain_ci`,
  moving-block to respect serial dependence).
- **Multiplicity:** 3 horizons × 2 cells = **6 tests** → Bonferroni. Each CI computed at
  `1 − 0.10/6` (≈ CI98.3) instead of CI90.
- **Pre-registered hit:** the Bonferroni-corrected CI lower bound on `ΔQLIKE_text` is **> 0**
  (text strictly helps) in **≥ 1** cell, **and** the point estimate is **≥ 0** in all other cells.

## Decision rule (fixed)

- **Hit → Corner A positive →** proceed to Corner B (forward-guidance/time axis → 2Y/5Y curve).
- **No hit → null →** report the null honestly and **stop**. Do **not** iterate on the feature
  definition, horizons, window, or signal transform to manufacture significance.

## Honest ceiling (acknowledged in advance)

Even a clean hit means "a marginally better FOMC-window volatility forecast," useful for
risk/options sizing and as a thesis result — **not** a tradeable equity edge. FOMC is ~8
scheduled meetings/year; the signal is low-frequency by construction.

## Artifacts

- Result JSON: `data/artifacts/corner_a_text_uncertainty/result.json` (per-horizon, per-cell
  QLIKE for HAR / ens / ens+text, ΔQLIKE point + Bonferroni CI, seed dispersion, feature
  coverage/variation diagnostics).
- This pre-registration is committed before that file exists.
