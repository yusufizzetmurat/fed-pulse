# Pre-registration — reverse direction: does the market predict the Fed?

**Date:** 2026-06-02 · **Status:** COMPLETE — null (primary); suggestive-but-not-significant directional lead. See Result.

## Result (2026-06-02)

Usable meetings after requiring complete pre-meeting features: **76** (2016-03-16
→ 2026-04-29, OOS = 36) — far below the ~130 estimate, because the
`pre_meeting_*` rate-expectation columns in `events.parquet` are only dense from
2016. Underpowered; read accordingly.

| test | M0 (persistence) | M1 (market) | statistic | verdict |
|---|---:|---:|---|---|
| **Primary** OOS MSE of stance `s` | 0.3327 | 0.3148 | incr-R² **+0.054**, CI90 **[−0.147, +0.237]**; DM p **0.335** | fails (CI ∋ 0, DM n.s.) |
| Secondary directional `sign(Δs)` | — | — | dir-acc **0.667**, binom p **0.033** | fails Bonferroni (0.05/3 = 0.0167) |

**Verdict: NULL by the decision rule.** The primary test fails both conditions —
market state does not lower stance MSE over persistence at a significant level,
and the incremental-R² CI includes 0. The directional secondary (the model calls
the *direction* of the hawkish/dovish shift 67% of the time) is suggestive that
markets weakly anticipate the Fed's turn, but it is a secondary test, does not
survive multiplicity, sits on only 36 OOS meetings, and is contradicted by the
flat primary. **It is a lead, not a result; not chased on this data per the
pre-registered rule.**

**Pre-registerable follow-up (NOT done here, to avoid post-hoc tuning):** rebuild
the pre-meeting rate-expectation features from FRED Treasury yields back to 2010
(roughly doubling n), re-pre-register, and test the directional channel as the
*primary* with proper power. The certainty-axis secondary was deferred:
`stance_daily.parquet` carries only `s`, so it needs re-scoring first.

Artifact: `data/artifacts/reverse_market_predicts_fed/result.json`.

## Motivation

The whole project established **text → market is null** (corners A–E, intraday
pivot, late-fusion rebuild). This tests the **reverse**: does pre-meeting market
state predict the *content* of the upcoming FOMC statement? There is a real
economic reason to expect signal here that does not exist in the forward
direction — the Fed has a **reaction function**: it responds to the economy and
financial conditions, and markets reprice rate expectations *before* the Fed
confirms them. A positive result would be a clean mirror of the forward null
(markets anticipate the Fed; the Fed does not move markets with text). A null is
also informative (the statement's tone is not anticipable from market state
beyond inertia).

## Target (text content)

Per-meeting **stance score** `s = P(hawkish) − P(dovish)` from the multi-axis
classifier, read on each FOMC statement date from
`data/artifacts/corner_b_text_rates/stance_daily.parquet` (`date`, `s`).
One row per **scheduled** FOMC meeting, 2010–2026 (calendar
`data/external/fomc_meetings_2010_2026.csv`, intermeeting/emergency actions
dropped) — n ≈ 130 after requiring complete features.

## Baseline = the Fed's own inertia

Stance is highly autocorrelated (hawkish/dovish regimes persist for years), so
the **only honest baseline is persistence**, not the unconditional mean. The
question is strictly whether market state adds information **beyond last
meeting's stance**:

- **M0 (baseline):** OLS `s_t ~ s_{t-1}` (fitted persistence + mean-reversion).
- **M1 (market):** Ridge `s_t ~ s_{t-1} + [pre-meeting market block]`.

M1 must beat M0 — recovering persistence is not a result.

## Pre-meeting feature block (fixed now, all strictly before the meeting)

Rate-expectation block — from `events.parquet` (`event_kind=="statement"`,
deduped to one row per `event_date`; all are strict-backward t-1 by construction
per `rates_event_features.py`):

1. `pre_meeting_implied_next_move_bps` — market-implied next policy move
2. `pre_meeting_slope_10y_2y` — yield-curve slope
3. `pre_meeting_trailing_2y_yield_change_5d_bps` — recent short-rate repricing
4. `pre_meeting_yield_2y` — short-rate level
5. `pre_meeting_days_since_last_rate_change` — policy inertia

Financial-conditions block — assembled from the market cache (`GSPC`, `VIX`),
using only closes on the **last trading day strictly before** the meeting date:

6. SPX trailing 22-trading-day log return
7. VIX level (last close before the meeting)
8. VIX trailing 22-day log change

**Leakage exclusions (committed):** `mp_surprise_level/path/info`,
`surprise_*` (daily_fusion), `ff_target_after`, SEP/dot-plot fields, and every
post-announcement reaction/forward column are **banned** — they incorporate the
decision itself (`mp_surprise.py:1135`). Only the eight features above + `s_{t-1}`.

## Protocol, metric, tests

- **Walk-forward expanding**, one meeting ahead: initial train = first 40
  meetings; predict each subsequent meeting t from a model fit on meetings < t.
  Features standardized on the train slice only; Ridge α chosen on the train
  slice (fixed grid, inner split) — no test-set tuning. Meetings are 6–8 weeks
  apart, so no embargo beyond the strict-prior feature construction is needed.
- **Primary metric:** OOS MSE of `s_t`. **Primary test:** one-sided
  Diebold–Mariano (Newey–West HAC) on per-meeting squared-error differences
  M0 − M1; plus OOS incremental R² of M1 over M0 with a moving-block bootstrap CI.
- **Secondary (Bonferroni ×3):** (a) directional accuracy of `sign(s_t − s_{t-1})`
  from M1's predicted change vs a 50% coin (binomial); (b) same pipeline on the
  **certainty** axis `c = P(certain) − P(uncertain)`; (c) M1 vs unconditional-mean
  (context only).

## Decision rule (committed now)

**Conclude "markets predict Fed stance beyond inertia"** iff **both**:
1. primary DM p < 0.05 (one-sided, M1 beats M0), **and**
2. OOS incremental-R²(M1 over M0) bootstrap 90% CI strictly > 0.

If positive, **replicate before any claim**: re-run on (a) the certainty axis and
(b) a held-out post-2019 block fit only on pre-2019 meetings — same discipline
that caught the Corner-C false positive. Otherwise **report the null**: the
statement's tone is not anticipable from pre-meeting market state beyond the
Fed's own persistence. No post-hoc feature/α tuning on the same data.
