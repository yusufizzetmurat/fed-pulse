# Pre-registration — powered confirmation of the market→Fed *directional* lead

**Date:** 2026-06-02 · **Status:** COMPLETE — NULL. The directional lead did not survive powering; reverse direction closed.

## Result (2026-06-02)

Usable meetings: **117** (2010-03-16 → 2026-04-29, OOS = 87) — ~3× the discovery's 36.

| test | discovery (n=36) | powered (n=86) | bar | verdict |
|---|---|---|---|---|
| **PRIMARY** directional acc `sign(predΔs)==sign(Δs)` | 0.667 | **0.477** (binom p 0.705) | p<0.05 & >0.5 | **fails (below chance)** |
| **Replication** pre-2016 OOS slice (fresh data) | — | **0.364** (n=11) | ≥ 0.55 | **fails** |
| Secondary Δs MSE: M1 vs mean-drift | — | 0.4355 vs 0.3615 (DM p 0.989) | M1 lower | **M1 worse** |

Verdict: null. On the powered sample directional accuracy falls below chance (0.477), the independent pre-2016 slice lands at 0.364, and the market-feature model predicts the stance shift worse than a constant drift. Both decision-rule conditions fail. The discovery's 0.667 (36 meetings, 2019+) does not generalise.

This closes the reverse direction. FOMC text ↔ markets is null in both directions: text does not predict markets (corners A–E, intraday, late-fusion), and markets do not predict the statement's stance/shift beyond the Fed's own inertia. The pre-registration plus independent-slice replication caught the false positive, mirroring Corner C→E. The one surviving forecasting edge remains the QLIKE-DLq RV ensemble (no text). Artifact: `data/artifacts/reverse_directional_followup/result.json`. See `reverse-market-predicts-fed-preregistration.md` (directional acc 0.667, p=0.033, secondary, n=36 OOS, 2016+ only — failed Bonferroni, flagged as underpowered).

## What changed and why

The first reverse test was starved: the `pre_meeting_*` rate features in `events.parquet` are dense only from 2016, so n collapsed to 76 (36 OOS). The one pulse was the direction of the stance shift. This follow-up (a) rebuilds the rate-expectation features from raw FRED/market data back to 2010, roughly doubling the sample, and (b) promotes the directional channel to the primary test (it was secondary in discovery), so it is no longer multiplicity-penalised. It must now clear an independent-slice replication, not just a powered p.

## Frame and target

One row per scheduled FOMC statement meeting, **2010–2026** (`events.parquet` `event_kind=="statement"`, deduped on `event_date`, intermeeting actions 2020-03-03/2020-03-15 dropped). Target is the stance shift `Δs_t = s_t − s_{t-1}`, where `s = P(hawk) − P(dove)` per statement date from `stance_daily.parquet`. The directional question: does pre-meeting market state predict the sign of the Fed's tone shift?

## Features (rebuilt back to 2010, all strictly pre-meeting; fixed now)

Rate-expectation block — from FRED `DGS1`, `DGS2`, `DFEDTARU` and cache `TNX` (10y), as of the last observation strictly before the meeting date:
1. `implied_next_move_bps` = (DGS1 − DFEDTARU)·100
2. `slope_10y_2y` = TNX(%) − DGS2
3. `trailing_2y_change_5d_bps` = (DGS2[t-1] − DGS2[t-6])·100
4. `yield_2y` = DGS2[t-1]
5. `days_since_last_rate_change` = days since DFEDTARU last moved

Financial-conditions block — cache `GSPC`, `VIX`, last close before the meeting:
6. SPX trailing 22-day log return · 7. VIX level · 8. VIX trailing 22-day log change

Leakage exclusions (unchanged): no `mp_surprise_*`, `ff_target_after`, SEP/dot-plot, or any at/after-announcement field. Model M1 = Ridge(Δs ~ the 8 features); standardize and choose α on the train slice only.

## Protocol, primary test, replication

- Walk-forward expanding, one meeting ahead, initial train = 30 meetings → OOS ≈ 105 (≈3× the discovery's 36).
- Primary: directional accuracy `sign(predΔs) == sign(Δs)` on the full OOS, one-sided binomial vs 0.5.
- Independent replication slice: directional accuracy on the pre-2016 OOS meetings only — data the discovery sample never contained. This is the clean out-of-sample check that separates a real reaction-function signal from a 2016–2026-specific artifact.
- Secondary (supportive, not decisive): Δs MSE of M1 vs a mean-drift baseline (Diebold–Mariano, Newey–West); and stance-level MSE vs persistence (the discovery's original primary), for continuity.

## Decision rule (committed now)

Confirm the directional reaction-function signal iff both:
1. full-OOS directional binomial p < 0.05 (one-sided, acc > 0.5), and
2. the pre-2016 replication slice also shows accuracy ≥ 0.55 (same direction, not a 2016+-only effect).

If the full-OOS test is not significant, the discovery lead was noise; report the null and close the reverse-direction question. If full-OOS is significant but the pre-2016 slice is < 0.55, report as regime-specific, not replicated (no general claim). No post-hoc feature/α/window tuning on this data; this is the one committed confirmation run.
