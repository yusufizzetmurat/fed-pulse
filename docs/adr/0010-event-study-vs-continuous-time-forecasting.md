# ADR 0010 — Event-study vs continuous-time forecasting

Status: accepted, in production.
Date: 2026-05-16.
Supersedes: nothing.
Superseded by: nothing.
References:
- `backend/app/forecasting/cross_asset_response.py` — event-study cross-section regression head (#148).
- `backend/app/forecasting/next_fomc_decision.py` — event-study ordinal-class head (#156).
- `backend/app/services/forecaster.py` — the continuous-time LSTM that predates the event-study framing.
- `backend/app/evaluation/lstm_baseline_appendix.py` — the honest negative-result appendix (#151) that runs the LSTM checkpoint against the v2 holdout.
- `../../fed-pulse.wiki/06_Deep_Learning_Roadmap.md` — the wiki appendix that frames the methodology argument.

## Context

Two framings are possible for "does FOMC text help short-horizon market
forecasting?":

1. **Continuous-time forecasting.** The model sees one row per trading
   day, threads a text-derived scalar into a 6-feature sequence
   window, and predicts close-price + volatility at t+1..t+10 trading
   days. The legacy LSTM forecaster (`services/forecaster.py`) is the
   continuous-time framing. Denominator: every trading day in the
   reporting window — roughly 6,500 rows for a 25-year v2 holdout.
2. **Event-study supervised learning.** The model sees one row per
   FOMC event (statement / minutes / speech / press conference / etc.),
   with the row's features and target conditioned on that event's
   announcement timestamp. The cross-asset response head
   (`forecasting/cross_asset_response.py`) and the next-FOMC decision
   classifier (`forecasting/next_fomc_decision.py`) are the event-study
   framing. Denominator: ~150 FOMC events in the same window.

The thesis question is whether fusing FOMC text with market history
improves short-horizon forecasts. The two framings have wildly
different signal-to-noise ratios for that question.

## Decision

**Treat event-study supervised learning as the primary methodology for
the headline reporting pack. Keep the continuous-time LSTM as an
honest baseline reported alongside; ship the LSTM appendix with the
two reference baselines (random-walk close, mean-reversion volatility)
so a reader can see exactly how much signal the LSTM extracts from the
text channel under the continuous-time framing.**

The argument:

- On the continuous-time framing, ~97% of the rows are non-event bars.
  The text-derived scalar for a non-event row is whatever the previous
  FOMC communication produced, which means the same scalar is repeated
  for ~15-20 days at a time. The LSTM is then trying to learn a
  market-dynamics signal from a near-constant feature; the gradient
  it can extract from text on those rows is structurally small.
- The event-study framing concentrates each text observation on the
  one row that matters: the day the policy stance is announced. Same
  text, same encoder, but the target is conditioned on the
  announcement so the gradient lands where the signal is.
- Macro-F1 / RMSE on the continuous-time framing have a denominator
  that includes the non-event rows; macro-F1 / RMSE on the event-study
  framing have a denominator of events. A 1% improvement under one
  framing is not commensurable with a 1% improvement under the other.

## Consequences

- The headline tables in the reporting pack are produced by the
  event-study heads. The dashboard's `/decisions` view reads
  ``data/artifacts/next_fomc/`` (event-study ordinal logit) and
  ``/analyze`` blends the event-study cross-asset response head with
  the legacy LSTM checkpoint for the close + volatility forecast.
- The LSTM continuous-time baseline appendix
  (``backend/app/evaluation/lstm_baseline_appendix.py``) ships in the
  reporting pack as an appendix, not a headline. The reader sees the
  LSTM checkpoint's RMSE / MAPE / directional accuracy against random-
  walk and mean-reversion baselines on the same v2 holdout, with
  block-bootstrap CIs.
- A continuous-time LSTM that fails to beat random-walk on close on
  most asset/horizon cells is consistent with the methodology
  argument above. The appendix surfaces that honestly rather than
  hiding it inside an aggregate macro-F1 that averages event days
  with non-event days.
- Future work: a same-day market-impact head (one row per intraday
  bar inside the announcement window) is the natural extension of the
  event-study framing. Deferred to a follow-up; tracked in the wiki
  appendix.
