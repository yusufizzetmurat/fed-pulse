# Corner D — Pre-registration: does FOMC text help forecast FX (dollar) volatility?

**Status:** pre-registered 2026-06-01, *before any result was inspected*. Committed before the run.

## Why this corner — closing the original project's loop

The project was *proposed* as a Central-Bank-Sentiment → **FX volatility** predictor, but it
drifted to SPX equities + rates and the FX target was **never actually tested**. Verified: the
dollar index appears only as a side correlation feature (`corr_dxy`), never as a forecast target.
So FX is **open, not null** — and closing it honestly is worth one disciplined shot, especially
because the dollar is arguably the **most Fed-sensitive** asset (the purest expression of relative
monetary policy). If FOMC text were ever going to help anywhere, FX volatility is a fair place.

## Hypotheses

- **H1:** FOMC text (stance `s = P(hawk) − P(dove)` and certainty `u = P(unc) − P(cert)`) improves
  the forecast of dollar-index realized volatility incrementally over a HAR persistence baseline.
- **H0:** the text features add nothing (ΔQLIKE = 0).

## Data

- **Target:** US Dollar Index (DXY, `DX-Y.NYB`) daily close, 1975→2026 (12,949 days). No intraday
  is available, so the realized-variance proxy is the **daily squared log return** `RV_t = r_t²`
  (the standard daily-only proxy — noisier than 5-min RV; acknowledged limitation). Forward target
  `y_t = log(mean(RV_{t+1..t+h}))` for h ∈ {1, 5, 22}.
- **Text:** stance `s` and certainty `u` from the served multi-axis classifier on FOMC statements,
  as-of forward-filled onto the DXY calendar (leak-safe: only statements with date ≤ t).

## Models (paired, walk-forward)

- **Baseline:** QLIKE-DLq ensemble (5 seeds × 5 folds, 300 epochs) on HAR lags
  `[logRV_t-1, mean-5, mean-22]` — same machinery that beats HAR on SPX RV.
- **Treatment:** baseline `⊕ [u, s]` (two text columns).
- Walk-forward, time-ordered, embargo = h+1; HAR + ensemble + standardizer fit train-fold only.

## Metric, significance, multiplicity

- **Metric:** out-of-sample QLIKE; `ΔQLIKE = QLIKE(base) − QLIKE(base+text)`.
- **Test:** moving-block bootstrap CI of the paired QLIKE gain (block = h).
- **Multiplicity:** 3 horizons → Bonferroni at family α = 0.10 → CI at 1−0.10/3 (≈ 96.7%,
  quantiles [0.0167, 0.9833]).
- **Pre-registered hit:** Bonferroni CI lower bound on `ΔQLIKE` **> 0** in **≥ 1** horizon.

## Decision rule

- **Hit →** FX behaves differently from equities — FOMC text helps dollar-vol forecasting; the
  original project's target reopens as a genuine result.
- **No hit →** FX confirms the SPX null on its own (and original) ground; text is closed on the
  project's founding target too. Report and stop; no iterating on the RV proxy or feature set.

## Honest ceiling & caveat

`RV = r²` is a noisy 1-observation proxy (DXY has no intraday/high-low on disk), so power is lower
than the SPX test; a clean null is still informative at n≈12.9k days, and a hit would warrant an
intraday-FX follow-up before any claim.

## Artifacts

- `data/artifacts/corner_d_fx_volatility/result.json`. Committed before it exists.

---

## Result (run 2026-06-01, 300 epochs × 5 seeds × 5 folds, n=12,948 days) — **NULL**

| horizon | QLIKE base | QLIKE +text | ΔQLIKE | Bonferroni CI | hit |
|---|---:|---:|---:|---|---|
| h1  | 1.5500 | 2.1019 | −0.5520 | [−0.640, −0.465] | no |
| h5  | 0.3718 | 0.3878 | −0.0160 | [−0.024, −0.009] | no |
| h22 | 0.1680 | 0.1754 | −0.0073 | [−0.016, +0.001] | no |

**0 / 3 hits.** Adding FOMC text (stance + certainty) makes the dollar-vol forecast **worse** at
every horizon — Bonferroni CIs sit at or below zero. **Verdict: NULL.**

**Reading it honestly:** the `r²` daily proxy is noisy (note the high base QLIKE at h1), so the DL
ensemble already strains on the target and the two extra text columns mostly inject noise — most
visibly at h1. But the direction is unambiguous across all horizons: text does not help, it hurts.
FX — the project's *original* and most Fed-sensitive target — behaves exactly like SPX. Caveat:
intraday FX (cleaner RV) isn't on disk; a clean null at n≈12.9k days is informative, and a positive
result would have been the one to chase with intraday data — but there is no positive result to chase.

**Decision:** null → the original FX target is closed too. Text is confirmed null for forecasting
on the very question the project was founded on.
