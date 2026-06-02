# Pre-registration — heterogeneous ensemble to push the QLIKE-RV edge

**Date:** 2026-06-02 · **Status:** pre-registered before looking at results.

## Motivation

The one genuine forecasting edge in fed-pulse is the QLIKE-trained DL ensemble
that beats HAR on realized volatility (`production_eval.json`: QLIKE 0.1973 /
0.1973 / 0.3266 vs HAR 0.2229 / 0.2194 / 0.3597 at h1/h5/h22; all 90% bootstrap
gain-CIs > 0). The production ensemble is **5 seeds of the *same* MLP** — it has
zero architectural diversity, and h22 is its weakest, highest-variance horizon.

A leverage-LSTM (σ-LSTM, `intraday_rv_arch.py`) is already implemented, is
architecturally complementary (processes an L=22 sequence with a leverage gate vs
the MLP's flat 11-feature row), and independently beats HAR at h1/h5 — but was
only ever run at 1 seed / 120 epochs and never combined with the MLP. Adding
diverse members is the standard way to lower ensemble error when members'
mistakes are partially independent.

## Hypothesis

**H1:** A heterogeneous ensemble (production MLP seeds + σ-LSTM seeds) achieves
lower pooled walk-forward QLIKE than the current MLP-only ensemble, at one or
more horizons.

## Design (fixed in advance)

Identical data + protocol to the canonical eval: `spx_5min_daily_rv.parquet`
(5,385 days), 5-fold expanding walk-forward, embargo `h+1`, QLIKE on σ²; HAR-
stacked residual learning; both model families QLIKE-trained at **300 epochs**.
To make the MLP and the σ-LSTM directly averageable, both are scored on the
**same day-set** — the σ-LSTM's scorable origins (days with a full L=22 lookback
and an in-range forward target); the first 21 days the MLP could score alone are
dropped (negligible vs n≈5,360). HAR is re-fit per fold on that day-set.

Members trained per fold:
- **MLP-QLIKE** seeds: 11, 22, 33, 44, 55, 66, 77, 88 (8 seeds).
- **σ-LSTM-QLIKE** seeds: 11, 22, 33 (3 seeds).

Ensembles (mean of member predictions **in log-RV space**, then exp for QLIKE):
- **A — MLP-5** (reproduces production): MLP{11,22,33,44,55}.
- **B — MLP-8** (count-matched control): MLP{11,22,33,44,55,66,77,88}.
- **C — Hetero-8** (treatment): MLP{11,22,33,44,55} + σLSTM{11,22,33} (8 members).

A vs B vs C is the clean factorial: A→C adds 3 diverse members; B→C swaps 3 MLP
members for 3 σ-LSTM members at fixed count, isolating *diversity* from *count*.

## Metric, tests, multiplicity

- **Primary metric:** pooled walk-forward QLIKE per horizon (lower better).
- **Primary test:** moving-block bootstrap (block=h, 1000 resamples) CI of the
  QLIKE gain **C over A** (`qlike(A) − qlike(C)`); > 0 ⇒ hetero wins.
- **Secondary test:** bootstrap gain CI **C over B** (diversity beyond count).
- **Multiplicity:** 3 horizons. Bonferroni — report the **96.7% CI**
  (α = 0.10/3) for the primary test; a horizon "wins" only if that corrected CI
  excludes 0.

## Decision rule (committed now)

**Promote the heterogeneous ensemble** iff, at ≥1 horizon, **all** hold:
1. C beats A — Bonferroni-corrected (96.7%) primary CI strictly > 0;
2. the gain is not merely a member-count effect — C-over-B 90% CI ≥ 0 (lower
   bound not below 0) at that horizon;
3. C does **not** significantly worsen any horizon (no Bonferroni CI strictly < 0
   for C-over-A anywhere).

Otherwise: **report the null**, keep the MLP-5 production ensemble, and record
that architectural diversity did not lower QLIKE on this series. No tuning of
seeds/epochs/members on the same data afterward — any positive result will be
re-run on an independent split (DXY or a held-out post-2020 block) before any
promotion, mirroring the Corner-A..E discipline.
