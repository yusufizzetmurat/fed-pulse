# Per-asset HAR-tercile baseline on ^NDX and ^DJI

## What this arm settles

The serving wrapper at `backend/app/services/har_tercile.py` is hardcoded to
SPX (`^GSPC`) and silently routes any `^NDX` / `^DJI` request through SPX
volatility regime probabilities. This run produces the per-asset macro-F1
numbers the routing fix needs, on the same evaluation protocol the SPX
baseline `0.687 / 0.685 / 0.654` was measured under.

## Protocol

- Realized variance proxy: daily `log_return^2`. yfinance carries only daily
  closes for `^NDX` and `^DJI`, so this is the only RV estimator available.
  The canonical SPX baseline uses 5-min intraday bars, which is the
  apples-to-oranges issue flagged below.
- HAR lags: `_har_lags(log(rv + EPS))` -> `[log_rv_t, mean last-5, mean last-22]`.
  Same construction as `app.data.intraday_rv_forecast._har_lags`.
- Forward target: `log(mean(rv[t+1..t+h]))`. Same as
  `app.data.fed_comms_dataset._forward_target` with `is_log=True`.
- Folds: dense `walk_forward_splits(n_valid, n_folds=5, embargo=23)` on the
  index of finite-target rows. Matches `app.data.fed_comms_regime.run`.
- Tercile cutoffs: `q33`, `q67` of the forward target on the **train slice**
  only, per fold. No test leakage.
- OLS HAR is deterministic. The official seed set `{11, 29, 47, 71, 97}`
  yields identical numbers. `n_seeds = 1` is reported.

## Headline numbers (pooled macro-F1 across 5 folds)

| Asset | h=1 | h=5 | h=22 |
| --- | ---: | ---: | ---: |
| ^GSPC (5-min RV, wiki baseline) | 0.687 | 0.685 | 0.654 |
| ^GSPC (daily-r^2 comparator) | 0.298 | 0.498 | 0.550 |
| ^NDX (daily-r^2) | 0.353 | 0.580 | 0.613 |
| ^DJI (daily-r^2) | 0.317 | 0.525 | 0.586 |

Per-fold mean +/- std (the spec asks for both):

| Asset | h=1 | h=5 | h=22 |
| --- | ---: | ---: | ---: |
| ^NDX | 0.296 +/- 0.053 | 0.500 +/- 0.068 | 0.514 +/- 0.064 |
| ^DJI | 0.294 +/- 0.096 | 0.463 +/- 0.087 | 0.489 +/- 0.085 |
| ^GSPC (daily) | 0.282 +/- 0.042 | 0.472 +/- 0.067 | 0.523 +/- 0.069 |

## What the numbers say

NDX and DJI both underperform the wiki SPX baseline by 5-37 pp depending on
horizon, but this is dominated by the RV-proxy difference, not the asset
difference. Once SPX is re-measured under the same daily-r^2 proxy, the
ranking flips:

| Comparison (pooled F1, deltas in pp) | h=1 | h=5 | h=22 |
| --- | ---: | ---: | ---: |
| NDX - SPX-daily | +5.5 | +8.3 | +6.3 |
| DJI - SPX-daily | +1.8 | +2.7 | +3.6 |

Under a like-for-like comparator NDX > DJI > SPX-daily on every horizon,
consistent with tech-sector vol mean reversion being easier to forecast than
broad-market vol mean reversion. The h=1 macro-F1 on the daily-r^2 proxy
floors near 0.30 (vs the 0.33 majority-class floor) because daily squared
returns are too noisy to be partitioned into stable terciles 1 step ahead;
the proxy noise dominates the HAR signal. By h=22 the averaging in the
forward target smooths it out and macro-F1 climbs to ~0.55-0.61.

## KS diagnostic (train vs test forward target distribution)

| Asset | h=1 mean KS | h=5 mean KS | h=22 mean KS |
| --- | ---: | ---: | ---: |
| ^NDX | 0.128 | 0.215 | 0.264 |
| ^DJI | 0.099 | 0.183 | 0.247 |

Per-fold KS stats are in the result JSON. Fold 1 (early NDX/DJI coverage
versus later vol regimes) has the largest train/test distribution gap as
expected; the q33/q67 cutoff fit on a 1985-1992 train slice does not match
the 1992-1999 test distribution, dragging h=1 macro-F1 down to 0.20 on NDX
fold 1 and 0.13 on DJI fold 1. This is the noisy-cutoff failure mode the
arm spec flagged.

## Assessment

This arm is HAR-tercile re-fitted on a different asset, so it does not "beat"
HAR-tercile; it is HAR-tercile, just with a different daily series and a
daily-r^2 proxy instead of 5-min intraday RV. The shipped result is:

1. The numerical answer to "what is HAR-tercile macro-F1 on NDX and DJI" on
   the canonical protocol. NDX clears SPX-daily by 5-8 pp; DJI by 2-4 pp.
   The paper's per-asset variation claim of 5-12 pp on equities (SPY/QQQ/IWM)
   replicates here on the NDX side.
2. The wiki SPX number 0.687 is not a like-for-like benchmark for NDX/DJI
   because it depends on intraday-RV which yfinance does not carry for those
   tickers. The serving wrapper that today routes any non-SPX request through
   the SPX 0.687 baseline overstates regime-classifier confidence on those
   tickers; the routing fix should expose the daily-r^2 NDX/DJI numbers, not
   the SPX intraday number, when serving NDX or DJI requests.

## Out of scope

Patching `app/services/har_tercile.py` to accept a `symbol` arg and load the
per-asset HAR coefficients from a new artifact slot. The arm spec notes the
patch but flags it out of scope for the read-only plan phase.

## Artifacts

- Script: `scripts/fit_per_asset_har_tercile.py`
- Per-fold JSON: `data/artifacts/har_tercile_per_asset_har/result.json`
- Synthesize-readable copy: `/tmp/fed-pulse/har-tercile-arms/per_asset_har.json`
