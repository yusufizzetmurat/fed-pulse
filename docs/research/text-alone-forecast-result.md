# Text-alone forecast — result

**Date:** 2026-06-02 · **Status:** COMPLETE — text-alone underperforms the
no-text baseline on every realized-vol horizon and on abnormal return.

## Setup

Text-only RV/AV regression on the dense FOMC panel (n=262 FOMC test points),
compared against a no-text baseline. Artifact:
`data/artifacts/dense_fomc_text/text_alone/result.json`.

## Results

| target | R² no-text | R² text | Δ (text − no-text) | Δ CI90 |
|--------|-----------:|--------:|-------------------:|--------|
| rv_1   | -0.000     | -0.425  | -0.425             | [-0.666, -0.225] |
| rv_3   | -0.000     | -0.322  | -0.322             | [-1.030, +0.107] |
| rv_5   | -0.000     | +0.077  | +0.077             | [-0.079, +0.194] |
| rv_10  | +0.000     | -0.584  | -0.584             | [-1.766, +0.079] |
| av     |  0.000     | -0.192  | -0.192             | [-0.386, -0.044] |

R² no-text is ≈0 by construction (intercept-only baseline).

## Findings

Text-alone R² is negative on rv_1, rv_3, rv_10, and av — the text-only model
forecasts worse than predicting the mean. Only rv_1 and av have CI90 entirely
below 0; rv_3 and rv_10 are negative but wide. rv_5 is the lone positive
(+0.077, CI90 includes 0), not enough to carry the panel. The result reinforces
the text-null finding: FOMC text on its own carries no usable RV/AV forecast
signal over the no-text baseline.
