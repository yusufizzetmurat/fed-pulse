# ADR 0008 — Conformal prediction for forecast confidence bands

Status: accepted, in production.
Date: 2026-05-16.
Supersedes: the previous fallback-only Gaussian-z band reporting.
References:
- `backend/app/evaluation/conformal.py` — the split-conformal implementation.
- `backend/app/services/forecaster.py::_build_confidence_bands` — the inference seam that swaps in conformal bands when the manifest exists.
- `frontend/lib/analyze/format.ts::bandLabel` — UI label that exposes the band source to the reader.
- `docs/benchmark-policy.md §"Calibrated uncertainty"` — empirical-coverage contract.

## Context

The forecaster predicts a point close-price and a point volatility for each horizon. The dashboard surfaces a confidence band around each prediction so a reader can tell how wide the model's uncertainty is. Two band methodologies were on the table:

1. **Gaussian-z fallback.** Multiply the in-sample residual standard deviation by the 80th-percentile z-score (1.2816) and apply symmetrically. Easy to compute, but the band width is parametric in a Gaussianity assumption that the residuals demonstrably violate (heavy tails around FOMC dates, asymmetric reaction to surprise hikes vs cuts).
2. **Split-conformal prediction.** Reserve a holdout fold for calibration, compute non-conformity scores against the model's point predictions on that fold, and emit bands whose width is the empirical 80th-percentile non-conformity score. Distribution-free, coverage-guaranteed up to the calibration sample size.

## Decision

Adopt split-conformal as the primary band source; keep Gaussian-z as the fallback when no calibration manifest is on disk.

The conformal manifest is produced during training-package construction and is named ``<checkpoint>.conformal.json``. When the manifest exists next to the checkpoint, ``_build_confidence_bands`` reads it and emits conformal bands; the API marks the band source in the response as ``"conformal"``. When the manifest is missing — which happens on a fresh bootstrap — the Gaussian-z fallback runs and the API marks the source as ``"gaussian_z"``.

The dashboard reads ``series.forecast_band_source`` and labels the band accordingly so a reader can distinguish a calibrated band from a parametric fallback at a glance (see ``frontend/lib/analyze/format.ts``).

## Consequences

- The dashboard never silently degrades to Gaussian-z without surfacing the methodology change. Reviewers and operators get the same signal the API carries.
- Conformal coverage targets the empirical 80% level by default. The level is configurable in the manifest; the API also exposes the realised ``conformal_coverage`` so the operator can see the actual hit-rate against the nominal target.
- The fallback is intentionally conservative. Gaussian-z over heavy tails produces a band that is too tight, not too wide. A reader who sees the "Gaussian-z" label should treat the band width as a lower bound and either re-run training-package construction (to populate the manifest) or apply their own buffer.
- Future work: extend the calibration to be per-asset and per-regime rather than the current pooled fit. Documented in ``docs/benchmark-policy.md``.
