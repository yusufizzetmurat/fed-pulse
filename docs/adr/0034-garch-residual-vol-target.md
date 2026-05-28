# ADR 0034 — GARCH(1,1)-residual variant of the forward-realised-vol target

Status: accepted, target column computed at build time; canonical sweep deferred to operator.
Date: 2026-05-28.
References:
- Issue #236 (closes).
- ADR 0027 — #305 surprise-decomposition target on the rates heads; the prior precedent for shipping an alternative target column alongside the raw target without disturbing the default path.
- `backend/app/data/garch_residual.py` — GARCH fit + residual helper.
- `backend/app/data/event_dataset_builder.py` — per-event invocation site; `forward_realized_vol_10d_garch_baseline` / `forward_realized_vol_10d_garch_residual` emitted on every row.
- `backend/app/models/config.py` — `FeatureVector.forward_realized_vol_10d_garch_baseline` / `forward_realized_vol_10d_garch_residual`.
- Bollerslev (1986), "Generalized autoregressive conditional heteroskedasticity." J. Econometrics.
- Hansen, Lunde (2005), "A forecast comparison of volatility models: does anything beat a GARCH(1,1)?" J. Applied Econometrics.

## Context

The Phase 9 V2 vol-regime target (`forward_realized_vol_10d`, #195) supervises the raw sample std of log returns over `[T+1, T+10]`. The quantity is heteroskedastic and heavy-tailed — exactly the regime neural networks struggle on. The classical GARCH(1,1) baseline captures most of the conditional-variance dynamics on equity returns (Hansen-Lunde 2005 — "does anything beat a GARCH(1,1)?" is the canonical negative finding in the volatility-forecasting literature); a neural forecaster that predicts the raw target spends most of its capacity re-learning what GARCH already explains.

The textbook hybrid is to subtract the GARCH baseline forecast from the raw target and supervise the residual instead. The residual isolates the *unanticipated* component of realised vol given the standard time-series model: the part the text + macro features can plausibly explain that GARCH cannot. The training surface is lower variance and closer to mean-zero, and the literature (Donaldson-Kamstra 1997; Engle-Patton 2001) reports consistent lift on equity vol forecasting when the decomposition is applied.

This sits opposite the #305 surprise-decomposition target on the rates heads. #305 isolates the FOMC-attributable component of the 5-day rates move via a 1-D projection onto the policy-surprise direction; #236 isolates the GARCH-unanticipated component of the forward-realised vol via a subtraction against a strict-prior conditional-variance forecast. Together the two targets give the §6 narrative a paired "decompose the target, supervise the part the text actually predicts" methodology axis on both regression surfaces.

## Decision

Compute two new columns at events.parquet build time, alongside the raw `forward_realized_vol_10d` column:

- `forward_realized_vol_10d_garch_baseline` — GARCH(1,1) 10-day-ahead 1-day-equivalent vol forecast, fitted on log returns of the asset's close series dated *strictly before* `event_date`.
- `forward_realized_vol_10d_garch_residual` — `forward_realized_vol_10d − forward_realized_vol_10d_garch_baseline`.

Both columns are frozen into the training package at build time (not computed at loader time) so the decomposition is reproducible byte-for-byte from the persisted parquet. The supervised target is the residual; the baseline is persisted alongside so the raw quantity is recoverable at inference (predicted vol = predicted residual + baseline).

### GARCH fit

Zero-mean GARCH(1,1) via `arch.arch_model(scaled, mean="Zero", vol="Garch", p=1, q=1)` from Kevin Sheppard's `arch` library (already on `backend/pyproject.toml` as `arch>=6.3`; the dep was added for `scripts/garch_baseline.py`'s §6.6 row). Returns are scaled to percentage units (× 100) before the fit because the QMLE optimiser is notoriously poorly-scaled on raw log returns of magnitude 1e-2; the forecast variance is de-scaled back to log-return units before the residual subtraction.

The forecast is `horizon=10` per-step variances; we collapse to the mean variance and square-root to get a 1-day-equivalent vol, which matches the scale of the sample std `_forward_realized_vol` reports over the same window. The subtraction is only meaningful if both legs sit on the same scale — that is why we collapse the per-step forecast and de-scale the percentage units before differencing rather than leaving the GARCH output in its native cumulative-variance form.

### Strict-prior contract

The fit consumes only closes whose date is strictly less than `event_date`. The slice is replicated from `_CloseSeries.index_strictly_before` so the contract is the same one `_volatility_shift`'s pre-event leg enforces. The forecast is conditional-on-fit and reads no close at or after `event_date`. The leak surface is identical to the strict-backward windows already audited under #350.

`tests/regression/test_feature_provenance_as_of.py` classifies both columns as target-only (lookback bars stay `None`, only the supervised event row carries the values). The audit doc `docs/feature-provenance-audit.md` documents the fit's strict-prior contract and the forecast's conditional-on-fit semantics; the regression test's inventory-coverage assertion gates any future addition.

### Edge cases

- `MIN_FIT_RETURNS = 252`. Below ~one trading year of strict-prior returns the GARCH(1,1) QMLE fit is dominated by initial-condition noise. Below the floor both columns degrade to `None` and the supervised row keeps the raw target intact. The events.parquet routinely carries events from 2010+, so the gate is rarely hit in practice; it exists for synthetic / truncated fixtures.
- Convergence failures (numerical, `LinAlgError`, ValueError from the optimiser) degrade to `None` via a broad `except Exception` around the fit. The arch library raises a grab-bag of error types on degenerate windows and we treat any failure as "no baseline this event."
- `forward_realized_vol_10d = None` (event within 10 td of the end of the price series) → residual is `None`; the baseline can still be computed and is persisted on its own.

### Default-off equivalent

There is no default-on / default-off knob on this PR. The columns land on every events.parquet built after this commit; older parquets validate cleanly against the schema because the columns are `required=False` (consistent with the #305 / #291 pattern). Downstream consumers that don't read the columns are byte-identical to the pre-#236 path — the FeatureVector dataclass has two new fields with `None` defaults and `as_rich_list` does not emit them.

The supervised-target switch (predict residual vs predict raw) lives on the training loop's target-column knob, not on the events.parquet build. A future training-time flag analogous to #305's `--rates-target-mode` would dispatch between `forward_realized_vol_10d` and `forward_realized_vol_10d_garch_residual` as the classifier's regression head's supervised target; that flag is a follow-up.

## Consequences

- Events.parquet grows by two nullable float columns. The schema validation is `required=False` so pre-#236 parquets still validate without the columns.
- The events.parquet build cost grows by ~50 ms per event (one GARCH(1,1) fit). Over ~150 FOMC events the marginal build time is ~8 s, negligible against the existing FRED / yfinance fetch latency.
- The `arch>=6.3` dep was already on `backend/pyproject.toml` for `scripts/garch_baseline.py`. No new dependency was added.
- The training-time switch (predict residual vs predict raw) is not wired in this PR. The §6.6 GARCH-residual row needs a canonical sweep against `forward_realized_vol_10d_garch_residual` as the supervised target, which is a Runpod follow-up.
- The §6 narrative now has a paired methodology axis: #305 decomposes the rates target, #236 decomposes the vol target. Each isolates the component a text + macro forecaster can plausibly predict from the unconditional baseline.
