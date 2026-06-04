# ADR 0034 — GARCH(1,1)-residual variant of the forward-realised-vol target

The Phase 9 V2 target supervises the raw sample std of log returns over `[T+1, T+10]`. That quantity is heteroskedastic and heavy-tailed, and Hansen-Lunde 2005 ("does anything beat a GARCH(1,1)?") is the standing negative finding in the volatility-forecasting literature. A neural forecaster predicting the raw target spends most of its capacity re-learning what GARCH already explains. The textbook fix is to subtract the GARCH baseline from the raw target and supervise the residual: the part the text + macro features can plausibly explain that GARCH cannot. Donaldson-Kamstra 1997 and Engle-Patton 2001 report consistent lift on equity vol when the decomposition is applied.

This pairs with the rates side. #305 / ADR 0027 isolates the FOMC-attributable component of the 5-day rates move via a projection onto the policy-surprise direction. #236 isolates the GARCH-unanticipated component of forward-realised vol via subtraction against a strict-prior conditional-variance forecast. Two decompositions, same idea: supervise the part of the target the text actually predicts.

## What lands

Two new nullable columns on every events.parquet built after this commit:

- `forward_realized_vol_10d_garch_baseline` — GARCH(1,1) 10-day-ahead 1-day-equivalent vol forecast, fitted on log returns strictly before `event_date`.
- `forward_realized_vol_10d_garch_residual` — raw target minus baseline.

Both are frozen into the parquet at build time so the decomposition is byte-reproducible. The baseline is persisted alongside the residual so the raw quantity stays recoverable at inference (predicted vol = predicted residual + baseline).

The fit uses `arch.arch_model(scaled, mean="Zero", vol="Garch", p=1, q=1)` from Sheppard's `arch` library (already on `backend/pyproject.toml` for `scripts/garch_baseline.py`). Returns are scaled to percentage units before the fit, since QMLE is poorly-scaled on raw log returns of magnitude 1e-2, and the forecast variance is de-scaled back to log-return units before the residual subtraction. The 10-step forecast collapses to the mean variance and square-roots to a 1-day-equivalent vol, matching the scale `_forward_realized_vol` reports over the same window. The subtraction is only meaningful with both legs on the same scale, which is why the collapse + de-scale happens before the difference.

## Strict-prior contract

The fit consumes only closes whose date is strictly less than `event_date`, replicated from `_CloseSeries.index_strictly_before` (the same gate `_volatility_shift`'s pre-event leg uses). The forecast is conditional-on-fit and reads no close at or after `event_date`. `tests/regression/test_feature_provenance_as_of.py` classifies both columns as target-only: lookback bars stay `None`, only the supervised row carries the values. The audit doc records the fit's strict-prior contract and the forecast's conditional-on-fit semantics.

## Edge cases

- `MIN_FIT_RETURNS = 252`. Below ~one trading year of strict-prior returns, QMLE is dominated by initial-condition noise. Below the floor both columns degrade to `None` and the raw target stays intact. Events.parquet routinely carries events from 2010+, so the gate is mostly for synthetic / truncated fixtures.
- Convergence failures (LinAlgError, optimiser ValueError, anything else the arch library throws on degenerate windows) degrade to `None` via a broad `except`.
- `forward_realized_vol_10d = None` (event within 10 td of the price series end) → residual is `None`; baseline can still be computed and is persisted on its own.

## What this PR does not do

No default-on / default-off knob. The columns land on every events.parquet built after this commit; pre-#236 parquets validate cleanly because the columns are `required=False`. Downstream consumers that don't read them are byte-identical to before: the FeatureVector dataclass has two new fields with `None` defaults and `as_rich_list` doesn't emit them.

The supervised-target switch (predict residual vs predict raw) lives on the training loop's target-column knob, not on the build. A training-time `--vol-target-mode` analogous to #305's `--rates-target-mode` is the follow-up.

## References

- `backend/app/data/garch_residual.py`, `backend/app/data/event_dataset_builder.py`
- ADR 0027 (rates-side decomposition precedent)
- Bollerslev (1986); Hansen, Lunde (2005)
