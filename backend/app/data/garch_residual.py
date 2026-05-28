"""GARCH(1,1) baseline + residual on the forward 10-day realised vol target (#236).

Fits a Zero-mean GARCH(1,1) on the asset's log returns dated *strictly
before* ``event_date`` and forecasts the next 10 trading days' conditional
volatility. The 10-day forecast is collapsed to a 1-day-equivalent std
(mean of the per-step variances, square-rooted) so the scale matches the
sample std of log returns over ``[T+1, T+10]`` that
:func:`app.data.event_dataset_builder._forward_realized_vol` reports.

The residual target is

    forward_realized_vol_10d_garch_residual
        = forward_realized_vol_10d - forward_realized_vol_10d_garch_baseline

so the supervised quantity isolates the *unanticipated* component of the
realised vol given a standard time-series model. Neural networks are
known to be poor at the raw heteroskedastic level but reasonable at the
residual once the classical conditional-variance baseline is stripped
off; the hybrid GARCH-NN recipe is the textbook decomposition.

Strict-prior contract: the fit consumes only returns whose ending close
is dated ``< event_date``. The forecast is conditional-on-fit and never
reads any close at or after ``event_date``. The leak surface is
identical to the strict-backward windows
:func:`app.data.event_dataset_builder._volatility_shift` uses on its
pre-event leg.

Convergence and edge cases:

- ``arch`` is an optional import (already on ``backend/pyproject.toml``
  as ``arch>=6.3``). Import is lazy so the module is importable in
  test environments that strip the dep.
- Below ``MIN_FIT_RETURNS`` strictly-prior returns the fit is skipped
  and the baseline / residual are ``None``. ~252 trading days (a year)
  is the documented floor for GARCH(1,1) convergence on equity returns.
- Convergence failures (numerical, missing-data, ``arch.LinAlgError``)
  degrade to ``None``; the supervised row keeps the raw target column
  intact, only the residual is missing.
- ``forward_realized_vol_10d`` itself is required to compute the
  residual; if the target is ``None`` (event too close to end of price
  series) the residual is also ``None``.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


# Exceptions the ``arch`` fit / forecast path actually raises on a
# degenerate window (singular Hessian, non-finite scores, scipy
# optimiser blow-up, malformed input). ``ConvergenceWarning`` is a
# Warning subclass and never propagates as an exception, so it is
# deliberately not in this tuple; the fit emits it via warnings.warn
# and we suppress it at the call site via ``show_warning=False``. The
# tuple is intentionally narrow: anything outside it (e.g. AttributeError
# from a typo on a module constant) must surface in CI rather than be
# silently swallowed and turned into a None column.
_ARCH_FIT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    np.linalg.LinAlgError,
    ValueError,
    RuntimeError,
    OverflowError,
)


# Minimum strictly-prior returns required to fit GARCH(1,1). ~252
# trading days (one year) is the documented floor for stable
# convergence on equity returns; below that the QMLE fit is dominated
# by initial-condition noise. The events.parquet routinely carries
# events dating back to 2010+ so the gate is rarely hit in practice;
# it exists to fail gracefully on synthetic / truncated fixtures.
MIN_FIT_RETURNS: int = 252

# Forecast horizon in trading days. Matches the 10-day window the
# raw target measures so the baseline and the realised quantity are
# defined over the same forward span.
GARCH_FORECAST_HORIZON: int = 10

# Returns are scaled to percentage units (× 100) before the fit. The
# ``arch`` optimiser is notoriously poorly-scaled on raw log returns
# (~1e-2 magnitude); percentage scaling brings the parameter space
# closer to unity and stabilises the QMLE step. The forecast variance
# is then de-scaled back to log-return units before exposure.
_PERCENT_SCALE: float = 100.0


@dataclass(frozen=True)
class GarchResidualResult:
    """Baseline + residual scalars on the supervised target row.

    All three fields carry log-return units (the same units the raw
    ``forward_realized_vol_10d`` column reports). ``baseline`` is the
    GARCH(1,1) 10-day-ahead 1-day-equivalent vol forecast; ``residual``
    is the raw target minus the baseline. Both are ``None`` when the
    fit was skipped (insufficient prior returns), failed to converge,
    or the raw target was itself missing.
    """

    baseline: float | None
    residual: float | None
    fit_returns_n: int


def _log_returns(closes: Sequence[float]) -> list[float]:
    out: list[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev <= 0 or cur <= 0:
            continue
        out.append(math.log(cur / prev))
    return out


def _forecast_one_step_equiv_vol(model_result, *, horizon: int) -> float | None:
    """Mean per-step variance over the next ``horizon`` steps, square-rooted.

    ``arch_model.forecast(horizon=h)`` returns a per-step variance path
    ``[v_1, v_2, ..., v_h]`` in the fit's units (here, percentage
    squared). Collapsing to the mean variance and square-rooting yields
    the 1-day-equivalent std over the window, which is the same scale
    the raw ``forward_realized_vol_10d`` column reports (sample std of
    log returns over ``[T+1, T+horizon]``).
    """

    try:
        forecast = model_result.forecast(horizon=horizon, reindex=False)
    except _ARCH_FIT_EXCEPTIONS:
        return None
    variance_row = forecast.variance.iloc[-1].to_numpy()
    if variance_row.size == 0:
        return None
    mean_var = float(variance_row.mean())
    if mean_var <= 0.0 or not math.isfinite(mean_var):
        return None
    return math.sqrt(mean_var)


def _fit_garch_and_forecast(
    returns: Sequence[float],
    *,
    horizon: int = GARCH_FORECAST_HORIZON,
) -> float | None:
    """Fit Zero-mean GARCH(1,1) and return the 1-day-equivalent forecast vol.

    Returns are expected in raw log-return units; the helper scales to
    percentage internally, fits, forecasts ``horizon`` steps, then
    de-scales the forecast back to log-return units. Returns ``None``
    when the fit fails for any reason (numerical, missing arch, etc).
    """

    if len(returns) < MIN_FIT_RETURNS:
        return None
    try:
        from arch import arch_model  # lazy import keeps the module importable without arch
    except ImportError:  # pragma: no cover - dep is locked in pyproject
        logger.warning(
            "arch package not importable; GARCH residual target will be None for every event"
        )
        return None
    try:
        scaled = np.asarray(returns, dtype=float) * _PERCENT_SCALE
        # show_warning=False suppresses the per-fit "DataScaleWarning"
        # the optimiser emits on tail-heavy windows; we control the
        # scaling explicitly so the warning is noise.
        model = arch_model(scaled, mean="Zero", vol="Garch", p=1, q=1, rescale=False)
        result = model.fit(disp="off", show_warning=False)
    except _ARCH_FIT_EXCEPTIONS:
        return None
    forecast_pct = _forecast_one_step_equiv_vol(result, horizon=horizon)
    if forecast_pct is None:
        return None
    return forecast_pct / _PERCENT_SCALE


def compute_garch_residual(
    *,
    prior_closes: Sequence[float],
    forward_realized_vol_10d: float | None,
    horizon: int = GARCH_FORECAST_HORIZON,
) -> GarchResidualResult:
    """Fit GARCH(1,1) on strict-prior closes and emit the baseline + residual.

    ``prior_closes`` is the asset close series dated strictly before the
    event (the caller slices it on ``index_strictly_before(event_date)``).
    ``forward_realized_vol_10d`` is the raw realised-vol target for the
    same supervised row.

    The helper computes log returns over ``prior_closes`` (dropping any
    non-positive closes), fits the GARCH(1,1), forecasts ``horizon``
    steps, collapses to a 1-day-equivalent vol, and returns the baseline
    + residual. Missing baseline (insufficient data or convergence
    failure) propagates ``None`` to the residual.
    """

    returns = _log_returns(prior_closes)
    baseline = _fit_garch_and_forecast(returns, horizon=horizon)
    if baseline is None or forward_realized_vol_10d is None:
        return GarchResidualResult(
            baseline=baseline,
            residual=None,
            fit_returns_n=len(returns),
        )
    raw = float(forward_realized_vol_10d)
    if not math.isfinite(raw):
        return GarchResidualResult(
            baseline=baseline,
            residual=None,
            fit_returns_n=len(returns),
        )
    return GarchResidualResult(
        baseline=float(baseline),
        residual=raw - float(baseline),
        fit_returns_n=len(returns),
    )


def compute_for_event(
    *,
    dates: Sequence[_dt.date],
    closes: Sequence[float],
    event_date: _dt.date,
    forward_realized_vol_10d: float | None,
    horizon: int = GARCH_FORECAST_HORIZON,
) -> GarchResidualResult:
    """Per-event entry point used by :mod:`event_dataset_builder`.

    Walks the (dates, closes) series, slices to the strict-prior window
    (``dates[i] < event_date``), and delegates to
    :func:`compute_garch_residual`. The strict-prior slice is the same
    one ``_CloseSeries.index_strictly_before`` exposes; we replicate it
    here to keep the helper testable without the full builder dependency.
    """

    if len(dates) != len(closes):
        raise ValueError(
            f"dates ({len(dates)}) and closes ({len(closes)}) length mismatch"
        )
    # Binary-search-style slice: dates are sorted ascending by contract.
    # A linear scan is fine because the asset series in practice carries
    # at most ~4000 daily bars and the slice is O(n) anyway in numpy land.
    cutoff_idx = 0
    for i, d in enumerate(dates):
        if d < event_date:
            cutoff_idx = i + 1
    if cutoff_idx <= 0:
        return GarchResidualResult(baseline=None, residual=None, fit_returns_n=0)
    prior_closes = closes[:cutoff_idx]
    return compute_garch_residual(
        prior_closes=prior_closes,
        forward_realized_vol_10d=forward_realized_vol_10d,
        horizon=horizon,
    )


__all__ = (
    "GARCH_FORECAST_HORIZON",
    "GarchResidualResult",
    "MIN_FIT_RETURNS",
    "compute_for_event",
    "compute_garch_residual",
)
