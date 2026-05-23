"""Path B Chunk 1: VIX term slope + yield-curve slope per-bar emission.

The slopes are computed in
:func:`app.data.event_dataset_builder._build_prior_window` from the
joined cross-asset row. These tests pin the formula and the
zero-default behaviour on missing inputs without exercising the full
yfinance fetch path.
"""

from __future__ import annotations

import datetime as dt
import math

import pytest


def _make_series(n: int = 30):
    from app.data.event_dataset_builder import _CloseSeries

    start = dt.date(2023, 1, 1)
    return _CloseSeries(
        dates=[start + dt.timedelta(days=i) for i in range(n)],
        close=[100.0 + i * 0.5 for i in range(n)],
        volume=[1_000_000.0] * n,
    )


def _cross_asset_lookup_constant(
    vix: float, vix3m: float, tnx: float, irx: float
) -> dict[dt.date, dict[str, float]]:
    """Same cross-asset row for every date, so derived slopes are constant."""

    base = _make_series().dates
    row = {
        "vix_close": vix,
        "vix3m_close": vix3m,
        "tnx_close": tnx,
        "irx_close": irx,
    }
    return {d: row for d in base}


def test_vix_term_slope_is_log_ratio_when_both_inputs_present() -> None:
    from app.data import event_dataset_builder as edb

    series = _make_series()
    lookup = _cross_asset_lookup_constant(vix=18.0, vix3m=20.0, tnx=4.2, irx=5.1)
    bars = edb._build_prior_window(
        series, as_of=series.dates[-1] + dt.timedelta(days=1), window_days=5,
        cross_asset_lookup=lookup,
    )
    assert bars and bars[0].vix_term_slope == pytest.approx(math.log(20.0 / 18.0))


def test_yield_curve_slope_is_tnx_minus_irx() -> None:
    from app.data import event_dataset_builder as edb

    series = _make_series()
    lookup = _cross_asset_lookup_constant(vix=18.0, vix3m=20.0, tnx=4.2, irx=5.1)
    bars = edb._build_prior_window(
        series, as_of=series.dates[-1] + dt.timedelta(days=1), window_days=5,
        cross_asset_lookup=lookup,
    )
    assert bars and bars[0].yield_curve_slope_10y_3m == pytest.approx(4.2 - 5.1)


def test_slopes_zero_when_inputs_missing() -> None:
    """Missing cross-asset inputs (pre-2002 VIX3M, holiday rows) must
    produce 0.0 slopes — never NaN, never raise — so the rich-feature
    block stays clean for the loader's per-fold scaler."""

    from app.data import event_dataset_builder as edb

    series = _make_series()
    # Lookup with vix3m = 0.0 simulates a pre-VIX3M date.
    lookup = _cross_asset_lookup_constant(vix=18.0, vix3m=0.0, tnx=0.0, irx=0.0)
    bars = edb._build_prior_window(
        series, as_of=series.dates[-1] + dt.timedelta(days=1), window_days=5,
        cross_asset_lookup=lookup,
    )
    assert bars
    bar = bars[0]
    assert bar.vix_term_slope == 0.0
    assert bar.yield_curve_slope_10y_3m == 0.0


def test_slopes_emitted_in_bars_to_json() -> None:
    """The JSON payload that lands in ``prior_bars_json`` must carry
    both raw closes and derived slopes so the loader can read them back
    without joining the cross-asset lookup again."""

    from app.data import event_dataset_builder as edb

    series = _make_series()
    lookup = _cross_asset_lookup_constant(vix=18.0, vix3m=20.0, tnx=4.2, irx=5.1)
    bars = edb._build_prior_window(
        series, as_of=series.dates[-1] + dt.timedelta(days=1), window_days=5,
        cross_asset_lookup=lookup,
    )
    blob = edb._bars_to_json(bars)
    for key in (
        "vix3m_close",
        "irx_close",
        "vix_term_slope",
        "yield_curve_slope_10y_3m",
    ):
        assert key in blob, f"prior-bars JSON missing key {key!r}"
