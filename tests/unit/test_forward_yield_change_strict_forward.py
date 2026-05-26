"""Pin strict-forward semantics for the rates-complex forward targets (#291).

The forward yield-change targets (``yield_2y_change_5d``,
``yield_5y_change_5d``, ``terminal_rate_change_5d``) must mirror the
convention pinned by ``test_forward_realized_vol_strict_forward.py`` for
the existing vol-regime target: ``yield[t-1]`` never appears in the
target window, ``yield[t]`` (the FOMC announcement-day close) is the
baseline, and ``yield[t+horizon]`` is the endpoint.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from app.data.rates_event_features import (
    FORWARD_TARGET_COLUMNS,
    forward_yield_change_bps,
)
from app.data.rates_panel import RatesPanelLookup


def _dense_business_calendar(start: _dt.date, n: int) -> list[_dt.date]:
    days: list[_dt.date] = []
    cursor = start
    while len(days) < n:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += _dt.timedelta(days=1)
    return days


def _build_lookup(
    column: str, observations: list[tuple[_dt.date, float]]
) -> RatesPanelLookup:
    pairs = sorted(observations)
    return RatesPanelLookup(
        dates_by_column={column: tuple(d for d, _ in pairs)},
        values_by_column={column: tuple(v for _, v in pairs)},
    )


def test_forward_change_uses_event_day_as_baseline_not_prior_day() -> None:
    """``yield[t-1]`` must NOT participate in the target window.

    Construct a series where the announcement-day jump is 100 bps
    (DGS2 jumps from 4.00 to 5.00 between t-1 and t) and every
    strictly-forward observation is flat at 5.00. Under the strict-
    forward convention the 5-day change is exactly zero (baseline =
    endpoint = 5.00). Under the deprecated ``yield[t+5] - yield[t-1]``
    convention it would be +100 bps.
    """

    calendar = _dense_business_calendar(_dt.date(2026, 1, 5), 30)
    event = calendar[10]  # t at index 10
    observations = []
    for i, day in enumerate(calendar):
        if i < 10:
            observations.append((day, 4.00))  # pre-event flat at 4.00
        else:
            observations.append((day, 5.00))  # event day + post-event flat at 5.00

    lookup = _build_lookup("treas_2y", observations)
    bps = forward_yield_change_bps(
        lookup, calendar, column="treas_2y", event_date=event, horizon=5
    )
    assert bps == pytest.approx(0.0, abs=1e-9), (
        "Strict-forward yield change with constant post-event yields "
        f"must collapse to 0; got {bps}. A non-zero value means the "
        "announcement-day jump leaked into the target window."
    )


def test_forward_change_endpoint_is_t_plus_horizon_close() -> None:
    """The endpoint must be ``yield[t+horizon]``, not ``yield[t+horizon-1]``."""

    calendar = _dense_business_calendar(_dt.date(2026, 1, 5), 30)
    event = calendar[10]
    observations = []
    for i, day in enumerate(calendar):
        if i <= 14:
            observations.append((day, 4.50))  # t=10 .. t+4 flat at 4.50
        elif i == 15:
            observations.append((day, 4.75))  # t+5 jumps to 4.75 (+25 bps)
        else:
            observations.append((day, 4.75))

    lookup = _build_lookup("treas_2y", observations)
    bps = forward_yield_change_bps(
        lookup, calendar, column="treas_2y", event_date=event, horizon=5
    )
    assert bps == pytest.approx(25.0, abs=1e-9), (
        f"5-day forward change should be +25 bps (4.50 → 4.75); got {bps}."
    )


def test_forward_change_returns_none_at_end_of_calendar() -> None:
    """An event within ``horizon`` trading days of the calendar end yields None."""

    calendar = _dense_business_calendar(_dt.date(2026, 1, 5), 12)
    event = calendar[10]  # only 1 trading day after t in the calendar
    observations = [(day, 4.50) for day in calendar]

    lookup = _build_lookup("treas_2y", observations)
    bps = forward_yield_change_bps(
        lookup, calendar, column="treas_2y", event_date=event, horizon=5
    )
    assert bps is None


def test_forward_change_returns_none_when_column_missing() -> None:
    """A column absent from the lookup degrades to None per row."""

    calendar = _dense_business_calendar(_dt.date(2026, 1, 5), 30)
    event = calendar[10]
    lookup = RatesPanelLookup(dates_by_column={}, values_by_column={})

    for fred_column, _ in FORWARD_TARGET_COLUMNS:
        assert forward_yield_change_bps(
            lookup, calendar, column=fred_column, event_date=event, horizon=5
        ) is None


def test_forward_change_uses_strict_forward_for_non_trading_day_event() -> None:
    """A weekend event date should snap forward to the next trading day."""

    calendar = _dense_business_calendar(_dt.date(2026, 1, 5), 30)
    # Calendar starts on Mon 2026-01-05; calendar[4] is Fri 2026-01-09.
    # Adding two calendar days lands on Sun 2026-01-11; ``t`` then
    # snaps to calendar[5] = Mon 2026-01-12 as the first trading day
    # on or after the event date.
    sunday_event = calendar[4] + _dt.timedelta(days=2)
    assert sunday_event.weekday() == 6, "test setup: event date must be a Sunday"

    observations = [(day, 4.00 + 0.1 * i) for i, day in enumerate(calendar)]
    # values: 4.0, 4.1, 4.2, ..., monotonically rising by 10 bps per
    # trading day. With t snapped to calendar[5] (yield 4.5),
    # t+5 = calendar[10] (yield 5.0). Strict-forward change = +50 bps.
    lookup = _build_lookup("treas_2y", observations)
    bps = forward_yield_change_bps(
        lookup, calendar, column="treas_2y", event_date=sunday_event, horizon=5
    )
    assert bps == pytest.approx(50.0, abs=1e-9), (
        f"Strict-forward snap should produce +50 bps; got {bps}."
    )
