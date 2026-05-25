"""Pin strict-backward semantics for the pre-meeting expectation features (#291)."""

from __future__ import annotations

import datetime as _dt

import pytest

from app.data.rates_event_features import (
    HALF_MOVE_BPS,
    STANDARD_MOVE_BPS,
    compute_pre_meeting_features,
    days_since_last_rate_change,
    implied_move_probabilities,
    implied_next_move_bps,
    trailing_yield_change_bps,
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
    columns: dict[str, list[tuple[_dt.date, float]]],
) -> RatesPanelLookup:
    dates: dict[str, tuple[_dt.date, ...]] = {}
    values: dict[str, tuple[float, ...]] = {}
    for column, observations in columns.items():
        pairs = sorted(observations)
        dates[column] = tuple(d for d, _ in pairs)
        values[column] = tuple(v for _, v in pairs)
    return RatesPanelLookup(dates_by_column=dates, values_by_column=values)


def test_pre_meeting_features_never_consume_event_day_observation() -> None:
    """No pre-meeting feature may read a yield published on event_date itself.

    Construct a series where the announcement-day yield jumps to a
    sentinel value (999.0). If any pre-meeting feature absorbs it the
    contract has regressed.
    """

    calendar = _dense_business_calendar(_dt.date(2026, 1, 5), 30)
    event = calendar[15]
    observations: dict[str, list[tuple[_dt.date, float]]] = {}
    for column in (
        "treas_1y",
        "treas_2y",
        "treas_5y",
        "treas_10y",
        "slope_10y_2y",
        "slope_10y_3m",
        "ff_target_upper",
    ):
        observations[column] = []
        for i, day in enumerate(calendar):
            if day < event:
                # Pre-event: stable plausible level so the strict-backward
                # answer is a real number, not the sentinel.
                observations[column].append((day, 1.00))
            else:
                # Event day + after: sentinel that no strict-backward
                # feature is allowed to read.
                observations[column].append((day, 999.0))

    lookup = _build_lookup(observations)
    features = compute_pre_meeting_features(lookup, calendar, event_date=event)

    for field, value in features.__dict__.items():
        if value is None:
            continue
        # None of the strict-backward features may equal the sentinel
        # or anywhere near 999. The trailing change is in bps, and
        # implied probabilities are bounded in [0, 1], so the absolute
        # ceiling check below is loose enough to catch any leak without
        # false positives.
        assert abs(value) < 500.0, (
            f"Pre-meeting feature {field!r} leaked the sentinel "
            f"event-day observation (value={value})"
        )


def test_trailing_yield_change_uses_strict_backward_endpoints() -> None:
    """`trailing_yield_change_bps` reads the t-1 / t-6 close, never t."""

    calendar = _dense_business_calendar(_dt.date(2026, 1, 5), 30)
    event = calendar[15]
    # Pre-event series rises by exactly 5 bps per trading day. The
    # 5-day trailing change at t-1 must therefore be 25 bps regardless
    # of what happens at t or after.
    observations = []
    for i, day in enumerate(calendar):
        if day < event:
            observations.append((day, 4.00 + 0.05 * i))  # 5 bps per day
        else:
            observations.append((day, 999.0))  # poisoned post-event values

    lookup = _build_lookup({"treas_2y": observations})
    bps = trailing_yield_change_bps(
        lookup, calendar, column="treas_2y", event_date=event, horizon=5
    )
    assert bps == pytest.approx(25.0, abs=1e-9), (
        f"5-day trailing yield change should be +25 bps; got {bps}. "
        "Non-25 value means the post-event sentinel leaked into the window."
    )


def test_implied_next_move_bps_computes_one_year_spread_at_t_minus_one() -> None:
    """``implied_next_move_bps`` reads the strict-backward 1y - FFR upper."""

    calendar = _dense_business_calendar(_dt.date(2026, 1, 5), 30)
    event = calendar[15]
    treas_1y = [
        (day, 5.25 if day < event else 999.0) for day in calendar
    ]
    ff_target_upper = [
        (day, 5.00 if day < event else 999.0) for day in calendar
    ]

    lookup = _build_lookup(
        {"treas_1y": treas_1y, "ff_target_upper": ff_target_upper}
    )
    bps = implied_next_move_bps(lookup, event)
    # (5.25 - 5.00) * 100 = +25 bps.
    assert bps == pytest.approx(25.0, abs=1e-9)


def test_implied_move_probabilities_at_canonical_thresholds() -> None:
    """The hike / cut / pause bucketing rules ramp linearly between thresholds."""

    # Within the no-move band: pause = 1.
    assert implied_move_probabilities(0.0) == (0.0, 0.0, 1.0)
    assert implied_move_probabilities(HALF_MOVE_BPS - 0.1) == (0.0, 0.0, 1.0)

    # At STANDARD_MOVE_BPS: full hike when positive, full cut when negative.
    assert implied_move_probabilities(STANDARD_MOVE_BPS) == (1.0, 0.0, 0.0)
    assert implied_move_probabilities(-STANDARD_MOVE_BPS) == (0.0, 1.0, 0.0)

    # At the midpoint between HALF and STANDARD: ramp value 0.5.
    mid = (HALF_MOVE_BPS + STANDARD_MOVE_BPS) / 2.0
    hike, cut, pause = implied_move_probabilities(mid)
    assert hike == pytest.approx(0.5)
    assert cut == 0.0
    assert pause == pytest.approx(0.5)
    # Probabilities always sum to 1.
    for implied in (-30.0, -15.0, -5.0, 0.0, 5.0, 15.0, 30.0):
        h, c, p = implied_move_probabilities(implied)
        assert h is not None and c is not None and p is not None
        assert h + c + p == pytest.approx(1.0, abs=1e-9)

    # None propagates.
    assert implied_move_probabilities(None) == (None, None, None)


def test_days_since_last_rate_change_walks_back_to_step() -> None:
    """The helper finds the most recent change in DFEDTARU before event_date."""

    calendar = _dense_business_calendar(_dt.date(2026, 1, 5), 30)
    event = calendar[20]
    # FF target jumps from 4.75 -> 5.00 at calendar[10] (10 trading days
    # before event). The helper measures calendar-day gap between the
    # last observation strictly before event (calendar[19]) and the
    # first observation at the new level (calendar[10]).
    rates = []
    for i, day in enumerate(calendar):
        if i < 10:
            rates.append((day, 4.75))
        else:
            rates.append((day, 5.00))

    lookup = _build_lookup({"ff_target_upper": rates})
    days = days_since_last_rate_change(lookup, event)
    expected = (calendar[19] - calendar[10]).days
    assert days == expected, (
        f"days_since_last_rate_change should equal the gap between "
        f"calendar[19] and calendar[10] = {expected}; got {days}."
    )


def test_days_since_last_rate_change_returns_none_when_no_change_in_lookback() -> None:
    """A flat target rate over the whole window returns None."""

    calendar = _dense_business_calendar(_dt.date(2026, 1, 5), 30)
    event = calendar[20]
    rates = [(day, 5.25) for day in calendar]
    lookup = _build_lookup({"ff_target_upper": rates})
    assert days_since_last_rate_change(lookup, event) is None


def test_compute_pre_meeting_features_returns_none_when_lookup_empty() -> None:
    """An empty lookup yields a PreMeetingFeatures with every field None."""

    calendar = _dense_business_calendar(_dt.date(2026, 1, 5), 30)
    event = calendar[10]
    lookup = RatesPanelLookup(dates_by_column={}, values_by_column={})
    features = compute_pre_meeting_features(lookup, calendar, event_date=event)
    for value in features.__dict__.values():
        assert value is None
