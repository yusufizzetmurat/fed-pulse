"""FOMC-day 14:00 ET cutoff guard.

The FOMC releases its policy statement at 2pm Eastern. Same-day market data
captured after that boundary embeds the announcement in the feature frame
and is a textbook lookahead bug. The guard at
``app.services.market_data.assert_fomc_day_market_cutoff`` is the seam where
feature-assembly callers must check. DST handling is automatic: the
comparison happens in America/New_York local time, so 14:00 ET corresponds
to 19:00 UTC in EST months and 18:00 UTC in EDT months and the assertion
fires above either boundary in the matching half of the year.
"""

from __future__ import annotations

import datetime as _dt
from zoneinfo import ZoneInfo

import pytest

from app.services.market_data import (
    FOMC_LOCAL_CUTOFF_TIME,
    FOMC_ZONE,
    assert_fomc_day_market_cutoff,
    is_fomc_day,
)


_NYC = ZoneInfo("America/New_York")


def _edt_fomc_day() -> _dt.date:
    # 2024-09-18 is a scheduled FOMC meeting in EDT (DST active).
    return _dt.date(2024, 9, 18)


def _est_fomc_day() -> _dt.date:
    # 2024-01-31 is a scheduled FOMC meeting in EST.
    return _dt.date(2024, 1, 31)


def test_is_fomc_day_recognises_scheduled_meeting() -> None:
    assert is_fomc_day(_edt_fomc_day()) is True
    assert is_fomc_day(_est_fomc_day()) is True


def test_is_fomc_day_returns_false_for_off_day() -> None:
    assert is_fomc_day(_dt.date(2024, 9, 14)) is False


def test_cutoff_passes_for_non_fomc_day() -> None:
    timestamp = _dt.datetime(2024, 9, 14, 23, 0, 0, tzinfo=_dt.timezone.utc)
    assert assert_fomc_day_market_cutoff(timestamp) is None


def test_cutoff_passes_at_or_before_14_00_et_in_edt() -> None:
    # 14:00 ET on 2024-09-18 (EDT) = 18:00 UTC. The guard must accept both
    # the local-time anchored 14:00 and the equivalent 18:00 UTC.
    meeting = _edt_fomc_day()
    on_cutoff_local = _dt.datetime.combine(meeting, FOMC_LOCAL_CUTOFF_TIME, tzinfo=_NYC)
    assert assert_fomc_day_market_cutoff(on_cutoff_local) is None

    on_cutoff_utc = on_cutoff_local.astimezone(_dt.timezone.utc)
    assert on_cutoff_utc.hour == 18  # EDT sanity check
    assert assert_fomc_day_market_cutoff(on_cutoff_utc) is None

    one_minute_earlier = on_cutoff_local - _dt.timedelta(minutes=1)
    assert assert_fomc_day_market_cutoff(one_minute_earlier) is None


def test_cutoff_passes_at_or_before_14_00_et_in_est() -> None:
    # 14:00 ET on 2024-01-31 (EST) = 19:00 UTC. The guard must accept that.
    meeting = _est_fomc_day()
    on_cutoff_local = _dt.datetime.combine(meeting, FOMC_LOCAL_CUTOFF_TIME, tzinfo=_NYC)
    assert assert_fomc_day_market_cutoff(on_cutoff_local) is None

    on_cutoff_utc = on_cutoff_local.astimezone(_dt.timezone.utc)
    assert on_cutoff_utc.hour == 19  # EST sanity check
    assert assert_fomc_day_market_cutoff(on_cutoff_utc) is None


def test_cutoff_raises_in_post_announcement_window_during_edt() -> None:
    # The pre-fix bug: 18:30 UTC on an EDT FOMC day = 14:30 EDT, which is
    # AFTER the 14:00 ET cutoff. The fixed-19:00-UTC implementation let
    # this slip through silently; the local-time comparison must reject.
    meeting = _edt_fomc_day()
    leak = _dt.datetime(meeting.year, meeting.month, meeting.day, 18, 30, tzinfo=_dt.timezone.utc)
    with pytest.raises(ValueError, match=r"FOMC 14:00 ET cutoff"):
        assert_fomc_day_market_cutoff(leak, feature_name="close_price")


def test_cutoff_raises_one_minute_after_14_00_et_in_edt() -> None:
    meeting = _edt_fomc_day()
    on_cutoff_local = _dt.datetime.combine(meeting, FOMC_LOCAL_CUTOFF_TIME, tzinfo=_NYC)
    after = on_cutoff_local + _dt.timedelta(minutes=1)
    with pytest.raises(ValueError, match=r"FOMC 14:00 ET cutoff"):
        assert_fomc_day_market_cutoff(after, feature_name="close_price")


def test_cutoff_raises_one_minute_after_14_00_et_in_est() -> None:
    meeting = _est_fomc_day()
    on_cutoff_local = _dt.datetime.combine(meeting, FOMC_LOCAL_CUTOFF_TIME, tzinfo=_NYC)
    after = on_cutoff_local + _dt.timedelta(minutes=1)
    with pytest.raises(ValueError, match=r"FOMC 14:00 ET cutoff"):
        assert_fomc_day_market_cutoff(after, feature_name="close_price")


def test_cutoff_accepts_naive_timestamps_as_utc() -> None:
    # Naive datetimes are interpreted as UTC. On an EDT meeting day,
    # 18:00 UTC corresponds to 14:00 EDT (on cutoff) and 19:00 UTC to
    # 15:00 EDT (after cutoff).
    meeting = _edt_fomc_day()
    naive_on_cutoff = _dt.datetime.combine(meeting, _dt.time(18, 0, 0))
    assert assert_fomc_day_market_cutoff(naive_on_cutoff) is None

    naive_after = _dt.datetime.combine(meeting, _dt.time(19, 0, 0))
    with pytest.raises(ValueError):
        assert_fomc_day_market_cutoff(naive_after)


def test_fomc_zone_is_new_york() -> None:
    # Locks the zone identity so a future refactor cannot drift to a fixed
    # UTC offset.
    assert FOMC_ZONE.key == "America/New_York"
    assert FOMC_LOCAL_CUTOFF_TIME == _dt.time(14, 0, 0)
