"""FOMC-day 14:00 ET cutoff guard.

The FOMC releases its policy statement at 2pm Eastern. Same-day market data
captured after that boundary embeds the announcement in the feature frame
and is a textbook lookahead bug. The guard at
``app.services.market_data.assert_fomc_day_market_cutoff`` is the seam where
feature-assembly callers must check.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from app.services.market_data import (
    FOMC_DAY_CUTOFF_UTC,
    assert_fomc_day_market_cutoff,
    is_fomc_day,
)


def _known_fomc_day() -> _dt.date:
    # 2024-09-18 is a scheduled FOMC meeting and lives in
    # data/external/fomc_meetings_2010_2026.csv. The cached helper warms on
    # first call.
    return _dt.date(2024, 9, 18)


def test_is_fomc_day_recognises_scheduled_meeting() -> None:
    assert is_fomc_day(_known_fomc_day()) is True


def test_is_fomc_day_returns_false_for_off_day() -> None:
    # The Saturday before a meeting is not itself an FOMC day.
    assert is_fomc_day(_dt.date(2024, 9, 14)) is False


def test_cutoff_passes_for_non_fomc_day() -> None:
    # Any time on a non-FOMC day is fine — the cutoff only applies on
    # meeting days. A 23:00 UTC feature on 2024-09-14 must not raise.
    timestamp = _dt.datetime(2024, 9, 14, 23, 0, 0, tzinfo=_dt.timezone.utc)
    assert assert_fomc_day_market_cutoff(timestamp) is None


def test_cutoff_passes_at_or_before_14_00_et() -> None:
    # 14:00 ET (EST) = 19:00 UTC, which matches FOMC_DAY_CUTOFF_UTC. Any
    # timestamp at or before that on an FOMC day is admissible.
    meeting = _known_fomc_day()
    on_cutoff = _dt.datetime.combine(meeting, FOMC_DAY_CUTOFF_UTC)
    assert assert_fomc_day_market_cutoff(on_cutoff) is None

    one_minute_earlier = on_cutoff - _dt.timedelta(minutes=1)
    assert assert_fomc_day_market_cutoff(one_minute_earlier) is None


def test_cutoff_raises_one_minute_after_14_00_et() -> None:
    # 14:01 ET on a meeting day. The boundary helper must reject so feature
    # assembly cannot silently leak post-announcement information.
    meeting = _known_fomc_day()
    on_cutoff = _dt.datetime.combine(meeting, FOMC_DAY_CUTOFF_UTC)
    after = on_cutoff + _dt.timedelta(minutes=1)
    with pytest.raises(ValueError, match=r"FOMC 14:00 ET cutoff"):
        assert_fomc_day_market_cutoff(after, feature_name="close_price")


def test_cutoff_accepts_naive_timestamps_as_utc() -> None:
    # Callers that pass a naive datetime get UTC by default. The cutoff
    # decision then matches the tz-aware case.
    meeting = _known_fomc_day()
    naive_before = _dt.datetime.combine(meeting, _dt.time(18, 0, 0))
    assert assert_fomc_day_market_cutoff(naive_before) is None

    naive_after = _dt.datetime.combine(meeting, _dt.time(20, 30, 0))
    with pytest.raises(ValueError):
        assert_fomc_day_market_cutoff(naive_after)
