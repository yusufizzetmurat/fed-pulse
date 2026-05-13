from __future__ import annotations

from datetime import date

from app.services.fomc_calendar import get_calendar, list_all_meetings


def test_meeting_schedule_is_chronologically_ordered():
    meetings = list_all_meetings()
    dates = [meeting.meeting_date for meeting in meetings]
    assert dates == sorted(dates)
    assert len(meetings) >= 24  # at least 3 years × 8 meetings


def test_get_calendar_splits_past_and_upcoming():
    calendar = get_calendar(as_of=date(2024, 9, 18), upcoming_limit=4, past_limit=4)
    assert len(calendar["upcoming"]) == 4
    assert len(calendar["past"]) == 4
    assert all(m.meeting_date < date(2024, 9, 18) for m in calendar["past"])
    assert all(m.meeting_date >= date(2024, 9, 18) for m in calendar["upcoming"])
    assert calendar["upcoming"][0].meeting_date == date(2024, 11, 6)


def test_meeting_serialises_to_iso_dict():
    meeting = list_all_meetings()[0]
    payload = meeting.to_dict()
    assert payload["meeting_date"] == meeting.meeting_date.isoformat()
    assert payload["meeting_type"] in {"scheduled", "unscheduled"}
    assert payload["statement_release_date"] is not None
