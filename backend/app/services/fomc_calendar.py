from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class FomcMeeting:
    meeting_date: date
    meeting_type: str  # "scheduled" | "unscheduled"
    statement_release_date: date | None
    minutes_release_date: date | None
    notes: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "meeting_date": self.meeting_date.isoformat(),
            "meeting_type": self.meeting_type,
            "statement_release_date": (
                self.statement_release_date.isoformat() if self.statement_release_date else None
            ),
            "minutes_release_date": (
                self.minutes_release_date.isoformat() if self.minutes_release_date else None
            ),
            "notes": self.notes,
        }


def _m(meeting: tuple[int, int, int], statement_offset_days: int = 0) -> FomcMeeting:
    meeting_date = date(*meeting)
    statement = date.fromordinal(meeting_date.toordinal() + statement_offset_days)
    minutes = date.fromordinal(meeting_date.toordinal() + 21)
    return FomcMeeting(
        meeting_date=meeting_date,
        meeting_type="scheduled",
        statement_release_date=statement,
        minutes_release_date=minutes,
    )


# Federal Reserve published FOMC meeting calendar. Two-day meetings; the statement
# release falls on the second day. Minutes publish three weeks later.
# Sourced from federalreserve.gov/monetarypolicy/fomccalendars.htm.
_SCHEDULE: tuple[FomcMeeting, ...] = (
    _m((2023, 1, 31), 1),
    _m((2023, 3, 21), 1),
    _m((2023, 5, 2), 1),
    _m((2023, 6, 13), 1),
    _m((2023, 7, 25), 1),
    _m((2023, 9, 19), 1),
    _m((2023, 10, 31), 1),
    _m((2023, 12, 12), 1),
    _m((2024, 1, 30), 1),
    _m((2024, 3, 19), 1),
    _m((2024, 4, 30), 1),
    _m((2024, 6, 11), 1),
    _m((2024, 7, 30), 1),
    _m((2024, 9, 17), 1),
    _m((2024, 11, 6), 1),
    _m((2024, 12, 17), 1),
    _m((2025, 1, 28), 1),
    _m((2025, 3, 18), 1),
    _m((2025, 4, 29), 1),
    _m((2025, 6, 17), 1),
    _m((2025, 7, 29), 1),
    _m((2025, 9, 16), 1),
    _m((2025, 10, 28), 1),
    _m((2025, 12, 9), 1),
    _m((2026, 1, 27), 1),
    _m((2026, 3, 17), 1),
    _m((2026, 4, 28), 1),
    _m((2026, 6, 16), 1),
    _m((2026, 7, 28), 1),
    _m((2026, 9, 15), 1),
    _m((2026, 10, 27), 1),
    _m((2026, 12, 8), 1),
    # 2027 schedule per federalreserve.gov/monetarypolicy/fomccalendars.htm.
    # Two-day meetings; statement releases on day two; minutes default to
    # ~21 days after the meeting start. Each date is tentative until
    # confirmed at the meeting immediately preceding it.
    _m((2027, 1, 26), 1),
    _m((2027, 3, 16), 1),
    _m((2027, 4, 27), 1),
    _m((2027, 6, 8), 1),
    _m((2027, 7, 27), 1),
    _m((2027, 9, 14), 1),
    _m((2027, 10, 26), 1),
    _m((2027, 12, 7), 1),
)


def get_calendar(
    *,
    as_of: date | None = None,
    upcoming_limit: int = 12,
    past_limit: int = 12,
) -> dict[str, list[FomcMeeting]]:
    reference = as_of or date.today()
    past = [m for m in _SCHEDULE if m.meeting_date < reference]
    upcoming = [m for m in _SCHEDULE if m.meeting_date >= reference]
    past.sort(key=lambda m: m.meeting_date, reverse=True)
    upcoming.sort(key=lambda m: m.meeting_date)
    return {
        "past": past[:past_limit],
        "upcoming": upcoming[:upcoming_limit],
    }


def list_all_meetings() -> tuple[FomcMeeting, ...]:
    return _SCHEDULE
