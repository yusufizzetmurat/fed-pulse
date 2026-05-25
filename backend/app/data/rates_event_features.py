"""Event-row helpers for the rates-complex heads (#291).

Two families of features:

- **Forward change targets** (strict-forward, in basis points). For each
  FOMC event at date ``t`` and each yield column ``y``, the 5-day target
  is ``yield_y[t+5] - yield_y[t]`` measured in raw bps (``× 100`` since
  FRED publishes yields as percent). Strict-forward semantics match the
  convention pinned by ``tests/unit/test_forward_realized_vol_strict_forward.py``
  for the existing realized-vol target: ``yield_y[t]`` is the close-of-day
  yield on the FOMC event date, ``yield_y[t+5]`` is the close 5 trading
  days later, and ``yield_y[t-1]`` never appears in the target window.
  This keeps the target measuring the *post-announcement drift* and not
  the announcement-day reaction, which the surprise-decomposition issue
  (#305) treats separately.

- **Pre-meeting expectation features** (strict-backward at ``t-1``). For
  each FOMC event the values are computed only from FRED observations
  with publication date strictly before the meeting date. A 5-day
  trailing yield change uses ``yield[t-1] - yield[t-6]`` in bps. The
  time-since-last-rate-change feature walks the DFEDTARU step series
  backward from ``t-1`` and counts trading days. The implied next-move
  proxy uses ``(1y Treasury - upper Fed Funds target)`` at ``t-1``,
  bucketed into hike / cut / pause probabilities under a 25-bps
  standard-move threshold.

The FRED-only implied-probability bucketing is the conservative baseline
for #291. The CME 30-day Fed Funds futures replacement is reserved for
#305 (surprise decomposition); the column names here are pinned so the
upgrade is a drop-in replacement.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.data.rates_panel import RatesPanelLookup


# Standard FOMC move increment, in basis points. Used by
# :func:`implied_move_probabilities` to bucket the 1y-implied next-move
# proxy into hike / cut / pause probabilities. The 12.5-bps half-move
# threshold is the canonical hike/cut detection boundary used by
# Bloomberg's World Interest Rate Probability product and by the
# fed-funds-futures-implied probability literature (Kuttner 2001, GSS
# 2005). The threshold is symmetric: ``|implied| < 12.5`` => pause;
# ``|implied| in [12.5, 25)`` => fractional hike or cut; ``|implied| >= 25``
# => full hike or cut.
STANDARD_MOVE_BPS: float = 25.0
HALF_MOVE_BPS: float = 12.5

# Columns covered by the strict-forward target family.
FORWARD_TARGET_COLUMNS: tuple[tuple[str, str], ...] = (
    # (rates_panel column, output target column name)
    ("treas_2y", "yield_2y_change_5d"),
    ("treas_5y", "yield_5y_change_5d"),
    # 1y constant-maturity Treasury proxies the terminal rate when CME
    # Fed Funds futures are unavailable. The column name keeps the
    # ``terminal_rate_*`` convention so the futures-based replacement
    # (#305) drops in without a schema migration.
    ("treas_1y", "terminal_rate_change_5d"),
)

# Columns covered by the strict-backward pre-meeting yield level family.
PRE_MEETING_LEVEL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("treas_1y", "pre_meeting_yield_1y"),
    ("treas_2y", "pre_meeting_yield_2y"),
    ("treas_5y", "pre_meeting_yield_5y"),
    ("treas_10y", "pre_meeting_yield_10y"),
    ("slope_10y_2y", "pre_meeting_slope_10y_2y"),
    ("slope_10y_3m", "pre_meeting_slope_10y_3m"),
)


# ---------------------------------------------------------------------------
# Trading-day arithmetic
# ---------------------------------------------------------------------------


def trading_day_offset(
    trading_calendar: Sequence[_dt.date],
    anchor: _dt.date,
    *,
    offset: int,
) -> _dt.date | None:
    """Return ``anchor`` shifted forward by ``offset`` trading days.

    The "first trading day on or after the anchor" is treated as position
    0 (the anchor itself when it is a trading day; otherwise the next
    one). A positive offset moves forward, a negative offset moves
    backward. Returns ``None`` when the resulting position falls outside
    the calendar.
    """

    import bisect as _bisect

    if not trading_calendar:
        return None

    idx_on_or_after = _bisect.bisect_left(trading_calendar, anchor)
    if idx_on_or_after >= len(trading_calendar):
        # Anchor sits past the last calendar day; fall back to the
        # last calendar entry as the reference point for forward
        # offsets that would never resolve.
        return None
    target_idx = idx_on_or_after + offset
    if target_idx < 0 or target_idx >= len(trading_calendar):
        return None
    return trading_calendar[target_idx]


def last_trading_day_strictly_before(
    trading_calendar: Sequence[_dt.date],
    anchor: _dt.date,
) -> _dt.date | None:
    """Return the largest calendar date strictly less than ``anchor``."""

    import bisect as _bisect

    if not trading_calendar:
        return None
    idx = _bisect.bisect_left(trading_calendar, anchor)
    if idx == 0:
        return None
    return trading_calendar[idx - 1]


# ---------------------------------------------------------------------------
# Forward change targets (strict-forward, bps)
# ---------------------------------------------------------------------------


def forward_yield_change_bps(
    lookup: "RatesPanelLookup",
    trading_calendar: Sequence[_dt.date],
    *,
    column: str,
    event_date: _dt.date,
    horizon: int = 5,
) -> float | None:
    """Strict-forward ``t+horizon`` yield change in basis points.

    Returns ``(yield[t+horizon] - yield[t]) * 100`` where ``t`` is the
    first trading day on or after ``event_date`` (the announcement-day
    close that already reflects the FOMC reaction) and ``t+horizon`` is
    the close ``horizon`` trading days later. Both endpoints are looked
    up via :meth:`RatesPanelLookup.yield_on_or_before`. Snapping the
    baseline forward to the next trading day matches the convention
    used by ``app.data.event_dataset_builder._forward_realized_vol``,
    which is pinned by ``test_forward_realized_vol_strict_forward.py``.

    Returns ``None`` when either endpoint is unavailable or when the
    calendar runs out before ``t+horizon``.
    """

    t = trading_day_offset(trading_calendar, event_date, offset=0)
    if t is None:
        return None
    base = lookup.yield_on_or_before(column, t)
    if base is None:
        return None

    t_plus_h = trading_day_offset(trading_calendar, event_date, offset=horizon)
    if t_plus_h is None:
        return None
    endpoint = lookup.yield_on_or_before(column, t_plus_h)
    if endpoint is None:
        return None
    return (endpoint - base) * 100.0


# ---------------------------------------------------------------------------
# Pre-meeting expectation features (strict-backward at t-1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreMeetingFeatures:
    """Bundle of strict-backward pre-meeting expectation features.

    Every field is computed from FRED observations with publication date
    strictly before the FOMC event date. A field is ``None`` when the
    underlying series has no observation before the event date (the
    earliest events in 2008 fall in this bucket for DFEDTARU before
    its FRED start date).
    """

    pre_meeting_yield_1y: float | None
    pre_meeting_yield_2y: float | None
    pre_meeting_yield_5y: float | None
    pre_meeting_yield_10y: float | None
    pre_meeting_slope_10y_2y: float | None
    pre_meeting_slope_10y_3m: float | None
    pre_meeting_trailing_2y_yield_change_5d_bps: float | None
    pre_meeting_implied_next_move_bps: float | None
    pre_meeting_implied_hike_prob: float | None
    pre_meeting_implied_cut_prob: float | None
    pre_meeting_implied_pause_prob: float | None
    pre_meeting_days_since_last_rate_change: int | None


def trailing_yield_change_bps(
    lookup: "RatesPanelLookup",
    trading_calendar: Sequence[_dt.date],
    *,
    column: str,
    event_date: _dt.date,
    horizon: int = 5,
) -> float | None:
    """Strict-backward trailing yield change in basis points.

    Returns ``(yield[t-1] - yield[t-1-horizon]) * 100`` where ``t-1`` is
    the last trading day strictly before ``event_date`` and ``t-1-horizon``
    is the close ``horizon`` trading days earlier. Both endpoints use
    :meth:`RatesPanelLookup.yield_strictly_before` so neither leaks any
    observation with publication date ``>= event_date``.

    Returns ``None`` when either endpoint is unavailable.
    """

    anchor = last_trading_day_strictly_before(trading_calendar, event_date)
    if anchor is None:
        return None

    endpoint = lookup.yield_strictly_before(column, event_date)
    if endpoint is None:
        return None

    earlier_anchor = trading_day_offset(
        trading_calendar, anchor, offset=-horizon
    )
    if earlier_anchor is None:
        return None
    # ``yield_strictly_before`` reads pub_date < target; ``earlier_anchor + 1``
    # is the smallest date that still admits the earlier_anchor publication.
    base = lookup.yield_strictly_before(column, earlier_anchor + _dt.timedelta(days=1))
    if base is None:
        return None
    return (endpoint - base) * 100.0


def days_since_last_rate_change(
    lookup: "RatesPanelLookup",
    event_date: _dt.date,
    *,
    column: str = "ff_target_upper",
    tolerance_bps: float = 0.5,
    max_lookback_days: int = 730,
) -> int | None:
    """Calendar days since the most recent change in ``ff_target_upper``.

    Walks the FRED DFEDTARU step series backward from the last observation
    strictly before ``event_date`` and returns the calendar-day gap to
    the first prior observation whose value differs by more than
    ``tolerance_bps`` (in percent: 0.5 bps == 0.005 percent). Returns
    ``None`` when no change is found within ``max_lookback_days`` or when
    the series has no observations before ``event_date``.

    ``tolerance_bps`` defaults to 0.5 bp to ignore micro-rounding in the
    published rate series; FOMC moves are 25 bps minimum, so the
    threshold is conservative.
    """

    dates = lookup.dates_by_column.get(column)
    values = lookup.values_by_column.get(column)
    if not dates or not values:
        return None

    import bisect as _bisect

    idx = _bisect.bisect_left(dates, event_date)
    if idx == 0:
        return None

    current_value = values[idx - 1]
    tolerance_pct = tolerance_bps / 100.0
    earliest_pos = idx - 1
    # Walk backward looking for the first observation differing from
    # ``current_value`` by more than the tolerance. The change date is
    # the observation *after* that one (the first one matching the new
    # level), so the gap is ``current_date - change_date``.
    walk_pos = idx - 2
    last_change_idx: int | None = None
    while walk_pos >= 0:
        if abs(values[walk_pos] - current_value) > tolerance_pct:
            last_change_idx = walk_pos + 1
            break
        walk_pos -= 1
    if last_change_idx is None:
        return None

    gap = (dates[earliest_pos] - dates[last_change_idx]).days
    # ``last_change_idx`` is the first observation with the *new* value;
    # the gap measures how many days the new level has been in place.
    if gap > max_lookback_days:
        return None
    return int(gap)


def implied_next_move_bps(
    lookup: "RatesPanelLookup",
    event_date: _dt.date,
) -> float | None:
    """FRED-only implied next-move proxy at ``t-1`` in basis points.

    Computed as ``(1y Treasury yield - upper Fed Funds target) * 100``
    using strict-backward observations at the last trading day before
    ``event_date``. A positive value indicates the 1y curve prices net
    hikes; a negative value prices net cuts.

    This is the conservative FRED-only baseline; #305 (surprise
    decomposition) replaces it with a CME 30-day Fed Funds futures
    construction.

    Returns ``None`` when either input is unavailable (FF target is
    absent before 2008-12-16 when DFEDTARU starts).
    """

    yield_1y = lookup.yield_strictly_before("treas_1y", event_date)
    ff_upper = lookup.yield_strictly_before("ff_target_upper", event_date)
    if yield_1y is None or ff_upper is None:
        return None
    return (yield_1y - ff_upper) * 100.0


def implied_move_probabilities(
    implied_bps: float | None,
) -> tuple[float | None, float | None, float | None]:
    """Bucket the implied next-move proxy into hike / cut / pause probs.

    Returns ``(hike_prob, cut_prob, pause_prob)``. Each probability is in
    ``[0, 1]`` and the three values sum to 1.0 within floating-point
    tolerance.

    Bucketing rules with :data:`STANDARD_MOVE_BPS` = 25 bps and
    :data:`HALF_MOVE_BPS` = 12.5 bps:

    - ``|implied| < 12.5`` => pause_prob = 1, hike = cut = 0
    - ``+12.5 <= implied < +25`` => hike_prob = (implied - 12.5)/12.5,
      pause_prob = 1 - hike_prob, cut = 0
    - ``implied >= +25`` => hike_prob = 1, pause = cut = 0
    - mirror cases for negative implied values

    Returns ``(None, None, None)`` when ``implied_bps`` is ``None``.
    """

    if implied_bps is None:
        return (None, None, None)

    abs_bps = abs(implied_bps)
    if abs_bps < HALF_MOVE_BPS:
        return (0.0, 0.0, 1.0)

    if abs_bps >= STANDARD_MOVE_BPS:
        directional = 1.0
    else:
        # Linearly ramp from 0 at ``HALF_MOVE_BPS`` to 1 at
        # ``STANDARD_MOVE_BPS``.
        directional = (abs_bps - HALF_MOVE_BPS) / (STANDARD_MOVE_BPS - HALF_MOVE_BPS)

    pause = 1.0 - directional
    if implied_bps > 0:
        return (directional, 0.0, pause)
    return (0.0, directional, pause)


def compute_pre_meeting_features(
    lookup: "RatesPanelLookup",
    trading_calendar: Sequence[_dt.date],
    *,
    event_date: _dt.date,
) -> PreMeetingFeatures:
    """Bundle every pre-meeting expectation feature for one FOMC event."""

    levels: dict[str, float | None] = {}
    for column, output_name in PRE_MEETING_LEVEL_COLUMNS:
        levels[output_name] = lookup.yield_strictly_before(column, event_date)

    trailing_2y = trailing_yield_change_bps(
        lookup, trading_calendar, column="treas_2y", event_date=event_date, horizon=5
    )
    implied_bps = implied_next_move_bps(lookup, event_date)
    hike_prob, cut_prob, pause_prob = implied_move_probabilities(implied_bps)
    days_since = days_since_last_rate_change(lookup, event_date)

    return PreMeetingFeatures(
        pre_meeting_yield_1y=levels["pre_meeting_yield_1y"],
        pre_meeting_yield_2y=levels["pre_meeting_yield_2y"],
        pre_meeting_yield_5y=levels["pre_meeting_yield_5y"],
        pre_meeting_yield_10y=levels["pre_meeting_yield_10y"],
        pre_meeting_slope_10y_2y=levels["pre_meeting_slope_10y_2y"],
        pre_meeting_slope_10y_3m=levels["pre_meeting_slope_10y_3m"],
        pre_meeting_trailing_2y_yield_change_5d_bps=trailing_2y,
        pre_meeting_implied_next_move_bps=implied_bps,
        pre_meeting_implied_hike_prob=hike_prob,
        pre_meeting_implied_cut_prob=cut_prob,
        pre_meeting_implied_pause_prob=pause_prob,
        pre_meeting_days_since_last_rate_change=days_since,
    )
