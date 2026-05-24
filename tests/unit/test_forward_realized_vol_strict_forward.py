"""Pin the strict-forward window for `_forward_realized_vol` + `_volatility_shift`.

Both helpers used to include the announcement-day close-to-close return
as the first bar of their post-event window. The strict-forward
convention (matching the textbook definition of ``RV_{t+1:t+window}``)
restricts the window to bars strictly after the event so the target /
post-event feature stays disjoint from any input feature derived from
``close[t]``. These tests pin the slicing so a future refactor cannot
silently regress to the deprecated convention.
"""

from __future__ import annotations

import datetime as _dt
import math

import pytest

from app.data import event_dataset_builder as edb


def _series_with_constructed_returns(closes: list[float]) -> edb._CloseSeries:
    """Build a `_CloseSeries` whose dates are dense trading days starting 2025-01-02."""

    base = _dt.date(2025, 1, 2)
    dates: list[_dt.date] = []
    cursor = base
    while len(dates) < len(closes):
        if cursor.weekday() < 5:  # Mon-Fri
            dates.append(cursor)
        cursor += _dt.timedelta(days=1)
    return edb._CloseSeries(
        dates=dates,
        close=[float(c) for c in closes],
        volume=[0.0] * len(closes),
    )


def test_forward_realized_vol_excludes_announcement_day_return() -> None:
    """The first log return in the target window must be `log(close[t+1]/close[t])`.

    Constructed so the announcement-day return is huge (close jumps 100 → 200, a
    log-return of +0.693) and every strictly-forward return is zero (constant
    close at 200). Under the strict-forward convention the std collapses to 0;
    under the deprecated convention it would be the std of one 0.693 plus nine
    zeros (~0.219).
    """

    # Layout:
    #   idx:  0   1   2   3   4   5   6   7   8   9   10   11
    #   day: t-1   t  t+1 t+2 ...                          t+10
    # Choose enough bars before t so `_volatility_shift`'s pre-window works
    # in the sibling test below.
    pre_bars = 30
    closes = (
        [100.0] * pre_bars              # bars before t
        + [200.0]                       # close at t (announcement-day jump)
        + [200.0] * 11                  # close at t+1 .. t+10 + 1 spare
    )
    series = _series_with_constructed_returns(closes)
    as_of = series.dates[pre_bars]      # t = bar at index pre_bars
    rv = edb._forward_realized_vol(series, as_of, window=10)
    assert rv is not None
    # 10 returns, all equal to 0 (constant closes 200, 200, ..., 200)
    # std of zeros is 0.0
    assert rv == pytest.approx(0.0, abs=1e-12), (
        f"Strict-forward RV should be 0 when post-event closes are flat; "
        f"got {rv}. A non-zero value means the announcement-day return is "
        "leaking into the target window."
    )


def test_forward_realized_vol_first_return_uses_close_t_as_denominator() -> None:
    """`log(close[t+1] / close[t])` should be the first return.

    Constructed so day-t close is 100, day-t+1 close is 110 (+9.5 % log
    return), all subsequent closes are 110 (zero returns). The standard
    deviation across the 10 returns of (ln(1.1), 0, 0, ..., 0) has a
    known closed form we can pin.
    """

    pre_bars = 30
    closes = [100.0] * pre_bars + [100.0, 110.0] + [110.0] * 10
    series = _series_with_constructed_returns(closes)
    as_of = series.dates[pre_bars]
    rv = edb._forward_realized_vol(series, as_of, window=10)
    assert rv is not None

    rets = [math.log(110.0 / 100.0)] + [0.0] * 9
    n = len(rets)
    mean = sum(rets) / n
    expected = math.sqrt(sum((v - mean) ** 2 for v in rets) / (n - 1))
    assert rv == pytest.approx(expected, abs=1e-12)


def test_forward_realized_vol_returns_none_at_end_of_series() -> None:
    """A bar within ``window`` days of the end has no full forward window."""

    pre_bars = 30
    # Only 5 strictly-forward bars available after `t`; window=10 -> None.
    closes = [100.0] * pre_bars + [100.0] * 5
    series = _series_with_constructed_returns(closes)
    as_of = series.dates[pre_bars]
    assert edb._forward_realized_vol(series, as_of, window=10) is None


def test_forward_realized_vol_accepts_exactly_window_forward_bars() -> None:
    """A bar with exactly ``window`` strictly-forward bars must NOT return None.

    Strict-forward needs ``window`` returns, which needs ``window+1`` closes
    starting from ``close[t]``. A series with ``t`` at index ``N - window - 1``
    has exactly that many bars and should produce a finite value.
    """

    pre_bars = 30
    window = 10
    closes = [100.0] * pre_bars + [100.0] + [101.0] * window
    series = _series_with_constructed_returns(closes)
    as_of = series.dates[pre_bars]
    rv = edb._forward_realized_vol(series, as_of, window=window)
    assert rv is not None
    assert math.isfinite(rv)


def test_volatility_shift_excludes_announcement_day_return() -> None:
    """The post-window in `_volatility_shift` must be strict-forward too.

    Construct a series where the announcement day jumps but the pre and
    post windows are both flat. Under strict-forward the shift is exactly
    ``std(post) - std(pre) = 0 - 0 = 0``; under the deprecated
    convention the post window would include the day-t jump and produce
    a positive shift.
    """

    window = edb.VOL_WINDOW_DAYS
    pre_bars = window + 5  # plenty of room before t
    closes = (
        [100.0] * pre_bars
        + [200.0]                       # close at t (announcement-day jump)
        + [200.0] * (window + 1)        # flat post-event
    )
    series = _series_with_constructed_returns(closes)
    as_of = series.dates[pre_bars]
    shift = edb._volatility_shift(series, as_of, window=window)
    assert shift is not None
    assert shift == pytest.approx(0.0, abs=1e-12), (
        f"Strict-forward vol shift should be 0 when both flanks are flat; "
        f"got {shift}. A non-zero value means the announcement-day return "
        "is leaking into the post-event window."
    )
