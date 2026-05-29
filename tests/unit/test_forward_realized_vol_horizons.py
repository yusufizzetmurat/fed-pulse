"""Multi-horizon forward realised vol generator tests (#480).

The events.parquet builder emits ``forward_realized_vol_{1,3,5,10,20,30}d``
off the same strict-forward generator (parametrised window). 10d remains
the canonical regime target; the auxiliary horizons ride alongside on
the parquet so downstream heads can be mounted without a rebuild. The
tests below pin:

1. Each window length is correct (the generator slices closes
   ``[T..T+window]`` and reports the sample-std of the log returns).
2. Cold-start (insufficient post-event data) collapses each horizon to
   ``None`` independently.
3. The 1d window's degenerate-n case is well-defined (``|log_return|``
   rather than ``None``).
"""

from __future__ import annotations

import datetime as _dt
import math

from app.data.event_dataset_builder import _CloseSeries, _forward_realized_vol


def _make_series(*, start: _dt.date, closes: list[float]) -> _CloseSeries:
    dates = [start + _dt.timedelta(days=i) for i in range(len(closes))]
    volume = [1_000_000.0] * len(closes)
    return _CloseSeries(dates=dates, close=closes, volume=volume)


def _sample_std(values: list[float]) -> float:
    n = len(values)
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5


def test_multi_horizon_window_lengths_match_sample_std() -> None:
    """For window in {3, 5, 10, 20, 30}, the generator equals std(log returns)."""

    start = _dt.date(2024, 1, 1)
    closes = [100.0 + 0.5 * i + (i % 7) * 0.3 for i in range(80)]
    series = _make_series(start=start, closes=closes)
    as_of = start + _dt.timedelta(days=20)

    # ``index_on_or_after`` lands on the bar at offset 20; the strict-
    # forward slice for window w is closes[20:20+w+1].
    t = 20
    for window in (3, 5, 10, 20, 30):
        got = _forward_realized_vol(series, as_of, window=window)
        forward_closes = closes[t : t + window + 1]
        rets = [
            math.log(forward_closes[i + 1] / forward_closes[i])
            for i in range(window)
        ]
        expected = _sample_std(rets)
        assert got is not None, f"window={window} should be defined here"
        assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-12), (
            f"window={window}: got {got!r}, expected {expected!r}"
        )


def test_window_1_collapses_to_absolute_log_return() -> None:
    """Degenerate-n case: sample std with ddof=1 is undefined for 1 obs.

    The 10d-canonical path is unaffected (always 10 returns). For the
    auxiliary ``forward_realized_vol_1d`` column the generator collapses
    to ``|log(close[T+1]/close[T])|`` so the column carries a real
    scalar instead of degenerating to ``None`` on every event.
    """

    start = _dt.date(2024, 1, 1)
    closes = [100.0, 101.5, 102.0, 99.5, 100.7]
    series = _make_series(start=start, closes=closes)
    as_of = start + _dt.timedelta(days=1)  # lands on index 1

    got = _forward_realized_vol(series, as_of, window=1)
    expected = abs(math.log(closes[2] / closes[1]))
    assert got is not None
    assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_cold_start_collapses_each_horizon_independently() -> None:
    """Each horizon degrades to None when its post-event window runs off the end."""

    start = _dt.date(2024, 1, 1)
    closes = [100.0 + i * 0.1 for i in range(15)]
    series = _make_series(start=start, closes=closes)
    # event at index 10 -- 4 strict-forward bars available (indices 11..14)
    as_of = start + _dt.timedelta(days=10)

    # window=3 needs indices 10..13 (in range -> defined)
    assert _forward_realized_vol(series, as_of, window=3) is not None
    # window=4 needs indices 10..14, the last index is valid only if
    # ``on_or_after + window < len(series)`` strictly; here
    # ``10 + 4 = 14 == len-1``, the generator requires
    # ``on_or_after + window >= len`` to return None, so 14 fails the
    # ``>= len(series)`` check -> defined.
    # Confirm boundary: window=4 is defined, window=5 is None.
    assert _forward_realized_vol(series, as_of, window=4) is not None
    assert _forward_realized_vol(series, as_of, window=5) is None
    # All the longer auxiliary horizons used by the multi-horizon
    # target columns also collapse to None at this cold-start boundary.
    assert _forward_realized_vol(series, as_of, window=10) is None
    assert _forward_realized_vol(series, as_of, window=20) is None
    assert _forward_realized_vol(series, as_of, window=30) is None


def test_10d_path_byte_identical_to_pre_480() -> None:
    """The 10d-canonical computation must not drift from the pre-#480 contract.

    The multi-horizon foundation is purely additive on the row schema
    and on the generator's window axis. The 10d target column is the
    canonical regime target under ADR 0015 + #322 and changing its
    numerical value would invalidate every promoted checkpoint. This
    test pins the contract: with 10 forward returns in scope, the
    generator returns sample-std (ddof=1).
    """

    start = _dt.date(2024, 1, 1)
    closes = [100.0 * math.exp(0.001 * i + 0.0005 * (i % 5)) for i in range(40)]
    series = _make_series(start=start, closes=closes)
    as_of = start + _dt.timedelta(days=5)
    t = 5
    forward_closes = closes[t : t + 11]
    rets = [
        math.log(forward_closes[i + 1] / forward_closes[i])
        for i in range(10)
    ]
    expected = _sample_std(rets)
    got = _forward_realized_vol(series, as_of, window=10)
    assert got is not None
    assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-12)
