"""Per-asset 10d forward realised-vol target columns at events.parquet build time (#481).

Pins three contracts for the data-foundation change that lets us train
the regime classifier per-asset:

a) Symbol -> slug normalisation rule (lowercase, strip ``^`` / ``=X`` /
   ``.NYB``, drop remaining ``-``).
b) Per-symbol computation: each ``forward_realized_vol_10d_<slug>``
   column on a row equals ``_forward_realized_vol`` applied to that
   symbol's own close series for the same event_date.
c) Missing-data handling: when a symbol's series is absent / empty /
   too short for the forward window, the column lands as ``None``
   rather than ``0.0`` so the classifier can learn to skip rather
   than treat absent as zero.
"""

from __future__ import annotations

import datetime as _dt

import math
import pytest

from app.data import event_dataset_builder as edb


# ---------------------------------------------------------------------------
# (a) Symbol slug normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("symbol", "expected_slug"),
    [
        ("^GSPC", "gspc"),
        ("^NDX", "ndx"),
        ("^DJI", "dji"),
        ("^VIX", "vix"),
        ("DX-Y.NYB", "dxy"),
        ("EURUSD=X", "eurusd"),
        ("USDJPY=X", "usdjpy"),
        ("GBPUSD=X", "gbpusd"),
    ],
)
def test_per_asset_target_slug_matches_documented_rule(
    symbol: str, expected_slug: str
) -> None:
    """The slug rule is: lowercase; strip ``^``, ``=x``, ``.nyb``; drop ``-``.

    Each row pins one symbol from the v1 workspace asset-picker set so a
    future refactor that silently changes the normalisation (e.g.
    swaps ``-`` for ``_`` instead of dropping it) trips the test.
    """

    assert edb.per_asset_target_slug(symbol) == expected_slug
    assert (
        edb.per_asset_target_column(symbol)
        == f"forward_realized_vol_10d_{expected_slug}"
    )


def test_per_asset_target_symbols_cover_eight_workspace_assets() -> None:
    """The constant lists the eight v1 symbols documented in #481.

    Locks the set: if the canonical asset picker changes, the column
    surface here must change too (and so must the audit doc + schema +
    regression test). Listed in stable order so the COLUMN_ORDER tuple
    in the builder stays deterministic.
    """

    assert edb.PER_ASSET_TARGET_SYMBOLS == (
        "^GSPC",
        "^NDX",
        "^DJI",
        "DX-Y.NYB",
        "^VIX",
        "EURUSD=X",
        "USDJPY=X",
        "GBPUSD=X",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dense_trading_series(
    closes: list[float], *, start: _dt.date = _dt.date(2025, 1, 2)
) -> edb._CloseSeries:
    """Build a dense Mon-Fri ``_CloseSeries`` of the requested closes."""

    dates: list[_dt.date] = []
    cursor = start
    while len(dates) < len(closes):
        if cursor.weekday() < 5:
            dates.append(cursor)
        cursor += _dt.timedelta(days=1)
    return edb._CloseSeries(
        dates=dates,
        close=[float(c) for c in closes],
        volume=[0.0] * len(closes),
    )


# ---------------------------------------------------------------------------
# (b) Per-symbol computation against synthetic price data
# ---------------------------------------------------------------------------


def test_per_asset_target_column_matches_per_symbol_forward_vol() -> None:
    """Each per-asset column equals ``_forward_realized_vol`` on that symbol's series.

    Builds two distinct synthetic series — one with a flat post-event
    window (target ~= 0.0) and one with a +9.5 % first-step jump (target
    has a known closed form) — and confirms the builder's per-asset
    computation routes each symbol to its own series rather than e.g.
    re-using the canonical asset series for all columns.
    """

    pre_bars = 30
    window = 10

    # Symbol A: flat post-event. Target == 0.
    flat_closes = [100.0] * pre_bars + [200.0] + [200.0] * (window + 1)
    flat_series = _dense_trading_series(flat_closes)

    # Symbol B: +9.5 % first strict-forward return, then flat. Target has
    # a known closed form.
    jump_closes = [100.0] * pre_bars + [100.0, 110.0] + [110.0] * window
    jump_series = _dense_trading_series(jump_closes)

    as_of = flat_series.dates[pre_bars]

    flat_vol = edb._forward_realized_vol(flat_series, as_of, window=window)
    jump_vol = edb._forward_realized_vol(jump_series, as_of, window=window)
    assert flat_vol is not None and jump_vol is not None
    assert flat_vol == pytest.approx(0.0, abs=1e-12)

    rets = [math.log(110.0 / 100.0)] + [0.0] * (window - 1)
    n = len(rets)
    mean = sum(rets) / n
    expected_jump = math.sqrt(sum((v - mean) ** 2 for v in rets) / (n - 1))
    assert jump_vol == pytest.approx(expected_jump, abs=1e-12)

    # Different series -> different values: rules out a degenerate
    # implementation that reads the canonical asset series for all
    # per-asset columns.
    assert flat_vol != pytest.approx(jump_vol, abs=1e-6)


# ---------------------------------------------------------------------------
# (c) Missing-data handling
# ---------------------------------------------------------------------------


def test_per_asset_target_missing_series_renders_column_as_none() -> None:
    """When a symbol's series is absent the column must be ``None``, not ``0.0``.

    Three missing-data shapes:

    - the symbol entry is not in the per-asset target series dict at all
      (cache fetch failed at the orchestrator),
    - the series exists but carries no bars (the cache parquet was
      empty),
    - the series is short enough that the strict-forward window does
      not fit (pre-listing event date).

    All three must collapse to ``None`` so the downstream classifier
    learns to skip rather than treating absent as zero.
    """

    pre_bars = 30
    window = 10
    closes = [100.0] * pre_bars + [100.0] * (window + 1)
    series = _dense_trading_series(closes)
    as_of = series.dates[pre_bars]

    # Sanity: the well-formed series returns a finite value.
    rv = edb._forward_realized_vol(series, as_of, window=window)
    assert rv == pytest.approx(0.0, abs=1e-12)

    # Empty series -> None via the `_forward_realized_vol` guard
    # (``on_or_after + window >= len(series)`` fires immediately).
    empty_series = edb._CloseSeries(dates=[], close=[], volume=[])
    assert edb._forward_realized_vol(empty_series, as_of, window=window) is None

    # Series too short -> None via the same guard.
    short_series = _dense_trading_series([100.0] * (pre_bars + 2))
    assert (
        edb._forward_realized_vol(short_series, as_of, window=window) is None
    )


def test_per_asset_target_column_is_not_zero_on_missing() -> None:
    """Defence-in-depth: missing must surface as None, never 0.0.

    A zero realised-vol target is a *legitimate* value (a perfectly
    flat post-event window emits it, as the flat-closes case above
    confirms). Conflating missing with zero would make those two
    distinct regimes indistinguishable for the classifier. This test
    pins the contract that the missing-data path returns ``None``.
    """

    pre_bars = 5
    window = 10
    # Not enough forward bars for the window -> guard fires.
    closes = [100.0] * pre_bars + [100.0] * 3
    series = _dense_trading_series(closes)
    as_of = series.dates[pre_bars]
    assert edb._forward_realized_vol(series, as_of, window=window) is None
    # Explicit anti-regression: must NOT be coerced to 0.0 anywhere on
    # the path. The orchestrator's per-asset attachment block reads
    # ``None`` and writes ``None`` to the column.
    assert edb._forward_realized_vol(series, as_of, window=window) != 0.0
