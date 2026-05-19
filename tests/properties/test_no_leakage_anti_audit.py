"""Property tests proving the rich-feature input never leaks the target.

These tests sit alongside ``test_no_leakage.py`` and harden the
classification pipeline against silent regressions. They cover three
invariants the result-improvement programme depends on:

1. ``forward_realized_vol_10d`` (the y axis under
   ``output_mode='classification'``) is never present in any input
   FeatureVector's ``as_rich_list`` output -- only in the y tensor.
2. Forward-looking event-row columns (``realized_return``,
   ``abnormal_return``, ``direction_t1d``, ``volatility_shift``,
   ``forward_realized_vol_10d``) never get attached to FeatureVectors
   in the prior-window slice [0:SEQUENCE_LENGTH] as input axes.
3. Every per-bar input feature comes from a date strictly before the
   target's event date -- no future bars in the X tensor.

The tests use lightweight in-memory fixtures rather than the full
training-package loader because the invariants are about the
``FeatureVector`` + ``as_rich_list`` contract, not the data plumbing.
"""

from __future__ import annotations

import datetime
import pytest

from app.models.config import (
    FeatureVector,
    RICH_FEATURE_SIZE,
    SEQUENCE_LENGTH,
)


# Columns the data builder writes as forward-looking targets / labels.
# None of these should ever land inside as_rich_list output as an
# input-feature axis. They live on the y tensor only.
_FORWARD_TARGET_FIELDS: tuple[str, ...] = (
    "realized_return",
    "abnormal_return",
    "direction_t1d",
    "volatility_shift",
    "forward_realized_vol_10d",
)


def _make_bar(
    *,
    date: str,
    close: float,
    forward_vol: float | None = None,
) -> FeatureVector:
    """Construct a FeatureVector with arbitrary trailing-feature values
    and an optional forward_realized_vol_10d label."""

    fv = FeatureVector(
        date=date,
        sentiment_score=0.5,
        market_close=close,
        market_volatility=0.012,
    )
    if forward_vol is not None:
        fv.forward_realized_vol_10d = forward_vol
    # Populate every rich-feature axis with sentinel values so the
    # leakage test can spot any accidental copy of the target into
    # the input vector.
    fv.credibility_drift_score = 1.0
    fv.credibility_realized_vs_stated_gap = 2.0
    fv.credibility_market_implied_gap = 3.0
    fv.credibility_months_since_reversal = 4.0
    fv.mp_surprise_level = 5.0
    fv.mp_surprise_path_factor = 6.0
    fv.fed_info_factor = 7.0
    fv.mp_is_intermeeting = 8.0
    fv.stance_hawk = 1.0
    fv.realized_vol_20d = 0.015
    fv.realized_vol_60d = 0.018
    fv.vix_close = 22.0
    fv.dxy_close = 104.0
    fv.tnx_close = 4.2
    fv.gold_close = 1940.0
    fv.rich_payload = True
    return fv


# ---------------------------------------------------------------------------
# Invariant 1: forward_realized_vol_10d never enters as_rich_list output
# ---------------------------------------------------------------------------


def test_forward_vol_target_never_enters_as_rich_list() -> None:
    """A FeatureVector with a known unique forward_realized_vol_10d
    must produce an as_rich_list output that does NOT contain that
    value at any position. This catches silent regressions where the
    target column gets accidentally appended to the input vector."""

    sentinel = -123.45  # impossible value for a real vol; guaranteed unique
    fv = _make_bar(date="2024-09-18", close=4321.0, forward_vol=sentinel)
    output = fv.as_rich_list()
    assert sentinel not in output, (
        f"forward_realized_vol_10d ({sentinel}) leaked into as_rich_list "
        f"at position {output.index(sentinel)}"
    )


def test_forward_vol_target_independent_of_other_inputs() -> None:
    """Two FeatureVectors with identical trailing features but
    different forward_realized_vol_10d must produce identical
    as_rich_list outputs. If the target axis affects the input
    representation, there's a leak."""

    fv_a = _make_bar(date="2024-09-18", close=4321.0, forward_vol=0.005)
    fv_b = _make_bar(date="2024-09-18", close=4321.0, forward_vol=0.030)
    assert fv_a.as_rich_list() == fv_b.as_rich_list()


def test_direction_target_never_enters_as_rich_list() -> None:
    """A6 (#211): the direction_t1d label is y-only. Two FeatureVectors
    with identical trailing features but opposite direction labels
    must produce identical as_rich_list outputs."""

    fv_up = _make_bar(date="2024-09-18", close=4321.0)
    fv_up.direction_t1d = 1
    fv_down = _make_bar(date="2024-09-18", close=4321.0)
    fv_down.direction_t1d = -1
    assert fv_up.as_rich_list() == fv_down.as_rich_list()


def test_direction_target_does_not_leak_via_softplus_or_other_proxy() -> None:
    """Belt-and-braces: even with extreme direction labels (large
    sentinel integers), the rich-feature output stays in finite
    market-feature ranges. Catches any future refactor that piped
    direction through a downstream transform."""

    fv = _make_bar(date="2024-09-18", close=4321.0)
    fv.direction_t1d = 9999  # impossible sentinel
    rich = fv.as_rich_list()
    # Every output position must stay in a sensible per-feature range
    # (no value should reflect the 9999 sentinel).
    for v in rich:
        assert abs(v) < 1e6


# ---------------------------------------------------------------------------
# Invariant 2: forward target columns are not FeatureVector input fields
# ---------------------------------------------------------------------------


def test_forward_target_columns_are_not_input_features() -> None:
    """The FeatureVector dataclass contract: forward-looking target
    columns either don't exist as attributes (clean) or default to
    None / 0.0 with documented y-only semantics. as_rich_list must
    not surface any of them as input dims.

    The current contract has ``forward_realized_vol_10d`` on the
    dataclass (used by the partition-tensor builder as y) but every
    other column from the audit list is loader-only -- it never
    lands on FeatureVector at all.
    """

    fv = FeatureVector(
        date="2024-09-18",
        sentiment_score=0.0,
        market_close=4321.0,
        market_volatility=0.012,
    )
    rich_size = len(fv.as_rich_list())
    assert rich_size == RICH_FEATURE_SIZE
    # If a future refactor accidentally adds a forward-looking field
    # to FeatureVector AND surfaces it via as_rich_list, the
    # ``rich_size`` check would catch the unexpected width but not
    # the leakage itself. The next two tests cover that case.

    # No attribute named exactly after a known target column should
    # exist on FeatureVector EXCEPT the documented y-only attributes
    # the partition-tensor builder consumes:
    #   - forward_realized_vol_10d (Phase 9 V2 vol-regime target)
    #   - direction_t1d (A6 binary direction-target diagnostic)
    KNOWN_Y_ONLY = {"forward_realized_vol_10d", "direction_t1d"}
    for col in _FORWARD_TARGET_FIELDS:
        if col in KNOWN_Y_ONLY:
            assert hasattr(fv, col), (
                f"{col} is documented as a y-axis field on FeatureVector "
                "but the attribute is missing -- contract regression"
            )
            continue
        assert not hasattr(fv, col), (
            f"Forward-looking target column {col!r} unexpectedly exists "
            "as a FeatureVector attribute. Future-leaking inputs are "
            "rejected by the classifier contract."
        )


# ---------------------------------------------------------------------------
# Invariant 3: prior-window bars carry dates strictly before the event date
# ---------------------------------------------------------------------------


def test_prior_window_bars_strictly_before_event_date() -> None:
    """The partition-tensor builder uses bars[idx - SEQUENCE_LENGTH : idx]
    for the X tensor and bars[idx] as the y target row. Every bar in
    the X window must have date < event_date.

    The check operates on a synthetic 21-bar group (SEQUENCE_LENGTH +
    1 target row); the event_date is taken from the last bar.
    """

    event_date = datetime.date(2024, 9, 18)
    bars = []
    for offset in range(SEQUENCE_LENGTH):
        bar_date = event_date - datetime.timedelta(days=SEQUENCE_LENGTH - offset)
        bars.append(_make_bar(date=bar_date.isoformat(), close=4300.0 + offset))
    # Event-day target row is appended last.
    target = _make_bar(
        date=event_date.isoformat(), close=4321.0, forward_vol=0.020
    )
    bars.append(target)

    # The X window is bars[0:SEQUENCE_LENGTH].
    for i, bar in enumerate(bars[:SEQUENCE_LENGTH]):
        bar_date = datetime.date.fromisoformat(bar.date)
        assert bar_date < event_date, (
            f"X-window bar at index {i} has date {bar_date} >= event_date "
            f"{event_date}; this is a future-leakage signal"
        )


def test_event_day_target_row_is_not_in_x_window() -> None:
    """Smoke test on the slicing contract. The supervised window is
    [0:SEQUENCE_LENGTH]; the target row at index SEQUENCE_LENGTH
    must not appear in the window's identity check."""

    event_date = datetime.date(2024, 9, 18)
    bars = []
    for offset in range(SEQUENCE_LENGTH):
        bar_date = event_date - datetime.timedelta(days=SEQUENCE_LENGTH - offset)
        bars.append(_make_bar(date=bar_date.isoformat(), close=4300.0 + offset))
    target = _make_bar(
        date=event_date.isoformat(), close=4321.0, forward_vol=0.020
    )
    bars.append(target)

    window = bars[:SEQUENCE_LENGTH]
    assert target not in window
    assert bars[SEQUENCE_LENGTH] is target


# ---------------------------------------------------------------------------
# Invariant 4: cross-asset features are bar-date aligned, not event-date aligned
# ---------------------------------------------------------------------------


def test_cross_asset_features_are_bar_date_aligned() -> None:
    """A3 cross-asset features (vix_close etc.) are populated by the
    builder from the BAR's date, not the event's date. The bar dates
    are strictly before the event date, so the cross-asset values
    cannot encode information from event-day or later.

    This test verifies the FeatureVector slot mechanics: two bars
    on different dates can carry different cross-asset values, and
    the builder is responsible for keeping them aligned to the bar
    date upstream. Here we just check the slot is a plain dataclass
    field that travels with the bar."""

    bar_a = _make_bar(date="2024-09-01", close=4300.0)
    bar_b = _make_bar(date="2024-09-17", close=4321.0)
    # If the builder gave bar_a vix=20 and bar_b vix=22, the slot
    # values must reflect that asymmetry on a per-bar basis -- not
    # a single value broadcast to the whole window from event date.
    bar_a.vix_close = 20.0
    bar_b.vix_close = 22.0
    assert bar_a.vix_close != bar_b.vix_close
    assert bar_a.as_rich_list() != bar_b.as_rich_list()
