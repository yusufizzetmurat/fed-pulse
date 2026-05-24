"""Cover the per-bank slicing helpers on the multi-axis classifier (D).

The classifier's eval pass now reports per-source macro-F1 alongside
the existing per-axis loss. Bootstrap CIs around those numbers are
filed as a follow-up; this file pins the slicing contract + the
inline macro-F1 helper.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.data.train_text_multi_axis_classifier import (
    _AxisRow,
    _macro_f1_from_arrays,
)


def test_axis_row_carries_source_default_empty() -> None:
    """Default constructor leaves ``source`` empty so existing call
    sites that pre-date the per-bank tracking keep working without
    being forced to specify the source explicitly."""

    row = _AxisRow(text="hello")
    assert row.source == ""


def test_axis_row_round_trips_source() -> None:
    row = _AxisRow(text="hello", source="gtfintechlab/federal_reserve_system")
    assert row.source == "gtfintechlab/federal_reserve_system"


def test_macro_f1_returns_zero_on_empty_input() -> None:
    assert _macro_f1_from_arrays([], [], n_classes=3) == 0.0


def test_macro_f1_returns_one_on_perfect_predictions() -> None:
    preds = [0, 1, 2, 0, 1, 2]
    tgts = [0, 1, 2, 0, 1, 2]
    assert _macro_f1_from_arrays(preds, tgts, n_classes=3) == pytest.approx(1.0)


def test_macro_f1_balanced_when_class_collapses() -> None:
    """All-zero predictions on a balanced target → only class 0 has
    nonzero F1; macro should average to ~1/3."""

    preds = [0] * 9
    tgts = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    f1 = _macro_f1_from_arrays(preds, tgts, n_classes=3)
    # class 0: precision=3/9, recall=3/3 -> F1=2*1/3*1/(1/3+1)=0.5
    # classes 1+2: zero
    # macro = 0.5/3
    assert f1 == pytest.approx(0.5 / 3.0, abs=1e-4)


def test_macro_f1_drops_mismatched_lengths() -> None:
    """Defensive contract: mismatched arrays return 0.0 rather than
    raising; the caller is expected to feed aligned rows."""

    assert _macro_f1_from_arrays([0, 1], [0, 1, 2], n_classes=3) == 0.0
