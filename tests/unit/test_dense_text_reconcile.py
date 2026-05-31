"""Tests for the text-reconciliation pure helpers."""

from __future__ import annotations
import numpy as np
import pytest
from app.data import dense_text_reconcile as dtr


def test_regime_thresholds_and_labels() -> None:
    y = np.arange(30, dtype=float)
    lo, hi = dtr._regime_thresholds(y)
    lab = dtr._to_regime(y, (lo, hi))
    assert set(lab.tolist()) == {0, 1, 2}
    assert (lab[:8] == 0).all() and (lab[-8:] == 2).all()


def test_macro_f1_perfect_and_chance() -> None:
    t = np.array([0, 0, 1, 1, 2, 2])
    assert dtr._macro_f1(t.copy(), t) == pytest.approx(1.0)
    assert 0.0 <= dtr._macro_f1(np.zeros(6, dtype=int), t) <= 0.5


def test_oos_r2() -> None:
    t = np.array([1.0, 2, 3, 4])
    assert dtr._oos_r2(t.copy(), t, 2.5) == pytest.approx(1.0)
    assert dtr._oos_r2(np.full(4, 2.5), t, 2.5) == pytest.approx(0.0)
