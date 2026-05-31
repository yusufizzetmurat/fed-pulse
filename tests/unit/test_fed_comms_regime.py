"""Pure-helper tests for vol-regime classification."""

from __future__ import annotations
import numpy as np
from app.data import fed_comms_regime as r


def test_labels_tercile_digitize() -> None:
    vals = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    thr = np.quantile(vals, [1 / 3, 2 / 3])
    lab = r._labels(vals, thr)
    assert lab.min() == 0 and lab.max() == 2
    assert set(np.unique(lab)) <= {0, 1, 2}


def test_macro_f1_perfect_and_floor() -> None:
    true = np.array([0, 0, 1, 1, 2, 2])
    assert r._macro_f1(true, true.copy()) == 1.0
    # all-one-class prediction → only that class can score, macro-F1 well below 1
    allzero = np.zeros_like(true)
    assert 0.0 < r._macro_f1(true, allzero) < 0.5


def test_block_f1_gap_ci_zero_when_identical() -> None:
    true = np.tile([0, 1, 2], 40)
    a = true.copy()
    lo, hi = r._block_f1_gap_ci(true, a, a.copy(), block=5, seed=0, n_boot=200)
    assert lo == 0.0 and hi == 0.0  # identical preds → zero gap everywhere
