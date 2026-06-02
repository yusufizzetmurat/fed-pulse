"""Tests for the intraday direction training harness (pure units)."""

from __future__ import annotations

import math

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from app.data import intraday_direction_train as idt


def test_bar_features_shape_and_first_return_zero() -> None:
    close = [100.0, 101.0, 100.0, 102.0]
    volume = [10.0, 20.0, 30.0, 40.0]
    feats = idt._bar_features(close, volume)
    assert feats.shape == (4, 3)
    assert feats[0, 0] == 0.0
    assert feats[1, 0] == pytest.approx(math.log(101.0 / 100.0))
    assert feats[2, 1] == pytest.approx(math.log(101.0 / 100.0) + math.log(100.0 / 101.0))


def test_bar_features_volume_z_is_zero_mean() -> None:
    feats = idt._bar_features([100.0, 101.0, 102.0], [10.0, 20.0, 30.0])
    assert feats[:, 2].mean() == pytest.approx(0.0, abs=1e-9)


def test_walk_forward_folds_are_expanding_and_future_only() -> None:
    folds = idt._walk_forward_folds(n=41, n_folds=4)
    assert len(folds) == 4
    for train_idx, test_idx in folds:
        assert max(train_idx) < min(test_idx)
        assert train_idx == list(range(len(train_idx)))
        assert all(0 <= i < 41 for i in test_idx)
    assert [len(t) for t, _ in folds] == sorted(len(t) for t, _ in folds)
    covered = [i for _, te in folds for i in te]
    assert len(covered) == len(set(covered))


def test_walk_forward_too_few_events_raises() -> None:
    with pytest.raises(ValueError, match="too few"):
        idt._walk_forward_folds(n=3, n_folds=4)


def test_directional_accuracy() -> None:
    assert idt._accuracy([1, 0, 1, 1], [1, 0, 0, 1]) == pytest.approx(0.75)


def test_majority_baseline_uses_train_majority() -> None:
    acc = idt._majority_baseline_accuracy(train_y=[0, 0, 0, 1], test_y=[0, 0, 0, 1])
    assert acc == pytest.approx(0.75)


def test_bootstrap_ci_brackets_point() -> None:
    correct = [1] * 30 + [0] * 10
    lo, point, hi = idt._bootstrap_ci(correct, n_boot=200, seed=11)
    assert point == pytest.approx(0.75)
    assert lo <= point <= hi
    assert 0.0 <= lo and hi <= 1.0


def test_standardize_fits_on_train_only() -> None:
    x = np.array([[[0.0], [2.0]], [[10.0], [12.0]]])
    xs = idt._standardize_per_fold(x, [0])
    assert xs[0, 0, 0] == pytest.approx(-1.0)
    assert xs[0, 1, 0] == pytest.approx(1.0)
    assert xs[1, 0, 0] == pytest.approx((10.0 - 1.0) / 1.0)


def test_build_arrays_shapes() -> None:
    df = pd.DataFrame(
        {
            "event_date": ["2024-03-20", "2024-01-31"],
            "text": ["dovish", "hawkish"],
            "pre_close": [[100.0, 101.0, 102.0], [50.0, 49.0, 50.0]],
            "pre_volume": [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
            "dir_immediate": [1, 0],
            "dir_delayed": [0, 1],
        }
    )
    x, text, y = idt.build_arrays(
        df, "immediate", embed_fn=lambda s: np.zeros(768, dtype=np.float32)
    )
    assert x.shape == (2, 3, 3)
    assert text.shape == (2, 768)
    assert y.shape == (2,)
    # sorted by event_date: 2024-01-31 first -> dir_immediate 0
    assert y.tolist() == [0, 1]
