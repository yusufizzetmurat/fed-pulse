"""Unit tests for the late-fusion experiment harness (leak-safety helpers)."""

from __future__ import annotations

import numpy as np
import pytest

from app.data.late_fusion_experiment import (
    _pca,
    _r2,
    _residualize,
    _standardize,
    walk_forward_splits,
)


def test_walk_forward_embargo_and_disjoint() -> None:
    splits = walk_forward_splits(n=120, n_folds=5, embargo=5)
    assert splits
    for train_idx, test_idx in splits:
        # train strictly before test, with an embargo gap
        assert train_idx.max() < test_idx.min()
        assert test_idx.min() - train_idx.max() - 1 >= 5 - 1  # embargo gap honored
        # disjoint
        assert len(np.intersect1d(train_idx, test_idx)) == 0


def test_standardize_uses_train_stats_only() -> None:
    train = np.array([[0.0], [2.0], [4.0]], dtype=np.float32)  # mean 2, std ~1.633
    test = np.array([[2.0]], dtype=np.float32)
    tr, te = _standardize(train, test)
    # train mean maps to ~0
    assert tr.mean() == pytest.approx(0.0, abs=1e-5)
    # test value == train mean -> standardized to ~0 using TRAIN stats
    assert te[0, 0] == pytest.approx(0.0, abs=1e-5)


def test_pca_reduces_to_k_and_is_train_fit() -> None:
    rng = np.random.default_rng(0)
    train = rng.standard_normal((40, 20)).astype(np.float32)
    test = rng.standard_normal((10, 20)).astype(np.float32)
    tr, te = _pca(train, test, k=5)
    assert tr.shape == (40, 5)
    assert te.shape == (10, 5)


def test_residualize_removes_struct_signal_on_train() -> None:
    rng = np.random.default_rng(1)
    struct = rng.standard_normal((50, 3)).astype(np.float32)
    # text is a linear function of struct + noise
    w = rng.standard_normal((3, 4)).astype(np.float32)
    text = struct @ w + 0.01 * rng.standard_normal((50, 4)).astype(np.float32)
    tr_res, _ = _residualize(text, struct, text, struct)
    # residual should be nearly orthogonal to struct on train (signal removed)
    corr = np.abs(struct.T @ tr_res).max()
    assert corr < 1e-2


def test_r2_known_values() -> None:
    y = np.array([1.0, 2.0, 3.0])
    # perfect prediction -> R2 = 1
    assert _r2(y, y, baseline=2.0) == pytest.approx(1.0)
    # predicting the baseline mean -> R2 = 0
    assert _r2(y, np.full(3, 2.0), baseline=2.0) == pytest.approx(0.0)
