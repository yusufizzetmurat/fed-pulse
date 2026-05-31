"""Smoke test for the fusion training helpers."""
from __future__ import annotations
import numpy as np
from app.data import fed_comms_train as t


def test_standardize_train_stats() -> None:
    x_tr = np.array([[1.0, 10.0], [3.0, 30.0]])
    x_te = np.array([[2.0, 20.0]])
    s_tr, s_te = t._standardize(x_tr, x_te)
    assert np.allclose(s_tr.mean(0), 0.0)
    assert np.allclose(s_te[0], 0.0)  # test midpoint maps to train mean → 0
    # zero-variance column must not divide by zero
    z_tr, z_te = t._standardize(np.array([[5.0], [5.0]]), np.array([[5.0]]))
    assert np.isfinite(z_tr).all() and np.isfinite(z_te).all()
