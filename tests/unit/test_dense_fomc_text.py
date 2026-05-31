"""Tests for the FOMC text marginal-test helpers (pure units)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from app.data import dense_fomc_text as dft


def test_statement_dates_and_text_dedups(tmp_path) -> None:
    events = pd.DataFrame(
        {
            "event_date": ["2020-01-29"] * 3 + ["2020-03-18", "2020-04-29"],
            "event_kind": ["statement", "statement", "statement", "minutes", "statement"],
            "text": ["hawkish", "hawkish", "hawkish", "mins", "dovish"],
        }
    )
    p = tmp_path / "events.parquet"
    events.to_parquet(p, index=False)
    out = dft.statement_dates_and_text(p)
    assert out == {"2020-01-29": "hawkish", "2020-04-29": "dovish"}


def test_pca_fit_transform_shape_and_train_centering() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(50, 20))
    proj = dft._pca_fit_transform(train, train, 4)
    assert proj.shape == (50, 4)
    # whitened: each train component ~unit variance
    assert np.allclose(proj.var(axis=0), 1.0, atol=0.2)


def test_statement_delta_first_is_zero_and_range() -> None:
    emb = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    d = dft._statement_delta(emb)
    assert d[0] == 0.0
    assert d[1] == pytest.approx(1.0)  # orthogonal -> cosine 0 -> delta 1
    assert d[2] == pytest.approx(1.0)


def test_ridge_recovers_linear_signal() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 3))
    y = 0.5 + 1.2 * x[:, 0] - 0.7 * x[:, 1]
    pred = dft._ridge_fit_predict(x[:150], y[:150], x[150:], alpha=1e-3)
    # low-reg ridge on noiseless linear data tracks it closely
    ss_res = np.sum((y[150:] - pred) ** 2)
    ss_tot = np.sum((y[150:] - y[150:].mean()) ** 2)
    assert 1 - ss_res / ss_tot > 0.95
