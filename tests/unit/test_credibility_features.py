from __future__ import annotations

import pytest

from app.features.credibility import (
    CredibilityVector,
    compute_credibility_vector,
    cosine_distance,
    drift_vs_prior,
    market_implied_gap,
    months_since_last_reversal,
    realized_vs_stated_gap,
)


def test_cosine_distance_basic_cases():
    assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert cosine_distance([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0)
    # Zero vectors and mismatched dims degrade to 0.0 cleanly.
    assert cosine_distance([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine_distance([1.0, 0.0], [1.0]) == 0.0


def test_drift_vs_prior_returns_zero_with_no_prior_context():
    assert drift_vs_prior([1.0, 0.0], []) == 0.0


def test_drift_vs_prior_grows_when_current_diverges_from_mean():
    prior = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    same = drift_vs_prior([1.0, 0.0, 0.0], prior)
    rotated = drift_vs_prior([0.0, 1.0, 0.0], prior)
    flipped = drift_vs_prior([-1.0, 0.0, 0.0], prior)
    assert same == pytest.approx(0.0)
    assert rotated == pytest.approx(1.0)
    assert flipped == pytest.approx(2.0)


def test_months_since_last_reversal_on_stable_stance():
    # Monotonically hawkish — no reversal in the window.
    assert months_since_last_reversal([0.6, 0.5, 0.7, 0.4]) == 4


def test_months_since_last_reversal_on_recent_flip():
    # Last flip is between the last two entries (dovish to hawkish).
    assert months_since_last_reversal([0.6, 0.5, 0.4, -0.2]) == 1


def test_months_since_last_reversal_short_history():
    assert months_since_last_reversal([0.5]) == 0
    assert months_since_last_reversal([]) == 0


def test_realized_vs_stated_gap_returns_pearson_correlation():
    # Perfectly correlated → 1.0
    stated = [0.1, 0.2, 0.3, 0.4]
    realized = [1.0, 2.0, 3.0, 4.0]
    assert realized_vs_stated_gap(stated, realized) == pytest.approx(1.0)
    # Perfectly anti-correlated → -1.0
    assert realized_vs_stated_gap(stated, list(reversed(realized))) == pytest.approx(-1.0)


def test_realized_vs_stated_gap_handles_short_series():
    assert realized_vs_stated_gap([0.1], [0.2]) == 0.0


def test_market_implied_gap_scales_and_clips():
    # 4pp gap → +1.0; -2pp gap → -0.5; missing inputs → 0.0.
    assert market_implied_gap(5.5, 1.5) == pytest.approx(1.0)
    assert market_implied_gap(2.0, 4.0) == pytest.approx(-0.5)
    assert market_implied_gap(None, 1.0) == 0.0
    assert market_implied_gap(1.0, None) == 0.0


def test_compute_credibility_vector_aggregates_all_four_axes():
    vector = compute_credibility_vector(
        current_embedding=[1.0, 0.0, 0.0],
        prior_embeddings=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        stance_history=[0.6, 0.5, 0.4, -0.2],
        stated_path=[0.1, 0.2, 0.3, 0.4],
        realized_path=[1.0, 2.0, 3.0, 4.0],
        sep_terminal=5.0,
        ois_terminal=4.0,
    )
    assert isinstance(vector, CredibilityVector)
    assert vector.drift_score == pytest.approx(0.0)
    assert vector.realized_vs_stated_gap == pytest.approx(1.0)
    assert vector.market_implied_gap == pytest.approx(0.25)
    assert vector.months_since_reversal == 1
    assert vector.as_list() == [0.0, 1.0, 0.25, 1.0]


def test_compute_credibility_vector_degrades_gracefully_when_inputs_missing():
    vector = compute_credibility_vector()
    assert vector.as_list() == [0.0, 0.0, 0.0, 0.0]
