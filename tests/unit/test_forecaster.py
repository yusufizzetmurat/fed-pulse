from __future__ import annotations

import pytest

pytest.importorskip("torch")

from app.services.forecaster import (  # noqa: E402
    FeatureVector,
    build_feature_vectors,
    build_last5_sequence,
    forecast_quantitative_series,
    parse_horizon_steps,
)


def _sample_vectors(n: int = 8) -> list[FeatureVector]:
    return [
        FeatureVector(
            date=f"2026-01-{idx + 1:02d}",
            sentiment_score=0.6,
            market_close=5000 + idx * 10,
            market_volatility=0.01 + idx * 0.0002,
        )
        for idx in range(n)
    ]


def test_parse_horizon_steps():
    assert parse_horizon_steps("1d") == 1
    assert parse_horizon_steps("10d") == 10
    assert parse_horizon_steps("invalid") == 3


def test_build_last5_sequence_padding():
    seq = build_last5_sequence(_sample_vectors(2), length=5)
    assert len(seq) == 5
    assert seq[0].date == "2026-01-01"


def test_build_feature_vectors_derives_change_signals():
    vectors = build_feature_vectors(
        [
            {"date": "2026-01-01", "close": 5000.0, "volatility_5d": 0.0100, "sentiment_score": 0.4},
            {"date": "2026-01-02", "close": 5100.0, "volatility_5d": 0.0115, "sentiment_score": 0.5},
        ]
    )
    assert len(vectors) == 2
    assert vectors[0].close_change_pct == 0.0
    assert vectors[0].volatility_change == 0.0
    assert vectors[1].close_change_pct == pytest.approx(0.02)
    assert vectors[1].volatility_change == pytest.approx(0.0015)


def test_forecast_quantitative_series_fast_shape():
    out = forecast_quantitative_series(_sample_vectors(10), forecast_mode="fast", horizon="3d")
    assert "prediction" in out and "series" in out
    assert out["prediction"]["horizon"] == "3d"
    assert len(out["series"]["forecast_close"]) == 3
    assert len(out["series"]["forecast_close_lower"]) == 3
    assert len(out["series"]["forecast_close_upper"]) == 3
    assert len(out["series"]["forecast_volatility"]) == 3
    assert len(out["series"]["forecast_volatility_lower"]) == 3
    assert len(out["series"]["forecast_volatility_upper"]) == 3
    assert len(out["series"]["timestamps"]) == 10
    assert out["series"]["forecast_confidence_level"] == 0.8
    assert all(
        lower <= point <= upper
        for lower, point, upper in zip(
            out["series"]["forecast_close_lower"],
            out["series"]["forecast_close"],
            out["series"]["forecast_close_upper"],
        )
    )
    assert "volatility_scale" in out["series"]


def test_forecast_quantitative_series_quick_train_shape():
    out = forecast_quantitative_series(_sample_vectors(12), forecast_mode="quick_train", horizon="5d")
    assert out["prediction"]["horizon"] == "5d"
    assert len(out["series"]["forecast_close"]) == 5
    assert len(out["series"]["forecast_volatility"]) == 5
    assert len(out["series"]["forecast_close_lower"]) == 5
    assert len(out["series"]["forecast_close_upper"]) == 5
    assert len(out["series"]["forecast_volatility_lower"]) == 5
    assert len(out["series"]["forecast_volatility_upper"]) == 5
