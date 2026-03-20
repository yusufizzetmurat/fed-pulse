from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from app.services.forecaster import (  # noqa: E402
    FeatureVector,
    ModelConfig,
    build_feature_vectors,
    build_last5_sequence,
    forecast_quantitative_series,
    inspect_training_data_sources,
    parse_horizon_steps,
    train_model,
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


def test_inspect_training_data_sources_reports_usable_and_insufficient_files(tmp_path):
    usable = {
        "records": [
            {"date": f"2026-01-{idx + 1:02d}", "close": 5000 + idx * 10, "volatility_5d": 0.01 + idx * 0.0001}
            for idx in range(7)
        ]
    }
    insufficient = {
        "records": [
            {"date": "2026-02-01", "close": 5200, "volatility_5d": 0.011},
            {"date": "2026-02-02", "close": 5210, "volatility_5d": 0.0112},
        ]
    }
    (tmp_path / "usable.json").write_text(json.dumps(usable), encoding="utf-8")
    (tmp_path / "insufficient.json").write_text(json.dumps(insufficient), encoding="utf-8")

    sequences, summaries = inspect_training_data_sources(tmp_path)

    assert len(sequences) == 1
    assert len(summaries) == 2
    statuses = {summary.path.name: summary.status for summary in summaries}
    assert statuses["usable.json"] == "usable"
    assert statuses["insufficient.json"] == "insufficient"


def test_forecast_quantitative_series_fast_shape():
    out = forecast_quantitative_series(_sample_vectors(10), forecast_mode="fast", horizon="3d")
    assert "prediction" in out and "series" in out and "model" in out
    assert out["prediction"]["horizon"] == "3d"
    assert out["model"]["runtime_mode"] == "fast"
    assert "hidden_size" in out["model"]
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
    assert out["model"]["runtime_mode"] == "quick_train"
    assert out["model"]["adaptation_epochs_completed"] is not None
    assert len(out["series"]["forecast_close"]) == 5
    assert len(out["series"]["forecast_volatility"]) == 5
    assert len(out["series"]["forecast_close_lower"]) == 5
    assert len(out["series"]["forecast_close_upper"]) == 5
    assert len(out["series"]["forecast_volatility_lower"]) == 5
    assert len(out["series"]["forecast_volatility_upper"]) == 5


def test_train_model_reports_model_config_and_metrics():
    result = train_model(
        vectors=_sample_vectors(10),
        model_config=ModelConfig(hidden_size=24, num_layers=1, dropout=0.05, head_hidden_size=12),
        epochs=3,
        batch_size=4,
        save_checkpoint=False,
        device="cpu",
    )

    assert result.model.lstm.hidden_size == 24
    assert result.summary.model_config.hidden_size == 24
    assert result.summary.model_config.num_layers == 1
    assert result.summary.metrics is not None
    assert result.summary.metrics.loss >= 0.0
    assert result.summary.metrics.combined_rmse >= 0.0


def test_train_model_checkpoint_contains_training_metadata(tmp_path):
    checkpoint_path = tmp_path / "forecaster.pt"
    result = train_model(
        vectors=_sample_vectors(10),
        model_config=ModelConfig(hidden_size=20, num_layers=2, dropout=0.10, head_hidden_size=10),
        epochs=2,
        batch_size=4,
        save_checkpoint=True,
        checkpoint_path=checkpoint_path,
        device="cpu",
    )

    payload = torch.load(checkpoint_path, map_location="cpu")

    assert checkpoint_path.exists()
    assert payload["model_config"]["hidden_size"] == 20
    assert payload["training_summary"]["model_config"]["num_layers"] == 2
    assert payload["training_summary"]["metrics"]["combined_rmse"] == pytest.approx(
        result.summary.metrics.combined_rmse
    )
