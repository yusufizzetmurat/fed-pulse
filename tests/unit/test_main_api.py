from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("yfinance")
from fastapi.testclient import TestClient  # noqa: E402

import app.main as main_mod


def test_analyze_happy_path_with_realized_overlay(monkeypatch):
    monkeypatch.setattr(
        main_mod,
        "analyze_text",
        lambda _: {"label": "POSITIVE", "score": 0.77, "raw": [{"label": "POSITIVE", "score": 0.77}]},
    )
    monkeypatch.setattr(
        main_mod,
        "fetch_market_snapshot",
        lambda **_: {
            "symbol": "^GSPC",
            "requested_date": "2026-03-15",
            "date_used": "2026-03-13",
            "lookback_days": 7,
            "close": 5600.0,
            "volatility_5d": 0.01,
        },
    )
    monkeypatch.setattr(
        main_mod,
        "fetch_market_history",
        lambda **_: [
            {"date": "2026-03-12", "close": 5580.0, "volatility_5d": 0.011},
            {"date": "2026-03-13", "close": 5600.0, "volatility_5d": 0.010},
        ],
    )
    monkeypatch.setattr(
        main_mod,
        "forecast_quantitative_series",
        lambda **_: {
            "prediction": {"close": 5610.0, "volatility": 0.012, "horizon": "3d"},
            "model": {
                "checkpoint_path": "backend/models/forecaster_best.pt",
                "checkpoint_loaded": True,
                "runtime_mode": "fast",
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.15,
                "head_hidden_size": 32,
                "best_loss": 0.0123,
                "combined_rmse": 0.0456,
                "adaptation_epochs_completed": None,
                "adaptation_best_epoch": None,
                "adaptation_loss": None,
                "adaptation_combined_rmse": None,
            },
            "series": {
                "timestamps": ["2026-03-12", "2026-03-13"],
                "history_close": [5580.0, 5600.0],
                "history_volatility": [0.011, 0.01],
                "forecast_timestamps": ["2026-03-13+1", "2026-03-13+2", "2026-03-13+3"],
                "forecast_close": [5605.0, 5608.0, 5610.0],
                "forecast_close_lower": [5589.0, 5588.0, 5587.0],
                "forecast_close_upper": [5621.0, 5628.0, 5633.0],
                "forecast_volatility": [0.0115, 0.0118, 0.012],
                "forecast_volatility_lower": [0.0110, 0.0111, 0.0112],
                "forecast_volatility_upper": [0.0120, 0.0125, 0.0128],
                "forecast_confidence_level": 0.8,
                "volatility_scale": {"suggested_ymin": 0.0, "suggested_ymax": 0.02},
            },
        },
    )
    monkeypatch.setattr(main_mod, "parse_horizon_steps", lambda _: 3)
    monkeypatch.setattr(
        main_mod,
        "fetch_realized_forward",
        lambda **_: [
            {"date": "2026-03-14", "close": 5606.0, "volatility_5d": 0.0112},
            {"date": "2026-03-15", "close": 5607.0, "volatility_5d": 0.0114},
            {"date": "2026-03-16", "close": 5609.0, "volatility_5d": 0.0117},
        ],
    )

    client = TestClient(main_mod.app)
    response = client.post(
        "/analyze",
        json={
            "text": "sample",
            "date": "2026-03-15",
            "symbol": "^GSPC",
            "forecast_mode": "fast",
            "horizon": "3d",
            "include_realized": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "sentiment" in payload and "prediction" in payload and "market" in payload and "series" in payload
    assert "model" in payload
    assert payload["series"]["realized_timestamps"] == ["2026-03-14", "2026-03-15", "2026-03-16"]
    assert payload["series"]["forecast_confidence_level"] == 0.8
    assert payload["model"]["checkpoint_loaded"] is True


def test_analyze_invalid_mode_returns_422():
    client = TestClient(main_mod.app)
    response = client.post(
        "/analyze",
        json={
            "text": "sample",
            "date": "2026-03-15",
            "symbol": "^GSPC",
            "forecast_mode": "bad_mode",
            "horizon": "3d",
        },
    )
    assert response.status_code == 422
