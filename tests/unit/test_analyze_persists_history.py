from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("torch")
pytest.importorskip("transformers")

from fastapi.testclient import TestClient  # noqa: E402

import app.main as main_mod  # noqa: E402


def _stub_market_path(monkeypatch):
    monkeypatch.setattr(
        main_mod,
        "analyze_text",
        lambda _: {
            "label": "HAWKISH",
            "score": 0.81,
            "raw": [{"label": "HAWKISH", "score": 0.81}],
        },
    )
    monkeypatch.setattr(
        main_mod,
        "fetch_market_snapshot",
        lambda **_: {
            "symbol": "^GSPC",
            "requested_date": "2026-03-15",
            "date_used": "2026-03-13",
            "lookback_days": 5,
            "close": 5000.0,
            "volatility_5d": 0.01,
        },
    )
    monkeypatch.setattr(
        main_mod,
        "fetch_market_history",
        lambda **_: [
            {"date": "2026-03-12", "close": 4980.0, "volatility_5d": 0.011},
            {"date": "2026-03-13", "close": 5000.0, "volatility_5d": 0.010},
        ],
    )
    monkeypatch.setattr(main_mod, "parse_horizon_steps", lambda _: 3)
    monkeypatch.setattr(main_mod, "fetch_forward_trading_dates", lambda **_: ["2026-03-16", "2026-03-17", "2026-03-18"])
    monkeypatch.setattr(
        main_mod,
        "forecast_quantitative_series",
        lambda **_: {
            "prediction": {"close": 5050.0, "volatility": 0.012, "horizon": "3d"},
            "model": {
                "checkpoint_path": "backend/models/forecaster_best.pt",
                "checkpoint_exists": True,
                "checkpoint_loaded": True,
                "runtime_mode": "fast",
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.15,
                "head_hidden_size": 32,
                "close_scale": 10000.0,
                "sequence_length": 5,
            },
            "series": {
                "timestamps": ["2026-03-12", "2026-03-13"],
                "history_close": [4980.0, 5000.0],
                "history_volatility": [0.011, 0.01],
                "forecast_timestamps": ["2026-03-16", "2026-03-17", "2026-03-18"],
                "forecast_close": [5020.0, 5040.0, 5050.0],
                "forecast_close_lower": [5000.0, 5015.0, 5020.0],
                "forecast_close_upper": [5040.0, 5060.0, 5080.0],
                "forecast_volatility": [0.011, 0.012, 0.012],
                "forecast_volatility_lower": [0.009, 0.010, 0.010],
                "forecast_volatility_upper": [0.013, 0.014, 0.015],
                "forecast_confidence_level": 0.8,
                "volatility_scale": {"suggested_ymin": 0.0, "suggested_ymax": 0.02},
            },
        },
    )


def test_analyze_persists_a_history_row_visible_via_get_history(monkeypatch):
    _stub_market_path(monkeypatch)
    monkeypatch.setattr(main_mod, "checkpoint_exists", lambda: True)
    client = TestClient(main_mod.app)
    pre = client.get("/history").json()
    assert pre["total"] == 0

    response = client.post(
        "/analyze",
        json={
            "text": "Recent indicators…",
            "date": "2026-03-15",
            "symbol": "^GSPC",
            "horizon": "3d",
            "include_realized": False,
        },
    )
    assert response.status_code == 200

    listing = client.get("/history").json()
    assert listing["total"] == 1
    row = listing["items"][0]
    assert row["symbol"] == "^GSPC"
    assert row["stance"] == "hawkish"
    assert row["predicted_close"] == pytest.approx(5050.0)
    assert row["document_date"] == "2026-03-15"
