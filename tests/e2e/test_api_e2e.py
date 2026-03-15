from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("yfinance")
from fastapi.testclient import TestClient  # noqa: E402

import app.main as main_mod


def test_documents_endpoint_reads_both_sources(tmp_path, monkeypatch):
    statements = [
        {"title": "FOMC Statement A", "date": "2026-02-01", "document_type": "Statement"},
    ]
    minutes = [
        {"title": "FOMC Minutes B", "date": "2026-03-01", "document_type": "Minutes"},
    ]

    (tmp_path / "fomc_statements.json").write_text(json.dumps(statements), encoding="utf-8")
    (tmp_path / "fomc_minutes.json").write_text(json.dumps(minutes), encoding="utf-8")
    monkeypatch.setattr(main_mod, "DATA_DIR", Path(tmp_path))

    client = TestClient(main_mod.app)
    response = client.get("/documents")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["documents"][0]["date"] == "2026-03-01"
    assert payload["documents"][1]["date"] == "2026-02-01"


def test_analyze_contract_e2e_with_real_forecaster(monkeypatch):
    monkeypatch.setattr(
        main_mod,
        "analyze_text",
        lambda _: {"label": "DOVISH", "score": 0.62, "raw": [{"label": "DOVISH", "score": 0.62}]},
    )
    monkeypatch.setattr(
        main_mod,
        "fetch_market_snapshot",
        lambda **_: {
            "symbol": "^GSPC",
            "requested_date": "2026-03-15",
            "date_used": "2026-03-13",
            "lookback_days": 7,
            "close": 5560.0,
            "volatility_5d": 0.012,
        },
    )
    monkeypatch.setattr(
        main_mod,
        "fetch_market_history",
        lambda **_: [
            {"date": "2026-03-07", "close": 5450.0, "volatility_5d": 0.0130},
            {"date": "2026-03-08", "close": 5460.0, "volatility_5d": 0.0127},
            {"date": "2026-03-09", "close": 5480.0, "volatility_5d": 0.0125},
            {"date": "2026-03-10", "close": 5510.0, "volatility_5d": 0.0122},
            {"date": "2026-03-11", "close": 5525.0, "volatility_5d": 0.0121},
            {"date": "2026-03-12", "close": 5540.0, "volatility_5d": 0.0120},
            {"date": "2026-03-13", "close": 5560.0, "volatility_5d": 0.0119},
        ],
    )
    monkeypatch.setattr(main_mod, "fetch_realized_forward", lambda **_: [])

    client = TestClient(main_mod.app)
    response = client.post(
        "/analyze",
        json={
            "text": "Policy statement sample",
            "date": "2026-03-15",
            "symbol": "^GSPC",
            "horizon": "3d",
            "forecast_mode": "fast",
            "include_realized": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"sentiment", "prediction", "market", "series"}
    assert payload["prediction"]["horizon"] == "3d"
    assert len(payload["series"]["forecast_close"]) == 3
    assert len(payload["series"]["forecast_volatility"]) == 3
