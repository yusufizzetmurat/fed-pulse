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
                "checkpoint_exists": True,
                "checkpoint_loaded": True,
                "runtime_mode": "fast",
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.15,
                "head_hidden_size": 32,
                "close_scale": 10000.0,
                "sequence_length": 5,
                "best_loss": 0.0123,
                "combined_rmse": 0.0456,
                "adaptation_epochs_completed": None,
                "adaptation_best_epoch": None,
                "adaptation_loss": None,
                "adaptation_combined_rmse": None,
                "decay_rate": 0.1234,
                "chunk_attention": {
                    "chunk_count": 2,
                    "weights": [0.6, 0.4],
                    "decay_coeffs": [1.0, 0.5],
                    "chunk_previews": ["intro...", "policy..."],
                    "lambda_value": 0.1234,
                },
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
    assert payload["model"]["decay_rate"] == pytest.approx(0.1234)
    assert payload["model"]["chunk_attention"]["chunk_count"] == 2
    assert payload["model"]["chunk_attention"]["weights"] == [0.6, 0.4]
    assert payload["model"]["chunk_attention"]["lambda_value"] == pytest.approx(0.1234)


def test_analyze_rejects_unknown_field():
    """Strict request schema (#265 Phase 2) — forecast_mode is no longer
    a field, so a stale payload that includes it must 422 rather than
    silently land on the fast-mode path."""

    client = TestClient(main_mod.app)
    response = client.post(
        "/analyze",
        json={
            "text": "sample",
            "date": "2026-03-15",
            "symbol": "^GSPC",
            "forecast_mode": "fast",
            "horizon": "3d",
        },
    )
    assert response.status_code == 422


def _write_statement_fixture(tmp_path, *, date_iso: str, text: str):
    import json

    payload = [
        {
            "date": date_iso,
            "title": f"FOMC Statement {date_iso}",
            "document_type": "Statement",
            "text": text,
        }
    ]
    (tmp_path / "fomc_statements.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "fomc_minutes.json").write_text("[]", encoding="utf-8")


def test_documents_by_date_returns_cleaned_text(monkeypatch, tmp_path):
    """The read path strips Implementation Note / voting roster / nav
    chrome so consumers (frontend prefill, downstream embeddings) never
    see the boilerplate even when the on-disk JSON predates the scraper
    hygiene wiring."""

    dirty = (
        "The Committee decided to maintain the target range for the federal "
        "funds rate. Voting for the monetary policy action were Jerome H. "
        "Powell, Chair; John C. Williams, Vice Chair; and Loretta J. Mester. "
        "Implementation Note issued January 29, 2020 Federal Reserve actions "
        "to support liquidity"
    )
    _write_statement_fixture(tmp_path, date_iso="2020-01-29", text=dirty)
    monkeypatch.setattr(main_mod, "DATA_DIR", tmp_path)

    client = TestClient(main_mod.app)
    response = client.get("/documents/by-date", params={"date": "2020-01-29", "kind": "statement"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "Implementation Note" not in payload["text"]
    assert "Jerome H. Powell" not in payload["text"]
    assert "Vice Chair" not in payload["text"]
    # Substantive policy sentence is preserved.
    assert "target range for the federal funds rate" in payload["text"]


def test_documents_by_date_preserves_dissent_signal(monkeypatch, tmp_path):
    dirty = (
        "The Committee decided to lower the target range. "
        "Voting for the monetary policy action were Jerome H. Powell, Chair; "
        "and Randal K. Quarles. Voting against this action was Loretta J. "
        "Mester, who preferred to reduce the target range for the federal "
        "funds rate to 1/2 to 3/4 percent at this meeting. "
        "Implementation Note issued March 15, 2020"
    )
    _write_statement_fixture(tmp_path, date_iso="2020-03-15", text=dirty)
    monkeypatch.setattr(main_mod, "DATA_DIR", tmp_path)

    client = TestClient(main_mod.app)
    response = client.get("/documents/by-date", params={"date": "2020-03-15", "kind": "statement"})
    assert response.status_code == 200, response.text
    cleaned_text = response.json()["text"]
    # Dissent signal preserved end-to-end through the endpoint.
    assert "Voting against this action was Loretta J. Mester" in cleaned_text
    assert "preferred to reduce the target range" in cleaned_text
    # Implementation Note + the "Voting for ..." roster are gone.
    assert "Implementation Note" not in cleaned_text
    assert "Voting for the monetary policy action" not in cleaned_text
    assert "Randal K. Quarles" not in cleaned_text


def test_symbols_endpoint_returns_only_trained_symbols():
    """GET /symbols exposes exactly the five tickers the HAR / RV /
    Expected-Volume models are trained against. Anything else would
    render an "unavailable" card silently in the picker, so the asset
    universe is pinned to the trained set here, in the on-disk JSON,
    and in the in-process fallback through
    ``app.models.config.SUPPORTED_SYMBOL_METADATA``."""

    from app.models.config import SUPPORTED_SYMBOLS

    client = TestClient(main_mod.app)
    response = client.get("/symbols")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get("symbols"), list)
    returned = [entry["symbol"] for entry in payload["symbols"]]
    assert returned == list(SUPPORTED_SYMBOLS)
    assert returned == ["^GSPC", "^NDX", "^DJI", "DX-Y.NYB", "EURUSD=X"]
    for entry in payload["symbols"]:
        assert set(entry.keys()) == {"symbol", "name", "category", "default_horizon"}
        assert entry["default_horizon"] == "10d"
