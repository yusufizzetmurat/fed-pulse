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


def _write_document_detail_fixtures(tmp_path):
    """Seed all three FOMC document caches the calendar viewer reads
    against so the path-based `/documents/{type}/{date}` endpoint can
    serve every kind from a single tmp_path."""

    import json

    (tmp_path / "fomc_statements.json").write_text(
        json.dumps(
            [
                {
                    "date": "2024-09-18",
                    "title": "Federal Reserve issues FOMC statement",
                    "document_type": "Statement",
                    "text": (
                        "The Committee decided to lower the target range for the "
                        "federal funds rate. Implementation Note issued September 18, 2024"
                    ),
                    "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240918a.htm",
                    "scraped_at_utc": "2026-05-30T00:00:00+00:00",
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "fomc_minutes.json").write_text(
        json.dumps(
            [
                {
                    "date": "2024-09-18",
                    "title": "FOMC Minutes 2024-09-18",
                    "document_type": "Minutes",
                    "text": "Minutes body discussing the September meeting in depth.",
                    "url": "https://www.federalreserve.gov/monetarypolicy/fomcminutes20240918.htm",
                    "scraped_at_utc": "2026-05-30T00:00:00+00:00",
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "press_conferences.json").write_text(
        json.dumps(
            [
                {
                    "date": "2024-09-18",
                    "title": "September 17-18, 2024 FOMC Meeting",
                    "document_type": "press_conference",
                    "text": "Chair Powell press conference transcript body.",
                    "url": "https://www.federalreserve.gov/monetarypolicy/fomcpresconf20240918.htm",
                    "scraped_at_utc": "2026-05-30T00:00:00+00:00",
                }
            ]
        ),
        encoding="utf-8",
    )


def test_document_detail_serves_statement_with_hygiene(monkeypatch, tmp_path):
    _write_document_detail_fixtures(tmp_path)
    monkeypatch.setattr(main_mod, "DATA_DIR", tmp_path)

    client = TestClient(main_mod.app)
    response = client.get("/documents/statement/2024-09-18")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["type"] == "statement"
    assert payload["date"] == "2024-09-18"
    assert payload["title"] == "Federal Reserve issues FOMC statement"
    # Hygiene strips the Implementation Note tail.
    assert "Implementation Note" not in payload["cleaned_text"]
    assert "target range for the federal funds rate" in payload["cleaned_text"]
    assert payload["source_url"].endswith("monetary20240918a.htm")
    assert payload["scraped_at"] == "2026-05-30T00:00:00+00:00"


def test_document_detail_serves_minutes_and_press_conference(monkeypatch, tmp_path):
    _write_document_detail_fixtures(tmp_path)
    monkeypatch.setattr(main_mod, "DATA_DIR", tmp_path)

    client = TestClient(main_mod.app)

    minutes = client.get("/documents/minutes/2024-09-18")
    assert minutes.status_code == 200
    assert minutes.json()["type"] == "minutes"
    assert "September meeting" in minutes.json()["cleaned_text"]

    press = client.get("/documents/press_conference/2024-09-18")
    assert press.status_code == 200
    assert press.json()["type"] == "press_conference"
    assert "press conference transcript" in press.json()["cleaned_text"]


def test_document_detail_404_when_missing(monkeypatch, tmp_path):
    _write_document_detail_fixtures(tmp_path)
    monkeypatch.setattr(main_mod, "DATA_DIR", tmp_path)

    client = TestClient(main_mod.app)
    response = client.get("/documents/statement/1999-01-01")
    assert response.status_code == 404


def test_document_detail_422_on_unknown_type(monkeypatch, tmp_path):
    _write_document_detail_fixtures(tmp_path)
    monkeypatch.setattr(main_mod, "DATA_DIR", tmp_path)

    client = TestClient(main_mod.app)
    response = client.get("/documents/speech/2024-09-18")
    assert response.status_code == 422


def test_load_rv_history_skips_spx_parquet_for_non_gspc_symbols(monkeypatch, tmp_path):
    """The SPX intraday RV parquet must only feed ^GSPC.

    Earlier the parquet branch fired whenever the file existed,
    regardless of the requested symbol, so a ^DJI / AAPL request would
    silently surface SPX rv tagged to the wrong ticker. The gate now
    routes everything other than ^GSPC through the yfinance fallback.
    """

    import pandas as pd
    from app.data import intraday_realized as intraday_mod

    parquet_path = tmp_path / "spx_5min_daily_rv.parquet"
    dates = pd.date_range("2026-02-01", periods=30, freq="B").strftime("%Y-%m-%d")
    pd.DataFrame({"date": dates, "rv": [1e-4 + i * 1e-6 for i in range(30)]}).to_parquet(
        parquet_path
    )
    monkeypatch.setattr(intraday_mod, "DEFAULT_RV_PARQUET", parquet_path)

    yf_calls: list[str] = []

    class _StubFrame:
        empty = False

        def __init__(self) -> None:
            import numpy as np

            self.index = pd.date_range("2026-02-01", periods=80, freq="B")
            close = pd.Series(100.0 + np.arange(80) * 0.5, index=self.index)
            self._df = pd.DataFrame({"Close": close})

        def __getitem__(self, key):
            return self._df[key]

    class _StubTicker:
        def __init__(self, symbol: str) -> None:
            yf_calls.append(symbol)

        def history(self, **_: object) -> _StubFrame:
            return _StubFrame()

    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", _StubTicker)

    # ^GSPC reads the parquet (no yfinance call).
    spx_rv, spx_dates = main_mod._load_rv_history("^GSPC")
    assert yf_calls == []
    assert spx_dates[-1] == dates[-1]
    assert spx_rv[-1] == pytest.approx(1e-4 + 29e-6)

    # ^DJI bypasses the parquet and falls back to yfinance.
    dji_rv, dji_dates = main_mod._load_rv_history("^DJI")
    assert yf_calls == ["^DJI"]
    assert dji_rv != spx_rv
    assert dji_dates != spx_dates


def test_forecast_realized_vol_surfaces_historical_bands(monkeypatch):
    """The /forecast/realized-vol endpoint surfaces walk-forward h=1
    bands so the VolatilityOutlookCard can render a "we covered" overlay
    behind the realized sparkline."""

    from app.services import rv_forecaster

    dates = [f"2026-03-{i + 1:02d}" for i in range(30)]
    rv = [1e-4 + i * 1e-6 for i in range(30)]
    monkeypatch.setattr(main_mod, "_load_rv_history", lambda symbol: (rv, dates))
    monkeypatch.setattr(
        rv_forecaster,
        "predict_rv",
        lambda hist: {
            "horizons": [
                {
                    "h": 1,
                    "point": 1.5e-4,
                    "band_lo_80": 7e-5,
                    "band_hi_80": 2.5e-4,
                    "band_lo_90": 6e-5,
                    "band_hi_90": 3e-4,
                    "qlike_model": 0.2,
                    "qlike_har": 0.25,
                    "coverage_empirical_90": 0.9,
                },
                {
                    "h": 5,
                    "point": 1.6e-4,
                    "band_lo_80": 8e-5,
                    "band_hi_80": 2.6e-4,
                    "band_lo_90": 7e-5,
                    "band_hi_90": 3.1e-4,
                    "qlike_model": 0.21,
                    "qlike_har": 0.26,
                    "coverage_empirical_90": 0.91,
                },
                {
                    "h": 22,
                    "point": 1.8e-4,
                    "band_lo_80": 9e-5,
                    "band_hi_80": 2.8e-4,
                    "band_lo_90": 8e-5,
                    "band_hi_90": 3.3e-4,
                    "qlike_model": 0.32,
                    "qlike_har": 0.36,
                    "coverage_empirical_90": 0.92,
                },
            ],
            "model_revision": "stub@2026-05-29",
        },
    )
    monkeypatch.setattr(
        rv_forecaster,
        "predict_rv_historical_bands",
        lambda hist, dts: [
            {
                "date": dts[i],
                "band_lo_80": 5e-5,
                "band_hi_80": 2e-4,
                "realized_rv": float(hist[i]),
            }
            for i in range(22, len(hist))
        ],
    )

    client = TestClient(main_mod.app)
    response = client.get("/forecast/realized-vol", params={"symbol": "^GSPC"})
    assert response.status_code == 200, response.text
    body = response.json()
    bands = body["historical_bands"]
    assert isinstance(bands, list)
    assert len(bands) == len(rv) - 22
    assert bands[0]["date"] == dates[22]
    assert bands[0]["band_lo_80"] == pytest.approx(5e-5)
    assert bands[0]["band_hi_80"] == pytest.approx(2e-4)
    assert bands[0]["realized_rv"] == pytest.approx(rv[22])


def test_analyze_does_not_carry_historical_bands(monkeypatch):
    """The /analyze response does not carry the walk-forward RV bands.

    Bands are owned by /forecast/realized-vol, the only consumer the
    frontend reads. Producing them on every /analyze call would pay the
    cost (parquet read + 38 ensemble forwards, plus a possible yfinance
    round-trip) for a field that AnalyzeResponse does not declare and no
    UI surface reads. Pinning this here so a future change that wires
    bands back into /analyze must also wire them onto the schema in the
    same commit.
    """

    captured: dict = {}

    def fake_record(request_payload, response_payload):
        captured["payload"] = response_payload

    monkeypatch.setattr(main_mod, "_record_history", fake_record)
    monkeypatch.setattr(
        main_mod,
        "analyze_text",
        lambda _: {"label": "neutral", "score": 0.0, "raw": []},
    )
    monkeypatch.setattr(
        main_mod,
        "fetch_market_snapshot",
        lambda **_: {
            "symbol": "^GSPC",
            "requested_date": "2026-03-23",
            "date_used": "2026-03-23",
            "lookback_days": 7,
            "close": 5600.0,
            "volatility_5d": 0.01,
        },
    )
    monkeypatch.setattr(
        main_mod,
        "fetch_market_history",
        lambda **_: [
            {"date": "2026-03-22", "close": 5580.0, "volatility_5d": 0.011},
            {"date": "2026-03-23", "close": 5600.0, "volatility_5d": 0.010},
        ],
    )
    monkeypatch.setattr(
        main_mod,
        "forecast_quantitative_series",
        lambda **_: {
            "prediction": {"close": 5605.0, "volatility": 0.011, "horizon": "3d"},
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
                "timestamps": ["2026-03-22", "2026-03-23"],
                "history_close": [5580.0, 5600.0],
                "history_volatility": [0.011, 0.01],
                "forecast_timestamps": ["2026-03-23+1"],
                "forecast_close": [5605.0],
                "forecast_close_lower": [5589.0],
                "forecast_close_upper": [5621.0],
                "forecast_volatility": [0.0115],
                "forecast_volatility_lower": [0.0110],
                "forecast_volatility_upper": [0.0120],
                "forecast_confidence_level": 0.8,
                "volatility_scale": {"suggested_ymin": 0.0, "suggested_ymax": 0.02},
            },
        },
    )
    monkeypatch.setattr(main_mod, "parse_horizon_steps", lambda _: 3)

    client = TestClient(main_mod.app)
    response = client.post(
        "/analyze",
        json={
            "text": "sample",
            "date": "2026-03-23",
            "symbol": "^GSPC",
            "horizon": "3d",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "historical_bands" not in body
    assert "payload" in captured, "history hook was not invoked"
    persisted = captured["payload"]
    assert "historical_bands" not in persisted


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


def test_fomc_calendar_exposes_text_availability_flags(monkeypatch, tmp_path):
    """Each calendar row carries three booleans that mirror whether the
    corresponding text record is present in the on-disk JSON caches.
    Statements and minutes are populated for the September 2024 meeting,
    the press conference is intentionally absent, and the November 2024
    meeting has nothing on file at all — the badges in the UI need this
    tri-source signal so the user can see at a glance what's collected."""

    import json

    sep_release = "2024-09-18"  # statement_release_date for 2024-09-17 meeting
    nov_release = "2024-11-07"  # statement_release_date for 2024-11-06 meeting

    (tmp_path / "fomc_statements.json").write_text(
        json.dumps(
            [
                {
                    "date": sep_release,
                    "title": f"FOMC Statement {sep_release}",
                    "document_type": "Statement",
                    "text": "stub",
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "fomc_minutes.json").write_text(
        json.dumps(
            [
                {
                    "date": sep_release,
                    "title": f"FOMC Minutes {sep_release}",
                    "document_type": "Minutes",
                    "text": "stub",
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "press_conferences.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(main_mod, "DATA_DIR", tmp_path)

    client = TestClient(main_mod.app)
    response = client.get(
        "/fomc/calendar",
        params={"as_of": "2024-11-06", "past_limit": 3, "upcoming_limit": 3},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["past"] and body["upcoming"]

    by_meeting: dict[str, dict[str, object]] = {}
    for row in body["past"] + body["upcoming"]:
        for key in (
            "statement_available",
            "minutes_available",
            "press_conference_available",
        ):
            assert isinstance(row[key], bool), row
        by_meeting[row["meeting_date"]] = row

    sep = by_meeting["2024-09-17"]
    assert sep["statement_release_date"] == sep_release
    assert sep["statement_available"] is True
    assert sep["minutes_available"] is True
    assert sep["press_conference_available"] is False

    nov = by_meeting["2024-11-06"]
    assert nov["statement_release_date"] == nov_release
    assert nov["statement_available"] is False
    assert nov["minutes_available"] is False
    assert nov["press_conference_available"] is False


def test_fomc_calendar_availability_matches_on_disk_caches():
    """Sanity-check that the production JSON caches under ``data/`` line
    up with the calendar flags for a handful of recent meetings. Pinning
    a few known-good releases here guards against silent drift between
    the schedule and the scraped text."""

    import json
    from pathlib import Path

    data_dir = Path(main_mod.DATA_DIR)
    statement_path = data_dir / "fomc_statements.json"
    minutes_path = data_dir / "fomc_minutes.json"
    presser_path = data_dir / "press_conferences.json"
    if not (statement_path.exists() and minutes_path.exists() and presser_path.exists()):
        pytest.skip("on-disk text caches not present in this environment")

    statement_dates = {
        row["date"] for row in json.loads(statement_path.read_text(encoding="utf-8"))
    }
    minutes_dates = {
        row["date"] for row in json.loads(minutes_path.read_text(encoding="utf-8"))
    }
    presser_dates = {
        row["date"] for row in json.loads(presser_path.read_text(encoding="utf-8"))
    }

    client = TestClient(main_mod.app)
    response = client.get(
        "/fomc/calendar",
        params={"as_of": "2026-01-01", "past_limit": 6, "upcoming_limit": 1},
    )
    assert response.status_code == 200
    body = response.json()
    sampled = 0
    for row in body["past"]:
        release = row.get("statement_release_date")
        if not isinstance(release, str):
            continue
        assert row["statement_available"] == (release in statement_dates), row
        assert row["minutes_available"] == (release in minutes_dates), row
        assert row["press_conference_available"] == (release in presser_dates), row
        sampled += 1
    assert sampled >= 3
