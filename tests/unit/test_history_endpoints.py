from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient  # noqa: E402

import app.db as db_module  # noqa: E402
import app.main as main_mod  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    db_module.reset_for_testing(f"sqlite:///{tmp_path / 'fed_pulse_history.db'}")
    return TestClient(main_mod.app)


def _seed(session, *, symbol="^GSPC", stance="HAWKISH", document_date="2024-09-18"):
    return db_module.persist_analysis_run(
        session,
        payload={
            "sentiment": {"label": stance, "score": 0.7, "raw": []},
            "prediction": {"close": 5050.0, "volatility": 0.012, "horizon": "3d"},
        },
        request={
            "text": "Recent indicators…",
            "date": document_date,
            "symbol": symbol,
            "horizon": "3d",
            "forecast_mode": "fast",
            "include_realized": False,
        },
        response={
            "sentiment": {"label": stance, "score": 0.7, "raw": []},
            "prediction": {"close": 5050.0, "volatility": 0.012, "horizon": "3d"},
            "market": {
                "symbol": symbol,
                "requested_date": document_date,
                "date_used": document_date,
                "lookback_days": 5,
                "close": 5000.0,
                "volatility_5d": 0.011,
            },
            "model": {},
            "series": {},
        },
    )


def test_history_endpoints_round_trip(client):
    session_iter = db_module.get_session()
    sess = next(session_iter)
    try:
        first = _seed(sess)
        _seed(sess, symbol="^NDX")
        _seed(sess, stance="DOVISH")
    finally:
        sess.close()

    response = client.get("/history")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert len(body["items"]) == 3

    filtered = client.get("/history", params={"symbol": "^NDX"})
    assert filtered.status_code == 200
    assert filtered.json()["total"] == 1

    detail = client.get(f"/history/{first.id}")
    assert detail.status_code == 200
    assert detail.json()["id"] == first.id
    assert "payload" in detail.json()

    deletion = client.delete(f"/history/{first.id}")
    assert deletion.status_code == 204

    missing = client.get(f"/history/{first.id}")
    assert missing.status_code == 404


def test_history_filter_by_stance_and_date(client):
    session_iter = db_module.get_session()
    sess = next(session_iter)
    try:
        _seed(sess, stance="HAWKISH", document_date="2024-09-18")
        _seed(sess, stance="DOVISH", document_date="2024-11-06")
    finally:
        sess.close()

    response = client.get("/history", params={"stance": "dovish"})
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["items"][0]["stance"] == "dovish"

    by_date = client.get("/history", params={"document_date": "2024-09-18"})
    assert by_date.status_code == 200
    assert by_date.json()["total"] == 1


def test_fomc_calendar_endpoint_returns_past_and_upcoming(client):
    response = client.get("/fomc/calendar", params={"as_of": "2024-09-18"})
    assert response.status_code == 200
    body = response.json()
    assert body["upcoming"]
    assert body["past"]
    assert body["upcoming"][0]["meeting_date"] >= "2024-09-18"
    assert body["past"][0]["meeting_date"] < "2024-09-18"


def test_fomc_calendar_validates_as_of(client):
    response = client.get("/fomc/calendar", params={"as_of": "not-a-date"})
    assert response.status_code == 422
