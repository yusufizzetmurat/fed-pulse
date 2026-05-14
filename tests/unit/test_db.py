from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")

from app import db as db_module


@pytest.fixture()
def session(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'fed_pulse_test.db'}"
    engine = db_module.reset_for_testing(database_url)
    assert engine is not None
    session_iter = db_module.get_session()
    sess = next(session_iter)
    try:
        yield sess
    finally:
        sess.close()


def _sample_request(**overrides):
    base = {
        "text": "Recent indicators suggest economic activity has continued to expand.",
        "date": "2024-09-18",
        "symbol": "^GSPC",
        "horizon": "3d",
        "forecast_mode": "fast",
        "include_realized": False,
    }
    base.update(overrides)
    return base


def _sample_response(**overrides):
    base = {
        "sentiment": {"label": "HAWKISH", "score": 0.81, "raw": []},
        "prediction": {"close": 5050.0, "volatility": 0.012, "horizon": "3d"},
        "market": {
            "symbol": "^GSPC",
            "requested_date": "2024-09-18",
            "date_used": "2024-09-18",
            "lookback_days": 5,
            "close": 5000.0,
            "volatility_5d": 0.011,
        },
        "model": {},
        "series": {},
    }
    base.update(overrides)
    return base


def test_persist_and_list_runs(session):
    db_module.persist_analysis_run(
        session,
        payload=_sample_response(),
        request=_sample_request(),
        response=_sample_response(),
    )
    db_module.persist_analysis_run(
        session,
        payload=_sample_response(),
        request=_sample_request(symbol="^NDX"),
        response=_sample_response(market={
            "symbol": "^NDX",
            "requested_date": "2024-09-18",
            "date_used": "2024-09-18",
            "lookback_days": 5,
            "close": 19000.0,
            "volatility_5d": 0.014,
        }),
    )
    rows, total = db_module.list_runs(session, limit=10, offset=0)
    assert total == 2
    assert len(rows) == 2

    filtered, total = db_module.list_runs(session, limit=10, offset=0, symbol="^NDX")
    assert total == 1
    assert filtered[0].symbol == "^NDX"


def test_delete_run_returns_true_when_row_exists(session):
    record = db_module.persist_analysis_run(
        session,
        payload=_sample_response(),
        request=_sample_request(),
        response=_sample_response(),
    )
    assert db_module.delete_run(session, record.id) is True
    assert db_module.get_run(session, record.id) is None
    assert db_module.delete_run(session, record.id) is False


def test_text_excerpt_truncates_long_input(session):
    long_text = "x" * 600
    record = db_module.persist_analysis_run(
        session,
        payload=_sample_response(),
        request=_sample_request(text=long_text),
        response=_sample_response(),
    )
    assert record.text_excerpt is not None
    assert len(record.text_excerpt) <= 281
    assert record.text_excerpt.endswith("…")
