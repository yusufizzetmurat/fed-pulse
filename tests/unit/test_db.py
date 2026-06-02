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


def test_to_summary_surfaces_signed_stance_score(session):
    """A dovish-leaning multi-axis payload yields a negative ``stance_score``.

    Guards the History chart: ``sentiment_score`` is unsigned confidence
    in [0, 1] and cannot encode direction. ``stance_score`` is the signed
    ``P(hawkish) - P(dovish)`` from the persisted ``multi_axis`` block and
    is what the chart's Y-axis must read off the summary.
    """

    payload = _sample_response(
        multi_axis={
            "stance": {
                "distribution": {
                    "hawkish": 0.15,
                    "neutral": 0.25,
                    "dovish": 0.60,
                },
            },
        },
    )
    record = db_module.persist_analysis_run(
        session,
        payload=payload,
        request=_sample_request(),
        response=payload,
    )
    summary = record.to_summary()
    assert "stance_score" in summary
    assert summary["stance_score"] is not None
    assert summary["stance_score"] == pytest.approx(0.15 - 0.60)
    assert summary["stance_score"] < 0


def test_to_summary_stance_score_none_when_multi_axis_absent(session):
    """Pre-multi-axis / regression-mode rows return ``stance_score=None``.

    The chart falls back to ``stanceToScore(row.stance)`` in that case so
    the X-axis still renders something, but the summary must not fabricate
    a zero — the ``None`` signal is what tells the frontend to fall back.
    """

    record = db_module.persist_analysis_run(
        session,
        payload=_sample_response(),
        request=_sample_request(),
        response=_sample_response(),
    )
    summary = record.to_summary()
    assert summary["stance_score"] is None
