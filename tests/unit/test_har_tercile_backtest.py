"""HAR-tercile backtest service + endpoint contract.

Exercises the service-layer logic against an in-memory SQLite ``analysis_runs``
table and the endpoint's symbol / limit validation. The realized-tercile
resolution path stubs the yfinance hop so the tests run hermetically.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient  # noqa: E402

import app.db as db_module  # noqa: E402
import app.main as main_mod  # noqa: E402
from app.services import har_tercile_backtest  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    db_module.reset_for_testing(
        f"sqlite:///{tmp_path / 'fed_pulse_har_backtest.db'}"
    )
    return TestClient(main_mod.app)


def _persist_run(
    session,
    *,
    symbol: str = "^GSPC",
    document_date: str = "2024-01-31",
    regime_argmax: str | None = "high",
    distribution: dict[str, float] | None = None,
    payload_extra: dict[str, Any] | None = None,
    created_at: datetime | None = None,
) -> db_module.AnalysisRun:
    """Insert a synthetic ``analysis_runs`` row.

    Builds the persisted payload directly via the ORM so the test does
    not depend on the analyze pipeline. The regime classification block
    matches the real /analyze response shape.
    """

    import uuid

    payload: dict[str, Any] = {}
    if regime_argmax is not None:
        dist = distribution or {regime_argmax: 0.7}
        payload["regime_classification"] = {
            "predicted_set": [regime_argmax],
            "set_label": f"[{regime_argmax}]",
            "set_size": 1,
            "coverage": 0.9,
            "distribution": dist,
            "argmax_class": regime_argmax,
            "bucket_source": "classification",
        }
    if payload_extra:
        payload.update(payload_extra)

    row = db_module.AnalysisRun(
        id=str(uuid.uuid4()),
        created_at=created_at or datetime.now(timezone.utc),
        symbol=symbol,
        document_date=document_date,
        horizon="3d",
        forecast_mode="fast",
        stance="hawkish",
        sentiment_score=0.7,
        predicted_close=5000.0,
        current_close=4990.0,
        predicted_volatility=0.012,
        payload=payload,
        text_excerpt=None,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def test_extract_predicted_tercile_from_regime_classification() -> None:
    payload = {
        "regime_classification": {
            "argmax_class": "calm",
            "distribution": {"calm": 0.62, "normal": 0.28, "high": 0.10},
        }
    }
    label, prob = har_tercile_backtest._extract_predicted_tercile(payload)
    assert label == "low"  # calm → low under the tercile mapping
    assert prob == pytest.approx(0.62)


def test_extract_predicted_tercile_prefers_har_baselines_block() -> None:
    payload = {
        "regime_classification": {"argmax_class": "calm", "distribution": {"calm": 0.9}},
        "har_baselines": {
            "horizons": [
                {
                    "h": 22,
                    "tercile": "high",
                    "tercile_probs": {"low": 0.1, "medium": 0.2, "high": 0.7},
                }
            ]
        },
    }
    label, prob = har_tercile_backtest._extract_predicted_tercile(payload)
    assert label == "high"
    assert prob == pytest.approx(0.7)


def test_extract_predicted_tercile_none_on_missing_payload() -> None:
    assert har_tercile_backtest._extract_predicted_tercile(None) == (None, None)
    assert har_tercile_backtest._extract_predicted_tercile({}) == (None, None)
    assert har_tercile_backtest._extract_predicted_tercile({"regime_classification": {}}) == (
        None,
        None,
    )


def test_bucket_against_cutoffs_layout() -> None:
    assert har_tercile_backtest._bucket_against_cutoffs(0.0, 1.0, 2.0) == "low"
    assert har_tercile_backtest._bucket_against_cutoffs(1.5, 1.0, 2.0) == "medium"
    assert har_tercile_backtest._bucket_against_cutoffs(3.0, 1.0, 2.0) == "high"
    # Boundary: value == q33 lands in medium (lower bound inclusive on
    # the upper bucket, matching np.digitize convention).
    assert har_tercile_backtest._bucket_against_cutoffs(1.0, 1.0, 2.0) == "medium"


def test_aggregate_metrics_overall_and_per_tercile() -> None:
    rows = [
        {"predicted_tercile": "low", "realized_tercile": "low", "correct": True},
        {"predicted_tercile": "low", "realized_tercile": "high", "correct": False},
        {"predicted_tercile": "medium", "realized_tercile": "medium", "correct": True},
        {"predicted_tercile": "high", "realized_tercile": "high", "correct": True},
        {"predicted_tercile": "high", "realized_tercile": None, "correct": None},
    ]
    metrics = har_tercile_backtest._aggregate_metrics(rows)
    assert metrics["total_runs"] == 5
    assert metrics["resolved_runs"] == 4
    assert metrics["accuracy_overall"] == pytest.approx(3 / 4)
    assert metrics["per_tercile_hit_rate"]["low"] == pytest.approx(0.5)
    assert metrics["per_tercile_hit_rate"]["medium"] == pytest.approx(1.0)
    assert metrics["per_tercile_hit_rate"]["high"] == pytest.approx(1.0)


def test_aggregate_metrics_zero_resolved_returns_none_accuracy() -> None:
    rows = [
        {"predicted_tercile": "low", "realized_tercile": None, "correct": None},
        {"predicted_tercile": "high", "realized_tercile": None, "correct": None},
    ]
    metrics = har_tercile_backtest._aggregate_metrics(rows)
    assert metrics["total_runs"] == 2
    assert metrics["resolved_runs"] == 0
    assert metrics["accuracy_overall"] is None
    assert metrics["per_tercile_hit_rate"] == {}


def test_build_backtest_orders_by_recency_and_filters_symbol(client, monkeypatch) -> None:
    # Stub the yfinance fallback so resolution lands on the cutoffs +
    # canned realized RV without touching the network.
    monkeypatch.setattr(
        har_tercile_backtest,
        "_fetch_realized_rv_yf",
        lambda event_date, symbol: 0.012,
    )
    monkeypatch.setattr(
        har_tercile_backtest,
        "_fetch_rv_history_for_cutoffs",
        lambda event_date, symbol: [0.005, 0.008, 0.011, 0.013, 0.015, 0.020],
    )

    session_iter = db_module.get_session()
    sess = next(session_iter)
    try:
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # Three ^GSPC runs at different prediction labels, one ^NDX
        # row that should be filtered out.
        _persist_run(sess, regime_argmax="calm", document_date="2024-01-31",
                     created_at=base)
        _persist_run(sess, regime_argmax="normal", document_date="2024-03-20",
                     created_at=base + timedelta(days=5))
        _persist_run(sess, regime_argmax="high", document_date="2024-05-01",
                     created_at=base + timedelta(days=10))
        _persist_run(sess, symbol="^NDX", regime_argmax="high",
                     document_date="2024-04-01",
                     created_at=base + timedelta(days=20))
    finally:
        sess.close()

    response = client.get(
        "/forecast/har-tercile-backtest",
        params={"symbol": "^GSPC", "limit": 10},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["symbol"] == "^GSPC"
    assert body["horizon"] == 10
    assert body["metrics"]["total_runs"] == 3
    # All resolved (yfinance stubbed to a constant). The realized
    # vol 0.012 falls in `medium` against the cutoffs derived from the
    # stub history (q33 ≈ 0.008, q67 ≈ 0.015).
    assert body["metrics"]["resolved_runs"] == 3
    # Rows come back in created_at desc → high, normal, calm.
    predicted = [row["predicted_tercile"] for row in body["rows"]]
    assert predicted == ["high", "medium", "low"]
    # Realized lands in medium (~0.012 vs q33≈0.008 / q67≈0.015).
    realized = [row["realized_tercile"] for row in body["rows"]]
    assert all(r == "medium" for r in realized)
    # Per-tercile hit-rate: only `normal` (mapped to medium) predicted the
    # right bucket. Others miss.
    per_t = body["metrics"]["per_tercile_hit_rate"]
    assert per_t["medium"] == pytest.approx(1.0)
    assert per_t["low"] == pytest.approx(0.0)
    assert per_t["high"] == pytest.approx(0.0)


def test_backtest_skips_rows_without_regime_card(client, monkeypatch) -> None:
    """Rows whose persisted payload has no regime card drop out entirely.

    The denominator stays honest: ``total_runs`` reflects only rows we
    could backtest, not every analysis_runs row blindly.
    """

    monkeypatch.setattr(
        har_tercile_backtest,
        "_fetch_realized_rv_yf",
        lambda event_date, symbol: 0.012,
    )
    monkeypatch.setattr(
        har_tercile_backtest,
        "_fetch_rv_history_for_cutoffs",
        lambda event_date, symbol: [0.005, 0.008, 0.012, 0.020],
    )

    session_iter = db_module.get_session()
    sess = next(session_iter)
    try:
        _persist_run(sess, regime_argmax=None, document_date="2024-01-31")
        _persist_run(sess, regime_argmax="high", document_date="2024-02-20")
    finally:
        sess.close()

    response = client.get("/forecast/har-tercile-backtest", params={"symbol": "^GSPC"})
    assert response.status_code == 200
    body = response.json()
    assert body["metrics"]["total_runs"] == 1


def test_backtest_uses_persisted_realized_rv_when_present(client, monkeypatch) -> None:
    """When the payload pins ``forward_realized_vol_10d``, no yfinance hop fires."""

    def _explode(event_date: str, symbol: str) -> float:  # pragma: no cover - guard
        raise AssertionError("yfinance fallback should not be invoked")

    monkeypatch.setattr(har_tercile_backtest, "_fetch_realized_rv_yf", _explode)
    monkeypatch.setattr(
        har_tercile_backtest,
        "_fetch_rv_history_for_cutoffs",
        lambda event_date, symbol: [0.005, 0.008, 0.012, 0.020],
    )

    session_iter = db_module.get_session()
    sess = next(session_iter)
    try:
        _persist_run(
            sess,
            regime_argmax="high",
            document_date="2024-02-20",
            payload_extra={"forward_realized_vol_10d": 0.018},
        )
    finally:
        sess.close()

    response = client.get("/forecast/har-tercile-backtest", params={"symbol": "^GSPC"})
    assert response.status_code == 200
    body = response.json()
    assert body["metrics"]["resolved_runs"] == 1
    row = body["rows"][0]
    assert row["realized_rv"] == pytest.approx(0.018)
    # 0.018 > q67 (0.012) -> high bucket -> prediction correct.
    assert row["realized_tercile"] == "high"
    assert row["correct"] is True


def test_endpoint_rejects_non_gspc_symbol(client) -> None:
    response = client.get(
        "/forecast/har-tercile-backtest", params={"symbol": "^NDX"}
    )
    assert response.status_code == 400
    body = response.json()
    assert body["detail"]["error"] == "symbol_unsupported"


def test_endpoint_rejects_out_of_range_limit(client) -> None:
    over = client.get(
        "/forecast/har-tercile-backtest", params={"symbol": "^GSPC", "limit": 51}
    )
    assert over.status_code == 422
    under = client.get(
        "/forecast/har-tercile-backtest", params={"symbol": "^GSPC", "limit": 0}
    )
    assert under.status_code == 422


def test_endpoint_returns_empty_state_with_no_runs(client) -> None:
    response = client.get(
        "/forecast/har-tercile-backtest", params={"symbol": "^GSPC"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rows"] == []
    assert body["metrics"]["total_runs"] == 0
    assert body["metrics"]["resolved_runs"] == 0
    assert body["metrics"]["accuracy_overall"] is None


def test_cutoffs_from_history_basic() -> None:
    q33, q67 = har_tercile_backtest._cutoffs_from_history(
        [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]
    )
    assert q33 == pytest.approx(0.003)
    assert q67 == pytest.approx(0.005)


def test_cutoffs_from_history_rejects_short_window() -> None:
    assert har_tercile_backtest._cutoffs_from_history([0.001, 0.002]) == (None, None)


def test_normalize_tercile_label_maps_all_known_inputs() -> None:
    fn = har_tercile_backtest._normalize_tercile_label
    assert fn("calm") == "low"
    assert fn("Normal") == "medium"
    assert fn("HIGH") == "high"
    assert fn("low") == "low"
    assert fn("medium") == "medium"
    assert fn("unknown") is None
    assert fn(None) is None
    assert fn("") is None


def test_realized_vol_from_log_returns_basic() -> None:
    rv = har_tercile_backtest._realized_vol_from_log_returns([0.0, 0.01, -0.005, 0.002])
    assert rv is not None
    assert math.isfinite(rv)
    assert rv > 0.0
