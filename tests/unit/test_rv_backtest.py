"""QLIKE-RV backtest service + endpoint contract.

Exercises the service-layer logic against an in-memory SQLite
``analysis_runs`` table with a stubbed RV predictor + RV history so
the tests run hermetically. The endpoint contract mirrors the
HAR-tercile backtest: 400 on a non-^GSPC symbol, 422 on out-of-range
limit, 200 with an aggregate coverage payload on the happy path.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient  # noqa: E402

import app.db as db_module  # noqa: E402
import app.main as main_mod  # noqa: E402
from app.services import rv_backtest  # noqa: E402
from app.services import rv_forecaster  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    db_module.reset_for_testing(
        f"sqlite:///{tmp_path / 'fed_pulse_rv_backtest.db'}"
    )
    return TestClient(main_mod.app)


def _persist_run(
    session,
    *,
    symbol: str = "^GSPC",
    document_date: str = "2024-01-31",
    created_at: datetime | None = None,
) -> db_module.AnalysisRun:
    """Insert a synthetic ``analysis_runs`` row for the RV backtest."""

    import uuid

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
        payload={},
        text_excerpt=None,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def _make_stub_predictor() -> Any:
    """Return a minimal _RvPredictor stand-in with h1 conformal quantiles set."""

    class _StubLinear:
        def __call__(self, x: Any) -> Any:
            import numpy as np

            class _T:
                def __init__(self, arr: np.ndarray) -> None:
                    self._arr = arr

                def cpu(self) -> "_T":
                    return self

                def numpy(self) -> np.ndarray:
                    return self._arr

            return _T(np.array([[0.0]], dtype=np.float32))

        def eval(self) -> "_StubLinear":
            return self

    n_feat = 11
    spec = {
        "model": "intraday_rv_production",
        "feature_order": [
            "har_daily", "har_weekly", "har_monthly",
            "rs_pos", "rs_neg", "bv", "rq", "rskew", "rkurt", "parkinson", "log_rvol",
        ],
        "date_last": "2026-05-29",
        "by_horizon": {
            "h1": {
                "har_coef": [-0.5, 0.6, 0.2, 0.1],
                "feat_mean": [0.0] * n_feat,
                "feat_std": [1.0] * n_feat,
                "resid_mean": 0.0,
                "resid_std": 0.5,
                "conformal_quantiles": {"0.20": 0.6, "0.10": 0.9},
                "seed_state_dicts": [],
                "n_oos_resid": 100,
            }
        },
    }
    inst = rv_forecaster._RvPredictor.__new__(rv_forecaster._RvPredictor)
    inst.model_dir = rv_forecaster.MODEL_DIR  # type: ignore[attr-defined]
    inst.spec = spec  # type: ignore[attr-defined]
    inst.eval = None  # type: ignore[attr-defined]
    inst.seed_models = {"h1": [_StubLinear() for _ in range(3)]}  # type: ignore[attr-defined]
    inst.revision = "stub@rv-backtest"  # type: ignore[attr-defined]
    return inst


@pytest.fixture()
def stub_predictor(monkeypatch: pytest.MonkeyPatch):
    """Pin the cached _RvPredictor so the service runs without HF Hub."""

    rv_forecaster._RvPredictor.reset()
    inst = _make_stub_predictor()
    monkeypatch.setattr(
        rv_forecaster._RvPredictor, "get", classmethod(lambda cls: inst)
    )
    yield inst
    rv_forecaster._RvPredictor.reset()


def _canned_rv_series(n: int = 60, start: str = "2024-01-01") -> tuple[list[float], list[str]]:
    """Build a deterministic RV series + ISO dates of length ``n``."""

    base = datetime.fromisoformat(start).date()
    rv = [1e-4 * (1.0 + (i % 5) * 0.1) for i in range(n)]
    dates = [(base + timedelta(days=i)).isoformat() for i in range(n)]
    return rv, dates


def test_aggregate_coverage_basic() -> None:
    rows = [
        {"realized_rv": 0.001, "in_band_80": True, "in_band_90": True},
        {"realized_rv": 0.002, "in_band_80": False, "in_band_90": True},
        {"realized_rv": 0.003, "in_band_80": True, "in_band_90": True},
        {"realized_rv": None, "in_band_80": None, "in_band_90": None},
    ]
    cov = rv_backtest._aggregate_coverage(rows)
    assert cov["total_runs"] == 4
    assert cov["resolved_runs"] == 3
    assert cov["pending_runs"] == 1
    assert cov["empirical_coverage_80"] == pytest.approx(2 / 3)
    assert cov["empirical_coverage_90"] == pytest.approx(1.0)
    assert cov["nominal_coverage_80"] == pytest.approx(0.80)
    assert cov["nominal_coverage_90"] == pytest.approx(0.90)


def test_aggregate_coverage_all_pending() -> None:
    rows = [
        {"realized_rv": None, "in_band_80": None, "in_band_90": None},
        {"realized_rv": None, "in_band_80": None, "in_band_90": None},
    ]
    cov = rv_backtest._aggregate_coverage(rows)
    assert cov["total_runs"] == 2
    assert cov["resolved_runs"] == 0
    assert cov["pending_runs"] == 2
    assert cov["empirical_coverage_80"] is None
    assert cov["empirical_coverage_90"] is None


def test_resolve_row_pending_inside_warmup(stub_predictor) -> None:
    import numpy as np

    rv_list, dates = _canned_rv_series(n=60)
    rv = np.asarray(rv_list, dtype=np.float64)
    # Event date sits inside the 22-day warmup window — pending row.
    target = dates[10]
    row = rv_backtest._resolve_row(
        event_date=target,
        rv=rv,
        dates=dates,
        predictor=stub_predictor,
    )
    assert row is not None
    assert row["realized_rv"] is None
    assert row["in_band_80"] is None
    assert row["in_band_90"] is None


def test_resolve_row_resolves_with_bands_outside_warmup(stub_predictor) -> None:
    import numpy as np

    rv_list, dates = _canned_rv_series(n=60)
    rv = np.asarray(rv_list, dtype=np.float64)
    # Index 30 is comfortably past the 22-day warmup horizon.
    target = dates[30]
    row = rv_backtest._resolve_row(
        event_date=target,
        rv=rv,
        dates=dates,
        predictor=stub_predictor,
    )
    assert row is not None
    assert row["realized_rv"] == pytest.approx(rv_list[30])
    # Bands bracket the point.
    assert row["band_lo_80"] <= row["point_forecast_rv"] <= row["band_hi_80"]
    assert row["band_lo_90"] <= row["band_lo_80"]
    assert row["band_hi_90"] >= row["band_hi_80"]
    assert isinstance(row["in_band_80"], bool)
    assert isinstance(row["in_band_90"], bool)


def test_resolve_row_returns_none_when_date_missing(stub_predictor) -> None:
    import numpy as np

    rv_list, dates = _canned_rv_series(n=60)
    rv = np.asarray(rv_list, dtype=np.float64)
    out = rv_backtest._resolve_row(
        event_date="2099-12-31",
        rv=rv,
        dates=dates,
        predictor=stub_predictor,
    )
    assert out is None


def test_get_rv_backtest_orders_by_recency_and_filters_symbol(
    client, monkeypatch, stub_predictor
) -> None:
    """Three ^GSPC rows resolve in created_at desc order; ^NDX row drops out."""

    rv_list, dates = _canned_rv_series(n=60)
    monkeypatch.setattr(
        rv_backtest, "_load_rv_series", lambda symbol: (list(rv_list), list(dates))
    )

    session_iter = db_module.get_session()
    sess = next(session_iter)
    try:
        base = datetime(2024, 6, 1, tzinfo=timezone.utc)
        _persist_run(sess, document_date=dates[30], created_at=base)
        _persist_run(
            sess, document_date=dates[40], created_at=base + timedelta(days=5)
        )
        _persist_run(
            sess, document_date=dates[50], created_at=base + timedelta(days=10)
        )
        _persist_run(
            sess,
            symbol="^NDX",
            document_date=dates[35],
            created_at=base + timedelta(days=20),
        )
    finally:
        sess.close()

    response = client.get(
        "/forecast/rv-backtest", params={"symbol": "^GSPC", "limit": 10}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["symbol"] == "^GSPC"
    assert body["horizon"] == 1
    assert body["coverage"]["total_runs"] == 3
    assert body["coverage"]["resolved_runs"] == 3
    event_dates = [row["event_date"] for row in body["rows"]]
    # Newest first.
    assert event_dates == [dates[50], dates[40], dates[30]]
    for row in body["rows"]:
        assert row["realized_rv"] is not None
        assert row["band_lo_80"] <= row["band_hi_80"]
        assert row["band_lo_90"] <= row["band_hi_90"]


def test_get_rv_backtest_emits_pending_row_when_date_missing(
    client, monkeypatch, stub_predictor
) -> None:
    """A persisted run whose event date sits outside the RV history is pending."""

    rv_list, dates = _canned_rv_series(n=60)
    monkeypatch.setattr(
        rv_backtest, "_load_rv_series", lambda symbol: (list(rv_list), list(dates))
    )

    session_iter = db_module.get_session()
    sess = next(session_iter)
    try:
        _persist_run(sess, document_date="2099-12-31")
    finally:
        sess.close()

    response = client.get("/forecast/rv-backtest", params={"symbol": "^GSPC"})
    assert response.status_code == 200
    body = response.json()
    assert body["coverage"]["total_runs"] == 1
    assert body["coverage"]["resolved_runs"] == 0
    assert body["coverage"]["pending_runs"] == 1
    assert body["coverage"]["empirical_coverage_80"] is None
    assert body["coverage"]["empirical_coverage_90"] is None
    row = body["rows"][0]
    assert row["realized_rv"] is None
    assert row["point_forecast_rv"] is None
    assert row["in_band_80"] is None
    assert row["in_band_90"] is None


def test_get_rv_backtest_aggregates_band_hits(
    client, monkeypatch, stub_predictor
) -> None:
    """Empirical coverage tracks ``in_band_*`` across resolved rows."""

    rv_list, dates = _canned_rv_series(n=60)
    monkeypatch.setattr(
        rv_backtest, "_load_rv_series", lambda symbol: (list(rv_list), list(dates))
    )

    session_iter = db_module.get_session()
    sess = next(session_iter)
    try:
        # Two rows comfortably past the warmup horizon.
        _persist_run(sess, document_date=dates[30])
        _persist_run(sess, document_date=dates[45])
    finally:
        sess.close()

    response = client.get("/forecast/rv-backtest", params={"symbol": "^GSPC"})
    assert response.status_code == 200
    body = response.json()
    assert body["coverage"]["total_runs"] == 2
    assert body["coverage"]["resolved_runs"] == 2
    # The exact coverage values depend on the stub; just assert finite + in [0, 1].
    assert 0.0 <= body["coverage"]["empirical_coverage_80"] <= 1.0
    assert 0.0 <= body["coverage"]["empirical_coverage_90"] <= 1.0


def test_endpoint_rejects_non_gspc_symbol(client) -> None:
    response = client.get("/forecast/rv-backtest", params={"symbol": "^NDX"})
    assert response.status_code == 400
    body = response.json()
    assert body["detail"]["error"] == "symbol_unsupported"


def test_endpoint_rejects_out_of_range_limit(client) -> None:
    over = client.get(
        "/forecast/rv-backtest", params={"symbol": "^GSPC", "limit": 51}
    )
    assert over.status_code == 422
    under = client.get(
        "/forecast/rv-backtest", params={"symbol": "^GSPC", "limit": 0}
    )
    assert under.status_code == 422


def test_endpoint_returns_empty_state_with_no_runs(client) -> None:
    response = client.get("/forecast/rv-backtest", params={"symbol": "^GSPC"})
    assert response.status_code == 200
    body = response.json()
    assert body["rows"] == []
    assert body["coverage"]["total_runs"] == 0
    assert body["coverage"]["resolved_runs"] == 0
    assert body["coverage"]["pending_runs"] == 0
    assert body["coverage"]["empirical_coverage_80"] is None
    assert body["coverage"]["empirical_coverage_90"] is None
    assert body["coverage"]["nominal_coverage_80"] == pytest.approx(0.80)
    assert body["coverage"]["nominal_coverage_90"] == pytest.approx(0.90)


def test_load_rv_series_requests_wide_history_window(monkeypatch) -> None:
    """The backtest must pull more RV history than the live forecast card.

    Regression test for the failure mode where the shared 60-day window
    capped older persisted FOMC event dates outside the dates index, so
    every event further than ~3 months back surfaced as pending. The
    backtest now requests a multi-quarter window so the trailing FOMC
    meetings can actually be aligned to the daily RV series.
    """

    captured: dict[str, Any] = {}

    def _fake_load_rv_history(symbol: str, days: int = 60) -> tuple[list[float], list[str]]:
        captured["symbol"] = symbol
        captured["days"] = days
        return ([0.0] * 30, ["2024-01-01"] * 30)

    import app.main as main_mod

    monkeypatch.setattr(main_mod, "_load_rv_history", _fake_load_rv_history)

    rv_backtest._load_rv_series("^GSPC")

    # 60 days = the live-forecast cap. The backtest must pass a strictly
    # larger ``days`` value so older event dates can land in the dates
    # index. ~2 years of trading days (504) is the current target.
    assert captured["symbol"] == "^GSPC"
    assert captured["days"] > 60
    assert captured["days"] >= 252


def test_main_load_rv_history_honors_days_argument(monkeypatch) -> None:
    """``_load_rv_history`` exposes a ``days`` knob the backtest passes."""

    import inspect

    sig = inspect.signature(main_mod._load_rv_history)
    assert "days" in sig.parameters
    assert sig.parameters["days"].default == main_mod._RV_HISTORY_DAYS
