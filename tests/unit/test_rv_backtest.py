"""QLIKE-RV backtest service + endpoint contract.

Exercises the on-demand backtest: stubs the FOMC calendar feed plus
the RV-history and realized-RV yfinance fetchers, then checks that
the h=1 point + 80% / 90% bands published at each meeting bracket
the realized variance on the bar one day forward. The endpoint half
exercises symbol + limit validation.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient  # noqa: E402

import app.db as db_module  # noqa: E402
import app.main as main_mod  # noqa: E402
from app.services import fomc_calendar, rv_backtest  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    db_module.reset_for_testing(
        f"sqlite:///{tmp_path / 'fed_pulse_rv_backtest.db'}"
    )
    return TestClient(main_mod.app)


def _meeting(d: str) -> fomc_calendar.FomcMeeting:
    """Build a minimal FomcMeeting stand-in for the calendar stub."""

    md = date.fromisoformat(d)
    return fomc_calendar.FomcMeeting(
        meeting_date=md,
        meeting_type="scheduled",
        statement_release_date=md,
        minutes_release_date=md,
    )


def _stub_calendar(monkeypatch, dates: list[str]) -> None:
    """Stub ``list_past_meetings`` to return the supplied meeting dates."""

    meetings = [_meeting(d) for d in dates]

    def _fake_list_past_meetings(*, as_of=None, limit: int = 10):
        return list(meetings[:limit])

    monkeypatch.setattr(
        fomc_calendar, "list_past_meetings", _fake_list_past_meetings
    )


def _stub_predict_rv(monkeypatch, mapping: dict[str, dict[str, float]]) -> None:
    """Stub ``predict_rv`` keyed off the first RV-history value.

    Each entry maps a ``rv_history[0]`` discriminator to a prediction
    dict containing ``point`` and the four band edges. The test
    composes synthetic RV histories whose first value selects the
    desired prediction, so the same stub serves multiple meeting
    dates without sequencing the calls.
    """

    from app.services import rv_forecaster as _rv

    def _fake_predict(rv_history, *_args, **_kwargs):
        key = f"{float(rv_history[0]):.6f}"
        spec = mapping[key]
        return {
            "horizons": [
                {
                    "h": 1,
                    "point": float(spec["point"]),
                    "band_lo_80": float(spec["lo80"]),
                    "band_hi_80": float(spec["hi80"]),
                    "band_lo_90": float(spec["lo90"]),
                    "band_hi_90": float(spec["hi90"]),
                    "qlike_model": None,
                    "qlike_har": None,
                    "coverage_empirical_90": float("nan"),
                },
                {
                    "h": 5,
                    "point": float(spec["point"]) * 1.05,
                    "band_lo_80": float(spec["lo80"]) * 1.05,
                    "band_hi_80": float(spec["hi80"]) * 1.05,
                    "band_lo_90": float(spec["lo90"]) * 1.05,
                    "band_hi_90": float(spec["hi90"]) * 1.05,
                    "qlike_model": None,
                    "qlike_har": None,
                    "coverage_empirical_90": float("nan"),
                },
            ],
            "model_revision": "stub-rev",
        }

    monkeypatch.setattr(_rv, "predict_rv", _fake_predict)


def _history(discriminator: float, length: int = 60) -> list[float]:
    """Compose a synthetic RV history of ``length`` bars.

    The first value carries the discriminator the ``predict_rv`` stub
    keys off; the remaining bars are uniform filler so the predict
    call sees a series at least as long as ``_MIN_RV_HISTORY``.
    """

    return [float(discriminator)] + [1e-4] * (length - 1)


def test_build_backtest_predicts_and_resolves_each_meeting(client, monkeypatch) -> None:
    """End-to-end: three calendar meetings -> three predicted/realized rows.

    Stubs the calendar feed, the RV-history fetcher (composing per-date
    histories whose first value selects the predicted bands), and the
    realized-RV fetcher. Verifies the rows come back in calendar order
    and that each in_band flag is computed against the bands the model
    would have emitted at that decision point.
    """

    _stub_calendar(monkeypatch, ["2024-05-01", "2024-03-20", "2024-01-31"])

    histories = {
        "2024-05-01": _history(0.000300),
        "2024-03-20": _history(0.000200),
        "2024-01-31": _history(0.000100),
    }

    def _hist(event_date: str, symbol: str) -> list[float]:
        return list(histories[event_date])

    monkeypatch.setattr(rv_backtest, "_fetch_rv_history", _hist)

    _stub_predict_rv(
        monkeypatch,
        {
            "0.000300": {
                "point": 2.5e-4,
                "lo80": 1.0e-4, "hi80": 4.0e-4,
                "lo90": 0.5e-4, "hi90": 5.0e-4,
            },
            "0.000200": {
                "point": 1.5e-4,
                "lo80": 0.8e-4, "hi80": 2.5e-4,
                "lo90": 0.5e-4, "hi90": 3.5e-4,
            },
            "0.000100": {
                "point": 1.0e-4,
                "lo80": 0.5e-4, "hi80": 1.8e-4,
                "lo90": 0.3e-4, "hi90": 2.5e-4,
            },
        },
    )

    realized_by_date = {
        "2024-05-01": 2.0e-4,   # inside 80 band
        "2024-03-20": 1.0e-4,   # inside 80 band
        "2024-01-31": 1.2e-4,   # inside 80 band
    }
    monkeypatch.setattr(
        rv_backtest,
        "_fetch_realized_rv_yf",
        lambda event_date, symbol: realized_by_date[event_date],
    )

    response = client.get(
        "/forecast/rv-backtest", params={"symbol": "^GSPC", "limit": 10}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["symbol"] == "^GSPC"
    assert body["horizon"] == 1

    # Calendar order preserved.
    event_dates = [row["event_date"] for row in body["rows"]]
    assert event_dates == ["2024-05-01", "2024-03-20", "2024-01-31"]

    for row in body["rows"]:
        assert row["realized_rv"] is not None
        assert row["band_lo_80"] <= row["point_forecast_rv"] <= row["band_hi_80"]
        assert row["band_lo_90"] <= row["band_lo_80"]
        assert row["band_hi_90"] >= row["band_hi_80"]
        assert row["in_band_80"] is True
        assert row["in_band_90"] is True

    coverage = body["coverage"]
    assert coverage["total_runs"] == 3
    assert coverage["resolved_runs"] == 3
    assert coverage["pending_runs"] == 0
    assert coverage["empirical_coverage_80"] == pytest.approx(1.0)
    assert coverage["empirical_coverage_90"] == pytest.approx(1.0)
    assert coverage["nominal_coverage_80"] == pytest.approx(0.80)
    assert coverage["nominal_coverage_90"] == pytest.approx(0.90)


def test_build_backtest_pending_row_when_realized_missing(client, monkeypatch) -> None:
    """A meeting whose forward bar has not resolved surfaces as pending."""

    _stub_calendar(monkeypatch, ["2024-06-12"])

    monkeypatch.setattr(
        rv_backtest,
        "_fetch_rv_history",
        lambda event_date, symbol: _history(0.000200),
    )
    _stub_predict_rv(
        monkeypatch,
        {
            "0.000200": {
                "point": 1.5e-4,
                "lo80": 0.8e-4, "hi80": 2.5e-4,
                "lo90": 0.5e-4, "hi90": 3.5e-4,
            },
        },
    )
    monkeypatch.setattr(
        rv_backtest,
        "_fetch_realized_rv_yf",
        lambda event_date, symbol: None,
    )

    response = client.get("/forecast/rv-backtest", params={"symbol": "^GSPC"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["rows"]) == 1
    row = body["rows"][0]
    assert row["event_date"] == "2024-06-12"
    assert row["point_forecast_rv"] == pytest.approx(1.5e-4)
    assert row["realized_rv"] is None
    assert row["in_band_80"] is None
    assert row["in_band_90"] is None
    cov = body["coverage"]
    assert cov["total_runs"] == 1
    assert cov["resolved_runs"] == 0
    assert cov["pending_runs"] == 1
    assert cov["empirical_coverage_80"] is None
    assert cov["empirical_coverage_90"] is None


def test_build_backtest_pending_row_when_history_too_short(client, monkeypatch) -> None:
    """A meeting with <60 days of trailing RV surfaces as a pending row."""

    _stub_calendar(monkeypatch, ["2024-01-31"])
    monkeypatch.setattr(
        rv_backtest,
        "_fetch_rv_history",
        lambda event_date, symbol: [1e-4] * 30,  # < _MIN_RV_HISTORY
    )

    # predict_rv must not be reached when the history gate fails.
    from app.services import rv_forecaster as _rv

    def _explode(*args, **kwargs):  # pragma: no cover - guard
        raise AssertionError("predict_rv called despite short history")

    monkeypatch.setattr(_rv, "predict_rv", _explode)
    monkeypatch.setattr(
        rv_backtest,
        "_fetch_realized_rv_yf",
        lambda event_date, symbol: 1.0e-4,
    )

    response = client.get("/forecast/rv-backtest", params={"symbol": "^GSPC"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["rows"]) == 1
    row = body["rows"][0]
    assert row["event_date"] == "2024-01-31"
    assert row["point_forecast_rv"] is None
    assert row["band_lo_80"] is None
    assert row["band_hi_80"] is None
    assert row["band_lo_90"] is None
    assert row["band_hi_90"] is None
    assert row["realized_rv"] is None
    assert row["in_band_80"] is None
    assert row["in_band_90"] is None
    cov = body["coverage"]
    assert cov["total_runs"] == 1
    assert cov["resolved_runs"] == 0
    assert cov["pending_runs"] == 1


def test_build_backtest_aggregate_matches_hand_computed_coverage(
    client, monkeypatch
) -> None:
    """Mixed hits + misses + pending row: aggregate coverage = hand calc."""

    _stub_calendar(
        monkeypatch,
        ["2024-09-18", "2024-07-31", "2024-05-01", "2024-03-20"],
    )

    histories = {
        "2024-09-18": _history(0.000300),
        "2024-07-31": _history(0.000300),
        "2024-05-01": _history(0.000200),
        "2024-03-20": _history(0.000100),
    }
    monkeypatch.setattr(
        rv_backtest,
        "_fetch_rv_history",
        lambda event_date, symbol: list(histories[event_date]),
    )
    _stub_predict_rv(
        monkeypatch,
        {
            "0.000300": {
                "point": 2.5e-4,
                "lo80": 1.0e-4, "hi80": 4.0e-4,
                "lo90": 0.5e-4, "hi90": 5.0e-4,
            },
            "0.000200": {
                "point": 1.5e-4,
                "lo80": 0.8e-4, "hi80": 2.5e-4,
                "lo90": 0.5e-4, "hi90": 3.5e-4,
            },
            "0.000100": {
                "point": 1.0e-4,
                "lo80": 0.5e-4, "hi80": 1.8e-4,
                "lo90": 0.3e-4, "hi90": 2.5e-4,
            },
        },
    )
    realized = {
        "2024-09-18": 2.0e-4,    # inside 80 band [1e-4, 4e-4] -> hit80, hit90
        "2024-07-31": 6.0e-4,    # outside 90 band [0.5e-4, 5e-4] -> miss80, miss90
        "2024-05-01": 3.0e-4,    # outside 80 band, inside 90 -> miss80, hit90
        "2024-03-20": None,      # pending
    }
    monkeypatch.setattr(
        rv_backtest,
        "_fetch_realized_rv_yf",
        lambda event_date, symbol: realized[event_date],
    )

    response = client.get(
        "/forecast/rv-backtest", params={"symbol": "^GSPC", "limit": 10}
    )
    assert response.status_code == 200
    body = response.json()
    cov = body["coverage"]
    # 4 rows total, 3 resolved, 1 in 80 band, 2 in 90 band.
    assert cov["total_runs"] == 4
    assert cov["resolved_runs"] == 3
    assert cov["pending_runs"] == 1
    assert cov["empirical_coverage_80"] == pytest.approx(1.0 / 3.0)
    assert cov["empirical_coverage_90"] == pytest.approx(2.0 / 3.0)


def test_build_backtest_dedupes_meeting_dates(client, monkeypatch) -> None:
    """Duplicate meeting dates in the calendar collapse to one row."""

    _stub_calendar(
        monkeypatch,
        ["2024-05-01", "2024-05-01", "2024-03-20"],
    )

    monkeypatch.setattr(
        rv_backtest,
        "_fetch_rv_history",
        lambda event_date, symbol: _history(0.000200),
    )
    _stub_predict_rv(
        monkeypatch,
        {
            "0.000200": {
                "point": 1.5e-4,
                "lo80": 0.8e-4, "hi80": 2.5e-4,
                "lo90": 0.5e-4, "hi90": 3.5e-4,
            },
        },
    )
    monkeypatch.setattr(
        rv_backtest,
        "_fetch_realized_rv_yf",
        lambda event_date, symbol: 1.2e-4,
    )

    response = client.get(
        "/forecast/rv-backtest", params={"symbol": "^GSPC", "limit": 10}
    )
    assert response.status_code == 200
    body = response.json()
    event_dates = [row["event_date"] for row in body["rows"]]
    assert event_dates == ["2024-05-01", "2024-03-20"]


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


def test_endpoint_returns_empty_state_with_no_meetings(client, monkeypatch) -> None:
    """No past meetings -> empty rows + zero-state coverage, 200 OK."""

    monkeypatch.setattr(
        fomc_calendar, "list_past_meetings", lambda *, as_of=None, limit=10: []
    )
    response = client.get("/forecast/rv-backtest", params={"symbol": "^GSPC"})
    assert response.status_code == 200
    body = response.json()
    assert body["rows"] == []
    cov = body["coverage"]
    assert cov["total_runs"] == 0
    assert cov["resolved_runs"] == 0
    assert cov["pending_runs"] == 0
    assert cov["empirical_coverage_80"] is None
    assert cov["empirical_coverage_90"] is None
    assert cov["nominal_coverage_80"] == pytest.approx(0.80)
    assert cov["nominal_coverage_90"] == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# TTL in-process cache on yfinance fetchers.
# ---------------------------------------------------------------------------


@pytest.fixture()
def reset_backtest_caches():
    """Force a clean cache around each cache-flavored test."""

    rv_backtest.reset_caches()
    yield
    rv_backtest.reset_caches()


def test_realized_rv_cache_hit_skips_underlying_fetch(monkeypatch, reset_backtest_caches) -> None:
    calls = {"n": 0}

    def _stub(event_date: str, symbol: str) -> float | None:
        calls["n"] += 1
        return 0.00042

    monkeypatch.setattr(rv_backtest, "_fetch_realized_rv_yf_uncached", _stub)

    first = rv_backtest._fetch_realized_rv_yf("2024-01-31", "^GSPC")
    second = rv_backtest._fetch_realized_rv_yf("2024-01-31", "^GSPC")

    assert first == pytest.approx(0.00042)
    assert second == pytest.approx(0.00042)
    assert calls["n"] == 1


def test_realized_rv_cache_ttl_expiry_refetches(monkeypatch, reset_backtest_caches) -> None:
    calls = {"n": 0}

    def _stub(event_date: str, symbol: str) -> float | None:
        calls["n"] += 1
        return 0.00099

    monkeypatch.setattr(rv_backtest, "_fetch_realized_rv_yf_uncached", _stub)

    rv_backtest._fetch_realized_rv_yf("2024-01-31", "^GSPC")
    assert calls["n"] == 1

    stale_ts = (
        rv_backtest._realized_rv_cache[("2024-01-31", "^GSPC")][0]
        - rv_backtest._BACKTEST_CACHE_TTL_SECONDS
        - 1.0
    )
    rv_backtest._realized_rv_cache[("2024-01-31", "^GSPC")] = (stale_ts, 0.00099)

    rv_backtest._fetch_realized_rv_yf("2024-01-31", "^GSPC")
    assert calls["n"] == 2


def test_rv_history_cache_hit_skips_underlying_fetch(monkeypatch, reset_backtest_caches) -> None:
    calls = {"n": 0}
    canned = [0.001, 0.002, 0.004, 0.008]

    def _stub(event_date: str, symbol: str) -> list[float]:
        calls["n"] += 1
        return list(canned)

    monkeypatch.setattr(rv_backtest, "_fetch_rv_history_uncached", _stub)

    first = rv_backtest._fetch_rv_history("2024-01-31", "^GSPC")
    second = rv_backtest._fetch_rv_history("2024-01-31", "^GSPC")

    assert first == canned
    assert second == canned
    assert calls["n"] == 1
    # The cache returns a copy so callers cannot mutate the cached list.
    second.append(99.0)
    third = rv_backtest._fetch_rv_history("2024-01-31", "^GSPC")
    assert third == canned


def test_reset_caches_clears_state(monkeypatch) -> None:
    monkeypatch.setattr(
        rv_backtest,
        "_fetch_realized_rv_yf_uncached",
        lambda event_date, symbol: 0.0007,
    )
    monkeypatch.setattr(
        rv_backtest,
        "_fetch_rv_history_uncached",
        lambda event_date, symbol: [0.001, 0.002, 0.003],
    )

    rv_backtest._fetch_realized_rv_yf("2024-03-20", "^GSPC")
    rv_backtest._fetch_rv_history("2024-03-20", "^GSPC")
    assert rv_backtest._realized_rv_cache
    assert rv_backtest._rv_history_cache

    rv_backtest.reset_caches()

    assert rv_backtest._realized_rv_cache == {}
    assert rv_backtest._rv_history_cache == {}


def test_list_past_meetings_orders_recent_first_and_respects_limit() -> None:
    """`fomc_calendar.list_past_meetings` returns at most ``limit`` meetings."""

    meetings = fomc_calendar.list_past_meetings(
        as_of=datetime(2025, 1, 1).date(), limit=3
    )
    assert len(meetings) == 3
    assert meetings[0].meeting_date == date(2024, 12, 17)
    assert all(
        meetings[i].meeting_date > meetings[i + 1].meeting_date
        for i in range(len(meetings) - 1)
    )
