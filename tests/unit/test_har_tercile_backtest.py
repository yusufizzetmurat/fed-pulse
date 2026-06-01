"""HAR-tercile backtest service + endpoint contract.

Exercises the on-demand backtest: stubs the FOMC calendar feed plus
the RV-history and realized-RV yfinance fetchers, then checks that
predicted / realized tercile pairs line up against the same cutoff
basis. The endpoint half exercises symbol + limit validation.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient  # noqa: E402

import app.db as db_module  # noqa: E402
import app.main as main_mod  # noqa: E402
from app.services import fomc_calendar, har_tercile_backtest  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    db_module.reset_for_testing(
        f"sqlite:///{tmp_path / 'fed_pulse_har_backtest.db'}"
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


def _stub_predict_har_regime(monkeypatch, mapping: dict[str, dict[str, Any]]) -> None:
    """Stub ``predict_har_regime`` keyed off the first RV-history value.

    Each entry maps a ``rv_history[0]`` discriminator to a prediction
    dict the production endpoint would return: ``{tercile, prob, q33,
    q67}``. The test composes synthetic RV histories whose first value
    selects the desired prediction, so the same stub serves multiple
    meeting dates without sequencing the calls.
    """

    from app.services import har_tercile as _ht

    def _fake_predict(rv_history, cutoffs_q33=None, cutoffs_q67=None, har_coef=None):
        key = f"{float(rv_history[0]):.6f}"
        spec = mapping[key]
        return {
            "horizons": [
                {
                    "h": 22,
                    "predicted_rv": spec["q33"] / 2.0 if spec["tercile"] == "low" else (
                        (spec["q33"] + spec["q67"]) / 2.0 if spec["tercile"] == "medium" else spec["q67"] * 2.0
                    ),
                    "tercile": spec["tercile"],
                    "tercile_probs": {
                        "low": 0.1, "medium": 0.2, "high": 0.7,
                    } if spec["tercile"] == "high" else (
                        {"low": 0.7, "medium": 0.2, "high": 0.1}
                        if spec["tercile"] == "low"
                        else {"low": 0.15, "medium": 0.7, "high": 0.15}
                    ),
                    "macro_f1": 0.65,
                    "macro_f1_source": "stub",
                    "qlike_model": None,
                    "qlike_har": None,
                }
            ],
            "cutoffs_q33": float(spec["q33"]),
            "cutoffs_q67": float(spec["q67"]),
            "model_revision": "stub-rev",
        }

    monkeypatch.setattr(_ht, "predict_har_regime", _fake_predict)


def test_build_backtest_predicts_and_resolves_each_meeting(client, monkeypatch) -> None:
    """End-to-end: three calendar meetings → three predicted/realized rows.

    Stubs the calendar feed, the RV-history fetcher (composing per-date
    histories whose first value selects the predicted bucket), and the
    realized-RV fetcher. Verifies the rows come back in calendar order
    and that each realized tercile is bucketed against the SAME q33/q67
    the prediction emitted.
    """

    _stub_calendar(monkeypatch, ["2024-05-01", "2024-03-20", "2024-01-31"])

    # Per-date histories: the first value carries a unique discriminator
    # that the predict_har_regime stub keys off, plus 21 filler bars so
    # the _MIN_RV_HISTORY gate (22) passes. The remaining bars are
    # deliberately uniform so the synthetic cutoffs the stub emits do
    # not depend on real quantile math.
    histories = {
        "2024-05-01": [0.000300] + [0.0001] * 21,  # → predicted high
        "2024-03-20": [0.000200] + [0.0001] * 21,  # → predicted medium
        "2024-01-31": [0.000100] + [0.0001] * 21,  # → predicted low
    }

    def _hist(event_date: str, symbol: str) -> list[float]:
        return list(histories[event_date])

    monkeypatch.setattr(
        har_tercile_backtest, "_fetch_rv_history_for_cutoffs", _hist
    )

    _stub_predict_har_regime(
        monkeypatch,
        {
            "0.000300": {"tercile": "high", "q33": 5e-5, "q67": 1.5e-4},
            "0.000200": {"tercile": "medium", "q33": 5e-5, "q67": 1.5e-4},
            "0.000100": {"tercile": "low", "q33": 5e-5, "q67": 1.5e-4},
        },
    )

    realized_by_date = {
        "2024-05-01": 2.0e-4,   # > q67 → high; predicted high → correct
        "2024-03-20": 1.0e-4,   # between q33/q67 → medium; predicted medium → correct
        "2024-01-31": 3.0e-5,   # < q33 → low; predicted low → correct
    }

    monkeypatch.setattr(
        har_tercile_backtest,
        "_fetch_realized_rv_yf",
        lambda event_date, symbol: realized_by_date[event_date],
    )

    response = client.get(
        "/forecast/har-tercile-backtest",
        params={"symbol": "^GSPC", "limit": 10},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["symbol"] == "^GSPC"
    assert body["horizon"] == 10

    # Calendar order preserved: stub returns the three dates in the
    # supplied order (most-recent-first), and the backtest must not
    # reshuffle them.
    event_dates = [row["event_date"] for row in body["rows"]]
    assert event_dates == ["2024-05-01", "2024-03-20", "2024-01-31"]

    predicted = [row["predicted_tercile"] for row in body["rows"]]
    assert predicted == ["high", "medium", "low"]

    realized = [row["realized_tercile"] for row in body["rows"]]
    assert realized == ["high", "medium", "low"]

    # Per-row correctness flags + aggregate accuracy: every row hits
    # the bucket the prediction called, so accuracy is 1.0 and each
    # per-tercile rate is 1.0.
    assert all(row["correct"] is True for row in body["rows"])
    metrics = body["metrics"]
    assert metrics["total_runs"] == 3
    assert metrics["resolved_runs"] == 3
    assert metrics["accuracy_overall"] == pytest.approx(1.0)
    assert metrics["per_tercile_hit_rate"]["low"] == pytest.approx(1.0)
    assert metrics["per_tercile_hit_rate"]["medium"] == pytest.approx(1.0)
    assert metrics["per_tercile_hit_rate"]["high"] == pytest.approx(1.0)


def test_build_backtest_pending_row_when_realized_missing(client, monkeypatch) -> None:
    """A meeting whose forward window has not resolved surfaces as pending."""

    _stub_calendar(monkeypatch, ["2024-06-12"])

    monkeypatch.setattr(
        har_tercile_backtest,
        "_fetch_rv_history_for_cutoffs",
        lambda event_date, symbol: [0.000200] + [0.0001] * 21,
    )
    _stub_predict_har_regime(
        monkeypatch,
        {"0.000200": {"tercile": "medium", "q33": 5e-5, "q67": 1.5e-4}},
    )
    monkeypatch.setattr(
        har_tercile_backtest,
        "_fetch_realized_rv_yf",
        lambda event_date, symbol: None,
    )

    response = client.get("/forecast/har-tercile-backtest", params={"symbol": "^GSPC"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["rows"]) == 1
    row = body["rows"][0]
    assert row["event_date"] == "2024-06-12"
    assert row["predicted_tercile"] == "medium"
    assert row["realized_tercile"] is None
    assert row["realized_rv"] is None
    assert row["correct"] is None
    assert body["metrics"]["total_runs"] == 1
    assert body["metrics"]["resolved_runs"] == 0
    assert body["metrics"]["accuracy_overall"] is None


def test_build_backtest_aggregate_matches_hand_computed_hit_rates(
    client, monkeypatch
) -> None:
    """Mixed hits + misses + pending row: aggregate KPI = hand calc."""

    _stub_calendar(
        monkeypatch,
        ["2024-09-18", "2024-07-31", "2024-05-01", "2024-03-20"],
    )

    histories = {
        "2024-09-18": [0.000300] + [0.0001] * 21,  # predicted high
        "2024-07-31": [0.000300] + [0.0001] * 21,  # predicted high (miss)
        "2024-05-01": [0.000200] + [0.0001] * 21,  # predicted medium (hit)
        "2024-03-20": [0.000100] + [0.0001] * 21,  # predicted low (pending)
    }
    monkeypatch.setattr(
        har_tercile_backtest,
        "_fetch_rv_history_for_cutoffs",
        lambda event_date, symbol: list(histories[event_date]),
    )
    _stub_predict_har_regime(
        monkeypatch,
        {
            "0.000300": {"tercile": "high", "q33": 5e-5, "q67": 1.5e-4},
            "0.000200": {"tercile": "medium", "q33": 5e-5, "q67": 1.5e-4},
            "0.000100": {"tercile": "low", "q33": 5e-5, "q67": 1.5e-4},
        },
    )
    realized = {
        "2024-09-18": 2.0e-4,   # high (hit)
        "2024-07-31": 3.0e-5,   # low (miss)
        "2024-05-01": 1.0e-4,   # medium (hit)
        "2024-03-20": None,     # pending
    }
    monkeypatch.setattr(
        har_tercile_backtest,
        "_fetch_realized_rv_yf",
        lambda event_date, symbol: realized[event_date],
    )

    response = client.get(
        "/forecast/har-tercile-backtest", params={"symbol": "^GSPC", "limit": 10}
    )
    assert response.status_code == 200
    body = response.json()
    metrics = body["metrics"]
    # 4 rows total, 3 resolved, 2 correct -> 2/3 overall accuracy.
    assert metrics["total_runs"] == 4
    assert metrics["resolved_runs"] == 3
    assert metrics["accuracy_overall"] == pytest.approx(2.0 / 3.0)
    per_t = metrics["per_tercile_hit_rate"]
    # Two `high` predictions resolved (one hit, one miss) → 0.5.
    assert per_t["high"] == pytest.approx(0.5)
    # One `medium` prediction resolved (hit) → 1.0.
    assert per_t["medium"] == pytest.approx(1.0)
    # Single `low` prediction never resolved, so it should be absent.
    assert "low" not in per_t


def test_build_backtest_dedupes_meeting_dates(client, monkeypatch) -> None:
    """Duplicate meeting dates in the calendar collapse to one row."""

    _stub_calendar(
        monkeypatch,
        ["2024-05-01", "2024-05-01", "2024-03-20"],
    )

    monkeypatch.setattr(
        har_tercile_backtest,
        "_fetch_rv_history_for_cutoffs",
        lambda event_date, symbol: [0.000200] + [0.0001] * 21,
    )
    _stub_predict_har_regime(
        monkeypatch,
        {"0.000200": {"tercile": "medium", "q33": 5e-5, "q67": 1.5e-4}},
    )
    monkeypatch.setattr(
        har_tercile_backtest,
        "_fetch_realized_rv_yf",
        lambda event_date, symbol: 1.0e-4,
    )

    response = client.get(
        "/forecast/har-tercile-backtest", params={"symbol": "^GSPC", "limit": 10}
    )
    assert response.status_code == 200
    body = response.json()
    event_dates = [row["event_date"] for row in body["rows"]]
    # The duplicate "2024-05-01" must not produce two rows.
    assert event_dates == ["2024-05-01", "2024-03-20"]


def test_build_backtest_skips_meetings_with_short_rv_history(
    client, monkeypatch
) -> None:
    """Meetings with too little leading RV history are dropped, not pending."""

    _stub_calendar(monkeypatch, ["2024-01-31"])
    monkeypatch.setattr(
        har_tercile_backtest,
        "_fetch_rv_history_for_cutoffs",
        lambda event_date, symbol: [0.0001] * 5,  # < _MIN_RV_HISTORY
    )

    # Ensure predict_har_regime is never called when the history gate fails.
    from app.services import har_tercile as _ht

    def _explode(*args, **kwargs):  # pragma: no cover - guard
        raise AssertionError("predict_har_regime called despite short history")

    monkeypatch.setattr(_ht, "predict_har_regime", _explode)
    monkeypatch.setattr(
        har_tercile_backtest,
        "_fetch_realized_rv_yf",
        lambda event_date, symbol: 1.0e-4,
    )

    response = client.get(
        "/forecast/har-tercile-backtest", params={"symbol": "^GSPC"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rows"] == []
    assert body["metrics"]["total_runs"] == 0


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


def test_endpoint_returns_empty_state_with_no_meetings(client, monkeypatch) -> None:
    """No past meetings → empty rows + zero-state metrics, 200 OK."""

    monkeypatch.setattr(
        fomc_calendar, "list_past_meetings", lambda *, as_of=None, limit=10: []
    )
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
    import numpy as np

    values = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]
    expected_q33, expected_q67 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    q33, q67 = har_tercile_backtest._cutoffs_from_history(values)
    assert q33 == pytest.approx(float(expected_q33))
    assert q67 == pytest.approx(float(expected_q67))


def test_cutoffs_from_history_rejects_short_window() -> None:
    assert har_tercile_backtest._cutoffs_from_history([0.001, 0.002]) == (None, None)


def test_cutoffs_from_history_matches_upstream_predict_har_regime() -> None:
    """Backtest's cutoff helper reproduces the upstream tercile cutoffs."""

    import numpy as np

    from app.services.har_tercile import _tercile_cutoffs

    rng = np.random.default_rng(11)
    series = (rng.standard_normal(60) * 0.01) ** 2 + 1e-6
    upstream_q33, upstream_q67 = _tercile_cutoffs(series)
    backtest_q33, backtest_q67 = har_tercile_backtest._cutoffs_from_history(
        series.tolist()
    )
    assert backtest_q33 == pytest.approx(upstream_q33)
    assert backtest_q67 == pytest.approx(upstream_q67)


def test_realized_vol_from_log_returns_basic() -> None:
    rv = har_tercile_backtest._realized_vol_from_log_returns([0.0, 0.01, -0.005, 0.002])
    assert rv is not None
    assert math.isfinite(rv)
    assert rv > 0.0


def test_realized_variance_matches_mean_squared_log_returns() -> None:
    """Realized stat is daily VARIANCE (mean of r**2), not std."""

    rets = [0.005, -0.004, 0.012, -0.003, 0.001, 0.0, -0.002, 0.004, 0.006, -0.007]
    rv = har_tercile_backtest._realized_variance_from_log_returns(rets)
    expected = sum(r * r for r in rets) / len(rets)
    assert rv == pytest.approx(expected)
    # Daily variance scale: for ~1% daily moves the variance is on the
    # order of 1e-4.
    assert rv < 1e-3


def test_realized_variance_aliases_legacy_name() -> None:
    rets = [0.01, -0.005, 0.002, 0.0]
    assert har_tercile_backtest._realized_vol_from_log_returns(rets) == pytest.approx(
        har_tercile_backtest._realized_variance_from_log_returns(rets)
    )


# ---------------------------------------------------------------------------
# TTL in-process cache on yfinance fetchers.
# ---------------------------------------------------------------------------


@pytest.fixture()
def reset_backtest_caches():
    """Force a clean cache around each cache-flavored test."""

    har_tercile_backtest.reset_caches()
    yield
    har_tercile_backtest.reset_caches()


def test_realized_rv_cache_hit_skips_underlying_fetch(monkeypatch, reset_backtest_caches) -> None:
    calls = {"n": 0}

    def _stub(event_date: str, symbol: str) -> float | None:
        calls["n"] += 1
        return 0.00042

    monkeypatch.setattr(
        har_tercile_backtest, "_fetch_realized_rv_yf_uncached", _stub
    )

    first = har_tercile_backtest._fetch_realized_rv_yf("2024-01-31", "^GSPC")
    second = har_tercile_backtest._fetch_realized_rv_yf("2024-01-31", "^GSPC")

    assert first == pytest.approx(0.00042)
    assert second == pytest.approx(0.00042)
    assert calls["n"] == 1


def test_realized_rv_cache_distinct_keys_do_not_collide(monkeypatch, reset_backtest_caches) -> None:
    calls: list[tuple[str, str]] = []

    def _stub(event_date: str, symbol: str) -> float | None:
        calls.append((event_date, symbol))
        return 0.001

    monkeypatch.setattr(
        har_tercile_backtest, "_fetch_realized_rv_yf_uncached", _stub
    )

    har_tercile_backtest._fetch_realized_rv_yf("2024-01-31", "^GSPC")
    har_tercile_backtest._fetch_realized_rv_yf("2024-02-20", "^GSPC")
    har_tercile_backtest._fetch_realized_rv_yf("2024-01-31", "^GSPC")  # cached
    har_tercile_backtest._fetch_realized_rv_yf("2024-01-31", "^NDX")

    assert calls == [
        ("2024-01-31", "^GSPC"),
        ("2024-02-20", "^GSPC"),
        ("2024-01-31", "^NDX"),
    ]


def test_realized_rv_cache_ttl_expiry_refetches(monkeypatch, reset_backtest_caches) -> None:
    calls = {"n": 0}

    def _stub(event_date: str, symbol: str) -> float | None:
        calls["n"] += 1
        return 0.00099

    monkeypatch.setattr(
        har_tercile_backtest, "_fetch_realized_rv_yf_uncached", _stub
    )

    har_tercile_backtest._fetch_realized_rv_yf("2024-01-31", "^GSPC")
    assert calls["n"] == 1

    stale_ts = (
        har_tercile_backtest._realized_rv_cache[("2024-01-31", "^GSPC")][0]
        - har_tercile_backtest._BACKTEST_CACHE_TTL_SECONDS
        - 1.0
    )
    har_tercile_backtest._realized_rv_cache[("2024-01-31", "^GSPC")] = (
        stale_ts,
        0.00099,
    )

    har_tercile_backtest._fetch_realized_rv_yf("2024-01-31", "^GSPC")
    assert calls["n"] == 2


def test_realized_rv_cache_caches_none_payload(monkeypatch, reset_backtest_caches) -> None:
    calls = {"n": 0}

    def _stub(event_date: str, symbol: str) -> float | None:
        calls["n"] += 1
        return None

    monkeypatch.setattr(
        har_tercile_backtest, "_fetch_realized_rv_yf_uncached", _stub
    )

    assert har_tercile_backtest._fetch_realized_rv_yf("2024-06-15", "^GSPC") is None
    assert har_tercile_backtest._fetch_realized_rv_yf("2024-06-15", "^GSPC") is None
    assert calls["n"] == 1


def test_rv_history_cache_hit_skips_underlying_fetch(monkeypatch, reset_backtest_caches) -> None:
    calls = {"n": 0}
    canned = [0.001, 0.002, 0.004, 0.008]

    def _stub(event_date: str, symbol: str) -> list[float]:
        calls["n"] += 1
        return list(canned)

    monkeypatch.setattr(
        har_tercile_backtest, "_fetch_rv_history_for_cutoffs_uncached", _stub
    )

    first = har_tercile_backtest._fetch_rv_history_for_cutoffs("2024-01-31", "^GSPC")
    second = har_tercile_backtest._fetch_rv_history_for_cutoffs("2024-01-31", "^GSPC")

    assert first == canned
    assert second == canned
    assert calls["n"] == 1
    # The cache returns a copy so callers cannot mutate the cached list.
    second.append(99.0)
    third = har_tercile_backtest._fetch_rv_history_for_cutoffs("2024-01-31", "^GSPC")
    assert third == canned


def test_rv_history_cache_ttl_expiry_refetches(monkeypatch, reset_backtest_caches) -> None:
    calls = {"n": 0}

    def _stub(event_date: str, symbol: str) -> list[float]:
        calls["n"] += 1
        return [0.001, 0.002, 0.003]

    monkeypatch.setattr(
        har_tercile_backtest, "_fetch_rv_history_for_cutoffs_uncached", _stub
    )

    har_tercile_backtest._fetch_rv_history_for_cutoffs("2024-02-20", "^GSPC")
    assert calls["n"] == 1

    stale_ts = (
        har_tercile_backtest._rv_history_cache[("2024-02-20", "^GSPC")][0]
        - har_tercile_backtest._BACKTEST_CACHE_TTL_SECONDS
        - 1.0
    )
    har_tercile_backtest._rv_history_cache[("2024-02-20", "^GSPC")] = (
        stale_ts,
        [0.001, 0.002, 0.003],
    )

    har_tercile_backtest._fetch_rv_history_for_cutoffs("2024-02-20", "^GSPC")
    assert calls["n"] == 2


def test_reset_caches_clears_state(monkeypatch) -> None:
    monkeypatch.setattr(
        har_tercile_backtest,
        "_fetch_realized_rv_yf_uncached",
        lambda event_date, symbol: 0.0007,
    )
    monkeypatch.setattr(
        har_tercile_backtest,
        "_fetch_rv_history_for_cutoffs_uncached",
        lambda event_date, symbol: [0.001, 0.002, 0.003],
    )

    har_tercile_backtest._fetch_realized_rv_yf("2024-03-20", "^GSPC")
    har_tercile_backtest._fetch_rv_history_for_cutoffs("2024-03-20", "^GSPC")
    assert har_tercile_backtest._realized_rv_cache
    assert har_tercile_backtest._rv_history_cache

    har_tercile_backtest.reset_caches()

    assert har_tercile_backtest._realized_rv_cache == {}
    assert har_tercile_backtest._rv_history_cache == {}


def test_list_past_meetings_orders_recent_first_and_respects_limit() -> None:
    """`fomc_calendar.list_past_meetings` returns at most ``limit`` meetings."""

    meetings = fomc_calendar.list_past_meetings(
        as_of=datetime(2025, 1, 1).date(), limit=3
    )
    assert len(meetings) == 3
    # Most recent meeting strictly before 2025-01-01 must come first.
    assert meetings[0].meeting_date == date(2024, 12, 17)
    # Sorted strictly descending.
    assert all(
        meetings[i].meeting_date > meetings[i + 1].meeting_date
        for i in range(len(meetings) - 1)
    )


def test_list_past_meetings_zero_limit_is_empty() -> None:
    assert fomc_calendar.list_past_meetings(limit=0) == []
