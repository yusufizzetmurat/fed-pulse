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


def test_backtest_emits_pending_row_when_yfinance_returns_none(
    client, monkeypatch
) -> None:
    """A predicted row with no resolvable realized RV must surface as pending.

    yfinance returning None (typical for future / unresolved meetings)
    flows through _resolve_realized_tercile and into the response as
    realized_tercile=None, realized_rv=None, correct=None — so the panel
    can render it as "pending" rather than dropping the row.
    """

    monkeypatch.setattr(
        har_tercile_backtest,
        "_fetch_realized_rv_yf",
        lambda event_date, symbol: None,
    )
    monkeypatch.setattr(
        har_tercile_backtest,
        "_fetch_rv_history_for_cutoffs",
        lambda event_date, symbol: [0.005, 0.008, 0.012, 0.020],
    )

    session_iter = db_module.get_session()
    sess = next(session_iter)
    try:
        _persist_run(sess, regime_argmax="normal", document_date="2024-02-20")
    finally:
        sess.close()

    response = client.get("/forecast/har-tercile-backtest", params={"symbol": "^GSPC"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["rows"]) == 1
    row = body["rows"][0]
    assert row["predicted_tercile"] in {"low", "medium", "high"}
    assert row["realized_tercile"] is None
    assert row["realized_rv"] is None
    assert row["correct"] is None
    # Aggregate metrics treat pending rows as un-denominated:
    # total_runs counts every backtest-able prediction, resolved_runs only
    # the ones with a realized outcome.
    assert body["metrics"]["total_runs"] == 1
    assert body["metrics"]["resolved_runs"] == 0
    assert body["metrics"]["accuracy_overall"] is None


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


def test_backtest_with_persisted_variance_cutoffs_does_not_force_high(client, monkeypatch) -> None:
    """Variance-space cutoffs + variance-space realized stat stay comparable.

    Guards against the contract-drift failure in the original review:
    if the realized statistic were a daily std (~1e-2 for 1% daily
    moves) while the persisted cutoffs are daily variance (~1e-4), the
    std would always exceed q67 and every row would land in ``high``.
    Stub a realistic ^GSPC forward window with mixed up/down 0.5% moves
    and confirm the backtest's variance-space realized stat bucks into
    ``low`` against variance-space cutoffs that bracket it tightly.
    """

    # Persisted cutoffs in variance space (q33 / q67 of a daily RV
    # series for a quiet regime; values pulled from the upstream
    # quantile semantics in ``services.har_tercile._tercile_cutoffs``).
    quiet_cutoffs = {"cutoffs_q33": 5e-5, "cutoffs_q67": 1.5e-4}

    def _calm_window(event_date: str, symbol: str) -> float | None:
        # 10 forward bars, all 0.5% daily log-return magnitude.
        rets = [0.005, -0.005] * 5
        return har_tercile_backtest._realized_variance_from_log_returns(rets)

    monkeypatch.setattr(har_tercile_backtest, "_fetch_realized_rv_yf", _calm_window)

    session_iter = db_module.get_session()
    sess = next(session_iter)
    try:
        _persist_run(
            sess,
            regime_argmax="high",
            document_date="2024-02-20",
            payload_extra={
                "har_baselines": {
                    "horizons": [
                        {
                            "h": 22,
                            "tercile": "high",
                            "tercile_probs": {"low": 0.1, "medium": 0.2, "high": 0.7},
                        }
                    ],
                    **quiet_cutoffs,
                }
            },
        )
    finally:
        sess.close()

    response = client.get("/forecast/har-tercile-backtest", params={"symbol": "^GSPC"})
    assert response.status_code == 200
    body = response.json()
    row = body["rows"][0]
    # Mean of 0.005**2 = 2.5e-5 — well below q33 (5e-5) so realized
    # tercile must resolve to ``low``, not ``high``. A regression to
    # the std-based realized stat would emit ~0.005 and force ``high``.
    assert row["realized_rv"] == pytest.approx(2.5e-5)
    assert row["realized_tercile"] == "low"
    # Prediction was "high"; realized resolved to "low" → miss. The
    # accuracy KPI is therefore exercised, not dominated by a phantom
    # "all-high" bucketing.
    assert row["correct"] is False


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
    # The cutoffs must match ``services.har_tercile._tercile_cutoffs``
    # byte-for-byte — i.e. ``np.quantile(values, [1/3, 2/3])`` with
    # linear interpolation — otherwise the backtest's per-tercile
    # hit-rate stops being a faithful proxy of the live endpoint's
    # bucketing.
    import numpy as np

    values = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]
    expected_q33, expected_q67 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    q33, q67 = har_tercile_backtest._cutoffs_from_history(values)
    assert q33 == pytest.approx(float(expected_q33))
    assert q67 == pytest.approx(float(expected_q67))


def test_cutoffs_from_history_rejects_short_window() -> None:
    assert har_tercile_backtest._cutoffs_from_history([0.001, 0.002]) == (None, None)


def test_cutoffs_from_history_matches_upstream_predict_har_regime() -> None:
    """Backtest's cutoff helper reproduces the upstream tercile cutoffs.

    The upstream ``services.har_tercile._tercile_cutoffs`` is the gold
    standard the backtest's accuracy KPI must reproduce. Driving both
    with the same RV series should yield byte-identical q33 / q67.
    """

    import numpy as np

    from app.services.har_tercile import _tercile_cutoffs

    rng = np.random.default_rng(11)
    series = (rng.standard_normal(60) * 0.01) ** 2 + 1e-6
    upstream_q33, upstream_q67 = _tercile_cutoffs(series)
    backtest_q33, backtest_q67 = har_tercile_backtest._cutoffs_from_history(series.tolist())
    assert backtest_q33 == pytest.approx(upstream_q33)
    assert backtest_q67 == pytest.approx(upstream_q67)


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


def test_realized_variance_matches_mean_squared_log_returns() -> None:
    """Realized stat is daily VARIANCE (mean of r**2), not std.

    Upstream ``main._load_rv_history`` writes per-bar RV as
    ``r * r``; the forward-window scalar must live in the same space
    so the per-bar variance cutoffs from ``predict_har_regime`` are
    apples-to-apples comparable. A regression here would silently
    inflate the panel's realized-vol column by ~sqrt(252) and dump
    every resolved row into the ``high`` bucket once cutoffs start
    persisting upstream.
    """

    rets = [0.005, -0.004, 0.012, -0.003, 0.001, 0.0, -0.002, 0.004, 0.006, -0.007]
    rv = har_tercile_backtest._realized_variance_from_log_returns(rets)
    expected = sum(r * r for r in rets) / len(rets)
    assert rv == pytest.approx(expected)
    # Daily variance scale: for ~1% daily moves the variance is on the
    # order of 1e-4. The pre-fix std-based helper would have returned a
    # value on the order of 1e-2 (the std), so the assertion guards
    # against accidental regression to the old convention.
    assert rv < 1e-3


def test_realized_variance_aliases_legacy_name() -> None:
    """The legacy ``_realized_vol_from_log_returns`` symbol aliases the variance form."""

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
    """A second call for the same key reads from cache, not the network."""

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
    assert calls["n"] == 1  # second call served from cache


def test_realized_rv_cache_distinct_keys_do_not_collide(monkeypatch, reset_backtest_caches) -> None:
    """Different ``(event_date, symbol)`` keys each hit the fetcher exactly once."""

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
    """Once the cached entry crosses the TTL it is refreshed via the fetcher."""

    calls = {"n": 0}

    def _stub(event_date: str, symbol: str) -> float | None:
        calls["n"] += 1
        return 0.00099

    monkeypatch.setattr(
        har_tercile_backtest, "_fetch_realized_rv_yf_uncached", _stub
    )

    har_tercile_backtest._fetch_realized_rv_yf("2024-01-31", "^GSPC")
    assert calls["n"] == 1

    # Forge a stale entry by rewriting the cache slot with a timestamp
    # well past the TTL horizon. A fresh call must re-invoke the fetcher.
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
    """A None result is still cached so unresolved rows don't retry every render."""

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
    """The trailing-60d history fetcher gets the same TTL treatment."""

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
    """Stale history entries trigger a refetch."""

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
    """``reset_caches()`` wipes both cache dicts so tests stay isolated."""

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
