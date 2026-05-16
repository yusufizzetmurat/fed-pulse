"""Tests for the /analyze real_train + /train-jobs Redis path (closes #103)."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("arq")
pytest.importorskip("fakeredis")

from arq.connections import ArqRedis  # noqa: E402
from fakeredis import FakeAsyncRedis, FakeServer  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.main as main_mod  # noqa: E402


def _fake_arq_pool_factory(server: FakeServer | None = None):
    """Return a zero-arg factory that constructs an ArqRedis on demand.

    The factory closes over a single ``FakeServer`` so every pool it
    produces shares the same in-memory key/value store. The pool itself
    is constructed inside the caller — when the caller is a request
    handler running in TestClient's anyio portal, the pool's internal
    ``asyncio.Queue`` binds to the portal's loop and the connection
    survives the round-trip. Constructing the pool in the test thread
    would bind it to a different loop and trip arq with
    ``RuntimeError: Queue is bound to a different event loop`` on the
    first ``enqueue_job`` call.
    """

    server = server if server is not None else FakeServer()

    def _build() -> ArqRedis:
        fake = FakeAsyncRedis(server=server)
        return ArqRedis(connection_pool=fake.connection_pool)

    _build.server = server  # type: ignore[attr-defined]
    return _build


def _fake_arq_pool(server: FakeServer | None = None) -> ArqRedis:
    """Eager pool variant for tests that explicitly want one instance.

    Most endpoint tests should use ``_fake_arq_pool_factory`` and
    install the factory on ``app.state.redis_pool``; the lazy ``_redis_pool``
    accessor unwraps the factory inside the request loop. Use the eager
    form only when the test calls ArqJob / pool methods itself outside
    a TestClient request (e.g. asserting state via ``asyncio.run``).
    """

    server = server if server is not None else FakeServer()
    fake = FakeAsyncRedis(server=server)
    return ArqRedis(connection_pool=fake.connection_pool)


@pytest.fixture(autouse=True)
def _clear_state(monkeypatch):
    """Each test sees an empty in-memory map and no Redis pool by default."""

    monkeypatch.setenv("FED_PULSE_DISABLE_REDIS_POOL", "1")
    with main_mod._train_jobs_lock:
        main_mod._train_jobs.clear()
    yield
    with main_mod._train_jobs_lock:
        main_mod._train_jobs.clear()
    # Teardown only acts on an eager ArqRedis instance. Factories
    # produce per-request pools that live and die inside the request
    # loop, so there is nothing for the test thread to close.
    pool = getattr(main_mod.app.state, "redis_pool", None)
    if isinstance(pool, ArqRedis):
        try:
            asyncio.run(pool.close(close_connection_pool=True))
        except Exception:
            pass
    main_mod.app.state.redis_pool = None


@pytest.fixture
def client() -> TestClient:
    return TestClient(main_mod.app)


def test_analyze_real_train_enqueues_into_redis(client):
    """When a Redis pool is attached, /analyze must enqueue a job through
    arq instead of spawning a daemon thread, and return the same
    accepted-response shape as before."""

    factory = _fake_arq_pool_factory()
    main_mod.app.state.redis_pool = factory

    response = client.post(
        "/analyze",
        json={
            "text": "sample text",
            "date": "2026-03-15",
            "symbol": "^GSPC",
            "forecast_mode": "real_train",
            "horizon": "3d",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    job_id = body["job_id"]
    assert job_id

    # The in-memory fallback dict must stay empty -- the Redis path is
    # the only source of truth when a pool is attached.
    with main_mod._train_jobs_lock:
        assert main_mod._train_jobs == {}

    # The job is now visible via arq.jobs.Job against a pool on the
    # asserting loop (asyncio.run creates a fresh loop, so we build
    # the pool inside the run).
    from arq.jobs import Job as ArqJob, JobStatus

    async def _status() -> JobStatus:
        pool = factory()
        return await ArqJob(job_id, pool).status()

    assert asyncio.run(_status()) == JobStatus.queued


def test_get_train_job_reads_from_redis(client):
    """``/train-jobs/{id}`` must shape an arq-stored job into the legacy
    response model so the frontend polling loop does not change."""

    main_mod.app.state.redis_pool = _fake_arq_pool_factory()

    response = client.post(
        "/analyze",
        json={
            "text": "x",
            "date": "2026-04-01",
            "symbol": "QQQ",
            "forecast_mode": "real_train",
            "horizon": "1d",
        },
    )
    job_id = response.json()["job_id"]

    detail = client.get(f"/train-jobs/{job_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["job_id"] == job_id
    assert body["status"] == "queued"
    assert body["result"] is None
    assert body["error"] is None


def test_get_train_job_missing_returns_404(client):
    main_mod.app.state.redis_pool = _fake_arq_pool_factory()

    response = client.get("/train-jobs/does-not-exist")
    assert response.status_code == 404


def test_list_train_jobs_reads_from_redis(client):
    """``/train-jobs`` listing must surface arq-stored jobs, not just the
    in-memory dict."""

    main_mod.app.state.redis_pool = _fake_arq_pool_factory()

    for symbol in ("^GSPC", "QQQ"):
        client.post(
            "/analyze",
            json={
                "text": "x",
                "date": "2026-04-01",
                "symbol": symbol,
                "forecast_mode": "real_train",
                "horizon": "1d",
            },
        )

    listing = client.get("/train-jobs")
    assert listing.status_code == 200
    body = listing.json()
    assert body["total"] == 2
    symbols = sorted(item["symbol"] for item in body["items"])
    assert symbols == ["QQQ", "^GSPC"]
    for item in body["items"]:
        assert item["status"] == "queued"


def test_real_train_state_survives_app_restart(client):
    """The headline contract for #103: a queued job must still be
    readable after the FastAPI app is torn down and a fresh pool
    (backed by the same Redis data) is bound.

    The two pools share a FakeServer but are otherwise independent
    ArqRedis objects, so the second client cannot read state through
    any in-process Python reference -- it has to round-trip through
    Redis. That is the contract: job state lives in Redis, not in the
    pool object.
    """

    server = FakeServer()
    factory = _fake_arq_pool_factory(server)
    main_mod.app.state.redis_pool = factory

    response = client.post(
        "/analyze",
        json={
            "text": "x",
            "date": "2026-04-01",
            "symbol": "^GSPC",
            "forecast_mode": "real_train",
            "horizon": "1d",
        },
    )
    job_id = response.json()["job_id"]

    # Drop the first factory, then bind a brand-new factory that shares
    # only the underlying FakeServer. From the endpoint's perspective
    # this is a process restart against the same Redis: no in-process
    # Python reference to the first pool survives.
    main_mod.app.state.redis_pool = None
    del factory
    fresh_factory = _fake_arq_pool_factory(server)
    main_mod.app.state.redis_pool = fresh_factory
    fresh_client = TestClient(main_mod.app)

    detail = fresh_client.get(f"/train-jobs/{job_id}")
    assert detail.status_code == 200
    assert detail.json()["job_id"] == job_id
    assert detail.json()["status"] == "queued"


def test_fast_mode_does_not_touch_redis(client, monkeypatch):
    """Regression contract from #103: fast-mode /analyze must stay
    synchronous and never hit Redis."""

    main_mod.app.state.redis_pool = _fake_arq_pool_factory()

    monkeypatch.setattr(
        main_mod,
        "analyze_text",
        lambda _: {
            "label": "POSITIVE",
            "score": 0.5,
            "raw": [{"label": "POSITIVE", "score": 0.5}],
        },
    )
    monkeypatch.setattr(
        main_mod,
        "fetch_market_snapshot",
        lambda **_: {
            "symbol": "^GSPC",
            "requested_date": "2026-03-15",
            "date_used": "2026-03-13",
            "lookback_days": 7,
            "close": 1.0,
            "volatility_5d": 0.01,
        },
    )
    monkeypatch.setattr(main_mod, "fetch_market_history", lambda **_: [])
    monkeypatch.setattr(main_mod, "parse_horizon_steps", lambda _: 1)
    monkeypatch.setattr(main_mod, "fetch_forward_trading_dates", lambda **_: [])
    monkeypatch.setattr(main_mod, "_record_history", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "checkpoint_exists", lambda: True)
    monkeypatch.setattr(
        main_mod,
        "forecast_quantitative_series",
        lambda **_: {
            "prediction": {"close": 1.0, "volatility": 0.01, "horizon": "3d"},
            "model": {
                "checkpoint_path": "x",
                "checkpoint_exists": True,
                "checkpoint_loaded": True,
                "runtime_mode": "fast",
                "hidden_size": 1,
                "num_layers": 1,
                "dropout": 0.0,
                "head_hidden_size": 1,
                "close_scale": 1.0,
                "sequence_length": 5,
            },
            "series": {
                "timestamps": [],
                "history_close": [],
                "history_volatility": [],
                "forecast_timestamps": [],
                "forecast_close": [],
                "forecast_close_lower": [],
                "forecast_close_upper": [],
                "forecast_volatility": [],
                "forecast_volatility_lower": [],
                "forecast_volatility_upper": [],
                "forecast_confidence_level": 0.8,
                "volatility_scale": {"suggested_ymin": 0.0, "suggested_ymax": 0.02},
            },
        },
    )

    response = client.post(
        "/analyze",
        json={
            "text": "x",
            "date": "2026-03-15",
            "symbol": "^GSPC",
            "forecast_mode": "fast",
            "horizon": "3d",
        },
    )
    assert response.status_code == 200
    # No real_train side-effects -- the in-memory map stays empty and the
    # Redis queue has no jobs.
    with main_mod._train_jobs_lock:
        assert main_mod._train_jobs == {}

    # Construct a fresh pool inside asyncio.run() so the connection
    # binds to that loop; factory-installed pools die with their
    # request loop.
    factory = main_mod.app.state.redis_pool

    async def _queue_len() -> int:
        check_pool = factory()
        return await check_pool.zcard("arq:queue")

    assert asyncio.run(_queue_len()) == 0
