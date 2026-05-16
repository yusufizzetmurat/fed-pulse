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


def _fake_arq_pool(server: FakeServer | None = None) -> ArqRedis:
    """Build an ArqRedis pool against a fakeredis backend.

    When ``server`` is None a fresh isolated FakeServer is allocated.
    Passing a shared FakeServer between two _fake_arq_pool() calls
    mimics two backend processes pointing at the same Redis: the pools
    are independent Python objects, but their underlying key/value
    store is the same. The restart-survival test relies on this to
    prove the contract that job state lives in Redis, not in the
    in-process pool.
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
    if getattr(main_mod.app.state, "redis_pool", None) is not None:
        try:
            asyncio.run(main_mod.app.state.redis_pool.close(close_connection_pool=True))
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

    pool = _fake_arq_pool()
    main_mod.app.state.redis_pool = pool

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

    # The job is now visible via arq.jobs.Job against the same pool.
    from arq.jobs import Job as ArqJob, JobStatus

    async def _status():
        return await ArqJob(job_id, pool).status()

    assert asyncio.run(_status()) == JobStatus.queued


def test_get_train_job_reads_from_redis(client):
    """``/train-jobs/{id}`` must shape an arq-stored job into the legacy
    response model so the frontend polling loop does not change."""

    pool = _fake_arq_pool()
    main_mod.app.state.redis_pool = pool

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
    pool = _fake_arq_pool()
    main_mod.app.state.redis_pool = pool

    response = client.get("/train-jobs/does-not-exist")
    assert response.status_code == 404


def test_list_train_jobs_reads_from_redis(client):
    """``/train-jobs`` listing must surface arq-stored jobs, not just the
    in-memory dict."""

    pool = _fake_arq_pool()
    main_mod.app.state.redis_pool = pool

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
    pool = _fake_arq_pool(server)
    main_mod.app.state.redis_pool = pool

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

    # Drop the first pool entirely, then bind a brand-new pool that
    # shares only the underlying FakeServer. From the endpoint's
    # perspective this is a process restart against the same Redis.
    main_mod.app.state.redis_pool = None
    del pool
    fresh_pool = _fake_arq_pool(server)
    main_mod.app.state.redis_pool = fresh_pool
    fresh_client = TestClient(main_mod.app)

    detail = fresh_client.get(f"/train-jobs/{job_id}")
    assert detail.status_code == 200
    assert detail.json()["job_id"] == job_id
    assert detail.json()["status"] == "queued"


def test_fast_mode_does_not_touch_redis(client, monkeypatch):
    """Regression contract from #103: fast-mode /analyze must stay
    synchronous and never hit Redis."""

    pool = _fake_arq_pool()
    main_mod.app.state.redis_pool = pool

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

    async def _queue_len():
        return await pool.zcard("arq:queue")

    assert asyncio.run(_queue_len()) == 0
