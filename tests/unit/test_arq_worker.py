"""Unit tests for the arq-backed real_train queue (closes #103)."""

from __future__ import annotations

import asyncio
import inspect

import pytest

pytest.importorskip("arq")
pytest.importorskip("fakeredis")

from arq.connections import ArqRedis  # noqa: E402
from arq.jobs import Job as ArqJob, JobStatus  # noqa: E402
from fakeredis import FakeAsyncRedis  # noqa: E402

import app.worker as worker_mod  # noqa: E402


def test_real_train_task_signature():
    """The arq task contract is ``(ctx, payload) -> dict``."""

    fn = worker_mod.real_train_task
    sig = inspect.signature(fn)
    params = list(sig.parameters)
    assert params == ["ctx", "payload"]
    assert inspect.iscoroutinefunction(fn)


def test_worker_settings_exposes_real_train_task():
    """``arq app.worker.WorkerSettings`` must find the task in ``functions``."""

    settings = worker_mod.WorkerSettings
    assert worker_mod.real_train_task in settings.functions
    assert hasattr(settings, "redis_settings")


def test_get_redis_settings_reads_env(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://example.invalid:6380/2")
    settings = worker_mod.get_redis_settings()
    assert settings.host in {"example.invalid", "example"}
    assert settings.port == 6380
    assert settings.database == 2


def test_real_train_history_length_constant():
    assert worker_mod.REAL_TRAIN_HISTORY_LENGTH == 252


def _fake_arq_pool() -> ArqRedis:
    """Wrap fakeredis in the same ArqRedis class arq uses internally so
    ``ArqJob`` round-trips behave the same way they do against real Redis."""

    fake = FakeAsyncRedis()
    return ArqRedis(connection_pool=fake.connection_pool)


_SAMPLE_PAYLOAD = {
    "text": "fed text",
    "date": "2026-03-15",
    "symbol": "^GSPC",
    "horizon": "3d",
    "forecast_mode": "real_train",
    "include_realized": False,
    "include_xai": False,
}


def test_enqueue_marks_job_as_queued_in_redis(monkeypatch):
    """An enqueued job must surface as ``JobStatus.queued`` to the API
    side -- this is the contract the /train-jobs/{id} endpoint depends on."""

    async def _run():
        pool = _fake_arq_pool()
        await pool.enqueue_job("real_train_task", _SAMPLE_PAYLOAD, _job_id="qid")
        job = ArqJob("qid", pool)
        return await job.status()

    assert asyncio.run(_run()) == JobStatus.queued


def test_real_train_task_returns_dict(monkeypatch):
    """Drive the task body directly with the heavy bits stubbed; assert
    the return shape arq stores under the job id."""

    monkeypatch.setattr(
        worker_mod,
        "_build_real_train_result",
        lambda _payload: {"sentiment": {"label": "neutral", "score": 0.0}},
    )
    monkeypatch.setattr(worker_mod, "append_audit_entry", lambda *a, **k: None)

    async def _run():
        return await worker_mod.real_train_task(
            {"job_id": "test-job-1"}, _SAMPLE_PAYLOAD
        )

    result = asyncio.run(_run())
    assert isinstance(result, dict)
    assert result["sentiment"]["label"] == "neutral"


def test_real_train_task_propagates_failure(monkeypatch):
    """A raising task body must re-raise so arq stores success=False; the
    listing endpoint maps that back to status='failed'."""

    def _boom(_payload):
        raise RuntimeError("forecast bootstrap failed")

    monkeypatch.setattr(worker_mod, "_build_real_train_result", _boom)
    monkeypatch.setattr(worker_mod, "append_audit_entry", lambda *a, **k: None)

    async def _run():
        await worker_mod.real_train_task({"job_id": "test-job-2"}, _SAMPLE_PAYLOAD)

    with pytest.raises(RuntimeError, match="forecast bootstrap failed"):
        asyncio.run(_run())
