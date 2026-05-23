"""Durable real_train queue backed by arq + Redis (closes #103).

The FastAPI process enqueues a ``real_train_task`` into Redis and a separate
``arq`` worker container drains the queue. Job state lives in Redis so it
survives a backend restart -- the in-process ``_train_jobs`` dict in
``app/main.py`` is now a graceful fallback for dev environments without
Redis.

The task function intentionally mirrors the body of the old
``_run_real_train_job`` daemon thread so behaviour stays identical aside
from where state lives.
"""

from __future__ import annotations

import os
from typing import Any

from arq.connections import RedisSettings

from app.audit import append_audit_entry
from app.logging import bind_run_id, clear_run_id, configure_logging, get_logger
from app.schemas import AnalyzeRequest

REAL_TRAIN_HISTORY_LENGTH = 252
REDIS_DSN_ENV = "REDIS_URL"
DEFAULT_REDIS_DSN = "redis://redis:6379"
REAL_TRAIN_QUEUE = "arq:queue"


def get_redis_settings() -> RedisSettings:
    """Build :class:`RedisSettings` from the ``REDIS_URL`` env var."""

    dsn = os.environ.get(REDIS_DSN_ENV, DEFAULT_REDIS_DSN)
    return RedisSettings.from_dsn(dsn)


def _build_real_train_result(payload: AnalyzeRequest) -> dict[str, Any]:
    """Reproduce the body of the legacy ``_run_real_train_job`` thread.

    Kept as a free function so both the arq task and tests can drive it
    without going through the queue. Heavy imports are deferred to call
    time so importing :mod:`app.worker` from a test never drags in torch.
    """

    from app.services.forecaster import (
        bootstrap_checkpoint,
        build_feature_vectors,
    )
    from app.services.market_data import fetch_market_history
    from app.services.sentiment import analyze_text

    # Late-imported because the main app module owns the response shape.
    from app.main import _build_analyze_response, _record_history

    sentiment = analyze_text(payload.text)
    market_history = fetch_market_history(
        target_date=payload.date,
        symbol=payload.symbol,
        history_length=REAL_TRAIN_HISTORY_LENGTH,
    )
    history_vectors = build_feature_vectors(
        market_history,
        sentiment_score=float(sentiment["score"]),
        document_date=payload.date,
    )
    # Real Train intentionally runs a stronger checkpoint update over 252-day context.
    bootstrap_checkpoint(
        vectors=history_vectors,
        epochs=120,
        batch_size=64,
        learning_rate=3e-4,
        validation_fraction=0.2,
        early_stopping_patience=12,
    )
    result = _build_analyze_response(
        payload, mode="real_train", history_length=REAL_TRAIN_HISTORY_LENGTH
    )
    _record_history(payload, result)
    return result


async def real_train_task(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Arq task that drives a Real-Train run.

    ``ctx`` is the standard arq context dict (job_id, redis, score, …).
    ``payload`` is the JSON-safe ``AnalyzeRequest.model_dump()`` from the
    enqueueing API call. Returns the same analyze response shape the
    in-process daemon thread used to write into ``_train_jobs[id]['result']``.
    Arq stores it in Redis under the job id and :class:`arq.jobs.Job`
    exposes it via ``result()`` / ``result_info()``.
    """

    job_id = str(ctx.get("job_id") or "")
    bind_run_id(job_id)
    request = AnalyzeRequest.model_validate(payload)
    try:
        result = _build_real_train_result(request)
        try:
            append_audit_entry(
                "real_train_completed",
                run_id=job_id,
                metadata={"symbol": request.symbol, "date": request.date},
            )
        except Exception:  # pragma: no cover -- audit row best-effort
            get_logger("fed_pulse").warning("audit_write_failed", run_id=job_id)
        return result
    except Exception as exc:
        try:
            append_audit_entry(
                "real_train_failed",
                run_id=job_id,
                metadata={
                    "symbol": request.symbol,
                    "date": request.date,
                    "error": str(exc),
                },
            )
        except Exception:  # pragma: no cover
            get_logger("fed_pulse").warning("audit_write_failed", run_id=job_id)
        # Re-raise so arq records the failure in JobResult.success=False; the
        # listing endpoint maps that back to status='failed' + error string.
        raise
    finally:
        clear_run_id()


async def _startup(ctx: dict[str, Any]) -> None:  # noqa: ARG001
    configure_logging()
    get_logger("fed_pulse").info("worker_startup", service="arq-worker")


async def _shutdown(ctx: dict[str, Any]) -> None:  # noqa: ARG001
    get_logger("fed_pulse").info("worker_shutdown", service="arq-worker")


class WorkerSettings:
    """Arq worker entrypoint -- ``arq app.worker.WorkerSettings``."""

    functions = [real_train_task]
    redis_settings = get_redis_settings()
    on_startup = _startup
    on_shutdown = _shutdown
    # Job results stick around in Redis long enough for the dashboard to
    # surface them; 7 days mirrors arq's default but is set explicitly so
    # the listing endpoint never silently drops a recent run.
    keep_result = 60 * 60 * 24 * 7
    max_jobs = 4
