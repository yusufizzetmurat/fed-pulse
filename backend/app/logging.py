from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog
from structlog.contextvars import bind_contextvars, unbind_contextvars

from app.config import settings


_RUN_ID: ContextVar[str | None] = ContextVar("fed_pulse_run_id", default=None)
_configured = False


def configure_logging(*, level: str | None = None, json_output: bool = True) -> None:
    """Idempotent structlog setup. Subsequent calls are no-ops."""

    global _configured
    if _configured:
        return

    log_level = (level or settings.log_level or "INFO").upper()
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level, logging.INFO),
    )

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.add_logger_name,
    ]
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, log_level, logging.INFO)),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    if not _configured:
        configure_logging()
    logger: structlog.stdlib.BoundLogger = (
        structlog.get_logger(name) if name else structlog.get_logger()
    )
    return logger


def bind_run_id(run_id: str | None) -> None:
    if run_id is None:
        clear_run_id()
        return
    _RUN_ID.set(run_id)
    bind_contextvars(run_id=run_id)


def current_run_id() -> str | None:
    return _RUN_ID.get()


def clear_run_id() -> None:
    # Unbind just `run_id`; other structlog context vars (model_version,
    # dataset_version, …) must survive the request teardown.
    _RUN_ID.set(None)
    unbind_contextvars("run_id")
