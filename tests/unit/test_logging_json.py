from __future__ import annotations

import io
import json
import logging

import pytest

pytest.importorskip("structlog")

import structlog  # noqa: E402

from app.logging import bind_run_id, clear_run_id, configure_logging


def _force_reconfigure():
    """Reset the structlog state so repeated `configure_logging` calls take
    effect inside the test process."""

    import app.logging as logging_module

    logging_module._configured = False
    structlog.reset_defaults()


def test_logging_emits_json_with_run_id(capsys):
    _force_reconfigure()
    # Re-bind stdout for the new handler.
    logging.getLogger().handlers = []
    configure_logging(level="INFO", json_output=True)
    bind_run_id("test-run-7")
    structlog.get_logger("fed_pulse.test").info("event", value=42)
    clear_run_id()
    captured = capsys.readouterr().out.strip().splitlines()
    assert captured, "expected at least one log line"
    payload = json.loads(captured[-1])
    assert payload["event"] == "event"
    assert payload["value"] == 42
    assert payload["run_id"] == "test-run-7"
