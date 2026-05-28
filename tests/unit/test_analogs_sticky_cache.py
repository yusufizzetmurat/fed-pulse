"""Sticky-failure cache for ``app.services.analogs._load_state`` (#410).

The pre-#410 cache stored ``None`` on a failed load; ``get_state()`` then
re-attempted the load on every request because ``_state is None`` matched
"never tried". A broken environment combined with a sweep produced ~1.5M
stack traces in 18 minutes. The contract pinned here:

* First failure: ``_load_state`` is invoked exactly once and the warning
  carries a structured ``reason=...`` field.
* Subsequent ``get_state`` calls return ``None`` without calling
  ``_load_state`` again and without emitting any further warning.
* ``reset_state`` clears the sticky failure so a fixed environment
  recovers without a process restart.
* The success path is unchanged: a healthy load is cached and reused.
"""

from __future__ import annotations

import logging

import pytest

from app.services import analogs as analogs_service


@pytest.fixture(autouse=True)
def _reset_singleton():
    analogs_service.reset_state()
    yield
    analogs_service.reset_state()


def test_first_failure_logs_once_and_caches(monkeypatch, caplog):
    """First failed load emits exactly one WARNING with a structured reason."""

    calls = {"n": 0}

    def _boom() -> analogs_service._LoadFailure:
        calls["n"] += 1
        return analogs_service._LoadFailure(reason="encoder_load_failed checkpoint=/nope")

    monkeypatch.setattr(analogs_service, "_load_state", _boom)

    with caplog.at_level(logging.WARNING, logger=analogs_service.__name__):
        result = analogs_service.get_state()

    assert result is None
    assert calls["n"] == 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "analogs_load_failed" in msg
    assert "reason=" in msg
    assert "encoder_load_failed" in msg


def test_subsequent_calls_return_cached_failure_silently(monkeypatch, caplog):
    """After the first failure, ``get_state`` is a no-op: no reload, no log."""

    calls = {"n": 0}

    def _boom() -> analogs_service._LoadFailure:
        calls["n"] += 1
        return analogs_service._LoadFailure(reason="bundle_missing path=/nope")

    monkeypatch.setattr(analogs_service, "_load_state", _boom)

    # Prime the sticky cache.
    with caplog.at_level(logging.WARNING, logger=analogs_service.__name__):
        analogs_service.get_state()
    caplog.clear()

    # The next 5 calls (simulating a sweep) must not retry or re-log.
    with caplog.at_level(logging.WARNING, logger=analogs_service.__name__):
        for _ in range(5):
            assert analogs_service.get_state() is None

    assert calls["n"] == 1, "sticky cache must not re-invoke _load_state"
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_reset_state_clears_sticky_failure(monkeypatch, caplog):
    """``reset_state`` lets a fixed environment recover without restart."""

    calls = {"n": 0}

    def _boom() -> analogs_service._LoadFailure:
        calls["n"] += 1
        return analogs_service._LoadFailure(reason="bundle_missing path=/nope")

    monkeypatch.setattr(analogs_service, "_load_state", _boom)

    with caplog.at_level(logging.WARNING, logger=analogs_service.__name__):
        analogs_service.get_state()
    assert calls["n"] == 1

    analogs_service.reset_state()

    with caplog.at_level(logging.WARNING, logger=analogs_service.__name__):
        analogs_service.get_state()

    assert calls["n"] == 2, "reset_state must clear the sticky failure"


def test_success_path_caches_state_and_does_not_trigger_failure_logic(monkeypatch, caplog):
    """Healthy loads still cache normally; the sticky path is failure-only."""

    sentinel_state = object()
    calls = {"n": 0}

    def _ok():
        calls["n"] += 1
        return sentinel_state

    monkeypatch.setattr(analogs_service, "_load_state", _ok)

    with caplog.at_level(logging.WARNING, logger=analogs_service.__name__):
        first = analogs_service.get_state()
        second = analogs_service.get_state()

    assert first is sentinel_state
    assert second is sentinel_state
    assert calls["n"] == 1
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
