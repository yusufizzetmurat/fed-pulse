"""CORS regression for /analyze/market under a fresh cold-start (#379).

The frontend fans /analyze, /analyze/market and /analyze/analogs out via
``Promise.allSettled``. On a fresh boot the three race into the
cold-start path and the slowest writer's loader can raise an opaque
exception; before #379 that escaped the narrow ``RuntimeError`` catch
on /analyze/market, fell through to ``ServerErrorMiddleware`` (outside
the CORS layer) and the browser saw ERR_FAILED / no
``Access-Control-Allow-Origin`` header.

These tests pin the contract that every /analyze/market response --
2xx success, 5xx cold-start failure, and the preflight ``OPTIONS`` --
carries the CORS allow-origin header so the browser never reports the
response as opaque.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")

from fastapi.testclient import TestClient  # noqa: E402

import app.main as main_mod  # noqa: E402


_ORIGIN = "http://localhost:3000"
_SAMPLE_REQUEST = {
    "text": "Inflation remains elevated.",
    "date": "2024-12-18",
    "symbol": "^GSPC",
    "horizon": "5d",
}


def _stub_market_history(monkeypatch: pytest.MonkeyPatch) -> None:
    import datetime as _dt

    def fake_market_history(*, target_date: str, symbol: str, history_length: int):
        return [
            {
                "date": (
                    _dt.date.fromisoformat(target_date)
                    - _dt.timedelta(days=history_length - i)
                ).isoformat(),
                "close": 100.0 + float(i),
                "volatility_5d": 0.01 + 0.0001 * i,
            }
            for i in range(history_length)
        ]

    monkeypatch.setattr(main_mod, "fetch_market_history", fake_market_history)
    monkeypatch.setattr(
        main_mod,
        "analyze_text",
        lambda _text: {"label": "neutral", "score": 0.0, "raw": []},
    )


@pytest.fixture
def client() -> TestClient:
    return TestClient(main_mod.app)


def _force_cold_start(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Pin ``checkpoint_exists`` to False and stub the bootstrap.

    Returns a counter dict the caller can inspect to confirm the
    bootstrap stub was actually invoked (i.e. we exercised the
    cold-start path, not the warm fast-path).
    """

    state: dict[str, Any] = {"calls": 0}

    def fake_bootstrap(_payload):
        state["calls"] += 1
        # Flip the existence flag so the second pass through
        # ``_ensure_cold_start`` (under the lock) early-returns.
        state["checkpoint"] = True

    def fake_exists() -> bool:
        return bool(state.get("checkpoint", False))

    monkeypatch.setattr(main_mod, "_bootstrap_cold_start", fake_bootstrap)
    monkeypatch.setattr(main_mod, "checkpoint_exists", fake_exists)
    return state


def test_analyze_market_preflight_carries_cors_header(client: TestClient) -> None:
    """OPTIONS /analyze/market must echo the allow-origin even when the
    backend has no checkpoint yet."""

    response = client.options(
        "/analyze/market",
        headers={
            "Origin": _ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == _ORIGIN


def test_analyze_market_cold_start_success_carries_cors_header(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh boot, no checkpoint on disk. The bootstrap stub runs, the
    panel builder returns an empty payload, and the response must
    still carry the allow-origin header."""

    state = _force_cold_start(monkeypatch)
    _stub_market_history(monkeypatch)
    monkeypatch.setattr(
        main_mod, "build_market_reaction_panel", lambda _vectors: None
    )

    response = client.post(
        "/analyze/market",
        headers={"Origin": _ORIGIN},
        json=_SAMPLE_REQUEST,
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == _ORIGIN
    assert state["calls"] == 1


def test_analyze_market_cold_start_runtime_error_carries_cors_header(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A contract-mismatch RuntimeError from the cold-start path must
    surface as a 503 with allow-origin still attached -- the structured
    failure must not bypass the CORS middleware."""

    def boom(_payload):
        raise RuntimeError("contract mismatch: unknown kwarg 'foo'")

    monkeypatch.setattr(main_mod, "checkpoint_exists", lambda: False)
    monkeypatch.setattr(main_mod, "_bootstrap_cold_start", boom)

    response = client.post(
        "/analyze/market",
        headers={"Origin": _ORIGIN},
        json=_SAMPLE_REQUEST,
    )
    assert response.status_code == 503
    assert response.headers.get("access-control-allow-origin") == _ORIGIN


def test_analyze_market_cold_start_unexpected_exception_carries_cors_header(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-RuntimeError (e.g. corrupted checkpoint produces
    ``pickle.UnpicklingError`` / ``EOFError``) from a concurrent
    cold-start race must also degrade to a structured 503 with CORS
    headers, not a bare 500 from ServerErrorMiddleware outside the
    CORS layer."""

    def boom(_payload):
        raise EOFError("Ran out of input")

    monkeypatch.setattr(main_mod, "checkpoint_exists", lambda: False)
    monkeypatch.setattr(main_mod, "_bootstrap_cold_start", boom)

    response = client.post(
        "/analyze/market",
        headers={"Origin": _ORIGIN},
        json=_SAMPLE_REQUEST,
    )
    assert response.status_code == 503
    assert response.headers.get("access-control-allow-origin") == _ORIGIN


def test_ensure_cold_start_serialises_concurrent_callers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_ensure_cold_start`` must invoke ``_bootstrap_cold_start`` at
    most once even when /analyze, /analyze/market and /analyze/analogs
    all race on a fresh boot. The double-checked locking pattern means
    the runner-up sees the leader's checkpoint and early-returns."""

    state = _force_cold_start(monkeypatch)

    async def _race() -> None:
        await asyncio.gather(
            main_mod._ensure_cold_start(_SAMPLE_REQUEST),
            main_mod._ensure_cold_start(_SAMPLE_REQUEST),
            main_mod._ensure_cold_start(_SAMPLE_REQUEST),
        )

    # Reset the module-level lock so a previous test's loop is not held.
    main_mod._cold_start_lock = asyncio.Lock()
    asyncio.run(_race())
    assert state["calls"] == 1
