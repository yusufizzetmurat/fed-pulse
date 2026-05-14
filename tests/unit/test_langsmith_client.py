"""LangSmith client wrapper tests.

The traced() decorator must:
- Pass through the wrapped function's return value identically.
- Become a no-op when LANGSMITH_API_KEY is unset (so unit tests do not
  need credentials).
- Use langsmith.traceable when LANGSMITH_API_KEY is set.
"""

from __future__ import annotations

import sys
import types

from app.services import langsmith_client


def test_traced_returns_callable() -> None:
    @langsmith_client.traced("my_call")
    def fn(x: int) -> int:
        return x + 1

    assert fn(2) == 3


def test_traced_passes_kwargs(monkeypatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    @langsmith_client.traced("my_call")
    def fn(*, x: int, y: int) -> int:
        return x * y

    assert fn(x=3, y=4) == 12


def test_traced_is_noop_when_key_unset(monkeypatch) -> None:
    """Without an API key the decorator should be transparent."""

    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    @langsmith_client.traced("my_call")
    def fn(x: int) -> int:
        return x * 10

    assert fn(7) == 70


def test_traced_uses_langsmith_traceable_when_key_is_set(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    calls = []

    def _traceable(*, run_type, name):
        calls.append((run_type, name))

        def _decorator(fn):
            def _wrapped(*args, **kwargs):
                return fn(*args, **kwargs) + 5

            return _wrapped

        return _decorator

    fake_langsmith = types.ModuleType("langsmith")
    fake_langsmith.traceable = _traceable
    monkeypatch.setitem(sys.modules, "langsmith", fake_langsmith)

    @langsmith_client.traced("my_call")
    def fn(x: int) -> int:
        return x + 1

    assert fn(2) == 8
    assert calls == [("llm", "my_call")]


def test_traced_falls_back_to_original_when_traceable_import_fails(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "langsmith", types.ModuleType("langsmith"))

    def fn(x: int) -> int:
        return x * 2

    decorated = langsmith_client.traced("my_call")(fn)
    assert decorated is fn
    assert decorated(3) == 6
