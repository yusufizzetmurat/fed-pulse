"""LangSmith client wrapper tests.

The traced() decorator must:
- Pass through the wrapped function's return value identically.
- Become a no-op when LANGSMITH_API_KEY is unset (so unit tests do not
  need credentials).
- Use langsmith.traceable when LANGSMITH_API_KEY is set.
"""

from __future__ import annotations

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
