"""Thin wrapper around the LangSmith Python SDK.

Provides the `traced(name)` decorator. When LANGSMITH_API_KEY is set,
calls go through `langsmith.traceable`. When it is not set (e.g. unit
tests), the decorator is a transparent passthrough so the calling code
runs without any LangSmith dependency at runtime.
"""

from __future__ import annotations

import os
from functools import wraps
from typing import Callable, TypeVar

T = TypeVar("T", bound=Callable)


def traced(name: str) -> Callable[[T], T]:
    """Decorate a function so it is traced by LangSmith when the API key is set.

    Behaviour:
    - LANGSMITH_API_KEY unset -> identity decorator; the wrapped function
      runs unchanged.
    - LANGSMITH_API_KEY set -> wrap with langsmith.traceable so each
      invocation produces a trace under the project the SDK is configured for.
    """

    def decorator(fn: T) -> T:
        if not os.environ.get("LANGSMITH_API_KEY"):
            return fn

        try:
            from langsmith import traceable  # type: ignore
        except Exception:  # pragma: no cover - import guard
            return fn

        traced_fn = traceable(run_type="llm", name=name)(fn)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            return traced_fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
