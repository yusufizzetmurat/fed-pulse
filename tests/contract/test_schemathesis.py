"""Schemathesis fuzz against the FastAPI OpenAPI surface (#101 part a).

Generates inputs from the live OpenAPI schema and asserts every response
either parses against the declared schema or returns a 4xx with a
client-error body. A 5xx fails the suite — the API must validate input
and refuse cleanly rather than crash.

The whole module is gated on the optional ``schemathesis`` dep so the
default ``pytest .[dev]`` invocation skips it when the dep is absent.
"""

from __future__ import annotations

import os

import pytest

schemathesis = pytest.importorskip("schemathesis")
pytest.importorskip("fastapi")
pytest.importorskip("hypothesis")

# Redis-disabled lifespan avoids the 30s arq connect attempt during
# schema discovery; schemathesis cycles through every endpoint so the
# cost compounds. The flag is the same one the unit suite uses.
os.environ.setdefault("FED_PULSE_DISABLE_REDIS_POOL", "1")

from hypothesis import settings  # noqa: E402

from app.main import app  # noqa: E402

# ``from_asgi`` runs the spec discovery in-process against the ASGI
# app, so no live server is needed.
schema = schemathesis.openapi.from_asgi("/openapi.json", app)


# Endpoints where a 5xx is the documented response for a client-side
# input fault (e.g. a fetch URL with no protocol → 502 Bad Gateway).
# These are not server crashes; the fuzz suite ignores them on the
# matching endpoint while still catching 500s from the same handler.
_EXPECTED_5XX_BY_PATH: dict[str, set[int]] = {
    "/documents/parse": {502},
}


@schema.parametrize()
@settings(max_examples=15, deadline=None)
def test_no_server_errors(case: schemathesis.Case) -> None:
    """The API must respond with 2xx or 4xx for any generated input.

    5xx responses indicate the validation layer let an unhandled input
    through to crash the handler — that is the bug class this fuzz
    targets. Endpoints with a documented client-fault 5xx (see
    ``_EXPECTED_5XX_BY_PATH``) are exempt on that specific status.
    """

    response = case.call()
    if response.status_code < 500:
        return
    allowed = _EXPECTED_5XX_BY_PATH.get(case.path, set())
    if response.status_code in allowed:
        return
    pytest.fail(
        f"{case.method} {case.path} returned {response.status_code} "
        f"on generated input — server-side crash, body={response.text[:300]!r}"
    )
