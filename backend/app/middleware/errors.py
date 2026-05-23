from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.logging import bind_run_id, clear_run_id, current_run_id, get_logger


class RunIdMiddleware(BaseHTTPMiddleware):
    """Bind a per-request `run_id` into the structlog context so every log line
    inside the request handler carries the same correlation id. The id is also
    surfaced on the request scope and on the response via `x-run-id`."""

    def __init__(self, app: ASGIApp, header_name: str = "x-run-id") -> None:
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        incoming = request.headers.get(self.header_name)
        run_id = incoming or str(uuid.uuid4())
        bind_run_id(run_id)
        request.state.run_id = run_id
        try:
            response = await call_next(request)
        finally:
            clear_run_id()
        response.headers[self.header_name] = run_id
        return response


def _resolve_run_id(request: Request) -> str | None:
    state_id = getattr(request.state, "run_id", None) if hasattr(request, "state") else None
    if state_id:
        return state_id
    header_id = request.headers.get("x-run-id")
    if header_id:
        return header_id
    return current_run_id()


def _payload(*, code: str, detail: Any, run_id: str | None) -> dict[str, Any]:
    return {"code": code, "detail": detail, "run_id": run_id}


def _json_safe(value: Any) -> Any:
    """Recursively coerce pydantic v2's error payloads into JSON-safe shapes.

    ``RequestValidationError.errors()`` can embed the raw ``Exception``
    object under ``ctx.error`` (pydantic v2 behaviour), which the
    default JSON encoder cannot serialise. Stringify any leftover
    non-trivial types so the JSONResponse never raises mid-handler.
    """

    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def register_error_handlers(app: FastAPI) -> None:
    """Map known exception classes to structured `{code, detail, run_id}`
    payloads. Bare `Exception` falls through to a 500 with the message
    redacted."""

    log = get_logger("fed_pulse.errors")

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        run_id = _resolve_run_id(request)
        errors = _json_safe(exc.errors())
        log.warning("validation_error", path=request.url.path, errors=errors)
        return JSONResponse(
            status_code=422,
            content=_payload(code="validation_error", detail=errors, run_id=run_id),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        run_id = _resolve_run_id(request)
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(code=f"http_{exc.status_code}", detail=exc.detail, run_id=run_id),
        )

    @app.exception_handler(ValueError)
    async def _value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        run_id = _resolve_run_id(request)
        log.info("value_error", path=request.url.path, detail=str(exc))
        return JSONResponse(
            status_code=422,
            content=_payload(code="value_error", detail=str(exc), run_id=run_id),
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        run_id = _resolve_run_id(request)
        log.exception(
            "unhandled_exception",
            path=request.url.path,
            exception_type=type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content=_payload(
                code="internal_error",
                detail="An unexpected error occurred while processing the request.",
                run_id=run_id,
            ),
        )
