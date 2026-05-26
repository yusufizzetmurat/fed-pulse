from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("structlog")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.middleware.errors import RunIdMiddleware, register_error_handlers


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RunIdMiddleware)
    register_error_handlers(app)

    @app.get("/boom")
    def _boom() -> dict:
        raise RuntimeError("synthetic explosion")

    @app.get("/value-error")
    def _value_error() -> dict:
        raise ValueError("bad input")

    @app.get("/ok")
    def _ok() -> dict:
        return {"ok": True}

    return app


def test_unhandled_exception_returns_structured_payload():
    client = TestClient(_build_app(), raise_server_exceptions=False)
    response = client.get("/boom")
    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "internal_error"
    assert body["run_id"] is not None
    assert "An unexpected error" in body["detail"]


def test_value_error_maps_to_422():
    client = TestClient(_build_app(), raise_server_exceptions=False)
    response = client.get("/value-error")
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "value_error"
    assert body["detail"] == "bad input"


def test_ok_route_returns_x_run_id_header():
    client = TestClient(_build_app())
    response = client.get("/ok")
    assert response.status_code == 200
    assert response.headers.get("x-run-id")
