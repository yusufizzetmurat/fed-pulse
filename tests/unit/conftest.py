from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_history_db():
    """Force every unit test that touches `app.db` onto a clean in-memory SQLite
    so test order can't flake the suite via a half-initialised file DB."""

    sqlalchemy = pytest.importorskip("sqlalchemy", reason="sqlalchemy not installed")  # noqa: F841
    from app import db as db_module

    db_module.reset_for_testing("sqlite:///:memory:")
    yield
