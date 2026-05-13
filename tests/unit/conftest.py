from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_history_db(tmp_path):
    """Force every unit test that touches `app.db` onto a clean per-test SQLite
    file so test order can't flake the suite and so every new SQLAlchemy
    connection sees the same `analysis_runs` table (in-memory `:memory:` would
    isolate each connection)."""

    pytest.importorskip("sqlalchemy", reason="sqlalchemy not installed")
    from app import db as db_module

    db_module.reset_for_testing(f"sqlite:///{tmp_path / 'fed_pulse_test.db'}")
    yield
