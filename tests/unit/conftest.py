from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_history_db(tmp_path_factory):
    """Force every unit test that touches `app.db` onto a clean per-test SQLite
    file so test order can't flake the suite. We use `tmp_path_factory` (not
    `tmp_path`) so the db file lands in a separate directory; otherwise tests
    that enumerate `tmp_path` for JSON fixtures would pick up the db file."""

    pytest.importorskip("sqlalchemy", reason="sqlalchemy not installed")
    from app import db as db_module

    db_dir = tmp_path_factory.mktemp("history_db")
    db_module.reset_for_testing(f"sqlite:///{db_dir / 'fed_pulse_test.db'}")
    yield
