from __future__ import annotations

from pathlib import Path

import pytest


def test_default_data_dir_is_a_path() -> None:
    from app.config import DATA_DIR

    assert isinstance(DATA_DIR, Path)


def test_settings_singleton_round_trip() -> None:
    from app.config import get_settings

    first = get_settings()
    second = get_settings()
    assert first is second


def test_fed_pulse_data_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from app import config

    override = tmp_path / "custom-data"
    override.mkdir()
    monkeypatch.setenv("FED_PULSE_DATA_DIR", str(override))
    config.get_settings.cache_clear()

    fresh = config.get_settings()
    assert fresh.data_dir == override


def test_hf_token_env_loads_as_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import config
    from pydantic import SecretStr

    monkeypatch.setenv("HF_TOKEN", "test-token-not-real")
    config.get_settings.cache_clear()
    fresh = config.get_settings()
    assert isinstance(fresh.hf_token, SecretStr)
    assert fresh.hf_token.get_secret_value() == "test-token-not-real"
