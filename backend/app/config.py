from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent


def _default_data_dir() -> Path:
    container_data = Path("/data")
    if container_data.exists():
        return container_data
    return REPO_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    data_dir: Path = Field(default_factory=_default_data_dir, validation_alias="FED_PULSE_DATA_DIR")
    model_checkpoint_dir: Path = Field(
        default=BACKEND_ROOT / "models",
        validation_alias="FED_PULSE_MODEL_CHECKPOINT_DIR",
    )
    log_level: str = Field(default="INFO", validation_alias="FED_PULSE_LOG_LEVEL")
    market_source: Literal["live", "snapshot"] = Field(
        default="live", validation_alias="FED_PULSE_MARKET_SOURCE"
    )

    hf_token: SecretStr | None = Field(default=None, validation_alias="HF_TOKEN")
    gemini_api_key: SecretStr | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    langsmith_api_key: SecretStr | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    voyage_api_key: SecretStr | None = Field(default=None, validation_alias="VOYAGE_API_KEY")
    kaggle_username: str | None = Field(default=None, validation_alias="KAGGLE_USERNAME")
    kaggle_key: SecretStr | None = Field(default=None, validation_alias="KAGGLE_KEY")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

DATA_DIR: Path = settings.data_dir
MODEL_CHECKPOINT_DIR: Path = settings.model_checkpoint_dir
