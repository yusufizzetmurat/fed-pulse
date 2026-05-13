from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import REPO_ROOT

SCHEMA_PATH = REPO_ROOT / "data" / "schema" / "labels.yaml"


class Stance(str, Enum):
    HAWKISH = "hawkish"
    DOVISH = "dovish"
    NEUTRAL = "neutral"


class Topic(str, Enum):
    INFLATION = "inflation"
    EMPLOYMENT = "employment"
    FINANCIAL_STABILITY = "financial_stability"
    GROWTH = "growth"


class MultiAxisLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stance: Stance
    factor: float | None = Field(default=None, ge=-1.0, le=1.0)
    certainty: float | None = Field(default=None, ge=0.0, le=1.0)
    topic: Topic | None = None

    @field_validator("factor", "certainty")
    @classmethod
    def _coerce_nan(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value != value:  # NaN check
            return None
        return value


@lru_cache(maxsize=1)
def load_schema(path: Path | None = None) -> dict[str, Any]:
    target = path or SCHEMA_PATH
    return yaml.safe_load(target.read_text(encoding="utf-8")) or {}


def sample_weight_for(provenance: str) -> float:
    weights = (load_schema().get("provenance") or {}).get("sample_weights") or {}
    return float(weights.get(provenance, 0.0))
