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

    @field_validator("factor", "certainty", mode="before")
    @classmethod
    def _coerce_nan(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, float) and value != value:  # NaN
            return None
        return value


@lru_cache(maxsize=1)
def load_schema(path: Path | None = None) -> dict[str, Any]:
    target = path or SCHEMA_PATH
    return yaml.safe_load(target.read_text(encoding="utf-8")) or {}


def sample_weight_for(provenance: str) -> float:
    weights = (load_schema().get("provenance") or {}).get("sample_weights") or {}
    return float(weights.get(provenance, 0.0))


def auxiliary_axis_weight_for(provenance: str) -> float:
    """Inclusion weight for auxiliary (non-stance) axes per provenance bucket.

    ``sample_weight_for`` is the strict-FOMC gate on the stance head: it
    returns 0.0 for ``peer_reviewed_cross_bank`` rows so they never
    contribute to FOMC stance supervision. The auxiliary axes
    (certainty / topic / factor / time) do not carry the same
    FOMC-distribution concern — the cross-bank corpora are useful
    encoder-side signal for those axes. This helper returns 1.0 for
    ``peer_reviewed_cross_bank`` so the encoder fine-tune can route
    those rows through the auxiliary heads even when their stance is
    masked, and otherwise falls back to the provenance-keyed sample
    weight so non-cross-bank buckets behave identically to the
    existing gate.
    """

    if provenance == "peer_reviewed_cross_bank":
        return 1.0
    return sample_weight_for(provenance)
