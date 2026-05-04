from __future__ import annotations

from typing import Any

from app.services.text_encoder import (
    FALLBACK_MODEL_ID,
    MODEL_ID,
    aggregate_label,
    encode_chunks,
    get_classifier,
)

__all__ = [
    "MODEL_ID",
    "FALLBACK_MODEL_ID",
    "analyze_text",
    "get_classifier",
]


def analyze_text(text: str) -> dict[str, Any]:
    return aggregate_label(encode_chunks(text))
