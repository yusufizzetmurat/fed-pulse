"""Thin wrapper around the Google Gemini SDK for FOMC-tone classification.

Exposes `score_passage(text, *, model)` that prompts the model with a
deterministic three-class instruction and returns a {label, confidence,
raw} dict. Caller-supplied `model` makes unit testing trivial via stubs.

Unparseable responses fall back to a neutral zero-confidence reading
rather than raising — the audit step counts parse failures separately
as a quality signal.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.services.langsmith_client import traced

_logger = logging.getLogger(__name__)

ALLOWED_LABELS = ("hawkish", "dovish", "neutral")

PROMPT_TEMPLATE = (
    "You classify U.S. Federal Open Market Committee passages by "
    "monetary-policy tone.\n"
    "Tone is exactly one of: hawkish, dovish, neutral.\n"
    'Return only a JSON object with two fields: "label" (one of the '
    'three tones) and "confidence" (a float in [0, 1]). No prose, no '
    "code fences, no explanation.\n\n"
    "Passage:\n<<<\n{passage}\n>>>"
)


def _parse_response(text: str) -> tuple[str, float]:
    """Extract (label, confidence) from a model response.

    Falls back to ('neutral', 0.0) on any parse failure or unknown label.
    """

    if not text:
        return "neutral", 0.0
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        payload = json.loads(cleaned)
    except Exception:
        return "neutral", 0.0
    if not isinstance(payload, dict):
        return "neutral", 0.0
    label = str(payload.get("label", "")).strip().lower()
    if label not in ALLOWED_LABELS:
        return "neutral", 0.0
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return label, max(0.0, min(1.0, confidence))


@traced("gemini.score_passage")
def score_passage(text: str, *, model: Any) -> dict[str, Any]:
    """Score one FOMC passage with the Gemini model.

    `model` must expose `generate_content(prompt, **kwargs)` returning
    an object with a `.text` attribute (matches both the real SDK
    adapter and the test stubs).
    """

    prompt = PROMPT_TEMPLATE.format(passage=text)
    response = model.generate_content(prompt)
    raw = getattr(response, "text", "") or ""
    label, confidence = _parse_response(raw)
    if confidence == 0.0:
        # A zero-confidence return out of ``_parse_response`` always
        # signals a parse / schema failure (the parser does not emit
        # zero on a real read). Surface the raw response so an operator
        # can spot prompt drift or model regressions without enabling
        # debug logging across the rest of the service.
        _logger.warning("gemini_parse_failure label=%s raw=%r", label, raw[:512])
    return {"label": label, "confidence": confidence, "raw": raw}


def load_model(model_name: str = "gemini-2.5-pro") -> _ModelAdapter:
    """Load a Gemini model. Imports the SDK lazily."""

    import os

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to fed-pulse/.env before running the live smoke."
        )

    from google import genai

    client = genai.Client(api_key=api_key)
    return _ModelAdapter(client, model_name)


class _ModelAdapter:
    """Adapter so the real SDK matches the stub interface (generate_content + .text)."""

    def __init__(self, client: Any, model_name: str):
        self._client = client
        self._model_name = model_name

    def generate_content(self, prompt: str, **kwargs: Any) -> Any:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        return response  # response has .text already


DEFAULT_EMBEDDING_DIM = 768


@traced("gemini.embed_text")
def embed_text(text: str, *, model: Any) -> list[float]:
    """Return a fixed-dim embedding for `text` using the supplied embedding model.

    `model` must expose `embed_content(content, **kwargs)` returning an
    object with `.embedding.values` (a list of floats). Empty input
    returns a zero vector of length DEFAULT_EMBEDDING_DIM rather than
    raising — callers can decide whether to skip those rows.
    """

    if not text:
        return [0.0] * DEFAULT_EMBEDDING_DIM
    response = model.embed_content(text)
    values = list(response.embedding.values)
    return [float(v) for v in values]


def load_embedding_model(model_name: str = "gemini-embedding-001") -> _EmbeddingAdapter:
    """Load a Gemini embedding model. Lazy import."""

    import os

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to fed-pulse/.env before running embedding precompute."
        )

    from google import genai

    client = genai.Client(api_key=api_key)
    return _EmbeddingAdapter(client, model_name)


class _EmbeddingAdapter:
    """Adapter so the real SDK matches the stub interface (embed_content + .embedding.values)."""

    def __init__(self, client: Any, model_name: str):
        self._client = client
        self._model_name = model_name

    def embed_content(self, content: str, **kwargs: Any) -> Any:
        response = self._client.models.embed_content(
            model=self._model_name,
            contents=content,
        )
        # The new google-genai client returns `.embeddings` (a list); we
        # adapt to the .embedding.values single shape used by the stub.
        values = response.embeddings[0].values
        wrapper = type("R", (), {"embedding": type("E", (), {"values": values})()})
        return wrapper()
