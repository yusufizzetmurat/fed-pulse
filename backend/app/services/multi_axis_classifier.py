"""Multi-axis text classifier inference service (#78 follow-up).

Wraps the trained ``TextMultiAxisClassifier`` checkpoint behind a
thread-safe singleton so the FastAPI handler at ``/analyze`` can
emit per-axis predictions without paying the cold-start cost on
every request. Mirrors the pattern in
``app.services.text_encoder.get_classifier``.

The classifier is treated as optional: if no checkpoint exists at
the configured path the service returns ``None`` for every prediction
and the /analyze handler falls back to populating only the stance
card from the legacy sentiment classifier. Cold-start training is
NOT triggered automatically — the classifier consumes a fixed
supervised corpus (events.parquet) and is trained out-of-band via
``python -m app.data.train_text_multi_axis_classifier``.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from app.config import MODEL_CHECKPOINT_DIR
from app.models.config import (
    MULTI_TASK_CERTAINTY_LABELS,
    MULTI_TASK_STANCE_LABELS,
    MULTI_TASK_TOPIC_LABELS,
)
from app.models.text_multi_axis_classifier import TextMultiAxisClassifier

DEFAULT_CHECKPOINT_PATH = MODEL_CHECKPOINT_DIR / "text_multi_axis_best.pt"
DEFAULT_MAX_LENGTH = 256

_logger = logging.getLogger(__name__)
_state: "_ClassifierState | None" = None
_state_lock = threading.Lock()


@dataclass(frozen=True)
class _ClassifierState:
    model: TextMultiAxisClassifier
    tokenizer: Any
    device: torch.device
    max_length: int
    encoder_alias: str


def _resolve_checkpoint_path() -> Path:
    override = (os.environ.get("FED_PULSE_TEXT_MULTI_AXIS_CHECKPOINT") or "").strip()
    if override:
        return Path(override)
    return DEFAULT_CHECKPOINT_PATH


def checkpoint_exists() -> bool:
    """Best-effort probe used by /analyze to decide whether to invoke the classifier."""

    return _resolve_checkpoint_path().exists()


def _load_state() -> _ClassifierState | None:
    """Build the singleton from the checkpoint payload.

    Returns ``None`` when the checkpoint is missing — the caller then
    routes /analyze through the legacy stance-only path without
    raising. Any other failure (missing transformers, malformed
    payload) is logged and also returns ``None`` so a broken
    classifier never blocks the rest of the analyze flow.
    """

    path = _resolve_checkpoint_path()
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        _logger.warning("multi_axis_checkpoint_load_failed path=%s", path, exc_info=True)
        return None

    metadata = payload.get("metadata") or {}
    encoder_alias = str(metadata.get("encoder_alias") or "finbert_fed_adjacent")
    head_hidden_size = int(metadata.get("head_hidden_size") or 128)
    dropout = float(metadata.get("dropout") or 0.1)

    try:
        from transformers import AutoTokenizer

        from app.models.registry import encoder_ref

        ref = encoder_ref(encoder_alias)
        if ref is None or not ref.revision:
            raise ValueError(
                f"Encoder alias {encoder_alias!r} is unpinned in registry.yaml"
            )
        tokenizer = AutoTokenizer.from_pretrained(ref.repo, revision=ref.revision)
        model = TextMultiAxisClassifier.from_encoder_alias(
            encoder_alias=encoder_alias,
            head_hidden_size=head_hidden_size,
            dropout=dropout,
        )
        state_dict = payload.get("model_state_dict") or {}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            _logger.warning(
                "multi_axis_checkpoint_partial_load missing=%d unexpected=%d",
                len(missing),
                len(unexpected),
            )
    except Exception:
        _logger.warning(
            "multi_axis_classifier_build_failed path=%s", path, exc_info=True
        )
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    max_length = int(
        (payload.get("training_args") or {}).get("max_length") or DEFAULT_MAX_LENGTH
    )
    return _ClassifierState(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        encoder_alias=encoder_alias,
    )


def get_loaded_encoder_alias() -> str | None:
    """Encoder alias backing the loaded classifier, or None when absent.

    Used by /analyze diagnostics to surface ``model.encoder_key`` so the
    workspace status bar and pipeline trace can show which encoder is
    live without poking at the file system.
    """

    state = _state
    if state is None:
        return None
    return state.encoder_alias


def get_classifier() -> _ClassifierState | None:
    """Return the lazily-loaded classifier singleton (or None when absent).

    Thread-safe: callers that race on first use see one load. Subsequent
    callers read the cached state without acquiring the lock.
    """

    global _state
    if _state is not None:
        return _state
    with _state_lock:
        if _state is None:
            _state = _load_state()
        return _state


def reset_classifier() -> None:
    """Drop the singleton so the next call rebuilds (test hook + post-train refresh)."""

    global _state
    with _state_lock:
        _state = None


@torch.no_grad()
def score_text(text: str) -> dict[str, Any] | None:
    """Run the classifier on ``text`` and return the per-axis prediction block.

    Returns ``None`` when no checkpoint is loaded. The output shape
    matches the ``MultiAxisBlock`` Pydantic schema in
    ``app.schemas`` — keys for stance / factor / certainty / topic
    each carrying ``label``, ``confidence``, and (where applicable)
    a per-class distribution.
    """

    state = get_classifier()
    if state is None:
        return None
    text = (text or "").strip()
    if not text:
        return None
    encoded = state.tokenizer(
        text,
        max_length=state.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(state.device)
    attention_mask = encoded["attention_mask"].to(state.device)
    logits = state.model(input_ids=input_ids, attention_mask=attention_mask)

    stance_probs = torch.softmax(logits["stance"], dim=-1)[0]
    stance_idx = int(stance_probs.argmax().item())
    stance_label = MULTI_TASK_STANCE_LABELS[stance_idx]
    stance_dist = {
        MULTI_TASK_STANCE_LABELS[i]: float(stance_probs[i].item())
        for i in range(len(MULTI_TASK_STANCE_LABELS))
    }

    factor_value = float(logits["factor"][0].item())

    certainty_probs = torch.softmax(logits["certainty"], dim=-1)[0]
    certainty_idx = int(certainty_probs.argmax().item())
    certainty_label = MULTI_TASK_CERTAINTY_LABELS[certainty_idx]
    certainty_dist = {
        MULTI_TASK_CERTAINTY_LABELS[i]: float(certainty_probs[i].item())
        for i in range(len(MULTI_TASK_CERTAINTY_LABELS))
    }

    topic_probs = torch.softmax(logits["topic"], dim=-1)[0]
    topic_idx = int(topic_probs.argmax().item())
    topic_label = MULTI_TASK_TOPIC_LABELS[topic_idx]
    topic_dist = {
        MULTI_TASK_TOPIC_LABELS[i]: float(topic_probs[i].item())
        for i in range(len(MULTI_TASK_TOPIC_LABELS))
    }

    return {
        "stance": {
            "label": stance_label,
            "confidence": float(stance_probs[stance_idx].item()),
            "distribution": stance_dist,
        },
        "factor": {
            "value": max(-1.0, min(1.0, factor_value)),
            # Factor regression confidence is not a probability; for now
            # we emit the absolute value as a proxy ("how far from
            # neutral") and the frontend renders it as the bar
            # magnitude. Calibration is a follow-up.
            "confidence": min(1.0, abs(factor_value)),
        },
        "certainty": {
            "label": certainty_label,
            "confidence": float(certainty_probs[certainty_idx].item()),
            "distribution": certainty_dist,
        },
        "topic": {
            "label": topic_label,
            "confidence": float(topic_probs[topic_idx].item()),
            "distribution": topic_dist,
        },
    }
