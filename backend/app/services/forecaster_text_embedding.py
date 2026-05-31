"""Pooled text embedding helper for the live forecaster serving path.

The training pipeline reads per-event embeddings from precomputed
parquet caches keyed by encoder alias. The live ``/analyze`` path
has no cache lookup for arbitrary user-pasted text, so we load the
encoder on demand, mean-pool its last hidden state over non-pad
tokens, and return the embedding as a Python list. The caller
passes the list through ``build_feature_vectors``'s
``text_embedding`` kwarg; ``FeatureVector.from_market_state``
forwards it onto ``text_embedding_pooled`` (the field the serving
forward path actually reads). Modifying the channel from this end
requires touching both hops, not just this one.

Singleton-cached after first load (one tokenizer + model + device).
Default encoder is ``finbert_fed_adjacent`` pinned via the
registry; override with ``FED_PULSE_FORECASTER_TEXT_ENCODER`` if a
different alias becomes the production choice.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from app.models.registry import encoder_ref

LOGGER = logging.getLogger(__name__)

DEFAULT_ENCODER_ALIAS = (
    os.environ.get("FED_PULSE_FORECASTER_TEXT_ENCODER") or "finbert_fed_adjacent"
).strip()

# Trim very long pastes before tokenisation so a 10MB body cannot
# stall the encoder.
MAX_TEXT_CHARS = 12_000
MAX_TOKENS = 512


@dataclass(frozen=True)
class _PooledEncoderState:
    encoder_alias: str
    tokenizer: Any
    model: Any
    device: torch.device


_state: _PooledEncoderState | None = None
_state_lock = threading.Lock()


def _load_state() -> _PooledEncoderState | None:
    global _state
    if _state is not None:
        return _state
    with _state_lock:
        if _state is not None:
            return _state
        try:
            ref = encoder_ref(DEFAULT_ENCODER_ALIAS)
        except Exception as exc:  # registry miss
            LOGGER.warning("forecaster_text_embedding: encoder_ref(%s) failed: %s", DEFAULT_ENCODER_ALIAS, exc)
            return None
        if ref is None or not ref.repo:
            LOGGER.warning("forecaster_text_embedding: alias %s unpinned", DEFAULT_ENCODER_ALIAS)
            return None
        kwargs: dict[str, Any] = {}
        if ref.revision:
            kwargs["revision"] = ref.revision
        if getattr(ref, "trust_remote_code", False):
            kwargs["trust_remote_code"] = True
        try:
            tokenizer = AutoTokenizer.from_pretrained(ref.repo, **kwargs)  # type: ignore[no-untyped-call]
            model = AutoModel.from_pretrained(ref.repo, **kwargs)
        except Exception as exc:
            LOGGER.warning(
                "forecaster_text_embedding: failed to load encoder %s @ %s: %s",
                ref.repo, ref.revision, exc,
            )
            return None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        _state = _PooledEncoderState(
            encoder_alias=DEFAULT_ENCODER_ALIAS,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )
        LOGGER.info(
            "forecaster_text_embedding: loaded %s @ %s on %s (hidden_size=%s)",
            ref.repo, ref.revision, device, getattr(model.config, "hidden_size", "?"),
        )
        return _state


@torch.no_grad()
def encode_text_pooled(text: str) -> list[float] | None:
    """Return a mean-pooled embedding for ``text`` as a flat list of floats.

    ``None`` when the encoder cannot be loaded or the text is empty
    after trimming. Mean-pool is computed over non-pad tokens.
    """

    cleaned = (text or "").strip()[:MAX_TEXT_CHARS]
    if not cleaned:
        return None
    state = _load_state()
    if state is None:
        return None
    enc = state.tokenizer(
        cleaned,
        max_length=MAX_TOKENS,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(state.device)
    attention_mask = enc["attention_mask"].to(state.device)
    outputs = state.model(input_ids=input_ids, attention_mask=attention_mask)
    hidden = outputs.last_hidden_state  # (1, T, H)
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    pooled_cpu = (summed / counts).squeeze(0).cpu().tolist()
    # Free the on-device intermediates explicitly so a long-lived server
    # process does not stack ~1.5 MB of GPU residency per call until the
    # next Python GC pass.
    del input_ids, attention_mask, outputs, hidden, mask, summed, counts
    return [float(v) for v in pooled_cpu]


def get_loaded_encoder_alias() -> str | None:
    """For diagnostics: which encoder alias the singleton is holding."""

    return _state.encoder_alias if _state is not None else None


def reset_state() -> None:
    """Drop the cached singleton (used by tests)."""

    global _state
    with _state_lock:
        _state = None
