from __future__ import annotations

import logging
import os
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import pipeline

from app.models.registry import revision_for

# Local fine-tuned FinBERT-FOMC seed-71 checkpoint with hawkish/dovish/neutral
# labels (see config.json id2label). Used as the default sentiment classifier
# so the live /analyze flow does not silently fall back to a generic news
# sentiment model. Override via the FED_PULSE_SENTIMENT_MODEL env var on any
# deployment where the local checkpoint is not present.
DEFAULT_LOCAL_CHECKPOINT = "/data/artifacts/phase3/pilot_finetune_20260505T142652Z/hf_checkpoints"
PRIMARY_HF_MODEL_ID = "gtfintechlab/FOMC-RoBERTa"
# gtfintechlab/FOMC-RoBERTa (Trillion Dollar Words) ships generic LABEL_0/1/2 in
# its config. Empirically verified mapping (probs ~1.0 on hawkish/dovish/neutral
# probe sentences): LABEL_0 = dovish, LABEL_1 = hawkish, LABEL_2 = neutral. The
# repo is gated; a token with gate access is required (else the strict guard /
# loud-fallback path engages instead of silently using distilbert).
_FOMC_ROBERTA_ID2LABEL = {0: "dovish", 1: "hawkish", 2: "neutral"}
# Last-resort fallback ONLY. Returns POSITIVE / NEGATIVE labels, NOT
# hawkish/dovish/neutral, so the frontend should refuse to map the output
# (see frontend/lib/analyze/format.ts::toStance).
FALLBACK_MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# Resolved at import time. The override path is read once; restart the
# backend to pick up a new value.
_OVERRIDE = (os.environ.get("FED_PULSE_SENTIMENT_MODEL") or "").strip()
MODEL_ID = _OVERRIDE or (
    DEFAULT_LOCAL_CHECKPOINT if Path(DEFAULT_LOCAL_CHECKPOINT).exists() else PRIMARY_HF_MODEL_ID
)

# When truthy, refuse to silently use the generic fallback classifier — raise
# instead. Embedding/training pipelines set this so cached vectors can never be
# built from the wrong encoder (the silent-fallback contamination bug: the
# primary HF repo can 404, and the run would otherwise proceed on distilbert
# sentiment vectors with no error).
STRICT_PRIMARY = (os.environ.get("FED_PULSE_REQUIRE_PRIMARY_SENTIMENT") or "").strip().lower() in {
    "1",
    "true",
    "yes",
}

# The model id the singleton actually loaded; differs from MODEL_ID on fallback.
_loaded_model_id: str | None = None

DEFAULT_MAX_TOKENS = 480
DEFAULT_STRIDE = 400
DEFAULT_CLASSIFIER_MAX_LENGTH = 512

# Stance edge-case gates. The stance classifier was trained on English
# FOMC text and reports stale class probabilities on degenerate input;
# the gates below let :func:`analyze_text` short-circuit with a
# ``status`` flag instead of feeding the model rubbish. Thresholds
# match :mod:`app.services.semantic_diff` so the wire surface stays
# consistent across descriptive panels.
STANCE_MIN_INPUT_TOKENS: int = 5
STANCE_LATIN_RATIO_THRESHOLD: float = 0.5


def _classify_stance_input(text: str) -> str | None:
    """Bucket ``text`` for the silent-null stance edge cases.

    Returns ``"no_input"`` / ``"non_english"`` when the stance head
    should short-circuit, or ``None`` when the input is healthy
    enough to run through the classifier. Order matches
    :func:`app.services.semantic_diff._classify_input` so a long
    block of CJK reports as ``non_english`` rather than
    ``no_input``.
    """

    if not text or not text.strip():
        return "no_input"
    stripped = "".join(text.split())
    if stripped:
        latin = sum(1 for ch in stripped if ord(ch) < 256)
        if (latin / len(stripped)) < STANCE_LATIN_RATIO_THRESHOLD:
            return "non_english"
    tokens = text.split()
    if len(tokens) < STANCE_MIN_INPUT_TOKENS:
        return "no_input"
    return None


_classifier = None
_classifier_lock = threading.Lock()
_classifier_load_count = 0
_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkEncoding:
    text: str
    embedding: list[float] = field(default_factory=list)
    scores: list[dict[str, float | str]] = field(default_factory=list)


def _resolve_pipeline_device() -> int:
    return 0 if torch.cuda.is_available() else -1


def _build_pipeline(model_id: str, device: int) -> Any:
    kwargs: dict[str, Any] = {
        "model": model_id,
        "return_all_scores": True,
        "device": device,
    }
    revision = revision_for(model_id)
    if revision is not None:
        kwargs["revision"] = revision
    clf = pipeline("text-classification", **kwargs)
    # FOMC-RoBERTa exposes only generic LABEL_0/1/2; remap to stance names so the
    # pipeline emits hawkish/dovish/neutral (the labels the rest of the stack and
    # the frontend's toStance() expect). The local fine-tune checkpoint already
    # carries proper labels, so this only touches the FOMC-RoBERTa fallback.
    if model_id == PRIMARY_HF_MODEL_ID:
        clf.model.config.id2label = dict(_FOMC_ROBERTA_ID2LABEL)
        clf.model.config.label2id = {v: k for k, v in _FOMC_ROBERTA_ID2LABEL.items()}
    return clf


def get_classifier() -> Any:
    global _classifier
    if _classifier is not None:
        return _classifier

    with _classifier_lock:
        if _classifier is not None:
            return _classifier

        device = _resolve_pipeline_device()
        attempts = [
            (MODEL_ID, device),
            (FALLBACK_MODEL_ID, device),
        ]
        if device != -1:
            attempts.extend(
                [
                    (MODEL_ID, -1),
                    (FALLBACK_MODEL_ID, -1),
                ]
            )

        last_error: Exception | None = None
        loaded_model_id: str | None = None
        for model_id, target_device in attempts:
            try:
                _classifier = _build_pipeline(model_id, target_device)
                global _classifier_load_count
                _classifier_load_count += 1
                loaded_model_id = model_id
                break
            except Exception as exc:
                last_error = exc
                _logger.warning(
                    "sentiment.classifier_load_failed",
                    extra={
                        "model_id": model_id,
                        "device": target_device,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
        if _classifier is None and last_error is not None:
            raise last_error
        global _loaded_model_id
        _loaded_model_id = loaded_model_id
        if loaded_model_id != MODEL_ID:
            # We picked a fallback. The fallback's label space (e.g.
            # POSITIVE / NEGATIVE for distilbert-sst-2) is NOT
            # hawkish/dovish/neutral; the frontend's toStance() refuses to
            # silently relabel POSITIVE -> hawkish, so the dashboard will
            # surface "Sentiment unavailable" until the primary model loads.
            if STRICT_PRIMARY:
                raise RuntimeError(
                    f"sentiment classifier fell back to {loaded_model_id!r} instead of "
                    f"the primary {MODEL_ID!r}; refusing because "
                    "FED_PULSE_REQUIRE_PRIMARY_SENTIMENT is set (the primary model is "
                    "unavailable — see preceding classifier_load_failed warnings)"
                )
            _logger.error(
                "sentiment.classifier_using_fallback",
                extra={
                    "primary_model_id": MODEL_ID,
                    "loaded_model_id": loaded_model_id,
                    "note": (
                        "Primary sentiment model failed to load. The active model's labels "
                        "are NOT hawkish/dovish/neutral; the dashboard will report stance "
                        "as unknown. Inspect the preceding classifier_load_failed warnings."
                    ),
                },
            )
    return _classifier


def get_loaded_model_id() -> str | None:
    """The model id the singleton actually loaded (differs from MODEL_ID on fallback)."""
    return _loaded_model_id


def assert_primary_model_loaded() -> None:
    """Raise unless the active classifier is the primary model, not the fallback.

    Build/training pipelines call this before producing cached embeddings so a
    silent fallback (e.g. the primary HF repo 404ing -> distilbert) can never
    contaminate the artifacts undetected.
    """
    get_classifier()
    if _loaded_model_id != MODEL_ID:
        raise RuntimeError(
            f"primary sentiment model {MODEL_ID!r} is not loaded "
            f"(active model: {_loaded_model_id!r}); refusing to build embeddings with "
            "a fallback encoder. Set FED_PULSE_SENTIMENT_MODEL to a valid model."
        )


def classifier_load_count() -> int:
    """How many times the underlying HF pipeline has been instantiated.

    Used by the lifespan-cache test to assert the model only loads once across
    repeated `/analyze` calls.
    """

    return _classifier_load_count


def warmup_classifier() -> None:
    """Force-load the classifier so the first `/analyze` request doesn't pay
    the cold-start cost. Called from the FastAPI lifespan."""

    get_classifier()


def resolve_ood_manifest_path() -> Path | None:
    """Locate the OOD calibration manifest beside the active checkpoint.

    Returns None when MODEL_ID is an HF hub id (no local manifest) or
    when no manifest has been written yet. Callers should treat the
    absence of a manifest as 'no OOD signal' and not surface ood_*
    fields on the response.
    """

    from app.evaluation.ood import OOD_MANIFEST_NAME  # lazy import: avoids cycle

    candidate = Path(MODEL_ID)
    if candidate.exists() and candidate.is_dir():
        manifest_path = candidate / OOD_MANIFEST_NAME
        if manifest_path.exists():
            return manifest_path
    return None


def split_into_chunks(
    text: str,
    classifier: Any = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    stride: int = DEFAULT_STRIDE,
) -> list[str]:
    if not text:
        return []
    if classifier is None:
        classifier = get_classifier()
    tokenizer = getattr(classifier, "tokenizer", None)
    if tokenizer is None:
        return [text]

    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if len(token_ids) <= max_tokens:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(token_ids):
            break
        start += stride
    return chunks or [text]


def _normalize_scores(output: Any) -> list[dict[str, float | str]]:
    if isinstance(output, list) and output and isinstance(output[0], list):
        output = output[0]

    if not isinstance(output, list):
        return []

    normalized: list[dict[str, float | str]] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "label": str(item.get("label", "")),
                "score": float(item.get("score", 0.0)),
            }
        )
    return normalized


def _pool_cls_embedding(model: Any, tokenizer: Any, chunk: str) -> list[float]:
    device = next(model.parameters()).device
    with torch.no_grad():
        inputs = tokenizer(
            chunk,
            truncation=True,
            max_length=DEFAULT_CLASSIFIER_MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(device)
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        cls_vec = hidden[:, 0, :].squeeze(0).detach().cpu().tolist()
    return list(cls_vec)


def encode_chunks(text: str, classifier: Any = None) -> list[ChunkEncoding]:
    if not text:
        return []
    if classifier is None:
        classifier = get_classifier()
    chunks = split_into_chunks(text, classifier=classifier)
    if not chunks:
        return []

    tokenizer = getattr(classifier, "tokenizer", None)
    model = getattr(classifier, "model", None)

    encodings: list[ChunkEncoding] = []
    for chunk in chunks:
        scores = _normalize_scores(
            classifier(chunk, truncation=True, max_length=DEFAULT_CLASSIFIER_MAX_LENGTH)
        )
        embedding: list[float] = []
        if model is not None and tokenizer is not None:
            embedding = _pool_cls_embedding(model, tokenizer, chunk)
        encodings.append(ChunkEncoding(text=chunk, embedding=embedding, scores=scores))
    return encodings


def analyze_text(text: str) -> dict[str, Any]:
    """Score ``text`` for stance + (when an OOD manifest is available)
    attach energy-based OOD diagnostics to the response.

    Issue #339 deliverable #2: the legacy ``app.services.sentiment``
    module was retired in favour of routing every caller through this
    function. When a :class:`TextMultiAxisClassifier` checkpoint is
    available (``app.services.multi_axis_classifier``), its stance head
    is the source of truth so the response carries the same per-axis
    prediction the model card publishes; otherwise the legacy chunked
    classifier remains the fallback. Returned shape stays
    ``{label, score, raw[]}`` so the /analyze + prepare_training_data
    + attention_ablation surfaces stay drop-in.

    Edge-case contract: empty / whitespace-only / majority-non-Latin
    inputs never reach the classifier. The function returns the
    standard ``{label: "UNKNOWN", score: 0.0, raw: []}`` block with
    an extra ``status`` key so callers can surface a parseable
    informational banner instead of a misleading stance label. The
    classifier path otherwise returns ``status="ok"``.
    """

    text_value = text or ""
    edge_status = _classify_stance_input(text_value)
    if edge_status is not None:
        return {"label": "UNKNOWN", "score": 0.0, "raw": [], "status": edge_status}
    response = _stance_from_multi_axis(text_value)
    if response is None:
        response = aggregate_label(encode_chunks(text_value))
    response.setdefault("status", "ok")

    manifest_path = resolve_ood_manifest_path()
    if manifest_path is None:
        return response

    # Lazy import: keep torch + numpy out of the import-only path.
    from app.evaluation.ood import load_manifest, score_text as score_text_ood

    manifest = load_manifest(manifest_path)
    if manifest is None:
        return response

    classifier = get_classifier()
    chunks = split_into_chunks(text_value, classifier=classifier)
    ood = score_text_ood(text_value, classifier=classifier, manifest=manifest, chunks=chunks)
    response["ood_energy"] = ood.get("ood_energy")
    response["ood_threshold"] = ood.get("ood_threshold")
    response["is_in_distribution"] = ood.get("is_in_distribution")
    return response


def _stance_from_multi_axis(text: str) -> dict[str, Any] | None:
    """Prefer the multi-axis stance head when a checkpoint is loaded.

    Returns the legacy ``{label, score, raw[]}`` dict shape so the
    /analyze surface and downstream callers do not need to special-case
    the source. ``None`` means "no checkpoint, fall back to the chunked
    classifier" rather than "no signal".
    """

    text_value = (text or "").strip()
    if not text_value:
        return None
    try:
        from app.services.multi_axis_classifier import (
            checkpoint_exists as _ma_checkpoint_exists,
            score_text as _ma_score_text,
        )
    except Exception:  # pragma: no cover -- defensive
        return None
    if not _ma_checkpoint_exists():
        return None
    try:
        block = _ma_score_text(text_value)
    except Exception:  # pragma: no cover -- never let inference crash
        _logger.warning("multi_axis_stance_route_failed", exc_info=True)
        return None
    if not block:
        return None
    stance = block.get("stance") if isinstance(block, dict) else None
    if not stance:
        return None
    distribution = stance.get("distribution") or {}
    raw = [{"label": str(name), "score": float(score)} for name, score in distribution.items()]
    return {
        "label": str(stance.get("label", "")),
        "score": float(stance.get("confidence", 0.0) or 0.0),
        "raw": raw,
    }


def aggregate_label(encodings: list[ChunkEncoding]) -> dict[str, Any]:
    if not encodings:
        return {"label": "UNKNOWN", "score": 0.0, "raw": []}

    aggregate: dict[str, float] = defaultdict(float)
    score_count = 0
    for encoding in encodings:
        if not encoding.scores:
            continue
        score_count += 1
        for item in encoding.scores:
            label = str(item["label"])
            aggregate[label] += float(item["score"])

    if not aggregate or score_count == 0:
        return {"label": "UNKNOWN", "score": 0.0, "raw": []}

    averaged: list[dict[str, float | str]] = [
        {"label": label, "score": score / score_count} for label, score in aggregate.items()
    ]
    best = max(averaged, key=lambda item: float(item["score"]))

    return {
        "label": str(best["label"]),
        "score": float(best["score"]),
        "raw": [{"label": str(item["label"]), "score": float(item["score"])} for item in averaged],
    }
