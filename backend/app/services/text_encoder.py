from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import pipeline

from app.models.registry import revision_for

MODEL_ID = "gtfintechlab/fomc-roberta-any-exp"
FALLBACK_MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_MAX_TOKENS = 480
DEFAULT_STRIDE = 400
DEFAULT_CLASSIFIER_MAX_LENGTH = 512

_classifier = None
_classifier_lock = threading.Lock()
_classifier_load_count = 0


@dataclass(frozen=True)
class ChunkEncoding:
    text: str
    embedding: list[float] = field(default_factory=list)
    scores: list[dict[str, float | str]] = field(default_factory=list)


def _resolve_pipeline_device() -> int:
    return 0 if torch.cuda.is_available() else -1


def _build_pipeline(model_id: str, device: int):
    kwargs: dict[str, Any] = {
        "model": model_id,
        "return_all_scores": True,
        "device": device,
    }
    revision = revision_for(model_id)
    if revision is not None:
        kwargs["revision"] = revision
    return pipeline("text-classification", **kwargs)


def get_classifier():
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
        for model_id, target_device in attempts:
            try:
                _classifier = _build_pipeline(model_id, target_device)
                global _classifier_load_count
                _classifier_load_count += 1
                break
            except Exception as exc:
                last_error = exc
        if _classifier is None and last_error is not None:
            raise last_error
    return _classifier


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


def split_into_chunks(
    text: str,
    classifier=None,
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


def _pool_cls_embedding(model, tokenizer, chunk: str) -> list[float]:
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


def encode_chunks(text: str, classifier=None) -> list[ChunkEncoding]:
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

    averaged = [
        {"label": label, "score": score / score_count}
        for label, score in aggregate.items()
    ]
    best = max(averaged, key=lambda item: float(item["score"]))

    return {
        "label": str(best["label"]),
        "score": float(best["score"]),
        "raw": [
            {"label": str(item["label"]), "score": float(item["score"])}
            for item in averaged
        ],
    }
