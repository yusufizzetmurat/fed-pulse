from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

from transformers import pipeline

MODEL_ID = "gtfintechlab/fomc-roberta-any-exp"
FALLBACK_MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
_classifier = None
_classifier_lock = threading.Lock()


def _get_classifier():
    global _classifier
    if _classifier is not None:
        return _classifier

    with _classifier_lock:
        if _classifier is not None:
            return _classifier

        try:
            _classifier = pipeline("text-classification", model=MODEL_ID, return_all_scores=True)
        except Exception:
            # Keep MVP usable if the target model is unavailable or private.
            _classifier = pipeline("text-classification", model=FALLBACK_MODEL_ID, return_all_scores=True)
    return _classifier


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


def _split_into_chunks(text: str, classifier, max_tokens: int = 480, stride: int = 400) -> list[str]:
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


def analyze_text(text: str) -> dict[str, Any]:
    classifier = _get_classifier()
    chunks = _split_into_chunks(text, classifier)
    aggregate: dict[str, float] = defaultdict(float)

    for chunk in chunks:
        outputs = classifier(chunk, truncation=True, max_length=512)
        for item in _normalize_scores(outputs):
            label = str(item["label"])
            aggregate[label] += float(item["score"])

    if not aggregate:
        return {"label": "UNKNOWN", "score": 0.0, "raw": []}

    averaged = [
        {"label": label, "score": score / len(chunks)}
        for label, score in aggregate.items()
    ]
    best = max(averaged, key=lambda item: float(item["score"]))

    return {
        "label": str(best["label"]),
        "score": float(best["score"]),
        "raw": [{"label": str(item["label"]), "score": float(item["score"])} for item in averaged],
    }
