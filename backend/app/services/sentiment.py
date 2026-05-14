from __future__ import annotations

from typing import Any

from app.services.text_encoder import (
    FALLBACK_MODEL_ID,
    MODEL_ID,
    aggregate_label,
    encode_chunks,
    get_classifier,
    resolve_ood_manifest_path,
    split_into_chunks,
)

__all__ = [
    "MODEL_ID",
    "FALLBACK_MODEL_ID",
    "analyze_text",
    "get_classifier",
]


def analyze_text(text: str) -> dict[str, Any]:
    """Score `text` for stance + (when an OOD manifest is available) attach
    energy-based OOD diagnostics to the response."""

    response = aggregate_label(encode_chunks(text))

    manifest_path = resolve_ood_manifest_path()
    if manifest_path is None:
        return response

    from app.evaluation.ood import load_manifest, score_text  # lazy: keep torch out of import-only paths

    manifest = load_manifest(manifest_path)
    if manifest is None:
        return response

    classifier = get_classifier()
    chunks = split_into_chunks(text, classifier=classifier)
    ood = score_text(text, classifier=classifier, manifest=manifest, chunks=chunks)
    response["ood_energy"] = ood.get("ood_energy")
    response["ood_threshold"] = ood.get("ood_threshold")
    response["is_in_distribution"] = ood.get("is_in_distribution")
    return response
