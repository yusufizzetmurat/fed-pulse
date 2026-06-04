"""Mahalanobis-distance OOD detection for the sentiment classifier.

The energy-based detector in :mod:`app.evaluation.ood` scores OOD by reading
classifier logit sharpness. FinBERT-FOMC happens to label clearly off-domain
text (recipes, code, generic news) with maximum confidence as ``neutral``,
so the energy signal does not separate FOMC from non-FOMC.

The standard OOD detector that does work on confidently miscalibrated
fine-tuned classifiers is Mahalanobis distance in the encoder's CLS
embedding space (Lee et al., NeurIPS 2018,
"A Simple Unified Framework for Detecting OOD Samples and Adversarial
Attacks"). Per class ``c``, fit a mean ``μ_c`` over the training-set CLS
embeddings. Fit a single tied covariance ``Σ`` across the whole training
set. For a new input ``x`` with CLS embedding ``z(x)``, compute

    d(x) = min_c  (z(x) − μ_c)ᵀ Σ⁻¹ (z(x) − μ_c)

and gate on a threshold calibrated to the training-set distance
distribution. The min over classes is the standard reduction; it asks
"is this point near any known class centroid".

This module exposes the building blocks. The serving path in
:mod:`app.services.text_encoder` loads the manifest produced by
``scripts/calibrate_ood_mahalanobis.py`` and calls
:func:`score_text_mahalanobis` per /analyze request.

The two manifests are independent. The Mahalanobis manifest sits at
``forecaster_best.ood_mahalanobis.json`` next to the checkpoint;
the energy manifest at ``forecaster_best.ood.json``. The serving
path prefers Mahalanobis when both are present.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

OOD_MAHALANOBIS_MANIFEST_NAME = "forecaster_best.ood_mahalanobis.json"
DEFAULT_THRESHOLD_PERCENTILE = 95.0
DEFAULT_SHRINKAGE = 0.01  # ridge added to Σ before inversion for numerical stability


@dataclass(frozen=True)
class MahalanobisManifest:
    """Persisted Mahalanobis OOD detector.

    ``class_means`` is a list of per-class mean vectors in embedding space
    (length n_classes; each vector has dim = hidden_size). ``cov_inverse``
    is the tied covariance inverse Σ⁻¹ flattened row-major. ``threshold``
    is the in-domain Mahalanobis distance ceiling; any input whose
    min-over-classes distance exceeds this is flagged out-of-distribution.
    """

    model_id: str
    embedding_dim: int
    class_labels: list[str]
    class_means: list[list[float]]
    cov_inverse: list[list[float]]
    threshold: float
    percentile: float
    shrinkage: float
    training_corpus_size: int
    training_distance_mean: float
    training_distance_std: float
    training_distance_min: float
    training_distance_max: float
    calibrated_at_utc: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def extract_cls_embedding(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    max_length: int = 512,
) -> torch.Tensor | None:
    """Return the pooled CLS embedding for ``text``.

    Looks at the top-layer hidden state at position 0. Returns ``None``
    on empty input or when the model does not expose ``output_hidden_states``.
    The tensor is detached and moved to CPU before return.
    """

    if not text or not text.strip():
        return None
    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        device = torch.device("cpu")
    raw_inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    if hasattr(raw_inputs, "to"):
        inputs = raw_inputs.to(device)
    elif isinstance(raw_inputs, dict):
        inputs = {key: value.to(device) for key, value in raw_inputs.items()}
    else:
        inputs = raw_inputs
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        return None
    # Last layer, batch index 0, position 0 (CLS).
    cls_vec: torch.Tensor = hidden_states[-1][0, 0]
    return cls_vec.detach().to("cpu")


def mahalanobis_distance(
    embedding: torch.Tensor,
    class_means: torch.Tensor,
    cov_inverse: torch.Tensor,
) -> tuple[float, int]:
    """Return ``(min_distance, argmin_class)``.

    ``class_means`` has shape ``[n_classes, d]``; ``cov_inverse`` has
    shape ``[d, d]``. The distance is computed for every class centroid
    and the minimum is returned. ``embedding`` should be the CLS vector
    from :func:`extract_cls_embedding`.
    """

    diffs = class_means - embedding  # [n_classes, d]
    quad = torch.einsum("nd,de,ne->n", diffs, cov_inverse, diffs)  # [n_classes]
    quad = quad.clamp(min=0.0)
    min_dist, min_idx = torch.min(quad, dim=0)
    return float(min_dist.item()), int(min_idx.item())


def fit_class_statistics(
    embeddings: torch.Tensor,
    labels: list[int],
    *,
    shrinkage: float = DEFAULT_SHRINKAGE,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Fit per-class means and tied covariance from training embeddings.

    Returns ``(class_means, cov_inverse, class_order)``. ``class_order``
    is the sorted unique label ids the means follow. The covariance is
    pooled within-class with Ledoit-Wolf-style ridge shrinkage:
    ``Σ_shrunk = (1 - α) Σ + α tr(Σ)/d * I`` where ``α = shrinkage``.
    This keeps the inverse numerically stable when the corpus is small
    relative to the embedding dimension (768-dim FinBERT on 5k rows is
    tight but workable).
    """

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {tuple(embeddings.shape)}")
    if embeddings.shape[0] != len(labels):
        raise ValueError("embeddings rows must match labels length")

    embeddings = embeddings.to(torch.float64)
    unique_labels = sorted(set(labels))
    means = []
    centered_parts = []
    n_total = 0
    for cls in unique_labels:
        idx = [i for i, label in enumerate(labels) if label == cls]
        if not idx:
            continue
        group = embeddings[idx]
        mu = group.mean(dim=0)
        means.append(mu)
        centered_parts.append(group - mu)
        n_total += group.shape[0]
    class_means = torch.stack(means, dim=0)  # [n_classes, d]
    centered = torch.cat(centered_parts, dim=0)
    cov = (centered.T @ centered) / max(n_total - len(unique_labels), 1)

    d = cov.shape[0]
    diag_avg = float(torch.diagonal(cov).mean().item())
    ridge = max(shrinkage, 1e-6)
    cov_shrunk = (1.0 - ridge) * cov + ridge * diag_avg * torch.eye(d, dtype=cov.dtype)
    # Compute the inverse via Cholesky → triangular solve for numerical safety.
    cov_inverse = torch.linalg.inv(cov_shrunk)
    return class_means.to(torch.float32), cov_inverse.to(torch.float32), unique_labels


def calibrate_threshold_mahalanobis(
    train_embeddings: torch.Tensor,
    train_labels: list[int],
    *,
    percentile: float = DEFAULT_THRESHOLD_PERCENTILE,
    shrinkage: float = DEFAULT_SHRINKAGE,
) -> tuple[float, list[float], torch.Tensor, torch.Tensor, list[int]]:
    """Fit class statistics and compute the threshold from training distances.

    Returns ``(threshold, raw_distances, class_means, cov_inverse, class_order)``.
    Callers persist the means + inverse + threshold via :class:`MahalanobisManifest`.
    """

    class_means, cov_inverse, class_order = fit_class_statistics(
        train_embeddings, train_labels, shrinkage=shrinkage
    )
    distances: list[float] = []
    for i in range(train_embeddings.shape[0]):
        d, _ = mahalanobis_distance(train_embeddings[i], class_means, cov_inverse)
        if math.isfinite(d):
            distances.append(d)
    if not distances:
        raise ValueError("calibration corpus produced zero valid distances")
    distances_sorted = sorted(distances)
    rank = max(0, min(len(distances_sorted) - 1, int(len(distances_sorted) * percentile / 100.0)))
    threshold = distances_sorted[rank]
    return threshold, distances, class_means, cov_inverse, class_order


def load_mahalanobis_manifest(path: Path | str) -> MahalanobisManifest | None:
    """Read a manifest from disk. Returns ``None`` if missing or malformed."""

    target = Path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    try:
        return MahalanobisManifest(
            model_id=str(payload["model_id"]),
            embedding_dim=int(payload["embedding_dim"]),
            class_labels=[str(s) for s in payload["class_labels"]],
            class_means=[[float(v) for v in row] for row in payload["class_means"]],
            cov_inverse=[[float(v) for v in row] for row in payload["cov_inverse"]],
            threshold=float(payload["threshold"]),
            percentile=float(payload["percentile"]),
            shrinkage=float(payload.get("shrinkage", DEFAULT_SHRINKAGE)),
            training_corpus_size=int(payload["training_corpus_size"]),
            training_distance_mean=float(payload["training_distance_mean"]),
            training_distance_std=float(payload["training_distance_std"]),
            training_distance_min=float(payload["training_distance_min"]),
            training_distance_max=float(payload["training_distance_max"]),
            calibrated_at_utc=str(payload["calibrated_at_utc"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def score_text_mahalanobis(
    text: str,
    *,
    classifier: Any,
    manifest: MahalanobisManifest,
    chunks: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compute the OOD signal for ``text`` against a Mahalanobis manifest.

    Mirrors the response shape of :func:`app.evaluation.ood.score_text` so
    the serving path stays drop-in: returns ``{ood_energy, ood_threshold,
    is_in_distribution}`` where ``ood_energy`` carries the Mahalanobis
    distance (lower = closer to training distribution, higher = farther).
    """

    if not text or not text.strip():
        return {
            "ood_energy": None,
            "ood_threshold": None,
            "is_in_distribution": None,
        }

    model = getattr(classifier, "model", None) or classifier
    tokenizer = getattr(classifier, "tokenizer", None)
    if model is None or tokenizer is None:
        return {
            "ood_energy": None,
            "ood_threshold": None,
            "is_in_distribution": None,
        }

    class_means = torch.tensor(manifest.class_means, dtype=torch.float32)
    cov_inverse = torch.tensor(manifest.cov_inverse, dtype=torch.float32)

    text_chunks = list(chunks) if chunks else [text]
    if not text_chunks:
        text_chunks = [text]

    per_chunk_distances: list[float] = []
    for chunk in text_chunks:
        embedding = extract_cls_embedding(model, tokenizer, chunk)
        if embedding is None:
            continue
        # Move to the same device/dtype as the class means.
        embedding = embedding.to(class_means.device, dtype=class_means.dtype)
        d, _ = mahalanobis_distance(embedding, class_means, cov_inverse)
        if math.isfinite(d):
            per_chunk_distances.append(d)

    if not per_chunk_distances:
        return {
            "ood_energy": None,
            "ood_threshold": float(manifest.threshold),
            "is_in_distribution": None,
        }

    # Aggregate per-chunk distances by mean. A single OOD chunk in a long
    # document shouldn't trip the gate; for that switch to max here, but
    # mean is the more stable default.
    doc_distance = sum(per_chunk_distances) / len(per_chunk_distances)
    return {
        "ood_energy": doc_distance,
        "ood_threshold": float(manifest.threshold),
        "is_in_distribution": doc_distance <= manifest.threshold,
    }


__all__ = [
    "DEFAULT_SHRINKAGE",
    "DEFAULT_THRESHOLD_PERCENTILE",
    "MahalanobisManifest",
    "OOD_MAHALANOBIS_MANIFEST_NAME",
    "calibrate_threshold_mahalanobis",
    "extract_cls_embedding",
    "fit_class_statistics",
    "load_mahalanobis_manifest",
    "mahalanobis_distance",
    "score_text_mahalanobis",
]
