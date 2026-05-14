"""Energy-based out-of-distribution detection for the sentiment classifier.

The live `/analyze` flow runs a fine-tuned FinBERT-FOMC seed-71 head against
arbitrary user input. Adversarial inputs that don't look like FOMC text
("good afternoon crypto bros") routinely come back as `hawkish 0.94`
because the model has nothing in-distribution to anchor on and falls back
to the training prior at high confidence.

This module ships a calibrated energy-based OOD signal that surfaces the
problem visibly. Approach: Liu et al. 2020 ("Energy-based Out-of-distribution
Detection"). For a classifier with K classes returning logits `f(x) in R^K`,
the **free energy** is

    E(x) = -T * logsumexp(f(x) / T)        (1)

with temperature `T = 1.0` by default. Lower energy → input lies near the
training distribution; higher energy → input is far from training and the
classifier's confidence is uncalibrated. A threshold is calibrated on the
training corpus (5th percentile of in-domain energies by default) and
persisted as a JSON manifest next to the checkpoint.

API surface:

- `logit_energy(model, tokenizer, text, temperature)`: per-text energy.
- `aggregate_energy(scores, mode)`: reduces per-chunk energies to one number.
- `OODManifest` + `load_manifest` + `OOD_MANIFEST_NAME`: persistence.
- `score_text(text, classifier, manifest)`: convenience for the API path.
- `calibrate_threshold(corpus, classifier, percentile)`: build a manifest.

No backwards-incompatible changes elsewhere; the API path returns
`ood_energy = None`, `is_in_distribution = None` when no manifest is
present, so the dashboard simply doesn't show an OOD signal until the
user runs `scripts/calibrate_ood.py`.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import torch

OOD_MANIFEST_NAME = "forecaster_best.ood.json"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_THRESHOLD_PERCENTILE = 95.0  # the 95th-percentile of training energies; anything above is OOD
DEFAULT_AGGREGATION: "EnergyAggregation" = "mean"

EnergyAggregation = Literal["mean", "max", "median"]


@dataclass(frozen=True)
class OODManifest:
    """Calibration manifest persisted alongside the sentiment checkpoint.

    `threshold` is the in-domain energy ceiling: a chunk-aggregated energy
    above this is flagged out-of-distribution. `percentile` is the percentile
    of training energies used to derive the threshold (e.g. 95.0 = top 5%
    of training energies are tolerated; anything above is OOD).
    """

    model_id: str
    threshold: float
    percentile: float
    temperature: float
    aggregation: EnergyAggregation
    training_corpus_size: int
    training_energy_mean: float
    training_energy_std: float
    training_energy_min: float
    training_energy_max: float
    calibrated_at_utc: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def logit_energy(
    model,
    tokenizer,
    text: str,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_length: int = 512,
) -> float:
    """Compute the free energy E(x) = -T * logsumexp(f(x) / T) for a single text.

    The model is queried once. For long texts the caller is responsible for
    chunking and aggregating via `aggregate_energy`. Returns a Python float
    so callers can sort, percentile, JSON-serialise without numpy.
    """

    if not text:
        return float("inf")
    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        # Defensive: a stub model without any parameters defaults to CPU.
        # Real HF models always have parameters; this branch covers tests
        # that inject a parameter-free toy model.
        device = torch.device("cpu")
    raw_inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    # HF's BatchEncoding has .to(); a plain dict (from a stub tokenizer in
    # tests) does not. Move the tensors to device by hand in that case.
    if hasattr(raw_inputs, "to"):
        inputs = raw_inputs.to(device)
    elif isinstance(raw_inputs, dict):
        inputs = {key: value.to(device) for key, value in raw_inputs.items()}
    else:
        inputs = raw_inputs
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(0)
    energy = -float(temperature) * torch.logsumexp(logits / float(temperature), dim=-1)
    return float(energy.item())


def aggregate_energy(
    chunk_energies: Iterable[float],
    *,
    mode: EnergyAggregation = DEFAULT_AGGREGATION,
) -> float:
    """Reduce per-chunk energies to a doc-level number.

    `mean` is the default — every chunk contributes equally, the doc-level
    energy averages out short noise. `max` is the most-OOD chunk; useful
    when one alien sentence in a long document should trigger the gate.
    `median` is robust to a single outlier chunk.
    """

    values = [float(e) for e in chunk_energies if math.isfinite(e)]
    if not values:
        return float("inf")
    if mode == "mean":
        return sum(values) / len(values)
    if mode == "max":
        return max(values)
    if mode == "median":
        return statistics.median(values)
    raise ValueError(f"unknown aggregation mode: {mode!r}")


def calibrate_threshold(
    training_texts: Iterable[str],
    *,
    classifier,
    percentile: float = DEFAULT_THRESHOLD_PERCENTILE,
    temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[float, list[float]]:
    """Compute the threshold for the OOD gate from a training-corpus iterable.

    Returns `(threshold, raw_energies)`. Caller persists via OODManifest.
    """

    model = getattr(classifier, "model", None) or classifier
    tokenizer = getattr(classifier, "tokenizer", None)
    if model is None or tokenizer is None:
        raise ValueError("classifier must expose .model and .tokenizer attributes")

    energies: list[float] = []
    for text in training_texts:
        try:
            energies.append(
                logit_energy(model, tokenizer, text, temperature=temperature)
            )
        except Exception:
            continue

    if not energies:
        raise ValueError("calibration corpus produced zero valid energies")
    energies_sorted = sorted(energies)
    # Higher percentile -> stricter gate (allows more training energies in)
    rank = max(0, min(len(energies_sorted) - 1, int(len(energies_sorted) * percentile / 100.0)))
    threshold = energies_sorted[rank]
    return threshold, energies


def load_manifest(path: Path | str) -> OODManifest | None:
    """Read a calibration manifest from disk. Returns None when the file is
    missing or malformed — callers fall back to "no OOD signal" gracefully.
    """

    target = Path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    try:
        return OODManifest(
            model_id=str(payload["model_id"]),
            threshold=float(payload["threshold"]),
            percentile=float(payload["percentile"]),
            temperature=float(payload["temperature"]),
            aggregation=str(payload.get("aggregation", DEFAULT_AGGREGATION)),  # type: ignore[arg-type]
            training_corpus_size=int(payload["training_corpus_size"]),
            training_energy_mean=float(payload["training_energy_mean"]),
            training_energy_std=float(payload["training_energy_std"]),
            training_energy_min=float(payload["training_energy_min"]),
            training_energy_max=float(payload["training_energy_max"]),
            calibrated_at_utc=str(payload["calibrated_at_utc"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def score_text(
    text: str,
    *,
    classifier,
    manifest: OODManifest,
    chunks: list[str] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper used by the API path.

    Computes the doc-level energy under the manifest's temperature +
    aggregation, then compares against the threshold to derive
    `is_in_distribution`. When ``chunks`` is provided the model is queried
    once per chunk; otherwise the whole text is fed in a single pass.
    """

    model = getattr(classifier, "model", None) or classifier
    tokenizer = getattr(classifier, "tokenizer", None)
    if model is None or tokenizer is None:
        return {
            "ood_energy": None,
            "ood_threshold": manifest.threshold,
            "is_in_distribution": None,
        }

    if chunks is None or not chunks:
        chunks = [text]
    chunk_energies = [
        logit_energy(model, tokenizer, chunk, temperature=manifest.temperature)
        for chunk in chunks
        if chunk
    ]
    if not chunk_energies:
        return {
            "ood_energy": None,
            "ood_threshold": manifest.threshold,
            "is_in_distribution": None,
        }
    doc_energy = aggregate_energy(chunk_energies, mode=manifest.aggregation)
    return {
        "ood_energy": doc_energy,
        "ood_threshold": manifest.threshold,
        "is_in_distribution": doc_energy <= manifest.threshold,
        "chunk_energies": chunk_energies,
    }
