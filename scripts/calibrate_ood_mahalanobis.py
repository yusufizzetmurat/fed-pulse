"""Calibrate the Mahalanobis-distance OOD detector for the sentiment
classifier.

Reads labelled texts from a registry JSONL, embeds each text with the
loaded classifier's encoder, fits per-class means + tied covariance,
computes training-set Mahalanobis distances, picks a percentile
threshold, and writes a manifest at ``--output`` (defaults to
``forecaster_best.ood_mahalanobis.json`` beside the sentiment
checkpoint).

Energy-based OOD (``scripts/calibrate_ood.py``) reads classifier logit
sharpness and fails to separate truly off-domain text when the
underlying head is confidently miscalibrated. The Mahalanobis detector
measures distance in the encoder's representation space and survives
that failure mode (Lee et al., NeurIPS 2018).

Usage inside the backend container::

    docker compose --profile gpu run --rm backend-gpu \\
        python -m scripts.calibrate_ood_mahalanobis \\
        --input /data/raw/phase2/source_registry.jsonl \\
        --output /data/artifacts/phase3/pilot_finetune_20260505T142652Z/hf_checkpoints/forecaster_best.ood_mahalanobis.json \\
        --max-rows 5000 \\
        --percentile 95.0

Once the manifest is on disk the live /analyze flow auto-loads it and
the response carries ood_energy (Mahalanobis distance) /
ood_threshold / is_in_distribution. The Mahalanobis manifest takes
precedence over the energy manifest when both are present.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch

from app.evaluation.ood_mahalanobis import (
    DEFAULT_SHRINKAGE,
    DEFAULT_THRESHOLD_PERCENTILE,
    OOD_MAHALANOBIS_MANIFEST_NAME,
    MahalanobisManifest,
    calibrate_threshold_mahalanobis,
    extract_cls_embedding,
)
from app.services.text_encoder import MODEL_ID, get_classifier

_DEFAULT_LABEL_KEY = "label"
_DEFAULT_TEXT_KEY = "text"


def _iter_labelled_rows(
    path: Path, *, text_key: str, label_key: str
) -> Iterable[tuple[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            text = row.get(text_key)
            label = row.get(label_key)
            if not isinstance(text, str) or not text.strip():
                continue
            if label is None:
                continue
            yield text.strip(), str(label)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="JSONL with one row per training document; expects `text` and `label`.",
    )
    parser.add_argument(
        "--output",
        default=str(Path("/data/artifacts") / OOD_MAHALANOBIS_MANIFEST_NAME),
        help=f"Output manifest path. Default places {OOD_MAHALANOBIS_MANIFEST_NAME} under /data/artifacts.",
    )
    parser.add_argument(
        "--text-key", default=_DEFAULT_TEXT_KEY,
    )
    parser.add_argument(
        "--label-key", default=_DEFAULT_LABEL_KEY,
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_THRESHOLD_PERCENTILE,
        help="Percentile of training distances used as the OOD threshold.",
    )
    parser.add_argument(
        "--shrinkage",
        type=float,
        default=DEFAULT_SHRINKAGE,
        help="Ridge shrinkage added to the covariance before inversion.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Cap the calibration corpus to N rows; 0 = no cap.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"input not found: {input_path}", file=sys.stderr)
        return 1

    rows = list(_iter_labelled_rows(input_path, text_key=args.text_key, label_key=args.label_key))
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    if not rows:
        print(f"no labelled rows read from {input_path}", file=sys.stderr)
        return 1

    label_universe = sorted({label for _, label in rows})
    label_to_idx = {label: idx for idx, label in enumerate(label_universe)}

    print(f"calibrating from {len(rows)} labelled texts in {input_path}")
    print(f"classes: {label_universe}")
    classifier = get_classifier()
    model = getattr(classifier, "model", None) or classifier
    tokenizer = getattr(classifier, "tokenizer", None)
    if model is None or tokenizer is None:
        print("classifier missing .model or .tokenizer attribute", file=sys.stderr)
        return 1

    embeddings_list: list[torch.Tensor] = []
    labels_list: list[int] = []
    for i, (text, label) in enumerate(rows):
        try:
            vec = extract_cls_embedding(model, tokenizer, text)
        except Exception:  # noqa: BLE001 — skip pathological rows, continue calibration
            continue
        if vec is None:
            continue
        embeddings_list.append(vec)
        labels_list.append(label_to_idx[label])
        if (i + 1) % 500 == 0:
            print(f"  embedded {i + 1}/{len(rows)} rows")

    if not embeddings_list:
        print("no embeddings produced", file=sys.stderr)
        return 1

    embeddings = torch.stack(embeddings_list, dim=0)
    threshold, distances, class_means, cov_inverse, class_order = (
        calibrate_threshold_mahalanobis(
            embeddings,
            labels_list,
            percentile=args.percentile,
            shrinkage=args.shrinkage,
        )
    )

    class_labels = [label_universe[idx] for idx in class_order]
    manifest = MahalanobisManifest(
        model_id=MODEL_ID,
        embedding_dim=int(embeddings.shape[1]),
        class_labels=class_labels,
        class_means=class_means.tolist(),
        cov_inverse=cov_inverse.tolist(),
        threshold=threshold,
        percentile=args.percentile,
        shrinkage=args.shrinkage,
        training_corpus_size=len(distances),
        training_distance_mean=statistics.fmean(distances),
        training_distance_std=statistics.pstdev(distances) if len(distances) > 1 else 0.0,
        training_distance_min=min(distances),
        training_distance_max=max(distances),
        calibrated_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    print(f"wrote OOD Mahalanobis manifest: {output_path}")
    print(f"  classes         : {class_labels}")
    print(f"  embedding dim   : {manifest.embedding_dim}")
    print(f"  threshold       : {manifest.threshold:.4f}")
    print(f"  percentile      : {manifest.percentile}")
    print(f"  training mean   : {manifest.training_distance_mean:.4f}")
    print(f"  training std    : {manifest.training_distance_std:.4f}")
    print(f"  training min/max: {manifest.training_distance_min:.4f} / {manifest.training_distance_max:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
