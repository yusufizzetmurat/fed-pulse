"""Calibrate the energy-based OOD threshold for the live sentiment classifier.

Reads texts from a registry JSONL or processed parquet, scores each
through the loaded classifier, and writes a manifest at the path passed
via `--output` (default: backend/models/forecaster_best.ood.json next to
the checkpoint).

Usage inside the backend container:

    docker compose --profile gpu run --rm backend-gpu \
        python -m scripts.calibrate_ood \
        --input /data/raw/phase2/source_registry.jsonl \
        --output /data/artifacts/phase3/pilot_finetune_20260505T142652Z/hf_checkpoints/forecaster_best.ood.json \
        --percentile 95.0 \
        --temperature 1.0 \
        --aggregation mean

Once the manifest is in place the live /analyze flow auto-loads it and
the response carries ood_energy / ood_threshold / is_in_distribution.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from app.evaluation.ood import (
    DEFAULT_AGGREGATION,
    DEFAULT_TEMPERATURE,
    DEFAULT_THRESHOLD_PERCENTILE,
    OOD_MANIFEST_NAME,
    OODManifest,
    calibrate_threshold,
)
from app.services.text_encoder import MODEL_ID, get_classifier


def _iter_jsonl_texts(path: Path, *, text_key: str = "text") -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = row.get(text_key) if isinstance(row, dict) else None
            if isinstance(value, str) and value.strip():
                yield value.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="JSONL file with one row per training document; expects a `text` field.",
    )
    parser.add_argument(
        "--output",
        default=str(Path("/data/artifacts") / OOD_MANIFEST_NAME),
        help=f"Output manifest path (default places {OOD_MANIFEST_NAME} under /data/artifacts).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_THRESHOLD_PERCENTILE,
        help="Percentile of in-domain energies to use as the OOD threshold.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature for the energy = -T * logsumexp(logits / T).",
    )
    parser.add_argument(
        "--aggregation",
        choices=["mean", "max", "median"],
        default=DEFAULT_AGGREGATION,
        help="How to reduce per-chunk energies to a doc-level number.",
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

    texts = list(_iter_jsonl_texts(input_path))
    if args.max_rows > 0:
        texts = texts[: args.max_rows]
    if not texts:
        print(f"no texts read from {input_path}", file=sys.stderr)
        return 1

    print(f"calibrating from {len(texts)} texts in {input_path}")
    classifier = get_classifier()
    threshold, energies = calibrate_threshold(
        texts,
        classifier=classifier,
        percentile=args.percentile,
        temperature=args.temperature,
    )

    manifest = OODManifest(
        model_id=MODEL_ID,
        threshold=threshold,
        percentile=args.percentile,
        temperature=args.temperature,
        aggregation=args.aggregation,
        training_corpus_size=len(energies),
        training_energy_mean=statistics.fmean(energies),
        training_energy_std=statistics.pstdev(energies) if len(energies) > 1 else 0.0,
        training_energy_min=min(energies),
        training_energy_max=max(energies),
        calibrated_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    print(f"wrote OOD manifest: {output_path}")
    print(f"  threshold       : {manifest.threshold:.4f}")
    print(f"  percentile      : {manifest.percentile}")
    print(f"  training mean   : {manifest.training_energy_mean:.4f}")
    print(f"  training std    : {manifest.training_energy_std:.4f}")
    print(f"  training min/max: {manifest.training_energy_min:.4f} / {manifest.training_energy_max:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
