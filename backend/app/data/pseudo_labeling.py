"""Pseudo-labelling pipeline for the unlabelled scraped FOMC corpus.

The teacher (Phase-4 fine-tune winner FinBERT-FOMC seed 71) scores each
unlabelled row in source_registry.jsonl; rows whose max class score
exceeds the threshold land in registry_pseudo.jsonl with
label_origin="pseudo" and full provenance metadata. Plan 4 layers an
LLM-as-judge second annotator + audit on top.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Sequence

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path("/data") if Path("/data").exists() else BACKEND_ROOT.parent / "data"
DEFAULT_INPUT = DEFAULT_DATA_DIR / "raw" / "phase2" / "source_registry.jsonl"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "interim" / "phase2" / "registry_pseudo.jsonl"
DEFAULT_AUDIT_DIR = DEFAULT_DATA_DIR / "artifacts" / "pseudo_label_audits"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score unlabelled scraped FOMC text with a fine-tuned teacher and write a pseudo-labelled registry."
    )
    parser.add_argument(
        "--teacher-checkpoint",
        required=True,
        help="Path to the fine-tuned teacher checkpoint directory (HF model dir).",
    )
    parser.add_argument(
        "--teacher-model-id",
        default="fomc_roberta_s71",
        help="Provenance label for the teacher (matches Phase-4 fine-tune batch encoder slot).",
    )
    parser.add_argument(
        "--teacher-model-version",
        default="phase4_finetune_v1",
        help="Provenance version string for the teacher.",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Source registry JSONL input.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Pseudo-labelled registry JSONL output.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Confidence threshold; rows with max class score below this are dropped.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Score at most this many rows; 0 means no limit (used for smoke).",
    )
    parser.add_argument(
        "--audit-dir",
        default=str(DEFAULT_AUDIT_DIR),
        help="Directory where threshold sweep and audit artefacts are written.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args()
    raise NotImplementedError("Wire orchestrator in Task 4.")


def load_teacher(checkpoint_path: str):
    """Load a fine-tuned text-classification pipeline from disk.

    Imports transformers lazily so the module can be imported in tests
    that stub the pipeline directly.
    """

    from transformers import (  # type: ignore
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TextClassificationPipeline,
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    return TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        truncation=True,
        max_length=512,
    )


def score_passages(passages: Iterable[str], *, pipeline) -> list[dict[str, Any]]:
    """Score a batch of passages and return one prediction dict per passage.

    Each prediction carries: predicted_label (argmax label string),
    max_score (float), scores (dict of label -> float).
    """

    raw = pipeline(list(passages), batch_size=8)
    predictions: list[dict[str, Any]] = []
    for entry in raw:
        scores = {item["label"]: float(item["score"]) for item in entry}
        top_label = max(scores, key=scores.get)
        predictions.append(
            {
                "predicted_label": top_label,
                "max_score": scores[top_label],
                "scores": scores,
            }
        )
    return predictions


if __name__ == "__main__":
    raise SystemExit(main())
