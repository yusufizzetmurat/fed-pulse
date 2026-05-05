"""Pseudo-labelling pipeline for the unlabelled scraped FOMC corpus.

The teacher (Phase-4 fine-tune winner FinBERT-FOMC seed 71) scores each
unlabelled row in source_registry.jsonl; rows whose max class score
exceeds the threshold land in registry_pseudo.jsonl with
label_origin="pseudo" and full provenance metadata. Plan 4 layers an
LLM-as-judge second annotator + audit on top.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path("/data") if Path("/data").exists() else BACKEND_ROOT.parent / "data"
DEFAULT_INPUT = DEFAULT_DATA_DIR / "raw" / "phase2" / "source_registry.jsonl"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "interim" / "phase2" / "registry_pseudo.jsonl"
DEFAULT_AUDIT_DIR = DEFAULT_DATA_DIR / "artifacts" / "pseudo_label_audits"


def threshold_sweep(
    predictions: list[dict[str, Any]], *, thresholds: tuple[float, ...] = (0.75, 0.85, 0.95)
) -> dict[str, Any]:
    """Yield + per-class distribution for each threshold over the same predictions.

    Returns a dict shaped for the project document precision/recall trade
    paragraph: {thresholds, total, yield, label_distribution}.
    """

    yield_by_tau: dict[str, int] = {}
    label_by_tau: dict[str, dict[str, int]] = {}
    for tau in thresholds:
        kept, _ = apply_threshold(predictions, threshold=tau)
        key = f"{tau}"
        yield_by_tau[key] = len(kept)
        label_by_tau[key] = dict(
            Counter(p["predicted_label"] for p in kept)
        )
    return {
        "thresholds": list(thresholds),
        "total": len(predictions),
        "yield": yield_by_tau,
        "label_distribution": label_by_tau,
    }


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
    pipeline = load_teacher(args.teacher_checkpoint)
    written = run_pseudo_labeling(
        input_path=Path(args.input),
        output_path=Path(args.output),
        teacher_pipeline=pipeline,
        threshold=args.threshold,
        teacher_model_id=args.teacher_model_id,
        teacher_model_version=args.teacher_model_version,
        max_rows=args.max_rows,
    )
    print(f"Pseudo-labelled rows written: {written}")
    print(f"Output: {args.output}")
    return 0


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


def apply_threshold(
    predictions: list[dict[str, Any]], *, threshold: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split predictions into kept (max_score >= threshold) and dropped."""

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for prediction in predictions:
        if prediction["max_score"] >= threshold:
            kept.append(prediction)
        else:
            dropped.append(prediction)
    return kept, dropped


def build_pseudo_row(
    source_row: dict[str, Any],
    prediction: dict[str, Any],
    *,
    teacher_model_id: str,
    teacher_model_version: str,
) -> dict[str, Any]:
    """Assemble a registry-shaped pseudo-labelled row.

    Preserves every field of source_row, sets label / label_origin from
    the prediction, and tacks on teacher provenance.
    """

    row = dict(source_row)
    row["label"] = prediction["predicted_label"]
    row["label_origin"] = "pseudo"
    row["teacher_model_id"] = teacher_model_id
    row["teacher_model_version"] = teacher_model_version
    row["teacher_max_score"] = float(prediction["max_score"])
    row["teacher_scores"] = dict(prediction["scores"])
    return row


def _read_registry_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _is_unlabelled(row: dict[str, Any]) -> bool:
    """A row is eligible for pseudo-labelling iff it has no human label."""

    return not str(row.get("label", "")).strip()


def run_pseudo_labeling(
    *,
    input_path: Path,
    output_path: Path,
    teacher_pipeline,
    threshold: float,
    teacher_model_id: str,
    teacher_model_version: str,
    max_rows: int = 0,
) -> int:
    """Score unlabelled rows and write the kept pseudo set as JSONL.

    Returns the number of rows written.
    """

    rows = _read_registry_jsonl(input_path)
    candidates = [row for row in rows if _is_unlabelled(row)]
    if max_rows > 0:
        candidates = candidates[:max_rows]

    if not candidates:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        return 0

    predictions = score_passages(
        (row["text"] for row in candidates), pipeline=teacher_pipeline
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row, prediction in zip(candidates, predictions):
            if prediction["max_score"] < threshold:
                continue
            pseudo_row = build_pseudo_row(
                row,
                prediction,
                teacher_model_id=teacher_model_id,
                teacher_model_version=teacher_model_version,
            )
            handle.write(json.dumps(pseudo_row) + "\n")
            written += 1
    return written


if __name__ == "__main__":
    raise SystemExit(main())
