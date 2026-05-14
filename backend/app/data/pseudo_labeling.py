"""Pseudo-labelling pipeline for the unlabelled scraped FOMC corpus.

The teacher (Phase-4 fine-tune winner FinBERT-FOMC seed 71) scores each
unlabelled row in source_registry.jsonl; rows whose aggregated score
exceeds the doc-level threshold land in registry_pseudo.jsonl with
label_origin="pseudo" and full provenance metadata.

Two scoring strategies are supported:

* ``doc_truncated`` — legacy path. Feeds the raw text to the HF pipeline
  which auto-truncates at 512 tokens. This is what the 2026-05-05 audit
  ran against; precision was 0.30 because FOMC minutes are 60-100k chars
  and the teacher only saw the boilerplate intro.
* ``chunk_max_pool`` (default) / ``chunk_mean_pool`` / ``chunk_vote`` —
  splits the text into 480-token chunks (the same windowing the chunk
  embedding store uses), scores each chunk with the teacher, and
  aggregates chunk-level predictions into a doc-level label. Chunks
  whose max confidence falls below ``tau_chunk`` are dropped from the
  aggregation. Recommended when the input documents are longer than the
  teacher's max sequence length.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from app.config import DATA_DIR as DEFAULT_DATA_DIR
DEFAULT_INPUT = DEFAULT_DATA_DIR / "raw" / "phase2" / "source_registry.jsonl"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "interim" / "phase2" / "registry_pseudo.jsonl"
DEFAULT_AUDIT_DIR = DEFAULT_DATA_DIR / "artifacts" / "pseudo_label_audits"

ChunkStrategy = Literal["doc_truncated", "chunk_max_pool", "chunk_mean_pool", "chunk_vote"]
CHUNK_STRATEGIES: tuple[ChunkStrategy, ...] = (
    "doc_truncated",
    "chunk_max_pool",
    "chunk_mean_pool",
    "chunk_vote",
)
DEFAULT_STRATEGY: ChunkStrategy = "chunk_max_pool"
DEFAULT_TAU_CHUNK = 0.50
DEFAULT_MAX_CHUNKS_PER_DOC = 64


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
    parser.add_argument(
        "--strategy",
        choices=list(CHUNK_STRATEGIES),
        default=DEFAULT_STRATEGY,
        help=(
            "Scoring strategy: doc_truncated (legacy, 512-token cut), "
            "chunk_max_pool (default; highest-confidence chunk wins), "
            "chunk_mean_pool (average per-class probabilities), "
            "chunk_vote (modal label across chunks above the floor)."
        ),
    )
    parser.add_argument(
        "--tau-chunk",
        type=float,
        default=DEFAULT_TAU_CHUNK,
        help="Chunk-level confidence floor; chunks below this drop out of the aggregation.",
    )
    parser.add_argument(
        "--max-chunks-per-doc",
        type=int,
        default=DEFAULT_MAX_CHUNKS_PER_DOC,
        help="Hard cap on chunks per document; chunks beyond this index are ignored.",
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
        strategy=args.strategy,
        tau_chunk=args.tau_chunk,
        max_chunks_per_doc=args.max_chunks_per_doc,
    )
    print(f"Pseudo-labelled rows written: {written}")
    print(f"Strategy: {args.strategy}, tau_doc={args.threshold}, tau_chunk={args.tau_chunk}")
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
        top_k=None,
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


def split_text_for_teacher(text: str, *, max_chunks: int = DEFAULT_MAX_CHUNKS_PER_DOC) -> list[str]:
    """Split a document into the same 480-token windows the chunk-embedding store uses.

    The import is lazy so the module can be loaded in tests that stub the
    teacher pipeline directly and never touch the HF classifier.
    """

    text = (text or "").strip()
    if not text:
        return []
    from app.services.text_encoder import get_classifier, split_into_chunks  # type: ignore

    chunks = split_into_chunks(text, classifier=get_classifier())
    if max_chunks > 0:
        chunks = chunks[:max_chunks]
    return chunks


def aggregate_chunk_predictions(
    chunk_predictions: list[dict[str, Any]],
    *,
    strategy: ChunkStrategy,
    tau_chunk: float,
) -> dict[str, Any]:
    """Reduce a list of per-chunk predictions to a single doc-level prediction.

    Each chunk prediction must carry ``predicted_label``, ``max_score`` and
    ``scores`` (per-class probabilities). ``tau_chunk`` is the chunk-level
    confidence floor — chunks below this are excluded from the aggregation.
    When no chunk clears the floor the doc-level prediction falls back to
    the highest-confidence chunk's argmax with a ``max_score`` of 0.0 so
    the outer doc-level threshold still filters it out.
    """

    if not chunk_predictions:
        return {
            "predicted_label": "",
            "max_score": 0.0,
            "scores": {},
            "chunk_count": 0,
            "chunks_above_floor": 0,
            "strategy": strategy,
            "tau_chunk": tau_chunk,
        }

    kept = [c for c in chunk_predictions if c["max_score"] >= tau_chunk]
    fallback = max(chunk_predictions, key=lambda c: c["max_score"])
    if not kept:
        return {
            "predicted_label": fallback["predicted_label"],
            "max_score": 0.0,
            "scores": dict(fallback["scores"]),
            "chunk_count": len(chunk_predictions),
            "chunks_above_floor": 0,
            "strategy": strategy,
            "tau_chunk": tau_chunk,
            "fallback_max_score": float(fallback["max_score"]),
        }

    if strategy == "chunk_max_pool":
        winner = max(kept, key=lambda c: c["max_score"])
        return {
            "predicted_label": winner["predicted_label"],
            "max_score": float(winner["max_score"]),
            "scores": dict(winner["scores"]),
            "chunk_count": len(chunk_predictions),
            "chunks_above_floor": len(kept),
            "strategy": strategy,
            "tau_chunk": tau_chunk,
        }

    if strategy == "chunk_mean_pool":
        labels = sorted({label for c in kept for label in c["scores"]})
        averaged: dict[str, float] = {}
        for label in labels:
            averaged[label] = sum(float(c["scores"].get(label, 0.0)) for c in kept) / len(kept)
        winning_label = max(averaged, key=averaged.get) if averaged else fallback["predicted_label"]
        return {
            "predicted_label": winning_label,
            "max_score": float(averaged.get(winning_label, 0.0)),
            "scores": averaged,
            "chunk_count": len(chunk_predictions),
            "chunks_above_floor": len(kept),
            "strategy": strategy,
            "tau_chunk": tau_chunk,
        }

    if strategy == "chunk_vote":
        votes = Counter(c["predicted_label"] for c in kept)
        top = votes.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            # Tie-break on mean confidence of the tied labels
            tied = {label for label, count in top if count == top[0][1]}
            mean_by_label: dict[str, float] = {}
            for label in tied:
                votes_for_label = [c for c in kept if c["predicted_label"] == label]
                mean_by_label[label] = sum(c["max_score"] for c in votes_for_label) / len(votes_for_label)
            winning_label = max(mean_by_label, key=mean_by_label.get)
        else:
            winning_label = top[0][0]
        winning_chunk = max(
            (c for c in kept if c["predicted_label"] == winning_label),
            key=lambda c: c["max_score"],
        )
        return {
            "predicted_label": winning_label,
            "max_score": float(winning_chunk["max_score"]),
            "scores": dict(winning_chunk["scores"]),
            "chunk_count": len(chunk_predictions),
            "chunks_above_floor": len(kept),
            "strategy": strategy,
            "tau_chunk": tau_chunk,
            "vote_counts": dict(votes),
        }

    raise ValueError(f"unknown aggregation strategy: {strategy!r}")


def score_passages_chunked(
    passages: Iterable[str],
    *,
    pipeline,
    strategy: ChunkStrategy = DEFAULT_STRATEGY,
    tau_chunk: float = DEFAULT_TAU_CHUNK,
    max_chunks_per_doc: int = DEFAULT_MAX_CHUNKS_PER_DOC,
    splitter=None,
) -> list[dict[str, Any]]:
    """Chunk-aware scoring path. One doc-level prediction per passage.

    ``splitter`` is injected for tests so we do not have to spin up the
    HF classifier. In production it defaults to ``split_text_for_teacher``.
    """

    if splitter is None:
        def splitter(text: str) -> list[str]:
            return split_text_for_teacher(text, max_chunks=max_chunks_per_doc)

    predictions: list[dict[str, Any]] = []
    for passage in passages:
        chunks = splitter(passage)
        chunk_predictions = score_passages(chunks, pipeline=pipeline) if chunks else []
        predictions.append(
            aggregate_chunk_predictions(
                chunk_predictions, strategy=strategy, tau_chunk=tau_chunk
            )
        )
    return predictions


def build_pseudo_row(
    source_row: dict[str, Any],
    prediction: dict[str, Any],
    *,
    teacher_model_id: str,
    teacher_model_version: str,
) -> dict[str, Any]:
    """Assemble a registry-shaped pseudo-labelled row.

    Preserves every field of source_row, sets label / label_origin from
    the prediction, and tacks on teacher provenance. When the prediction
    carries chunk-level diagnostics (``chunk_count``, ``strategy``, …)
    those are persisted under ``teacher_aggregation`` for downstream
    auditing.
    """

    row = dict(source_row)
    row["label"] = prediction["predicted_label"]
    row["label_origin"] = "pseudo"
    row["teacher_model_id"] = teacher_model_id
    row["teacher_model_version"] = teacher_model_version
    row["teacher_max_score"] = float(prediction["max_score"])
    row["teacher_scores"] = dict(prediction["scores"])

    diagnostics: dict[str, Any] = {}
    for key in ("strategy", "tau_chunk", "chunk_count", "chunks_above_floor", "vote_counts", "fallback_max_score"):
        if key in prediction:
            diagnostics[key] = prediction[key]
    if diagnostics:
        row["teacher_aggregation"] = diagnostics
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
    strategy: ChunkStrategy = DEFAULT_STRATEGY,
    tau_chunk: float = DEFAULT_TAU_CHUNK,
    max_chunks_per_doc: int = DEFAULT_MAX_CHUNKS_PER_DOC,
    splitter=None,
) -> int:
    """Score unlabelled rows and write the kept pseudo set as JSONL.

    Returns the number of rows written. ``strategy`` selects between the
    legacy doc-truncated path and the new chunk-aware aggregator; see the
    module docstring for the trade-offs. ``splitter`` is exposed for tests
    so the chunk-aware path can run without the HF classifier.
    """

    if strategy not in CHUNK_STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy!r}; allowed: {CHUNK_STRATEGIES}")

    rows = _read_registry_jsonl(input_path)
    candidates = [row for row in rows if _is_unlabelled(row)]
    if max_rows > 0:
        candidates = candidates[:max_rows]

    if not candidates:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        return 0

    if strategy == "doc_truncated":
        predictions = score_passages(
            (row["text"] for row in candidates), pipeline=teacher_pipeline
        )
    else:
        predictions = score_passages_chunked(
            (row["text"] for row in candidates),
            pipeline=teacher_pipeline,
            strategy=strategy,
            tau_chunk=tau_chunk,
            max_chunks_per_doc=max_chunks_per_doc,
            splitter=splitter,
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
