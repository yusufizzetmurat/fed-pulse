"""Cross-source transfer evaluation.

Given a stance-classification checkpoint trained on FOMC text and a training
package whose `registry_normalized.jsonl` carries labelled rows from multiple
``source_type`` strata (FOMC statements / minutes / meeting transcripts,
chair / governor speeches, congressional testimony, press conferences, Beige
Book, regional research, Op-Fed external corpus, etc.), compute per-source
macro-F1, accuracy, and per-class precision / recall / F1.

Inference-only. The trained checkpoint is held fixed; rows are filtered by
``source_type`` and scored against the model's existing weights. No
re-training, no leak — the eval simply asks "does the FOMC-trained model
generalise to source X?" for each source the registry carries labelled rows
for.

Output schema mirrors `cross_bank_transfer` so downstream aggregation can
share code paths: ``matrix.csv`` with one row per ``(encoder, source)`` pair
plus per-source ``support`` so under-populated cells are visible.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.data.finetune_pilot import (
    ID2LABEL,
    LABELS,
    _compute_classification_metrics,
    _latency_summary,
)


# Canonical FOMC-side source_type strata that ship through the registry
# today. Extend by adding the source_type string here; the harness filters
# rows by exact match on the registry's ``source_type`` column.
CROSS_SOURCE_TYPES: tuple[str, ...] = (
    "fomc_statement",
    "fomc_minutes",
    "fomc_meeting_transcript",
    "fomc_press_conference",
    "chair_speech",
    "governor_speech",
    "congressional_testimony",
    "beige_book",
    "regional_research",
    "ny_fed_liberty_street",
)


@dataclass(frozen=True)
class CrossSourceRow:
    """A labelled registry row read for the cross-source eval."""

    record_id: str
    text: str
    label: str
    event_date: str
    source: str
    source_type: str
    provenance: str


@dataclass(frozen=True)
class CrossSourceResult:
    source_type: str
    encoder_alias: str
    checkpoint: str
    support: int
    macro_f1: float
    weighted_f1: float
    accuracy: float
    per_class: dict[str, dict[str, float]]
    label_support: dict[str, int]
    latency_ms_p50: float
    latency_ms_p95: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "encoder_alias": self.encoder_alias,
            "checkpoint": self.checkpoint,
            "support": self.support,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "accuracy": self.accuracy,
            "per_class": self.per_class,
            "label_support": self.label_support,
            "latency_ms": {"p50": self.latency_ms_p50, "p95": self.latency_ms_p95},
        }


def load_cross_source_rows(
    package_dir: Path,
    *,
    include_zero_weight: bool = False,
) -> list[CrossSourceRow]:
    """Read every labelled row from the package registry that carries a
    canonical ``source_type``.

    ``include_zero_weight=False`` drops rows with ``sample_weight==0`` so
    cross-bank corpora (peer-reviewed-cross-bank provenance) and unlabelled
    archive rows (scraped, label="") do not enter the eval set. The base
    eval is FOMC-side cross-source only — cross-bank rides on its own
    harness (``app.evaluation.cross_bank_transfer``).
    """

    path = package_dir / "registry_normalized.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing registry: {path}")

    rows: list[CrossSourceRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        label = str(payload.get("mapped_label", "")).strip().lower()
        if label not in LABELS:
            continue
        text = str(payload.get("text", "")).strip()
        if not text:
            continue
        try:
            sample_weight = float(payload.get("sample_weight", 1.0))
        except (TypeError, ValueError):
            sample_weight = 1.0
        if not include_zero_weight and sample_weight == 0.0:
            continue
        source_type = str(payload.get("source_type", "")).strip()
        if source_type not in CROSS_SOURCE_TYPES:
            continue
        rows.append(
            CrossSourceRow(
                record_id=str(payload.get("record_id", "")).strip(),
                text=text,
                label=label,
                event_date=str(payload.get("event_date", "")).strip(),
                source=str(payload.get("source", "")).strip(),
                source_type=source_type,
                provenance=str(payload.get("provenance", "")).strip(),
            )
        )
    return rows


def group_by_source_type(
    rows: Iterable[CrossSourceRow],
) -> dict[str, list[CrossSourceRow]]:
    """Bucket rows by ``source_type``."""

    buckets: dict[str, list[CrossSourceRow]] = {}
    for row in rows:
        buckets.setdefault(row.source_type, []).append(row)
    return buckets


def _predict_with_model(
    rows: list[CrossSourceRow],
    *,
    checkpoint: str,
    max_length: int = 256,
    batch_size: int = 32,
) -> tuple[list[str], list[str], list[float]]:
    """Run inference using a HuggingFace classification checkpoint."""

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Honour the canonical TDW label order (dovish, hawkish, neutral) per the
    # cross_bank_transfer note — some legacy checkpoints carry an inverted
    # id2label that the patch script normalised in place.
    id2label = ID2LABEL

    y_true: list[str] = []
    y_pred: list[str] = []
    latencies: list[float] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch_texts = [r.text for r in batch_rows]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            t0 = time.perf_counter()
            logits = model(**enc).logits
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            per_item_ms = elapsed_ms / max(len(batch_rows), 1)
            latencies.extend([per_item_ms] * len(batch_rows))
            preds = logits.argmax(dim=-1).tolist()
            for row, pred_idx in zip(batch_rows, preds):
                y_true.append(row.label)
                y_pred.append(id2label[int(pred_idx)])
    return y_true, y_pred, latencies


def evaluate_source(
    rows: list[CrossSourceRow],
    *,
    source_type: str,
    encoder_alias: str,
    checkpoint: str,
    max_length: int = 256,
    batch_size: int = 32,
    predict_fn: Any = None,
) -> CrossSourceResult:
    """Score a single ``source_type`` slice end-to-end.

    ``predict_fn`` is injected by tests with a deterministic stub; production
    code passes ``None`` to fall through to the HF inference path.
    """

    if not rows:
        raise ValueError(
            f"No labelled rows for source_type={source_type!r} in registry."
        )
    record_ids = [r.record_id for r in rows if r.record_id]
    if record_ids and len(record_ids) != len(set(record_ids)):
        raise ValueError(
            f"Duplicate record_ids for source_type={source_type!r} — "
            "re-run ingest_sources to regenerate the registry."
        )

    if predict_fn is None:
        y_true, y_pred, latencies = _predict_with_model(
            rows, checkpoint=checkpoint, max_length=max_length, batch_size=batch_size
        )
    else:
        y_true, y_pred, latencies = predict_fn(rows)

    cls = _compute_classification_metrics(y_true, y_pred)
    latency = _latency_summary(latencies)
    label_support = dict(Counter(r.label for r in rows))
    return CrossSourceResult(
        source_type=source_type,
        encoder_alias=encoder_alias,
        checkpoint=checkpoint,
        support=len(rows),
        macro_f1=cls["macro_f1"],
        weighted_f1=cls["weighted_f1"],
        accuracy=cls["accuracy"],
        per_class=cls["per_class"],
        label_support=label_support,
        latency_ms_p50=latency["p50_ms"],
        latency_ms_p95=latency["p95_ms"],
    )


def build_matrix(
    *,
    package_dir: Path,
    encoder_checkpoints: dict[str, str],
    source_types: list[str] | None = None,
    max_length: int = 256,
    batch_size: int = 32,
    predict_fn: Any = None,
) -> dict[str, Any]:
    """Build the per-(encoder, source_type) cross-source transfer payload.

    ``encoder_checkpoints`` is ``{alias: checkpoint_path}``. The harness runs
    inference once per (alias, source_type) cell. Cells with zero rows are
    emitted as ``support=0`` with empty metrics so the under-populated
    sources stay visible in the CSV.
    """

    rows = load_cross_source_rows(package_dir)
    buckets = group_by_source_type(rows)
    targets = source_types or list(CROSS_SOURCE_TYPES)

    matrix: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "training_package_id": package_dir.name,
        "encoders": list(encoder_checkpoints.keys()),
        "source_types": list(targets),
        "per_source_counts": {st: len(buckets.get(st, [])) for st in targets},
        "cells": [],
        "failures": [],
    }

    for encoder_alias, checkpoint in encoder_checkpoints.items():
        for source_type in targets:
            slice_rows = buckets.get(source_type, [])
            if not slice_rows:
                matrix["cells"].append(
                    {
                        "encoder_alias": encoder_alias,
                        "checkpoint": checkpoint,
                        "source_type": source_type,
                        "support": 0,
                        "status": "no_rows",
                    }
                )
                continue
            try:
                result = evaluate_source(
                    slice_rows,
                    source_type=source_type,
                    encoder_alias=encoder_alias,
                    checkpoint=checkpoint,
                    max_length=max_length,
                    batch_size=batch_size,
                    predict_fn=predict_fn,
                )
            except Exception as exc:  # noqa: BLE001 — surface per-cell failure
                matrix["failures"].append(
                    {
                        "encoder_alias": encoder_alias,
                        "checkpoint": checkpoint,
                        "source_type": source_type,
                        "error": str(exc),
                    }
                )
                continue
            cell = result.to_dict()
            cell["status"] = "ok"
            matrix["cells"].append(cell)

    return matrix


def render_csv(matrix: dict[str, Any]) -> str:
    fieldnames = [
        "encoder_alias",
        "checkpoint",
        "source_type",
        "status",
        "support",
        "dovish_n",
        "hawkish_n",
        "neutral_n",
        "macro_f1",
        "weighted_f1",
        "accuracy",
        "latency_ms_p50",
        "latency_ms_p95",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for cell in matrix.get("cells", []):
        per_label = cell.get("label_support") or {}
        latency = cell.get("latency_ms") or {}
        writer.writerow(
            {
                "encoder_alias": cell.get("encoder_alias", ""),
                "checkpoint": cell.get("checkpoint", ""),
                "source_type": cell.get("source_type", ""),
                "status": cell.get("status", ""),
                "support": cell.get("support", 0),
                "dovish_n": per_label.get("dovish", 0),
                "hawkish_n": per_label.get("hawkish", 0),
                "neutral_n": per_label.get("neutral", 0),
                "macro_f1": _fmt(cell.get("macro_f1")),
                "weighted_f1": _fmt(cell.get("weighted_f1")),
                "accuracy": _fmt(cell.get("accuracy")),
                "latency_ms_p50": _fmt(latency.get("p50")),
                "latency_ms_p95": _fmt(latency.get("p95")),
            }
        )
    return buffer.getvalue()


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""


def _parse_encoder_spec(spec: str) -> dict[str, str]:
    """Parse ``alias=checkpoint[,alias=checkpoint]`` into a dict.

    Unlike the cross-bank transfer-matrix CLI we deliberately accept exactly
    one checkpoint per alias here — the cross-source eval reports a point
    estimate per (encoder, source) cell. Multiple checkpoints are a
    follow-up; if you need per-seed CIs, run this harness once per seed and
    aggregate downstream.
    """

    out: dict[str, str] = {}
    if not spec:
        return out
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"--encoder-checkpoints entry {piece!r} missing 'alias=path'")
        alias, path = piece.split("=", 1)
        alias = alias.strip()
        path = path.strip()
        if not alias or not path:
            raise ValueError(f"--encoder-checkpoints entry {piece!r} has empty alias or path")
        if alias in out:
            raise ValueError(
                f"--encoder-checkpoints alias {alias!r} duplicated; pass exactly one path per alias."
            )
        out[alias] = path
    return out


def _parse_source_types(spec: str) -> list[str] | None:
    if not spec:
        return None
    out: list[str] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token not in CROSS_SOURCE_TYPES:
            raise ValueError(
                f"unknown source_type {token!r}; allowed: {CROSS_SOURCE_TYPES}"
            )
        out.append(token)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the cross-source transfer matrix.")
    parser.add_argument("--training-package-id", required=True)
    parser.add_argument(
        "--encoder-checkpoints",
        required=True,
        help=(
            "Comma-separated alias=path pairs (exactly one path per alias). "
            "Example: finbert_fed_adjacent=/path/to/ckpt"
        ),
    )
    parser.add_argument(
        "--source-types",
        default="",
        help=(
            "Comma-separated source_type strata to score. Defaults to the "
            "full canonical set; use this to restrict the matrix to a subset."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    from app.config import DATA_DIR

    args = _parse_args()
    package_dir = DATA_DIR / "processed" / args.training_package_id
    if not package_dir.exists():
        raise SystemExit(f"Training package not found: {package_dir}")
    encoder_checkpoints = _parse_encoder_spec(args.encoder_checkpoints)
    if not encoder_checkpoints:
        raise SystemExit("No encoder checkpoints provided.")
    source_types = _parse_source_types(args.source_types)

    matrix = build_matrix(
        package_dir=package_dir,
        encoder_checkpoints=encoder_checkpoints,
        source_types=source_types,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = DATA_DIR / "artifacts" / "v2_cross_source" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "matrix.json").write_text(
        json.dumps(matrix, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "matrix.csv").write_text(render_csv(matrix), encoding="utf-8")
    print(f"[cross_source_transfer] wrote artefacts to {output_dir}")
    return 0


__all__ = [
    "CROSS_SOURCE_TYPES",
    "CrossSourceResult",
    "CrossSourceRow",
    "build_matrix",
    "evaluate_source",
    "group_by_source_type",
    "load_cross_source_rows",
    "render_csv",
]


if __name__ == "__main__":
    raise SystemExit(main())
