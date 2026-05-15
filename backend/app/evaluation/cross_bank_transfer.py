"""Zero-shot cross-CB transfer evaluation.

Given an NLP checkpoint trained on FOMC stance labels and a held-out central
bank from the gtfintechlab cross-bank pool (ECB / BoJ / BoE / BoC / RBA),
compute macro-F1, accuracy, per-class precision/recall, and per-axis metrics
(stance + time + certainty when available) on that bank's labeled rows.

The eval reads cross-bank rows directly from the training package's
``registry_normalized.jsonl`` — they live there with ``sample_weight=0`` so
they are masked from training but still indexable for evaluation.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.data.finetune_pilot import (
    ID2LABEL,
    LABEL2ID,
    LABELS,
    _compute_classification_metrics,
    _latency_summary,
    _load_registry_rows,
    EvalRow,
)

CROSS_BANK_SOURCES = (
    "gtfintechlab_european_central_bank",
    "gtfintechlab_bank_of_japan",
    "gtfintechlab_bank_of_england",
    "gtfintechlab_bank_of_canada",
    "gtfintechlab_reserve_bank_of_australia",
)


@dataclass(frozen=True)
class CrossBankResult:
    bank: str
    checkpoint: str
    support: int
    macro_f1: float
    weighted_f1: float
    accuracy: float
    per_class: dict[str, dict[str, float]]
    latency_ms_p50: float
    latency_ms_p95: float
    per_axis: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bank": self.bank,
            "checkpoint": self.checkpoint,
            "support": self.support,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "accuracy": self.accuracy,
            "per_class": self.per_class,
            "latency_ms": {"p50": self.latency_ms_p50, "p95": self.latency_ms_p95},
            "per_axis": self.per_axis,
        }


def load_cross_bank_rows(package_dir: Path, bank_source: str) -> list[EvalRow]:
    """Read cross-bank labeled rows for ``bank_source`` from the package registry."""

    all_rows = _load_registry_rows(package_dir, include_zero_weight=True)
    return [row for row in all_rows if row.source == bank_source]


def _read_raw_registry_extras(package_dir: Path, bank_source: str) -> dict[str, dict[str, Any]]:
    """Return {record_id: multi_axis_extras} for the bank's rows.

    Used by ``_compute_per_axis_metrics`` to surface stance + time + certainty
    breakdowns where the gtfintechlab schema carries them.
    """

    out: dict[str, dict[str, Any]] = {}
    path = package_dir / "registry_normalized.jsonl"
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(payload.get("source", "")) != bank_source:
            continue
        rid = str(payload.get("record_id", ""))
        if rid:
            extras = payload.get("multi_axis_extras") or {}
            if isinstance(extras, dict):
                out[rid] = extras
    return out


def _slice_by_axis_value(
    rows: Iterable[EvalRow],
    extras_by_id: dict[str, dict[str, Any]],
    *,
    axis_field: str,
) -> dict[str, list[EvalRow]]:
    buckets: dict[str, list[EvalRow]] = {}
    for row in rows:
        extras = extras_by_id.get(row.record_id, {})
        value = str(extras.get(axis_field, "") or "").strip()
        if not value:
            continue
        buckets.setdefault(value, []).append(row)
    return buckets


def _compute_per_axis_metrics(
    rows: list[EvalRow],
    y_true: list[str],
    y_pred: list[str],
    extras_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute macro-F1 sliced by each multi-axis extras field, when present."""

    axis_fields = (
        ("time", "gtfintechlab_time_label"),
        ("certainty", "gtfintechlab_certain_label"),
    )
    per_axis: dict[str, dict[str, Any]] = {}
    for axis_name, axis_field in axis_fields:
        buckets = _slice_by_axis_value(rows, extras_by_id, axis_field=axis_field)
        if not buckets:
            continue
        slice_metrics: dict[str, dict[str, Any]] = {}
        for bucket_value, bucket_rows in buckets.items():
            ids = {row.record_id for row in bucket_rows}
            sub_true = [t for row, t in zip(rows, y_true) if row.record_id in ids]
            sub_pred = [p for row, p in zip(rows, y_pred) if row.record_id in ids]
            if not sub_true:
                continue
            cls = _compute_classification_metrics(sub_true, sub_pred)
            slice_metrics[bucket_value] = {
                "support": len(sub_true),
                "macro_f1": cls["macro_f1"],
                "accuracy": cls["accuracy"],
            }
        if slice_metrics:
            per_axis[axis_name] = slice_metrics
    return per_axis


def _label_support(rows: list[EvalRow]) -> Counter[str]:
    return Counter(row.label for row in rows)


def _predict_with_model(
    rows: list[EvalRow],
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
    model.eval()
    device = next(model.parameters()).device

    # Honour LABEL2ID order, not the loaded checkpoint's id2label, because some
    # legacy checkpoints had inverted classes. The canonical TDW order is
    # (dovish, hawkish, neutral).
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
            elapsed_ms = (time.perf_counter() - t0) * 1000
            per_item_ms = elapsed_ms / max(len(batch_rows), 1)
            latencies.extend([per_item_ms] * len(batch_rows))
            preds = logits.argmax(dim=-1).tolist()
            for row, pred_idx in zip(batch_rows, preds):
                y_true.append(row.label)
                y_pred.append(id2label[int(pred_idx)])

    return y_true, y_pred, latencies


def evaluate_cross_bank(
    *,
    package_dir: Path,
    bank_source: str,
    checkpoint: str,
    max_length: int = 256,
    batch_size: int = 32,
    predict_fn=None,
) -> CrossBankResult:
    """Evaluate ``checkpoint`` zero-shot on the bank's cross-bank rows.

    ``predict_fn`` is injected by tests with a deterministic stub; production
    code passes ``None`` to use the HF inference path.
    """

    if bank_source not in CROSS_BANK_SOURCES:
        raise ValueError(
            f"Unknown bank_source {bank_source!r}. Allowed: {CROSS_BANK_SOURCES}"
        )
    rows = load_cross_bank_rows(package_dir, bank_source)
    if not rows:
        raise ValueError(
            f"No labeled rows for {bank_source!r} in registry. Re-run ingest_sources to fetch."
        )

    if predict_fn is None:
        y_true, y_pred, latencies = _predict_with_model(
            rows, checkpoint=checkpoint, max_length=max_length, batch_size=batch_size
        )
    else:
        y_true, y_pred, latencies = predict_fn(rows)

    cls = _compute_classification_metrics(y_true, y_pred)
    latency = _latency_summary(latencies)
    extras = _read_raw_registry_extras(package_dir, bank_source)
    per_axis = _compute_per_axis_metrics(rows, y_true, y_pred, extras)

    return CrossBankResult(
        bank=bank_source,
        checkpoint=checkpoint,
        support=len(rows),
        macro_f1=cls["macro_f1"],
        weighted_f1=cls["weighted_f1"],
        accuracy=cls["accuracy"],
        per_class=cls["per_class"],
        latency_ms_p50=latency["p50_ms"],
        latency_ms_p95=latency["p95_ms"],
        per_axis=per_axis,
    )


__all__ = [
    "CROSS_BANK_SOURCES",
    "CrossBankResult",
    "LABELS",
    "evaluate_cross_bank",
    "load_cross_bank_rows",
]
