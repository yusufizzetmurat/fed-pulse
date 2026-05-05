"""Source-type stratified analysis over fine-tune-batch predictions.

Reads per-row test predictions emitted by
app.data.phase3_finetune_batch / phase3_finetune_pilot, joins them to
source_type from the source registry, and emits a per-source-type
metrics table (macro-F1, accuracy, per-class precision/recall/F1).
The output answers the project document's cross-source generalisation
question: does FinBERT-FOMC stay accurate when applied to chair
speeches, testimony, etc. (issue #40).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path("/data") if Path("/data").exists() else BACKEND_ROOT.parent / "data"


def join_predictions_to_source_type(
    predictions: Iterable[dict[str, Any]],
    registry: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach `source_type` from the registry to each prediction by record_id.

    Predictions whose record_id is not in the registry are dropped
    silently — they cannot be stratified.
    """

    by_id: dict[str, str] = {}
    for row in registry:
        rid = str(row.get("record_id", ""))
        if rid:
            by_id[rid] = str(row.get("source_type", "") or "unknown")

    joined: list[dict[str, Any]] = []
    for prediction in predictions:
        rid = str(prediction.get("record_id", ""))
        if rid not in by_id:
            continue
        out = dict(prediction)
        out["source_type"] = by_id[rid]
        joined.append(out)
    return joined


def compute_stratified_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Group rows by source_type and compute macro / per-class metrics per group."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        st = str(row.get("source_type", "") or "unknown")
        groups.setdefault(st, []).append(row)

    output: dict[str, Any] = {}
    for source_type, group in groups.items():
        gold = [str(r.get("mapped_label", "")).lower() for r in group]
        pred = [str(r.get("predicted_label", "")).lower() for r in group]
        accuracy = sum(g == p for g, p in zip(gold, pred)) / len(group)
        per_class = _per_class_prf(pred, gold)
        macro_f1 = (
            sum(c["f1"] for c in per_class.values()) / len(per_class)
            if per_class
            else 0.0
        )
        output[source_type] = {
            "support": len(group),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_class": per_class,
        }
    return output


def _per_class_prf(pred: Sequence[str], gold: Sequence[str]) -> dict[str, dict[str, float]]:
    labels = sorted(set(pred) | set(gold))
    out: dict[str, dict[str, float]] = {}
    for label in labels:
        if not label:
            continue
        tp = sum(1 for p, g in zip(pred, gold) if p == label and g == label)
        fp = sum(1 for p, g in zip(pred, gold) if p == label and g != label)
        fn = sum(1 for p, g in zip(pred, gold) if p != label and g == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        support = sum(1 for g in gold if g == label)
        out[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute source-type stratified metrics from a fine-tune-batch run."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="JSONL of per-row predictions ({record_id, mapped_label, predicted_label}).",
    )
    parser.add_argument(
        "--registry",
        default=str(DEFAULT_DATA_DIR / "raw" / "phase2" / "source_registry.jsonl"),
        help="Source registry JSONL (provides source_type per record_id).",
    )
    parser.add_argument("--output", required=True, help="Output JSON path for the stratified table.")
    args = parser.parse_args()

    predictions = _read_jsonl(Path(args.predictions))
    registry = _read_jsonl(Path(args.registry))

    joined = join_predictions_to_source_type(predictions, registry)
    metrics = compute_stratified_metrics(joined)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Stratified metrics written to {output}")
    print(json.dumps({st: m["support"] for st, m in metrics.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
