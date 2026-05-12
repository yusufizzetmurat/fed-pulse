"""Phase 3 fine-tune pilot: single-seed, single-fold fine-tune of FinBERT-FOMC.

Demonstrates that the deep-learning pipeline can clear the majority-class floor
when a model is actually trained on our data, rather than used zero-shot.
Scope is intentionally narrow: one model, one seed, one fold.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from app.config import DATA_DIR as DEFAULT_DATA_DIR
DEFAULT_ARTIFACT_ROOT = DEFAULT_DATA_DIR / "artifacts" / "phase3"

LABELS = ("dovish", "neutral", "hawkish")
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}

DEFAULT_CHECKPOINT = "ZiweiChen/FinBERT-FOMC"


@dataclass
class EvalRow:
    text: str
    label: str
    event_date: str
    record_id: str = ""


def _set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None


def _load_registry_rows(package_dir: Path) -> list[EvalRow]:
    path = package_dir / "registry_normalized.jsonl"
    rows: list[EvalRow] = []
    if not path.exists():
        raise SystemExit(f"Missing registry: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        label = str(payload.get("mapped_label", "")).strip().lower()
        text = str(payload.get("text", "")).strip()
        event_date = str(payload.get("event_date", "")).strip()
        record_id = str(payload.get("record_id", "")).strip()
        if label in LABELS and text and event_date:
            rows.append(EvalRow(text=text, label=label, event_date=event_date, record_id=record_id))
    return rows


def _load_fold(package_dir: Path, fold_id: str) -> dict[str, Any]:
    path = package_dir / "fold_manifest_expanding_walk_forward.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for fold in payload.get("folds", []):
        if fold.get("fold_id") == fold_id:
            return fold
    raise SystemExit(f"Fold {fold_id} not found in {path}")


def _split_by_fold(rows: list[EvalRow], fold: dict[str, Any]) -> tuple[list[EvalRow], list[EvalRow]]:
    train_end = str(fold["train_end"])
    test_start = str(fold["test_start"])
    test_end = str(fold["test_end"])
    train_rows = [r for r in rows if r.event_date <= train_end]
    test_rows = [r for r in rows if test_start <= r.event_date <= test_end]
    return train_rows, test_rows


class TextClassificationDataset(Dataset):
    def __init__(self, rows: list[EvalRow], tokenizer: Any, max_length: int = 256) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        enc = self.tokenizer(
            row.text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(LABEL2ID[row.label], dtype=torch.long),
        }


def _compute_classification_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    support = Counter(y_true)
    per_class: dict[str, dict[str, float]] = {}
    total = len(y_true)
    correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)

    weighted_f1_sum = 0.0
    macro_values: list[float] = []
    for label in LABELS:
        tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support[label]}
        macro_values.append(f1)
        weighted_f1_sum += f1 * support[label]

    return {
        "macro_f1": statistics.mean(macro_values) if macro_values else 0.0,
        "weighted_f1": weighted_f1_sum / total if total else 0.0,
        "accuracy": correct / total if total else 0.0,
        "per_class": per_class,
    }


def _latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0}
    values = sorted(latencies_ms)

    def _pct(p: float) -> float:
        idx = int((len(values) - 1) * p)
        return values[idx]

    return {"p50_ms": _pct(0.50), "p95_ms": _pct(0.95)}


def write_predictions_jsonl(
    *,
    record_ids: list[str],
    gold_labels: list[str],
    predicted_labels: list[str],
    output_path: Path,
) -> None:
    """Persist per-row test predictions as JSONL.

    Each line: {record_id, mapped_label, predicted_label}. Consumed by
    app.data.source_type_stratified_analysis to compute per-source-type
    metrics without re-running training. Raises ValueError if the three
    input lists have different lengths.
    """
    if not (len(record_ids) == len(gold_labels) == len(predicted_labels)):
        raise ValueError(
            "record_ids, gold_labels, predicted_labels must have the same length; "
            f"got {len(record_ids)}, {len(gold_labels)}, {len(predicted_labels)}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for rid, gold, pred in zip(record_ids, gold_labels, predicted_labels):
            handle.write(
                json.dumps(
                    {"record_id": rid, "mapped_label": gold, "predicted_label": pred}
                )
                + "\n"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 fine-tune pilot.")
    parser.add_argument("--training-package-id", required=True)
    parser.add_argument("--fold-id", default="wf_fold_2")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--owner", default="unknown")
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    return parser.parse_args()


def run_one(args: argparse.Namespace, *, artifact_dir: Path | None = None) -> dict[str, Any]:
    _set_all_seeds(args.seed)

    package_dir = DEFAULT_DATA_DIR / "processed" / args.training_package_id
    if not package_dir.exists():
        raise SystemExit(f"Training package not found: {package_dir}")

    rows = _load_registry_rows(package_dir)
    if not rows:
        raise SystemExit("Empty registry.")

    fold = _load_fold(package_dir, args.fold_id)
    train_rows, test_rows = _split_by_fold(rows, fold)
    if not train_rows or not test_rows:
        raise SystemExit(
            f"Insufficient rows for fold {args.fold_id}: train={len(train_rows)} test={len(test_rows)}"
        )
    print(f"[pilot] fold={args.fold_id} train_rows={len(train_rows)} test_rows={len(test_rows)}")

    hf_token = _hf_token()
    if hf_token:
        print(f"[pilot] using HF token (len={len(hf_token)})")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        token=hf_token,
    )

    train_ds = TextClassificationDataset(train_rows, tokenizer, max_length=args.max_length)

    run_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if artifact_dir is None:
        artifact_dir = Path(args.artifact_root) / f"pilot_finetune_{run_token}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = artifact_dir / "hf_checkpoints"

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        data_seed=args.seed,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        disable_tqdm=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds)

    print(f"[pilot] fine-tuning {args.checkpoint} for {args.epochs} epochs on {len(train_rows)} rows")
    train_output = trainer.train()
    train_runtime = float(train_output.metrics.get("train_runtime", 0.0))
    train_loss = float(train_output.metrics.get("train_loss", 0.0))
    print(f"[pilot] train_runtime={train_runtime:.1f}s train_loss={train_loss:.4f}")

    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    print(f"[pilot] model and tokenizer saved to {checkpoint_dir}")

    # Inference mode (disables dropout, batchnorm-updates, etc.)
    model.train(False)
    device = next(model.parameters()).device
    y_true: list[str] = []
    y_pred: list[str] = []
    latencies_ms: list[float] = []
    batch_size = args.eval_batch_size
    with torch.no_grad():
        for start in range(0, len(test_rows), batch_size):
            batch_rows = test_rows[start : start + batch_size]
            batch_texts = [r.text for r in batch_rows]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            t0 = time.perf_counter()
            logits = model(**enc).logits
            elapsed_ms = (time.perf_counter() - t0) * 1000
            per_item_ms = elapsed_ms / max(len(batch_rows), 1)
            latencies_ms.extend([per_item_ms] * len(batch_rows))
            batch_preds = logits.argmax(dim=-1).tolist()
            for row, pred_idx in zip(batch_rows, batch_preds):
                y_true.append(row.label)
                y_pred.append(ID2LABEL[int(pred_idx)])

    cls_metrics = _compute_classification_metrics(y_true, y_pred)
    latency = _latency_summary(latencies_ms)
    print(
        f"[pilot] macro_f1={cls_metrics['macro_f1']:.4f} "
        f"acc={cls_metrics['accuracy']:.4f} weighted_f1={cls_metrics['weighted_f1']:.4f}"
    )

    metrics = {
        "pipeline": "phase3_finetune_pilot",
        "owner": args.owner,
        "checkpoint": args.checkpoint,
        "fold_id": args.fold_id,
        "seed": args.seed,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_runtime_s": train_runtime,
        "train_loss": train_loss,
        "classification": cls_metrics,
        "latency": latency,
        "training_package_id": args.training_package_id,
        "started_at_utc": run_token,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }

    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[pilot] metrics written to {artifact_dir / 'metrics.json'}")

    predictions_path = artifact_dir / "predictions.jsonl"
    write_predictions_jsonl(
        record_ids=[r.record_id for r in test_rows],
        gold_labels=y_true,
        predicted_labels=y_pred,
        output_path=predictions_path,
    )
    print(f"[pilot] predictions written to {predictions_path}")

    return metrics


def main() -> int:
    args = _parse_args()
    run_one(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
