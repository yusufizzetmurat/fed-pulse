from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import pipeline

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path("/data") if Path("/data").exists() else BACKEND_ROOT.parent / "data"
DEFAULT_ARTIFACT_ROOT = DEFAULT_DATA_DIR / "artifacts" / "phase3"
DEFAULT_SEEDS = [11, 29, 47, 71, 97]
LABELS = ("hawkish", "dovish", "neutral")
LABEL_TO_INDEX = {"dovish": -1, "neutral": 0, "hawkish": 1}

MODEL_SPECS = {
    "bert": {
        "model_version": "mv_bert_base_uncased_fast_v1.0.0",
        "checkpoints": ["bert-base-uncased"],
    },
    "finbert": {
        "model_version": "mv_finbert_fast_v1.0.0",
        "checkpoints": ["ProsusAI/finbert"],
    },
    "fomc_roberta": {
        "model_version": "mv_fomc_roberta_fast_v1.0.0",
        "checkpoints": [
            "gtfintechlab/fomc-roberta-any-exp",
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "distilbert-base-uncased-finetuned-sst-2-english",
        ],
    },
}


@dataclass
class EvalRow:
    text: str
    label: str
    event_date: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute Phase 3 baseline evaluation runs. "
            "Mode 'smoke' runs one model/seed gate; mode 'full' runs all official seeds/candidates."
        )
    )
    parser.add_argument("--training-package-id", required=True, help="Training package id under data/processed.")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--model", choices=tuple(MODEL_SPECS.keys()), default="bert")
    parser.add_argument("--seed", type=int, default=11, help="Smoke-mode seed.")
    parser.add_argument("--owner", default="unknown")
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument(
        "--max-eval-rows-per-fold",
        type=int,
        default=300,
        help="Deterministic cap for per-fold test evaluation rows.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size.")
    return parser.parse_args()


def _load_registry_rows(package_dir: Path) -> list[EvalRow]:
    path = package_dir / "registry_normalized.jsonl"
    rows: list[EvalRow] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        label = str(payload.get("mapped_label", "")).strip().lower()
        text = str(payload.get("text", "")).strip()
        event_date = str(payload.get("event_date", "")).strip()
        if label in LABELS and text and event_date:
            rows.append(EvalRow(text=text, label=label, event_date=event_date))
    return rows


def _load_fold_manifest(package_dir: Path) -> list[dict[str, Any]]:
    path = package_dir / "fold_manifest_expanding_walk_forward.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds = payload.get("folds")
    if not isinstance(folds, list):
        return []
    return [fold for fold in folds if isinstance(fold, dict)]


def _fallback_folds(rows: list[EvalRow], fold_count: int = 3) -> list[dict[str, Any]]:
    unique_dates = sorted({row.event_date for row in rows})
    if len(unique_dates) < 12:
        return []
    block = max(1, len(unique_dates) // (fold_count + 2))
    folds = []
    for idx in range(1, fold_count + 1):
        train_end_idx = min(len(unique_dates) - 3 * block, idx * block + block)
        val_start_idx = train_end_idx + 1
        val_end_idx = min(val_start_idx + block - 1, len(unique_dates) - block - 1)
        test_start_idx = val_end_idx + 1
        test_end_idx = min(test_start_idx + block - 1, len(unique_dates) - 1)
        if test_start_idx >= len(unique_dates):
            break
        folds.append(
            {
                "fold_id": f"wf_fold_{idx}",
                "train_start": unique_dates[0],
                "train_end": unique_dates[train_end_idx],
                "val_start": unique_dates[val_start_idx],
                "val_end": unique_dates[val_end_idx],
                "test_start": unique_dates[test_start_idx],
                "test_end": unique_dates[test_end_idx],
            }
        )
    return folds


def _rows_for_test_window(rows: list[EvalRow], test_start: str, test_end: str) -> list[EvalRow]:
    return [row for row in rows if test_start <= row.event_date <= test_end]


def _sample_rows(rows: list[EvalRow], seed: int, cap: int) -> list[EvalRow]:
    if cap <= 0 or len(rows) <= cap:
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), cap))
    return [rows[idx] for idx in indices]


def _map_prediction_label(raw_label: str) -> str:
    label = str(raw_label or "").strip().lower().replace("-", "_")
    if "hawk" in label or "positive" in label or label in {"label_2", "2"}:
        return "hawkish"
    if "dov" in label or "negative" in label or label in {"label_0", "0"}:
        return "dovish"
    if "neutral" in label or "mixed" in label or label in {"label_1", "1"}:
        return "neutral"
    return "neutral"


def _infer_labels(
    classifier: Any,
    texts: list[str],
    *,
    batch_size: int,
) -> tuple[list[str], list[float]]:
    if not texts:
        return [], []

    predictions: list[str] = []
    latencies_ms: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        t0 = time.perf_counter()
        outputs = classifier(batch, truncation=True, max_length=512)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_item_ms = elapsed_ms / max(len(batch), 1)
        latencies_ms.extend([per_item_ms] * len(batch))

        for output in outputs:
            if isinstance(output, list):
                best = max(output, key=lambda item: float(item.get("score", 0.0)))
                predictions.append(_map_prediction_label(str(best.get("label", ""))))
            elif isinstance(output, dict):
                predictions.append(_map_prediction_label(str(output.get("label", ""))))
            else:
                predictions.append("neutral")
    return predictions, latencies_ms


def _compute_classification_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    support = Counter(y_true)
    per_class: dict[str, dict[str, float]] = {}
    total = len(y_true)
    correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)

    weighted_f1_sum = 0.0
    macro_f1_values: list[float] = []
    for label in LABELS:
        tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support[label]}
        macro_f1_values.append(f1)
        weighted_f1_sum += f1 * support[label]

    macro_f1 = statistics.mean(macro_f1_values) if macro_f1_values else 0.0
    weighted_f1 = weighted_f1_sum / total if total else 0.0
    accuracy = correct / total if total else 0.0
    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "per_class": per_class,
    }


def _compute_rmse_mape(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    if not y_true:
        return {"rmse": 0.0, "mape": 0.0}

    squared_errors = []
    ape = []
    for truth, pred in zip(y_true, y_pred):
        t_val = LABEL_TO_INDEX[truth]
        p_val = LABEL_TO_INDEX[pred]
        err = p_val - t_val
        squared_errors.append(err * err)
        denom = max(abs(t_val), 1)
        ape.append(abs(err) / denom)
    rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
    mape = (sum(ape) / len(ape)) * 100
    return {"rmse": rmse, "mape": mape}


def _latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0}
    values = sorted(latencies_ms)

    def _pct(p: float) -> float:
        idx = int((len(values) - 1) * p)
        return values[idx]

    return {"p50_ms": _pct(0.50), "p95_ms": _pct(0.95)}


def _run_single(
    *,
    model_key: str,
    seed: int,
    rows: list[EvalRow],
    folds: list[dict[str, Any]],
    artifact_dir: Path,
    max_eval_rows_per_fold: int,
    batch_size: int,
) -> dict[str, Any]:
    model_spec = MODEL_SPECS[model_key]
    classifier = None
    checkpoint_used = ""
    checkpoint_errors: list[str] = []
    for checkpoint in model_spec["checkpoints"]:
        try:
            classifier = pipeline(
                "text-classification",
                model=checkpoint,
                return_all_scores=True,
            )
            checkpoint_used = checkpoint
            break
        except Exception as exc:  # pragma: no cover - network/model availability variability
            checkpoint_errors.append(f"{checkpoint}: {exc}")
    if classifier is None:
        raise RuntimeError(
            "Unable to load any checkpoint for "
            f"{model_key}. Errors: {' | '.join(checkpoint_errors)}"
        )

    y_true: list[str] = []
    y_pred: list[str] = []
    latencies_ms: list[float] = []
    fold_rows_summary: list[dict[str, Any]] = []

    for fold in folds:
        fold_id = str(fold.get("fold_id", "wf_fold"))
        test_start = str(fold.get("test_start", ""))
        test_end = str(fold.get("test_end", ""))
        test_rows = _rows_for_test_window(rows, test_start, test_end)
        sampled = _sample_rows(test_rows, seed=seed, cap=max_eval_rows_per_fold)

        texts = [row.text for row in sampled]
        truths = [row.label for row in sampled]
        preds, fold_latencies = _infer_labels(classifier, texts, batch_size=batch_size)

        y_true.extend(truths)
        y_pred.extend(preds)
        latencies_ms.extend(fold_latencies)
        fold_rows_summary.append(
            {
                "fold_id": fold_id,
                "test_start": test_start,
                "test_end": test_end,
                "test_rows_total": len(test_rows),
                "test_rows_evaluated": len(sampled),
            }
        )

    cls_metrics = _compute_classification_metrics(y_true, y_pred)
    reg_like_metrics = _compute_rmse_mape(y_true, y_pred)
    latency = _latency_summary(latencies_ms)

    metrics = {
        "model_key": model_key,
        "model_version": model_spec["model_version"],
        "checkpoint": checkpoint_used,
        "checkpoint_fallback_used": checkpoint_used != model_spec["checkpoints"][0],
        "seed": seed,
        "evaluated_rows": len(y_true),
        "classification": cls_metrics,
        "error_proxy": reg_like_metrics,
        "latency": latency,
        "folds": fold_rows_summary,
    }

    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d')}_epv1_{model_spec['model_version']}_s{seed}"
    run_dir = artifact_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"run_id": run_id, **metrics}


def main() -> int:
    args = _parse_args()
    package_dir = DEFAULT_DATA_DIR / "processed" / args.training_package_id
    if not package_dir.exists():
        raise SystemExit(f"Training package not found: {package_dir}")

    rows = _load_registry_rows(package_dir)
    if not rows:
        raise SystemExit(f"No labeled rows found in {package_dir / 'registry_normalized.jsonl'}")

    folds = _load_fold_manifest(package_dir)
    if not folds:
        folds = _fallback_folds(rows)
    if not folds:
        raise SystemExit("No usable folds available for execution.")

    artifact_root = Path(args.artifact_root)
    run_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = artifact_root / f"execution_{run_token}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "smoke":
        run_plan = [(args.model, args.seed)]
    else:
        run_plan = [(model_key, seed) for model_key in MODEL_SPECS for seed in DEFAULT_SEEDS]

    all_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for model_key, seed in run_plan:
        print(f"[phase3] running model={model_key} seed={seed}")
        try:
            result = _run_single(
                model_key=model_key,
                seed=seed,
                rows=rows,
                folds=folds,
                artifact_dir=artifact_dir,
                max_eval_rows_per_fold=args.max_eval_rows_per_fold,
                batch_size=args.batch_size,
            )
            all_results.append(result)
            print(
                f"[phase3] done run_id={result['run_id']} macro_f1={result['classification']['macro_f1']:.4f} "
                f"rmse={result['error_proxy']['rmse']:.4f} mape={result['error_proxy']['mape']:.2f}%"
            )
        except Exception as exc:  # pragma: no cover - keep full batch resilient
            failures.append({"model_key": model_key, "seed": seed, "error": str(exc)})
            print(f"[phase3] failed model={model_key} seed={seed}: {exc}")

    summary = {
        "mode": args.mode,
        "owner": args.owner,
        "training_package_id": args.training_package_id,
        "started_at_utc": run_token,
        "official_seeds": DEFAULT_SEEDS,
        "results": all_results,
        "failures": failures,
    }
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[phase3] summary written to {artifact_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
