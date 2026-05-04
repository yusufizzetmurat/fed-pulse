"""Phase 3 fine-tune full batch: 3 encoders × 5 official seeds on fold 2.

Wraps :mod:`app.data.phase3_finetune_pilot` and aggregates results into a
single ``aggregate.json`` with mean ± std per encoder. Closes GitHub issue #20
(SRS FR-30).
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from app.data.phase3_finetune_pilot import DEFAULT_ARTIFACT_ROOT, run_one

ENCODERS: dict[str, str] = {
    "bert_base_uncased": "bert-base-uncased",
    "finbert": "ProsusAI/finbert",
    "fomc_roberta": "ZiweiChen/FinBERT-FOMC",
}
OFFICIAL_SEEDS: tuple[int, ...] = (11, 29, 47, 71, 97)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 fine-tune full batch (3 encoders × 5 seeds).")
    parser.add_argument("--training-package-id", required=True)
    parser.add_argument("--fold-id", default="wf_fold_2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--owner", default="unknown")
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument(
        "--encoders",
        nargs="+",
        choices=tuple(ENCODERS.keys()),
        default=tuple(ENCODERS.keys()),
        help="Encoder family keys to include.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(OFFICIAL_SEEDS),
        help="Seeds to run per encoder (defaults to the official set).",
    )
    return parser.parse_args()


def _build_run_args(args: argparse.Namespace, *, checkpoint: str, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        training_package_id=args.training_package_id,
        fold_id=args.fold_id,
        checkpoint=checkpoint,
        seed=seed,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        owner=args.owner,
        artifact_root=args.artifact_root,
    )


def _summarize(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    def _values(getter) -> list[float]:
        out: list[float] = []
        for entry in metrics_list:
            try:
                value = getter(entry)
            except (KeyError, TypeError):
                continue
            if value is None:
                continue
            out.append(float(value))
        return out

    def _mean_std(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 0.0, "count": 0}
        return {
            "mean": float(statistics.mean(values)),
            "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
            "count": len(values),
        }

    return {
        "macro_f1": _mean_std(_values(lambda m: m["classification"]["macro_f1"])),
        "weighted_f1": _mean_std(_values(lambda m: m["classification"]["weighted_f1"])),
        "accuracy": _mean_std(_values(lambda m: m["classification"]["accuracy"])),
        "p50_ms": _mean_std(_values(lambda m: m["latency"]["p50_ms"])),
        "p95_ms": _mean_std(_values(lambda m: m["latency"]["p95_ms"])),
        "train_runtime_s": _mean_std(_values(lambda m: m["train_runtime_s"])),
    }


def _release_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    args = _parse_args()
    selected_encoders: dict[str, str] = {key: ENCODERS[key] for key in args.encoders}
    seeds = list(args.seeds)

    batch_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_dir = Path(args.artifact_root) / f"finetune_batch_{batch_token}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total_runs = len(selected_encoders) * len(seeds)
    print(f"[batch] encoders={list(selected_encoders.keys())} seeds={seeds} total={total_runs}")
    print(f"[batch] artifact root: {batch_dir}")

    run_index = 0
    for encoder_key, checkpoint in selected_encoders.items():
        for seed in seeds:
            run_index += 1
            run_args = _build_run_args(args, checkpoint=checkpoint, seed=seed)
            run_dir = batch_dir / f"{encoder_key}_s{seed}"
            print(f"\n[batch] ({run_index}/{total_runs}) encoder={encoder_key} seed={seed}")
            try:
                metrics = run_one(run_args, artifact_dir=run_dir)
                metrics_with_meta = dict(metrics)
                metrics_with_meta["encoder_key"] = encoder_key
                metrics_with_meta["status"] = "succeeded"
                runs.append(metrics_with_meta)
            except Exception as exc:
                print(f"[batch] FAILED encoder={encoder_key} seed={seed}: {exc}")
                failures.append(
                    {
                        "encoder_key": encoder_key,
                        "checkpoint": checkpoint,
                        "seed": seed,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            finally:
                _release_gpu()

    by_encoder: dict[str, dict[str, Any]] = {}
    for encoder_key in selected_encoders:
        encoder_runs = [r for r in runs if r.get("encoder_key") == encoder_key]
        by_encoder[encoder_key] = {
            "checkpoint": selected_encoders[encoder_key],
            "run_count": len(encoder_runs),
            "summary": _summarize(encoder_runs),
            "per_seed": {
                str(r["seed"]): {
                    "macro_f1": r["classification"]["macro_f1"],
                    "weighted_f1": r["classification"]["weighted_f1"],
                    "accuracy": r["classification"]["accuracy"],
                    "p95_ms": r["latency"]["p95_ms"],
                    "train_runtime_s": r["train_runtime_s"],
                }
                for r in encoder_runs
            },
        }

    aggregate = {
        "pipeline": "phase3_finetune_batch",
        "owner": args.owner,
        "training_package_id": args.training_package_id,
        "fold_id": args.fold_id,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "max_length": args.max_length,
        "seeds": seeds,
        "encoders": list(selected_encoders.keys()),
        "started_at_utc": batch_token,
        "ended_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "total_runs": total_runs,
        "succeeded": len(runs),
        "failed": len(failures),
        "by_encoder": by_encoder,
        "failures": failures,
    }
    (batch_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"\n[batch] aggregate written to {batch_dir / 'aggregate.json'}")
    print(f"[batch] succeeded={len(runs)} failed={len(failures)} of {total_runs}")
    for encoder_key, payload in by_encoder.items():
        macro = payload["summary"]["macro_f1"]
        print(
            f"[batch] {encoder_key}: macro_f1={macro['mean']:.4f} ± {macro['std']:.4f} "
            f"(n={macro['count']})"
        )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
