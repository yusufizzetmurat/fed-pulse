"""Per-rates-head walk-forward sweep runner (#292).

Trains the three rates heads (``2y`` / ``5y`` / ``terminal``) on the
official 5-seed × 4-fold protocol and emits a per-head MAE-bps +
directional-accuracy + R² panel keyed by fold and seed. The aggregated
summary is the headline the §16 finalization-roadmap table reads.

Usage::

    docker compose run --rm backend python -m scripts.run_rates_heads_sweep \\
        --training-package-id <id> \\
        --output artifacts/experiments/rates_heads_sweep.json \\
        --seeds 11 29 47 71 97 \\
        --epochs 40 \\
        --rates-head-mode regression

Output JSON shape::

    {
      "seeds": [11, 29, 47, 71, 97],
      "fold_ids": ["wf_fold_1", ...],
      "trials": [
        {"seed": 11, "fold_id": "wf_fold_1", "rates_metrics": {...}, "scalers": {...}},
        ...
      ],
      "summary": {
        "2y":      {"mae_bps_mean": float, "mae_bps_std": float, ...},
        "5y":      {...},
        "terminal": {...}
      }
    }
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from app.config import BACKEND_ROOT
from app.models.rates_heads import RATES_HEAD_NAMES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-package-id",
        required=True,
        help="Training-package ID under ``backend/artifacts/training_packages/<id>``.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Defaults to "
            "``artifacts/experiments/rates_heads_sweep.json``."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[11, 29, 47, 71, 97],
        help="Official seed set. Default mirrors docs/benchmark-policy.md.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Epochs per cell (default 40).",
    )
    parser.add_argument(
        "--rates-heads",
        nargs="+",
        choices=list(RATES_HEAD_NAMES),
        default=list(RATES_HEAD_NAMES),
        help="Subset of rates heads to run (defaults to all three).",
    )
    parser.add_argument(
        "--rates-head-mode",
        choices=("regression", "classification", "dual"),
        default="regression",
        help="Per-head training mode shared across the sweep.",
    )
    parser.add_argument(
        "--rates-alpha",
        type=float,
        default=0.5,
        help="alpha for ``--rates-head-mode=dual`` joint loss.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Hidden size shared across folds.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=None,
        help=(
            "Subset of walk-forward fold IDs to evaluate. Defaults to "
            "every fold present in the training package's "
            "fold_manifest_expanding_walk_forward.json."
        ),
    )
    return parser.parse_args()


def _resolve_fold_ids(training_package_id: str, override: list[str] | None) -> list[str]:
    if override:
        return list(override)
    from app.training.loaders import _read_fold_manifest, _resolve_training_package_dir

    package_dir = _resolve_training_package_dir(training_package_id)
    manifest = _read_fold_manifest(package_dir)
    if not manifest:
        raise RuntimeError(
            "fold_manifest_expanding_walk_forward.json is empty / missing "
            f"for training_package_id={training_package_id!r}"
        )
    return sorted(manifest.keys())


def _train_single(
    *,
    training_package_id: str,
    fold_id: str,
    seed: int,
    rates_heads: tuple[str, ...],
    rates_head_mode: str,
    rates_alpha: float,
    hidden_size: int,
    epochs: int,
) -> dict[str, Any]:
    from app.models.config import ModelConfig
    from app.training.loaders import load_walk_forward_split
    from app.training.loop import train_model

    config = ModelConfig(
        output_mode="classification",
        n_classes=3,
        hidden_size=hidden_size,
        head_hidden_size=hidden_size // 2,
        rates_heads=rates_heads,
        rates_head_mode=rates_head_mode,
        rates_alpha=rates_alpha,
    )
    split = load_walk_forward_split(training_package_id, fold_id=fold_id)
    result = train_model(
        model_config=config,
        train_sequence_groups=split.train,
        val_sequence_groups=split.val,
        test_sequence_groups=split.test,
        fold_id=fold_id,
        protocol="walk-forward",
        epochs=epochs,
        seed=seed,
        save_checkpoint=False,
        use_compile=False,
        use_amp=False,
    )
    metrics = result.summary.metrics
    # #317 finding #6: persist the per-fold rates_quantile_edges into
    # the canonical fold_manifest so inference can read back the same
    # tertile cutoffs the model trained against (the per-trial JSON
    # also keeps them but is per-seed; the fold_manifest is the
    # cross-seed source of truth).
    try:
        from app.data.build_training_package import (
            update_fold_manifest_rates_quantile_edges,
        )
        from app.training.loaders import _resolve_training_package_dir

        package_dir = _resolve_training_package_dir(training_package_id)
        update_fold_manifest_rates_quantile_edges(
            package_dir, fold_id, result.summary.rates_quantile_edges
        )
    except Exception:  # pragma: no cover -- never let upsert break the sweep
        pass
    return {
        "seed": seed,
        "fold_id": fold_id,
        "epochs_completed": result.summary.epochs_completed,
        "rates_metrics": metrics.rates_metrics if metrics else None,
        "rates_scalers": result.summary.rates_scalers,
        "rates_quantile_edges": result.summary.rates_quantile_edges,
    }


def _aggregate(trials: list[dict[str, Any]], rates_heads: tuple[str, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for head in rates_heads:
        mae_samples: list[float] = []
        dir_samples: list[float] = []
        r2_samples: list[float] = []
        for trial in trials:
            block = (trial.get("rates_metrics") or {}).get(head) or {}
            mae_payload = block.get("mae_bps")
            if isinstance(mae_payload, dict):
                p = mae_payload.get("point")
                if p is not None and math.isfinite(float(p)):
                    mae_samples.append(float(p))
            dir_payload = block.get("directional_accuracy")
            if isinstance(dir_payload, dict):
                p = dir_payload.get("point")
                if p is not None and math.isfinite(float(p)):
                    dir_samples.append(float(p))
            r2_payload = block.get("r_squared")
            if isinstance(r2_payload, dict):
                p = r2_payload.get("point")
                if p is not None and math.isfinite(float(p)):
                    r2_samples.append(float(p))

        def _mean_std(samples: list[float]) -> tuple[float | None, float | None]:
            if not samples:
                return None, None
            if len(samples) == 1:
                return float(samples[0]), 0.0
            return float(statistics.mean(samples)), float(statistics.stdev(samples))

        mae_mean, mae_std = _mean_std(mae_samples)
        dir_mean, dir_std = _mean_std(dir_samples)
        r2_mean, r2_std = _mean_std(r2_samples)
        summary[head] = {
            "n_trials": len(mae_samples),
            "mae_bps_mean": mae_mean,
            "mae_bps_std": mae_std,
            "directional_accuracy_mean": dir_mean,
            "directional_accuracy_std": dir_std,
            "r_squared_mean": r2_mean,
            "r_squared_std": r2_std,
        }
    return summary


def _resolve_output(path: Path | None) -> Path:
    if path is not None:
        return path
    return BACKEND_ROOT.parent / "artifacts" / "experiments" / "rates_heads_sweep.json"


def main() -> None:
    args = _parse_args()
    fold_ids = _resolve_fold_ids(args.training_package_id, args.folds)
    rates_heads = tuple(args.rates_heads)
    trials: list[dict[str, Any]] = []
    for seed in args.seeds:
        for fold_id in fold_ids:
            print(
                f"[rates_heads_sweep] seed={seed} fold={fold_id} mode={args.rates_head_mode}",
                flush=True,
            )
            trial = _train_single(
                training_package_id=args.training_package_id,
                fold_id=fold_id,
                seed=int(seed),
                rates_heads=rates_heads,
                rates_head_mode=args.rates_head_mode,
                rates_alpha=float(args.rates_alpha),
                hidden_size=int(args.hidden_size),
                epochs=int(args.epochs),
            )
            trials.append(trial)
    summary = _aggregate(trials, rates_heads)
    payload = {
        "seeds": list(args.seeds),
        "fold_ids": fold_ids,
        "rates_heads": list(rates_heads),
        "rates_head_mode": args.rates_head_mode,
        "rates_alpha": float(args.rates_alpha),
        "trials": trials,
        "summary": summary,
    }
    output_path = _resolve_output(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[rates_heads_sweep] wrote {output_path}", flush=True)
    for head, stats in summary.items():
        print(
            f"[rates_heads_sweep] {head}: "
            f"mae_bps={stats['mae_bps_mean']} ± {stats['mae_bps_std']}  "
            f"dir_acc={stats['directional_accuracy_mean']} ± {stats['directional_accuracy_std']}  "
            f"r2={stats['r_squared_mean']} ± {stats['r_squared_std']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
