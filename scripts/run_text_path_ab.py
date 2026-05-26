"""Text-path A/B comparison runner (issue #327).

Exercises three configurations on the canonical fold protocol:

- ``broadcast_static`` -- the pre-#327 path; one pooled text vector
  tiled across every lookback bar (``text_channel='scalar'``).
- ``per_bar`` (Arm A) -- the recurrent forecaster with
  ``text_channel='per_bar'`` so each lookback bar gets its own
  pooled-text projection.
- ``flat_mlp`` (Arm B) -- the no-sequence-wrap comparator; flat MLP on
  ``[pooled_market || pooled_text_adapter || rich]``.

The runner pins ``input_size=RICH_FEATURE_SIZE`` explicitly per the
#322 follow-up contract (without it, the canonical fold protocol
crashes when ``rich_features=True`` widens the per-bar payload).
Head mode defaults to ``dual`` so the comparison reads off the new
canonical training objective (ADR 0015 / #322 + the PR #354 default
flip).

Output JSON shape mirrors ``dual_head_comparison_canonical.json``::

    {
      "arms": ["broadcast_static", "per_bar", "flat_mlp"],
      "seeds": [...],
      "fold_ids": [...],
      "training_package_id": "...",
      "trials": {
        "broadcast_static": [ {"seed": ..., "folds": [...]}, ... ],
        "per_bar":          [ ... ],
        "flat_mlp":         [ ... ]
      },
      "summary": {
        "broadcast_static": {
          "regime_f1_macro": {"mean": ..., "std": ..., "min": ..., "max": ..., "n": ...} | None,
          "regression_rmse_log_rv": ...
        },
        ...
      }
    }

Usage::

    docker compose run --rm backend python -m scripts.run_text_path_ab \\
        --training-package-id <id> \\
        --output artifacts/experiments/text_path_ab.json \\
        --seeds 11 29 47 71 97 \\
        --epochs 40 \\
        --regression-alpha 0.5 \\
        --text-encoder finbert_fed_adjacent
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from app.config import BACKEND_ROOT


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
            "``artifacts/experiments/text_path_ab.json``."
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
        "--head-mode",
        choices=("classification", "regression", "dual"),
        default="dual",
        help=(
            "Head mode for the comparison. Defaults to ``dual`` so the "
            "A/B reads off the post-#322 canonical objective."
        ),
    )
    parser.add_argument(
        "--regression-alpha",
        type=float,
        default=0.5,
        help="alpha for head_mode='dual' joint loss.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Hidden size shared across all three arms.",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("broadcast_static", "per_bar", "flat_mlp"),
        default=["broadcast_static", "per_bar", "flat_mlp"],
        help="Subset of arms to evaluate (defaults to all three).",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=None,
        help=(
            "Subset of walk-forward fold IDs to evaluate. Defaults to "
            "every fold in the package's "
            "fold_manifest_expanding_walk_forward.json."
        ),
    )
    parser.add_argument(
        "--text-encoder",
        type=str,
        default="finbert_fed_adjacent",
        help=(
            "Encoder alias driving the loader's pooled embedding cache. "
            "Defaults to the canonical FOMC encoder."
        ),
    )
    parser.add_argument(
        "--text-adapter-dim",
        type=int,
        default=64,
        help=(
            "Projection target for the text adapter. Held constant "
            "across arms so the parameter count is comparable."
        ),
    )
    parser.add_argument(
        "--text-adapter-warm-start",
        type=Path,
        default=None,
        help=(
            "Optional warm-start state_dict for the text adapter. "
            "When set the recurrent arms load the persisted weights "
            "into ``text_adapter`` at construction time."
        ),
    )
    return parser.parse_args()


def _resolve_fold_ids(training_package_id: str, override: list[str] | None) -> list[str]:
    if override:
        return list(override)
    from app.training.loaders import (
        _read_fold_manifest,
        _resolve_training_package_dir,
    )

    package_dir = _resolve_training_package_dir(training_package_id)
    manifest = _read_fold_manifest(package_dir)
    if not manifest:
        raise RuntimeError(
            "fold_manifest_expanding_walk_forward.json is empty / missing "
            f"for training_package_id={training_package_id!r}; provide "
            "--folds explicitly."
        )
    return sorted(manifest.keys())


def _resolve_output_path(arg: Path | None) -> Path:
    if arg is not None:
        return arg
    base = BACKEND_ROOT.parent / "artifacts" / "experiments"
    base.mkdir(parents=True, exist_ok=True)
    return base / "text_path_ab.json"


def _trial_metrics(summary: Any) -> dict[str, float | None]:
    test = getattr(summary, "test_metrics", None) or getattr(summary, "metrics", None)
    if test is None:
        return {}
    return {
        "regime_f1_macro": getattr(test, "regime_f1_macro", None),
        "regime_accuracy": getattr(test, "regime_accuracy", None),
        "regime_loss": getattr(test, "regime_loss", None),
        "regression_rmse_log_rv": getattr(test, "regression_rmse_log_rv", None),
        "regression_mae_log_rv": getattr(test, "regression_mae_log_rv", None),
        "regression_loss": getattr(test, "regression_loss", None),
    }


def _summary_stats(values: list[float]) -> dict[str, float] | None:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return None
    return {
        "mean": statistics.fmean(finite),
        "std": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "max": max(finite),
        "n": len(finite),
    }


def _arm_config(arm: str, args: argparse.Namespace) -> dict[str, Any]:
    """Translate an arm name into a (config-kwargs, text-channel) pair."""

    # RICH_FEATURE_SIZE is mandatory (#322 follow-up). All three arms
    # consume the rich-feature payload, so the input_size on the
    # ModelConfig must be widened explicitly.
    from app.models.config import RICH_FEATURE_SIZE

    base: dict[str, Any] = {
        "input_size": RICH_FEATURE_SIZE,
        "output_mode": "classification",
        "head_mode": args.head_mode,
        "regression_alpha": float(args.regression_alpha),
        "n_classes": 3,
        "hidden_size": int(args.hidden_size),
        "text_adapter_dim": int(args.text_adapter_dim),
    }
    if arm == "broadcast_static":
        base["architecture"] = "lstm"
        base["text_channel"] = "scalar"
    elif arm == "per_bar":
        base["architecture"] = "lstm"
        base["text_channel"] = "per_bar"
    elif arm == "flat_mlp":
        base["architecture"] = "flat_mlp"
        base["text_channel"] = "scalar"
    else:
        raise ValueError(f"unknown arm: {arm!r}")
    return base


def _run_one_cell(
    arm: str,
    seed: int,
    args: argparse.Namespace,
    *,
    fold_ids: list[str],
) -> dict[str, Any]:
    """Train + evaluate one (arm, seed) cell across every fold."""

    from app.models.config import ModelConfig
    from app.training.loaders import load_walk_forward_split
    from app.training.loop import train_model

    config_kwargs = _arm_config(arm, args)
    config = ModelConfig(**config_kwargs)

    per_fold: list[dict[str, Any]] = []
    for fold_id in fold_ids:
        split = load_walk_forward_split(
            training_package_id=args.training_package_id,
            fold_id=fold_id,
            rich_features=True,
            text_encoder=str(args.text_encoder),
        )
        train_kwargs: dict[str, Any] = {
            "model_config": config,
            "train_sequence_groups": split.train,
            "val_sequence_groups": split.val,
            "test_sequence_groups": split.test,
            "fold_id": split.fold_id,
            "protocol": split.protocol,
            "epochs": int(args.epochs),
            "seed": int(seed),
            "save_checkpoint": False,
        }
        if args.text_adapter_warm_start and arm != "flat_mlp":
            train_kwargs["text_adapter_warm_start"] = str(
                args.text_adapter_warm_start
            )
        result = train_model(**train_kwargs)
        per_fold.append(
            {
                "fold_id": split.fold_id,
                "metrics": _trial_metrics(result.summary),
            }
        )

    return {
        "arm": arm,
        "seed": seed,
        "config": config_kwargs,
        "folds": per_fold,
    }


def main() -> int:
    args = _parse_args()
    output_path = _resolve_output_path(args.output)
    print(f"[text_path_ab] writing -> {output_path}")

    fold_ids = _resolve_fold_ids(args.training_package_id, args.folds)
    print(f"[text_path_ab] folds={fold_ids}")

    trials: dict[str, list[dict[str, Any]]] = {arm: [] for arm in args.arms}
    for arm in args.arms:
        for seed in args.seeds:
            print(
                f"[text_path_ab] arm={arm} seed={seed} epochs={args.epochs}",
                flush=True,
            )
            trials[arm].append(
                _run_one_cell(arm, seed, args, fold_ids=fold_ids)
            )

    summary: dict[str, Any] = {}
    for arm, trial_list in trials.items():
        per_fold_f1: list[float] = []
        per_fold_rmse: list[float] = []
        for trial in trial_list:
            for fold in trial["folds"]:
                metrics = fold.get("metrics", {}) or {}
                f1 = metrics.get("regime_f1_macro")
                rmse = metrics.get("regression_rmse_log_rv")
                if f1 is not None:
                    per_fold_f1.append(float(f1))
                if rmse is not None:
                    per_fold_rmse.append(float(rmse))
        summary[arm] = {
            "regime_f1_macro": _summary_stats(per_fold_f1),
            "regression_rmse_log_rv": _summary_stats(per_fold_rmse),
        }

    payload = {
        "arms": list(args.arms),
        "seeds": list(args.seeds),
        "fold_ids": fold_ids,
        "epochs": int(args.epochs),
        "head_mode": str(args.head_mode),
        "regression_alpha": float(args.regression_alpha),
        "training_package_id": args.training_package_id,
        "text_encoder": str(args.text_encoder),
        "text_adapter_dim": int(args.text_adapter_dim),
        "text_adapter_warm_start": (
            str(args.text_adapter_warm_start)
            if args.text_adapter_warm_start
            else None
        ),
        "trials": trials,
        "summary": summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[text_path_ab] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
