"""Family zero-out smoke (#505 A.1.b).

Trains a baseline cell with every per-family rich-feature flag ON,
then re-trains once per family with that family's loader flag flipped
off. Reports the regime_f1_macro delta per family.

A family whose delta is exactly 0.000 is almost always a silent zeros
bug in the feature path, not a real null result. Flag it.

Default budget: 1 seed, 1 fold, gru, 8 epochs => ~30 GPU-min on H200
for the full 11-family sweep + baseline (12 cells).

Usage:

    python -m scripts.run_family_zeroout_smoke \\
        --training-package-id <tp_id> \\
        --output backend/artifacts/audits/pre_sweep_<tp_id>/family_zeroout.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_FOLD = "wf_fold_3"
DEFAULT_SEED = 11
DEFAULT_EPOCHS = 8
DEFAULT_HIDDEN = 128

FAMILIES: tuple[str, ...] = (
    "credibility",
    "linguistic",
    "mp_surprise",
    "multi_axis",
    "llm_features",
    "retrieval_analogs",
    "regime_conditioning",
    "sep",
    "press_conf",
    "statement_delta",
    "vote_features",
)


def _baseline_flags() -> dict[str, bool]:
    """Loader kwargs that turn every family ON."""

    return {f"use_{family}": True for family in FAMILIES}


def _run_one_arm(
    *,
    training_package_id: str,
    fold_id: str,
    seed: int,
    epochs: int,
    hidden_size: int,
    loader_flags: dict[str, bool],
) -> dict[str, Any]:
    """Train one cell and return the test metrics."""

    from app.models.config import RICH_FEATURE_SIZE, ModelConfig
    from app.training.loaders import load_walk_forward_split
    from app.training.loop import train_model

    config = ModelConfig(
        input_size=RICH_FEATURE_SIZE,
        output_mode="classification",
        head_mode="classification",
        n_classes=3,
        hidden_size=hidden_size,
    )
    split = load_walk_forward_split(
        training_package_id=training_package_id,
        fold_id=fold_id,
        rich_features=True,
        **loader_flags,
    )
    result = train_model(
        model_config=config,
        train_sequence_groups=split.train,
        val_sequence_groups=split.val,
        test_sequence_groups=split.test,
        fold_id=split.fold_id,
        protocol=split.protocol,
        epochs=epochs,
        seed=seed,
        save_checkpoint=False,
    )
    test = (
        getattr(result.summary, "test_metrics", None)
        or getattr(result.summary, "metrics", None)
    )
    return {
        "regime_f1_macro": float(getattr(test, "regime_f1_macro", 0.0) or 0.0),
        "regime_accuracy": float(getattr(test, "regime_accuracy", 0.0) or 0.0),
    }


def run_smoke(
    *,
    training_package_id: str,
    fold_id: str = DEFAULT_FOLD,
    seed: int = DEFAULT_SEED,
    epochs: int = DEFAULT_EPOCHS,
    hidden_size: int = DEFAULT_HIDDEN,
) -> dict[str, Any]:
    """Run the baseline + one zero-out per family. Return the report."""

    baseline_flags = _baseline_flags()
    print(
        f"[family-zeroout] baseline (all families ON), "
        f"fold={fold_id} seed={seed} epochs={epochs}",
        flush=True,
    )
    baseline_metrics = _run_one_arm(
        training_package_id=training_package_id,
        fold_id=fold_id,
        seed=seed,
        epochs=epochs,
        hidden_size=hidden_size,
        loader_flags=baseline_flags,
    )

    per_family: list[dict[str, Any]] = []
    flagged_silent_zero: list[str] = []
    for family in FAMILIES:
        flags = dict(baseline_flags)
        flags[f"use_{family}"] = False
        print(
            f"[family-zeroout] family={family} OFF", flush=True
        )
        metrics = _run_one_arm(
            training_package_id=training_package_id,
            fold_id=fold_id,
            seed=seed,
            epochs=epochs,
            hidden_size=hidden_size,
            loader_flags=flags,
        )
        delta = metrics["regime_f1_macro"] - baseline_metrics["regime_f1_macro"]
        is_zero = delta == 0.0
        if is_zero:
            flagged_silent_zero.append(family)
        per_family.append(
            {
                "family": family,
                "metrics": metrics,
                "delta_f1": delta,
                "is_silent_zero_suspect": is_zero,
            }
        )

    return {
        "training_package_id": training_package_id,
        "fold_id": fold_id,
        "seed": seed,
        "epochs": epochs,
        "hidden_size": hidden_size,
        "baseline_metrics": baseline_metrics,
        "per_family": per_family,
        "flagged_silent_zero": flagged_silent_zero,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Family zero-out smoke (#505 A.1.b)")
    parser.add_argument("--training-package-id", required=True)
    parser.add_argument("--fold-id", default=DEFAULT_FOLD)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_smoke(
        training_package_id=args.training_package_id,
        fold_id=args.fold_id,
        seed=args.seed,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
    )
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[family-zeroout] wrote {args.output}")
    else:
        sys.stdout.write(text + "\n")
    if report["flagged_silent_zero"]:
        print(
            "[family-zeroout] silent-zero suspects: "
            + ", ".join(report["flagged_silent_zero"]),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
