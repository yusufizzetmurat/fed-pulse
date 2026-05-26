"""Three-way derived-text-features ablation runner (#309).

Trains the forecaster under three configurations on the same
walk-forward fold protocol:

- ``baseline``: ``use_derived_text_features=True`` -- the pre-#309
  baseline (per-sentence sentiment / stance / certainty / topic slots
  flow into the recurrent core).
- ``ablation``: ``use_derived_text_features=False`` -- the document-
  level encoder text path is the only text-derived signal.
- ``replacement``: ``use_derived_text_features=False`` + pre-meeting
  rates columns from #291. The replacement arm only runs when the
  rates_panel.parquet artefact is available; otherwise the runner
  skips it and documents the skip in the output JSON.

The output JSON is keyed by configuration with per-fold macro-F1 +
bootstrap CI numbers so the §16 finalization-roadmap table can read
the comparison off a single file.

Usage::

    docker compose run --rm backend python -m scripts.run_derived_features_ablation \\
        --training-package-id <id> \\
        --output artifacts/experiments/derived_features_ablation.json \\
        --seeds 11 29 47 71 97 \\
        --bootstrap-samples 500
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from app.config import BACKEND_ROOT


# The replacement arm pulls in #291 pre-meeting columns. When the
# rates_panel.parquet artefact is missing (the canonical
# pre-#312 corpus), the runner skips the replacement arm and documents
# the skip on the output payload so the table can show "n/a -- #291 not
# materialised".
_REPLACEMENT_ARM_DATA = (
    BACKEND_ROOT.parent / "data" / "processed" / "rates_panel.parquet"
)


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
            "``artifacts/experiments/derived_features_ablation.json``."
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
        "--hidden-size",
        type=int,
        default=64,
        help="Hidden size shared across all three configurations.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=500,
        help="Block-bootstrap iterations for the CI columns.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=11,
        help="Seed for the bootstrap RNG so the CI numbers reproduce.",
    )
    return parser.parse_args()


def _resolve_output_path(arg: Path | None) -> Path:
    if arg is not None:
        return arg
    base = BACKEND_ROOT.parent / "artifacts" / "experiments"
    base.mkdir(parents=True, exist_ok=True)
    return base / "derived_features_ablation.json"


def _trial_metrics(summary: Any) -> dict[str, float | None]:
    test = getattr(summary, "test_metrics", None) or getattr(summary, "metrics", None)
    if test is None:
        return {}
    return {
        "regime_f1_macro": getattr(test, "regime_f1_macro", None),
        "regime_accuracy": getattr(test, "regime_accuracy", None),
        "regime_loss": getattr(test, "regime_loss", None),
    }


def _bootstrap_ci(
    values: list[float],
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float] | None:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return None
    if len(finite) < 2 or samples <= 0:
        return {
            "mean": statistics.fmean(finite),
            "lo": min(finite),
            "hi": max(finite),
            "n": len(finite),
        }
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        resampled = [rng.choice(finite) for _ in finite]
        means.append(statistics.fmean(resampled))
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = max(0, int(math.floor(alpha * len(means))))
    hi_idx = min(len(means) - 1, int(math.ceil((1.0 - alpha) * len(means))) - 1)
    return {
        "mean": statistics.fmean(finite),
        "lo": means[lo_idx],
        "hi": means[hi_idx],
        "n": len(finite),
    }


def _run_one_cell(
    configuration: str,
    seed: int,
    *,
    training_package_id: str,
    epochs: int,
    hidden_size: int,
    use_derived: bool,
) -> dict[str, Any]:
    from app.models.config import ModelConfig
    from app.training.loaders import load_walk_forward_split
    from app.training.loop import train_model

    splits = load_walk_forward_split(
        training_package_id=training_package_id,
        rich_features=True,
    )
    config = ModelConfig(
        output_mode="classification",
        n_classes=3,
        hidden_size=hidden_size,
        use_derived_text_features=use_derived,
    )

    per_fold: list[dict[str, Any]] = []
    for split in splits:
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
        per_fold.append(
            {
                "fold_id": split.fold_id,
                "metrics": _trial_metrics(result.summary),
            }
        )

    return {
        "configuration": configuration,
        "seed": seed,
        "use_derived_text_features": use_derived,
        "training_package_id": training_package_id,
        "folds": per_fold,
    }


def _configurations() -> list[tuple[str, dict[str, Any]]]:
    """Return the three configuration definitions in fixed order.

    Each entry is a ``(name, kwargs)`` pair the runner forwards to
    :func:`_run_one_cell`. The replacement arm carries a ``"requires"``
    marker so the runner can decide to skip it when the dependency is
    missing.
    """

    return [
        ("baseline", {"use_derived": True}),
        ("ablation", {"use_derived": False}),
        (
            "replacement",
            {
                "use_derived": False,
                "requires": str(_REPLACEMENT_ARM_DATA),
            },
        ),
    ]


def main() -> int:
    args = _parse_args()
    output_path = _resolve_output_path(args.output)
    print(f"[derived_features_ablation] writing -> {output_path}")

    trials: dict[str, list[dict[str, Any]]] = {}
    skipped: dict[str, str] = {}
    for name, kwargs in _configurations():
        required = kwargs.pop("requires", None)
        if required is not None and not Path(required).exists():
            skipped[name] = (
                f"required artefact {required} is not on disk; "
                "the replacement arm needs the #291 pre-meeting "
                "rates columns. Re-run after #291 lands."
            )
            print(
                f"[derived_features_ablation] SKIP {name}: {skipped[name]}",
                flush=True,
            )
            continue
        trials[name] = []
        for seed in args.seeds:
            print(
                f"[derived_features_ablation] {name} seed={seed} "
                f"epochs={args.epochs}",
                flush=True,
            )
            trials[name].append(
                _run_one_cell(
                    name,
                    seed,
                    training_package_id=args.training_package_id,
                    epochs=args.epochs,
                    hidden_size=args.hidden_size,
                    **kwargs,
                )
            )

    summary: dict[str, Any] = {}
    for name, trial_list in trials.items():
        per_fold_f1: list[float] = []
        for trial in trial_list:
            for fold in trial["folds"]:
                metrics = fold.get("metrics", {}) or {}
                f1 = metrics.get("regime_f1_macro")
                if f1 is not None:
                    per_fold_f1.append(float(f1))
        summary[name] = _bootstrap_ci(
            per_fold_f1,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )

    payload = {
        "configurations": list(trials.keys()),
        "skipped": skipped,
        "seeds": list(args.seeds),
        "epochs": args.epochs,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "training_package_id": args.training_package_id,
        "trials": trials,
        "summary": summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[derived_features_ablation] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
