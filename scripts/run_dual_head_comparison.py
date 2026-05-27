"""Three-way dual-head comparison runner (#304).

Trains the same model under three head-mode configurations on the same
walk-forward fold protocol and emits a per-trial JSON keyed by
configuration so the §16 finalization-roadmap table can read the
results off a single file. Each configuration runs over the official
seed set and writes the per-fold ``regime_f1_macro`` (classification
surface) and ``regression_rmse_log_rv`` (regression surface) so the
table can compare both axes at a glance.

Usage::

    docker compose run --rm backend python -m scripts.run_dual_head_comparison \\
        --training-package-id <id> \\
        --output artifacts/experiments/dual_head_comparison.json \\
        --seeds 11 29 47 71 97 \\
        --epochs 40 \\
        --regression-alpha 0.5

The output JSON has the structure::

    {
      "head_modes": ["classification", "regression", "dual"],
      "seeds": [...],
      "trials": {
        "classification": [ { "seed": 11, "metrics": {...}, ... }, ... ],
        "regression":     [ ... ],
        "dual":           [ ... ]
      },
      "summary": {
        "classification": { "regime_f1_macro_mean": float, "regime_f1_macro_std": float, ... },
        ...
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
from app.training.runtime_compat import ensure_compile_safe


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
            "``artifacts/experiments/dual_head_comparison.json``."
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
        "--regression-alpha",
        type=float,
        default=0.5,
        help="alpha for head_mode='dual' joint loss.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Hidden size shared across all three head modes.",
    )
    parser.add_argument(
        "--head-modes",
        nargs="+",
        choices=("classification", "regression", "dual"),
        default=["classification", "regression", "dual"],
        help="Subset of head modes to evaluate (defaults to all three).",
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
    # #305 rates-head target derivation. Mirrors the same flag on
    # ``app.train_forecaster`` (name + choices + default). ``raw`` keeps
    # this runner byte-identical to the pre-#305 canonical sweep.
    parser.add_argument(
        "--rates-target-mode",
        type=str,
        choices=("raw", "fomc_attributable"),
        default="raw",
        help=(
            "Rates-head target derivation. ``raw`` (default) keeps the "
            "observed ``yield_<tenor>_change_5d`` bps move; "
            "``fomc_attributable`` predicts the 1-D projection onto the "
            "strict-prior policy-surprise direction. See ADR 0027."
        ),
    )
    # #306 retrieval-augmented input features. Off by default so the
    # canonical sweep stays byte-identical.
    parser.add_argument(
        "--use-retrieval-analogs",
        dest="use_retrieval_analogs",
        action="store_true",
        help=(
            "Attach the 5-dim retrieval-analog summary block to every "
            "supervised event. Default off."
        ),
    )
    parser.add_argument(
        "--no-retrieval-analogs",
        dest="use_retrieval_analogs",
        action="store_false",
        help="Disable the retrieval-analog block (default).",
    )
    # #307 macro-regime conditioning. Off by default so the canonical
    # sweep stays byte-identical.
    parser.add_argument(
        "--use-regime-conditioning",
        dest="use_regime_conditioning",
        action="store_true",
        help=(
            "Attach the 3-scalar macro-regime indicator block and mount "
            "the multiplicative gate over the rich-feature slice. "
            "Default off."
        ),
    )
    parser.add_argument(
        "--no-regime-conditioning",
        dest="use_regime_conditioning",
        action="store_false",
        help="Disable the macro-regime block + gate (default).",
    )
    parser.set_defaults(
        use_retrieval_analogs=False,
        use_regime_conditioning=False,
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
            f"for training_package_id={training_package_id!r}; provide "
            "--folds explicitly."
        )
    return sorted(manifest.keys())


def _resolve_output_path(arg: Path | None) -> Path:
    if arg is not None:
        return arg
    base = BACKEND_ROOT.parent / "artifacts" / "experiments"
    base.mkdir(parents=True, exist_ok=True)
    return base / "dual_head_comparison.json"


def _trial_metrics(summary: Any) -> dict[str, float | None]:
    """Pull the headline numbers out of a TrainingRunSummary."""

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


def _resolve_auto_rates_heads(
    rates_target_mode: str,
    rates_heads: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Auto-activate the canonical rates-head set under FOMC-attributable.

    ``--rates-target-mode`` only steers the rates heads' supervised
    target; it has no observable effect unless at least one rates head
    is mounted. The canonical-comparison sweep does not expose a
    ``--rates-heads`` flag (it sweeps head_modes, not rates heads), so
    when the operator opts the runner into ``fomc_attributable`` we
    mount the same canonical set (``2y``, ``5y``, ``terminal``) the
    ``make rates-heads-sweep`` target uses. ``raw`` (default) keeps
    ``rates_heads=()`` so the pre-#401 canonical sweep stays
    byte-identical.
    """

    from app.models.rates_heads import RATES_HEAD_NAMES

    if rates_heads:
        return tuple(rates_heads)
    if rates_target_mode != "raw":
        return tuple(RATES_HEAD_NAMES)
    return ()


def _run_one_cell(
    head_mode: str,
    seed: int,
    *,
    training_package_id: str,
    fold_ids: list[str],
    epochs: int,
    regression_alpha: float,
    hidden_size: int,
    rates_target_mode: str = "raw",
    use_retrieval_analogs: bool = False,
    use_regime_conditioning: bool = False,
) -> dict[str, Any]:
    # Imports happen here so the script is importable without a torch
    # install (useful for doc-only environments).
    from app.models.config import RICH_FEATURE_SIZE, ModelConfig
    from app.training.loaders import load_walk_forward_split
    from app.training.loop import train_model

    rates_heads = _resolve_auto_rates_heads(rates_target_mode, rates_heads=None)
    config = ModelConfig(
        input_size=RICH_FEATURE_SIZE,
        output_mode="classification",
        head_mode=head_mode,
        regression_alpha=regression_alpha,
        n_classes=3,
        hidden_size=hidden_size,
        rates_heads=rates_heads,
        rates_target_mode=rates_target_mode,
        use_regime_conditioning=use_regime_conditioning,
    )

    per_fold: list[dict[str, Any]] = []
    for fold_id in fold_ids:
        split = load_walk_forward_split(
            training_package_id=training_package_id,
            fold_id=fold_id,
            rich_features=True,
            use_retrieval_analogs=use_retrieval_analogs,
            use_regime_conditioning=use_regime_conditioning,
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
        per_fold.append(
            {
                "fold_id": split.fold_id,
                "metrics": _trial_metrics(result.summary),
            }
        )

    return {
        "head_mode": head_mode,
        "seed": seed,
        "regression_alpha": regression_alpha,
        "training_package_id": training_package_id,
        "folds": per_fold,
    }


def main() -> int:
    ensure_compile_safe()
    args = _parse_args()
    output_path = _resolve_output_path(args.output)
    print(f"[dual_head_comparison] writing -> {output_path}")

    fold_ids = _resolve_fold_ids(args.training_package_id, args.folds)
    print(f"[dual_head_comparison] folds={fold_ids}")
    if str(args.rates_target_mode) != "raw":
        print(
            "[dual_head_comparison] auto-activating rates heads for "
            f"rates_target_mode={args.rates_target_mode}"
        )

    trials: dict[str, list[dict[str, Any]]] = {mode: [] for mode in args.head_modes}
    for head_mode in args.head_modes:
        for seed in args.seeds:
            print(
                f"[dual_head_comparison] head_mode={head_mode} seed={seed} "
                f"epochs={args.epochs}",
                flush=True,
            )
            trials[head_mode].append(
                _run_one_cell(
                    head_mode,
                    seed,
                    training_package_id=args.training_package_id,
                    fold_ids=fold_ids,
                    epochs=args.epochs,
                    regression_alpha=args.regression_alpha,
                    hidden_size=args.hidden_size,
                    rates_target_mode=str(args.rates_target_mode),
                    use_retrieval_analogs=bool(args.use_retrieval_analogs),
                    use_regime_conditioning=bool(args.use_regime_conditioning),
                )
            )

    summary: dict[str, Any] = {}
    for head_mode, trial_list in trials.items():
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
        summary[head_mode] = {
            "regime_f1_macro": _summary_stats(per_fold_f1),
            "regression_rmse_log_rv": _summary_stats(per_fold_rmse),
        }

    payload = {
        "head_modes": args.head_modes,
        "seeds": list(args.seeds),
        "fold_ids": fold_ids,
        "epochs": args.epochs,
        "regression_alpha": args.regression_alpha,
        "training_package_id": args.training_package_id,
        "rates_target_mode": str(args.rates_target_mode),
        "rates_heads": list(
            _resolve_auto_rates_heads(str(args.rates_target_mode), rates_heads=None)
        ),
        "use_retrieval_analogs": bool(args.use_retrieval_analogs),
        "use_regime_conditioning": bool(args.use_regime_conditioning),
        "trials": trials,
        "summary": summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[dual_head_comparison] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
