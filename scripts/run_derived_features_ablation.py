"""Three-way derived-text-features ablation runner (#309).

Trains the forecaster under three configurations on the same
walk-forward fold protocol:

- ``baseline``: every derived-text slot stays on -- the pre-#309
  baseline (per-sentence sentiment / stance / certainty / topic slots
  flow into the recurrent core).
- ``derived_ablation`` (alias ``ablation``): the five derived-feature
  columns surfaced in the §16 walkthrough (``sentiment_score``,
  ``stance_label``, ``factor_labels``, ``certainty_score``,
  ``topic_label``) are zeroed in place on every FeatureVector before
  the per-fold scaler fit. The document-level encoder text path is
  the only text-derived signal that survives.
- ``derived_replacement`` (alias ``replacement``): same narrow
  five-column zero, plus the #291 pre-meeting rates columns once
  the loader carries them. The arm runs only when
  ``data/external/fred/rates_panel.parquet`` is on disk; otherwise it
  reports ``skipped`` in the output JSON. Wiring the 12 pre-meeting
  columns into the FeatureVector input slots that the ablation arm
  zeros is tracked under #315; until that lands the arm emits the
  same input tensor as ``derived_ablation`` and surfaces the deferral
  on ``replacement_arm`` in the manifest.

The narrow five-column zeroing is the methodology contract this
runner ships. The broader text-family zero (linguistic / mp_surprise
/ multi-axis / LLM-features blocks together) is owned by the
per-family ablation (#334). This ablation isolates the question the
issue body asks: do the five derived-feature columns specifically
carry forecaster-relevant signal over the document-level encoder
path. See ADR 0039 for the methodology framing.

The output JSON is keyed by configuration with per-fold macro-F1 +
bootstrap CI numbers so the §16 finalization-roadmap table can read
the comparison off a single file.

Usage::

    docker compose run --rm backend python -m scripts.run_derived_features_ablation \\
        --training-package-id <id> \\
        --arm derived_ablation \\
        --output artifacts/experiments/derived_features_ablation.json \\
        --seeds 11 29 47 71 97 \\
        --bootstrap-samples 500

Omit ``--arm`` to sweep all three arms in one invocation (the legacy
shape the PR #314 runner shipped). Pass ``--arm`` to run a single
cell (matches the per-family / L-M runners and lets a Runpod queue
schedule one arm per job).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable

from app.config import BACKEND_ROOT


# The five derived-feature columns the §16 walkthrough surfaces and
# the issue body names verbatim. Mapped to the FeatureVector attribute
# slots that ``as_rich_list`` writes into the per-bar tensor:
#
# - ``sentiment_score``  : per-bar position [0]
# - ``stance_label``     : stance one-hot at positions [29:32]
#                          (stance_hawk / stance_dove / stance_neutral)
# - ``factor_labels``    : multi-task ``factor`` axis -- carried via
#                          ``mt_aux`` rather than a per-bar slot. The
#                          narrow ablation masks the auxiliary loss
#                          contribution by collapsing ``factor_mask``
#                          to False so the head no longer reads the
#                          axis.
# - ``certainty_score``  : per-bar position [33] (certain_label_certain)
# - ``topic_label``      : multi-task ``topic`` axis (mt_aux only;
#                          same mask-to-False treatment as factor).
#
# ``stance_missing`` (slot [34]) is intentionally NOT zeroed: it's a
# missingness flag that tells the head "no stance signal available",
# which is precisely what the ablation establishes. Flipping it would
# leak a "stance unknown" signal that the baseline arm does not see
# either (baseline already keeps stance_missing at whatever the
# upstream loader set it to).
_FIVE_DERIVED_COLUMNS: tuple[str, ...] = (
    "sentiment_score",
    "stance_label",
    "factor_labels",
    "certainty_score",
    "topic_label",
)

# The FeatureVector attributes the narrow zeroer overwrites. Kept
# separate from ``_FIVE_DERIVED_COLUMNS`` (the conceptual names the
# issue body uses) so the conceptual list reads cleanly in the
# manifest and the attribute list stays the operational contract.
_FIVE_DERIVED_FV_ATTRS: tuple[str, ...] = (
    "sentiment_score",
    "stance_hawk",
    "stance_dove",
    "stance_neutral",
    "certain_label_certain",
)

# The multi-task aux axes the narrow zeroer masks off. The masks
# drop to all-False so the auxiliary loss contribution from each axis
# vanishes; the target tensors themselves are untouched.
_FIVE_DERIVED_MT_AUX_AXES: tuple[str, ...] = ("factor", "topic")


# The replacement arm pulls in #291 pre-meeting columns from
# ``data/external/fred/rates_panel.parquet`` (the canonical post-#312
# location). When the parquet is missing the runner skips the arm and
# documents the skip on the output payload so the table can show
# "n/a -- #291 not materialised".
_REPLACEMENT_ARM_DATA = (
    BACKEND_ROOT.parent / "data" / "external" / "fred" / "rates_panel.parquet"
)
# The pre-meeting wiring needed to actually inject the 12 rates columns
# into the FeatureVector slots that ``_zero_derived_text_features``
# clears is tracked separately; the runner reports the deferral on the
# manifest so downstream readers know which arm ran.
_REPLACEMENT_ARM_DEFERRAL_TICKET = (
    "deferred: pre-meeting wiring tracked in #315"
)


# Arm vocabulary. ``baseline`` is the canonical pipeline byte-identical
# to the pre-#309 head. ``derived_ablation`` flips the narrow
# five-column zero on every FeatureVector before the per-fold scaler
# fit. ``derived_replacement`` does the same zero and is the slot
# the #315 pre-meeting columns will fill once that issue lands. The
# legacy arm names (``ablation`` / ``replacement``) stay as aliases so
# the PR #314 CLI surface keeps working.
_ARM_CHOICES: tuple[str, ...] = (
    "baseline",
    "derived_ablation",
    "derived_replacement",
)
_ARM_ALIASES: dict[str, str] = {
    "ablation": "derived_ablation",
    "replacement": "derived_replacement",
}


def _canonicalise_arm(name: str) -> str:
    """Resolve a CLI arm string to one of ``_ARM_CHOICES``.

    The PR #314 runner shipped with ``ablation`` / ``replacement`` arm
    names; #309's methodology rewrite renames them so the §16 table
    can read the columns as ``derived_ablation`` / ``derived_replacement``
    against the per-family runner's ``zero_<family>`` vocabulary.
    Legacy strings resolve to the new names without breaking callers.
    """

    return _ARM_ALIASES.get(name, name)


def _zero_five_derived_columns_inplace(
    sequences: Iterable[list[Any]],
) -> None:
    """Zero the five derived-feature columns on every per-bar FeatureVector.

    Mirrors the per-family runner's ``_zero_per_bar_market_aux``
    pattern: the loader has no flag for the narrow five-column slice
    (the existing ``use_multi_axis`` / ``use_mp_surprise`` flags zero
    families much wider than the §16 question scopes), so the runner
    walks the loaded sequences and overwrites the slots in place
    BEFORE ``train_model`` fits the per-fold RobustScaler. That
    ordering is the load-bearing piece -- the scaler sees the zero
    column on the train slice and locks the median + IQR at the
    no-signal state, then applies the same transform to val + test so
    the post-scale value stays a literal 0 across the partition.

    The multi-task ``factor`` and ``topic`` aux axes have no per-bar
    slot in ``as_rich_list``; they ride the auxiliary loss path. The
    matching mask collapse runs inside ``train_model`` and is enabled
    by ``ModelConfig.use_derived_text_features=False``; the runner
    flips that flag on both ablation arms so the aux contribution
    drops out alongside the per-bar zeroing.
    """

    for sequence in sequences:
        for fv in sequence:
            for attr in _FIVE_DERIVED_FV_ATTRS:
                if hasattr(fv, attr):
                    setattr(fv, attr, 0.0)


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
    parser.add_argument(
        "--replacement-arm-status",
        type=str,
        default=_REPLACEMENT_ARM_DEFERRAL_TICKET,
        help=(
            "Status string emitted on the manifest when the replacement "
            "arm runs without wiring the pre-meeting columns. Set to "
            "``ready`` once the FeatureVector loader carries the 12 "
            "pre-meeting columns from #291 into the input slots the "
            "narrow five-column zero clears."
        ),
    )
    parser.add_argument(
        "--arm",
        type=str,
        choices=list(_ARM_CHOICES) + list(_ARM_ALIASES.keys()),
        default=None,
        help=(
            "Run a single arm and exit. ``baseline`` keeps the canonical "
            "pipeline, ``derived_ablation`` zeros the five derived-feature "
            "columns in place before the per-fold scaler fit, and "
            "``derived_replacement`` is the #315-blocked slot the "
            "pre-meeting rates columns will fill. Legacy aliases "
            "``ablation`` / ``replacement`` resolve to the canonical names. "
            "Omit to sweep all three arms in one invocation (the PR #314 "
            "default shape)."
        ),
    )
    return parser.parse_args()


def _resolve_output_path(arg: Path | None) -> Path:
    if arg is not None:
        return arg
    base = BACKEND_ROOT.parent / "artifacts" / "experiments"
    base.mkdir(parents=True, exist_ok=True)
    return base / "derived_features_ablation.json"


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
    mean = statistics.fmean(finite)
    # ``std`` is the across-observation sample std on the finite
    # measurements (5 seeds x N folds). The §16 narrative reads as
    # ``mean ± std`` per arm; the bootstrap (lo / hi) numbers carry
    # the CI separately for callers that want the resampled spread.
    std = statistics.stdev(finite) if len(finite) >= 2 else 0.0
    if len(finite) < 2 or samples <= 0:
        return {
            "mean": mean,
            "std": std,
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
        "mean": mean,
        "std": std,
        "lo": means[lo_idx],
        "hi": means[hi_idx],
        "n": len(finite),
    }


def _run_one_cell(
    configuration: str,
    seed: int,
    *,
    training_package_id: str,
    fold_ids: list[str],
    epochs: int,
    hidden_size: int,
    use_derived: bool,
) -> dict[str, Any]:
    from app.models.config import ModelConfig
    from app.training.loaders import load_walk_forward_split
    from app.training.loop import train_model

    config = ModelConfig(
        output_mode="classification",
        n_classes=3,
        hidden_size=hidden_size,
        use_derived_text_features=use_derived,
    )

    per_fold: list[dict[str, Any]] = []
    for fold_id in fold_ids:
        split = load_walk_forward_split(
            training_package_id=training_package_id,
            fold_id=fold_id,
            rich_features=True,
        )
        # Narrow five-column zeroing on every FeatureVector before the
        # per-fold scaler fit. The ``use_derived_text_features=False``
        # config flag downstream collapses the multi-task factor /
        # topic aux masks inside ``train_model``; this pass clears the
        # per-bar sentiment + stance one-hot + certainty slots the
        # ``as_rich_list`` writer would otherwise hand the scaler.
        # Baseline (``use_derived=True``) skips both -- the per-fold
        # tensor that lands on ``train_model`` is byte-identical to
        # the canonical pipeline.
        if not use_derived:
            _zero_five_derived_columns_inplace(split.train)
            _zero_five_derived_columns_inplace(split.val)
            _zero_five_derived_columns_inplace(split.test)
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
        ("derived_ablation", {"use_derived": False}),
        (
            "derived_replacement",
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

    fold_ids = _resolve_fold_ids(args.training_package_id, args.folds)
    print(f"[derived_features_ablation] folds={fold_ids}")

    selected_arm = _canonicalise_arm(args.arm) if args.arm is not None else None
    if selected_arm is not None:
        print(f"[derived_features_ablation] arm={selected_arm}")

    trials: dict[str, list[dict[str, Any]]] = {}
    skipped: dict[str, str] = {}
    replacement_arm_status: str | None = None
    for name, kwargs in _configurations():
        if selected_arm is not None and name != selected_arm:
            continue
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
        if name == "derived_replacement":
            # The arm is gated on the rates_panel.parquet being present.
            # Wiring the 12 pre-meeting columns into the FeatureVector
            # slots that the ablation zeros is tracked separately; the
            # status string surfaces the deferral on the manifest so
            # downstream readers can tell which arm code path ran.
            replacement_arm_status = str(args.replacement_arm_status)
            print(
                f"[derived_features_ablation] replacement arm status: "
                f"{replacement_arm_status}",
                flush=True,
            )
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
                    fold_ids=fold_ids,
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
        "arm_choices": list(_ARM_CHOICES),
        "arm_aliases": dict(_ARM_ALIASES),
        "arm": selected_arm,
        "five_derived_columns": list(_FIVE_DERIVED_COLUMNS),
        "five_derived_fv_attrs": list(_FIVE_DERIVED_FV_ATTRS),
        "five_derived_mt_aux_axes": list(_FIVE_DERIVED_MT_AUX_AXES),
        "skipped": skipped,
        "seeds": list(args.seeds),
        "fold_ids": fold_ids,
        "epochs": args.epochs,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "training_package_id": args.training_package_id,
        "trials": trials,
        "summary": summary,
    }
    if replacement_arm_status is not None:
        payload["replacement_arm"] = replacement_arm_status
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[derived_features_ablation] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
