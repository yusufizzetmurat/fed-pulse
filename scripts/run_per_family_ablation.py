"""Per-family rich-feature ablation runner (#334).

Trains the forecaster under the canonical training package + walk-forward
fold protocol while zeroing one rich-feature family at a time (and a
cumulative chain that drops families in dominance order). The output JSON
backs the §6 per-family ablation table and the substitution-finding
narrative in the wiki.

Seven rich-feature families are ablated:

- ``linguistic``       -- 15-dim per-document linguistic block (joined
                          on ``text_hash``)
- ``credibility``      -- 4-vector sourced from the event row
- ``mp_surprise``      -- 4-vector joined on ``event_date``
- ``multi_axis``       -- 6-dim stance / time / certainty block
- ``realised_vol``     -- per-bar 20d + 60d realised vol horizons
- ``cross_asset``      -- per-bar VIX/DXY/TNX/Gold + VIX3M/IRX +
                          VIX term-slope + 10y-3m curve slope (8-dim)
- ``llm_features``     -- 35-dim B1 catalogue one-hot block + missing
                          flag (off by default; the runner switches it
                          on for the baseline so the ``zero_llm`` cell
                          reads as a real ablation)

Each ablation zeros the family **before** the per-fold scaler fit in
``train_model``. The mechanism follows the #309 contract:

- The first five families (linguistic / credibility / mp_surprise /
  multi_axis / llm_features) are zeroed via the existing loader flags
  on :func:`load_walk_forward_split` -- those flags already write
  literal 0.0 into the FeatureVector slice before the scaler sees the
  payload.
- The realised-vol and cross-asset families have no loader flag because
  PR #207 / #208 fanned them out as direct FeatureVector attributes.
  The runner walks the loaded sequences in-place and zeros those
  attributes before passing the split to ``train_model``; the same
  pre-scaler ordering is preserved.

Output JSON shape (``backend/artifacts/experiments/per_family_ablation.json``)::

    {
      "families": [...],
      "cells": ["baseline", "zero_linguistic", ..., "cumulative_drop_*"],
      "seeds": [...],
      "fold_ids": [...],
      "training_package_id": "...",
      "epochs": 40,
      "regression_alpha": 0.5,
      "head_mode": "dual",
      "trials": {
        "<cell>": [ {"seed": ..., "folds": [{"fold_id": ..., "metrics": {...}}]}, ... ]
      },
      "summary": {
        "<cell>": {
          "regime_f1_macro": {"mean": ..., "lo": ..., "hi": ..., "n": ...} | None,
          "regression_rmse_log_rv": {...} | None
        }
      },
      "post_350_note": "..."     # caption marker
    }

Usage::

    docker compose --profile gpu run --rm backend-gpu \\
        python -m scripts.run_per_family_ablation \\
        --training-package-id <id> \\
        --output artifacts/experiments/per_family_ablation.json \\
        --seeds 11 29 47 71 97 \\
        --epochs 40 \\
        --head-mode dual \\
        --regression-alpha 0.5

The runner is re-runnable; the output path is overwritten on each call.
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
from app.training.runtime_compat import ensure_compile_safe


# Canonical family order. The per-family cell labels follow
# ``zero_<family>`` so the JSON keys sort stably. The cumulative chain
# drops families in dominance order -- text first (smallest lift in the
# §6.3 tier table), then the two market families that landed in the A2 /
# A3 chunks, finishing on the credibility 4-vector that joins straight
# off the event row. ``zero_cumulative_text`` is read as "drop the text
# block entirely"; ``zero_cumulative_text_market_aux`` drops text + the
# auxiliary market families; ``zero_cumulative_text_market_aux_cred``
# drops everything except the legacy 6-feature market path. The chain
# is intentionally short -- the table only needs three cumulative cells
# to anchor the "rich market dominates" claim against the per-family
# columns.
_FAMILIES: tuple[str, ...] = (
    "linguistic",
    "credibility",
    "mp_surprise",
    "multi_axis",
    "realised_vol",
    "cross_asset",
    "llm_features",
)


_CUMULATIVE_CHAIN: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "cumulative_drop_text",
        ("linguistic", "multi_axis", "llm_features"),
    ),
    (
        "cumulative_drop_text_market_aux",
        (
            "linguistic",
            "multi_axis",
            "llm_features",
            "realised_vol",
            "cross_asset",
        ),
    ),
    (
        "cumulative_drop_text_market_aux_cred_mp",
        (
            "linguistic",
            "multi_axis",
            "llm_features",
            "realised_vol",
            "cross_asset",
            "credibility",
            "mp_surprise",
        ),
    ),
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
            "``artifacts/experiments/per_family_ablation.json``."
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
        help="Hidden size shared across every cell.",
    )
    parser.add_argument(
        "--head-mode",
        choices=("classification", "regression", "dual"),
        default="dual",
        help=(
            "Head mode for every cell. Defaults to ``dual`` so the "
            "table reads off the post-#322 canonical objective."
        ),
    )
    parser.add_argument(
        "--regression-alpha",
        type=float,
        default=0.5,
        help="alpha for head_mode='dual' joint loss.",
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
            "every fold in the package's "
            "fold_manifest_expanding_walk_forward.json."
        ),
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        default=None,
        help=(
            "Subset of cell labels to run (defaults to baseline + "
            "every per-family cell + the cumulative chain). Useful for "
            "resuming a partial sweep."
        ),
    )
    parser.add_argument(
        "--post-350-status",
        type=str,
        default="pre-#350",
        help=(
            "Caption-level marker recording whether the run was made "
            "against the pre-#350 or post-#350 feature surface. Stored "
            "verbatim on the output JSON so the wiki table can cite the "
            "correct vintage."
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
    # #305 rates-head target derivation. Mirrors the flag on
    # ``app.train_forecaster``; default ``raw`` keeps the per-family
    # ablation byte-identical to the pre-#305 path.
    parser.add_argument(
        "--rates-target-mode",
        type=str,
        choices=("raw", "fomc_attributable"),
        default="raw",
        help=(
            "Rates-head target derivation. ``raw`` (default) keeps the "
            "observed yield change in bps; ``fomc_attributable`` "
            "projects onto the strict-prior policy-surprise direction. "
            "See ADR 0027."
        ),
    )
    # #306 retrieval-augmented input features. The per-family ablation
    # zeros document-level rich-feature families directly on the loaded
    # FeatureVector slices; the retrieval-analog block lives in a
    # separate per-bar slot, so the families and the analog block are
    # orthogonal -- a cell that zeros ``linguistic`` does NOT zero the
    # analog summary scalars, and ``zero_llm`` likewise leaves the
    # analog block intact. Enabling the flag widens the per-bar feature
    # surface for every cell in the sweep.
    parser.add_argument(
        "--use-retrieval-analogs",
        dest="use_retrieval_analogs",
        action="store_true",
        help=(
            "Attach the 5-dim retrieval-analog summary block. Default "
            "off; the block is orthogonal to per-family zeroing."
        ),
    )
    parser.add_argument(
        "--no-retrieval-analogs",
        dest="use_retrieval_analogs",
        action="store_false",
        help="Disable the retrieval-analog block (default).",
    )
    # #307 macro-regime conditioning. Off by default so the per-family
    # ablation stays byte-identical to the pre-#307 path.
    parser.add_argument(
        "--use-regime-conditioning",
        dest="use_regime_conditioning",
        action="store_true",
        help=(
            "Attach the 3-scalar macro-regime block and mount the "
            "multiplicative gate over the rich-feature slice. Default off."
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


def _resolve_output_path(arg: Path | None) -> Path:
    if arg is not None:
        return arg
    base = BACKEND_ROOT.parent / "artifacts" / "experiments"
    base.mkdir(parents=True, exist_ok=True)
    return base / "per_family_ablation.json"


def _resolved_rates_heads_for_payload(rates_target_mode: str) -> list[str]:
    """Surface the auto-activated rates-head set in the output JSON.

    Mirrors the auto-activation policy inside ``_run_one_cell``: when
    ``--rates-target-mode != raw`` we mount the canonical rates head set
    so the per-family ablation actually exercises the flag. The payload
    records the resolved tuple so readers can confirm the run was
    distinct from the ``raw`` baseline.
    """

    from app.models.rates_heads import RATES_HEAD_NAMES

    if rates_target_mode != "raw":
        return list(RATES_HEAD_NAMES)
    return []


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


def _zero_per_bar_market_aux(
    sequences: list[list[Any]],
    *,
    zero_realised_vol: bool,
    zero_cross_asset: bool,
) -> None:
    """Zero per-bar realised-vol / cross-asset FeatureVector attrs in place.

    The loader has no flag for these families (they were fanned out as
    direct ``FeatureVector`` attributes by PRs #207 / #208), so the
    runner walks the loaded sequences and overwrites the slices before
    the per-fold scaler in ``train_model`` sees them. Mirrors the #309
    pre-scaler-fit pattern.
    """

    if not (zero_realised_vol or zero_cross_asset):
        return
    for sequence in sequences:
        for fv in sequence:
            if zero_realised_vol:
                fv.realized_vol_20d = 0.0
                fv.realized_vol_60d = 0.0
            if zero_cross_asset:
                fv.vix_close = 0.0
                fv.dxy_close = 0.0
                fv.tnx_close = 0.0
                fv.gold_close = 0.0
                fv.vix3m_close = 0.0
                fv.irx_close = 0.0
                fv.vix_term_slope = 0.0
                fv.yield_curve_slope_10y_3m = 0.0


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


def _cell_flags(zero_set: frozenset[str]) -> dict[str, bool]:
    """Translate a set of families-to-zero into loader + post-load flags.

    Returns the kwargs for :func:`load_walk_forward_split` and the two
    booleans that drive the in-place per-bar zeroing pass. The loader
    accepts ``use_*`` flags for the document-level families;
    ``realised_vol`` / ``cross_asset`` ride the post-load walker.
    """

    return {
        "loader": {
            "use_credibility": "credibility" not in zero_set,
            "use_linguistic": "linguistic" not in zero_set,
            "use_mp_surprise": "mp_surprise" not in zero_set,
            "use_multi_axis": "multi_axis" not in zero_set,
            # Baseline turns LLM features on so the zero_llm cell reads
            # as a real ablation rather than a no-op (the default loader
            # path emits llm_features=None on every cell).
            "use_llm_features": "llm_features" not in zero_set,
        },
        "zero_realised_vol": "realised_vol" in zero_set,
        "zero_cross_asset": "cross_asset" in zero_set,
    }


def _resolve_cells(
    override: list[str] | None,
) -> list[tuple[str, frozenset[str]]]:
    """Build the canonical cell list.

    Order: baseline -> per-family individual zeros (in ``_FAMILIES``
    order) -> cumulative chain (in ``_CUMULATIVE_CHAIN`` order).
    """

    cells: list[tuple[str, frozenset[str]]] = [("baseline", frozenset())]
    for family in _FAMILIES:
        cells.append((f"zero_{family}", frozenset({family})))
    for label, families in _CUMULATIVE_CHAIN:
        cells.append((label, frozenset(families)))
    if override:
        allowed = set(override)
        known_labels = [c[0] for c in cells]
        cells = [c for c in cells if c[0] in allowed]
        if not cells:
            raise ValueError(
                f"--cells={override!r} did not match any of the "
                f"known labels: {known_labels}"
            )
    return cells


def _run_one_cell(
    cell_label: str,
    zero_set: frozenset[str],
    seed: int,
    args: argparse.Namespace,
    *,
    fold_ids: list[str],
) -> dict[str, Any]:
    """Train + evaluate one (cell, seed) cell across every fold."""

    from app.models.config import ModelConfig, RICH_FEATURE_SIZE
    from app.models.rates_heads import RATES_HEAD_NAMES
    from app.training.loaders import load_walk_forward_split
    from app.training.loop import train_model

    flags = _cell_flags(zero_set)

    rates_target_mode = str(getattr(args, "rates_target_mode", "raw"))
    # #401 follow-up: ``--rates-target-mode`` is a no-op unless at least
    # one rates head is mounted. The per-family ablation runner does not
    # expose a ``--rates-heads`` flag (it sweeps rich-feature families,
    # not rates heads), so we auto-activate the canonical set when the
    # operator opts into ``fomc_attributable``. ``raw`` (default) leaves
    # ``rates_heads=()`` so the pre-#401 ablation stays byte-identical.
    rates_heads = (
        tuple(RATES_HEAD_NAMES) if rates_target_mode != "raw" else ()
    )

    config = ModelConfig(
        input_size=RICH_FEATURE_SIZE,
        output_mode="classification",
        head_mode=str(args.head_mode),
        regression_alpha=float(args.regression_alpha),
        n_classes=3,
        hidden_size=int(args.hidden_size),
        rates_heads=rates_heads,
        rates_target_mode=rates_target_mode,
        use_regime_conditioning=bool(
            getattr(args, "use_regime_conditioning", False)
        ),
    )

    per_fold: list[dict[str, Any]] = []
    for fold_id in fold_ids:
        split = load_walk_forward_split(
            training_package_id=args.training_package_id,
            fold_id=fold_id,
            rich_features=True,
            text_encoder=str(args.text_encoder),
            use_retrieval_analogs=bool(
                getattr(args, "use_retrieval_analogs", False)
            ),
            use_regime_conditioning=bool(
                getattr(args, "use_regime_conditioning", False)
            ),
            **flags["loader"],
        )
        # Pre-scaler-fit zeroing for the per-bar families that the
        # loader has no flag for. Same contract as #309's
        # ``_zero_derived_text_features`` rewrite -- the scaler in
        # ``train_model`` only sees zeros for those columns, so
        # post-scaling values stay literal 0.
        _zero_per_bar_market_aux(
            split.train,
            zero_realised_vol=flags["zero_realised_vol"],
            zero_cross_asset=flags["zero_cross_asset"],
        )
        _zero_per_bar_market_aux(
            split.val,
            zero_realised_vol=flags["zero_realised_vol"],
            zero_cross_asset=flags["zero_cross_asset"],
        )
        _zero_per_bar_market_aux(
            split.test,
            zero_realised_vol=flags["zero_realised_vol"],
            zero_cross_asset=flags["zero_cross_asset"],
        )

        result = train_model(
            model_config=config,
            train_sequence_groups=split.train,
            val_sequence_groups=split.val,
            test_sequence_groups=split.test,
            fold_id=split.fold_id,
            protocol=split.protocol,
            epochs=int(args.epochs),
            seed=int(seed),
            save_checkpoint=False,
        )
        per_fold.append(
            {
                "fold_id": split.fold_id,
                "metrics": _trial_metrics(result.summary),
            }
        )

    return {
        "cell": cell_label,
        "seed": seed,
        "zeroed_families": sorted(zero_set),
        "loader_flags": flags["loader"],
        "zero_realised_vol": flags["zero_realised_vol"],
        "zero_cross_asset": flags["zero_cross_asset"],
        "folds": per_fold,
    }


def main() -> int:
    ensure_compile_safe()
    args = _parse_args()
    output_path = _resolve_output_path(args.output)
    print(f"[per_family_ablation] writing -> {output_path}")

    fold_ids = _resolve_fold_ids(args.training_package_id, args.folds)
    print(f"[per_family_ablation] folds={fold_ids}")

    cells = _resolve_cells(args.cells)
    print(f"[per_family_ablation] cells={[c[0] for c in cells]}")
    if str(args.rates_target_mode) != "raw":
        print(
            "[per_family_ablation] auto-activating rates heads for "
            f"rates_target_mode={args.rates_target_mode}"
        )

    trials: dict[str, list[dict[str, Any]]] = {}
    for cell_label, zero_set in cells:
        print(
            f"[per_family_ablation] >>> cell={cell_label} "
            f"zeroed={sorted(zero_set) or '[]'}",
            flush=True,
        )
        cell_trials: list[dict[str, Any]] = []
        for seed in args.seeds:
            print(
                f"[per_family_ablation] {cell_label} seed={seed} "
                f"epochs={args.epochs}",
                flush=True,
            )
            cell_trials.append(
                _run_one_cell(
                    cell_label,
                    zero_set,
                    seed,
                    args,
                    fold_ids=fold_ids,
                )
            )
        trials[cell_label] = cell_trials

    summary: dict[str, dict[str, Any]] = {}
    for cell_label, cell_trials in trials.items():
        f1_values: list[float] = []
        rmse_values: list[float] = []
        for trial in cell_trials:
            for fold in trial["folds"]:
                metrics = fold.get("metrics", {}) or {}
                f1 = metrics.get("regime_f1_macro")
                if f1 is not None:
                    f1_values.append(float(f1))
                rmse = metrics.get("regression_rmse_log_rv")
                if rmse is not None:
                    rmse_values.append(float(rmse))
        summary[cell_label] = {
            "regime_f1_macro": _bootstrap_ci(
                f1_values,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            ),
            "regression_rmse_log_rv": _bootstrap_ci(
                rmse_values,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            ),
        }

    payload: dict[str, Any] = {
        "families": list(_FAMILIES),
        "cells": [c[0] for c in cells],
        "seeds": list(args.seeds),
        "fold_ids": fold_ids,
        "epochs": int(args.epochs),
        "hidden_size": int(args.hidden_size),
        "head_mode": str(args.head_mode),
        "regression_alpha": float(args.regression_alpha),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "training_package_id": args.training_package_id,
        "text_encoder": str(args.text_encoder),
        "rates_target_mode": str(args.rates_target_mode),
        "rates_heads": _resolved_rates_heads_for_payload(
            str(args.rates_target_mode)
        ),
        "use_retrieval_analogs": bool(args.use_retrieval_analogs),
        "use_regime_conditioning": bool(args.use_regime_conditioning),
        "post_350_status": str(args.post_350_status),
        "cumulative_chain": [
            {"label": label, "families": list(families)}
            for label, families in _CUMULATIVE_CHAIN
        ],
        "trials": trials,
        "summary": summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[per_family_ablation] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
