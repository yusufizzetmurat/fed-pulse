"""Phase A architecture sweep — random-search HP on every architecture.

Runs ``python -m app.train_forecaster --sweep --random-search ...`` once
per architecture in ``{lstm_attn, gru, tcn, transformer, tft}``, all on
the Tier 5 surface (rich + LLM features, no NLP encoder) so the ensemble
aggregator downstream compares architectures cleanly. Reuses every
existing flag on the trainer — this is a thin orchestrator, no model
code lives here.

Per-architecture artefact:
``<report_root>/<package_id>/<architecture>/forecaster_sweep_results.json``.

The follow-on ``python -m app.evaluation.ensemble_aggregator`` reads the
parent directory and aligns the per-architecture trials on each
``(fold_id, seed)`` cell.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


_DEFAULT_REPORT_ROOT = Path("data/artifacts/regime_arch_sweep")

# TFT is intentionally absent from the canonical sweep targets. See
# ``docs/adr/0020-tft-architecture-comparison-exclusion.md`` and the
# §6.7 footnote in ``fed-pulse.wiki/06_Deep_Learning_Roadmap.md`` --
# the in-repo generic-head evaluation is architecture-mismatch with
# TFT's native quantile + VSN inductive bias; the 0.3803 result lives
# in the wiki only as historical record. Pass ``--architectures tft``
# explicitly to opt back in for a one-off investigation.
_DEFAULT_ARCHITECTURES = (
    "lstm_attn",
    "gru",
    "tcn",
    "transformer",
)


def _build_common_args(args: argparse.Namespace) -> list[str]:
    """Tier-invariant flags. Fold / seed / classification dispatch are
    constant across architectures; only ``--architectures`` and the
    output path differ per cell."""

    return [
        "--training-package-id",
        args.training_package_id,
        "--sweep",
        "--random-search",
        "--random-search-samples",
        str(args.random_search_samples),
        "--random-search-seed",
        str(args.random_search_seed),
        "--seeds",
        *[str(s) for s in args.seeds],
        "--folds",
        *args.folds,
        "--hidden-sizes",
        *[str(h) for h in args.hidden_sizes],
        "--num-layers-grid",
        *[str(n) for n in args.num_layers],
        "--dropouts",
        *[str(d) for d in args.dropouts],
        "--learning-rates",
        *[str(lr) for lr in args.learning_rates],
        "--weight-decays",
        *[str(wd) for wd in args.weight_decays],
        "--output-mode",
        "classification",
        "--vol-regime-classes",
        str(args.vol_regime_classes),
        "--rich-features",
    ]


def _arch_args(
    architecture: str,
    args: argparse.Namespace,
    report_path: Path,
) -> list[str]:
    common = _build_common_args(args)
    arch_cmd = [
        *common,
        "--architectures",
        architecture,
        "--report-path",
        str(report_path),
    ]
    if args.use_llm_features:
        arch_cmd.append("--use-llm-features")
    if args.text_encoder and args.text_encoder != "none":
        arch_cmd.extend(["--text-encoder", args.text_encoder])
        arch_cmd.extend(["--text-adapter-dims", *[str(d) for d in args.text_adapter_dims]])
    else:
        arch_cmd.extend(["--text-encoder", "none"])
    return arch_cmd


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase A architecture sweep -- random-search HP per architecture. "
            "Mirrors the tier orchestrator pattern but the moving axis is "
            "the model architecture, not the input-feature family."
        )
    )
    parser.add_argument("--training-package-id", required=True)
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=list(_DEFAULT_ARCHITECTURES),
        help=(
            "Architectures to sweep. Default: lstm_attn, gru, tcn, transformer. "
            "TFT is excluded from the canonical comparison per ADR 0020; "
            "pass --architectures tft explicitly to opt in."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[11, 29, 47, 71, 97],
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=["wf_fold_1", "wf_fold_2", "wf_fold_3", "wf_fold_4"],
    )
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        default=[128, 256],
        help="A5 best-cell neighbourhood. Default: 128, 256.",
    )
    parser.add_argument("--num-layers", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0.1, 0.2])
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=[3e-4, 1e-3],
    )
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[0.0, 1e-4])
    parser.add_argument("--text-adapter-dims", nargs="+", type=int, default=[64])
    parser.add_argument("--vol-regime-classes", type=int, default=3)
    parser.add_argument(
        "--text-encoder",
        default="none",
        help=(
            "Text encoder alias. Default ``none`` keeps the surface comparable "
            "to Tier 5; pass ``finbert_fed_adjacent`` to add the NLP channel."
        ),
    )
    parser.add_argument(
        "--use-llm-features",
        dest="use_llm_features",
        action="store_true",
        help="Attach the B1 LLM-features block. Default on.",
    )
    parser.add_argument(
        "--no-llm-features",
        dest="use_llm_features",
        action="store_false",
        help="Disable the B1 LLM-features block.",
    )
    parser.set_defaults(use_llm_features=True)
    parser.add_argument(
        "--random-search-samples",
        type=int,
        default=20,
        help="Number of HP combos to sample per architecture. Default 20.",
    )
    parser.add_argument(
        "--random-search-seed",
        type=int,
        default=42,
        help="RNG seed for the HP sampler. Default 42.",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=_DEFAULT_REPORT_ROOT,
        help="Root for per-architecture report JSONs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report_root = args.report_root / args.training_package_id
    report_root.mkdir(parents=True, exist_ok=True)

    for arch in args.architectures:
        arch_dir = report_root / arch
        arch_dir.mkdir(parents=True, exist_ok=True)
        report_path = arch_dir / "forecaster_sweep_results.json"
        cmd = [
            sys.executable,
            "-m",
            "app.train_forecaster",
            *_arch_args(arch, args, report_path),
        ]
        print(f"[regime_arch_sweep] running architecture={arch} -> {report_path}")
        print(f"[regime_arch_sweep] cmd: {shlex.join(cmd)}")
        if args.dry_run:
            continue
        result = subprocess.run(cmd, env=os.environ.copy())
        if result.returncode != 0:
            print(
                f"[regime_arch_sweep] architecture {arch} exited with status "
                f"{result.returncode}; bailing on remaining architectures."
            )
            return result.returncode
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
