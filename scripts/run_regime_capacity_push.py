"""Phase B capacity push — random-search on hidden / schedule / wd.

Runs ``python -m app.train_forecaster --sweep --random-search`` at the
Tier 5 surface (rich + LLM, no NLP encoder) with the capacity-push grid
(hidden in {256, 384, 512}, lr_schedule in {plateau, cosine_warmup},
weight-decay in {0, 1e-4}) so the headline LSTM cell from A5 has a
chance to lift further on a wider model + warmup-cosine schedule.

The downstream ``regime-pooled-aggregate`` reports the pooled-fold
macro-F1 with bootstrap CI on the resulting JSON.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


_DEFAULT_REPORT_ROOT = Path("data/artifacts/regime_capacity_push")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase B capacity push -- random-search on hidden, LR "
            "schedule, and weight decay at the Tier 5 surface."
        )
    )
    parser.add_argument("--training-package-id", required=True)
    parser.add_argument("--architecture", default="lstm")
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
        default=[256, 384, 512],
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
    parser.add_argument(
        "--lr-schedules",
        nargs="+",
        choices=("plateau", "cosine_warmup"),
        default=["plateau", "cosine_warmup"],
    )
    parser.add_argument(
        "--text-encoder",
        default="none",
        help="Default ``none`` keeps the Tier 5 surface (no NLP) consistent with the A5 + B1 baseline.",
    )
    parser.add_argument(
        "--use-llm-features",
        dest="use_llm_features",
        action="store_true",
    )
    parser.add_argument(
        "--no-llm-features",
        dest="use_llm_features",
        action="store_false",
    )
    parser.set_defaults(use_llm_features=True)
    parser.add_argument("--vol-regime-classes", type=int, default=3)
    parser.add_argument("--random-search-samples", type=int, default=40)
    parser.add_argument("--random-search-seed", type=int, default=42)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=_DEFAULT_REPORT_ROOT,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _build_command(args: argparse.Namespace, report_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "app.train_forecaster",
        "--training-package-id",
        args.training_package_id,
        "--sweep",
        "--random-search",
        "--random-search-samples",
        str(args.random_search_samples),
        "--random-search-seed",
        str(args.random_search_seed),
        "--architectures",
        args.architecture,
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
        "--lr-schedules",
        *args.lr_schedules,
        "--output-mode",
        "classification",
        "--vol-regime-classes",
        str(args.vol_regime_classes),
        "--rich-features",
        "--text-encoder",
        args.text_encoder,
        "--report-path",
        str(report_path),
    ]
    if args.use_llm_features:
        cmd.append("--use-llm-features")
    return cmd


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report_root = args.report_root / args.training_package_id
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / "forecaster_sweep_results.json"
    cmd = _build_command(args, report_path)
    print(f"[regime_capacity_push] running -> {report_path}")
    print(f"[regime_capacity_push] cmd: {shlex.join(cmd)}")
    if args.dry_run:
        return 0
    result = subprocess.run(cmd, env=os.environ.copy())
    return result.returncode


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
