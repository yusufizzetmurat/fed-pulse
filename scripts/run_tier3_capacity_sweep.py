"""Tier 3 NLP capacity sweep for the vol-regime classifier (#198).

Tracks whether the NLP integration redundancy seen in the 3-tier baseline
(PR #197, §6.3 of 06_Deep_Learning_Roadmap.md) survives a small
adapter-dim × encoder search. Holds architecture / seeds / folds /
HP fixed except for the two NLP knobs.

Grid:
    adapter dims: 32, 64, 128
    encoders:     finbert_fed_adjacent, bge_large_en_v15
    architecture: lstm
    seeds:        official 5
    folds:        wf_fold_1 .. wf_fold_4

Total trials: 3 × 2 × 5 × 4 = 120.

Per-encoder JSON lands at
``data/artifacts/regime_baseline_tiers/<pkg>/tier3_capacity_sweep/<encoder>/forecaster_sweep_results.json``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


_DEFAULT_REPORT_ROOT = Path("data/artifacts/regime_baseline_tiers")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tier 3 NLP capacity sweep (#198)")
    p.add_argument("--training-package-id", required=True)
    p.add_argument(
        "--encoders",
        nargs="+",
        default=["finbert_fed_adjacent", "bge_large_en_v15"],
        help="Cached encoder aliases to sweep.",
    )
    p.add_argument(
        "--adapter-dims",
        nargs="+",
        type=int,
        default=[32, 64, 128],
        help="Text-adapter output dimensions to sweep.",
    )
    p.add_argument("--architecture", default="lstm")
    p.add_argument(
        "--seeds", nargs="+", type=int, default=[11, 29, 47, 71, 97]
    )
    p.add_argument(
        "--folds",
        nargs="+",
        default=["wf_fold_1", "wf_fold_2", "wf_fold_3", "wf_fold_4"],
    )
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--vol-regime-classes", type=int, default=3)
    p.add_argument("--report-root", type=Path, default=_DEFAULT_REPORT_ROOT)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    pkg_root = args.report_root / args.training_package_id / "tier3_capacity_sweep"
    pkg_root.mkdir(parents=True, exist_ok=True)

    for encoder in args.encoders:
        encoder_dir = pkg_root / encoder
        encoder_dir.mkdir(parents=True, exist_ok=True)
        report_path = encoder_dir / "forecaster_sweep_results.json"
        cmd = [
            sys.executable,
            "-m",
            "app.train_forecaster",
            "--training-package-id",
            args.training_package_id,
            "--sweep",
            "--rich-features",
            "--architectures",
            args.architecture,
            "--seeds",
            *[str(s) for s in args.seeds],
            "--folds",
            *args.folds,
            "--hidden-sizes",
            str(args.hidden_size),
            "--num-layers-grid",
            str(args.num_layers),
            "--dropouts",
            str(args.dropout),
            "--learning-rates",
            str(args.learning_rate),
            "--weight-decays",
            str(args.weight_decay),
            "--text-encoder",
            encoder,
            "--text-adapter-dims",
            *[str(d) for d in args.adapter_dims],
            "--output-mode",
            "classification",
            "--vol-regime-classes",
            str(args.vol_regime_classes),
            "--report-path",
            str(report_path),
        ]
        print(f"[tier3_capacity] encoder={encoder} -> {report_path}", flush=True)
        print(f"[tier3_capacity] cmd: {shlex.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        env = os.environ.copy()
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(
                f"[tier3_capacity] encoder={encoder} exited {result.returncode}; "
                "stopping remaining encoders.",
                file=sys.stderr,
            )
            return result.returncode
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
