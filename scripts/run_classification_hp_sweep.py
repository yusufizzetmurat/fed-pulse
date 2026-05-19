"""A5 (#210): HP sweep on the vol-regime classifier at the wider feature set.

After A1 (class weighting) + A2 (vol horizons) + A3 (cross-asset / VIX)
the feature input width is 41 dimensions. The 3-tier baseline still
uses a single hyperparameter combo (hidden=64, layers=2, dropout=0.2,
lr=1e-3, weight_decay=1e-4). A3's accuracy lift / macro-F1 dip pattern
suggests the regularisation point that was right for the 35-dim input
is now too loose at 41 dims.

This script runs a random-search HP sweep on Tier 2 (Market + Rich) and
Tier 3 (Market + Rich + NLP). Each tier emits per-trial JSON under a
fresh directory so the aggregator can read them independently.

Grid:
    hidden        : 32 / 64 / 128 / 256
    num_layers    : 1 / 2 / 3
    dropout       : 0.1 / 0.2 / 0.3 / 0.4
    learning_rate : 1e-3 / 5e-4 / 3e-4 / 1e-4
    weight_decay  : 0 / 1e-4 / 1e-3
    text_adapter  : 32 / 64 / 128 (tier 3 only)

5 official seeds × 4 walk-forward folds × random-search M=20 samples.

Total trials per tier:
    Tier 2: 20 × 5 × 4 = 400
    Tier 3: 20 × 5 × 4 = 400

About 3-4 hours total on a 4080.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


_DEFAULT_REPORT_ROOT = Path("data/artifacts/regime_baseline_tiers")


def _build_args(args: argparse.Namespace, tier: str, report_path: Path) -> list[str]:
    base = [
        "--training-package-id",
        args.training_package_id,
        "--sweep",
        "--rich-features",
        "--architectures",
        "lstm",
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
        "--random-search",
        "--random-search-samples",
        str(args.random_search_samples),
        "--random-search-seed",
        str(args.random_search_seed),
        "--output-mode",
        "classification",
        "--vol-regime-classes",
        str(args.vol_regime_classes),
        "--report-path",
        str(report_path),
    ]
    if tier == "tier2_market_rich":
        base += ["--text-encoder", "none"]
    elif tier == "tier3_market_rich_nlp":
        base += [
            "--text-encoder",
            args.nlp_text_encoder,
            "--text-adapter-dims",
            *[str(d) for d in args.text_adapter_dims],
        ]
    else:
        raise ValueError(f"Unknown tier: {tier!r}")
    return base


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-package-id", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47, 71, 97])
    p.add_argument(
        "--folds",
        nargs="+",
        default=["wf_fold_1", "wf_fold_2", "wf_fold_3", "wf_fold_4"],
    )
    p.add_argument("--hidden-sizes", nargs="+", type=int, default=[32, 64, 128, 256])
    p.add_argument("--num-layers", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument(
        "--dropouts", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4]
    )
    p.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=[1e-3, 5e-4, 3e-4, 1e-4],
    )
    p.add_argument(
        "--weight-decays", nargs="+", type=float, default=[0.0, 1e-4, 1e-3]
    )
    p.add_argument(
        "--text-adapter-dims", nargs="+", type=int, default=[32, 64, 128]
    )
    p.add_argument(
        "--nlp-text-encoder", default="finbert_fed_adjacent"
    )
    p.add_argument("--random-search-samples", type=int, default=20)
    p.add_argument("--random-search-seed", type=int, default=42)
    p.add_argument("--vol-regime-classes", type=int, default=3)
    p.add_argument(
        "--report-root", type=Path, default=_DEFAULT_REPORT_ROOT
    )
    p.add_argument(
        "--tiers",
        nargs="+",
        choices=("tier2_market_rich", "tier3_market_rich_nlp"),
        default=("tier2_market_rich", "tier3_market_rich_nlp"),
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    pkg_root = args.report_root / args.training_package_id / "classification_hp_sweep"
    pkg_root.mkdir(parents=True, exist_ok=True)

    for tier in args.tiers:
        tier_dir = pkg_root / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        report_path = tier_dir / "forecaster_sweep_results.json"
        cmd_args = _build_args(args, tier, report_path)
        cmd = [sys.executable, "-m", "app.train_forecaster", *cmd_args]
        print(f"[hp_sweep] tier={tier} -> {report_path}", flush=True)
        print(f"[hp_sweep] cmd: {shlex.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        env = os.environ.copy()
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(
                f"[hp_sweep] tier {tier} exited {result.returncode}; "
                "bailing on remaining tiers.",
                file=sys.stderr,
            )
            return result.returncode
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
