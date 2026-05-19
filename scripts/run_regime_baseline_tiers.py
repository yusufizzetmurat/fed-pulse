"""3-tier baseline harness for the Phase 9 V2 vol-regime classifier.

Runs three increasingly rich classification baselines so the aggregate
table can show the marginal lift of each input family on top of the
market-only floor:

    tier1_market_only         -- 6-feature legacy input (close, vol,
                                  close_change_pct, vol_change,
                                  elapsed_time, sentiment_score). No
                                  credibility, no linguistic, no
                                  multi-axis, no text embeddings.
    tier2_market_rich         -- 35-dim rich-feature input. Adds
                                  credibility + linguistic + mp-surprise
                                  + multi-axis families; no embeddings.
    tier3_market_rich_nlp     -- Three-stream input: tier-2 rich +
                                  pooled text embeddings from the
                                  configured encoder (default
                                  finbert_fed_adjacent). This is the
                                  full Phase 9 V2 input contract.

Each tier reuses the same architecture / fold / seed grid so the only
moving variable between tiers is the input-feature family. The harness
shells out to ``python -m app.train_forecaster`` per tier; the per-tier
JSON lands in ``data/artifacts/regime_baseline_tiers/<package_id>/<tier>/``
and the post-run aggregator (``app.evaluation.forecaster_sweep_aggregator``)
can read each directory independently.

Usage example::

    python scripts/run_regime_baseline_tiers.py \\
        --training-package-id <pkg> \\
        --seeds 11 29 47 71 97 \\
        --folds wf_fold_1 wf_fold_2 wf_fold_3 wf_fold_4 \\
        --architectures lstm \\
        --vol-regime-classes 3 \\
        --nlp-text-encoder finbert_fed_adjacent
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


_DEFAULT_REPORT_ROOT = Path("data/artifacts/regime_baseline_tiers")


def _build_common_args(args: argparse.Namespace) -> list[str]:
    """Tier-invariant flags. Architecture, seeds, folds, and the
    classification dispatch all stay constant across the three tiers."""

    return [
        "--training-package-id",
        args.training_package_id,
        "--sweep",
        "--architectures",
        *args.architectures,
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
    ]


def _tier_args(
    tier: str,
    args: argparse.Namespace,
    report_path: Path,
) -> list[str]:
    """Return the tier-specific flag overlay on top of the common args."""

    common = _build_common_args(args)
    if tier == "tier1_market_only":
        return [
            *common,
            "--no-rich-features",
            "--text-encoder",
            "none",
            "--report-path",
            str(report_path),
        ]
    if tier == "tier2_market_rich":
        return [
            *common,
            "--rich-features",
            "--text-encoder",
            "none",
            "--report-path",
            str(report_path),
        ]
    if tier == "tier3_market_rich_nlp":
        return [
            *common,
            "--rich-features",
            "--text-encoder",
            args.nlp_text_encoder,
            "--text-adapter-dims",
            *[str(d) for d in args.text_adapter_dims],
            "--report-path",
            str(report_path),
        ]
    raise ValueError(f"Unknown tier: {tier!r}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "3-tier baseline harness for Phase 9 V2 (#195) vol-regime "
            "classification. Runs Market-Only, Market+Rich, and "
            "Market+Rich+NLP-Embeddings as separate sweeps so the marginal "
            "lift of each input family is measurable."
        )
    )
    parser.add_argument(
        "--training-package-id",
        required=True,
        help="Training package containing events.parquet + splits + folds.",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["lstm"],
        help="Architectures to sweep. Defaults to lstm only; the headline "
        "tier comparison should not vary architecture.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[11, 29, 47, 71, 97],
        help="Official seed set. Default: 5 official seeds.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=["wf_fold_1", "wf_fold_2", "wf_fold_3", "wf_fold_4"],
        help="Walk-forward folds.",
    )
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[64])
    parser.add_argument("--num-layers", nargs="+", type=int, default=[2])
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0.2])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-3])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[1e-4])
    parser.add_argument(
        "--text-adapter-dims",
        nargs="+",
        type=int,
        default=[64],
        help="Adapter dims for tier 3 only.",
    )
    parser.add_argument(
        "--nlp-text-encoder",
        default="finbert_fed_adjacent",
        help="Text encoder used in tier 3. Use ``finbert`` / "
        "``finbert_fomc`` / ``bge_large_en_v15`` / ``voyage_finance_2`` "
        "to swap the embedding source.",
    )
    parser.add_argument(
        "--vol-regime-classes",
        type=int,
        default=3,
        help="Class count for the vol-regime head. Default 3 "
        "(calm / normal / high).",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=_DEFAULT_REPORT_ROOT,
        help="Root directory for per-tier JSON artefacts.",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        choices=("tier1_market_only", "tier2_market_rich", "tier3_market_rich_nlp"),
        default=("tier1_market_only", "tier2_market_rich", "tier3_market_rich_nlp"),
        help="Subset of tiers to run. Default: all three.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the per-tier command instead of executing it. "
        "Used by unit tests and pre-flight checks.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report_root = args.report_root / args.training_package_id
    report_root.mkdir(parents=True, exist_ok=True)

    for tier in args.tiers:
        tier_dir = report_root / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        report_path = tier_dir / "forecaster_sweep_results.json"
        cmd_args = _tier_args(tier, args, report_path)
        cmd = [sys.executable, "-m", "app.train_forecaster", *cmd_args]
        print(f"[regime_tiers] running {tier} -> {report_path}", flush=True)
        print(f"[regime_tiers] cmd: {shlex.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        env = os.environ.copy()
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(
                f"[regime_tiers] tier {tier} exited with status "
                f"{result.returncode}; bailing on remaining tiers.",
                file=sys.stderr,
            )
            return result.returncode
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
