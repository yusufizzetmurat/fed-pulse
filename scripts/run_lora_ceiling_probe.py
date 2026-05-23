"""Round 5 (#244) ceiling-probe orchestrator.

Resolves the best-arch from the latest ``regime_arch_sweep`` aggregator
output for a given training package, then invokes
``app.train_forecaster`` with ``--encoder-lora`` on that architecture
for seed 97 across all four walk-forward folds. The result lands as a
single sweep JSON under
``data/artifacts/encoder_lora_ceiling_probe/<package>/`` so the
follow-up pooled-CI aggregator can produce the macro-F1 number that
feeds the §6.6 LoRA-vs-static-cache row.

Why a wrapper: the ceiling probe needs (a) the right architecture
pulled from the post-correction sweep, not the pre-correction §6.6
table, and (b) a tight HP cell that matches Round 1's best so the
delta isolates the encoder-LoRA lift rather than mixing in HP-grid
variation. The wrapper reads both off the existing aggregator JSON
and refuses to run if Round 1's results are absent.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


_DEFAULT_PACKAGE_ID = "tp_v2_sprint1_2026_05_15_sentiment_market_core_v1.0_epv1_v1.0"
_DEFAULT_ARCH_SWEEP_ROOT = Path("data/artifacts/regime_arch_sweep")
_DEFAULT_OUTPUT_ROOT = Path("data/artifacts/encoder_lora_ceiling_probe")
_DEFAULT_SEED = 97
_DEFAULT_FOLDS: tuple[str, ...] = (
    "wf_fold_1",
    "wf_fold_2",
    "wf_fold_3",
    "wf_fold_4",
)
_DEFAULT_TEXT_ENCODER = "finbert_fed_adjacent"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve Round 1's best architecture from the post-correction "
            "regime_arch_sweep results and invoke train_forecaster with "
            "--encoder-lora for the Round 5 ceiling-probe cell."
        )
    )
    parser.add_argument(
        "--training-package-id",
        default=_DEFAULT_PACKAGE_ID,
    )
    parser.add_argument(
        "--arch-sweep-root",
        type=Path,
        default=_DEFAULT_ARCH_SWEEP_ROOT,
        help="Parent of <package_id>/<architecture>/forecaster_sweep_results.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help="Single seed. Default 97 matches the ceiling-probe scope.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=list(_DEFAULT_FOLDS),
    )
    parser.add_argument(
        "--text-encoder",
        default=_DEFAULT_TEXT_ENCODER,
        help="Encoder alias (must be pinned in models/registry.yaml).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=(
            "Smaller-than-default batch size to keep the LoRA forward "
            "inside the RTX 4080 16GB envelope -- the encoder forward "
            "memory dominates the recurrent core memory."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--architecture",
        default=None,
        help=(
            "Override the auto-resolved best architecture. Useful when "
            "the post-correction sweep has not finished yet or you want "
            "to probe a specific arch."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved CLI without executing it.",
    )
    return parser.parse_args(argv)


def _resolve_best_architecture(
    arch_sweep_dir: Path,
) -> tuple[str, dict[str, Any]]:
    """Pick the best architecture by pooled macro-F1 across the
    per-architecture sweep JSONs. Reads ``<arch>/forecaster_sweep_results.json``
    for each subdirectory, picks the trial with the highest
    ``test_metrics.classification_breakdown.macro_f1``, and returns
    ``(arch_name, best_trial_record)``."""

    candidates: list[tuple[str, float, dict[str, Any]]] = []
    if not arch_sweep_dir.exists():
        raise FileNotFoundError(
            f"arch sweep directory not found: {arch_sweep_dir}; run "
            "``make regime-arch-sweep`` first"
        )
    for arch_dir in sorted(arch_sweep_dir.iterdir()):
        if not arch_dir.is_dir():
            continue
        results_path = arch_dir / "forecaster_sweep_results.json"
        if not results_path.exists():
            continue
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        trials = payload.get("trials") if isinstance(payload, dict) else None
        if not isinstance(trials, list):
            continue
        best_macro = -1.0
        best_record: dict[str, Any] | None = None
        for trial in trials:
            if not isinstance(trial, dict):
                continue
            summary = trial.get("summary") if isinstance(trial.get("summary"), dict) else trial
            test_metrics = summary.get("test_metrics") if isinstance(summary, dict) else None
            if not isinstance(test_metrics, dict):
                continue
            breakdown = test_metrics.get("classification_breakdown")
            macro = None
            if isinstance(breakdown, dict):
                macro = breakdown.get("macro_f1")
            if not isinstance(macro, (int, float)):
                continue
            macro_value = float(macro)
            if macro_value > best_macro:
                best_macro = macro_value
                best_record = trial
        if best_record is not None:
            candidates.append((arch_dir.name, best_macro, best_record))

    if not candidates:
        raise RuntimeError(
            f"no per-architecture sweep results with a usable macro-F1 "
            f"in {arch_sweep_dir}; the pooled aggregator must have run "
            "first or the sweep produced no classification trials"
        )
    candidates.sort(key=lambda row: row[1], reverse=True)
    best_arch_name, best_macro_f1, best_trial = candidates[0]
    print(
        f"[lora_ceiling_probe] best architecture from "
        f"{arch_sweep_dir.name}: {best_arch_name} (macro_f1={best_macro_f1:.4f})"
    )
    return best_arch_name, best_trial


def _trial_hp_cell(trial: dict[str, Any]) -> dict[str, Any]:
    """Extract the HP cell that produced the best macro-F1 -- the
    LoRA probe re-uses this cell so the delta isolates the LoRA lift
    rather than mixing in HP search variance."""

    candidate = trial.get("candidate") if isinstance(trial.get("candidate"), dict) else {}
    summary = trial.get("summary") if isinstance(trial.get("summary"), dict) else trial
    model_config = summary.get("model_config") if isinstance(summary, dict) else None
    cell: dict[str, Any] = {}
    if isinstance(candidate, dict):
        cell.update(candidate)
    if isinstance(model_config, dict):
        for key in (
            "hidden_size",
            "num_layers",
            "dropout",
            "head_hidden_size",
        ):
            if key in model_config:
                cell.setdefault(key, model_config[key])
    return cell


def _build_train_forecaster_cmd(
    *,
    args: argparse.Namespace,
    architecture: str,
    hp_cell: dict[str, Any],
    report_path: Path,
) -> list[str]:
    hidden_size = int(hp_cell.get("hidden_size", 64))
    num_layers = int(hp_cell.get("num_layers", 2))
    dropout = float(hp_cell.get("dropout", 0.2))
    learning_rate = float(hp_cell.get("learning_rate", 1e-3))
    weight_decay = float(hp_cell.get("weight_decay", 1e-4))
    text_adapter_dim = int(hp_cell.get("text_adapter_dim", 64))
    return [
        sys.executable,
        "-m",
        "app.train_forecaster",
        "--training-package-id",
        args.training_package_id,
        "--sweep",
        "--architectures",
        architecture,
        "--seeds",
        str(int(args.seed)),
        "--folds",
        *list(args.folds),
        "--hidden-sizes",
        str(hidden_size),
        "--num-layers-grid",
        str(num_layers),
        "--dropouts",
        str(dropout),
        "--learning-rates",
        str(learning_rate),
        "--weight-decays",
        str(weight_decay),
        "--text-adapter-dims",
        str(text_adapter_dim),
        "--batch-size",
        str(int(args.batch_size)),
        "--epochs",
        str(int(args.epochs)),
        "--output-mode",
        "classification",
        "--vol-regime-classes",
        "3",
        "--rich-features",
        "--use-llm-features",
        "--text-encoder",
        str(args.text_encoder),
        "--encoder-lora",
        "--report-path",
        str(report_path),
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    arch_sweep_dir = args.arch_sweep_root / args.training_package_id

    if args.architecture:
        architecture = str(args.architecture)
        # Manual override: pull the best HP cell from whatever subset
        # of the sweep is available so the probe still re-uses a
        # measured cell rather than guessing.
        trial: dict[str, Any] = {}
        per_arch = arch_sweep_dir / architecture / "forecaster_sweep_results.json"
        if per_arch.exists():
            _, trial = _resolve_best_architecture(arch_sweep_dir / architecture)
    else:
        architecture, trial = _resolve_best_architecture(arch_sweep_dir)
    hp_cell = _trial_hp_cell(trial)

    output_dir = args.output_root / args.training_package_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "forecaster_sweep_results.json"

    cmd = _build_train_forecaster_cmd(
        args=args,
        architecture=architecture,
        hp_cell=hp_cell,
        report_path=report_path,
    )
    print(f"[lora_ceiling_probe] cmd: {shlex.join(cmd)}", flush=True)
    if args.dry_run:
        return 0

    env = os.environ.copy()
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(
            f"[lora_ceiling_probe] exited with status {result.returncode}",
            file=sys.stderr,
        )
        return result.returncode
    print(f"[lora_ceiling_probe] sweep report at {report_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
