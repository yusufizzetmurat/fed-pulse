"""Pooled-fold macro-F1 aggregator with bootstrap CIs.

Per-fold macro-F1 reporting at n ≈ 50 per fold has a confidence-band
wider than the lifts most feature families deliver, so the headline
"Tier 3 beats Tier 2 by +0.031" sits inside the per-fold spread. This
aggregator pools the test-partition predictions from every walk-forward
fold into a single n ≈ 200 evaluation, recomputes macro-F1 on the
pooled cells, and reports a bootstrap CI on the pooled statistic —
the statistically honest version of "macro-F1 across the full headline
holdout".

Input contract: one or more ``forecaster_sweep_results.json`` files
produced by ``app.train_forecaster --sweep --output-mode classification``.
Each per-trial summary must carry ``test_metrics.predictions`` and
``test_metrics.targets`` (PR #226 wired both onto :class:`EvaluationMetrics`
behind ``record_row_predictions=True`` at the test-partition call site).

Trials are grouped by ``(architecture, hp_combo_id, seed)`` if those
fields are present; otherwise by ``(architecture, seed)``. Within a
group, the four fold-tagged trials are pooled. The aggregator selects
the single best HP-cell per architecture via the existing
``selection_metric`` field on the parent JSON, then reports the pooled
macro-F1 of that cell with a block-bootstrap CI.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from app.evaluation.bootstrap import BootstrapCI, block_bootstrap_ci
from app.evaluation.classification_breakdown import (
    ClassificationBreakdown,
    compute_classification_breakdown,
)


@dataclasses.dataclass(frozen=True)
class PooledCell:
    """One pooled-fold cell — the headline row of the output table."""

    architecture: str
    hp_combo_id: str | None
    seeds: tuple[int, ...]
    folds: tuple[str, ...]
    n_pooled: int
    breakdown: ClassificationBreakdown
    macro_f1_ci: BootstrapCI

    def to_dict(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "hp_combo_id": self.hp_combo_id,
            "seeds": list(self.seeds),
            "folds": list(self.folds),
            "n_pooled": self.n_pooled,
            "breakdown": self.breakdown.to_dict(),
            "macro_f1_ci": {
                "point": self.macro_f1_ci.point,
                "lo": self.macro_f1_ci.lo,
                "hi": self.macro_f1_ci.hi,
                "coverage": self.macro_f1_ci.coverage,
                "n_resamples": self.macro_f1_ci.n_resamples,
                "block_size": self.macro_f1_ci.block_size,
            },
        }


def _trial_summary(trial: Mapping[str, object]) -> Mapping[str, object]:
    summary = trial.get("summary") if isinstance(trial.get("summary"), Mapping) else trial
    if not isinstance(summary, Mapping):
        raise ValueError("trial record is missing the 'summary' block")
    return summary


def _test_predictions(trial: Mapping[str, object]) -> tuple[list[int], list[int]] | None:
    summary = _trial_summary(trial)
    test_metrics = summary.get("test_metrics")
    if not isinstance(test_metrics, Mapping):
        return None
    preds = test_metrics.get("predictions")
    targets = test_metrics.get("targets")
    if not isinstance(preds, list) or not isinstance(targets, list):
        return None
    if len(preds) != len(targets):
        raise ValueError(
            f"test_metrics.predictions length {len(preds)} != targets length {len(targets)}"
        )
    return [int(x) for x in preds], [int(x) for x in targets]


def _cell_key(trial: Mapping[str, object]) -> tuple[str, str | None]:
    """Group by (architecture, hp_combo_id) so the same HP cell across
    folds + seeds becomes one row. The seed is rolled into the pool."""
    arch = str(trial.get("architecture", "?"))
    summary = _trial_summary(trial)
    hp = summary.get("hp_combo_id") or trial.get("hp_combo_id")
    if hp is None:
        # Fall back to a stable HP fingerprint when hp_combo_id is not on
        # the per-trial record (single-HP runs).
        cfg = summary.get("model_config")
        if isinstance(cfg, Mapping):
            hp_parts = [
                f"h={cfg.get('hidden_size')}",
                f"L={cfg.get('num_layers')}",
                f"d={cfg.get('dropout')}",
                f"lr={summary.get('learning_rate')}",
                f"wd={summary.get('weight_decay')}",
            ]
            hp = "|".join(hp_parts)
    return arch, str(hp) if hp is not None else None


def _select_best_cell(  # noqa: C901
    cells: Mapping[tuple[str, str | None], list[Mapping[str, object]]],
    selection_metric: str,
) -> dict[str, tuple[str, str | None]]:
    """Return best (arch, hp) per architecture by the per-cell mean of the
    selected per-trial metric (lower-is-better for regression metrics,
    higher-is-better for macro-F1)."""

    higher_is_better = selection_metric in {"macro_f1", "regime_f1_macro"}

    def cell_score(rows: Iterable[Mapping[str, object]]) -> float:
        values: list[float] = []
        for trial in rows:
            summary = _trial_summary(trial)
            test_metrics = summary.get("test_metrics")
            if not isinstance(test_metrics, Mapping):
                continue
            if selection_metric in {"macro_f1", "regime_f1_macro"}:
                breakdown = test_metrics.get("classification_breakdown")
                if isinstance(breakdown, Mapping):
                    point = breakdown.get("macro_f1")
                    if isinstance(point, int | float):
                        values.append(float(point))
            else:
                point = test_metrics.get(selection_metric)
                if isinstance(point, int | float):
                    values.append(float(point))
        if not values:
            return float("-inf") if higher_is_better else float("inf")
        return sum(values) / len(values)

    by_arch: dict[str, tuple[str, str | None]] = {}
    for (arch, hp), rows in cells.items():
        score = cell_score(rows)
        existing = by_arch.get(arch)
        if existing is None:
            by_arch[arch] = (arch, hp)
            continue
        existing_score = cell_score(cells[existing])
        if higher_is_better:
            if score > existing_score:
                by_arch[arch] = (arch, hp)
        else:
            if score < existing_score:
                by_arch[arch] = (arch, hp)
    return by_arch


def pool_cell(  # noqa: PLR0913
    trials: Sequence[Mapping[str, object]],
    *,
    n_classes: int = 3,
    block_size: int = 20,
    n_resamples: int = 1000,
    coverage: float = 0.95,
    bootstrap_seed: int = 11,
) -> PooledCell:
    """Pool the test-partition predictions from ``trials`` into a single
    confusion matrix + breakdown + bootstrap CI on macro-F1."""

    pooled_predictions: list[int] = []
    pooled_targets: list[int] = []
    seeds: list[int] = []
    folds: list[str] = []
    architecture = "?"
    hp_combo_id: str | None = None
    for trial in trials:
        rows = _test_predictions(trial)
        if rows is None:
            continue
        preds, targets = rows
        pooled_predictions.extend(preds)
        pooled_targets.extend(targets)
        seed = trial.get("seed")
        if isinstance(seed, int):
            seeds.append(seed)
        fold = trial.get("fold_id")
        if isinstance(fold, str):
            folds.append(fold)
        architecture = str(trial.get("architecture", architecture))
        if hp_combo_id is None:
            _, hp = _cell_key(trial)
            hp_combo_id = hp

    breakdown = compute_classification_breakdown(
        predictions=pooled_predictions,
        targets=pooled_targets,
        n_classes=n_classes,
    )

    # Bootstrap CI on macro-F1: resample the row-aligned pairs, recompute
    # macro-F1 on each resample, take the empirical CI. Block-bootstrap
    # ordering follows :func:`block_bootstrap_ci` so the helper stays the
    # single CI implementation in the codebase.
    n = len(pooled_predictions)
    if n == 0:
        macro_ci = BootstrapCI(
            point=float("nan"),
            lo=float("nan"),
            hi=float("nan"),
            coverage=coverage,
            n_resamples=n_resamples,
            block_size=block_size,
        )
    else:
        # The aggregator uses :func:`block_bootstrap_ci` on the per-row
        # macro-F1 contribution; macro-F1 itself is a non-linear function
        # of the confusion matrix, so we resample the rows directly and
        # recompute macro-F1 per resample. The internal RNG is seeded for
        # reproducibility.
        import random

        rng = random.Random(bootstrap_seed)
        resampled_macros: list[float] = []
        for _ in range(n_resamples):
            # Block resample by drawing ⌈n/block_size⌉ random starts and
            # taking a contiguous block of length ``block_size`` from each.
            idx: list[int] = []
            n_blocks = max(1, (n + block_size - 1) // block_size)
            for _ in range(n_blocks):
                start = rng.randint(0, max(0, n - block_size))
                idx.extend(range(start, min(n, start + block_size)))
            idx = idx[:n]
            resample_preds = [pooled_predictions[i] for i in idx]
            resample_targets = [pooled_targets[i] for i in idx]
            resample_breakdown = compute_classification_breakdown(
                predictions=resample_preds,
                targets=resample_targets,
                n_classes=n_classes,
            )
            resampled_macros.append(float(resample_breakdown.macro_f1))
        resampled_macros.sort()
        alpha = (1.0 - coverage) / 2.0
        lo_idx = max(0, min(n_resamples - 1, int(alpha * n_resamples)))
        hi_idx = max(0, min(n_resamples - 1, int((1.0 - alpha) * n_resamples) - 1))
        macro_ci = BootstrapCI(
            point=float(breakdown.macro_f1),
            lo=resampled_macros[lo_idx],
            hi=resampled_macros[hi_idx],
            coverage=coverage,
            n_resamples=n_resamples,
            block_size=block_size,
        )

    return PooledCell(
        architecture=architecture,
        hp_combo_id=hp_combo_id,
        seeds=tuple(sorted(set(seeds))),
        folds=tuple(sorted(set(folds))),
        n_pooled=len(pooled_predictions),
        breakdown=breakdown,
        macro_f1_ci=macro_ci,
    )


def aggregate(  # noqa: C901, PLR0913
    sweep_jsons: Sequence[Mapping[str, object]],
    *,
    n_classes: int = 3,
    block_size: int = 20,
    n_resamples: int = 1000,
    coverage: float = 0.95,
    bootstrap_seed: int = 11,
    selection_metric: str | None = None,
) -> list[PooledCell]:
    """Aggregate every ``forecaster_sweep_results.json`` into per-(arch,
    best-HP) pooled cells. ``selection_metric`` falls back to the
    metric recorded on the first sweep JSON; on missing it picks
    ``macro_f1``."""

    cells: dict[tuple[str, str | None], list[Mapping[str, object]]] = defaultdict(list)
    inferred_metric = "macro_f1"
    classification_detected = False
    for blob in sweep_jsons:
        trials = blob.get("trials", [])
        metric = blob.get("selection_metric")
        if isinstance(metric, str) and selection_metric is None:
            inferred_metric = metric
        if not isinstance(trials, list):
            continue
        for trial in trials:
            if not isinstance(trial, Mapping):
                continue
            cells[_cell_key(trial)].append(trial)
            # Detect classification mode off the per-trial model_config
            # so the default selection metric on a classification sweep
            # is ``macro_f1`` rather than the meaningless ``combined_rmse``
            # the trainer writes at the top level. Sweep JSONs do not
            # surface ``output_mode`` at the top so we peek at the first
            # trial that exposes it.
            if not classification_detected:
                summary = _trial_summary(trial)
                cfg = summary.get("model_config") if isinstance(summary, Mapping) else None
                if (
                    isinstance(cfg, Mapping)
                    and str(cfg.get("output_mode", "")).lower() == "classification"
                ):
                    classification_detected = True

    # Classification sweeps' top-level ``selection_metric`` is almost
    # always ``combined_rmse`` (the trainer writes the same field
    # regardless of output mode). That metric is undefined under
    # classification — every trial reports inf for it, so the
    # per-cell selection collapses to whichever cell pandas iterates
    # first. Override to macro-F1 once we know the sweep is
    # classification-typed.
    if selection_metric is None and classification_detected:
        inferred_metric = "macro_f1"

    selection = selection_metric or inferred_metric
    best_by_arch = _select_best_cell(cells, selection)
    pooled_rows: list[PooledCell] = []
    for arch in sorted(best_by_arch.keys()):
        key = best_by_arch[arch]
        cell_rows = cells[key]
        pooled = pool_cell(
            cell_rows,
            n_classes=n_classes,
            block_size=block_size,
            n_resamples=n_resamples,
            coverage=coverage,
            bootstrap_seed=bootstrap_seed,
        )
        pooled_rows.append(pooled)
    return pooled_rows


def _render_markdown(rows: Sequence[PooledCell]) -> str:
    lines: list[str] = []
    lines.append("# Pooled-fold macro-F1 with block-bootstrap CI")
    lines.append("")
    lines.append(
        "Test-partition predictions pooled across every walk-forward fold "
        "in the input sweep JSONs. Macro-F1 recomputed on the pooled "
        "(prediction, target) pairs; bootstrap CI by row-level block resample."
    )
    lines.append("")
    lines.append("| Architecture | n_pooled | seeds | folds | macro-F1 | 95% CI |")
    lines.append("| --- | ---: | --- | --- | ---: | --- |")
    for row in rows:
        seeds = ",".join(str(s) for s in row.seeds) or "—"
        folds = ",".join(row.folds) or "—"
        ci = row.macro_f1_ci
        ci_band = f"[{ci.lo:.4f}, {ci.hi:.4f}]"
        lines.append(
            f"| {row.architecture} | {row.n_pooled} | {seeds} | {folds} | "
            f"{ci.point:.4f} | {ci_band} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pool test-partition predictions across walk-forward folds "
            "and report macro-F1 with a block-bootstrap CI per "
            "architecture × best-HP cell."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing one or more ``forecaster_sweep_results.json`` "
            "files (one per tier or per architecture). The aggregator walks "
            "the tree recursively and ingests every matching file."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path (default: <input-dir>/pooled_test_macro_f1.json).",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Output markdown path (default: <input-dir>/pooled_test_macro_f1.md).",
    )
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=11)
    parser.add_argument(
        "--selection-metric",
        default=None,
        help=(
            "Per-cell selection metric. Defaults to the metric recorded "
            "on the input JSONs (typically ``combined_rmse`` for legacy "
            "runs; macro_f1 for classification-mode sweeps)."
        ),
    )
    args = parser.parse_args(argv)

    input_dir: Path = args.input_dir
    if not input_dir.exists():
        raise SystemExit(f"input dir not found: {input_dir}")

    sweep_blobs: list[Mapping[str, object]] = []
    for candidate in sorted(input_dir.rglob("forecaster_sweep_results.json")):
        try:
            with candidate.open("r", encoding="utf-8") as fh:
                blob = json.load(fh)
            if isinstance(blob, Mapping):
                sweep_blobs.append(blob)
        except json.JSONDecodeError as exc:
            print(f"[regime_pooled_aggregator] skipping {candidate}: {exc}")

    if not sweep_blobs:
        raise SystemExit(f"no forecaster_sweep_results.json found under {input_dir}")

    rows = aggregate(
        sweep_blobs,
        n_classes=args.n_classes,
        block_size=args.block_size,
        n_resamples=args.n_resamples,
        coverage=args.coverage,
        bootstrap_seed=args.bootstrap_seed,
        selection_metric=args.selection_metric,
    )

    output_json = args.output_json or (input_dir / "pooled_test_macro_f1.json")
    output_markdown = args.output_markdown or (input_dir / "pooled_test_macro_f1.md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps({"rows": [row.to_dict() for row in rows]}, indent=2)
    )
    output_markdown.write_text(_render_markdown(rows))
    print(f"[regime_pooled_aggregator] wrote {output_json} + {output_markdown}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
