"""Multi-architecture ensemble aggregator.

Reads N per-architecture sweep JSONs (one per architecture under a shared
training package), aligns the test-partition predictions on each
``(fold_id, seed)`` cell, and reports macro-F1 for three classical
ensemble strategies — **mean-logit / mean-softmax / plurality-vote**.

Inputs to the aggregator are the same ``forecaster_sweep_results.json``
files :mod:`app.evaluation.regime_pooled_aggregator` consumes. Trials
must carry ``test_metrics.predictions``, ``test_metrics.targets``, and
``test_metrics.class_scores`` (PR #226 wired all three onto
:class:`EvaluationMetrics`).

The cell alignment contract: trials from different architecture JSONs
are matched by ``(fold_id, seed)``. The per-architecture best HP cell
is selected first via the same selection-metric path as
:mod:`regime_pooled_aggregator`, then the selected trials from each
architecture are aligned and averaged.

Three strategies:

1. **mean-logit** -- average ``log(class_score)`` element-wise across
   architectures, take ``argmax``. Treats each model as a calibrated
   logit producer; reduces to per-class log-prob averaging.
2. **mean-softmax** -- average ``class_scores`` (already softmax
   probabilities) element-wise, take ``argmax``. The default for
   uncalibrated heads with overlapping class supports.
3. **plurality-vote** -- per-row majority vote across architectures'
   argmax predictions. Ties broken by lowest class index.

Macro-F1 reported with the same block-bootstrap CI surface as the
pooled-fold aggregator, so the two outputs sit cleanly next to each
other in the report appendix.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from app.evaluation.bootstrap import BootstrapCI
from app.evaluation.classification_breakdown import (
    ClassificationBreakdown,
    compute_classification_breakdown,
)
from app.evaluation.regime_pooled_aggregator import (
    _cell_key,
    _select_best_cell,
    _test_predictions,
    _trial_summary,
)


_STRATEGIES = ("mean_logit", "mean_softmax", "plurality_vote")


@dataclasses.dataclass(frozen=True)
class EnsembleCell:
    strategy: str
    architectures: tuple[str, ...]
    fold_id: str | None
    seed: int | None
    n_rows: int
    breakdown: ClassificationBreakdown

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "architectures": list(self.architectures),
            "fold_id": self.fold_id,
            "seed": self.seed,
            "n_rows": self.n_rows,
            "breakdown": self.breakdown.to_dict(),
        }


@dataclasses.dataclass(frozen=True)
class EnsemblePooled:
    strategy: str
    architectures: tuple[str, ...]
    n_pooled: int
    breakdown: ClassificationBreakdown
    macro_f1_ci: BootstrapCI

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "architectures": list(self.architectures),
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


def _trial_class_scores(trial: Mapping[str, object]) -> list[list[float]] | None:
    summary = _trial_summary(trial)
    test_metrics = summary.get("test_metrics")
    if not isinstance(test_metrics, Mapping):
        return None
    scores = test_metrics.get("class_scores")
    if not isinstance(scores, list) or not scores:
        return None
    rows: list[list[float]] = []
    for row in scores:
        if not isinstance(row, list):
            return None
        rows.append([float(x) for x in row])
    return rows


def _mean_logit(per_arch_scores: Sequence[Sequence[float]]) -> int:
    eps = 1e-12
    n_classes = len(per_arch_scores[0])
    log_sums = [0.0] * n_classes
    for scores in per_arch_scores:
        for c, p in enumerate(scores):
            log_sums[c] += math.log(max(eps, float(p)))
    best = 0
    best_val = log_sums[0]
    for c in range(1, n_classes):
        if log_sums[c] > best_val:
            best = c
            best_val = log_sums[c]
    return best


def _mean_softmax(per_arch_scores: Sequence[Sequence[float]]) -> int:
    n_classes = len(per_arch_scores[0])
    sums = [0.0] * n_classes
    for scores in per_arch_scores:
        for c, p in enumerate(scores):
            sums[c] += float(p)
    best = 0
    best_val = sums[0]
    for c in range(1, n_classes):
        if sums[c] > best_val:
            best = c
            best_val = sums[c]
    return best


def _plurality_vote(per_arch_argmaxes: Sequence[int]) -> int:
    counter = Counter(per_arch_argmaxes)
    # Break ties by lowest class index.
    best_count = max(counter.values())
    return min(c for c, n in counter.items() if n == best_count)


def _align_per_cell(
    selected_trials_per_arch: Mapping[str, Sequence[Mapping[str, object]]]
) -> dict[tuple[str | None, int | None], dict[str, Mapping[str, object]]]:
    """Group selected trials by ``(fold_id, seed)`` so the ensemble
    averages run only on cells where every architecture has data.
    Cells with incomplete coverage are dropped."""

    by_cell: dict[tuple[str | None, int | None], dict[str, Mapping[str, object]]] = (
        defaultdict(dict)
    )
    for arch, trials in selected_trials_per_arch.items():
        for trial in trials:
            fold = trial.get("fold_id")
            seed_value = trial.get("seed")
            fold_key = fold if isinstance(fold, str) else None
            seed_key = int(seed_value) if isinstance(seed_value, int) else None
            by_cell[(fold_key, seed_key)][arch] = trial
    return by_cell


def _ensemble_predictions_for_cell(  # noqa: C901
    arch_trials: Mapping[str, Mapping[str, object]],
    strategy: str,
) -> tuple[list[int], list[int]] | None:
    """For a single (fold, seed) cell shared by every architecture,
    produce the ensemble's predictions + the (shared) targets."""

    if not arch_trials:
        return None

    arch_payloads: dict[str, tuple[list[int], list[int], list[list[float]] | None]] = {}
    n_rows: int | None = None
    pooled_targets: list[int] | None = None
    for arch, trial in arch_trials.items():
        rows = _test_predictions(trial)
        if rows is None:
            return None
        preds, targets = rows
        if pooled_targets is None:
            pooled_targets = targets
            n_rows = len(targets)
        elif targets != pooled_targets:
            # If two architectures disagree on the target sequence
            # ordering for the same (fold, seed) cell, the underlying
            # event ordering has drifted and the ensemble alignment is
            # unsafe. Skip the cell.
            return None
        scores = _trial_class_scores(trial)
        arch_payloads[arch] = (preds, targets, scores)

    if pooled_targets is None or n_rows is None:
        return None

    ensemble_preds: list[int] = []
    for i in range(n_rows):
        per_arch_argmaxes: list[int] = []
        per_arch_scores: list[list[float]] = []
        for _arch, (preds, _targets, scores) in arch_payloads.items():
            per_arch_argmaxes.append(int(preds[i]))
            if scores is not None and i < len(scores):
                per_arch_scores.append(scores[i])
        if strategy == "plurality_vote":
            ensemble_preds.append(_plurality_vote(per_arch_argmaxes))
        elif strategy == "mean_logit":
            if not per_arch_scores:
                # Fall back to plurality when scores are unavailable.
                ensemble_preds.append(_plurality_vote(per_arch_argmaxes))
            else:
                ensemble_preds.append(_mean_logit(per_arch_scores))
        elif strategy == "mean_softmax":
            if not per_arch_scores:
                ensemble_preds.append(_plurality_vote(per_arch_argmaxes))
            else:
                ensemble_preds.append(_mean_softmax(per_arch_scores))
        else:
            raise ValueError(f"unknown ensemble strategy: {strategy!r}")
    return ensemble_preds, list(pooled_targets)


def aggregate(  # noqa: C901, PLR0913
    per_arch_sweep_blobs: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    strategies: Sequence[str] = _STRATEGIES,
    n_classes: int = 3,
    block_size: int = 20,
    n_resamples: int = 1000,
    coverage: float = 0.95,
    bootstrap_seed: int = 11,
    selection_metric: str | None = None,
) -> dict[str, list[EnsembleCell] | list[EnsemblePooled]]:
    """Compute per-cell + all-cells-pooled ensemble macro-F1 for each
    strategy. Returns a dict with keys ``per_cell`` (list) and
    ``pooled`` (list)."""

    # Step 1: select the best (arch, hp) from each architecture's sweep.
    per_arch_best_trials: dict[str, list[Mapping[str, object]]] = {}
    for arch, blobs in per_arch_sweep_blobs.items():
        cells: dict[tuple[str, str | None], list[Mapping[str, object]]] = defaultdict(
            list
        )
        inferred_metric = "macro_f1"
        for blob in blobs:
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
        if not cells:
            continue
        chosen = _select_best_cell(cells, selection_metric or inferred_metric)
        if arch in chosen:
            per_arch_best_trials[arch] = list(cells[chosen[arch]])

    if len(per_arch_best_trials) < 2:
        raise ValueError(
            "ensemble aggregator requires at least 2 architectures with "
            f"selectable cells; got {sorted(per_arch_best_trials.keys())!r}"
        )

    aligned = _align_per_cell(per_arch_best_trials)

    architectures = tuple(sorted(per_arch_best_trials.keys()))

    per_cell_rows: list[EnsembleCell] = []
    pooled_rows: list[EnsemblePooled] = []
    for strategy in strategies:
        all_preds: list[int] = []
        all_targets: list[int] = []
        for (fold_id, seed), arch_trials in sorted(
            aligned.items(),
            key=lambda kv: ((kv[0][0] or ""), (kv[0][1] or -1)),
        ):
            if set(arch_trials.keys()) != set(architectures):
                # Skip cells where some architecture is missing.
                continue
            payload = _ensemble_predictions_for_cell(arch_trials, strategy)
            if payload is None:
                continue
            ens_preds, targets = payload
            cell_breakdown = compute_classification_breakdown(
                predictions=ens_preds,
                targets=targets,
                n_classes=n_classes,
            )
            per_cell_rows.append(
                EnsembleCell(
                    strategy=strategy,
                    architectures=architectures,
                    fold_id=fold_id,
                    seed=seed,
                    n_rows=len(targets),
                    breakdown=cell_breakdown,
                )
            )
            all_preds.extend(ens_preds)
            all_targets.extend(targets)

        pooled_breakdown = compute_classification_breakdown(
            predictions=all_preds,
            targets=all_targets,
            n_classes=n_classes,
        )
        # Block-bootstrap CI on the pooled ensemble macro-F1. Same logic
        # as :func:`regime_pooled_aggregator.pool_cell` for parity in the
        # report appendix.
        import random

        rng = random.Random(bootstrap_seed)
        resampled: list[float] = []
        n = len(all_preds)
        if n == 0:
            ci = BootstrapCI(
                point=float("nan"),
                lo=float("nan"),
                hi=float("nan"),
                coverage=coverage,
                n_resamples=n_resamples,
                block_size=block_size,
            )
        else:
            for _ in range(n_resamples):
                idx: list[int] = []
                n_blocks = max(1, (n + block_size - 1) // block_size)
                for _ in range(n_blocks):
                    start = rng.randint(0, max(0, n - block_size))
                    idx.extend(range(start, min(n, start + block_size)))
                idx = idx[:n]
                rs_preds = [all_preds[i] for i in idx]
                rs_targets = [all_targets[i] for i in idx]
                rs_breakdown = compute_classification_breakdown(
                    predictions=rs_preds,
                    targets=rs_targets,
                    n_classes=n_classes,
                )
                resampled.append(float(rs_breakdown.macro_f1))
            resampled.sort()
            alpha = (1.0 - coverage) / 2.0
            lo_idx = max(0, min(n_resamples - 1, int(alpha * n_resamples)))
            hi_idx = max(
                0, min(n_resamples - 1, int((1.0 - alpha) * n_resamples) - 1)
            )
            ci = BootstrapCI(
                point=float(pooled_breakdown.macro_f1),
                lo=resampled[lo_idx],
                hi=resampled[hi_idx],
                coverage=coverage,
                n_resamples=n_resamples,
                block_size=block_size,
            )

        pooled_rows.append(
            EnsemblePooled(
                strategy=strategy,
                architectures=architectures,
                n_pooled=len(all_preds),
                breakdown=pooled_breakdown,
                macro_f1_ci=ci,
            )
        )

    return {"per_cell": per_cell_rows, "pooled": pooled_rows}


def _render_markdown(payload: Mapping[str, object]) -> str:
    lines: list[str] = []
    lines.append("# Multi-architecture ensemble macro-F1")
    lines.append("")
    pooled = payload.get("pooled", [])
    architectures: tuple[str, ...] = ()
    if isinstance(pooled, list) and pooled:
        first = pooled[0]
        if isinstance(first, EnsemblePooled):
            architectures = first.architectures
    if architectures:
        lines.append(
            "Ensembling architectures: " + ", ".join(f"`{a}`" for a in architectures)
        )
        lines.append("")
    lines.append("| Strategy | n_pooled | macro-F1 | 95% CI |")
    lines.append("| --- | ---: | ---: | --- |")
    if isinstance(pooled, list):
        for row in pooled:
            if isinstance(row, EnsemblePooled):
                ci = row.macro_f1_ci
                lines.append(
                    f"| {row.strategy} | {row.n_pooled} | {ci.point:.4f} | "
                    f"[{ci.lo:.4f}, {ci.hi:.4f}] |"
                )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute multi-architecture ensemble macro-F1 (mean-logit, "
            "mean-softmax, plurality-vote) by aligning test-partition "
            "predictions on each (fold_id, seed) cell from per-architecture "
            "sweep JSONs."
        )
    )
    parser.add_argument(
        "--arch-sweep-dir",
        type=Path,
        required=True,
        help=(
            "Root directory with one subdirectory per architecture, each "
            "containing a ``forecaster_sweep_results.json``. The directory "
            "layout produced by ``scripts/run_regime_architecture_sweep.py``."
        ),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=11)
    parser.add_argument(
        "--selection-metric",
        default=None,
        help="Per-architecture HP selection metric; defaults to the sweep JSON's value.",
    )
    args = parser.parse_args(argv)

    root: Path = args.arch_sweep_dir
    if not root.exists():
        raise SystemExit(f"arch sweep dir not found: {root}")

    per_arch_blobs: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for json_path in sorted(root.rglob("forecaster_sweep_results.json")):
        arch = json_path.parent.name
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                blob = json.load(fh)
            if isinstance(blob, Mapping):
                per_arch_blobs[arch].append(blob)
        except json.JSONDecodeError as exc:
            print(f"[ensemble_aggregator] skipping {json_path}: {exc}")

    if not per_arch_blobs:
        raise SystemExit(f"no forecaster_sweep_results.json under {root}")

    result = aggregate(
        per_arch_blobs,
        n_classes=args.n_classes,
        block_size=args.block_size,
        n_resamples=args.n_resamples,
        coverage=args.coverage,
        bootstrap_seed=args.bootstrap_seed,
        selection_metric=args.selection_metric,
    )

    output_json = args.output_json or (root / "ensemble_results.json")
    output_markdown = args.output_markdown or (root / "ensemble_results.md")
    output_json.write_text(
        json.dumps(
            {
                "per_cell": [row.to_dict() for row in result.get("per_cell", [])],
                "pooled": [row.to_dict() for row in result.get("pooled", [])],
            },
            indent=2,
        )
    )
    output_markdown.write_text(_render_markdown(result))
    print(f"[ensemble_aggregator] wrote {output_json} + {output_markdown}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
