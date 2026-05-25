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
from typing import Callable, Literal

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

# Phase 5 multi-run defaults. The redundancy guard threshold is the
# Cohen-kappa above which two components are treated as duplicate
# voters in the ensemble — keep it configurable rather than baked
# into a magic-number constant.
DEFAULT_REDUNDANCY_KAPPA_THRESHOLD = 0.85
DEFAULT_CONFORMAL_ALPHA = 0.2


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


# ---------------------------------------------------------------------------
# Phase 5 multi-run logit-average ensemble (#5 of the cross-bank work
# ladder). The single-run aggregator above selects one HP cell per
# architecture from a single sweep JSON; the multi-run path below
# pools across N distinct training runs (one run = one config x seed
# x architecture combination), averages logits per (fold, trial),
# and emits per-fold conformal prediction sets via the existing
# conformal helpers.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RunSpec:
    """One training-run participant in the Phase 5 multi-run ensemble.

    ``run_id`` is the manifest id used to disambiguate runs in logs and
    in the agreement-matrix output. ``architecture`` is the architecture
    family (``"transformer"``, ``"lstm"``, ``"tft"``, ...). ``encoder_alias``
    is the text-encoder family alias from ``backend/app/models/registry.yaml``
    when relevant (e.g., ``"bert_base"``, ``"finbert"``); for runs without
    a text encoder, set it to a stable placeholder like ``"none"``.
    ``seed`` is the training-time random seed. ``weight`` is the logit-
    average weight (default 1.0 = uniform); the aggregator normalises
    weights to sum to one before averaging.

    ``results_path`` is the absolute or relative path to the run's
    ``forecaster_sweep_results.json`` (or equivalent) blob. The loader
    uses the same JSON shape the single-run aggregator already consumes
    so no new on-disk contract is introduced.
    """

    run_id: str
    architecture: str
    encoder_alias: str
    seed: int
    results_path: str
    weight: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "architecture": self.architecture,
            "encoder_alias": self.encoder_alias,
            "seed": self.seed,
            "results_path": self.results_path,
            "weight": self.weight,
        }


@dataclasses.dataclass(frozen=True)
class MultiRunFoldResult:
    """Per-fold result emitted by :func:`aggregate_multi_run_ensemble`.

    ``coverage`` + ``avg_set_size`` come from the per-fold conformal
    wrapper, applied to the averaged softmax probabilities.
    """

    fold_id: str
    n_rows: int
    breakdown: ClassificationBreakdown
    coverage: float
    avg_set_size: float
    softmax_quantile: float

    def to_dict(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "n_rows": self.n_rows,
            "breakdown": self.breakdown.to_dict(),
            "coverage": self.coverage,
            "avg_set_size": self.avg_set_size,
            "softmax_quantile": self.softmax_quantile,
        }


@dataclasses.dataclass(frozen=True)
class MultiRunEnsembleResult:
    """Top-level Phase 5 ensemble result.

    ``kept_run_ids`` lists the run_ids that survived the redundancy
    guard. ``dropped_run_ids`` lists the ones dropped + the run_id they
    were redundant with. ``agreement`` is the pairwise Cohen-kappa
    matrix across the kept components.
    """

    kept_run_ids: tuple[str, ...]
    dropped_run_ids: tuple[tuple[str, str, float], ...]
    agreement: dict[tuple[str, str], float]
    per_fold: tuple[MultiRunFoldResult, ...]
    pooled_breakdown: ClassificationBreakdown
    calibration_strategy: str
    conformal_alpha: float

    def to_dict(self) -> dict[str, object]:
        return {
            "kept_run_ids": list(self.kept_run_ids),
            "dropped_run_ids": [
                {"dropped": d, "redundant_with": k, "kappa": float(kappa)}
                for d, k, kappa in self.dropped_run_ids
            ],
            "agreement": {
                f"{a}|{b}": float(v) for (a, b), v in self.agreement.items()
            },
            "per_fold": [row.to_dict() for row in self.per_fold],
            "pooled_breakdown": self.pooled_breakdown.to_dict(),
            "calibration_strategy": self.calibration_strategy,
            "conformal_alpha": self.conformal_alpha,
        }


def _load_run_trials(spec: RunSpec) -> list[Mapping[str, object]]:
    """Read the run's sweep JSON and return its trial list.

    Uses the same loader path as :func:`main` above so the multi-run
    aggregator inherits the single-run aggregator's on-disk contract.
    Trials without ``test_metrics.predictions`` / ``class_scores`` are
    skipped silently — the multi-run logic asserts coverage downstream.
    """

    path = Path(spec.results_path)
    if not path.exists():
        raise FileNotFoundError(
            f"run {spec.run_id!r} results not found at {path!s}"
        )
    blob = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(blob, Mapping):
        raise ValueError(
            f"run {spec.run_id!r} results blob is not a JSON object: {path!s}"
        )
    trials = blob.get("trials", [])
    if not isinstance(trials, list):
        raise ValueError(
            f"run {spec.run_id!r} results blob has no 'trials' list: {path!s}"
        )
    out: list[Mapping[str, object]] = []
    for trial in trials:
        if isinstance(trial, Mapping):
            out.append(trial)
    return out


def _trial_id(trial: Mapping[str, object]) -> tuple[str, int | None]:
    """Stable identifier for a single trial. The multi-run aggregator
    aligns trials across runs by ``(fold_id, seed)``; if seed is absent
    it falls through to ``None`` so the layout-mismatch guard fires
    fail-loud."""

    fold = trial.get("fold_id")
    seed = trial.get("seed")
    fold_key = str(fold) if fold is not None else ""
    seed_key = int(seed) if isinstance(seed, int) else None
    return fold_key, seed_key


def _softmax(logits: Sequence[float]) -> list[float]:
    """Numerically stable softmax for a single row of logits."""

    if not logits:
        return []
    m = max(float(x) for x in logits)
    exps = [math.exp(float(x) - m) for x in logits]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(logits)] * len(logits)
    return [e / total for e in exps]


def _row_logits(trial: Mapping[str, object]) -> list[list[float]] | None:
    """Pull per-row class scores as logits.

    The current training loop emits ``test_metrics.class_scores`` as
    softmax probabilities; we convert to logits via ``log(max(eps, p))``
    so the multi-run averaging is in log-space (equivalent to averaging
    log-probabilities). This matches the ``mean_logit`` strategy in
    the single-run aggregator above.
    """

    scores = _trial_class_scores(trial)
    if scores is None:
        return None
    eps = 1e-12
    return [[math.log(max(eps, float(p))) for p in row] for row in scores]


def _cohen_kappa(preds_a: Sequence[int], preds_b: Sequence[int]) -> float:
    """Cohen's kappa between two prediction sequences over the same rows.

    Defined as ``(p_o - p_e) / (1 - p_e)`` where ``p_o`` is observed
    agreement and ``p_e`` is the chance-agreement implied by each
    rater's marginal class distribution. Returns 1.0 on identical
    predictions; 0.0 when the chance-agreement saturates (i.e. both
    raters always predict the same single class).
    """

    if len(preds_a) != len(preds_b):
        raise ValueError(
            f"preds_a length {len(preds_a)} != preds_b length {len(preds_b)}"
        )
    n = len(preds_a)
    if n == 0:
        return float("nan")
    classes = sorted({int(x) for x in preds_a} | {int(x) for x in preds_b})
    if len(classes) <= 1:
        return 1.0 if all(int(a) == int(b) for a, b in zip(preds_a, preds_b)) else 0.0
    agree = sum(1 for a, b in zip(preds_a, preds_b) if int(a) == int(b))
    p_o = agree / n
    marg_a = {c: 0 for c in classes}
    marg_b = {c: 0 for c in classes}
    for a, b in zip(preds_a, preds_b):
        marg_a[int(a)] += 1
        marg_b[int(b)] += 1
    p_e = sum((marg_a[c] / n) * (marg_b[c] / n) for c in classes)
    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if p_o >= 1.0 - 1e-12 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def _validate_run_layouts(
    specs: Sequence[RunSpec],
    trials_by_run: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[tuple[tuple[str, int | None], ...], dict[str, dict[tuple[str, int | None], Mapping[str, object]]]]:
    """Confirm every run has the same ``(fold_id, seed)`` trial set.

    Returns the canonical trial-id sequence (sorted, derived from the
    first run) and a per-run lookup keyed by trial id. Raises
    ``ValueError`` on any mismatch — fail-loud per the deliverables.
    """

    if not specs:
        raise ValueError("multi-run ensemble requires at least one RunSpec")
    canonical: list[tuple[str, int | None]] | None = None
    lookup: dict[
        str, dict[tuple[str, int | None], Mapping[str, object]]
    ] = {}
    for spec in specs:
        trials = trials_by_run.get(spec.run_id, [])
        per_run: dict[tuple[str, int | None], Mapping[str, object]] = {}
        for trial in trials:
            tid = _trial_id(trial)
            per_run[tid] = trial
        lookup[spec.run_id] = per_run
        ids = sorted(per_run.keys())
        if canonical is None:
            canonical = ids
        elif ids != canonical:
            missing = sorted(set(canonical) - set(ids))
            extra = sorted(set(ids) - set(canonical))
            raise ValueError(
                f"run {spec.run_id!r} fold/trial layout mismatch: "
                f"missing={missing!r}, extra={extra!r}"
            )
    if canonical is None:
        canonical = []
    return tuple(canonical), lookup


def _agreement_matrix(
    run_ids: Sequence[str],
    preds_by_run: Mapping[str, Sequence[int]],
) -> dict[tuple[str, str], float]:
    """Pairwise Cohen-kappa across run pairs.

    Symmetric: the matrix records both ``(a, b)`` and ``(b, a)`` so
    callers can drop either direction without re-checking ordering.
    The diagonal carries 1.0.
    """

    out: dict[tuple[str, str], float] = {}
    for i, a in enumerate(run_ids):
        out[(a, a)] = 1.0
        for b in run_ids[i + 1 :]:
            kappa = _cohen_kappa(preds_by_run[a], preds_by_run[b])
            out[(a, b)] = kappa
            out[(b, a)] = kappa
    return out


def per_fold_conformal_calibration(
    fold_softmax: Sequence[Sequence[float]],
    fold_targets: Sequence[int],
    *,
    conformal_alpha: float = DEFAULT_CONFORMAL_ALPHA,
) -> tuple[float, float, float]:
    """Per-fold conformal calibration on averaged softmax probabilities.

    Wires through the existing
    :func:`app.evaluation.conformal.calibrate_classification_conformal` +
    :func:`predict_conformal_set` +
    :func:`empirical_classification_coverage` helpers introduced in
    PR #279. The fold's own scores fit the threshold and feed the
    coverage check; no global calibration tier is used (calibration
    leak guard path (b) in the Phase 5 deliverables).

    Returns ``(softmax_quantile, coverage, avg_set_size)``. Any of
    the three may be ``nan`` when the fold has too few rows for the
    finite-sample-corrected quantile (the underlying helper raises
    ``ValueError`` on an empty conformity-score set).
    """

    from app.evaluation.conformal import (
        calibrate_classification_conformal,
        empirical_classification_coverage,
        predict_conformal_set,
    )

    try:
        softmax_q = calibrate_classification_conformal(
            softmax_scores=fold_softmax,
            true_classes=fold_targets,
            alpha=conformal_alpha,
        )
    except ValueError:
        return float("nan"), float("nan"), float("nan")
    if not math.isfinite(softmax_q):
        return float("nan"), float("nan"), float("nan")
    sets = [predict_conformal_set(row, softmax_q) for row in fold_softmax]
    coverage = empirical_classification_coverage(sets, fold_targets)
    avg_set_size = (
        sum(len(s) for s in sets) / len(sets) if sets else float("nan")
    )
    return float(softmax_q), float(coverage), float(avg_set_size)


def _drop_redundant_runs(
    run_ids: Sequence[str],
    agreement: Mapping[tuple[str, str], float],
    *,
    threshold: float,
    log: Callable[[str], None],
) -> tuple[tuple[str, ...], tuple[tuple[str, str, float], ...]]:
    """Drop runs whose pairwise kappa with any kept run exceeds threshold.

    Greedy left-to-right pass: the first run is always kept; each
    subsequent run is dropped if any already-kept run has agreement
    above the threshold. The dropped run is logged with the kept run
    id it was redundant with and the kappa value.
    """

    kept: list[str] = []
    dropped: list[tuple[str, str, float]] = []
    for run_id in run_ids:
        redundant_with: str | None = None
        kappa_with: float = float("nan")
        for k in kept:
            kappa = float(agreement.get((run_id, k), 0.0))
            if kappa > threshold:
                redundant_with = k
                kappa_with = kappa
                break
        if redundant_with is None:
            kept.append(run_id)
        else:
            dropped.append((run_id, redundant_with, kappa_with))
            log(
                f"[ensemble_aggregator] dropping run {run_id!r} "
                f"(kappa={kappa_with:.3f} with kept run "
                f"{redundant_with!r} exceeds threshold {threshold:.3f})"
            )
    return tuple(kept), tuple(dropped)


def aggregate_multi_run_ensemble(  # noqa: C901, PLR0913
    run_specs: Sequence[RunSpec],
    *,
    ground_truth_loader: Callable[[RunSpec], list[Mapping[str, object]]] | None = None,
    calibration_strategy: Literal["conformal_per_fold"] = "conformal_per_fold",
    redundancy_kappa_threshold: float = DEFAULT_REDUNDANCY_KAPPA_THRESHOLD,
    conformal_alpha: float = DEFAULT_CONFORMAL_ALPHA,
    n_classes: int = 3,
    log: Callable[[str], None] | None = None,
) -> MultiRunEnsembleResult:
    """Phase 5 calibrated logit-average ensemble across N training runs.

    See module docstring + ``RunSpec`` for the input contract. The
    aggregator:

    1. Loads each run's per-fold per-trial logits via
       ``ground_truth_loader`` (defaults to reading the spec's
       ``results_path`` and parsing the standard sweep JSON).
    2. Validates that all runs share the same ``(fold_id, seed)``
       trial layout — raises ``ValueError`` on any mismatch.
    3. Computes pairwise Cohen-kappa across runs and drops any run
       whose agreement with an already-kept run exceeds
       ``redundancy_kappa_threshold``.
    4. Averages logits across the kept runs per (fold, trial) using
       the spec weights (normalised to sum to one).
    5. Fits a per-fold conformal threshold on the averaged softmax
       probabilities and emits per-fold coverage + average set size.
    6. Pools predictions across folds for a single headline macro-F1.

    The conformal path reuses the per-fold APS calibration the
    existing single-run wrapper applies, so the calibration carries
    the same coverage guarantee as the headline classifier (modulo
    the exchangeability assumption holding across the averaged-run
    logits, which is the standard split-conformal claim).
    """

    if calibration_strategy != "conformal_per_fold":
        raise ValueError(
            f"unsupported calibration_strategy: {calibration_strategy!r}; "
            "Phase 5 ships only the per-fold conformal path"
        )
    if not run_specs:
        raise ValueError("aggregate_multi_run_ensemble requires >=1 RunSpec")

    log_fn = log if log is not None else print

    loader = ground_truth_loader if ground_truth_loader is not None else _load_run_trials
    trials_by_run = {spec.run_id: list(loader(spec)) for spec in run_specs}

    canonical_ids, lookup = _validate_run_layouts(run_specs, trials_by_run)
    if not canonical_ids:
        raise ValueError(
            "multi-run ensemble received an empty trial layout — every "
            "run has zero trials; nothing to aggregate"
        )

    # Step: assemble per-run argmax predictions so we can compute the
    # pairwise agreement matrix before deciding who to keep.
    preds_by_run: dict[str, list[int]] = {}
    for spec in run_specs:
        per_run = lookup[spec.run_id]
        flat: list[int] = []
        for tid in canonical_ids:
            trial = per_run[tid]
            rows = _test_predictions(trial)
            if rows is None:
                raise ValueError(
                    f"run {spec.run_id!r} trial {tid!r} missing "
                    "test_metrics.predictions/targets"
                )
            preds, _targets = rows
            flat.extend(preds)
        preds_by_run[spec.run_id] = flat

    run_ids_in_order = [spec.run_id for spec in run_specs]
    agreement = _agreement_matrix(run_ids_in_order, preds_by_run)
    kept, dropped = _drop_redundant_runs(
        run_ids_in_order,
        agreement,
        threshold=redundancy_kappa_threshold,
        log=log_fn,
    )

    kept_set = set(kept)
    kept_specs = [spec for spec in run_specs if spec.run_id in kept_set]

    # Normalise weights across kept specs.
    raw_weights = [float(spec.weight) for spec in kept_specs]
    total_w = sum(raw_weights)
    if total_w <= 0:
        raise ValueError(
            "kept runs have non-positive weight sum; cannot normalise"
        )
    weights = [w / total_w for w in raw_weights]

    # Step: per-fold logit averaging + conformal calibration.
    per_fold_results: list[MultiRunFoldResult] = []
    pooled_preds: list[int] = []
    pooled_targets: list[int] = []

    # Group canonical_ids by fold so the conformal calibration is per-
    # fold by construction (the leak-guard requirement in the deliverables).
    fold_buckets: dict[str, list[tuple[str, int | None]]] = defaultdict(list)
    for tid in canonical_ids:
        fold_buckets[tid[0]].append(tid)

    for fold_id, fold_tids in sorted(fold_buckets.items()):
        fold_preds: list[int] = []
        fold_targets: list[int] = []
        fold_softmax: list[list[float]] = []
        for tid in fold_tids:
            # Reference targets from the first kept spec; the layout
            # validator already confirmed alignment across runs, but
            # do a defensive equality check on targets per trial to
            # catch row-ordering drift inside a single trial.
            ref_rows = _test_predictions(lookup[kept_specs[0].run_id][tid])
            if ref_rows is None:
                raise ValueError(
                    f"kept run {kept_specs[0].run_id!r} missing test_metrics "
                    f"on trial {tid!r}"
                )
            _ref_preds, ref_targets = ref_rows
            n_rows_trial = len(ref_targets)

            # Per-row aggregated logits across kept runs.
            avg_logits: list[list[float]] = [
                [0.0] * n_classes for _ in range(n_rows_trial)
            ]
            for spec, w in zip(kept_specs, weights):
                logits = _row_logits(lookup[spec.run_id][tid])
                if logits is None:
                    raise ValueError(
                        f"run {spec.run_id!r} trial {tid!r} missing "
                        "test_metrics.class_scores"
                    )
                if len(logits) != n_rows_trial:
                    raise ValueError(
                        f"run {spec.run_id!r} trial {tid!r} class_scores "
                        f"length {len(logits)} != targets length {n_rows_trial}"
                    )
                _, ref_targets_b = _test_predictions(lookup[spec.run_id][tid])  # type: ignore[misc]
                if ref_targets_b != ref_targets:
                    raise ValueError(
                        f"run {spec.run_id!r} trial {tid!r} target sequence "
                        "differs from reference run; row ordering drifted"
                    )
                for i, row in enumerate(logits):
                    if len(row) != n_classes:
                        raise ValueError(
                            f"run {spec.run_id!r} trial {tid!r} row {i} has "
                            f"{len(row)} classes; expected {n_classes}"
                        )
                    for c in range(n_classes):
                        avg_logits[i][c] += w * float(row[c])

            # Softmax once at the end.
            for row in avg_logits:
                probs = _softmax(row)
                fold_softmax.append(probs)
                fold_preds.append(
                    max(range(n_classes), key=lambda c: probs[c])
                )
            fold_targets.extend(int(y) for y in ref_targets)

        # Per-fold conformal calibration on the averaged softmax. This
        # is the leak-safe path: the averaged logits at fold F never
        # touched fold F training, because each component trained on
        # its own walk-forward partition; the conformal threshold is
        # fit on the fold's own held-out scores rather than a global
        # calibration tier.
        softmax_q, coverage, avg_set_size = per_fold_conformal_calibration(
            fold_softmax,
            fold_targets,
            conformal_alpha=conformal_alpha,
        )

        breakdown = compute_classification_breakdown(
            predictions=fold_preds,
            targets=fold_targets,
            n_classes=n_classes,
        )
        per_fold_results.append(
            MultiRunFoldResult(
                fold_id=fold_id,
                n_rows=len(fold_preds),
                breakdown=breakdown,
                coverage=float(coverage),
                avg_set_size=float(avg_set_size),
                softmax_quantile=float(softmax_q),
            )
        )
        pooled_preds.extend(fold_preds)
        pooled_targets.extend(fold_targets)

    pooled_breakdown = compute_classification_breakdown(
        predictions=pooled_preds,
        targets=pooled_targets,
        n_classes=n_classes,
    )

    return MultiRunEnsembleResult(
        kept_run_ids=kept,
        dropped_run_ids=dropped,
        agreement=agreement,
        per_fold=tuple(per_fold_results),
        pooled_breakdown=pooled_breakdown,
        calibration_strategy=calibration_strategy,
        conformal_alpha=float(conformal_alpha),
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
