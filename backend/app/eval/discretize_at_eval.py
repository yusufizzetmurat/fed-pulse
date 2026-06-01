"""Discretize-at-eval companion metric for the dual-head defense (#498).

Reads dual_head_comparison_*.json sweep artefacts, locates the regression
head's per-cell payload, and -- when per-event log(RV) predictions and
the train-fitted tertile bin edges are present -- buckets the
predictions into {0,1,2} class indices and recomputes macro-F1 against
the ground-truth labels. The bucketed F1 is the "regression-discretized"
arm of the three-way comparison:

  classification F1  vs  regression-discretized F1  vs  dual F1

When all three arms are populated per (seed, fold_id), a paired
Wilcoxon signed-rank between regression-discretized and dual tests the
null "post-hoc bucketing matches the dual head".

The current dual_head_comparison runner persists summary stats
(``regression_rmse_log_rv``, ``regression_mae_log_rv``) per cell but
does NOT persist per-event ``regression_predictions`` /
``regression_targets`` / ``bin_edges``. Cells lacking those fields emit
a structured warning and are skipped, with ``runner_extension_required``
flagged in the output payload. The bucketing math here is wired so a
future runner extension that records those per-event arrays surfaces a
populated table without any further changes here.

CLI::

    python -m app.eval.discretize_at_eval \\
        --sweep-artefact backend/artifacts/experiments/dual_head_comparison_canonical.json \\
        --output backend/artifacts/experiments/discretize_at_eval.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import warnings
from pathlib import Path
from typing import Any, Sequence

from app.config import BACKEND_ROOT
from app.eval.paired_comparisons import (
    block_bootstrap_ci_deltas,
    effect_size,
    wilcoxon_signed_rank,
)
from app.evaluation.classification_breakdown import compute_classification_breakdown

_N_CLASSES = 3
_DEFAULT_ARTEFACT = (
    BACKEND_ROOT / "artifacts" / "experiments" / "dual_head_comparison_canonical.json"
)
_DEFAULT_OUTPUT = BACKEND_ROOT / "artifacts" / "experiments" / "discretize_at_eval.json"


def bucketize(predictions: Sequence[float], bin_edges: Sequence[float]) -> list[int]:
    """Map continuous predictions to class indices using ``bin_edges`` cutoffs.

    Cutoff semantics match ``app.training.loaders.vol_regime_class_for``:
    a value below cutoff[i] lands in class i; a value at or above the
    last cutoff lands in class ``len(bin_edges)``. For a 3-class
    classifier ``bin_edges`` is the two-element list ``[t_1, t_2]``.
    """
    out: list[int] = []
    for v in predictions:
        cls = len(bin_edges)
        for i, cutoff in enumerate(bin_edges):
            if v < cutoff:
                cls = i
                break
        out.append(cls)
    return out


def discretized_macro_f1(
    predictions: Sequence[float],
    targets: Sequence[int],
    bin_edges: Sequence[float],
) -> float:
    """Bucket regression predictions then compute macro-F1 vs targets."""
    pred_classes = bucketize(predictions, bin_edges)
    bd = compute_classification_breakdown(
        predictions=pred_classes,
        targets=list(targets),
        n_classes=_N_CLASSES,
    )
    return float(bd.macro_f1)


@dataclasses.dataclass(frozen=True)
class DiscretizedCell:
    head_mode: str
    seed: int
    fold_id: str
    n_events: int
    f1_macro_discretized: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _extract_classification_cells(
    payload: dict[str, Any],
    head_mode: str,
    metric: str,
) -> dict[tuple[int, str], float]:
    cells: dict[tuple[int, str], float] = {}
    mode_trials: list[Any] = (payload.get("trials") or {}).get(head_mode) or []
    for trial in mode_trials:
        seed = int(trial.get("seed", -1))
        for fold in trial.get("folds") or []:
            fid = str(fold.get("fold_id", ""))
            v = (fold.get("metrics") or {}).get(metric)
            if v is not None and seed >= 0:
                cells[(seed, fid)] = float(v)
    return cells


def extract_discretized_cells(
    sweep_path: Path,
    payload: dict[str, Any] | None = None,
) -> tuple[list[DiscretizedCell], int]:
    """Pull DiscretizedCell rows from the regression arm of a sweep artefact.

    Returns ``(cells, n_skipped)``. Cells without per-event
    ``regression_predictions`` + ``regression_targets`` + ``bin_edges``
    are skipped with a structured warning and counted in ``n_skipped``;
    they correspond to the current artefact format which only persists
    aggregate RMSE/MAE.

    ``payload`` optionally lets the caller skip the file read when it
    has already parsed the JSON; useful inside batch loops where the
    same file is consumed twice.
    """
    if payload is None:
        payload = json.loads(sweep_path.read_text(encoding="utf-8"))
    mode_trials: list[Any] = (payload.get("trials") or {}).get("regression") or []
    cells: list[DiscretizedCell] = []
    n_skipped = 0
    for trial in mode_trials:
        seed = int(trial.get("seed", -1))
        for fold in trial.get("folds") or []:
            fold_id = str(fold.get("fold_id", ""))
            metrics = fold.get("metrics") or {}
            preds = metrics.get("regression_predictions")
            targets = metrics.get("regression_targets")
            edges = metrics.get("bin_edges") or fold.get("bin_edges")
            if preds is None or targets is None or edges is None:
                n_skipped += 1
                warnings.warn(
                    f"[discretize_at_eval] {sweep_path.name} "
                    f"head=regression seed={seed} fold={fold_id}: "
                    "missing per-event regression_predictions / "
                    "regression_targets / bin_edges; skipping "
                    "(runner extension required)",
                    stacklevel=2,
                )
                continue
            if len(preds) != len(targets):
                n_skipped += 1
                warnings.warn(
                    f"[discretize_at_eval] {sweep_path.name} "
                    f"head=regression seed={seed} fold={fold_id}: "
                    f"length mismatch preds={len(preds)} targets={len(targets)};"
                    " skipping",
                    stacklevel=2,
                )
                continue
            f1 = discretized_macro_f1(
                [float(x) for x in preds],
                [int(t) for t in targets],
                [float(e) for e in edges],
            )
            cells.append(
                DiscretizedCell(
                    head_mode="regression_discretized",
                    seed=seed,
                    fold_id=fold_id,
                    n_events=len(preds),
                    f1_macro_discretized=f1,
                )
            )
    return cells, n_skipped


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mu = sum(values) / n
    if n == 1:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    return mu, math.sqrt(var)


@dataclasses.dataclass(frozen=True)
class ArmSummary:
    arm: str
    n_cells: int
    f1_mean: float
    f1_std: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class PairedTest:
    label_a: str
    label_b: str
    n_pairs: int
    mean_delta: float
    std_delta: float
    wilcoxon_stat: float
    p_value: float
    effect_size_d: float
    ci_lo: float
    ci_hi: float
    ci_coverage: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _paired(  # noqa: PLR0913 -- bootstrap knobs surface as named kwargs by design
    cells_a: dict[tuple[int, str], float],
    cells_b: dict[tuple[int, str], float],
    label_a: str,
    label_b: str,
    *,
    coverage: float,
    n_resamples: int,
    bootstrap_seed: int,
) -> PairedTest:
    shared = sorted(set(cells_a) & set(cells_b), key=lambda k: (k[1], k[0]))
    deltas = [cells_a[k] - cells_b[k] for k in shared]
    fold_labels = [k[1] for k in shared]
    n = len(deltas)
    if n == 0:
        return PairedTest(
            label_a=label_a,
            label_b=label_b,
            n_pairs=0,
            mean_delta=float("nan"),
            std_delta=float("nan"),
            wilcoxon_stat=float("nan"),
            p_value=float("nan"),
            effect_size_d=float("nan"),
            ci_lo=float("nan"),
            ci_hi=float("nan"),
            ci_coverage=coverage,
        )
    mu = sum(deltas) / n
    var = sum((d - mu) ** 2 for d in deltas) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    stat, pval = wilcoxon_signed_rank(deltas)
    es = effect_size(deltas)
    _, lo, hi = block_bootstrap_ci_deltas(
        deltas,
        fold_labels,
        n_resamples=n_resamples,
        coverage=coverage,
        seed=bootstrap_seed,
    )
    return PairedTest(
        label_a=label_a,
        label_b=label_b,
        n_pairs=n,
        mean_delta=mu,
        std_delta=sd,
        wilcoxon_stat=stat,
        p_value=pval,
        effect_size_d=es,
        ci_lo=lo,
        ci_hi=hi,
        ci_coverage=coverage,
    )


def compute_discretize_at_eval(
    sweep_paths: Sequence[Path],
    *,
    n_resamples: int = 1000,
    coverage: float = 0.95,
    bootstrap_seed: int = 11,
) -> dict[str, Any]:
    """Build the three-way comparison table from a list of sweep artefacts.

    The output payload carries:
      - ``arm_summaries``: mean ± std macro-F1 per arm
      - ``paired_tests``: regression_discretized vs dual + regression_discretized
        vs classification, paired Wilcoxon
      - ``runner_extension_required``: True when zero regression cells carried
        per-event predictions (current artefact format)
    """
    all_disc_cells: list[DiscretizedCell] = []
    disc_by_cell: dict[tuple[int, str], float] = {}
    cls_by_cell: dict[tuple[int, str], float] = {}
    dual_by_cell: dict[tuple[int, str], float] = {}
    skipped_total = 0
    for sweep_path in sweep_paths:
        payload = json.loads(sweep_path.read_text(encoding="utf-8"))
        cells, n_skipped = extract_discretized_cells(sweep_path, payload=payload)
        skipped_total += n_skipped
        # Merge across artefacts: last writer wins on key collision
        # uniformly across all three arms (disc / cls / dual). The
        # caller pins the canonical artefact first; any companion
        # artefacts overlay deliberately so a follow-up sweep can
        # supersede a stale cell without rebuilding the canonical pass.
        for c in cells:
            disc_by_cell[(c.seed, c.fold_id)] = c.f1_macro_discretized
        all_disc_cells.extend(cells)
        cls_by_cell.update(
            _extract_classification_cells(
                payload,
                "classification",
                "regime_f1_macro",
            )
        )
        dual_by_cell.update(
            _extract_classification_cells(
                payload,
                "dual",
                "regime_f1_macro",
            )
        )

    disc_f1 = [c.f1_macro_discretized for c in all_disc_cells]
    disc_mu, disc_sd = _mean_std(disc_f1)
    cls_mu, cls_sd = _mean_std(list(cls_by_cell.values()))
    dual_mu, dual_sd = _mean_std(list(dual_by_cell.values()))

    arm_summaries = [
        ArmSummary("classification", len(cls_by_cell), cls_mu, cls_sd),
        ArmSummary("regression_discretized", len(all_disc_cells), disc_mu, disc_sd),
        ArmSummary("dual", len(dual_by_cell), dual_mu, dual_sd),
    ]

    paired: list[PairedTest] = []
    if disc_by_cell:
        paired.append(
            _paired(
                disc_by_cell,
                dual_by_cell,
                "regression_discretized",
                "dual",
                coverage=coverage,
                n_resamples=n_resamples,
                bootstrap_seed=bootstrap_seed,
            )
        )
        paired.append(
            _paired(
                disc_by_cell,
                cls_by_cell,
                "regression_discretized",
                "classification",
                coverage=coverage,
                n_resamples=n_resamples,
                bootstrap_seed=bootstrap_seed,
            )
        )

    return {
        "sweep_artefacts": [str(p) for p in sweep_paths],
        "n_discretized_cells": len(all_disc_cells),
        "n_skipped_missing_predictions": skipped_total,
        "runner_extension_required": len(all_disc_cells) == 0,
        "discretized_cells": [c.to_dict() for c in all_disc_cells],
        "arm_summaries": [s.to_dict() for s in arm_summaries],
        "paired_tests": [p.to_dict() for p in paired],
    }


def _render_markdown(result: dict[str, Any]) -> str:
    arms = result.get("arm_summaries") or []
    lines = [
        "| Arm | n | F1 macro (mean ± std) |",
        "|---|---:|---|",
    ]
    for s in arms:
        mu = s["f1_mean"]
        sd = s["f1_std"]
        if mu != mu:
            cell = "n/a"
        else:
            cell = f"{mu:.4f} ± {sd:.4f}"
        lines.append(f"| {s['arm']} | {s['n_cells']} | {cell} |")
    paired = result.get("paired_tests") or []
    if paired:
        lines.append("")
        lines.append("| A vs B | n | mean Δ | W | p | d | CI lo | CI hi |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

        def _f(v: float) -> str:
            return "n/a" if v != v else f"{v:.4f}"

        for r in paired:
            lines.append(
                f"| {r['label_a']} vs {r['label_b']} | {r['n_pairs']}"
                f" | {_f(r['mean_delta'])} | {_f(r['wilcoxon_stat'])}"
                f" | {_f(r['p_value'])} | {_f(r['effect_size_d'])}"
                f" | {_f(r['ci_lo'])} | {_f(r['ci_hi'])} |"
            )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Discretize-at-eval companion metric for the dual-head defense.",
    )
    p.add_argument("--sweep-artefact", nargs="+", default=[str(_DEFAULT_ARTEFACT)])
    p.add_argument("--n-resamples", type=int, default=1000)
    p.add_argument("--coverage", type=float, default=0.95)
    p.add_argument("--bootstrap-seed", type=int, default=11)
    p.add_argument("--output", default=str(_DEFAULT_OUTPUT))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = compute_discretize_at_eval(
        [Path(s) for s in args.sweep_artefact],
        n_resamples=args.n_resamples,
        coverage=args.coverage,
        bootstrap_seed=args.bootstrap_seed,
    )
    result["markdown"] = _render_markdown(result)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    if result["runner_extension_required"]:
        print(
            "No regression cells carried per-event predictions + bin_edges; "
            f"{result['n_skipped_missing_predictions']} cells skipped. "
            "Table populates once the dual-head runner persists "
            "regression_predictions / regression_targets / bin_edges."
        )
    print(_render_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
