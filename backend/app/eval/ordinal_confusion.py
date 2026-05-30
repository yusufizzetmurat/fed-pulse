"""Ordinal confusion-matrix decomposition for the regime classifier (#496).

Reads per-cell classification_breakdown payloads from sweep artefacts.
Post-#523 artefacts carry the breakdown; pre-#523 artefacts do not and
are skipped with a structured warning.

For cells with a breakdown, computes:
  adjacent error rate    = off-by-1 errors / total errors
  non-adjacent error rate = off-by-2 errors / total errors (calm <-> high)
  ordinal accuracy        = 1 - non_adjacent_error_rate

Aggregates per arm (head_mode): mean ± std across seed x fold cells.

CLI::

    python -m app.eval.ordinal_confusion \\
        --sweep-artefact backend/artifacts/experiments/dual_head_comparison_canonical.json \\
        --output backend/artifacts/experiments/ordinal_confusion.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from app.config import BACKEND_ROOT

_DEFAULT_ARTEFACT = (
    BACKEND_ROOT / "artifacts" / "experiments" / "dual_head_comparison_canonical.json"
)
_DEFAULT_OUTPUT = (
    BACKEND_ROOT / "artifacts" / "experiments" / "ordinal_confusion.json"
)


def decompose_ordinal(
    confusion: Sequence[Sequence[int]],
) -> dict[str, float]:
    """Adjacent / non-adjacent error rates and ordinal accuracy.

    Rows = true class, columns = predicted class. Adjacent = |true-pred|==1;
    non-adjacent = |true-pred|>=2.
    """
    n = len(confusion)
    total_errors = 0
    adjacent = 0
    non_adjacent = 0
    for r in range(n):
        for c in range(n):
            count = int(confusion[r][c])
            if r == c:
                continue
            total_errors += count
            dist = abs(r - c)
            if dist == 1:
                adjacent += count
            else:
                non_adjacent += count
    adj_rate = adjacent / total_errors if total_errors > 0 else 0.0
    nonadj_rate = non_adjacent / total_errors if total_errors > 0 else 0.0
    return {
        "total_errors": total_errors,
        "adjacent_errors": adjacent,
        "non_adjacent_errors": non_adjacent,
        "adjacent_error_rate": adj_rate,
        "non_adjacent_error_rate": nonadj_rate,
        "ordinal_accuracy": 1.0 - nonadj_rate,
    }


@dataclasses.dataclass(frozen=True)
class OrdinalCell:
    head_mode: str
    seed: int
    fold_id: str
    adjacent_error_rate: float
    non_adjacent_error_rate: float
    ordinal_accuracy: float
    total_errors: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def extract_cells(sweep_path: Path) -> list[OrdinalCell]:
    """Pull OrdinalCell records from a sweep artefact.

    Cells missing classification_breakdown emit a warning and are skipped
    (pre-#523 artefact format).
    """
    payload = json.loads(sweep_path.read_text(encoding="utf-8"))
    trials_by_mode: dict[str, list[Any]] = payload.get("trials") or {}
    cells: list[OrdinalCell] = []
    for head_mode, mode_trials in trials_by_mode.items():
        for trial in mode_trials:
            seed = int(trial.get("seed", -1))
            for fold in trial.get("folds") or []:
                fold_id = str(fold.get("fold_id", ""))
                metrics = fold.get("metrics") or {}
                breakdown = metrics.get("classification_breakdown")
                if breakdown is None:
                    warnings.warn(
                        f"[ordinal_confusion] {sweep_path.name} "
                        f"head={head_mode} seed={seed} fold={fold_id}: "
                        "no classification_breakdown; skipping (pre-#523 artefact)",
                        stacklevel=2,
                    )
                    continue
                cm_raw = breakdown.get("confusion_matrix")
                if not cm_raw or not isinstance(cm_raw, list):
                    warnings.warn(
                        f"[ordinal_confusion] {sweep_path.name} "
                        f"head={head_mode} seed={seed} fold={fold_id}: "
                        "confusion_matrix missing or malformed; skipping",
                        stacklevel=2,
                    )
                    continue
                d = decompose_ordinal(cm_raw)
                cells.append(OrdinalCell(
                    head_mode=head_mode,
                    seed=seed,
                    fold_id=fold_id,
                    adjacent_error_rate=d["adjacent_error_rate"],
                    non_adjacent_error_rate=d["non_adjacent_error_rate"],
                    ordinal_accuracy=d["ordinal_accuracy"],
                    total_errors=int(d["total_errors"]),
                ))
    return cells


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
    head_mode: str
    n_cells: int
    mean_adjacent_error_rate: float
    std_adjacent_error_rate: float
    mean_non_adjacent_error_rate: float
    std_non_adjacent_error_rate: float
    mean_ordinal_accuracy: float
    std_ordinal_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def aggregate_by_arm(cells: Sequence[OrdinalCell]) -> list[ArmSummary]:
    by_arm: dict[str, list[OrdinalCell]] = defaultdict(list)
    for c in cells:
        by_arm[c.head_mode].append(c)
    summaries: list[ArmSummary] = []
    for head_mode in sorted(by_arm):
        arm = by_arm[head_mode]
        adj_mu, adj_sd = _mean_std([c.adjacent_error_rate for c in arm])
        na_mu, na_sd = _mean_std([c.non_adjacent_error_rate for c in arm])
        oa_mu, oa_sd = _mean_std([c.ordinal_accuracy for c in arm])
        summaries.append(ArmSummary(
            head_mode=head_mode,
            n_cells=len(arm),
            mean_adjacent_error_rate=adj_mu,
            std_adjacent_error_rate=adj_sd,
            mean_non_adjacent_error_rate=na_mu,
            std_non_adjacent_error_rate=na_sd,
            mean_ordinal_accuracy=oa_mu,
            std_ordinal_accuracy=oa_sd,
        ))
    return summaries


def compute_ordinal_confusion(sweep_paths: Sequence[Path]) -> dict[str, Any]:
    all_cells: list[OrdinalCell] = []
    for p in sweep_paths:
        all_cells.extend(extract_cells(p))
    summaries = aggregate_by_arm(all_cells)

    def _f(v: float) -> float | None:
        return None if v != v else round(v, 6)

    return {
        "n_cells_total": len(all_cells),
        "cells": [c.to_dict() for c in all_cells],
        "arm_summaries": [
            {
                "head_mode": s.head_mode,
                "n_cells": s.n_cells,
                "adjacent_error_rate": {
                    "mean": _f(s.mean_adjacent_error_rate),
                    "std": _f(s.std_adjacent_error_rate),
                },
                "non_adjacent_error_rate": {
                    "mean": _f(s.mean_non_adjacent_error_rate),
                    "std": _f(s.std_non_adjacent_error_rate),
                },
                "ordinal_accuracy": {
                    "mean": _f(s.mean_ordinal_accuracy),
                    "std": _f(s.std_ordinal_accuracy),
                },
            }
            for s in summaries
        ],
    }


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "| Head mode | n | Adj err (mean±std) | Non-adj err (mean±std) | Ordinal acc (mean±std) |",
        "|---|---:|---|---|---|",
    ]
    for s in result.get("arm_summaries", []):
        def _cell(d: dict[str, Any]) -> str:
            mu, sd = d.get("mean"), d.get("std")
            if mu is None:
                return "n/a"
            return f"{mu:.4f} ± {sd:.4f}" if sd is not None else f"{mu:.4f}"
        lines.append(
            f"| {s['head_mode']} | {s['n_cells']}"
            f" | {_cell(s['adjacent_error_rate'])}"
            f" | {_cell(s['non_adjacent_error_rate'])}"
            f" | {_cell(s['ordinal_accuracy'])} |"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ordinal confusion-matrix decomposition for regime classifier."
    )
    p.add_argument("--sweep-artefact", nargs="+", default=[str(_DEFAULT_ARTEFACT)])
    p.add_argument("--output", default=str(_DEFAULT_OUTPUT))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = compute_ordinal_confusion([Path(s) for s in args.sweep_artefact])
    result["markdown"] = _render_markdown(result)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    if result["n_cells_total"] == 0:
        print("No cells with classification_breakdown (pre-#523 artefacts); table pending.")
    else:
        print(_render_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
