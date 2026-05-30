"""Paired statistical tests on sweep config comparisons (#497).

Reads dual_head_comparison_*.json sweep artefacts, matches cells by
(seed, fold_id), and reports:
  - Paired Wilcoxon signed-rank p-value (two-sided) via scipy
  - Cohen's d: mean(delta) / std(delta)
  - Block-bootstrap CI on mean(delta), blocking by fold
  - Holm-Bonferroni correction across the comparison family

CLI::

    python -m app.eval.paired_comparisons \\
        --comparisons classification,dual classification,regression regression,dual \\
        --metric regime_f1_macro \\
        --output backend/artifacts/experiments/paired_comparisons.json
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

_DEFAULT_ARTEFACT = (
    BACKEND_ROOT / "artifacts" / "experiments" / "dual_head_comparison_canonical.json"
)
_DEFAULT_OUTPUT = (
    BACKEND_ROOT / "artifacts" / "experiments" / "paired_comparisons.json"
)


def _extract_cells(
    sweep_path: Path,
    head_mode: str,
    metric: str,
) -> dict[tuple[int, str], float]:
    payload = json.loads(sweep_path.read_text(encoding="utf-8"))
    mode_trials: list[Any] = (payload.get("trials") or {}).get(head_mode) or []
    cells: dict[tuple[int, str], float] = {}
    for trial in mode_trials:
        seed = int(trial.get("seed", -1))
        for fold in trial.get("folds") or []:
            fid = str(fold.get("fold_id", ""))
            v = (fold.get("metrics") or {}).get(metric)
            if v is not None and seed >= 0:
                cells[(seed, fid)] = float(v)
    return cells


def extract_paired_deltas(
    sweep_path: Path,
    label_a: str,
    label_b: str,
    metric: str,
) -> tuple[list[float], list[str]]:
    """Extract paired (a - b) deltas matched by (seed, fold_id).

    Sort order is (fold_id, seed) so the resulting deltas are grouped by
    fold; the per-fold groups are the blocks the bootstrap below relies
    on to respect within-fold correlation across seeds.
    """
    cells_a = _extract_cells(sweep_path, label_a, metric)
    cells_b = _extract_cells(sweep_path, label_b, metric)
    shared = sorted(set(cells_a) & set(cells_b), key=lambda k: (k[1], k[0]))
    deltas = [cells_a[k] - cells_b[k] for k in shared]
    fold_labels = [k[1] for k in shared]
    return deltas, fold_labels


def wilcoxon_signed_rank(
    deltas: Sequence[float],
) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test via scipy.stats.wilcoxon.

    Zero deltas are dropped per the standard convention. Returns NaN
    for both statistic and p-value when fewer than 2 nonzero deltas
    remain — scipy raises on n<2 in some versions and returns a
    degenerate value in others, neither of which is a defensible
    headline-table cell. The warning fires at <3 because n=2 is
    technically valid but rank-tied and underpowered.
    """
    from scipy.stats import wilcoxon

    nonzero = [d for d in deltas if d != 0.0]
    if len(nonzero) < 2:
        return float("nan"), float("nan")
    if len(nonzero) < 3:
        warnings.warn(
            f"[paired_comparisons] only {len(nonzero)} nonzero deltas; "
            "Wilcoxon p-value may be unreliable",
            stacklevel=2,
        )
    stat, pval = wilcoxon(nonzero, alternative="two-sided")
    return float(stat), float(pval)


def effect_size(deltas: Sequence[float]) -> float:
    """Cohen's d analogue: mean(delta) / std(delta, ddof=1).

    Returns NaN when the deltas are constant (sd at or below float noise
    against the mean magnitude); otherwise float64 round-off would let
    a 0/0 case surface as a spurious huge ratio.
    """
    n = len(deltas)
    if n < 2:
        return float("nan")
    mu = sum(deltas) / n
    var = sum((d - mu) ** 2 for d in deltas) / (n - 1)
    sd = math.sqrt(var) if var > 0 else 0.0
    tol = 1e-12 * max(1.0, abs(mu))
    if sd <= tol:
        return float("nan")
    return mu / sd


def block_bootstrap_ci_deltas(
    deltas: Sequence[float],
    fold_labels: Sequence[str],
    *,
    n_resamples: int = 1000,
    coverage: float = 0.95,
    seed: int = 11,
) -> tuple[float, float, float]:
    """Block-bootstrap CI on mean(delta), one block per fold.

    Relies on ``extract_paired_deltas`` returning deltas sorted by
    ``(fold_id, seed)`` so contiguous slices map to per-fold groups.
    ``block_size`` is set to the per-fold group size (deltas/n_folds);
    a moving-blocks resample then preserves the within-fold seed
    correlation. Returns (point, lo, hi).
    """
    from app.evaluation.bootstrap import block_bootstrap_ci

    if not deltas:
        return float("nan"), float("nan"), float("nan")
    n_folds = len(set(fold_labels))
    block_size = max(1, len(deltas) // max(1, n_folds))
    ci = block_bootstrap_ci(
        list(deltas),
        block_size=block_size,
        n_resamples=n_resamples,
        coverage=coverage,
        seed=seed,
    )
    return float(ci.point), float(ci.lo), float(ci.hi)


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni corrected p-values (in original input order)."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    corrected = [0.0] * n
    running_max = 0.0
    for rank, (orig_idx, pval) in enumerate(indexed):
        adjusted = min(1.0, pval * (n - rank))
        adjusted = max(adjusted, running_max)
        running_max = adjusted
        corrected[orig_idx] = adjusted
    return corrected


@dataclasses.dataclass(frozen=True)
class ComparisonResult:
    label_a: str
    label_b: str
    metric: str
    n_pairs: int
    mean_delta: float
    std_delta: float
    wilcoxon_stat: float
    p_value: float
    p_value_holm: float
    effect_size_d: float
    ci_lo: float
    ci_hi: float
    ci_coverage: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def compare_two(  # noqa: PLR0913 -- bootstrap knobs + Holm placeholder are independent kwargs by design
    sweep_path: Path,
    label_a: str,
    label_b: str,
    metric: str,
    *,
    n_resamples: int = 1000,
    coverage: float = 0.95,
    bootstrap_seed: int = 11,
    p_value_holm: float = float("nan"),
) -> ComparisonResult:
    deltas, fold_labels = extract_paired_deltas(sweep_path, label_a, label_b, metric)
    n = len(deltas)
    if n == 0:
        return ComparisonResult(
            label_a=label_a, label_b=label_b, metric=metric, n_pairs=0,
            mean_delta=float("nan"), std_delta=float("nan"),
            wilcoxon_stat=float("nan"), p_value=float("nan"),
            p_value_holm=p_value_holm, effect_size_d=float("nan"),
            ci_lo=float("nan"), ci_hi=float("nan"), ci_coverage=coverage,
        )
    mu = sum(deltas) / n
    var = sum((d - mu) ** 2 for d in deltas) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    stat, pval = wilcoxon_signed_rank(deltas)
    es = effect_size(deltas)
    point, lo, hi = block_bootstrap_ci_deltas(
        deltas, fold_labels,
        n_resamples=n_resamples, coverage=coverage, seed=bootstrap_seed,
    )
    return ComparisonResult(
        label_a=label_a, label_b=label_b, metric=metric, n_pairs=n,
        mean_delta=mu, std_delta=sd,
        wilcoxon_stat=stat, p_value=pval,
        p_value_holm=p_value_holm, effect_size_d=es,
        ci_lo=lo, ci_hi=hi, ci_coverage=coverage,
    )


def compute_paired_comparisons(  # noqa: PLR0913 -- bootstrap knobs surface as named kwargs by design
    sweep_paths: Sequence[Path],
    comparison_pairs: Sequence[tuple[str, str]],
    metric: str,
    *,
    n_resamples: int = 1000,
    coverage: float = 0.95,
    bootstrap_seed: int = 11,
) -> list[ComparisonResult]:
    """Run Wilcoxon + bootstrap for every (a, b) pair, then apply Holm correction."""
    raw: list[ComparisonResult] = []
    for sweep_path in sweep_paths:
        for label_a, label_b in comparison_pairs:
            raw.append(compare_two(
                sweep_path, label_a, label_b, metric,
                n_resamples=n_resamples, coverage=coverage,
                bootstrap_seed=bootstrap_seed,
            ))
    corrected = holm_bonferroni([r.p_value for r in raw])
    return [dataclasses.replace(r, p_value_holm=corrected[i]) for i, r in enumerate(raw)]


def _render_markdown(results: list[ComparisonResult]) -> str:
    lines = [
        "| A vs B | n | mean Δ | W | p | p (Holm) | d | CI lo | CI hi |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        def _f(v: float) -> str:
            return "n/a" if v != v else f"{v:.4f}"
        lines.append(
            f"| {r.label_a} vs {r.label_b} | {r.n_pairs}"
            f" | {_f(r.mean_delta)} | {_f(r.wilcoxon_stat)}"
            f" | {_f(r.p_value)} | {_f(r.p_value_holm)}"
            f" | {_f(r.effect_size_d)} | {_f(r.ci_lo)} | {_f(r.ci_hi)} |"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paired statistical tests on sweep config comparisons."
    )
    p.add_argument(
        "--sweep-artefacts",
        nargs="+",
        default=[str(_DEFAULT_ARTEFACT)],
    )
    p.add_argument(
        "--comparisons",
        nargs="+",
        default=["classification,dual", "classification,regression", "regression,dual"],
        help="Comma-separated head-mode pairs, e.g. 'classification,dual'.",
    )
    p.add_argument("--metric", default="regime_f1_macro")
    p.add_argument("--n-resamples", type=int, default=1000)
    p.add_argument("--coverage", type=float, default=0.95)
    p.add_argument("--bootstrap-seed", type=int, default=11)
    p.add_argument("--output", default=str(_DEFAULT_OUTPUT))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sweep_paths = [Path(s) for s in args.sweep_artefacts]
    pairs: list[tuple[str, str]] = []
    for token in args.comparisons:
        parts = token.split(",")
        if len(parts) != 2:
            raise SystemExit(f"comparison must be 'a,b'; got {token!r}")
        pairs.append((parts[0].strip(), parts[1].strip()))
    results = compute_paired_comparisons(
        sweep_paths=sweep_paths,
        comparison_pairs=pairs,
        metric=args.metric,
        n_resamples=args.n_resamples,
        coverage=args.coverage,
        bootstrap_seed=args.bootstrap_seed,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "metric": args.metric,
        "comparisons": [r.to_dict() for r in results],
        "markdown": _render_markdown(results),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(_render_markdown(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
