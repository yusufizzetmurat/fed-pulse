"""Paired Wilcoxon signed-rank tests on a head-mode comparison sweep (#497).

Reads a sweep JSON of the shape produced by
``scripts/run_dual_head_comparison.py``, matches per-(seed, fold) cells
across arms, and emits a paired test per arm-arm comparison. Sharpens
the §6.7 / §6.10 headline from "looks better" into "is better, p<X".

The 25 cells per arm (5 seeds × 5 folds) are NOT independent — the
same fold appears with 5 different seeds and vice versa — so a paired
test on the per-(seed, fold) delta is the right framing.

Holm-Bonferroni correction is applied across the comparison family
when multiple arm-arm pairs are tested in the same run.

Usage:

    python -m scripts.run_paired_comparison_tests \
        --input backend/artifacts/experiments/dual_head_comparison_canonical.json \
        --metric regime_f1_macro

Outputs a single JSON to stdout (or to ``--output`` if set) with one
row per pair carrying ``mean_delta``, ``std_delta``, ``wilcoxon_W``,
``wilcoxon_p``, ``holm_p`` (corrected), and ``n_pairs``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from itertools import combinations
from pathlib import Path
from typing import Any


def _collect_per_cell(
    arm_trials: list[dict[str, Any]],
    metric: str,
) -> dict[tuple[int, str], float]:
    """Return a {(seed, fold_id): metric} mapping for one arm."""

    out: dict[tuple[int, str], float] = {}
    for trial in arm_trials:
        seed = int(trial.get("seed", -1))
        for fold in trial.get("folds", []) or []:
            fold_id = str(fold.get("fold_id", ""))
            metrics = fold.get("metrics") or {}
            value = metrics.get(metric)
            if value is None:
                continue
            try:
                out[(seed, fold_id)] = float(value)
            except (TypeError, ValueError):
                continue
    return out


def _wilcoxon_signed_rank(deltas: list[float]) -> tuple[float, float, int]:
    """Compute the Wilcoxon signed-rank W statistic and a two-sided
    p-value for ``H0: deltas symmetric around zero``.

    Implemented by hand (not via scipy) so the script has zero non-
    stdlib runtime deps. Uses the normal approximation when n >= 10
    (the canonical 25-cell sweep gives n = 25 so the approximation is
    well-justified); for smaller n falls back to an exact enumeration
    over signed permutations.

    Returns ``(W, p_two_sided, n_nonzero)``.
    """

    nonzero = [d for d in deltas if d != 0]
    n = len(nonzero)
    if n == 0:
        return 0.0, 1.0, 0

    abs_ranked = sorted(enumerate(nonzero), key=lambda kv: abs(kv[1]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs_ranked[j + 1][1]) == abs(abs_ranked[i][1]):
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[abs_ranked[k][0]] = avg_rank
        i = j + 1

    w_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nonzero) if d < 0)
    w_stat = min(w_plus, w_minus)

    if n >= 10:
        mean_w = n * (n + 1) / 4.0
        var_w = n * (n + 1) * (2 * n + 1) / 24.0
        z = (w_stat - mean_w) / math.sqrt(var_w) if var_w > 0 else 0.0
        # Two-sided p via normal CDF
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
        return float(w_stat), max(min(p, 1.0), 0.0), n

    # Exact enumeration for n < 10
    from itertools import product

    target = w_stat
    count = 0
    total = 0
    for signs in product((-1, 1), repeat=n):
        signed = [s * r for s, r in zip(signs, ranks)]
        w_pos = sum(r for r in signed if r > 0)
        w_neg = -sum(r for r in signed if r < 0)
        w_min = min(w_pos, w_neg)
        if w_min <= target:
            count += 1
        total += 1
    p = count / total
    return float(w_stat), max(min(p, 1.0), 0.0), n


def _holm_correct(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction on a vector of p-values."""

    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    n = len(p_values)
    corrected = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * p_values[idx]
        running_max = max(running_max, adj)
        corrected[idx] = min(running_max, 1.0)
    return corrected


def run_paired_tests(
    sweep_payload: dict[str, Any],
    metric: str,
) -> list[dict[str, Any]]:
    """Run paired tests for every arm-arm pair in ``sweep_payload``."""

    trials_block = sweep_payload.get("trials") or {}
    if not trials_block:
        raise ValueError("input JSON has no 'trials' block")
    arms = list(trials_block.keys())
    per_arm_cells: dict[str, dict[tuple[int, str], float]] = {
        arm: _collect_per_cell(trials_block[arm], metric=metric)
        for arm in arms
    }

    pair_payloads: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for arm_a, arm_b in combinations(arms, 2):
        a_cells = per_arm_cells[arm_a]
        b_cells = per_arm_cells[arm_b]
        shared = sorted(set(a_cells) & set(b_cells))
        if not shared:
            continue
        deltas = [b_cells[k] - a_cells[k] for k in shared]
        mean_delta = statistics.fmean(deltas)
        std_delta = (
            statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
        )
        w_stat, p_two, n_nonzero = _wilcoxon_signed_rank(deltas)
        pair_payloads.append(
            {
                "metric": metric,
                "arm_a": arm_a,
                "arm_b": arm_b,
                "n_pairs": len(shared),
                "n_nonzero_pairs": n_nonzero,
                "mean_delta_b_minus_a": mean_delta,
                "std_delta": std_delta,
                "effect_size_d": (
                    mean_delta / std_delta if std_delta > 0 else None
                ),
                "wilcoxon_W": w_stat,
                "wilcoxon_p_two_sided": p_two,
            }
        )
        raw_p_values.append(p_two)

    holm = _holm_correct(raw_p_values) if raw_p_values else []
    for payload, p_corr in zip(pair_payloads, holm):
        payload["holm_corrected_p"] = p_corr
    return pair_payloads


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired Wilcoxon signed-rank tests on a head-mode "
            "comparison sweep. See #497."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Path to a sweep JSON (e.g. "
            "backend/artifacts/experiments/dual_head_comparison_canonical.json)."
        ),
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="regime_f1_macro",
        help="Per-fold metric to compare (default: regime_f1_macro).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON to this path; default stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = run_paired_tests(payload, metric=args.metric)
    report = {
        "source": str(args.input),
        "metric": args.metric,
        "pairs": rows,
    }
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[paired-tests] wrote {args.output}")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
