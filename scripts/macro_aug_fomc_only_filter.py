"""FOMC-only subset re-aggregation of the macro-augmented sweep result.

The macro-augmented arch-sweep evaluates models on a test pool that
mixes FOMC events and macro-release events (CPI, NFP). The headline
macro-F1 is the pooled CI over that mixed pool. For an apples-to-
apples comparison against the Chunk-1 headline (FOMC-only), we need
to re-pool the same trial predictions filtered to FOMC test rows.

Each test prediction position lines up with an entry in
``WalkForwardSplit.test_event_dates`` for its trial's fold. Macro
events live on CPI / NFP release dates; FOMC events live on FOMC
statement / minutes / press-conference / speech dates. Because the
two sets of dates almost never overlap on the same calendar day, we
can classify each test position as macro-vs-FOMC purely by whether
its event_date is in the FRED release-calendar set.

Output: a markdown table with both pooled-CI numbers (full pool +
FOMC-only subset) for every architecture in the sweep dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.evaluation.bootstrap import block_bootstrap_ci  # noqa: E402
from app.evaluation.classification_breakdown import (  # noqa: E402
    compute_classification_breakdown,
)


def _load_fold_test_dates(package_dir: Path, fold_id: str) -> list[str]:
    """Read the test partition's per-row event_date list for one fold.

    Returns dates in the same order the trainer's
    ``WalkForwardSplit.test`` partition iterates them — i.e. the order
    the per-trial predictions arrive in.
    """

    from app.training.loaders import load_walk_forward_split

    split = load_walk_forward_split(
        training_package_id=package_dir.name,
        fold_id=fold_id,
        rich_features=True,
    )
    return list(split.test_event_dates)


def _macro_event_dates(registry_jsonl: Path) -> set[str]:
    """Return the set of event_dates that came from the FRED release
    calendar (i.e. macro_release rows). Read from the augmented
    registry directly so we do not depend on the events.parquet
    survivors."""

    dates: set[str] = set()
    with registry_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("source", "")) == "fred_macro_releases":
                date = str(row.get("event_date", ""))[:10]
                if date:
                    dates.add(date)
    return dates


def _iter_arch_dirs(sweep_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in sweep_dir.iterdir()
        if path.is_dir() and (path / "forecaster_sweep_results.json").exists()
    )


def _select_best_hp_cell(payload: dict[str, Any]) -> str:
    """Replicate the aggregator's per-architecture HP-cell selection:
    pick the hp_combo_id with the highest pooled macro-F1 across folds
    + seeds. We use the same selection criterion so the FOMC-only
    subset evaluates the same cell as the headline."""

    by_cell: dict[Any, list[tuple[int, int]]] = defaultdict(list)
    for trial in payload.get("trials", []):
        summary = trial.get("summary", trial)
        if not isinstance(summary, dict):
            continue
        test_metrics = summary.get("test_metrics") or {}
        preds = test_metrics.get("predictions")
        targets = test_metrics.get("targets")
        if not isinstance(preds, list) or not isinstance(targets, list):
            continue
        hp = trial.get("hp_combo_id")
        # Pool by HP cell across folds + seeds for the selection step.
        by_cell[hp].extend(zip(preds, targets))

    best_cell: Any = None
    best_macro = -1.0
    for hp, pairs in by_cell.items():
        if not pairs:
            continue
        preds = [int(p) for p, _ in pairs]
        targets = [int(t) for _, t in pairs]
        breakdown = compute_classification_breakdown(preds, targets, n_classes=3)
        if breakdown.macro_f1 > best_macro:
            best_macro = breakdown.macro_f1
            best_cell = hp
    return best_cell


def _aggregate_arch(
    arch_dir: Path,
    *,
    macro_dates: set[str],
    package_dir: Path,
) -> dict[str, Any]:
    payload = json.loads((arch_dir / "forecaster_sweep_results.json").read_text())
    best_cell = _select_best_hp_cell(payload)
    fold_dates_cache: dict[str, list[str]] = {}

    pooled_full: list[tuple[int, int]] = []
    pooled_fomc: list[tuple[int, int]] = []
    fold_ids_seen: set[str] = set()
    seeds_seen: set[int] = set()

    for trial in payload.get("trials", []):
        if trial.get("hp_combo_id") != best_cell:
            continue
        summary = trial.get("summary", trial)
        fold_id = summary.get("fold_id") or trial.get("fold_id")
        seed = trial.get("seed") or summary.get("seed")
        test_metrics = summary.get("test_metrics") or {}
        preds = test_metrics.get("predictions")
        targets = test_metrics.get("targets")
        if not (
            isinstance(preds, list)
            and isinstance(targets, list)
            and isinstance(fold_id, str)
        ):
            continue
        if fold_id not in fold_dates_cache:
            fold_dates_cache[fold_id] = _load_fold_test_dates(package_dir, fold_id)
        test_dates = fold_dates_cache[fold_id]
        if len(preds) > len(test_dates):
            # Should not happen: more predictions than test rows.
            for p, t in zip(preds, targets):
                pooled_full.append((int(p), int(t)))
            continue
        if len(preds) < len(test_dates):
            # The trainer drops trailing test rows whose forward
            # target window is incomplete (10-day forward vol). The
            # ``test_event_dates`` list mirrors the raw partition; trim
            # from the END so positions line up with the predictions
            # the trainer actually emitted.
            test_dates = test_dates[: len(preds)]
        for p, t, d in zip(preds, targets, test_dates):
            pooled_full.append((int(p), int(t)))
            if str(d)[:10] not in macro_dates:
                pooled_fomc.append((int(p), int(t)))
        fold_ids_seen.add(fold_id)
        if isinstance(seed, int):
            seeds_seen.add(seed)

    full_breakdown = compute_classification_breakdown(
        [p for p, _ in pooled_full], [t for _, t in pooled_full], n_classes=3
    ) if pooled_full else None
    fomc_breakdown = compute_classification_breakdown(
        [p for p, _ in pooled_fomc], [t for _, t in pooled_fomc], n_classes=3
    ) if pooled_fomc else None

    full_ci = (
        block_bootstrap_ci(
            [p == t for p, t in pooled_full],  # ignored; we use macro-F1 path below
            block_size=20,
            n_resamples=1000,
        )
        if pooled_full
        else None
    )

    def _ci_macro(pairs: list[tuple[int, int]]) -> dict[str, float] | None:
        import random
        if not pairs:
            return None
        rng = random.Random(11)
        n = len(pairs)
        block_size = 20
        macros: list[float] = []
        for _ in range(1000):
            sample: list[tuple[int, int]] = []
            while len(sample) < n:
                start = rng.randrange(0, n)
                sample.extend(pairs[start:start + block_size])
            sample = sample[:n]
            br = compute_classification_breakdown(
                [p for p, _ in sample], [t for _, t in sample], n_classes=3
            )
            macros.append(br.macro_f1)
        macros.sort()
        lo = macros[int(0.025 * len(macros))]
        hi = macros[int(0.975 * len(macros))]
        return {"lo": lo, "hi": hi}

    full_ci_macro = _ci_macro(pooled_full)
    fomc_ci_macro = _ci_macro(pooled_fomc)

    return {
        "arch": arch_dir.name,
        "best_hp_combo": best_cell,
        "n_full": len(pooled_full),
        "n_fomc": len(pooled_fomc),
        "n_macro": len(pooled_full) - len(pooled_fomc),
        "seeds": sorted(seeds_seen),
        "folds": sorted(fold_ids_seen),
        "macro_f1_full": full_breakdown.macro_f1 if full_breakdown else None,
        "macro_f1_fomc": fomc_breakdown.macro_f1 if fomc_breakdown else None,
        "ci_full": full_ci_macro,
        "ci_fomc": fomc_ci_macro,
    }


def _format_md(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Macro-aug sweep — full pool vs FOMC-only subset macro-F1")
    lines.append("")
    lines.append(
        "Both columns use the same per-architecture best HP cell from the "
        "macro-augmented sweep (selected by full-pool macro-F1). The "
        "FOMC-only column re-pools the same trial predictions after "
        "dropping macro-release rows by event_date."
    )
    lines.append("")
    lines.append("| Arch | n_full | n_fomc | macro-F1 (full) | 95% CI (full) | macro-F1 (FOMC-only) | 95% CI (FOMC-only) |")
    lines.append("|---|---:|---:|---:|---|---:|---|")
    for row in rows:
        full_ci = row.get("ci_full") or {}
        fomc_ci = row.get("ci_fomc") or {}
        lines.append(
            f"| {row['arch']} | {row['n_full']} | {row['n_fomc']} | "
            f"{row['macro_f1_full']:.4f} | "
            f"[{full_ci.get('lo', 0.0):.4f}, {full_ci.get('hi', 0.0):.4f}] | "
            f"{row['macro_f1_fomc']:.4f} | "
            f"[{fomc_ci.get('lo', 0.0):.4f}, {fomc_ci.get('hi', 0.0):.4f}] |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--registry-jsonl", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    macro_dates = _macro_event_dates(args.registry_jsonl)
    print(f"[fomc-only] {len(macro_dates)} macro-release event dates loaded")

    rows: list[dict[str, Any]] = []
    for arch_dir in _iter_arch_dirs(args.sweep_dir):
        print(f"[fomc-only] aggregating {arch_dir.name} ...")
        row = _aggregate_arch(
            arch_dir, macro_dates=macro_dates, package_dir=args.package_dir
        )
        rows.append(row)
        print(
            f"  best_hp_combo={row['best_hp_combo']}  "
            f"n_full={row['n_full']}  n_fomc={row['n_fomc']}  "
            f"macro_f1_full={row['macro_f1_full']:.4f}  "
            f"macro_f1_fomc={row['macro_f1_fomc']:.4f}"
        )

    out_json = args.output_json or args.sweep_dir / "fomc_only_subset_macro_f1.json"
    out_md = args.output_md or args.sweep_dir / "fomc_only_subset_macro_f1.md"
    out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_format_md(rows), encoding="utf-8")
    print(f"[fomc-only] wrote {out_json} + {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
