"""Per-fold class distribution + dual baselines table (#500).

Reads events.parquet + the fold manifest, fits the per-fold tertile
cutoffs the trainer would fit (train slice only, n_classes=3), and
computes two baseline F1s per fold:

- majority_class_f1: predict the most-frequent TRAIN class on every
  test row.
- stratified_random_f1: sample test predictions from the TRAIN class
  distribution; averaged over a fixed seed grid for stability.

The headline encoder F1 stays where it is reported (the sweep
artifact). This script lands the per-fold reference column so the
canonical comparison reports against a proper baseline rather than
an implicit 0.33 random-chance.

Usage:

    python -m scripts.run_per_fold_class_baselines \\
        --training-package-id <tp_id> \\
        --output backend/artifacts/experiments/per_fold_class_baselines.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_SEEDS: tuple[int, ...] = (11, 29, 47, 71, 97)
DEFAULT_N_CLASSES = 3


def _fit_quantile_cutoffs(
    forward_vols: np.ndarray, *, n_classes: int = DEFAULT_N_CLASSES
) -> tuple[float, ...]:
    """Mirror of ``backend/app/training/loaders.fit_vol_regime_quantiles``.

    Local copy keeps this script importable without the heavy training
    surface; the math is small and the contract is pinned by the
    matching unit test.
    """

    arr = forward_vols[np.isfinite(forward_vols)]
    if arr.size < n_classes:
        return ()
    qs = [(i + 1) / n_classes for i in range(n_classes - 1)]
    cutoffs = np.quantile(arr, qs)
    return tuple(float(c) for c in cutoffs)


def _label_with_cutoffs(
    forward_vols: np.ndarray, cutoffs: tuple[float, ...]
) -> np.ndarray:
    """Map continuous vol values to class indices using fitted cutoffs."""

    labels = np.full(forward_vols.shape, -1, dtype=np.int64)
    finite = np.isfinite(forward_vols)
    if not cutoffs:
        return labels
    values = forward_vols[finite]
    classes = np.zeros(values.shape, dtype=np.int64)
    for idx, cutoff in enumerate(cutoffs):
        classes = np.where(values >= cutoff, idx + 1, classes)
    labels[finite] = classes
    return labels


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro F1 over ``n_classes`` classes, equal weighting per class.

    Local impl avoids a sklearn import so the script runs in
    minimal-dependency CI environments. Matches
    ``sklearn.metrics.f1_score(average='macro')`` against the same
    inputs (verified by the unit test).
    """

    per_class_f1: list[float] = []
    for c in range(n_classes):
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        if tp + fp == 0 or tp + fn == 0:
            per_class_f1.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            per_class_f1.append(0.0)
        else:
            per_class_f1.append(2 * precision * recall / (precision + recall))
    return float(np.mean(per_class_f1))


def _load_fold_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds = payload.get("folds")
    if not isinstance(folds, list):
        raise ValueError(f"fold manifest at {path} has no 'folds' list")
    return folds


def _select_event_dates(
    df_events: pd.DataFrame, dates: list[str]
) -> pd.DataFrame:
    """Filter events.parquet to the requested event_dates."""

    if not dates:
        return df_events.iloc[0:0]
    return df_events[df_events["event_date"].astype(str).isin(dates)]


def _select_event_date_range(
    df_events: pd.DataFrame, start: str, end: str
) -> pd.DataFrame:
    """Filter events.parquet to event_date in [start, end] inclusive
    (the manifest convention from training_package_builder)."""

    if not start or not end:
        return df_events.iloc[0:0]
    dates = df_events["event_date"].astype(str)
    return df_events[(dates >= start) & (dates <= end)]


def _compute_fold_row(
    df_events: pd.DataFrame,
    fold: dict[str, Any],
    *,
    n_classes: int,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    fold_id = str(fold.get("fold_id") or fold.get("id") or "")

    # Two manifest shapes supported: explicit lists of event_dates per
    # split, OR start/end ranges (the production manifest from
    # training_package_builder uses ranges; tests use explicit lists).
    train_dates = [str(d) for d in (fold.get("train") or [])]
    test_dates = [str(d) for d in (fold.get("test") or [])]
    if train_dates or test_dates:
        train_df = _select_event_dates(df_events, train_dates)
        test_df = _select_event_dates(df_events, test_dates)
    else:
        train_df = _select_event_date_range(
            df_events,
            str(fold.get("train_start") or ""),
            str(fold.get("train_end") or ""),
        )
        test_df = _select_event_date_range(
            df_events,
            str(fold.get("test_start") or ""),
            str(fold.get("test_end") or ""),
        )

    train_vols = train_df["forward_realized_vol_10d"].to_numpy(dtype=np.float64)
    test_vols = test_df["forward_realized_vol_10d"].to_numpy(dtype=np.float64)

    cutoffs = _fit_quantile_cutoffs(train_vols, n_classes=n_classes)
    train_labels = _label_with_cutoffs(train_vols, cutoffs)
    test_labels = _label_with_cutoffs(test_vols, cutoffs)

    test_mask = test_labels >= 0
    y_true = test_labels[test_mask]
    if y_true.size == 0:
        return {
            "fold_id": fold_id,
            "n_train": int(train_df.shape[0]),
            "n_test": 0,
            "cutoffs": list(cutoffs),
            "test_class_counts": [0] * n_classes,
            "majority_class_idx": None,
            "majority_class_f1": None,
            "stratified_random_f1": None,
        }

    train_valid = train_labels[train_labels >= 0]
    train_counts = np.bincount(train_valid, minlength=n_classes).astype(int)
    test_counts = np.bincount(y_true, minlength=n_classes).astype(int)
    train_total = int(train_counts.sum())
    train_probs = (
        train_counts.astype(np.float64) / train_total
        if train_total > 0
        else np.full(n_classes, 1.0 / n_classes)
    )
    majority_idx = int(np.argmax(train_counts))
    majority_pred = np.full(y_true.shape, majority_idx, dtype=np.int64)
    majority_f1 = _macro_f1(y_true, majority_pred, n_classes)

    per_seed_strat: list[float] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        strat_pred = rng.choice(
            n_classes, size=y_true.shape, replace=True, p=train_probs
        )
        per_seed_strat.append(
            _macro_f1(y_true, strat_pred.astype(np.int64), n_classes)
        )
    strat_mean = float(np.mean(per_seed_strat))
    strat_std = float(np.std(per_seed_strat))

    return {
        "fold_id": fold_id,
        "n_train": int(train_df.shape[0]),
        "n_test": int(y_true.size),
        "cutoffs": list(cutoffs),
        "train_class_counts": [int(c) for c in train_counts],
        "test_class_counts": [int(c) for c in test_counts],
        "majority_class_idx": majority_idx,
        "majority_class_f1": majority_f1,
        "stratified_random_f1": strat_mean,
        "stratified_random_f1_std": strat_std,
        "stratified_seeds": list(seeds),
    }


def compute_per_fold_baselines(
    events_parquet: Path,
    fold_manifest: Path,
    *,
    n_classes: int = DEFAULT_N_CLASSES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    df_events = pd.read_parquet(events_parquet)
    if "event_date" not in df_events.columns:
        raise ValueError("events.parquet missing 'event_date' column")
    if "forward_realized_vol_10d" not in df_events.columns:
        raise ValueError(
            "events.parquet missing 'forward_realized_vol_10d' column"
        )
    folds = _load_fold_manifest(fold_manifest)

    per_fold_rows = [
        _compute_fold_row(df_events, fold, n_classes=n_classes, seeds=seeds)
        for fold in folds
    ]

    majority_values = [
        row["majority_class_f1"]
        for row in per_fold_rows
        if row.get("majority_class_f1") is not None
    ]
    strat_values = [
        row["stratified_random_f1"]
        for row in per_fold_rows
        if row.get("stratified_random_f1") is not None
    ]
    summary: dict[str, Any] = {
        "majority_class_f1_pooled_mean": (
            float(np.mean(majority_values)) if majority_values else None
        ),
        "stratified_random_f1_pooled_mean": (
            float(np.mean(strat_values)) if strat_values else None
        ),
        "n_folds": len(per_fold_rows),
        "n_classes": int(n_classes),
    }
    return {
        "events_parquet": str(events_parquet),
        "fold_manifest": str(fold_manifest),
        "per_fold": per_fold_rows,
        "summary": summary,
    }


def _render_summary_md(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines: list[str] = []
    lines.append("# Per-fold class baselines (#500)")
    lines.append("")
    lines.append(f"- events.parquet: `{report['events_parquet']}`")
    lines.append(f"- fold manifest: `{report['fold_manifest']}`")
    lines.append(
        f"- folds: {summary['n_folds']} · classes: {summary['n_classes']}"
    )
    lines.append("")
    lines.append(
        "| fold | n_train | n_test | train_counts | test_counts | "
        "majority_class | majority_F1 | stratified_F1 |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|"
    )
    for row in report["per_fold"]:
        train_counts = row.get("train_class_counts") or []
        test_counts = row.get("test_class_counts") or []
        majority = row.get("majority_class_idx")
        mf1 = row.get("majority_class_f1")
        sf1 = row.get("stratified_random_f1")
        mf1_str = f"{mf1:.4f}" if mf1 is not None else "n/a"
        sf1_str = f"{sf1:.4f}" if sf1 is not None else "n/a"
        lines.append(
            f"| {row['fold_id']} | {row['n_train']} | {row['n_test']} | "
            f"{train_counts} | {test_counts} | {majority} | "
            f"{mf1_str} | {sf1_str} |"
        )
    lines.append("")
    pooled_m = summary["majority_class_f1_pooled_mean"]
    pooled_s = summary["stratified_random_f1_pooled_mean"]
    if pooled_m is not None and pooled_s is not None:
        lines.append(
            f"Pooled mean: majority-class {pooled_m:.4f}, "
            f"stratified-random {pooled_s:.4f}"
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-fold class distribution + dual baselines (#500)."
    )
    parser.add_argument("--training-package-id", default=None)
    parser.add_argument("--events-parquet", type=Path, default=None)
    parser.add_argument("--fold-manifest", type=Path, default=None)
    parser.add_argument("--n-classes", type=int, default=DEFAULT_N_CLASSES)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Seed grid used to average the stratified-random F1.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.training_package_id and not args.events_parquet:
        base = Path(__file__).resolve().parent.parent / "data" / "processed" / args.training_package_id
        events = base / "events.parquet"
        manifest = base / "fold_manifest_expanding_walk_forward.json"
        return events, manifest
    if args.events_parquet and args.fold_manifest:
        return args.events_parquet, args.fold_manifest
    raise SystemExit(
        "either --training-package-id or both --events-parquet and "
        "--fold-manifest must be supplied"
    )


def main() -> int:
    args = _parse_args()
    events_parquet, fold_manifest = _resolve_paths(args)
    report = compute_per_fold_baselines(
        events_parquet=events_parquet,
        fold_manifest=fold_manifest,
        n_classes=int(args.n_classes),
        seeds=tuple(int(s) for s in args.seeds),
    )
    summary_md = _render_summary_md(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        md_path = args.output.with_suffix(".md")
        md_path.write_text(summary_md, encoding="utf-8")
        print(f"[per-fold-baselines] wrote {args.output} + {md_path}")
    else:
        sys.stdout.write(summary_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
