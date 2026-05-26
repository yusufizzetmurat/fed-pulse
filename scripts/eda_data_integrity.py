"""Data integrity audit on a published training package (#200).

Produces the per-fold, per-source, per-target diagnostics the report
appendix needs so every headline number can be traced back to a
reproducible row count + class balance. Outputs land under
``data/artifacts/eda/<training_package_id>/`` and the wiki page at
``../fed-pulse.wiki/15_Data_Integrity_Report.md`` reads from them.

Run with::

    python scripts/eda_data_integrity.py --training-package-id <pkg>

The script is read-only on the package directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


def _resolve_package_dir(training_package_id: str) -> Path:
    """Find the events.parquet for ``training_package_id`` regardless of
    whether we're running inside the backend container (``/data`` mount)
    or directly on the host filesystem."""

    candidates = [
        Path(f"/data/processed/{training_package_id}"),
        Path(f"data/processed/{training_package_id}"),
        Path(f"backend/data/processed/{training_package_id}"),
    ]
    for c in candidates:
        if (c / "events.parquet").exists():
            return c
    raise FileNotFoundError(
        f"events.parquet not found for {training_package_id}; tried: "
        + ", ".join(str(c) for c in candidates)
    )


def _resolve_output_root(training_package_id: str) -> Path:
    base = Path(os.environ.get("FED_PULSE_DATA_DIR", "data"))
    out = base / "artifacts" / "eda" / training_package_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_fold_manifest(pkg_dir: Path) -> dict[str, dict[str, str]] | None:
    p = pkg_dir / "fold_manifest_expanding_walk_forward.json"
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    folds = payload.get("folds") if isinstance(payload, dict) else None
    if not isinstance(folds, list):
        return None
    out: dict[str, dict[str, str]] = {}
    for f in folds:
        if not isinstance(f, dict):
            continue
        fid = str(f.get("fold_id", "")).strip()
        if fid:
            out[fid] = {
                "train_start": str(f.get("train_start", "")),
                "train_end": str(f.get("train_end", "")),
                "val_start": str(f.get("val_start", "")),
                "val_end": str(f.get("val_end", "")),
                "test_start": str(f.get("test_start", "")),
                "test_end": str(f.get("test_end", "")),
            }
    return out


def _fold_partition(event_date: str, fold: dict[str, str]) -> str | None:
    """Map an event_date to its train/val/test slot under a fold's
    chronological boundaries. Returns ``None`` when the date falls
    outside every window."""

    if not event_date:
        return None
    if fold["train_start"] <= event_date <= fold["train_end"]:
        return "train"
    if fold["val_start"] <= event_date <= fold["val_end"]:
        return "val"
    if fold["test_start"] <= event_date <= fold["test_end"]:
        return "test"
    return None


def _per_fold_partition_counts(
    df: pd.DataFrame, folds: dict[str, dict[str, str]]
) -> dict[str, dict[str, int]]:
    """Returns ``{fold_id: {partition: row_count}}`` using event_date bounds."""

    counts: dict[str, dict[str, int]] = {}
    for fid, fold in folds.items():
        partitions: dict[str, int] = defaultdict(int)
        for event_date in df["event_date"].astype(str):
            slot = _fold_partition(event_date, fold)
            if slot:
                partitions[slot] += 1
        counts[fid] = dict(partitions)
    return counts


def _per_fold_class_balance(
    df: pd.DataFrame,
    folds: dict[str, dict[str, str]],
    *,
    n_classes: int = 3,
) -> dict[str, dict[str, dict[int, int]]]:
    """Per-fold, per-partition class membership counts.

    The class boundary is fitted on each fold's TRAIN slice only --
    matches what the training-loop quantile fitter does, so the audit
    measures the same boundary the optimiser saw.
    """

    out: dict[str, dict[str, dict[int, int]]] = {}
    qs = [(i + 1) / n_classes for i in range(n_classes - 1)]
    for fid, fold in folds.items():
        train_mask = df["event_date"].between(fold["train_start"], fold["train_end"])
        train_vols = df.loc[train_mask, "forward_realized_vol_10d"].dropna()
        if train_vols.empty:
            out[fid] = {}
            continue
        cutoffs = list(train_vols.quantile(qs))
        partition_counts: dict[str, dict[int, int]] = {}
        for partition in ("train", "val", "test"):
            mask = df["event_date"].between(
                fold[f"{partition}_start"], fold[f"{partition}_end"]
            )
            vols = df.loc[mask, "forward_realized_vol_10d"]
            class_counts = defaultdict(int)
            for v in vols:
                if pd.isna(v):
                    class_counts[-1] += 1
                    continue
                cls = next(
                    (i for i, q in enumerate(cutoffs) if v < q), len(cutoffs)
                )
                class_counts[cls] += 1
            partition_counts[partition] = dict(class_counts)
        out[fid] = partition_counts
        out[fid]["__cutoffs__"] = {i: float(c) for i, c in enumerate(cutoffs)}
    return out


def _per_source_counts(df: pd.DataFrame) -> dict[str, int]:
    if "source" not in df.columns:
        return {}
    return df["source"].astype(str).value_counts().to_dict()


def _missingness_audit(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """For every analysis-relevant column, report rows + null fraction."""

    columns = [
        "forward_realized_vol_10d",
        "axis_stance",
        "credibility_drift_score",
        "credibility_realized_vs_stated_gap",
        "credibility_market_implied_gap",
        "credibility_months_since_reversal",
        "mp_surprise_level",
        "mp_surprise_path_factor",
        "fed_info_factor",
        "realized_return",
        "abnormal_return",
        "volatility_shift",
        "direction_t1d",
    ]
    out: dict[str, dict[str, float | int]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        total = len(df)
        nulls = int(df[col].isna().sum())
        out[col] = {
            "total_rows": int(total),
            "null_rows": nulls,
            "null_fraction": round(nulls / total, 6) if total else 0.0,
        }
    return out


def _event_date_timeline(df: pd.DataFrame) -> dict[str, int]:
    """Per-year row counts so the timeline coverage is plot-able."""

    if "event_date" not in df.columns:
        return {}
    years = df["event_date"].astype(str).str[:4]
    return dict(years.value_counts().sort_index().items())


def _quantile_drift_table(
    fold_class_balance: dict[str, dict[str, dict[int, int]]]
) -> dict[str, dict[int, float]]:
    """Surface the per-fold cutoffs in one table so the drift across
    walk-forward folds is visible at a glance."""

    out: dict[str, dict[int, float]] = {}
    for fid, partitions in fold_class_balance.items():
        cutoffs = partitions.get("__cutoffs__", {})
        if cutoffs:
            out[fid] = cutoffs
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="EDA / data-integrity audit (#200)")
    p.add_argument("--training-package-id", required=True)
    p.add_argument("--n-classes", type=int, default=3)
    args = p.parse_args(argv)

    pkg_dir = _resolve_package_dir(args.training_package_id)
    out_dir = _resolve_output_root(args.training_package_id)

    df = pd.read_parquet(pkg_dir / "events.parquet")
    folds = _load_fold_manifest(pkg_dir) or {}

    summary: dict[str, object] = {
        "training_package_id": args.training_package_id,
        "package_dir": str(pkg_dir),
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": sorted(df.columns.tolist()),
        "fold_manifest": folds,
        "per_fold_partition_counts": _per_fold_partition_counts(df, folds),
        "per_fold_class_balance": _per_fold_class_balance(
            df, folds, n_classes=args.n_classes
        ),
        "per_source_counts": _per_source_counts(df),
        "missingness": _missingness_audit(df),
        "event_date_timeline": _event_date_timeline(df),
    }
    summary["per_fold_quantile_drift"] = _quantile_drift_table(
        summary["per_fold_class_balance"]  # type: ignore[arg-type]
    )

    out_path = out_dir / "data_integrity.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {out_path}", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
