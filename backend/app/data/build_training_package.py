from __future__ import annotations

import argparse
import json
import math
import shutil
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path("/data") if Path("/data").exists() else BACKEND_ROOT.parent / "data"
DEFAULT_INPUT = DEFAULT_DATA_DIR / "interim" / "phase2" / "registry_quality_passed.jsonl"
DEFAULT_REPORT_DIR = DEFAULT_DATA_DIR / "interim" / "phase2" / "quality_reports"
DEFAULT_PROCESSED_ROOT = DEFAULT_DATA_DIR / "processed"
DEFAULT_PROTOCOL = "evaluation_protocol_v1"


@dataclass
class FoldRange:
    fold_id: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str
    train_rows: int
    val_rows: int
    test_rows: int
    class_distribution: dict[str, int]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build versioned Phase 2 training package with train/val/test splits and walk-forward folds."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Quality-passed JSONL registry.")
    parser.add_argument(
        "--quality-report-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="Quality report directory to copy into package.",
    )
    parser.add_argument("--dataset-version", required=True, help="Dataset version identifier.")
    parser.add_argument("--feature-version", required=True, help="Feature version identifier.")
    parser.add_argument("--training-package-id", default="", help="Training package id. Auto-generated when omitted.")
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL, help="Evaluation protocol identifier.")
    parser.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT), help="Processed package root.")
    parser.add_argument("--min-train-ratio", type=float, default=0.6, help="Min train date ratio for fold seed.")
    parser.add_argument("--fold-count", type=int, default=5, help="Target walk-forward fold count.")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _auto_training_package_id(dataset_version: str, feature_version: str, protocol: str) -> str:
    protocol_short = "epv1" if protocol == "evaluation_protocol_v1" else protocol
    return f"tp_{dataset_version}_{feature_version}_{protocol_short}_v1.0"


def _time_split(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    n = len(rows)
    train_end = max(1, int(n * 0.7))
    val_end = max(train_end + 1, int(n * 0.85))
    val_end = min(val_end, n)
    return rows[:train_end], rows[train_end:val_end], rows[val_end:]


def _rows_between(rows: list[dict[str, Any]], start_date: str, end_date: str) -> list[dict[str, Any]]:
    return [r for r in rows if start_date <= str(r.get("event_date", "")) <= end_date]


def _build_folds(rows: list[dict[str, Any]], min_train_ratio: float, fold_count: int) -> list[FoldRange]:
    unique_dates = sorted({str(r.get("event_date", "")) for r in rows if r.get("event_date")})
    if len(unique_dates) < 8:
        return []

    min_train_dates = max(3, int(len(unique_dates) * min_train_ratio))
    remaining = len(unique_dates) - min_train_dates
    window = max(1, remaining // max(fold_count + 1, 2))

    folds: list[FoldRange] = []
    for i in range(1, fold_count + 1):
        train_end_idx = min_train_dates + (i - 1) * window - 1
        val_start_idx = train_end_idx + 1
        val_end_idx = val_start_idx + window - 1
        test_start_idx = val_end_idx + 1
        test_end_idx = test_start_idx + window - 1
        if test_end_idx >= len(unique_dates):
            break

        train_dates = (unique_dates[0], unique_dates[train_end_idx])
        val_dates = (unique_dates[val_start_idx], unique_dates[val_end_idx])
        test_dates = (unique_dates[test_start_idx], unique_dates[test_end_idx])

        train_rows = _rows_between(rows, *train_dates)
        val_rows = _rows_between(rows, *val_dates)
        test_rows = _rows_between(rows, *test_dates)
        cls_count = Counter(str(r.get("mapped_label", "")) for r in test_rows if r.get("mapped_label"))

        folds.append(
            FoldRange(
                fold_id=f"wf_fold_{i}",
                train_start=train_dates[0],
                train_end=train_dates[1],
                val_start=val_dates[0],
                val_end=val_dates[1],
                test_start=test_dates[0],
                test_end=test_dates[1],
                train_rows=len(train_rows),
                val_rows=len(val_rows),
                test_rows=len(test_rows),
                class_distribution=dict(cls_count),
            )
        )
    return folds


def _maybe_write_parquet(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return False
    try:
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        return True
    except Exception:
        return False


def main() -> int:
    warnings.warn(
        "app.data.build_training_package is deprecated. Use app.data.training_package_builder instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    args = _parse_args()
    input_path = Path(args.input)
    report_dir = Path(args.quality_report_dir)
    processed_root = Path(args.processed_root)
    rows = _read_jsonl(input_path)
    if not rows:
        print(f"No quality-passed rows found at {input_path}")
        return 1

    # Official NLP package should contain mapped labels only.
    supervised_rows = [r for r in rows if str(r.get("mapped_label", "")).strip()]
    supervised_rows.sort(key=lambda r: (str(r.get("event_date", "")), str(r.get("record_id", ""))))
    if len(supervised_rows) < 10:
        print("Insufficient mapped rows to build training package.")
        return 1

    training_package_id = args.training_package_id or _auto_training_package_id(
        args.dataset_version, args.feature_version, args.protocol
    )
    package_dir = processed_root / training_package_id
    quality_out_dir = package_dir / "quality_reports"
    package_dir.mkdir(parents=True, exist_ok=True)
    quality_out_dir.mkdir(parents=True, exist_ok=True)

    # Base registry artifact
    registry_jsonl = package_dir / "registry_normalized.jsonl"
    _write_jsonl(registry_jsonl, supervised_rows)
    parquet_written = _maybe_write_parquet(package_dir / "registry_normalized.parquet", supervised_rows)

    # Train/val/test split artifact
    train_rows, val_rows, test_rows = _time_split(supervised_rows)
    split_rows: list[dict[str, Any]] = []
    for row in train_rows:
        split_rows.append({**row, "split_tag": "train"})
    for row in val_rows:
        split_rows.append({**row, "split_tag": "val"})
    for row in test_rows:
        split_rows.append({**row, "split_tag": "test"})
    splits_jsonl = package_dir / "splits_train_val_test.jsonl"
    _write_jsonl(splits_jsonl, split_rows)
    _maybe_write_parquet(package_dir / "splits_train_val_test.parquet", split_rows)

    folds = _build_folds(supervised_rows, min_train_ratio=args.min_train_ratio, fold_count=args.fold_count)
    fold_manifest = {
        "evaluation_protocol": args.protocol,
        "dataset_version": args.dataset_version,
        "feature_version": args.feature_version,
        "training_package_id": training_package_id,
        "fold_count": len(folds),
        "folds": [asdict(fold) for fold in folds],
    }
    (package_dir / "fold_manifest_expanding_walk_forward.json").write_text(
        json.dumps(fold_manifest, indent=2), encoding="utf-8"
    )

    # Copy quality reports
    if report_dir.exists():
        for item in report_dir.glob("*.json"):
            shutil.copy2(item, quality_out_dir / item.name)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_protocol": args.protocol,
        "dataset_version": args.dataset_version,
        "feature_version": args.feature_version,
        "training_package_id": training_package_id,
        "input_rows": len(rows),
        "supervised_rows": len(supervised_rows),
        "split_counts": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "source_counts": dict(Counter(str(r.get("source", "")) for r in supervised_rows)),
        "label_counts": dict(Counter(str(r.get("mapped_label", "")) for r in supervised_rows)),
        "registry_parquet_written": parquet_written,
    }
    (package_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Training package created: {package_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

