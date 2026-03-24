from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path("/data") if Path("/data").exists() else BACKEND_ROOT.parent / "data"
DEFAULT_INPUT = DEFAULT_DATA_DIR / "interim" / "phase2" / "registry_labeled.jsonl"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "interim" / "phase2" / "registry_quality_passed.jsonl"
DEFAULT_REPORT_DIR = DEFAULT_DATA_DIR / "interim" / "phase2" / "quality_reports"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 quality and leakage checks with exact and near-duplicate blocking."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Labeled registry JSONL input.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Quality-passed registry JSONL output.")
    parser.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="Output directory for dedup/leakage reports.",
    )
    parser.add_argument(
        "--near-threshold",
        type=float,
        default=0.97,
        help="SequenceMatcher threshold for near-duplicate blocking.",
    )
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


def _is_valid_date(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except Exception:
        return False


def _exact_dedup(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    seen: dict[str, str] = {}
    for row in rows:
        key = f"{row.get('event_date','')}::{row.get('text_hash','')}"
        rid = str(row.get("record_id", ""))
        if key in seen:
            dropped.append({"record_id": rid, "kept_record_id": seen[key], "reason": "exact_text_hash_duplicate"})
            continue
        seen[key] = rid
        kept.append(row)
    report = {
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "dropped_rows": len(dropped),
        "dropped": dropped[:5000],
    }
    return kept, report


def _near_dedup(rows: list[dict[str, Any]], threshold: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_date[str(row.get("event_date", ""))].append(row)

    kept: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    for _, group in by_date.items():
        group_sorted = sorted(group, key=lambda r: (str(r.get("source", "")), str(r.get("record_id", ""))))
        survivors: list[dict[str, Any]] = []
        for row in group_sorted:
            text = str(row.get("text", ""))
            rid = str(row.get("record_id", ""))
            duplicate_of = None
            best_ratio = 0.0
            for existing in survivors:
                ratio = SequenceMatcher(None, text, str(existing.get("text", ""))).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                if ratio >= threshold:
                    duplicate_of = str(existing.get("record_id", ""))
                    break
            if duplicate_of:
                blocked.append(
                    {
                        "record_id": rid,
                        "blocked_by": duplicate_of,
                        "event_date": str(row.get("event_date", "")),
                        "similarity": round(best_ratio, 5),
                        "reason": "near_duplicate_blocked",
                    }
                )
                continue
            survivors.append(row)
        kept.extend(survivors)

    report = {
        "threshold": threshold,
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "blocked_rows": len(blocked),
        "blocked": blocked[:5000],
    }
    return kept, report


def _leakage_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    invalid_dates = [str(r.get("record_id", "")) for r in rows if not _is_valid_date(str(r.get("event_date", "")))]
    source_record_dupes: list[dict[str, str]] = []
    seen: dict[str, str] = {}
    for row in rows:
        key = f"{row.get('source','')}::{row.get('source_record_id','')}"
        rid = str(row.get("record_id", ""))
        if key in seen:
            source_record_dupes.append({"record_id": rid, "first_seen_record_id": seen[key], "key": key})
        else:
            seen[key] = rid
    return {
        "invalid_event_dates": invalid_dates[:5000],
        "source_record_duplicates": source_record_dupes[:5000],
        "status": "pass" if not invalid_dates and not source_record_dupes else "fail",
    }


def _distribution_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts = Counter(str(r.get("source", "")) for r in rows)
    label_counts = Counter(str(r.get("mapped_label", "")) for r in rows if r.get("mapped_label"))
    source_label_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        source = str(row.get("source", ""))
        label = str(row.get("mapped_label", "")) or "unlabeled"
        source_label_counts.setdefault(source, {})
        source_label_counts[source][label] = source_label_counts[source].get(label, 0) + 1
    return {
        "source_counts": dict(source_counts),
        "mapped_label_counts": dict(label_counts),
        "source_label_counts": source_label_counts,
    }


def main() -> int:
    warnings.warn(
        "app.data.quality_checks is deprecated. Use app.data.quality_validation instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(input_path)
    if not rows:
        print(f"No input rows found at {input_path}")
        return 1

    exact_kept, exact_report = _exact_dedup(rows)
    near_kept, near_report = _near_dedup(exact_kept, threshold=args.near_threshold)
    leakage = _leakage_report(near_kept)
    distribution = _distribution_report(near_kept)

    _write_jsonl(output_path, near_kept)
    (report_dir / "dedup_report.json").write_text(json.dumps(exact_report, indent=2), encoding="utf-8")
    (report_dir / "near_duplicate_report.json").write_text(json.dumps(near_report, indent=2), encoding="utf-8")
    (report_dir / "leakage_report.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    (report_dir / "distribution_report.json").write_text(json.dumps(distribution, indent=2), encoding="utf-8")

    print(f"Quality-passed registry written to {output_path}")
    print(f"Reports written to {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

