"""Audit a training package's events.parquet for column coverage.

Run before any sweep to confirm every feature family the trainer flag
surface targets is materialised on the dataset. The sweep loop is
expensive — a sparse or missing column burns GPU hours producing
fallback baselines instead of the methodology cell the operator wanted.

Usage:

    python -m scripts.audit_training_package_coverage \
        --training-package-id tp_v3_phase3_<version>

Exits non-zero when a required column is missing or under-populated.
Optional columns are reported but do not fail the audit (they reflect
ongoing data work).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROCESSED = REPO_ROOT / "data" / "processed"

# Sparse threshold: a required column with less than this share of rows
# populated counts as a failure. Set conservatively — most columns
# should be populated on most rows.
DEFAULT_SPARSE_THRESHOLD_PCT = 50.0

# Required column families. Each entry: family label → list of column
# names that must exist AND be at least `SPARSE_THRESHOLD_PCT` populated.
REQUIRED_FAMILIES: dict[str, list[str]] = {
    "rates_panel": [
        "forward_yield_2y_change_5d",
        "forward_yield_terminal_change_5d",
    ],
    "garch_residual": [
        "forward_realized_vol_10d_garch_baseline",
        "forward_realized_vol_10d_garch_residual",
    ],
    "statement_delta": [
        "statement_delta_inserted",
        "statement_delta_deleted",
        "statement_delta_substituted_pairs",
        "statement_delta_embedding",
    ],
    "vote_tally": [
        "votes_for",
        "votes_against",
        "dissent_count",
        "dissent_direction",
    ],
    "press_conference": [
        "qa_text",
        "qa_embedding",
        "has_press_conf",
    ],
    "sep_next_year": [
        "ffr_median_next_year",
    ],
    "canonical_target": [
        "forward_realized_vol_10d",
    ],
}

# Optional column families: presence reported but absence does NOT fail.
# These reflect feature additions that may or may not be on the current
# dataset version depending on which builders ran.
OPTIONAL_FAMILIES: dict[str, list[str]] = {
    "multi_horizon": [
        "forward_realized_vol_1d",
        "forward_realized_vol_3d",
        "forward_realized_vol_5d",
        "forward_realized_vol_20d",
        "forward_realized_vol_30d",
    ],
    "per_asset_targets": [
        "forward_realized_vol_10d_gspc",
        "forward_realized_vol_10d_ndx",
        "forward_realized_vol_10d_dji",
        "forward_realized_vol_10d_dxy",
        "forward_realized_vol_10d_vix",
        "forward_realized_vol_10d_eurusd",
        "forward_realized_vol_10d_usdjpy",
        "forward_realized_vol_10d_gbpusd",
    ],
}

EXPECTED_SOURCE_TYPES = {
    "fomc_statement",
    "fomc_minutes",
    "fomc_meeting_transcript",
    "press_conference",
    "speech",
    "testimony",
    "beige_book",
    "ny_fed",
    "op_fed",
    "gss_factor_decomposition",
}


def _format_pct(n_pop: int, n_total: int) -> str:
    pct = 100.0 * n_pop / n_total if n_total else 0.0
    return f"{pct:5.1f}%"


def _check_column(
    df: pd.DataFrame, column: str, sparse_threshold: float
) -> tuple[str, bool, int, int]:
    """Return (status, ok, n_populated, n_total).

    status is one of: 'missing', 'empty', 'sparse', 'ok'. ok is False
    when the column should fail the audit.
    """
    n = len(df)
    if column not in df.columns:
        return "missing", False, 0, n
    populated = df[column].notna().sum()
    populated_int = int(populated)
    if populated_int == 0:
        return "empty", False, 0, n
    pct = 100.0 * populated_int / n if n else 0.0
    if pct < sparse_threshold:
        return "sparse", False, populated_int, n
    return "ok", True, populated_int, n


def audit(
    events_parquet: Path, sparse_threshold: float = DEFAULT_SPARSE_THRESHOLD_PCT
) -> int:
    if not events_parquet.exists():
        print(f"events.parquet not found at {events_parquet}", file=sys.stderr)
        return 2

    df = pd.read_parquet(events_parquet)
    n_total = len(df)
    print(f"events.parquet: {events_parquet}")
    print(f"rows: {n_total}, columns: {len(df.columns)}")
    print()

    failures: list[str] = []

    print("=== REQUIRED FAMILIES ===")
    for family, columns in REQUIRED_FAMILIES.items():
        print(f"\n[{family}]")
        for column in columns:
            status, ok, n_pop, n = _check_column(df, column, sparse_threshold)
            marker = "OK " if ok else "FAIL"
            line = (
                f"  {marker} {column:50s} {_format_pct(n_pop, n)} ({n_pop}/{n}) [{status}]"
            )
            print(line)
            if not ok:
                failures.append(f"{family}: {column} ({status})")

    print("\n=== OPTIONAL FAMILIES (presence reported) ===")
    for family, columns in OPTIONAL_FAMILIES.items():
        print(f"\n[{family}]")
        for column in columns:
            if column in df.columns:
                n_pop = int(df[column].notna().sum())
                print(
                    f"  PRESENT {column:48s} {_format_pct(n_pop, n_total)} ({n_pop}/{n_total})"
                )
            else:
                print(f"  ABSENT  {column}")

    print("\n=== SOURCE TYPE COVERAGE ===")
    if "source_type" in df.columns:
        counts = df["source_type"].value_counts(dropna=False)
        for kind, count in counts.items():
            in_expected = (
                "expected" if str(kind) in EXPECTED_SOURCE_TYPES else "unexpected"
            )
            print(f"  {str(kind):40s} {count:6d}  [{in_expected}]")
        missing_kinds = EXPECTED_SOURCE_TYPES - set(map(str, counts.index))
        if missing_kinds:
            print("\n  Expected source_types with zero rows:")
            for kind in sorted(missing_kinds):
                print(f"    - {kind}")
    else:
        print("  source_type column not present")
        failures.append("source_type: column missing")

    print()
    if failures:
        print("=== AUDIT FAILED ===")
        for line in failures:
            print(f"  - {line}")
        print(
            "\nDo not run sweeps against this training package until the failures "
            "are resolved. Either re-run the source ingestion + data-prep "
            "rebuild, or drop the affected sweep from the batch."
        )
        return 1

    print("=== AUDIT PASSED ===")
    print("Every required column is present and populated above the sparse threshold.")
    print("Optional columns are reported above; absent ones reflect ongoing data work.")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a training package's events.parquet for column coverage "
            "before running a sweep."
        )
    )
    parser.add_argument(
        "--training-package-id",
        type=str,
        required=False,
        help=(
            "Training package id; resolves to "
            "data/processed/<id>/events.parquet. Mutually exclusive with "
            "--events-parquet."
        ),
    )
    parser.add_argument(
        "--events-parquet",
        type=Path,
        required=False,
        help="Direct path to an events.parquet file.",
    )
    parser.add_argument(
        "--sparse-threshold-pct",
        type=float,
        default=DEFAULT_SPARSE_THRESHOLD_PCT,
        help=(
            "Required columns populated below this share of rows fail the "
            "audit (default 50.0)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.training_package_id and not args.events_parquet:
        print(
            "one of --training-package-id or --events-parquet is required",
            file=sys.stderr,
        )
        return 2
    if args.events_parquet:
        target = args.events_parquet
    else:
        target = DEFAULT_PROCESSED / args.training_package_id / "events.parquet"
    return audit(target, sparse_threshold=args.sparse_threshold_pct)


if __name__ == "__main__":
    sys.exit(main())
