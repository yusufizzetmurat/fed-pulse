"""Audit a training package's events.parquet + sidecars for column coverage.

Run before any sweep to confirm every feature family the trainer flag
surface targets is materialised. The sweep loop is expensive — a sparse
or missing column burns GPU hours producing fallback baselines instead
of the methodology cell the operator wanted.

The audit splits checks into two layers:

- **events.parquet columns**: the canonical row schema emitted by
  ``backend/app/data/event_dataset_builder.py`` and validated by
  ``backend/app/data/schemas.py::EventRowSchema``.
- **Sidecar parquets**: lookup files that the training loaders consume
  alongside events.parquet (press-conference Q&A, SEP projections,
  per-asset close caches). These do not live on events.parquet but
  drive feature flags the trainer reads.

Both layers are inventoried; only events.parquet failures gate the
sweep, sidecar gaps are reported as warnings (the trainer's flags
default off and degrade gracefully).

Usage:

    python -m scripts.audit_training_package_coverage \
        --training-package-id tp_v3_phase3_<version>

Exits non-zero when a required events.parquet column is missing or
under-populated. Sidecar gaps are reported but do not fail the audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROCESSED = REPO_ROOT / "data" / "processed"
DEFAULT_EXTERNAL = REPO_ROOT / "data" / "external"

# Sparse threshold: a required column with less than this share of rows
# populated counts as a failure. Set conservatively.
DEFAULT_SPARSE_THRESHOLD_PCT = 50.0

# Required events.parquet column families. Each entry: family label →
# (column names that must exist AND be at least sparse_threshold
# populated, optional issue/PR reference). Column names verified against
# ``backend/app/data/schemas.py`` and the loader sidecars.
#
# Note: press_conference QA, SEP projections, and per-asset close caches
# do NOT live on events.parquet — they live in sidecar parquets and are
# reported under SIDECAR_FILES below.
REQUIRED_EVENT_FAMILIES: dict[str, list[str]] = {
    "canonical_target": [
        "forward_realized_vol_10d",
    ],
    "rates_panel_pre_meeting": [
        "pre_meeting_yield_1y",
        "pre_meeting_yield_2y",
        "pre_meeting_yield_5y",
        "pre_meeting_yield_10y",
    ],
    "rates_panel_change_5d": [
        "yield_2y_change_5d",
        "yield_5y_change_5d",
        "terminal_rate_change_5d",
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
}

# Optional events.parquet column families: presence reported, absence
# does NOT fail the audit. Reflects feature additions that may or may
# not have a builder firing on the current rebuild.
OPTIONAL_EVENT_FAMILIES: dict[str, list[str]] = {
    "multi_horizon_vol": [
        "forward_realized_vol_1d",
        "forward_realized_vol_3d",
        "forward_realized_vol_5d",
        "forward_realized_vol_20d",
        "forward_realized_vol_30d",
    ],
    "per_asset_vol": [
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

# Sidecar parquet files the training loader reads alongside events.parquet.
# Each entry: family label → (relative path under the training-package dir
# OR a list of fallback paths under data/external/, required columns on
# the sidecar). Missing sidecars do not fail the audit — they degrade
# trainer flags that default off — but the report surfaces them so the
# operator knows which feature flags will silently no-op.
SIDECAR_FILES: dict[str, dict[str, object]] = {
    "press_conference_qa": {
        "package_path": "qa_lookup.parquet",
        "external_paths": ["fomc_press_conferences/qa_lookup.parquet"],
        "expected_columns": ["qa_text", "has_press_conf"],
    },
    "sep_projections": {
        "package_path": "sep_projections.parquet",
        "external_paths": ["fred/sep_projections.parquet"],
        "expected_columns": [
            "ffr_median_current_year",
            "ffr_median_next_year",
            "ffr_median_longer_run",
        ],
    },
}

# Canonical event_kind values per ``backend/app/data/schemas.py::
# _ALLOWED_EVENT_KIND``. event_kind is the document-type axis (what kind
# of FOMC artefact is this row) and is distinct from source_type (which
# data provider). The narrower event_kind set above is what
# events.parquet rows carry.
EXPECTED_EVENT_KINDS = {
    "statement",
    "minutes",
    "press_conference",
    "speech",
    "testimony",
    "macro_release",
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


def _audit_event_families(
    df: pd.DataFrame, sparse_threshold: float
) -> list[str]:
    """Walk required event-schema families. Return list of failure
    descriptions; empty list means everything passed.
    """
    failures: list[str] = []

    print("=== REQUIRED EVENTS.PARQUET FAMILIES ===")
    for family, columns in REQUIRED_EVENT_FAMILIES.items():
        print(f"\n[{family}]")
        for column in columns:
            status, ok, n_pop, n = _check_column(df, column, sparse_threshold)
            marker = "OK " if ok else "FAIL"
            line = (
                f"  {marker} {column:55s} {_format_pct(n_pop, n)} "
                f"({n_pop}/{n}) [{status}]"
            )
            print(line)
            if not ok:
                failures.append(f"{family}: {column} ({status})")

    print("\n=== OPTIONAL EVENTS.PARQUET FAMILIES (presence reported) ===")
    n_total = len(df)
    for family, columns in OPTIONAL_EVENT_FAMILIES.items():
        print(f"\n[{family}]")
        for column in columns:
            if column in df.columns:
                n_pop = int(df[column].notna().sum())
                print(
                    f"  PRESENT {column:53s} {_format_pct(n_pop, n_total)} "
                    f"({n_pop}/{n_total})"
                )
            else:
                print(f"  ABSENT  {column}")

    return failures


def _audit_event_kind_distribution(df: pd.DataFrame) -> None:
    """Print the event_kind value_counts. The current schema uses the
    ``event_kind`` column for the corpus-diversity axis; the canonical
    catalog lives in ``backend/app/data/source_type.py``. Missing event
    kinds are reported but do not fail the audit — they reflect
    upstream-data gaps tracked separately.
    """
    print("\n=== EVENT_KIND DISTRIBUTION ===")
    if "event_kind" not in df.columns:
        print("  event_kind column not present on this training package")
        return
    counts = df["event_kind"].value_counts(dropna=False)
    for kind, count in counts.items():
        in_expected = (
            "canonical" if str(kind) in EXPECTED_EVENT_KINDS else "unknown"
        )
        print(f"  {str(kind):40s} {count:6d}  [{in_expected}]")
    missing = EXPECTED_EVENT_KINDS - set(map(str, counts.index))
    if missing:
        print("\n  Canonical event kinds with zero rows on this package:")
        for kind in sorted(missing):
            print(f"    - {kind}")
        print(
            "\n  (Adapter ingestion gaps are tracked in #485; missing "
            "event_kinds do NOT fail the audit.)"
        )


def _resolve_sidecar(
    package_dir: Path, package_path: str, external_paths: list[str]
) -> tuple[Path | None, str]:
    """Return (resolved path or None, source label).

    Walks the package directory first (sidecar shipped with the TP),
    then falls back to the repo's data/external/ tree.
    """
    candidate = package_dir / package_path
    if candidate.exists():
        return candidate, "package"
    for relative in external_paths:
        candidate = DEFAULT_EXTERNAL / relative
        if candidate.exists():
            return candidate, "external"
    return None, "missing"


def _audit_sidecars(package_dir: Path, sparse_threshold: float) -> None:
    """Walk sidecar parquets. Report-only; sidecar gaps do not fail
    the audit (the corresponding trainer flags default off).
    """
    print("\n=== SIDECAR PARQUETS (report only — flags default off) ===")
    for family, spec in SIDECAR_FILES.items():
        print(f"\n[{family}]")
        pkg_path = str(spec["package_path"])
        ext_paths = list(spec["external_paths"])  # type: ignore[arg-type]
        expected = list(spec["expected_columns"])  # type: ignore[arg-type]
        resolved, source_label = _resolve_sidecar(
            package_dir, pkg_path, ext_paths
        )
        if resolved is None:
            print(f"  ABSENT  expected at {pkg_path} (package) or one of:")
            for relative in ext_paths:
                print(f"            data/external/{relative}")
            continue
        print(f"  PRESENT {resolved.relative_to(REPO_ROOT)} [{source_label}]")
        try:
            sidecar_df = pd.read_parquet(resolved)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR   could not read sidecar: {exc}")
            continue
        n_rows = len(sidecar_df)
        for column in expected:
            if column not in sidecar_df.columns:
                print(f"  MISS    {column}  (sidecar column missing)")
                continue
            n_pop = int(sidecar_df[column].notna().sum())
            print(
                f"  COL     {column:50s} {_format_pct(n_pop, n_rows)} "
                f"({n_pop}/{n_rows})"
            )


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

    failures = _audit_event_families(df, sparse_threshold)
    _audit_event_kind_distribution(df)
    _audit_sidecars(events_parquet.parent, sparse_threshold)

    print()
    if failures:
        print("=== AUDIT FAILED ===")
        for line in failures:
            print(f"  - {line}")
        print(
            "\nDo not run sweeps against this training package until the "
            "failures are resolved. Either re-run the ingestion + data-prep "
            "rebuild, or drop the affected sweep from the batch."
        )
        return 1

    print("=== AUDIT PASSED ===")
    print(
        "Every required events.parquet column is present and populated "
        "above the sparse threshold. Sidecar gaps and event-kind coverage "
        "are reported above; address them per #485 if the cross-source "
        "matrix is on the sweep batch."
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a training package's events.parquet + sidecar coverage "
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
            "audit (default 50.0). At 0.0 any non-empty column passes; at "
            "100.0 only fully-populated columns pass."
        ),
    )
    parser.add_argument(
        "--json-summary",
        type=Path,
        required=False,
        help="Optional path to write a one-line JSON summary of the audit.",
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
    exit_code = audit(target, sparse_threshold=args.sparse_threshold_pct)
    if args.json_summary:
        args.json_summary.parent.mkdir(parents=True, exist_ok=True)
        args.json_summary.write_text(
            json.dumps(
                {
                    "events_parquet": str(target),
                    "exit_code": exit_code,
                    "sparse_threshold_pct": args.sparse_threshold_pct,
                }
            )
        )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
