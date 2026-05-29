"""Per-column population audit on events.parquet (#505 A.1.a).

For every column declared in ``backend/app/data/schemas.py::_EVENT_ROW_COLUMNS``
that's actually present on the parquet, compute:

- ``non_null_rate`` overall + per fold + per event_kind + per source
- ``n_distinct``
- ``std`` (numeric columns only)

Flag any ``required=True`` column whose empirical non-null rate is below
the configured threshold (default 100%). Exit non-zero if any required
column fails the gate, so the script can be wired into the sweep-launch
pre-flight.

Output:

- ``column_population.csv`` — long-format, one row per (column, slice).
- ``summary.md`` — human-readable summary of the gate result + flagged
  columns.
- ``summary.json`` — machine-readable summary for CI / dashboards.

Usage:

    python -m scripts.run_pre_sweep_column_audit \\
        --events-parquet data/processed/<tp>/events.parquet \\
        --fold-manifest data/processed/<tp>/fold_manifest_expanding_walk_forward.json \\
        --output-dir backend/artifacts/audits/pre_sweep_<date>/

Both ``--fold-manifest`` and ``--output-dir`` are optional. Without the
fold manifest, per-fold breakdowns are skipped (the report header notes
this) and the gate continues to run on the overall populations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_THRESHOLD_PCT = 100.0


def _load_event_schema_columns() -> dict[str, dict[str, bool]]:
    """Return a mapping of column name -> {required, nullable} from the
    pandera schema. Imported lazily so the script can be smoke-tested
    on machines without the full backend dependency set.
    """

    from app.data.schemas import _EVENT_ROW_COLUMNS  # noqa: PLC0415

    return {
        name: {"required": bool(col.required), "nullable": bool(col.nullable)}
        for name, col in _EVENT_ROW_COLUMNS.items()
    }


def _load_fold_manifest(
    path: Path,
    event_dates: pd.Series | None = None,
) -> dict[str, list[str]] | None:
    """Return ``{fold_id: [event_date strings in that fold's TEST split]}``.

    The production manifest (``fold_manifest_expanding_walk_forward.json``)
    declares each fold as a ``[test_start, test_end]`` ISO-date range, not
    as an enumerated event list — so the reader expands the range against
    ``event_dates`` (the ``event_date`` column from the parquet under
    audit) and emits the intersection per fold. We also accept the legacy
    enumerated forms (``test`` / ``test_event_dates``) so older fixtures
    keep working.

    Returns ``None`` when the manifest can't be parsed or no fold could be
    resolved; the audit falls back to overall-only populations in that
    case.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    folds = payload.get("folds")
    if not isinstance(folds, list):
        return None

    date_strings: list[str] | None = None
    if event_dates is not None:
        date_strings = [str(d) for d in event_dates.tolist() if d is not None]

    out: dict[str, list[str]] = {}
    for entry in folds:
        if not isinstance(entry, dict):
            continue
        fold_id = entry.get("fold_id") or entry.get("id")
        if not fold_id:
            continue
        legacy_dates = entry.get("test") or entry.get("test_event_dates")
        if isinstance(legacy_dates, list):
            out[str(fold_id)] = [str(d) for d in legacy_dates if d]
            continue
        test_start = entry.get("test_start")
        test_end = entry.get("test_end")
        if not test_start or not test_end or date_strings is None:
            continue
        lo, hi = str(test_start), str(test_end)
        out[str(fold_id)] = [d for d in date_strings if lo <= d <= hi]
    return out or None


def _populate_per_slice(
    df: pd.DataFrame,
    column: str,
    slice_label: str,
    slice_filter: pd.Series | None,
    rows: list[dict[str, Any]],
) -> None:
    """Append one row to the report for ``(column, slice_label)``."""

    if slice_filter is not None:
        slice_df = df.loc[slice_filter, column]
    else:
        slice_df = df[column]
    total = int(len(slice_df))
    if total == 0:
        rows.append(
            {
                "column": column,
                "slice_kind": slice_label.split(":", 1)[0],
                "slice_value": slice_label.split(":", 1)[-1]
                if ":" in slice_label
                else "",
                "rows": 0,
                "non_null": 0,
                "non_null_rate": None,
                "n_distinct": 0,
                "std": None,
            }
        )
        return
    non_null = int(slice_df.notna().sum())
    rate = float(non_null / total)
    n_distinct = int(slice_df.dropna().nunique())
    std: float | None = None
    if pd.api.types.is_numeric_dtype(slice_df):
        std_value = slice_df.std(skipna=True)
        if pd.notna(std_value):
            std = float(std_value)
    rows.append(
        {
            "column": column,
            "slice_kind": slice_label.split(":", 1)[0],
            "slice_value": slice_label.split(":", 1)[-1]
            if ":" in slice_label
            else "",
            "rows": total,
            "non_null": non_null,
            "non_null_rate": rate,
            "n_distinct": n_distinct,
            "std": std,
        }
    )


def audit_column_populations(
    events_parquet: Path,
    fold_manifest: Path | None = None,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the column-population audit and return ``(df, summary)``.

    ``df`` is the long-format population CSV. ``summary`` carries the
    pass/fail verdict + the list of flagged required columns.
    """

    if not events_parquet.exists():
        raise SystemExit(f"events.parquet not found at {events_parquet}")

    schema_cols = _load_event_schema_columns()
    df_events = pd.read_parquet(events_parquet)

    fold_test_dates: dict[str, list[str]] | None = None
    if fold_manifest is not None and fold_manifest.exists():
        event_dates = (
            df_events["event_date"].astype(str)
            if "event_date" in df_events.columns
            else None
        )
        fold_test_dates = _load_fold_manifest(fold_manifest, event_dates)

    rows: list[dict[str, Any]] = []
    flagged_required: list[dict[str, Any]] = []

    threshold_rate = float(threshold_pct) / 100.0

    event_kind_values = (
        sorted(df_events["event_kind"].dropna().unique().tolist())
        if "event_kind" in df_events.columns
        else []
    )
    source_values = (
        sorted(df_events["source"].dropna().unique().tolist())
        if "source" in df_events.columns
        else []
    )

    # Build per-slice filters once so we don't re-evaluate per column.
    slice_filters: list[tuple[str, pd.Series | None]] = [("overall:", None)]
    for kind in event_kind_values:
        slice_filters.append(
            (f"event_kind:{kind}", df_events["event_kind"] == kind)
        )
    for src in source_values:
        slice_filters.append((f"source:{src}", df_events["source"] == src))
    if fold_test_dates is not None and "event_date" in df_events.columns:
        evdates = df_events["event_date"].astype(str)
        for fold_id, dates in fold_test_dates.items():
            slice_filters.append(
                (f"fold:{fold_id}", evdates.isin(dates))
            )

    advisory_nullable: list[dict[str, Any]] = []

    for column in sorted(df_events.columns):
        for slice_label, slice_filter in slice_filters:
            _populate_per_slice(
                df_events, column, slice_label, slice_filter, rows
            )

        if column not in schema_cols:
            continue
        if not schema_cols[column]["required"]:
            continue
        # The strict gate fires only when the schema marks the column
        # as both required AND non-nullable. Columns declared
        # ``nullable=True`` are populated only on the source/window
        # subset that emits them (e.g. ``axis_stance`` ships on HF-style
        # corpora; ``realized_return`` is null when the prior price
        # window won't resolve), so flagging them against an overall
        # non-null floor double-counts a population gap the schema
        # already accepts. We still report them as advisory.
        non_null = int(df_events[column].notna().sum())
        non_null_rate = float(non_null / max(len(df_events), 1))
        if non_null_rate + 1e-12 >= threshold_rate:
            continue
        entry = {
            "column": column,
            "non_null_rate": non_null_rate,
            "n_rows": int(len(df_events)),
            "n_non_null": non_null,
            "threshold_rate": threshold_rate,
        }
        if schema_cols[column]["nullable"]:
            advisory_nullable.append(entry)
        else:
            flagged_required.append(entry)

    schema_only_columns = [
        c for c in schema_cols if c not in df_events.columns
    ]
    schema_only_required_missing = [
        c for c in schema_only_columns if schema_cols[c]["required"]
    ]

    summary: dict[str, Any] = {
        "events_parquet": str(events_parquet),
        "n_rows": int(len(df_events)),
        "n_columns": int(len(df_events.columns)),
        "schema_columns_present": sorted(
            set(df_events.columns) & set(schema_cols)
        ),
        "schema_columns_absent": sorted(schema_only_columns),
        "required_columns_missing_from_parquet": sorted(
            schema_only_required_missing
        ),
        "required_columns_under_threshold": flagged_required,
        "nullable_required_columns_under_threshold": advisory_nullable,
        "threshold_pct": float(threshold_pct),
        "fold_manifest_used": bool(fold_test_dates),
        "pass": (
            len(schema_only_required_missing) == 0 and len(flagged_required) == 0
        ),
    }

    return pd.DataFrame(rows), summary


def _render_summary_md(summary: dict[str, Any]) -> str:
    """Render the human-readable summary."""

    verdict = "PASS" if summary["pass"] else "FAIL"
    lines: list[str] = []
    lines.append(f"# Pre-sweep column audit — {verdict}")
    lines.append("")
    lines.append(f"- events.parquet: `{summary['events_parquet']}`")
    lines.append(
        f"- rows: {summary['n_rows']} · columns on parquet: {summary['n_columns']}"
    )
    lines.append(f"- threshold for required-column gate: {summary['threshold_pct']}%")
    lines.append(
        "- per-fold breakdown: "
        + ("yes" if summary["fold_manifest_used"] else "no (fold manifest not loaded)")
    )
    lines.append("")
    if summary["required_columns_missing_from_parquet"]:
        lines.append("## Required columns absent from events.parquet")
        for col in summary["required_columns_missing_from_parquet"]:
            lines.append(f"- `{col}`")
        lines.append("")
    if summary["required_columns_under_threshold"]:
        lines.append(
            f"## Required columns below {summary['threshold_pct']}% non-null"
        )
        for entry in summary["required_columns_under_threshold"]:
            pct = 100.0 * entry["non_null_rate"]
            lines.append(
                f"- `{entry['column']}` — {pct:.2f}% "
                f"({entry['n_non_null']}/{entry['n_rows']})"
            )
        lines.append("")
    advisory = summary.get("nullable_required_columns_under_threshold") or []
    if advisory:
        lines.append(
            f"## Advisory: nullable required columns below {summary['threshold_pct']}% non-null"
        )
        lines.append(
            "These columns are declared ``nullable=True`` in the schema, so "
            "the population gap is allowed by design and does not fail the "
            "gate. Listed for source/window-mix visibility."
        )
        for entry in advisory:
            pct = 100.0 * entry["non_null_rate"]
            lines.append(
                f"- `{entry['column']}` — {pct:.2f}% "
                f"({entry['n_non_null']}/{entry['n_rows']})"
            )
        lines.append("")
    if summary["pass"]:
        lines.append(
            "All schema columns required as non-nullable are present and at full population."
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Per-column population audit on events.parquet (#505 A.1.a). "
            "Gates the next sweep on required-column populations."
        )
    )
    parser.add_argument(
        "--events-parquet",
        type=Path,
        required=True,
        help="Path to events.parquet for the training package being audited.",
    )
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=None,
        help=(
            "Optional fold manifest (JSON) to enable per-fold breakdowns. "
            "Defaults to skip per-fold."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Where to write column_population.csv + summary.md + summary.json. "
            "Defaults to printing summary.md to stdout only."
        ),
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=DEFAULT_THRESHOLD_PCT,
        help=(
            "Required-column non-null gate (default: 100.0). Set lower to "
            "wave through sparse columns during incremental rebuilds."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    df_population, summary = audit_column_populations(
        events_parquet=args.events_parquet,
        fold_manifest=args.fold_manifest,
        threshold_pct=args.threshold_pct,
    )
    summary_md = _render_summary_md(summary)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        df_population.to_csv(
            args.output_dir / "column_population.csv", index=False
        )
        (args.output_dir / "summary.md").write_text(
            summary_md, encoding="utf-8"
        )
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        print(f"[pre-sweep-audit] artefacts under {args.output_dir}")
    else:
        sys.stdout.write(summary_md)
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
