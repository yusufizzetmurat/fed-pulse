"""Tests for the pre-sweep per-column population audit (#505 A.1.a)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_pre_sweep_column_audit import (
    _render_summary_md,
    audit_column_populations,
)


def _make_minimal_events(n: int = 10) -> pd.DataFrame:
    """Build a synthetic events.parquet frame that satisfies every
    schema-required column. The pinned schema in
    ``backend/app/data/schemas.py::_EVENT_ROW_COLUMNS`` is the
    authoritative list; this helper covers the canonical-arm required
    set plus the indexing columns the audit slices on.
    """

    from app.data.schemas import _EVENT_ROW_COLUMNS

    required_cols = {
        name
        for name, col in _EVENT_ROW_COLUMNS.items()
        if col.required
    }

    string_defaults = {
        "event_kind": "statement",
        "document_id": "doc_0",
        "text_hash": "hash_0",
        "source": "scraped_fed",
        "source_record_id": "rec_0",
        "as_of_ts": "2024-06-12T18:00:00Z",
        "document_type": "statement",
        "label_origin": "model",
        "license_scope": "public",
        "citation_ref": "test_citation",
        "text": "test",
        "asset_symbol": "^GSPC",
        "prior_window_sha256": "0" * 64,
        "prior_bars_json": "[]",
        "event_date": "2024-06-12",
        "realized_date": "2024-06-26",
        "concurrent_macro_release": False,
    }

    data: dict[str, list] = {}
    for col in required_cols:
        if col in string_defaults:
            data[col] = [string_defaults[col]] * n
        else:
            # Default numeric required columns to 0.0; schemas with
            # int/bool types coerce on read.
            data[col] = [0.0] * n
    return pd.DataFrame(data)


def test_audit_passes_when_every_required_column_is_present_and_populated(
    tmp_path: Path,
) -> None:
    df = _make_minimal_events(n=10)
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    population, summary = audit_column_populations(parquet)

    assert summary["pass"] is True
    assert summary["required_columns_under_threshold"] == []
    assert summary["required_columns_missing_from_parquet"] == []
    assert summary["n_rows"] == 10
    # Overall slice present for every column on the parquet.
    overall = population[population["slice_kind"] == "overall"]
    assert len(overall) == len(df.columns)


def test_audit_fails_when_required_column_absent(tmp_path: Path) -> None:
    """Drop a column that the schema marks ``required=True`` and confirm
    the audit reports it as missing.
    """

    df = _make_minimal_events(n=10).drop(columns=["event_date"])
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    _, summary = audit_column_populations(parquet)
    assert summary["pass"] is False
    assert "event_date" in summary["required_columns_missing_from_parquet"]


def test_audit_fails_when_required_column_partially_null(tmp_path: Path) -> None:
    df = _make_minimal_events(n=10)
    df.loc[5:, "event_date"] = None  # half-empty required column
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    _, summary = audit_column_populations(parquet)
    flagged = {
        entry["column"]
        for entry in summary["required_columns_under_threshold"]
    }
    assert "event_date" in flagged
    assert summary["pass"] is False


def test_audit_treats_partial_nullable_required_column_as_advisory(
    tmp_path: Path,
) -> None:
    """``axis_stance`` is declared ``required=True, nullable=True``: the
    column must exist on the parquet, but individual rows may legitimately
    carry ``None`` (only HF-style multi-axis sources ship it). The audit
    must surface the population gap as advisory rather than failing the
    strict gate.
    """

    df = _make_minimal_events(n=10)
    df["axis_stance"] = [None] * 10  # 0% non-null, allowed by schema
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    _, summary = audit_column_populations(parquet)
    flagged = {
        entry["column"] for entry in summary["required_columns_under_threshold"]
    }
    advisory = {
        entry["column"]
        for entry in summary["nullable_required_columns_under_threshold"]
    }
    assert "axis_stance" not in flagged
    assert "axis_stance" in advisory
    assert summary["pass"] is True


def test_audit_threshold_override_relaxes_required_gate(
    tmp_path: Path,
) -> None:
    df = _make_minimal_events(n=10)
    df.loc[5:, "event_date"] = None  # 50% non-null
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    # At threshold 40% the required-column gate passes.
    _, summary = audit_column_populations(parquet, threshold_pct=40.0)
    assert summary["pass"] is True


def test_audit_groups_by_event_kind_and_source(tmp_path: Path) -> None:
    df = _make_minimal_events(n=6)
    df["event_kind"] = ["statement"] * 3 + ["minutes"] * 3
    df["source"] = ["scraped_fed"] * 2 + ["kaggle"] * 2 + ["op_fed"] * 2
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    population, _ = audit_column_populations(parquet)

    kinds = set(
        population[population["slice_kind"] == "event_kind"]["slice_value"]
    )
    sources = set(
        population[population["slice_kind"] == "source"]["slice_value"]
    )
    assert {"statement", "minutes"} <= kinds
    assert {"scraped_fed", "kaggle", "op_fed"} <= sources


def test_audit_uses_fold_manifest_when_provided(tmp_path: Path) -> None:
    df = _make_minimal_events(n=6)
    df["event_date"] = [
        "2024-01-01",
        "2024-02-01",
        "2024-03-01",
        "2024-04-01",
        "2024-05-01",
        "2024-06-01",
    ]
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)
    manifest = {
        "folds": [
            {
                "fold_id": "wf_fold_1",
                "test": ["2024-01-01", "2024-02-01", "2024-03-01"],
            },
            {
                "fold_id": "wf_fold_2",
                "test": ["2024-04-01", "2024-05-01", "2024-06-01"],
            },
        ]
    }
    manifest_path = tmp_path / "fold_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    population, summary = audit_column_populations(
        parquet, fold_manifest=manifest_path
    )
    assert summary["fold_manifest_used"] is True
    fold_slices = set(
        population[population["slice_kind"] == "fold"]["slice_value"]
    )
    assert {"wf_fold_1", "wf_fold_2"} <= fold_slices


def test_audit_expands_fold_manifest_with_test_start_end_ranges(
    tmp_path: Path,
) -> None:
    """The production manifest writer emits per-fold ``test_start`` /
    ``test_end`` ISO-date ranges, not enumerated date lists. The reader
    must expand the range against ``event_date`` so the per-fold breakdown
    actually fires.
    """

    df = _make_minimal_events(n=6)
    df["event_date"] = [
        "2024-01-01",
        "2024-02-01",
        "2024-03-01",
        "2024-04-01",
        "2024-05-01",
        "2024-06-01",
    ]
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)
    manifest = {
        "folds": [
            {
                "fold_id": "wf_fold_1",
                "test_start": "2024-01-01",
                "test_end": "2024-03-01",
            },
            {
                "fold_id": "wf_fold_2",
                "test_start": "2024-04-01",
                "test_end": "2024-06-01",
            },
        ]
    }
    manifest_path = tmp_path / "fold_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    population, summary = audit_column_populations(
        parquet, fold_manifest=manifest_path
    )
    assert summary["fold_manifest_used"] is True
    fold_slices = population[population["slice_kind"] == "fold"]
    by_fold = dict(zip(fold_slices["slice_value"], fold_slices["rows"]))
    # Each fold's test window covers exactly three of the six synthetic events.
    assert by_fold.get("wf_fold_1") == 3
    assert by_fold.get("wf_fold_2") == 3


def test_audit_skips_fold_breakdown_on_corrupt_manifest(tmp_path: Path) -> None:
    df = _make_minimal_events(n=2)
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{not json", encoding="utf-8")

    _, summary = audit_column_populations(
        parquet, fold_manifest=manifest_path
    )
    assert summary["fold_manifest_used"] is False


def test_audit_raises_on_missing_parquet(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="not found"):
        audit_column_populations(tmp_path / "missing.parquet")


def test_summary_md_lists_flagged_columns() -> None:
    summary = {
        "events_parquet": "x.parquet",
        "n_rows": 100,
        "n_columns": 10,
        "schema_columns_present": [],
        "schema_columns_absent": [],
        "required_columns_missing_from_parquet": ["foo"],
        "required_columns_under_threshold": [
            {
                "column": "bar",
                "non_null_rate": 0.5,
                "n_non_null": 50,
                "n_rows": 100,
                "threshold_rate": 1.0,
            }
        ],
        "nullable_required_columns_under_threshold": [],
        "threshold_pct": 100.0,
        "fold_manifest_used": False,
        "pass": False,
    }
    md = _render_summary_md(summary)
    assert "FAIL" in md
    assert "`foo`" in md
    assert "`bar`" in md
    assert "50.00%" in md


def test_summary_md_pass_message() -> None:
    summary = {
        "events_parquet": "x.parquet",
        "n_rows": 100,
        "n_columns": 10,
        "schema_columns_present": [],
        "schema_columns_absent": [],
        "required_columns_missing_from_parquet": [],
        "required_columns_under_threshold": [],
        "nullable_required_columns_under_threshold": [],
        "threshold_pct": 100.0,
        "fold_manifest_used": True,
        "pass": True,
    }
    md = _render_summary_md(summary)
    assert "PASS" in md
    assert "non-nullable" in md


def test_summary_md_renders_advisory_section() -> None:
    summary = {
        "events_parquet": "x.parquet",
        "n_rows": 100,
        "n_columns": 10,
        "schema_columns_present": [],
        "schema_columns_absent": [],
        "required_columns_missing_from_parquet": [],
        "required_columns_under_threshold": [],
        "nullable_required_columns_under_threshold": [
            {
                "column": "axis_stance",
                "non_null_rate": 0.0,
                "n_non_null": 0,
                "n_rows": 100,
                "threshold_rate": 1.0,
            }
        ],
        "threshold_pct": 100.0,
        "fold_manifest_used": True,
        "pass": True,
    }
    md = _render_summary_md(summary)
    assert "PASS" in md
    assert "Advisory" in md
    assert "`axis_stance`" in md
