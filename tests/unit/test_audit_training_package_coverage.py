"""Tests for the coverage audit script.

The required column families exercised here are the actual events.parquet
column names from ``backend/app/data/schemas.py``. The audit also walks
sidecar parquets (press-conf Q&A, SEP projections); those are reported
but do NOT fail the audit.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.audit_training_package_coverage import (
    REQUIRED_EVENT_FAMILIES,
    _check_column,
    audit,
)


def _all_required_event_columns() -> list[str]:
    columns: list[str] = []
    for cols in REQUIRED_EVENT_FAMILIES.values():
        columns.extend(cols)
    return columns


def _make_full_event_dataframe(n: int = 10) -> pd.DataFrame:
    """Build a synthetic events.parquet frame with every required column
    populated. The string-typed columns get text; numeric columns get
    floats.
    """
    string_cols = {
        "statement_delta_inserted",
        "statement_delta_deleted",
        "statement_delta_substituted_pairs",
        "dissent_direction",
    }
    list_cols = {"statement_delta_embedding"}
    data: dict[str, list[object]] = {}
    for column in _all_required_event_columns():
        if column in string_cols:
            data[column] = ["text"] * n
        elif column in list_cols:
            data[column] = [[0.0]] * n
        else:
            data[column] = [0.05] * n
    data["event_kind"] = ["statement"] * n
    return pd.DataFrame(data)


def test_check_column_missing() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})
    status, ok, populated, total = _check_column(df, "y", 50.0)
    assert status == "missing"
    assert ok is False
    assert populated == 0
    assert total == 3


def test_check_column_empty() -> None:
    df = pd.DataFrame({"x": [None, None, None]})
    status, ok, populated, total = _check_column(df, "x", 50.0)
    assert status == "empty"
    assert ok is False
    assert populated == 0


def test_check_column_sparse() -> None:
    df = pd.DataFrame({"x": [1.0, None, None, None]})
    status, ok, populated, total = _check_column(df, "x", 50.0)
    assert status == "sparse"
    assert ok is False
    assert populated == 1
    assert total == 4


def test_check_column_ok() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    status, ok, populated, total = _check_column(df, "x", 50.0)
    assert status == "ok"
    assert ok is True
    assert populated == 4


def test_audit_passes_on_full_event_dataframe(tmp_path: Path) -> None:
    df = _make_full_event_dataframe(n=10)
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    result = audit(parquet)
    assert result == 0


def test_audit_fails_when_required_rates_column_missing(tmp_path: Path) -> None:
    df = _make_full_event_dataframe(n=10).drop(columns=["yield_2y_change_5d"])
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    result = audit(parquet)
    assert result == 1


def test_audit_fails_when_required_column_sparse(tmp_path: Path) -> None:
    df = _make_full_event_dataframe(n=10)
    df["statement_delta_inserted"] = ["text"] + [None] * 9
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    result = audit(parquet)
    assert result == 1


def test_audit_fails_when_canonical_target_missing(tmp_path: Path) -> None:
    df = _make_full_event_dataframe(n=10).drop(
        columns=["forward_realized_vol_10d"]
    )
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    result = audit(parquet)
    assert result == 1


def test_audit_returns_two_when_file_missing(tmp_path: Path) -> None:
    target = tmp_path / "does-not-exist.parquet"
    result = audit(target)
    assert result == 2


def test_audit_passes_when_event_kind_includes_unknown(tmp_path: Path) -> None:
    df = _make_full_event_dataframe(n=10)
    df.loc[0, "event_kind"] = "unrecognised_kind"
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    # Unknown event_kind values are reported but do not fail the audit.
    result = audit(parquet)
    assert result == 0


def test_audit_passes_when_event_kind_column_missing(tmp_path: Path) -> None:
    df = _make_full_event_dataframe(n=10).drop(columns=["event_kind"])
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    # Missing event_kind column is reported but does NOT fail the audit
    # (corpus-diversity gaps are tracked separately under #485).
    result = audit(parquet)
    assert result == 0


def test_sparse_threshold_override_relaxes_audit(tmp_path: Path) -> None:
    df = _make_full_event_dataframe(n=10)
    df["statement_delta_inserted"] = ["text"] + [None] * 9
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    result = audit(parquet, sparse_threshold=5.0)
    assert result == 0


def test_audit_reports_sidecar_absence_without_failing(tmp_path: Path) -> None:
    """No sidecar parquets present alongside events.parquet — audit
    still passes since sidecar gaps degrade trainer flags that default
    off. The report visibly notes the absence; the exit code is 0.
    """
    df = _make_full_event_dataframe(n=10)
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    result = audit(parquet)
    assert result == 0
