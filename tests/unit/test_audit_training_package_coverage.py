"""Smoke tests for the coverage audit script."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_training_package_coverage import (
    REQUIRED_FAMILIES,
    _check_column,
    audit,
)


def _all_required_columns() -> list[str]:
    columns: list[str] = []
    for cols in REQUIRED_FAMILIES.values():
        columns.extend(cols)
    return columns


def _make_full_dataframe(n: int = 10) -> pd.DataFrame:
    data: dict[str, list[float | str | int]] = {}
    for column in _all_required_columns():
        if column in {
            "statement_delta_inserted",
            "statement_delta_deleted",
            "statement_delta_substituted_pairs",
            "qa_text",
            "dissent_direction",
        }:
            data[column] = ["text"] * n
        elif column == "statement_delta_embedding" or column == "qa_embedding":
            data[column] = [[0.0]] * n  # type: ignore[list-item]
        elif column == "has_press_conf":
            data[column] = [1.0] * n
        elif column == "dissent_count":
            data[column] = [0.0] * n
        elif column == "votes_for" or column == "votes_against":
            data[column] = [10.0] * n
        else:
            data[column] = [0.05] * n
    data["source_type"] = ["fomc_statement"] * n
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


def test_audit_passes_on_full_dataframe(tmp_path: Path) -> None:
    df = _make_full_dataframe(n=10)
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    result = audit(parquet)
    assert result == 0


def test_audit_fails_when_required_column_missing(tmp_path: Path) -> None:
    df = _make_full_dataframe(n=10)
    df = df.drop(columns=["forward_yield_2y_change_5d"])
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    result = audit(parquet)
    assert result == 1


def test_audit_fails_when_required_column_sparse(tmp_path: Path) -> None:
    df = _make_full_dataframe(n=10)
    df["statement_delta_inserted"] = ["text"] + [None] * 9  # type: ignore[list-item]
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    result = audit(parquet)
    assert result == 1


def test_audit_returns_two_when_file_missing(tmp_path: Path) -> None:
    target = tmp_path / "does-not-exist.parquet"
    result = audit(target)
    assert result == 2


def test_audit_passes_when_source_type_includes_unexpected_kinds(
    tmp_path: Path,
) -> None:
    df = _make_full_dataframe(n=10)
    df.loc[0, "source_type"] = "novel_kind"
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    result = audit(parquet)
    assert result == 0


def test_audit_fails_when_source_type_column_missing(tmp_path: Path) -> None:
    df = _make_full_dataframe(n=10).drop(columns=["source_type"])
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    result = audit(parquet)
    assert result == 1


def test_sparse_threshold_override_relaxes_audit(tmp_path: Path) -> None:
    df = _make_full_dataframe(n=10)
    df["statement_delta_inserted"] = ["text"] + [None] * 9  # type: ignore[list-item]
    parquet = tmp_path / "events.parquet"
    df.to_parquet(parquet)

    pytest.importorskip("pandas")
    result = audit(parquet, sparse_threshold=5.0)
    assert result == 0
