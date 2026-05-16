"""Coverage of the loader IO + parsing helpers.

These hit the per-file ingestion branches (JSON, JSONL, CSV) and the
extract_record_groups dispatch the rest of the loader pipeline relies
on. They also cover the read_chunk_embedding_lookup absence paths
(missing parquet, missing key column) since those are the lookup's
clean-failure contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.training.loaders import (
    _extract_record_groups,
    _is_record_mapping_list,
    _load_csv_records,
    _load_json_records,
    _load_jsonl_records,
    _read_chunk_embedding_lookup,
)


def test_load_json_records_returns_top_level_list(tmp_path: Path) -> None:
    p = tmp_path / "in.json"
    p.write_text(json.dumps([{"a": 1}, {"b": 2}, "skipped"]), encoding="utf-8")
    out = _load_json_records(p)
    assert out == [{"a": 1}, {"b": 2}]


def test_load_json_records_extracts_records_key(tmp_path: Path) -> None:
    p = tmp_path / "in.json"
    p.write_text(json.dumps({"records": [{"a": 1}, {"b": 2}]}), encoding="utf-8")
    out = _load_json_records(p)
    assert out == [{"a": 1}, {"b": 2}]


def test_load_json_records_empty_when_no_recognised_shape(tmp_path: Path) -> None:
    p = tmp_path / "in.json"
    p.write_text(json.dumps({"unrecognised": [{"a": 1}]}), encoding="utf-8")
    assert _load_json_records(p) == []


def test_load_jsonl_records_skips_blank_and_non_object(tmp_path: Path) -> None:
    p = tmp_path / "in.jsonl"
    p.write_text(
        '{"a": 1}\n\n[1, 2]\n{"b": 2}\n',
        encoding="utf-8",
    )
    out = _load_jsonl_records(p)
    assert out == [{"a": 1}, {"b": 2}]


def test_load_csv_records_returns_dict_rows(tmp_path: Path) -> None:
    p = tmp_path / "in.csv"
    p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    out = _load_csv_records(p)
    assert out == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


def test_is_record_mapping_list_accepts_homogeneous_list_of_dicts() -> None:
    assert _is_record_mapping_list([{"a": 1}, {"b": 2}]) is True


def test_is_record_mapping_list_rejects_heterogeneous_input() -> None:
    assert _is_record_mapping_list([{"a": 1}, 5]) is False
    assert _is_record_mapping_list({"a": 1}) is False
    assert _is_record_mapping_list([]) is True  # empty list is trivially homogeneous


def test_extract_record_groups_flat_list() -> None:
    payload = [{"a": 1}, {"a": 2}]
    groups = _extract_record_groups(payload)
    assert groups == [payload]


def test_extract_record_groups_nested_records_key() -> None:
    payload = [{"records": [{"a": 1}]}, {"records": [{"a": 2}]}]
    groups = _extract_record_groups(payload)
    # Each ``records`` value becomes its own group.
    assert len(groups) == 2
    assert groups[0] == [{"a": 1}]
    assert groups[1] == [{"a": 2}]


def test_read_chunk_embedding_lookup_returns_empty_when_revision_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An encoder absent from the registry returns ``({}, {})``."""

    import app.models.registry as registry_module

    monkeypatch.setattr(registry_module, "revision_for", lambda alias: None)
    lookup, dates = _read_chunk_embedding_lookup(
        "not-a-registered-encoder", cache_dir=tmp_path
    )
    assert lookup == {}
    assert dates == {}


def test_read_chunk_embedding_lookup_returns_empty_when_parquet_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Missing cache parquet returns ``({}, {})`` cleanly."""

    import app.models.registry as registry_module

    monkeypatch.setattr(registry_module, "revision_for", lambda alias: "v1")
    lookup, dates = _read_chunk_embedding_lookup("finbert", cache_dir=tmp_path)
    assert lookup == {}
    assert dates == {}
