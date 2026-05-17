from __future__ import annotations

import pytest

from app.data import event_dataset_builder as edb
from app.data.event_dataset_builder import (
    _RegistryRow,
    _derive_stance_by_date,
    _resolve_embedding_path,
)


def _row(event_date: str, mapped_label: str | None) -> _RegistryRow:
    return _RegistryRow(
        source="hf_fomc_communication",
        source_record_id=f"rec-{event_date}-{mapped_label}",
        document_type="fomc_statement",
        event_date=event_date,
        text="dummy",
        mapped_label=mapped_label,
        multi_axis_extras={},
        axes={"stance": None, "time": None, "certainty": None, "factor": None, "topic": None},
    )


def test_derive_stance_by_date_returns_sorted_means() -> None:
    rows = [
        _row("2024-09-18", "hawkish"),
        _row("2024-09-18", "neutral"),
        _row("2024-07-31", "dovish"),
        _row("2024-11-07", "hawkish"),
    ]
    out = _derive_stance_by_date(rows)
    assert out == (
        ("2024-07-31", -1.0),
        ("2024-09-18", 0.5),    # mean of hawkish (1.0) + neutral (0.0)
        ("2024-11-07", 1.0),
    )


def test_derive_stance_by_date_drops_unmapped_rows() -> None:
    rows = [
        _row("2024-09-18", None),                # unmapped → skipped
        _row("2024-09-18", "not-a-stance"),      # unknown label → skipped
        _row("2024-09-18", "hawkish"),
    ]
    out = _derive_stance_by_date(rows)
    assert out == (("2024-09-18", 1.0),)


def test_derive_stance_by_date_empty_on_zero_mapped_rows() -> None:
    rows = [_row("2024-09-18", None), _row("2024-11-07", None)]
    assert _derive_stance_by_date(rows) == ()


def test_resolve_embedding_path_none_sentinel(tmp_path) -> None:
    assert _resolve_embedding_path(None) is None
    assert _resolve_embedding_path("") is None
    assert _resolve_embedding_path("none") is None
    assert _resolve_embedding_path("NONE") is None
    assert _resolve_embedding_path("  None  ") is None


def test_resolve_embedding_path_existing_file(tmp_path) -> None:
    p = tmp_path / "fake.parquet"
    p.write_bytes(b"")
    assert _resolve_embedding_path(str(p)) == p


def test_resolve_embedding_path_missing_file_is_hard_error(tmp_path) -> None:
    missing = tmp_path / "no-such-file.parquet"
    with pytest.raises(SystemExit) as exc_info:
        _resolve_embedding_path(str(missing))
    assert "missing" in str(exc_info.value).lower()


def test_safe_credibility_logs_warning_when_inputs_empty(caplog) -> None:
    edb._CREDIBILITY_EMPTY_INPUTS_WARNED = False
    with caplog.at_level("WARNING", logger="app.data.event_dataset_builder"):
        edb._safe_credibility(
            "2024-09-18T14:00:00+00:00",
            {"embedding_path": None, "stance_by_date": (), "fred_cache_dir": None},
        )
    assert any(
        "credibility inputs empty" in rec.message for rec in caplog.records
    )
