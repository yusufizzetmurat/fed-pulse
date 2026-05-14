"""Smoke tests for the external-corpora ingestion paths."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pytest

from app.data.ingest_sources import (
    _OP_FED_STANCE_MAP,
    _iter_lucca_trebbi_records,
    _iter_op_fed_records,
)


def _write_op_fed_csv(path: Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "",
        "Unnamed: 0",
        "unique_id",
        "speaker",
        "sentence",
        "utterance",
        "-5 sentences",
        "-200+ tokens",
        "1_opinion",
        "2_mp",
        "3_mp_context",
        "4_stance_nli",
        "5_stance_nli_context",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            full = {col: row.get(col, "") for col in columns}
            writer.writerow(full)


def test_op_fed_loader_maps_stance_labels(tmp_path: Path) -> None:
    csv_path = tmp_path / "opfed_v1.csv"
    _write_op_fed_csv(
        csv_path,
        [
            {
                "unique_id": "19811222_189_9",
                "speaker": "MR. TRUMAN.",
                "sentence": '"Japan in particular seems to be a serious problem."',
                "1_opinion": "yes",
                "2_mp": "no",
            },
            {
                "unique_id": "19911001_128_5",
                "speaker": "MR. KELLEY.",
                "sentence": '"the question is very much outstanding."',
                "1_opinion": "yes",
                "2_mp": "yes",
                "3_mp_context": "utterance",
                "4_stance_nli": "contradiction",
                "5_stance_nli_context": "utterance",
            },
            {
                "unique_id": "20070131_55_3",
                "speaker": "CHAIR BERNANKE.",
                "sentence": "We need to tighten policy to contain inflation expectations.",
                "1_opinion": "yes",
                "2_mp": "yes",
                "4_stance_nli": "entailment",
            },
            {
                "unique_id": "20080130_60_7",
                "speaker": "MR. KOHN.",
                "sentence": "The outlook is balanced for the time being.",
                "1_opinion": "no",
                "2_mp": "yes",
                "4_stance_nli": "neutral",
            },
            {
                "unique_id": "19850315_70_1",
                "speaker": "MR. LINDSEY.",
                "sentence": "I am uncertain.",
                "1_opinion": "ambiguous",
                "2_mp": "ambiguous",
                "4_stance_nli": "ambiguous",
            },
            {
                "unique_id": "",  # rejected by loader: empty id
                "speaker": "X",
                "sentence": "noop",
            },
        ],
    )

    records = _iter_op_fed_records(csv_path)

    assert len(records) == 5  # the empty-id row is dropped
    assert all(r["source"] == "op_fed" for r in records)
    assert all(r["source_type"] == "fomc_meeting_transcript" for r in records)
    assert all(r["provenance"] == "peer_reviewed" for r in records)
    assert all(r["license_scope"] == "mit" for r in records)

    stance_dist = Counter(r["label"] for r in records)
    assert stance_dist["dovish"] == 1  # contradiction -> dovish
    assert stance_dist["hawkish"] == 1  # entailment -> hawkish
    assert stance_dist["neutral"] == 1
    assert stance_dist[""] == 2  # ambiguous + no-stance rows stay unlabelled

    # Dates parsed from the YYYYMMDD prefix of unique_id
    assert {r["event_date"] for r in records} == {
        "1981-12-22",
        "1991-10-01",
        "2007-01-31",
        "2008-01-30",
        "1985-03-15",
    }

    # Multi-axis extras carry the underlying annotation fields through
    sample = next(r for r in records if r["source_record_id"] == "19911001_128_5")
    assert sample["multi_axis_extras"] == {
        "op_fed_opinion": "yes",
        "op_fed_mp": "yes",
        "op_fed_mp_context": "utterance",
        "op_fed_stance_nli": "contradiction",
        "op_fed_stance_nli_context": "utterance",
    }


def test_op_fed_stance_map_covers_three_keys() -> None:
    assert _OP_FED_STANCE_MAP == {
        "entailment": "hawkish",
        "contradiction": "dovish",
        "neutral": "neutral",
    }


def test_op_fed_loader_returns_empty_on_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "absent.csv"
    with pytest.warns(UserWarning, match="Op-Fed CSV not found"):
        records = _iter_op_fed_records(missing)
    assert records == []


def test_lucca_trebbi_loader_thresholds_categorical_label(tmp_path: Path) -> None:
    csv_path = tmp_path / "lt_index.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["meeting_date", "hawkish_dovish_index"])
        writer.writeheader()
        writer.writerow({"meeting_date": "2004-12-14", "hawkish_dovish_index": "0.8"})
        writer.writerow({"meeting_date": "2008-12-16", "hawkish_dovish_index": "-0.7"})
        writer.writerow({"meeting_date": "2018-06-13", "hawkish_dovish_index": "0.1"})
        writer.writerow({"meeting_date": "2020-04-29", "hawkish_dovish_index": "not-a-number"})  # dropped

    records = _iter_lucca_trebbi_records(csv_path)
    assert len(records) == 3
    by_date = {r["event_date"]: r for r in records}

    assert by_date["2004-12-14"]["label"] == "hawkish"
    assert by_date["2008-12-16"]["label"] == "dovish"
    assert by_date["2018-06-13"]["label"] == ""  # below the threshold

    assert all(r["source"] == "lucca_trebbi_index" for r in records)
    assert all(r["provenance"] == "peer_reviewed" for r in records)
    assert by_date["2018-06-13"]["multi_axis_extras"] == {"lucca_trebbi_index": 0.1}


def test_lucca_trebbi_loader_warns_on_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "absent.csv"
    with pytest.warns(UserWarning, match="Lucca-Trebbi CSV not found"):
        records = _iter_lucca_trebbi_records(missing)
    assert records == []
