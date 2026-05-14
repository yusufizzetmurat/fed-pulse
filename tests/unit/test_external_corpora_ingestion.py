"""Smoke tests for the external-corpora ingestion paths."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pytest

from app.data.ingest_sources import (
    _OP_FED_STANCE_MAP,
    _iter_gss_factors_records,
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


def _write_gss_factors_csv(path: Path, rows: list[dict[str, str]]) -> None:
    columns = ["meeting_date", "target_factor", "path_factor", "fomc_statement"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def _write_gss_surprises_csv(path: Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "meeting_date",
        "surprise_30min_bp",
        "surprise_1hour_bp",
        "surprise_1day_bp",
        "diff_wide_minus_tight",
        "diff_daily_minus_tight",
        "flags_raw",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def test_gss_factors_loader_emits_one_row_per_meeting_with_factor_extras(tmp_path: Path) -> None:
    factors_path = tmp_path / "gss_factors.csv"
    surprises_path = tmp_path / "gss_surprises.csv"
    _write_gss_factors_csv(
        factors_path,
        [
            {"meeting_date": "1994-08-16", "target_factor": "10.7", "path_factor": "-8.3", "fomc_statement": "T"},
            {"meeting_date": "2001-01-03", "target_factor": "-32.3", "path_factor": "22.8", "fomc_statement": "T"},
            {"meeting_date": "1990-02-08", "target_factor": "0.3", "path_factor": "5.8", "fomc_statement": ""},
        ],
    )
    _write_gss_surprises_csv(
        surprises_path,
        [
            {
                "meeting_date": "2001-01-03",
                "surprise_30min_bp": "-39.3",
                "surprise_1hour_bp": "-36.5",
                "surprise_1day_bp": "-38.2",
                "diff_wide_minus_tight": "1.1",
                "diff_daily_minus_tight": "2.8",
                "flags_raw": "T",
            }
        ],
    )

    records = _iter_gss_factors_records(factors_path, surprises_path)
    assert len(records) == 3
    by_date = {r["event_date"]: r for r in records}

    assert all(r["source"] == "gss_factor" for r in records)
    assert all(r["source_type"] == "fomc_statement" for r in records)
    assert all(r["provenance"] == "peer_reviewed" for r in records)
    assert all(r["license_scope"] == "research_only" for r in records)
    assert all(r["citation_ref"] == "gurkaynak_sack_swanson_2005_ijcb" for r in records)
    assert all(r["label"] == "" for r in records)  # factor axis is continuous

    factors_row = by_date["2001-01-03"]
    extras = factors_row["multi_axis_extras"]
    assert extras["gss_target_factor"] == pytest.approx(-32.3)
    assert extras["gss_path_factor"] == pytest.approx(22.8)
    assert extras["gss_fomc_statement"] is True
    # The surprise row merges onto the same date
    assert extras["surprise_30min_bp"] == pytest.approx(-39.3)
    assert extras["diff_daily_minus_tight"] == pytest.approx(2.8)

    no_surprise_row = by_date["1990-02-08"]
    assert no_surprise_row["multi_axis_extras"]["gss_fomc_statement"] is False
    assert "surprise_30min_bp" not in no_surprise_row["multi_axis_extras"]
    assert "GSS factor decomposition for 1990-02-08" in no_surprise_row["text"]


def test_gss_factors_loader_warns_on_missing_factors_file(tmp_path: Path) -> None:
    missing = tmp_path / "absent.csv"
    with pytest.warns(UserWarning, match="GSS factors CSV not found"):
        records = _iter_gss_factors_records(missing, surprises_csv=None)
    assert records == []


def test_gss_factors_loader_handles_missing_surprises_file(tmp_path: Path) -> None:
    factors_path = tmp_path / "gss_factors.csv"
    _write_gss_factors_csv(
        factors_path,
        [{"meeting_date": "1990-02-08", "target_factor": "0.3", "path_factor": "5.8", "fomc_statement": ""}],
    )
    records = _iter_gss_factors_records(factors_path, surprises_csv=tmp_path / "absent.csv")
    assert len(records) == 1
    extras = records[0]["multi_axis_extras"]
    assert extras["gss_target_factor"] == pytest.approx(0.3)
    assert extras["gss_path_factor"] == pytest.approx(5.8)
    assert "surprise_30min_bp" not in extras  # gracefully degrades when surprises CSV is absent


def test_extract_gss_factors_parses_appendix_text() -> None:
    from scripts.extract_gss_factors import extract_factors, extract_surprises

    factor_text = (
        "Date Factor Factor Statement? Date Factor Factor Statement? Date Factor Factor Statement?\n"
        "8-Feb-90 0.3 5.8 16-Aug-94 10.7 -8.3 T 15-Nov-00 2.7 2.2 T\n"
        "28-Mar-90 1.5 -3.3 27-Sep-94 -4.3 7.9 19-Dec-00 7.5 -3.9 T\n"
    )
    factor_rows = extract_factors(factor_text)
    assert {r["meeting_date"] for r in factor_rows} == {
        "1990-02-08",
        "1990-03-28",
        "1994-08-16",
        "1994-09-27",
        "2000-11-15",
        "2000-12-19",
    }
    aug_94 = next(r for r in factor_rows if r["meeting_date"] == "1994-08-16")
    assert aug_94["target_factor"] == 10.7
    assert aug_94["path_factor"] == -8.3
    assert aug_94["fomc_statement"] == "T"

    surprise_text = (
        "4-Feb-94 16.3 15.2 11.7 -4.7 -1.2 T\n"
        "17-Sep-01 omitted omitted omitted omitted omitted T\n"
    )
    surprise_rows = extract_surprises(surprise_text)
    assert len(surprise_rows) == 2
    feb_94 = next(r for r in surprise_rows if r["meeting_date"] == "1994-02-04")
    assert feb_94["surprise_30min_bp"] == 16.3
    sep_01 = next(r for r in surprise_rows if r["meeting_date"] == "2001-09-17")
    assert sep_01["surprise_30min_bp"] is None
    assert sep_01["surprise_1day_bp"] is None
