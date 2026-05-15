"""Smoke tests for the external-corpora ingestion paths."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pytest

from typing import Any

from app.data.ingest_sources import (
    GTFINTECHLAB_CROSS_BANK_DATASETS,
    GTFINTECHLAB_FED_DATASET_ID,
    VTASCA_FOMC_ARCHIVE_DATASET_ID,
    _DATASET_REVISIONS,
    _GTFINTECHLAB_STANCE_MAP,
    _OP_FED_STANCE_MAP,
    _dataset_revision,
    _iter_fomc_archive_records,
    _iter_gss_factors_records,
    _iter_gtfintechlab_cross_bank_records,
    _iter_gtfintechlab_federal_reserve_records,
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
    """Locate the parser at repo root and exercise its two regexes against synthetic
    appendix text. The script lives outside the backend package so pytest CI (which
    runs with ``PYTHONPATH=backend``) does not see it on the import path by default."""

    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "extract_gss_factors.py"
    if not script.exists():
        pytest.skip(f"extraction script not present at {script}")
    spec = importlib.util.spec_from_file_location("extract_gss_factors", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    factor_text = (
        "Date Factor Factor Statement? Date Factor Factor Statement? Date Factor Factor Statement?\n"
        "8-Feb-90 0.3 5.8 16-Aug-94 10.7 -8.3 T 15-Nov-00 2.7 2.2 T\n"
        "28-Mar-90 1.5 -3.3 27-Sep-94 -4.3 7.9 19-Dec-00 7.5 -3.9 T\n"
    )
    factor_rows = module.extract_factors(factor_text)
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
    surprise_rows = module.extract_surprises(surprise_text)
    assert len(surprise_rows) == 2
    feb_94 = next(r for r in surprise_rows if r["meeting_date"] == "1994-02-04")
    assert feb_94["surprise_30min_bp"] == 16.3
    sep_01 = next(r for r in surprise_rows if r["meeting_date"] == "2001-09-17")
    assert sep_01["surprise_30min_bp"] is None
    assert sep_01["surprise_1day_bp"] is None


def _install_fake_datasets(monkeypatch, payload: dict[tuple[str, str], list[dict]]) -> None:
    """Inject a fake `datasets` module wired for the multi-config gtfintechlab loader."""
    import sys
    import types

    fake = types.SimpleNamespace()
    captured: dict[str, Any] = {"revisions": []}
    fake.get_dataset_config_names = lambda dataset, revision=None: sorted({k[0] for k in payload})
    fake.get_dataset_split_names = lambda dataset, config, revision=None: sorted(
        k[1] for k in payload if k[0] == config
    )

    def _load(dataset, config, split=None, revision=None):
        captured["revisions"].append(revision)
        return payload[(config, split)]

    fake.load_dataset = _load
    fake._captured = captured
    monkeypatch.setitem(sys.modules, "datasets", fake)
    return captured


def _install_fake_datasets_module(monkeypatch, rows: list[dict]) -> None:
    """Inject a fake `datasets` module wired for single-split iter-of-rows loaders (vtasca)."""
    import sys
    import types

    fake = types.SimpleNamespace()
    captured: dict[str, Any] = {"revisions": []}

    def _load(dataset_id, **kw):
        captured["revisions"].append(kw.get("revision"))
        return iter(rows)

    fake.load_dataset = _load
    fake._captured = captured
    monkeypatch.setitem(sys.modules, "datasets", fake)
    return captured


def test_gtfintechlab_stance_map_covers_canonical_classes() -> None:
    assert set(_GTFINTECHLAB_STANCE_MAP) == {"hawkish", "dovish", "neutral"}
    assert _GTFINTECHLAB_STANCE_MAP["hawkish"] == "hawkish"
    assert _GTFINTECHLAB_STANCE_MAP["dovish"] == "dovish"
    assert _GTFINTECHLAB_STANCE_MAP["neutral"] == "neutral"


def test_iter_gtfintechlab_federal_reserve_records_maps_multi_axis(monkeypatch) -> None:
    payload = {
        ("5768", "train"): [
            {
                "sentences": "Inflation pressures remain elevated.",
                "stance_label": "hawkish",
                "time_label": "forward looking",
                "certain_label": "certain",
                "year": 2022,
            },
            {
                "sentences": "The Committee maintained accommodative policy.",
                "stance_label": "dovish",
                "time_label": "not forward looking",
                "certain_label": "uncertain",
                "year": 2021,
            },
            {
                "sentences": "Activity has expanded at a moderate pace.",
                "stance_label": "neutral",
                "time_label": "not forward looking",
                "certain_label": "certain",
                "year": 2014,
            },
        ],
        ("5768", "test"): [
            {
                "sentences": "",  # empty text — should be dropped
                "stance_label": "hawkish",
                "time_label": "forward looking",
                "certain_label": "certain",
                "year": 2020,
            },
            {
                "sentences": "Inflation pressures remain elevated.",  # duplicate of train row
                "stance_label": "hawkish",
                "time_label": "forward looking",
                "certain_label": "certain",
                "year": 2022,
            },
        ],
        ("78516", "train"): [
            {
                "sentences": "Risks to the outlook are roughly balanced.",
                "stance_label": "NEUTRAL",  # case-insensitive
                "time_label": "forward looking",
                "certain_label": "certain",
                "year": 2016,
            },
            {
                "sentences": "Empty year row should drop.",
                "stance_label": "hawkish",
                "time_label": "forward looking",
                "certain_label": "certain",
                "year": None,
            },
        ],
    }
    _install_fake_datasets(monkeypatch, payload)

    records = _iter_gtfintechlab_federal_reserve_records()

    assert len(records) == 4  # empty-text, empty-year, and duplicate dropped
    assert all(r["source"] == "gtfintechlab_federal_reserve_system" for r in records)
    assert all(r["provenance"] == "peer_reviewed" for r in records)
    assert all(r["license_scope"] == "research_only" for r in records)
    assert {r["label"] for r in records} == {"hawkish", "dovish", "neutral"}

    by_text = {r["text"]: r for r in records}
    elevated = by_text["Inflation pressures remain elevated."]
    assert elevated["event_date"] == "2022-01-01"
    assert elevated["multi_axis_extras"]["gtfintechlab_time_label"] == "forward looking"
    assert elevated["multi_axis_extras"]["gtfintechlab_certain_label"] == "certain"
    assert elevated["multi_axis_extras"]["gtfintechlab_config"] == "5768"
    # First-seen-wins dedup keeps the test-split copy (splits iterate alphabetically).
    assert elevated["multi_axis_extras"]["gtfintechlab_split"] == "test"

    balanced = by_text["Risks to the outlook are roughly balanced."]
    assert balanced["label"] == "neutral"  # case-insensitive stance match
    assert balanced["event_date"] == "2016-01-01"


def test_iter_gtfintechlab_federal_reserve_pins_revision_and_derives_source_record_id(
    monkeypatch,
) -> None:
    """Regression: source_record_id must be content-derived (not positional idx),
    and load_dataset must receive the pinned revision from _DATASET_REVISIONS."""
    payload = {
        ("5768", "train"): [
            {
                "sentences": "First sentence about inflation.",
                "stance_label": "hawkish",
                "time_label": "forward looking",
                "certain_label": "certain",
                "year": 2022,
            },
        ],
        ("5768", "test"): [
            {
                "sentences": "Second sentence about employment.",
                "stance_label": "dovish",
                "time_label": "not forward looking",
                "certain_label": "certain",
                "year": 2021,
            },
        ],
    }
    captured = _install_fake_datasets(monkeypatch, payload)

    records = _iter_gtfintechlab_federal_reserve_records()

    pinned = _DATASET_REVISIONS[GTFINTECHLAB_FED_DATASET_ID]
    assert captured["revisions"] == [pinned, pinned]  # called once per config/split combo
    assert all(r["multi_axis_extras"]["gtfintechlab_dataset_revision"] == pinned for r in records)
    # source_record_id is the 16-char prefix of sha256(normalized_text); not a positional ":<idx>".
    for record in records:
        assert ":" not in record["source_record_id"]
        assert len(record["source_record_id"]) == 16
        assert all(c in "0123456789abcdef" for c in record["source_record_id"])


def test_gtfintechlab_cross_bank_dataset_list_covers_five_banks() -> None:
    bank_keys = [item[0] for item in GTFINTECHLAB_CROSS_BANK_DATASETS]
    assert bank_keys == [
        "european_central_bank",
        "bank_of_japan",
        "bank_of_england",
        "bank_of_canada",
        "reserve_bank_of_australia",
    ]
    for bank_key, hf_id, _document_type in GTFINTECHLAB_CROSS_BANK_DATASETS:
        assert hf_id.startswith("gtfintechlab/")
        assert bank_key in hf_id.split("/", 1)[1]


def test_iter_gtfintechlab_cross_bank_records_tags_provenance(monkeypatch) -> None:
    # Same row schema as federal_reserve_system; the cross-bank loader
    # iterates a fixed list of (bank, dataset_id) tuples so we mock all five
    # by returning the same single-row payload regardless of dataset_id.
    import sys
    import types

    fake = types.SimpleNamespace()
    fake.get_dataset_config_names = lambda dataset, revision=None: ["default"]
    fake.get_dataset_split_names = lambda dataset, config, revision=None: ["train"]

    def _fake_load(dataset_id, config, split=None, revision=None):
        # Encode the dataset_id into the sentence so the dedupe doesn't collapse
        # rows across banks.
        return [
            {
                "sentences": f"{dataset_id} — Inflation pressures are easing.",
                "stance_label": "dovish",
                "time_label": "not forward looking",
                "certain_label": "certain",
                "year": 2024,
            }
        ]

    fake.load_dataset = _fake_load
    monkeypatch.setitem(sys.modules, "datasets", fake)

    records = _iter_gtfintechlab_cross_bank_records()

    assert len(records) == len(GTFINTECHLAB_CROSS_BANK_DATASETS) == 5
    assert all(r["provenance"] == "peer_reviewed_cross_bank" for r in records)
    sources = {r["source"] for r in records}
    assert sources == {
        "gtfintechlab_european_central_bank",
        "gtfintechlab_bank_of_japan",
        "gtfintechlab_bank_of_england",
        "gtfintechlab_bank_of_canada",
        "gtfintechlab_reserve_bank_of_australia",
    }
    assert all(r["label"] == "dovish" for r in records)
    assert all(r["event_date"] == "2024-01-01" for r in records)


def test_iter_fomc_archive_records_routes_statements_and_minutes(monkeypatch) -> None:
    _install_fake_datasets_module(
        monkeypatch,
        [
            {
                "Date": "2024-09-18",
                "Release Date": "2024-09-18",
                "Type": "Statement",
                "Text": "Recent indicators suggest economic activity has been expanding.",
            },
            {
                "Date": "2024-09-18",
                "Release Date": "2024-10-09",
                "Type": "Minutes",
                "Text": "The Committee discussed the staff outlook for inflation.",
            },
            {
                "Date": "2024-11-07",
                "Release Date": "2024-11-07",
                "Type": "Statement",
                "Text": "",  # empty text → dropped
            },
            {
                "Date": "",  # missing Date falls back to Release Date.
                "Release Date": "",
                "Type": "Statement",
                "Text": "No date at all — should drop.",
            },
            {
                "Date": "2024-12-18",
                "Release Date": "2024-12-18",
                "Type": "Speech",  # unrecognised type → dropped
                "Text": "Speech text.",
            },
        ],
    )

    records = _iter_fomc_archive_records()

    assert len(records) == 2
    assert all(r["source"] == "vtasca_fomc_archive" for r in records)
    assert all(r["provenance"] == "scraped" for r in records)
    assert all(r["license_scope"] == "public_source_scrape_terms_required" for r in records)
    assert all(r["label"] == "" for r in records)
    assert all(r["label_origin"] == "pseudo" for r in records)

    by_type = {r["document_type"]: r for r in records}
    assert by_type["statement"]["source_type"] == "fomc_statement"
    assert by_type["minutes"]["source_type"] == "fomc_minutes"

    minutes_row = by_type["minutes"]
    # Minutes release on 2024-10-09 differs from event_date 2024-09-18 — flagged in extras.
    assert minutes_row["multi_axis_extras"]["release_date"] == "2024-10-09"
    statement_row = by_type["statement"]
    # Statement release equals event date so release_date is omitted from extras.
    # vtasca_dataset_revision is set unconditionally for reproducibility.
    assert "release_date" not in statement_row.get("multi_axis_extras", {})
    assert statement_row["multi_axis_extras"]["vtasca_dataset_revision"] == _DATASET_REVISIONS[
        VTASCA_FOMC_ARCHIVE_DATASET_ID
    ]


def test_iter_fomc_archive_records_source_record_id_discriminates_distinct_text(
    monkeypatch,
) -> None:
    """Two corrected releases on the same date+document_type with distinct text
    must produce distinct source_record_ids (text_hash discriminator)."""
    _install_fake_datasets_module(
        monkeypatch,
        [
            {
                "Date": "2024-09-18",
                "Release Date": "2024-09-18",
                "Type": "Statement",
                "Text": "Original September statement language.",
            },
            {
                "Date": "2024-09-18",
                "Release Date": "2024-09-19",
                "Type": "Statement",
                "Text": "Corrected September statement language.",
            },
        ],
    )

    records = _iter_fomc_archive_records()

    assert len(records) == 2
    ids = [r["source_record_id"] for r in records]
    assert len(set(ids)) == 2
    for record in records:
        assert record["source_record_id"].startswith("2024-09-18:statement:")


def test_iter_fomc_archive_records_pins_revision(monkeypatch) -> None:
    captured = _install_fake_datasets_module(
        monkeypatch,
        [
            {
                "Date": "2024-09-18",
                "Release Date": "2024-09-18",
                "Type": "Statement",
                "Text": "Sample statement.",
            },
        ],
    )

    records = _iter_fomc_archive_records()

    expected_revision = _DATASET_REVISIONS[VTASCA_FOMC_ARCHIVE_DATASET_ID]
    assert captured["revisions"] == [expected_revision]
    assert records[0]["multi_axis_extras"]["vtasca_dataset_revision"] == expected_revision


def test_dataset_revision_returns_pinned_or_none() -> None:
    assert _dataset_revision(GTFINTECHLAB_FED_DATASET_ID) == _DATASET_REVISIONS[GTFINTECHLAB_FED_DATASET_ID]
    assert _dataset_revision(VTASCA_FOMC_ARCHIVE_DATASET_ID) == _DATASET_REVISIONS[VTASCA_FOMC_ARCHIVE_DATASET_ID]
    assert _dataset_revision("unknown/dataset") is None


def test_iter_fomc_archive_records_dedupes_by_text_hash(monkeypatch) -> None:
    _install_fake_datasets_module(
        monkeypatch,
        [
            {
                "Date": "2024-09-18",
                "Release Date": "2024-09-18",
                "Type": "Statement",
                "Text": "Duplicate statement text.",
            },
            {
                "Date": "2024-09-18",
                "Release Date": "2024-09-18",
                "Type": "Statement",
                "Text": "Duplicate statement text.",
            },
        ],
    )

    records = _iter_fomc_archive_records()

    assert len(records) == 1
