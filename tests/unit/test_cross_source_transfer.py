"""Unit tests for app.evaluation.cross_source_transfer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation import cross_source_transfer as cst


def _write_registry(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


@pytest.fixture
def package_with_multiple_source_types(tmp_path: Path) -> Path:
    package_dir = tmp_path / "processed" / "tp_v1"
    _write_registry(
        package_dir / "registry_normalized.jsonl",
        [
            # Two FOMC statements (in-domain reference).
            {
                "record_id": "stmt_1",
                "text": "policy is appropriate to return inflation to target",
                "event_date": "2024-01-31",
                "source_type": "fomc_statement",
                "source": "hf_fomc_communication",
                "mapped_label": "neutral",
                "provenance": "peer_reviewed",
                "sample_weight": 1.0,
            },
            {
                "record_id": "stmt_2",
                "text": "the committee will respond to risks",
                "event_date": "2024-03-20",
                "source_type": "fomc_statement",
                "source": "hf_fomc_communication",
                "mapped_label": "hawkish",
                "provenance": "peer_reviewed",
                "sample_weight": 1.0,
            },
            # Three meeting-transcript sentences (Op-Fed-style).
            {
                "record_id": "tx_1",
                "text": "we should tighten further to restore price stability",
                "event_date": "2007-08-07",
                "source_type": "fomc_meeting_transcript",
                "source": "op_fed",
                "mapped_label": "hawkish",
                "provenance": "peer_reviewed",
                "sample_weight": 1.0,
            },
            {
                "record_id": "tx_2",
                "text": "conditions argue for accommodation at this meeting",
                "event_date": "2007-08-07",
                "source_type": "fomc_meeting_transcript",
                "source": "op_fed",
                "mapped_label": "dovish",
                "provenance": "peer_reviewed",
                "sample_weight": 1.0,
            },
            {
                "record_id": "tx_3",
                "text": "the staff outlook is consistent with the current stance",
                "event_date": "2007-09-18",
                "source_type": "fomc_meeting_transcript",
                "source": "op_fed",
                "mapped_label": "neutral",
                "provenance": "peer_reviewed",
                "sample_weight": 1.0,
            },
            # Cross-bank row with sample_weight=0 — must be dropped by default.
            {
                "record_id": "ecb_1",
                "text": "ECB inflation outlook firm",
                "event_date": "2024-02-01",
                "source_type": "fomc_statement",
                "source": "gtfintechlab_european_central_bank",
                "mapped_label": "hawkish",
                "provenance": "peer_reviewed_cross_bank",
                "sample_weight": 0.0,
            },
            # Unlabelled row.
            {
                "record_id": "scrape_1",
                "text": "unlabelled archive text",
                "event_date": "2010-06-23",
                "source_type": "fomc_statement",
                "source": "vtasca_fomc_archive",
                "mapped_label": "",
                "provenance": "scraped",
                "sample_weight": 0.0,
            },
        ],
    )
    return package_dir


def test_load_cross_source_rows_drops_zero_weight_and_unlabelled(
    package_with_multiple_source_types: Path,
) -> None:
    rows = cst.load_cross_source_rows(package_with_multiple_source_types)
    record_ids = {r.record_id for r in rows}
    assert record_ids == {"stmt_1", "stmt_2", "tx_1", "tx_2", "tx_3"}


def test_group_by_source_type_partitions_rows(
    package_with_multiple_source_types: Path,
) -> None:
    rows = cst.load_cross_source_rows(package_with_multiple_source_types)
    buckets = cst.group_by_source_type(rows)
    assert set(buckets.keys()) == {"fomc_statement", "fomc_meeting_transcript"}
    assert len(buckets["fomc_meeting_transcript"]) == 3
    assert len(buckets["fomc_statement"]) == 2


def test_evaluate_source_with_oracle_predictor(
    package_with_multiple_source_types: Path,
) -> None:
    rows = cst.load_cross_source_rows(package_with_multiple_source_types)
    transcripts = cst.group_by_source_type(rows)["fomc_meeting_transcript"]

    def oracle(rs):
        labels = [r.label for r in rs]
        return list(labels), list(labels), [0.5] * len(labels)

    result = cst.evaluate_source(
        transcripts,
        source_type="fomc_meeting_transcript",
        encoder_alias="oracle",
        checkpoint="oracle",
        predict_fn=oracle,
    )
    assert result.support == 3
    assert result.accuracy == pytest.approx(1.0)
    assert result.macro_f1 == pytest.approx(1.0)
    assert result.label_support == {"hawkish": 1, "dovish": 1, "neutral": 1}


def test_evaluate_source_with_constant_predictor(
    package_with_multiple_source_types: Path,
) -> None:
    rows = cst.load_cross_source_rows(package_with_multiple_source_types)
    transcripts = cst.group_by_source_type(rows)["fomc_meeting_transcript"]

    def always_neutral(rs):
        return [r.label for r in rs], ["neutral"] * len(rs), [0.4] * len(rs)

    result = cst.evaluate_source(
        transcripts,
        source_type="fomc_meeting_transcript",
        encoder_alias="constant",
        checkpoint="constant",
        predict_fn=always_neutral,
    )
    assert result.macro_f1 < 0.6
    assert result.accuracy == pytest.approx(1 / 3)


def test_build_matrix_emits_per_source_counts(
    package_with_multiple_source_types: Path,
) -> None:
    def oracle(rs):
        labels = [r.label for r in rs]
        return list(labels), list(labels), [0.3] * len(labels)

    matrix = cst.build_matrix(
        package_dir=package_with_multiple_source_types,
        encoder_checkpoints={"oracle": "oracle"},
        predict_fn=oracle,
    )
    counts = matrix["per_source_counts"]
    assert counts["fomc_statement"] == 2
    assert counts["fomc_meeting_transcript"] == 3
    assert counts["chair_speech"] == 0  # under-populated stays visible

    statuses = {(c["source_type"], c["status"]) for c in matrix["cells"]}
    assert ("fomc_statement", "ok") in statuses
    assert ("fomc_meeting_transcript", "ok") in statuses
    assert ("chair_speech", "no_rows") in statuses


def test_render_csv_carries_per_class_counts(
    package_with_multiple_source_types: Path,
) -> None:
    def oracle(rs):
        labels = [r.label for r in rs]
        return list(labels), list(labels), [0.3] * len(labels)

    matrix = cst.build_matrix(
        package_dir=package_with_multiple_source_types,
        encoder_checkpoints={"oracle": "oracle"},
        predict_fn=oracle,
    )
    csv_text = cst.render_csv(matrix)
    assert csv_text.splitlines()[0].split(",")[:5] == [
        "encoder_alias",
        "checkpoint",
        "source_type",
        "status",
        "support",
    ]
    # Per-class counts must surface so under-populated cells are visible.
    assert "dovish_n" in csv_text.splitlines()[0]
    assert "hawkish_n" in csv_text.splitlines()[0]
    assert "neutral_n" in csv_text.splitlines()[0]


def test_evaluate_source_rejects_empty_rows() -> None:
    with pytest.raises(ValueError, match="No labelled rows"):
        cst.evaluate_source(
            [],
            source_type="fomc_statement",
            encoder_alias="x",
            checkpoint="x",
            predict_fn=lambda rs: ([], [], []),
        )


def test_load_cross_source_rows_includes_zero_weight_when_flag_set(
    package_with_multiple_source_types: Path,
) -> None:
    rows = cst.load_cross_source_rows(
        package_with_multiple_source_types, include_zero_weight=True
    )
    assert any(r.record_id == "ecb_1" for r in rows)


def test_parse_encoder_spec_rejects_duplicate_aliases() -> None:
    with pytest.raises(ValueError, match="duplicated"):
        cst._parse_encoder_spec("a=x,a=y")


def test_parse_source_types_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown source_type"):
        cst._parse_source_types("not_a_real_source")
