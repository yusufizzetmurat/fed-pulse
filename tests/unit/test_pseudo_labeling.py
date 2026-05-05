from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.data import pseudo_labeling


def test_parse_args_requires_teacher_checkpoint() -> None:
    with pytest.raises(SystemExit):
        pseudo_labeling._parse_args([])


def test_parse_args_accepts_threshold_flag() -> None:
    args = pseudo_labeling._parse_args(
        [
            "--teacher-checkpoint",
            "/some/path",
            "--threshold",
            "0.95",
        ]
    )
    assert args.teacher_checkpoint == "/some/path"
    assert args.threshold == 0.95


def test_parse_args_default_threshold_is_0_85() -> None:
    args = pseudo_labeling._parse_args(
        ["--teacher-checkpoint", "/some/path"]
    )
    assert args.threshold == 0.85
    assert args.teacher_model_id == "fomc_roberta_s71"
    assert args.teacher_model_version == "phase4_finetune_v1"


class _StubPipeline:
    """Stand-in for transformers.pipeline that the teacher loader returns."""

    def __init__(self, label_to_score: list[list[dict[str, float]]]):
        self._batches = label_to_score

    def __call__(self, texts, **kwargs):
        # Mirror the transformers `text-classification` shape with
        # return_all_scores=True: list of [list of {label, score}].
        out = []
        for _ in texts:
            out.append(self._batches.pop(0))
        return out


def test_score_passages_returns_one_prediction_per_passage_with_label_and_confidence() -> None:
    pipeline = _StubPipeline(
        [
            [
                {"label": "hawkish", "score": 0.92},
                {"label": "dovish", "score": 0.05},
                {"label": "neutral", "score": 0.03},
            ],
            [
                {"label": "hawkish", "score": 0.30},
                {"label": "dovish", "score": 0.40},
                {"label": "neutral", "score": 0.30},
            ],
        ]
    )

    predictions = pseudo_labeling.score_passages(
        ["Strong tightening signal.", "Mixed signals on the labor market."],
        pipeline=pipeline,
    )

    assert len(predictions) == 2
    assert predictions[0]["predicted_label"] == "hawkish"
    assert predictions[0]["max_score"] == pytest.approx(0.92)
    assert predictions[0]["scores"] == {"hawkish": 0.92, "dovish": 0.05, "neutral": 0.03}
    assert predictions[1]["predicted_label"] == "dovish"
    assert predictions[1]["max_score"] == pytest.approx(0.40)


def test_apply_threshold_partitions_predictions_into_kept_and_dropped() -> None:
    predictions = [
        {"predicted_label": "hawkish", "max_score": 0.92, "scores": {}},
        {"predicted_label": "dovish", "max_score": 0.40, "scores": {}},
        {"predicted_label": "neutral", "max_score": 0.85, "scores": {}},
    ]
    kept, dropped = pseudo_labeling.apply_threshold(predictions, threshold=0.85)
    assert len(kept) == 2
    assert len(dropped) == 1
    assert kept[0]["predicted_label"] == "hawkish"
    assert dropped[0]["predicted_label"] == "dovish"


def test_build_pseudo_row_carries_provenance_and_label() -> None:
    source_row = {
        "record_id": "abc123",
        "source": "scraped_fed",
        "source_record_id": "fomc_minutes.json:12",
        "document_type": "minutes",
        "source_type": "fomc_minutes",
        "event_date": "2024-01-31",
        "title": "FOMC Meeting Minutes",
        "text": "Some passage text",
        "text_hash": "deadbeef",
        "license_scope": "public_source_scrape_terms_required",
        "citation_ref": "federalreserve_primary_source",
        "ingested_at_utc": "2024-01-31T00:00:00+00:00",
    }
    prediction = {
        "predicted_label": "hawkish",
        "max_score": 0.92,
        "scores": {"hawkish": 0.92, "dovish": 0.05, "neutral": 0.03},
    }
    row = pseudo_labeling.build_pseudo_row(
        source_row,
        prediction,
        teacher_model_id="fomc_roberta_s71",
        teacher_model_version="phase4_finetune_v1",
    )

    assert row["record_id"] == "abc123"
    assert row["label"] == "hawkish"
    assert row["label_origin"] == "pseudo"
    assert row["teacher_model_id"] == "fomc_roberta_s71"
    assert row["teacher_model_version"] == "phase4_finetune_v1"
    assert row["teacher_max_score"] == pytest.approx(0.92)
    assert row["teacher_scores"] == prediction["scores"]
    assert row["source"] == "scraped_fed"
    assert row["source_type"] == "fomc_minutes"
    assert row["text"] == "Some passage text"


def _write_registry_fixture(path: Path) -> None:
    rows = [
        {
            "record_id": "r1",
            "source": "scraped_fed",
            "source_record_id": "fomc_minutes.json:0",
            "document_type": "minutes",
            "source_type": "fomc_minutes",
            "event_date": "2024-01-31",
            "title": "FOMC Meeting Minutes",
            "text": "Hawkish passage about tightening monetary policy.",
            "text_hash": "hash1",
            "license_scope": "public_source_scrape_terms_required",
            "citation_ref": "federalreserve_primary_source",
            "ingested_at_utc": "2024-01-31T00:00:00+00:00",
            "label": "",
            "label_origin": "pseudo",
        },
        {
            "record_id": "r2",
            "source": "scraped_fed",
            "source_record_id": "fomc_minutes.json:1",
            "document_type": "minutes",
            "source_type": "fomc_minutes",
            "event_date": "2024-03-20",
            "title": "FOMC Meeting Minutes",
            "text": "Mixed signals on growth.",
            "text_hash": "hash2",
            "license_scope": "public_source_scrape_terms_required",
            "citation_ref": "federalreserve_primary_source",
            "ingested_at_utc": "2024-03-20T00:00:00+00:00",
            "label": "",
            "label_origin": "pseudo",
        },
        {
            "record_id": "r3",
            "source": "kaggle_fed_statements_minutes",
            "source_record_id": "kaggle:0",
            "document_type": "statement",
            "source_type": "fomc_statement",
            "event_date": "2024-04-15",
            "title": "Statement",
            "text": "A statement that already has a label.",
            "text_hash": "hash3",
            "license_scope": "source_terms_required",
            "citation_ref": "kaggle_drlexus_fed_statements_and_minutes",
            "ingested_at_utc": "2024-04-15T00:00:00+00:00",
            "label": "hawkish",
            "label_origin": "human",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_run_pseudo_labeling_scores_only_unlabelled_rows_and_writes_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "registry.jsonl"
    output_path = tmp_path / "registry_pseudo.jsonl"
    _write_registry_fixture(input_path)

    pipeline = _StubPipeline(
        [
            [
                {"label": "hawkish", "score": 0.92},
                {"label": "dovish", "score": 0.05},
                {"label": "neutral", "score": 0.03},
            ],
            [
                {"label": "neutral", "score": 0.40},
                {"label": "hawkish", "score": 0.35},
                {"label": "dovish", "score": 0.25},
            ],
        ]
    )

    written = pseudo_labeling.run_pseudo_labeling(
        input_path=input_path,
        output_path=output_path,
        teacher_pipeline=pipeline,
        threshold=0.85,
        teacher_model_id="fomc_roberta_s71",
        teacher_model_version="phase4_finetune_v1",
    )

    assert written == 1

    pseudo_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(pseudo_rows) == 1
    pseudo = pseudo_rows[0]
    assert pseudo["record_id"] == "r1"
    assert pseudo["label"] == "hawkish"
    assert pseudo["label_origin"] == "pseudo"
    assert pseudo["teacher_model_id"] == "fomc_roberta_s71"
    assert pseudo["teacher_max_score"] == pytest.approx(0.92)
    assert all(p["record_id"] != "r3" for p in pseudo_rows)
