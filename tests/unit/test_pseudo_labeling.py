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


def test_default_strategy_is_chunk_vote() -> None:
    """The CPU smoke on 2026-05-14 showed chunk_max_pool collapses to all-hawkish
    (same failure mode as doc_truncated). chunk_vote surfaces class diversity by
    taking the modal label across chunks above tau_chunk. The module-level
    default and the CLI default must agree on chunk_vote so a call without
    --strategy reproduces the data-supported behaviour."""

    assert pseudo_labeling.DEFAULT_STRATEGY == "chunk_vote"
    args = pseudo_labeling._parse_args(["--teacher-checkpoint", "/some/path"])
    assert args.strategy == "chunk_vote"


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
        # Deterministic single-chunk splitter so the chunk-aware default strategy
        # does not load the (gated) HF tokenizer — matches the sibling chunked
        # tests and keeps this test independent of model availability.
        splitter=lambda text: [text],
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


def _chunk_pred(label: str, max_score: float, scores: dict[str, float] | None = None) -> dict[str, float]:
    """Shorthand for an aggregator chunk-prediction dict."""

    base = {"hawkish": 0.0, "dovish": 0.0, "neutral": 0.0}
    if scores is not None:
        base.update(scores)
    base[label] = max_score
    return {"predicted_label": label, "max_score": max_score, "scores": base}


def test_aggregate_chunk_predictions_max_pool_picks_highest_confidence_chunk_above_floor() -> None:
    chunks = [
        _chunk_pred("neutral", 0.55),
        _chunk_pred("hawkish", 0.92),  # winner
        _chunk_pred("dovish", 0.40),  # below floor 0.5 — dropped
    ]
    aggregate = pseudo_labeling.aggregate_chunk_predictions(
        chunks, strategy="chunk_max_pool", tau_chunk=0.5
    )
    assert aggregate["predicted_label"] == "hawkish"
    assert aggregate["max_score"] == pytest.approx(0.92)
    assert aggregate["chunk_count"] == 3
    assert aggregate["chunks_above_floor"] == 2
    assert aggregate["strategy"] == "chunk_max_pool"


def test_aggregate_chunk_predictions_mean_pool_averages_per_class_probabilities_across_chunks() -> None:
    chunks = [
        {"predicted_label": "hawkish", "max_score": 0.7,
         "scores": {"hawkish": 0.7, "dovish": 0.2, "neutral": 0.1}},
        {"predicted_label": "hawkish", "max_score": 0.6,
         "scores": {"hawkish": 0.6, "dovish": 0.3, "neutral": 0.1}},
        {"predicted_label": "dovish", "max_score": 0.55,
         "scores": {"hawkish": 0.2, "dovish": 0.55, "neutral": 0.25}},
    ]
    aggregate = pseudo_labeling.aggregate_chunk_predictions(
        chunks, strategy="chunk_mean_pool", tau_chunk=0.5
    )
    assert aggregate["predicted_label"] == "hawkish"
    assert aggregate["scores"]["hawkish"] == pytest.approx((0.7 + 0.6 + 0.2) / 3)
    assert aggregate["scores"]["dovish"] == pytest.approx((0.2 + 0.3 + 0.55) / 3)
    assert aggregate["chunks_above_floor"] == 3


def test_aggregate_chunk_predictions_vote_returns_modal_label_with_tiebreak_on_mean_confidence() -> None:
    chunks = [
        _chunk_pred("hawkish", 0.6),
        _chunk_pred("hawkish", 0.55),
        _chunk_pred("dovish", 0.95),  # higher confidence but minority
        _chunk_pred("dovish", 0.90),
    ]
    aggregate = pseudo_labeling.aggregate_chunk_predictions(
        chunks, strategy="chunk_vote", tau_chunk=0.5
    )
    # Both labels have 2 votes; tiebreak picks the label with higher mean confidence.
    assert aggregate["predicted_label"] == "dovish"
    assert aggregate["max_score"] == pytest.approx(0.95)
    assert aggregate["vote_counts"] == {"hawkish": 2, "dovish": 2}


def test_aggregate_chunk_predictions_falls_back_when_no_chunk_clears_the_floor() -> None:
    chunks = [
        _chunk_pred("neutral", 0.45),
        _chunk_pred("hawkish", 0.40),
    ]
    aggregate = pseudo_labeling.aggregate_chunk_predictions(
        chunks, strategy="chunk_max_pool", tau_chunk=0.5
    )
    # max_score is 0.0 so the doc-level threshold (tau_doc) will discard
    # the row, but we still report the fallback label for diagnostics.
    assert aggregate["max_score"] == 0.0
    assert aggregate["predicted_label"] == "neutral"
    assert aggregate["chunks_above_floor"] == 0
    assert aggregate["fallback_max_score"] == pytest.approx(0.45)


def test_aggregate_chunk_predictions_handles_empty_input() -> None:
    aggregate = pseudo_labeling.aggregate_chunk_predictions(
        [], strategy="chunk_max_pool", tau_chunk=0.5
    )
    assert aggregate["predicted_label"] == ""
    assert aggregate["chunk_count"] == 0
    assert aggregate["max_score"] == 0.0


def test_aggregate_chunk_predictions_rejects_unknown_strategy() -> None:
    chunks = [_chunk_pred("hawkish", 0.9)]
    with pytest.raises(ValueError, match="unknown aggregation strategy"):
        pseudo_labeling.aggregate_chunk_predictions(
            chunks, strategy="not_a_strategy", tau_chunk=0.5  # type: ignore[arg-type]
        )


def test_score_passages_chunked_routes_each_doc_through_its_chunks_and_aggregates() -> None:
    # One doc with three chunks: middle chunk dominates with max confidence.
    chunk_batches = [
        [
            {"label": "neutral", "score": 0.55},
            {"label": "hawkish", "score": 0.30},
            {"label": "dovish", "score": 0.15},
        ],
        [
            {"label": "hawkish", "score": 0.95},
            {"label": "neutral", "score": 0.03},
            {"label": "dovish", "score": 0.02},
        ],
        [
            {"label": "neutral", "score": 0.40},
            {"label": "dovish", "score": 0.35},
            {"label": "hawkish", "score": 0.25},
        ],
    ]
    pipeline = _StubPipeline(chunk_batches)
    splitter = lambda text: ["chunk1", "chunk2", "chunk3"]

    predictions = pseudo_labeling.score_passages_chunked(
        ["long document text"],
        pipeline=pipeline,
        strategy="chunk_max_pool",
        tau_chunk=0.5,
        splitter=splitter,
    )
    assert len(predictions) == 1
    aggregate = predictions[0]
    assert aggregate["predicted_label"] == "hawkish"
    assert aggregate["max_score"] == pytest.approx(0.95)
    assert aggregate["chunk_count"] == 3
    assert aggregate["chunks_above_floor"] == 2  # 0.55 + 0.95 above floor


def test_run_pseudo_labeling_dispatches_chunk_strategy_and_persists_diagnostics(tmp_path: Path) -> None:
    input_path = tmp_path / "source_registry.jsonl"
    output_path = tmp_path / "registry_pseudo.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "record_id": "doc-1",
                "source": "scraped_fed",
                "source_record_id": "fomc_minutes_2023-09-20",
                "event_date": "2023-09-20",
                "text": "doc-1 long text",
                "label": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    chunk_batches = [
        [
            {"label": "neutral", "score": 0.40},
            {"label": "dovish", "score": 0.30},
            {"label": "hawkish", "score": 0.30},
        ],
        [
            {"label": "hawkish", "score": 0.91},
            {"label": "neutral", "score": 0.07},
            {"label": "dovish", "score": 0.02},
        ],
    ]
    pipeline = _StubPipeline(chunk_batches)
    splitter = lambda text: ["chunk-a", "chunk-b"]

    written = pseudo_labeling.run_pseudo_labeling(
        input_path=input_path,
        output_path=output_path,
        teacher_pipeline=pipeline,
        threshold=0.85,
        teacher_model_id="fomc_roberta_s71",
        teacher_model_version="phase4_finetune_v1",
        strategy="chunk_max_pool",
        tau_chunk=0.5,
        splitter=splitter,
    )

    assert written == 1
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["label"] == "hawkish"
    assert rows[0]["teacher_aggregation"]["strategy"] == "chunk_max_pool"
    assert rows[0]["teacher_aggregation"]["chunk_count"] == 2
    assert rows[0]["teacher_aggregation"]["chunks_above_floor"] == 1
    assert rows[0]["teacher_aggregation"]["tau_chunk"] == 0.5


def test_run_pseudo_labeling_rejects_unknown_strategy(tmp_path: Path) -> None:
    input_path = tmp_path / "registry.jsonl"
    output_path = tmp_path / "pseudo.jsonl"
    input_path.write_text(
        json.dumps({"record_id": "x", "text": "t", "event_date": "2024-01-01", "label": ""}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown strategy"):
        pseudo_labeling.run_pseudo_labeling(
            input_path=input_path,
            output_path=output_path,
            teacher_pipeline=_StubPipeline([]),
            threshold=0.5,
            teacher_model_id="t",
            teacher_model_version="v",
            strategy="garbage",  # type: ignore[arg-type]
        )


def test_threshold_sweep_reports_yield_and_label_distribution_per_threshold() -> None:
    predictions = [
        {"predicted_label": "hawkish", "max_score": 0.92, "scores": {}},
        {"predicted_label": "hawkish", "max_score": 0.80, "scores": {}},
        {"predicted_label": "dovish", "max_score": 0.78, "scores": {}},
        {"predicted_label": "dovish", "max_score": 0.40, "scores": {}},
        {"predicted_label": "neutral", "max_score": 0.97, "scores": {}},
    ]

    sweep = pseudo_labeling.threshold_sweep(predictions, thresholds=(0.75, 0.85, 0.95))

    assert sweep["thresholds"] == [0.75, 0.85, 0.95]
    assert sweep["total"] == 5
    assert sweep["yield"] == {"0.75": 4, "0.85": 2, "0.95": 1}
    assert sweep["label_distribution"]["0.85"] == {"hawkish": 1, "neutral": 1}
    assert sweep["label_distribution"]["0.95"] == {"neutral": 1}
