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


# ----- continuous-arm (GSS factor decomposition) ----------------------------


@pytest.fixture
def package_with_gss_rows(tmp_path: Path) -> Path:
    package_dir = tmp_path / "processed" / "tp_gss"
    _write_registry(
        package_dir / "registry_normalized.jsonl",
        [
            # GSS rows ship sample_weight=0 by design (eval-only, never
            # enter training) — the continuous arm must still pick them up.
            {
                "record_id": "gss_1994-08-16",
                "text": "GSS factor decomposition for 1994-08-16: target=+10.70 bp, path=-8.30 bp",
                "event_date": "1994-08-16",
                "source_type": "gss_factor_decomposition",
                "source": "gurkaynak_sack_swanson_2005",
                "mapped_label": "",
                "provenance": "peer_reviewed",
                "sample_weight": 0.0,
                "multi_axis_extras": {
                    "gss_target_factor": 10.7,
                    "gss_path_factor": -8.3,
                    "gss_fomc_statement": True,
                },
            },
            {
                "record_id": "gss_2001-01-03",
                "text": "GSS factor decomposition for 2001-01-03: target=-32.30 bp, path=+22.80 bp",
                "event_date": "2001-01-03",
                "source_type": "gss_factor_decomposition",
                "source": "gurkaynak_sack_swanson_2005",
                "mapped_label": "",
                "provenance": "peer_reviewed",
                "sample_weight": 0.0,
                "multi_axis_extras": {
                    "gss_target_factor": -32.3,
                    "gss_path_factor": 22.8,
                    "gss_fomc_statement": True,
                },
            },
            {
                "record_id": "gss_1990-02-08",
                "text": "GSS factor decomposition for 1990-02-08: target=+0.30 bp, path=+5.80 bp",
                "event_date": "1990-02-08",
                "source_type": "gss_factor_decomposition",
                "source": "gurkaynak_sack_swanson_2005",
                "mapped_label": "",
                "provenance": "peer_reviewed",
                "sample_weight": 0.0,
                "multi_axis_extras": {
                    "gss_target_factor": 0.3,
                    "gss_path_factor": 5.8,
                    "gss_fomc_statement": False,
                },
            },
        ],
    )
    return package_dir


def test_load_cross_source_rows_picks_up_continuous_rows(
    package_with_gss_rows: Path,
) -> None:
    rows = cst.load_cross_source_rows(package_with_gss_rows)
    assert len(rows) == 3
    assert all(r.source_type == "gss_factor_decomposition" for r in rows)
    assert all(r.label == "" for r in rows)
    by_id = {r.record_id: r for r in rows}
    assert by_id["gss_2001-01-03"].multi_axis_extras["gss_target_factor"] == pytest.approx(-32.3)


def test_evaluate_continuous_source_with_perfect_oracle(
    package_with_gss_rows: Path,
) -> None:
    rows = cst.load_cross_source_rows(package_with_gss_rows)

    def perfect_oracle(rs):
        # Return the target factor itself as the "score". A perfect
        # predictor must hit Pearson r = 1.0 and zero z-scored RMSE.
        scores = [float(r.multi_axis_extras["gss_target_factor"]) for r in rs]
        return scores, [0.5] * len(rs)

    result = cst.evaluate_continuous_source(
        rows,
        source_type="gss_factor_decomposition",
        encoder_alias="oracle",
        checkpoint="oracle",
        predict_fn=perfect_oracle,
    )
    assert result.support == 3
    target = result.targets["gss_target_factor"]
    assert target["support"] == 3
    assert target["pearson_r"] == pytest.approx(1.0, abs=1e-9)
    assert target["spearman_r"] == pytest.approx(1.0, abs=1e-9)
    assert target["zscore_rmse"] == pytest.approx(0.0, abs=1e-9)


def test_evaluate_continuous_source_with_anti_correlated_predictor(
    package_with_gss_rows: Path,
) -> None:
    rows = cst.load_cross_source_rows(package_with_gss_rows)

    def anti(rs):
        return [-float(r.multi_axis_extras["gss_target_factor"]) for r in rs], [0.4] * len(rs)

    result = cst.evaluate_continuous_source(
        rows,
        source_type="gss_factor_decomposition",
        encoder_alias="anti",
        checkpoint="anti",
        predict_fn=anti,
    )
    target = result.targets["gss_target_factor"]
    assert target["pearson_r"] == pytest.approx(-1.0, abs=1e-9)
    assert target["spearman_r"] == pytest.approx(-1.0, abs=1e-9)


def test_evaluate_continuous_source_rejects_score_count_mismatch(
    package_with_gss_rows: Path,
) -> None:
    rows = cst.load_cross_source_rows(package_with_gss_rows)
    with pytest.raises(ValueError, match="scores for"):
        cst.evaluate_continuous_source(
            rows,
            source_type="gss_factor_decomposition",
            encoder_alias="x",
            checkpoint="x",
            predict_fn=lambda rs: ([0.1], [0.1]),
        )


def test_evaluate_continuous_source_rejects_empty_rows() -> None:
    with pytest.raises(ValueError, match="No continuous rows"):
        cst.evaluate_continuous_source(
            [],
            source_type="gss_factor_decomposition",
            encoder_alias="x",
            checkpoint="x",
            predict_fn=lambda rs: ([], []),
        )


def test_evaluate_continuous_source_rejects_unconfigured_type(
    package_with_gss_rows: Path,
) -> None:
    rows = cst.load_cross_source_rows(package_with_gss_rows)
    with pytest.raises(ValueError, match="No continuous targets configured"):
        cst.evaluate_continuous_source(
            rows,
            source_type="fomc_statement",  # not in CONTINUOUS_TARGETS
            encoder_alias="x",
            checkpoint="x",
            predict_fn=lambda rs: ([0.0] * len(rs), [0.1] * len(rs)),
        )


def test_build_matrix_dispatches_continuous_arm(
    package_with_gss_rows: Path,
) -> None:
    def oracle(rs):
        return [float(r.multi_axis_extras["gss_target_factor"]) for r in rs], [0.3] * len(rs)

    matrix = cst.build_matrix(
        package_dir=package_with_gss_rows,
        encoder_checkpoints={"oracle": "oracle"},
        source_types=["gss_factor_decomposition", "fomc_statement"],
        continuous_predict_fn=oracle,
    )
    counts = matrix["per_source_counts"]
    assert counts["gss_factor_decomposition"] == 3
    assert counts["fomc_statement"] == 0

    by_type = {(c["source_type"], c["status"]): c for c in matrix["cells"]}
    gss_cell = by_type[("gss_factor_decomposition", "ok")]
    assert gss_cell["kind"] == "continuous"
    assert gss_cell["support"] == 3
    assert gss_cell["targets"]["gss_target_factor"]["pearson_r"] == pytest.approx(1.0, abs=1e-9)
    # Empty stance source still emits a no_rows cell, tagged kind=stance.
    stance_cell = by_type[("fomc_statement", "no_rows")]
    assert stance_cell["kind"] == "stance"


def test_render_continuous_csv_one_row_per_target(
    package_with_gss_rows: Path,
) -> None:
    def oracle(rs):
        return [float(r.multi_axis_extras["gss_target_factor"]) for r in rs], [0.3] * len(rs)

    matrix = cst.build_matrix(
        package_dir=package_with_gss_rows,
        encoder_checkpoints={"oracle": "oracle"},
        source_types=["gss_factor_decomposition"],
        continuous_predict_fn=oracle,
    )
    csv_text = cst.render_continuous_csv(matrix)
    header = csv_text.splitlines()[0].split(",")
    assert header[:7] == [
        "encoder_alias",
        "checkpoint",
        "source_type",
        "status",
        "support",
        "target_key",
        "paired_support",
    ]
    # Two rows: one for gss_target_factor, one for gss_path_factor.
    body = csv_text.splitlines()[1:]
    assert len(body) == 2
    keys = {line.split(",")[5] for line in body}
    assert keys == {"gss_target_factor", "gss_path_factor"}


def test_render_csv_skips_continuous_cells(
    package_with_gss_rows: Path,
) -> None:
    def oracle(rs):
        return [float(r.multi_axis_extras["gss_target_factor"]) for r in rs], [0.3] * len(rs)

    matrix = cst.build_matrix(
        package_dir=package_with_gss_rows,
        encoder_checkpoints={"oracle": "oracle"},
        source_types=["gss_factor_decomposition"],
        continuous_predict_fn=oracle,
    )
    # Stance CSV must NOT contain the continuous cell — its per-class
    # columns don't apply and would render as zeros, masking the cell.
    csv_text = cst.render_csv(matrix)
    body_lines = [line for line in csv_text.splitlines()[1:] if line.strip()]
    assert body_lines == []


def test_pearson_and_spearman_helpers_handle_degenerate_inputs() -> None:
    assert cst._pearson([1.0], [1.0]) is None
    assert cst._pearson([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]) is None
    assert cst._spearman([], []) is None
    assert cst._zscore_rmse([1.0, 1.0], [2.0, 3.0]) is None
    # Monotone agreement -> Spearman 1, Pearson positive but < 1 on non-linear.
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [1.0, 4.0, 9.0, 16.0]
    assert cst._spearman(xs, ys) == pytest.approx(1.0, abs=1e-9)
    assert (cst._pearson(xs, ys) or 0.0) > 0.9
