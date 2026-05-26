"""Tests for the per-axis × per-encoder table aggregator (#82)."""

from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.multi_axis_table import (
    build_rows,
    render_markdown,
    write_csv,
    write_json,
)


def _aggregate_per_axis() -> dict:
    """Synthetic 2-encoder × 2-axis fixture mirroring the future
    multi-axis bake-off output shape."""

    return {
        "by_encoder": {
            "encoder_a": {
                "per_axis": {
                    "stance": {
                        "per_seed": {
                            "11": {"macro_f1": 0.60, "weighted_f1": 0.62, "accuracy": 0.61},
                            "29": {"macro_f1": 0.62, "weighted_f1": 0.63, "accuracy": 0.63},
                            "47": {"macro_f1": 0.58, "weighted_f1": 0.60, "accuracy": 0.59},
                            "71": {"macro_f1": 0.61, "weighted_f1": 0.62, "accuracy": 0.60},
                            "97": {"macro_f1": 0.63, "weighted_f1": 0.64, "accuracy": 0.64},
                        }
                    },
                    "factor": {
                        "per_seed": {
                            "11": {"macro_f1": 0.48, "weighted_f1": 0.49, "accuracy": 0.51},
                            "29": {"macro_f1": 0.50, "weighted_f1": 0.51, "accuracy": 0.52},
                            "47": {"macro_f1": 0.47, "weighted_f1": 0.48, "accuracy": 0.50},
                        }
                    },
                }
            },
            "encoder_b": {
                "per_axis": {
                    "stance": {
                        "per_seed": {
                            "11": {"macro_f1": 0.65, "weighted_f1": 0.66, "accuracy": 0.66},
                            "29": {"macro_f1": 0.66, "weighted_f1": 0.67, "accuracy": 0.67},
                            "47": {"macro_f1": 0.64, "weighted_f1": 0.65, "accuracy": 0.65},
                        }
                    },
                    "factor": {
                        "per_seed": {
                            "11": {"macro_f1": 0.50, "weighted_f1": 0.51, "accuracy": 0.52},
                            "29": {"macro_f1": 0.52, "weighted_f1": 0.53, "accuracy": 0.54},
                        }
                    },
                }
            },
        }
    }


def test_build_rows_covers_every_axis_encoder_metric_combination() -> None:
    rows = build_rows(_aggregate_per_axis())
    # 2 encoders × 2 axes × 3 metrics = 12 rows
    assert len(rows) == 12
    triples = {(r.encoder, r.axis, r.metric) for r in rows}
    expected = {
        (enc, axis, metric)
        for enc in ("encoder_a", "encoder_b")
        for axis in ("stance", "factor")
        for metric in ("macro_f1", "weighted_f1", "accuracy")
    }
    assert triples == expected


def test_rows_carry_bootstrap_ci_when_multi_seed() -> None:
    rows = build_rows(_aggregate_per_axis())
    for row in rows:
        if row.n > 1:
            assert row.ci_lo is not None and row.ci_hi is not None
            assert row.ci_lo <= row.mean <= row.ci_hi


def test_legacy_per_seed_shape_falls_through_as_stance_only() -> None:
    legacy = {
        "by_encoder": {
            "encoder_legacy": {
                "per_seed": {
                    "11": {"macro_f1": 0.70, "weighted_f1": 0.71, "accuracy": 0.72},
                    "29": {"macro_f1": 0.72, "weighted_f1": 0.73, "accuracy": 0.74},
                }
            }
        }
    }
    rows = build_rows(legacy)
    assert {r.axis for r in rows} == {"stance"}
    assert {r.encoder for r in rows} == {"encoder_legacy"}
    assert {r.metric for r in rows} == {"macro_f1", "weighted_f1", "accuracy"}


def test_csv_and_markdown_outputs_round_trip(tmp_path: Path) -> None:
    rows = build_rows(_aggregate_per_axis())
    csv_path = tmp_path / "table.csv"
    json_path = tmp_path / "table.json"
    md_path = tmp_path / "table.md"

    write_csv(rows, csv_path)
    write_json(rows, json_path)
    md_path.write_text(render_markdown(rows), encoding="utf-8")

    # CSV header is fixed; the row count matches build_rows.
    csv_lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert csv_lines[0] == "axis,encoder,metric,n,mean,std,ci_lo,ci_hi"
    assert len(csv_lines) - 1 == len(rows)

    # JSON payload mirrors the row dataclasses minus the raw samples.
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(payload["rows"]) == len(rows)
    for emitted in payload["rows"]:
        assert "samples" not in emitted

    # Markdown groups by axis; both axes get their own section.
    md = md_path.read_text(encoding="utf-8")
    assert "### Axis: `stance`" in md
    assert "### Axis: `factor`" in md
    # Each metric × encoder pair shows up exactly once in the relevant section.
    assert md.count("encoder_a") == 2 * 3  # 2 axes × 3 metrics
    assert md.count("encoder_b") == 2 * 3


def test_empty_aggregate_emits_no_rows() -> None:
    assert build_rows({}) == []
    assert build_rows({"by_encoder": {}}) == []


def test_axis_block_without_per_seed_is_skipped() -> None:
    aggregate = {
        "by_encoder": {
            "encoder_x": {
                "per_axis": {
                    "stance": {"per_seed": {"11": {"macro_f1": 0.5}}},
                    "factor": {},  # malformed / empty axis
                }
            }
        }
    }
    rows = build_rows(aggregate)
    axes = {r.axis for r in rows}
    assert axes == {"stance"}
