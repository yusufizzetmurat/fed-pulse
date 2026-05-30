"""Unit tests for the E2E event trace smoke (#505 A.1.c)."""

from __future__ import annotations

import math

from scripts.run_e2e_event_trace import (
    CORE_SLICES,
    SliceReport,
    _render_summary_md,
    _summarise_slice,
)


def test_summarise_slice_flags_all_zero_non_missing_block() -> None:
    vec = [0.0] * 50
    sl = slice(10, 14)
    report = _summarise_slice("credibility", sl, False, vec)
    assert report.dim == 4
    assert report.finite == 4
    assert report.non_zero == 0
    assert report.ok is False
    assert "silent zeros" in report.note


def test_summarise_slice_passes_on_one_non_zero() -> None:
    vec = [0.0] * 50
    vec[12] = 0.7
    sl = slice(10, 14)
    report = _summarise_slice("credibility", sl, False, vec)
    assert report.non_zero == 1
    assert report.ok is True


def test_summarise_slice_passes_zero_on_missing_flag() -> None:
    """A missing-flag scalar at 0 means 'value present' and is the
    expected good state for a rich-feature event.
    """

    vec = [0.0] * 50
    sl = slice(40, 41)
    report = _summarise_slice("llm_missing", sl, True, vec)
    assert report.ok is True
    assert report.non_zero == 0


def test_summarise_slice_flags_non_finite_on_missing() -> None:
    vec = [0.0] * 50
    vec[40] = math.nan
    sl = slice(40, 41)
    report = _summarise_slice("llm_missing", sl, True, vec)
    assert report.finite == 0
    assert report.ok is False


def test_summarise_slice_truncates_sample_to_eight() -> None:
    vec = [float(i) for i in range(50)]
    sl = slice(10, 30)
    report = _summarise_slice("big", sl, False, vec)
    assert len(report.sample_values) == 8
    assert report.sample_values == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]


def test_core_slices_table_has_market_and_missing_flag_pair() -> None:
    """Sanity check on the slices table the script walks. Lock the
    market entry first; lock that llm_feature_missing carries the
    is_missing_flag bit.
    """

    names = [name for name, _, _ in CORE_SLICES]
    assert names[0] == "market"
    flag_entries = {name: flag for name, _, flag in CORE_SLICES}
    assert flag_entries["llm_feature_missing"] is True
    assert flag_entries["retrieval_analog_missing"] is True
    assert flag_entries["llm_feature"] is False


def test_render_summary_md_emits_pass_verdict() -> None:
    report = {
        "training_package_id": "tp_x",
        "event_date": "2024-09-18",
        "fold_id": "wf_fold_3",
        "partition_found_in": "test",
        "rich_feature_size_const": 100,
        "rich_list_length": 100,
        "length_matches_const": True,
        "slices": [
            {
                "name": "market",
                "start": 0,
                "stop": 6,
                "dim": 6,
                "finite": 6,
                "non_zero": 6,
                "is_missing_flag": False,
                "sample_values": [1.0, 2.0],
                "ok": True,
                "note": "ok",
            }
        ],
        "failures": [],
        "pass": True,
    }
    md = _render_summary_md(report)
    assert "PASS" in md
    assert "All slices populated" in md
    assert "`tp_x`" in md


def test_render_summary_md_emits_fail_verdict_and_lists_failures() -> None:
    report = {
        "training_package_id": "tp_y",
        "event_date": "2024-09-18",
        "fold_id": "wf_fold_3",
        "partition_found_in": "test",
        "rich_feature_size_const": 100,
        "rich_list_length": 100,
        "length_matches_const": True,
        "slices": [
            {
                "name": "linguistic",
                "start": 10,
                "stop": 25,
                "dim": 15,
                "finite": 15,
                "non_zero": 0,
                "is_missing_flag": False,
                "sample_values": [0.0, 0.0, 0.0],
                "ok": False,
                "note": "every position is zero or non-finite (possible silent zeros)",
            }
        ],
        "failures": ["linguistic"],
        "pass": False,
    }
    md = _render_summary_md(report)
    assert "FAIL" in md
    assert "`linguistic`" in md
    assert "silent zeros" in md
