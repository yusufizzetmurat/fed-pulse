"""Tests for the paired Wilcoxon analysis script (#497)."""

from __future__ import annotations

import math

import pytest

from scripts.run_paired_comparison_tests import (
    _holm_correct,
    _wilcoxon_signed_rank,
    run_paired_tests,
)


def _make_sweep_payload(
    arm_a_values: list[float],
    arm_b_values: list[float],
    fold_ids: list[str] | None = None,
    seeds: list[int] | None = None,
) -> dict[str, object]:
    """Build a sweep JSON shape with one arm per name. Both arms get the
    same (seed, fold) grid so pairing succeeds.
    """

    seeds = seeds or [11, 29, 47, 71, 97]
    fold_ids = fold_ids or [
        "wf_fold_1",
        "wf_fold_2",
        "wf_fold_3",
        "wf_fold_4",
        "wf_fold_5",
    ]
    expected = len(seeds) * len(fold_ids)
    assert len(arm_a_values) == expected
    assert len(arm_b_values) == expected

    def _trials(values: list[float]) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        cursor = 0
        for seed in seeds:
            fold_block = []
            for fold_id in fold_ids:
                fold_block.append(
                    {
                        "fold_id": fold_id,
                        "metrics": {
                            "regime_f1_macro": values[cursor],
                        },
                    }
                )
                cursor += 1
            out.append({"seed": seed, "folds": fold_block})
        return out

    return {
        "trials": {
            "arm_a": _trials(arm_a_values),
            "arm_b": _trials(arm_b_values),
        }
    }


def test_wilcoxon_zero_deltas_returns_p_one() -> None:
    w, p, n = _wilcoxon_signed_rank([0.0] * 5)
    assert p == 1.0
    assert n == 0


def test_wilcoxon_all_positive_deltas_normal_approx() -> None:
    # n = 25, all positive => W = 0 (min(W+, W-) = 0), p should be tiny.
    deltas = [0.01 * (i + 1) for i in range(25)]
    w, p, n = _wilcoxon_signed_rank(deltas)
    assert n == 25
    assert w == 0.0
    assert p < 0.001


def test_wilcoxon_symmetric_small_n_exact() -> None:
    # n=4, balanced signs => p should be ~1.0 (no preference for either side)
    deltas = [0.5, -0.5, 0.3, -0.3]
    w, p, n = _wilcoxon_signed_rank(deltas)
    assert n == 4
    assert p == pytest.approx(1.0, abs=0.1)


def test_holm_correction_orders_correctly() -> None:
    raw = [0.001, 0.01, 0.04, 0.06]
    corrected = _holm_correct(raw)
    # Holm: multiply smallest by 4, next by 3, etc.
    # 0.001 * 4 = 0.004
    # 0.01 * 3 = 0.03
    # 0.04 * 2 = 0.08
    # 0.06 * 1 = 0.06 (but must be >= 0.08 from running max)
    assert corrected[0] == pytest.approx(0.004)
    assert corrected[1] == pytest.approx(0.03)
    assert corrected[2] == pytest.approx(0.08)
    assert corrected[3] == pytest.approx(0.08)  # bumped by running max


def test_run_paired_tests_recovers_known_delta_direction() -> None:
    """Arm B uniformly beats Arm A by +0.01 → mean_delta == +0.01,
    Wilcoxon p should be very small.
    """

    arm_a = [0.40] * 25
    arm_b = [0.41] * 25
    payload = _make_sweep_payload(arm_a, arm_b)
    rows = run_paired_tests(payload, metric="regime_f1_macro")
    assert len(rows) == 1
    row = rows[0]
    assert row["arm_a"] == "arm_a"
    assert row["arm_b"] == "arm_b"
    assert row["n_pairs"] == 25
    assert row["mean_delta_b_minus_a"] == pytest.approx(0.01, abs=1e-9)
    # All deltas identical → Wilcoxon W = 0 (every nonzero rank is on
    # the positive side); p < 0.001 under normal approx.
    assert row["wilcoxon_p_two_sided"] < 0.001
    # Holm with one pair is identity (multiplier = 1)
    assert row["holm_corrected_p"] == pytest.approx(
        row["wilcoxon_p_two_sided"]
    )


def test_run_paired_tests_no_delta_returns_p_one() -> None:
    arm_a = [0.40] * 25
    arm_b = [0.40] * 25
    payload = _make_sweep_payload(arm_a, arm_b)
    rows = run_paired_tests(payload, metric="regime_f1_macro")
    assert rows[0]["wilcoxon_p_two_sided"] == 1.0
    assert rows[0]["n_nonzero_pairs"] == 0


def test_run_paired_tests_handles_missing_cells() -> None:
    payload = _make_sweep_payload([0.40] * 25, [0.45] * 25)
    # Drop one fold from arm_b to simulate a missing cell
    payload["trials"]["arm_b"][0]["folds"][0]["metrics"]["regime_f1_macro"] = None
    rows = run_paired_tests(payload, metric="regime_f1_macro")
    assert rows[0]["n_pairs"] == 24


def test_run_paired_tests_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="trials"):
        run_paired_tests({}, metric="regime_f1_macro")
