"""Unit tests for the LSTM continuous-time baseline appendix (#151).

These tests cover the pure-math helpers + the no-look-ahead contract on
the reference baselines. The GPU-bound checkpoint-inference path is
tested at integration time; until the GPU sweep lands,
``run_baseline_appendix`` emits an empty result and we assert that
behaviour explicitly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from app.evaluation.lstm_baseline_appendix import (
    compute_directional_cell,
    compute_mape_cell,
    compute_rmse_cell,
    directional_accuracy,
    mape,
    random_walk_close,
    rmse,
    rolling_mean_volatility,
    run_baseline_appendix,
)


# ---------------------------------------------------------------------------
# Pure metric helpers
# ---------------------------------------------------------------------------


def test_rmse_matches_textbook_definition() -> None:
    # rmse([1, 2, 3], [1, 2, 3]) = 0
    assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    # rmse([1, 2], [2, 4]) = sqrt(((1-2)^2 + (2-4)^2) / 2) = sqrt(2.5)
    assert rmse([1.0, 2.0], [2.0, 4.0]) == pytest.approx(math.sqrt(2.5))


def test_rmse_skips_non_finite_rows() -> None:
    out = rmse([1.0, float("nan"), 3.0], [1.0, 99.0, 3.0])
    # NaN row skipped -> 2 finite, both exact -> RMSE 0.
    assert out == 0.0
    assert math.isnan(rmse([], []))


def test_rmse_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match=r"predictions \(2\) and targets \(3\)"):
        rmse([1.0, 2.0], [1.0, 2.0, 3.0])


def test_mape_skips_zero_targets() -> None:
    # Zero target rows are skipped (division by zero); the test asserts
    # the helper does not silently inject +inf or NaN.
    out = mape([10.0, 5.0, 4.0], [10.0, 0.0, 4.0])
    # Two usable rows (10 vs 10 -> 0, 4 vs 4 -> 0) -> mean 0.
    assert out == 0.0


def test_directional_accuracy_counts_matching_signs() -> None:
    # Three forward rows: pred up, target up -> hit; pred up, target down -> miss;
    # pred down, target down -> hit. 2/3.
    out = directional_accuracy(
        predictions=[1.5, 1.5, 0.5],
        targets=[1.5, 0.5, 0.5],
        previous=[1.0, 1.0, 1.0],
    )
    assert out == pytest.approx(2.0 / 3.0)


def test_directional_accuracy_returns_nan_on_empty() -> None:
    assert math.isnan(directional_accuracy([], [], []))


# ---------------------------------------------------------------------------
# Reference baselines: no-look-ahead contract
# ---------------------------------------------------------------------------


def test_random_walk_close_returns_input_directly() -> None:
    # rwc[t] = prev_closes[t] -> output is the input by construction.
    out = random_walk_close([1.0, 2.0, 3.0, 4.0])
    assert out == [1.0, 2.0, 3.0, 4.0]


def test_rolling_mean_volatility_uses_strict_prior_window() -> None:
    """The mean at index ``i`` must depend on indices strictly < i.

    Concretely: if we set realised_vol[i] to a wildly different value
    *after* computing the rolling mean, the prediction at index ``i``
    must be unchanged. That is the no-look-ahead contract.
    """

    history = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    pristine = rolling_mean_volatility(history, window=3)

    # Perturb index 3 by an order of magnitude.
    perturbed_history = list(history)
    perturbed_history[3] = 999.0
    perturbed = rolling_mean_volatility(perturbed_history, window=3)

    # The predictions at indices < 3 must be byte-identical because they
    # depend only on strictly-earlier rows. At index 3 the prediction
    # uses indices 0..2 (strictly before 3) and must still be unchanged.
    # Only indices >= 4 see the perturbation (index 4's window now
    # includes the perturbed index 3).
    for i in range(4):
        if math.isnan(pristine[i]):
            assert math.isnan(perturbed[i])
        else:
            assert pristine[i] == perturbed[i]


def test_rolling_mean_volatility_first_row_is_nan() -> None:
    # No history before index 0 -> NaN.
    out = rolling_mean_volatility([0.5, 0.5, 0.5], window=2)
    assert math.isnan(out[0])
    # Index 1 sees [0.5] -> mean 0.5; index 2 sees [0.5, 0.5] -> 0.5.
    assert out[1] == 0.5
    assert out[2] == 0.5


def test_rolling_mean_volatility_rejects_zero_window() -> None:
    with pytest.raises(ValueError, match="window must be >= 1"):
        rolling_mean_volatility([0.1, 0.2], window=0)


# ---------------------------------------------------------------------------
# Cell helpers — bootstrap CIs sit in expected ranges.
# ---------------------------------------------------------------------------


def test_compute_rmse_cell_emits_finite_ci_bounds() -> None:
    # 60 paired rows with small noise -> RMSE in a stable range; CI
    # bounds should bracket the point estimate.
    rng = [0.001 * i for i in range(60)]
    preds = [5000.0 + r for r in rng]
    targets = [5000.5 + r for r in rng]
    cell = compute_rmse_cell(
        asset="^GSPC",
        horizon="3d",
        model="lstm",
        predictions=preds,
        targets=targets,
        n_bootstrap=200,
        block_size=10,
        seed=11,
    )
    assert cell.n == 60
    assert cell.point == pytest.approx(0.5, abs=0.05)
    assert cell.ci_low <= cell.point <= cell.ci_high


def test_compute_mape_cell_skips_zero_targets() -> None:
    preds = [100.0, 105.0, 110.0]
    targets = [100.0, 0.0, 110.0]
    cell = compute_mape_cell(
        asset="^GSPC",
        horizon="3d",
        model="lstm",
        predictions=preds,
        targets=targets,
        n_bootstrap=200,
        block_size=2,
        seed=11,
    )
    # 1 of 3 rows skipped (zero target) -> n == 2; both surviving rows
    # are exact matches -> point 0.
    assert cell.n == 2
    assert cell.point == 0.0


def test_compute_directional_cell_handles_perfect_hit_rate() -> None:
    preds = [101.0, 99.0, 102.0, 98.0]
    targets = [102.0, 98.5, 103.0, 97.0]
    previous = [100.0, 100.0, 100.0, 100.0]
    cell = compute_directional_cell(
        asset="^GSPC",
        horizon="1d",
        model="lstm",
        predictions=preds,
        targets=targets,
        previous=previous,
        n_bootstrap=200,
        block_size=2,
        seed=11,
    )
    assert cell.n == 4
    assert cell.point == 1.0


def test_compute_directional_cell_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="must have equal length"):
        compute_directional_cell(
            asset="x", horizon="1d", model="lstm",
            predictions=[1.0, 2.0],
            targets=[1.0, 2.0, 3.0],
            previous=[1.0, 2.0],
        )


# ---------------------------------------------------------------------------
# CLI smoke: empty result + JSON write when no checkpoint is on disk.
# ---------------------------------------------------------------------------


def test_run_baseline_appendix_writes_empty_when_checkpoint_missing(tmp_path: Path) -> None:
    output = tmp_path / "out"
    result = run_baseline_appendix(
        training_package_id="tp-smoke",
        output_dir=output,
        checkpoint_path=tmp_path / "no-such-checkpoint.pt",
    )
    assert result.cells == []
    payload_path = output / "lstm_baseline_appendix.json"
    assert payload_path.exists()
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert payload["training_package_id"] == "tp-smoke"
    assert payload["cells"] == []
    assert payload["note"]  # smoke message present.
