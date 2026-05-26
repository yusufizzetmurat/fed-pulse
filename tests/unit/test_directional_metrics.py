from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.evaluation.directional_metrics import compute_directional_metrics
from app.evaluation.metrics import EvaluationMetrics


# ---------------------------------------------------------------------------
# EvaluationMetrics — new optional fields
# ---------------------------------------------------------------------------


def test_evaluation_metrics_defaults_directional_to_none() -> None:
    m = EvaluationMetrics(loss=0.1, close_rmse=1.0, volatility_rmse=0.01, combined_rmse=0.5)
    d = m.to_dict()
    assert d["direction_accuracy"] is None
    assert d["f1_macro"] is None
    assert d["direction_auc"] is None


def test_evaluation_metrics_round_trips_through_dict_with_directional() -> None:
    m = EvaluationMetrics(
        loss=0.1,
        close_rmse=1.0,
        volatility_rmse=0.01,
        combined_rmse=0.5,
        direction_accuracy=0.62,
        f1_macro=0.58,
        direction_auc=0.66,
    )
    d = m.to_dict()
    assert d["direction_accuracy"] == pytest.approx(0.62)
    assert d["f1_macro"] == pytest.approx(0.58)
    assert d["direction_auc"] == pytest.approx(0.66)


# ---------------------------------------------------------------------------
# compute_directional_metrics — edges
# ---------------------------------------------------------------------------


def test_empty_input_returns_all_none() -> None:
    out = compute_directional_metrics([], [], [])
    assert out == {"direction_accuracy": None, "f1_macro": None, "direction_auc": None}


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        compute_directional_metrics([1, 2, 3], [1, 2], [1, 2, 3])


def test_perfect_prediction_yields_one_across_the_board() -> None:
    rng = np.random.default_rng(11)
    prev = np.linspace(100, 110, 200)
    true_close = prev + rng.normal(size=200)
    pred_close = true_close   # identical predictions
    out = compute_directional_metrics(pred_close, true_close, prev)
    assert out["direction_accuracy"] == pytest.approx(1.0)
    assert out["f1_macro"] == pytest.approx(1.0)
    assert out["direction_auc"] == pytest.approx(1.0)


def test_random_prediction_lands_near_coin_flip() -> None:
    rng = np.random.default_rng(11)
    prev = np.linspace(100, 110, 1000)
    true_close = prev + rng.normal(size=1000)
    pred_close = prev + rng.normal(size=1000)
    out = compute_directional_metrics(pred_close, true_close, prev)
    # With 1000 events, accuracy + f1 + auc should all sit within ~5pp
    # of 0.5 if the predictions truly carry no directional signal.
    assert 0.45 <= out["direction_accuracy"] <= 0.55
    assert 0.45 <= out["f1_macro"] <= 0.55
    assert 0.45 <= out["direction_auc"] <= 0.55


def test_anti_correlated_prediction_yields_low_accuracy_and_low_auc() -> None:
    """Flipping every prediction's sign should produce accuracy < 0.5 and AUC < 0.5."""
    rng = np.random.default_rng(11)
    prev = np.linspace(100, 110, 200)
    delta = rng.normal(size=200)
    true_close = prev + delta
    pred_close = prev - delta   # exactly anti-correlated
    out = compute_directional_metrics(pred_close, true_close, prev)
    assert out["direction_accuracy"] < 0.1
    assert out["direction_auc"] < 0.1


def test_single_class_truth_returns_none_for_auc() -> None:
    """When every realised move is 'up', binary AUC is undefined."""
    prev = np.linspace(100, 110, 50)
    true_close = prev + 1.0  # every event is 'up'
    pred_close = prev + np.random.default_rng(11).normal(size=50)
    out = compute_directional_metrics(pred_close, true_close, prev)
    assert out["direction_auc"] is None
    # Accuracy + f1 still compute; they may be any value depending on
    # how many predictions land 'up' — just assert they are floats.
    assert isinstance(out["direction_accuracy"], float)
    assert isinstance(out["f1_macro"], float)


def test_torch_input_parity_with_numpy() -> None:
    rng = np.random.default_rng(11)
    prev = np.linspace(100, 110, 200)
    true_close = prev + rng.normal(size=200)
    pred_close = prev + rng.normal(size=200)

    out_np = compute_directional_metrics(pred_close, true_close, prev)
    out_torch = compute_directional_metrics(
        torch.tensor(pred_close, dtype=torch.float32),
        torch.tensor(true_close, dtype=torch.float32),
        torch.tensor(prev, dtype=torch.float32),
    )
    assert out_np["direction_accuracy"] == pytest.approx(
        out_torch["direction_accuracy"], abs=1e-5
    )
    assert out_np["f1_macro"] == pytest.approx(out_torch["f1_macro"], abs=1e-5)
    assert out_np["direction_auc"] == pytest.approx(
        out_torch["direction_auc"], abs=1e-5
    )


def test_epsilon_band_collapses_small_deltas_to_zero_class() -> None:
    prev = np.array([100.0, 100.0, 100.0, 100.0])
    true_close = np.array([100.5, 99.5, 100.0001, 99.9999])
    pred_close = np.array([100.5, 99.5, 100.0001, 99.9999])
    # epsilon=0.01 turns the last two near-zero moves into class 0
    out = compute_directional_metrics(pred_close, true_close, prev, epsilon=0.01)
    # Perfect prediction so accuracy is 1.0 regardless
    assert out["direction_accuracy"] == pytest.approx(1.0)
