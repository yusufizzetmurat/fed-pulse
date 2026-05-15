"""Regression tests for the credibility_features flag on ForecasterModel.

The default-off invariant is the critical one: when ``credibility_features``
is False (the default), the model must behave byte-identically to the v1
forecaster so the existing determinism test in ``tests/regression/`` and every
on-disk checkpoint keep loading.
"""

from __future__ import annotations

import torch

from app.models.config import CREDIBILITY_FEATURE_DIM, ModelConfig
from app.models.lstm import ForecasterModel


def _seed(seed: int = 11) -> None:
    torch.manual_seed(seed)


def test_credibility_off_is_default() -> None:
    config = ModelConfig()
    assert config.credibility_features is False
    model = ForecasterModel(**config.to_dict())
    assert model.credibility_features is False
    assert model.credibility_dim == 0


def test_credibility_off_model_forward_matches_v1_shape() -> None:
    _seed()
    model_off = ForecasterModel()
    x = torch.randn(2, 5, model_off.input_size)
    out = model_off(x)
    assert out.shape == (2, 2)


def test_credibility_off_state_dict_keys_unchanged_from_v1() -> None:
    """Adding the credibility_features parameter must not add new tensors when
    the feature is off — otherwise loading older checkpoints would break."""

    model_off = ForecasterModel(credibility_features=False)
    keys_off = set(model_off.state_dict().keys())

    # Compare against a model built with the legacy constructor surface (no
    # credibility_features kwarg, simulating older code paths). Same keys
    # expected.
    legacy = ForecasterModel(
        input_size=model_off.input_size,
        hidden_size=model_off.hidden_size,
        num_layers=model_off.num_layers,
        dropout=model_off.dropout,
        head_hidden_size=model_off.head_hidden_size,
        initial_decay_rate=model_off.initial_decay_rate,
    )
    assert keys_off == set(legacy.state_dict().keys())


def test_credibility_on_extends_lstm_input_size() -> None:
    model = ForecasterModel(credibility_features=True)
    assert model.credibility_features is True
    assert model.credibility_dim == CREDIBILITY_FEATURE_DIM
    assert model.lstm_input_size == model.input_size + CREDIBILITY_FEATURE_DIM


def test_credibility_on_forward_requires_credibility_tensor() -> None:
    model = ForecasterModel(credibility_features=True)
    x = torch.randn(2, 5, model.input_size)
    try:
        model(x)
    except ValueError as exc:
        assert "credibility" in str(exc).lower()
    else:
        raise AssertionError("forward must reject missing credibility tensor")


def test_credibility_on_forward_accepts_per_batch_vector() -> None:
    model = ForecasterModel(credibility_features=True)
    x = torch.randn(2, 5, model.input_size)
    credibility = torch.tensor(
        [[0.1, -0.2, 0.0, 6.0], [0.3, 0.4, -0.1, 12.0]], dtype=torch.float32
    )
    out = model(x, credibility=credibility)
    assert out.shape == (2, 2)


def test_credibility_on_rejects_wrong_dim() -> None:
    model = ForecasterModel(credibility_features=True)
    x = torch.randn(1, 5, model.input_size)
    wrong = torch.zeros((1, 3))
    try:
        model(x, credibility=wrong)
    except ValueError as exc:
        assert "shape" in str(exc).lower()
    else:
        raise AssertionError("forward must reject wrong-dim credibility tensor")


def test_modelconfig_round_trip_preserves_credibility_flag() -> None:
    config = ModelConfig(credibility_features=True)
    rebuilt = ModelConfig(**config.to_dict())
    assert rebuilt.credibility_features is True
