"""Lock the encoder_lora flag round-trip through the model factory.

Background: ``build_forecaster`` pops ``encoder_lora`` from kwargs
because the recurrent ``ForecasterModel`` does not consume it. Before
this contract was added, the popped flag was discarded — so
``ModelConfig.from_model(trained_model)`` always read ``False`` back
and every Round 5 LoRA cell's persisted summary lied about whether
LoRA had actually been active. These tests pin the factory to stamp
the value onto the module as a plain attribute, and pin
``from_model`` to read it back faithfully.
"""

from __future__ import annotations

import pytest


def test_factory_stamps_encoder_lora_onto_built_model() -> None:
    from app.models.config import ModelConfig
    from app.models.factory import build_forecaster

    cfg_on = ModelConfig(architecture="gru", encoder_lora=True)
    model = build_forecaster(cfg_on)
    assert getattr(model, "encoder_lora", False) is True


def test_factory_default_is_false() -> None:
    from app.models.config import ModelConfig
    from app.models.factory import build_forecaster

    cfg_off = ModelConfig(architecture="gru")  # default
    model = build_forecaster(cfg_off)
    assert getattr(model, "encoder_lora", False) is False


@pytest.mark.parametrize("flag", [True, False])
def test_model_config_round_trips_encoder_lora_via_from_model(flag: bool) -> None:
    """Every persisted run summary reads its model_config back via
    ``ModelConfig.from_model(trained_model)``; the flag the factory was
    asked to enable must survive that round-trip."""

    from app.models.config import ModelConfig
    from app.models.factory import build_forecaster

    cfg_in = ModelConfig(architecture="gru", encoder_lora=flag)
    model = build_forecaster(cfg_in)
    cfg_out = ModelConfig.from_model(model)
    assert cfg_out.encoder_lora is flag
