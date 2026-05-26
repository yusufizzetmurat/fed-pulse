"""Cover the multi-architecture forecaster factory.

Locks the dispatch surface used by both the sweep harness and the training
loop: each of the six architectures must return a runnable module that
accepts ``(batch, 20, 6)`` input and emits ``(batch, 2)`` output, and the
default ``architecture="lstm"`` must return an instance of the canonical
``ForecasterModel`` wrapper so on-disk checkpoints keep loading.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.models import FORECASTER_ARCHITECTURES  # noqa: E402
from app.models.config import FEATURE_SIZE, SEQUENCE_LENGTH, ModelConfig  # noqa: E402
from app.models.factory import build_forecaster  # noqa: E402
from app.models.lstm import ForecasterModel  # noqa: E402


def _fresh(seed: int = 11) -> None:
    torch.manual_seed(seed)


def test_registry_lists_nine_architectures() -> None:
    # #327 added ``flat_mlp`` as Arm B of the text-path A/B comparison.
    # Older callers that iterate the registry to dispatch a recurrent
    # core should filter out ``flat_mlp`` explicitly -- it has no
    # recurrent core to dispatch.
    assert set(FORECASTER_ARCHITECTURES) == {
        "lstm",
        "lstm_attn",
        "gru",
        "tcn",
        "transformer",
        "dlinear",
        "informer",
        "tft",
        "flat_mlp",
    }


def test_default_config_returns_lstm_forecaster_model() -> None:
    _fresh()
    model = build_forecaster(ModelConfig())
    assert isinstance(model, ForecasterModel)
    assert model.model_type == "lstm"


def test_factory_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError) as exc:
        build_forecaster(ModelConfig(architecture="not_a_thing"))
    message = str(exc.value).lower()
    assert "architecture" in message or "unknown" in message


@pytest.mark.parametrize("architecture", list(FORECASTER_ARCHITECTURES))
def test_each_architecture_emits_two_outputs(architecture: str) -> None:
    """Every architecture must respect the shared (batch, 20, 6) -> (batch, 2) shape."""

    _fresh()
    # ``transformer`` requires hidden_size divisible by 4 (num_heads=4); the
    # default 64 satisfies this. ``dlinear`` is pinned to SEQUENCE_LENGTH=20.
    config = ModelConfig(architecture=architecture)
    model = build_forecaster(config)
    model.eval()
    # ``flat_mlp`` is a research-only Arm B class that does not subclass
    # ForecasterResearchModel (no recurrent core). The shape contract
    # below still holds across all nine architectures, but the class
    # check only applies to the eight recurrent variants.
    if architecture != "flat_mlp":
        assert isinstance(model, ForecasterModel)
    assert model.model_type == architecture

    batch = 4
    x = torch.randn(batch, SEQUENCE_LENGTH, FEATURE_SIZE)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (batch, 2), (
        f"architecture={architecture}: expected (batch, 2); got {tuple(out.shape)}"
    )
    # Volatility (column 1) must stay non-negative; the wrapper softpluses it
    # before returning so this invariant should hold across all architectures.
    assert torch.all(out[:, 1] >= 0.0), (
        f"architecture={architecture}: volatility output went negative"
    )


def test_credibility_features_flow_through_factory() -> None:
    _fresh()
    config = ModelConfig(architecture="lstm", credibility_features=True)
    model = build_forecaster(config)
    assert isinstance(model, ForecasterModel)
    assert model.credibility_features is True

    x = torch.randn(2, SEQUENCE_LENGTH, FEATURE_SIZE)
    credibility = torch.zeros(2, model.credibility_dim, dtype=torch.float32)
    with torch.no_grad():
        out = model(x, credibility=credibility)
    assert out.shape == (2, 2)


def test_factory_accepts_plain_dict_config() -> None:
    """Callers that round-trip ModelConfig via JSON pass dicts; the factory must accept them."""

    _fresh()
    model = build_forecaster(ModelConfig(architecture="gru").to_dict())
    assert isinstance(model, ForecasterModel)
    assert model.model_type == "gru"


def test_factory_dispatches_informer() -> None:
    """architecture=\"informer\" returns a ForecasterModel whose core is the Informer encoder."""

    from app.models.informer import InformerEncoder

    _fresh()
    model = build_forecaster(ModelConfig(architecture="informer"))
    model.eval()
    assert isinstance(model, ForecasterModel)
    assert model.model_type == "informer"
    assert isinstance(model.recurrent_core, InformerEncoder)
    x = torch.randn(4, SEQUENCE_LENGTH, FEATURE_SIZE)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 2)
    assert torch.all(out[:, 1] >= 0.0)


def test_factory_dispatches_tft() -> None:
    """architecture=\"tft\" returns a ForecasterModel whose core is the TFT encoder."""

    from app.models.tft import TFTEncoder

    _fresh()
    model = build_forecaster(ModelConfig(architecture="tft"))
    model.eval()
    assert isinstance(model, ForecasterModel)
    assert model.model_type == "tft"
    assert isinstance(model.recurrent_core, TFTEncoder)
    x = torch.randn(4, SEQUENCE_LENGTH, FEATURE_SIZE)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 2)
    assert torch.all(out[:, 1] >= 0.0)
