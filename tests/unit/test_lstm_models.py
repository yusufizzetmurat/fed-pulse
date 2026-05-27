"""Regression guard for the legacy ``ForecasterModel`` alias.

Issue #327 added ``text_channel='per_bar'`` to the recurrent path. The
pre-#336 ``ForecasterModel`` class lived in ``app.models.lstm`` and was
the construction surface most call sites still import through the
back-compat shim (per ADR 0016). The burn-down recorded that the
legacy alias path was still rejecting the new enum value because it
had not been re-pointed at the research class that consumes per-bar
inputs. This test pins the alias contract: constructing the legacy
``ForecasterModel`` with ``text_channel='per_bar'`` must succeed and
the resulting module must accept a per-bar text tensor on its forward
path.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.models.config import RICH_FEATURE_SIZE  # noqa: E402
from app.models.lstm import ForecasterModel  # noqa: E402
from app.models.research_model import ForecasterResearchModel  # noqa: E402

SEQ_LEN = 20
TEXT_IN_DIM = 32
TEXT_ADAPTER_DIM = 16


def _kwargs() -> dict[str, object]:
    return {
        "input_size": RICH_FEATURE_SIZE,
        "hidden_size": 16,
        "num_layers": 1,
        "head_hidden_size": 16,
        "text_embedding_dim": TEXT_IN_DIM,
        "text_adapter_dim": TEXT_ADAPTER_DIM,
    }


def test_legacy_alias_accepts_per_bar_text_channel():
    """``ForecasterModel(text_channel='per_bar')`` must not raise."""

    model = ForecasterModel(text_channel="per_bar", **_kwargs())
    assert model.text_channel == "per_bar"


def test_legacy_alias_resolves_to_research_class():
    """The shim re-exports the research class so per-bar plumbing is live."""

    assert ForecasterModel is ForecasterResearchModel


def test_legacy_alias_per_bar_forward_shape():
    """Per-bar forward through the legacy alias produces the canonical (B, 2) head."""

    model = ForecasterModel(text_channel="per_bar", **_kwargs())
    model.eval()
    x = torch.zeros((2, SEQ_LEN, RICH_FEATURE_SIZE))
    per_bar = torch.zeros((2, SEQ_LEN, TEXT_IN_DIM))
    with torch.no_grad():
        out = model(x, text_embedding_per_bar=per_bar)
    assert out.shape == (2, 2)


def test_services_forecaster_alias_accepts_per_bar():
    """The ``app.services.forecaster.ForecasterModel`` re-export honours per_bar."""

    from app.services.forecaster import ForecasterModel as ServicesForecasterModel

    model = ServicesForecasterModel(text_channel="per_bar", **_kwargs())
    assert model.text_channel == "per_bar"
    assert isinstance(model, ForecasterResearchModel)
