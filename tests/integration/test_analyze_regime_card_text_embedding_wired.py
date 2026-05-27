"""Verify the regime card threads ``text_embedding`` into the model forward (#339).

The canonical text-mounted forecaster checkpoint
(``forecaster_canonical`` revision ``de318540``) trains the
``text_adapter`` against pooled prior-N statement embeddings. Before
#339 the inference path did not pass the matching ``text_embedding``
kwarg, so ``ForecasterBase`` raised ``ValueError`` inside
``forward_multi_task``, the ``_safe_regime_classification`` try/except
swallowed it, and the regime card silently rendered ``None``.

This test stands a fake text-mounted forecaster in place of the singleton
and asserts the inference helpers wire the pooled vector + missing flag
through to ``forward_multi_task`` with the contract the training-time
loader emits (``text_embedding`` shape ``(1, text_embedding_dim)``,
``text_embedding_missing`` shape ``(1, 1)``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from app.models.config import FeatureVector  # noqa: E402
from app.services import forecaster as forecaster_module  # noqa: E402


class _FakeTextMountedModel(torch.nn.Module):
    """Minimal stand-in for ``ForecasterServingModel`` exposing the
    attributes ``build_regime_classification_card`` consults."""

    def __init__(self, *, text_embedding_dim: int = 4, credibility_dim: int = 4) -> None:
        super().__init__()
        self.text_embedding_dim = text_embedding_dim
        self.text_adapter_dim = text_embedding_dim
        self._text_path_active = True
        self.credibility_features = True
        self.credibility_dim = credibility_dim
        self.input_size = 6
        self.output_mode = "classification"
        self.head_mode = "regression"
        self.regression_head = torch.nn.Identity()
        self.vol_regime_quantiles = (-0.5, 0.5)
        # Track the kwargs the production code threaded into the forward.
        self.last_kwargs: dict[str, torch.Tensor] = {}
        self._device_param = torch.nn.Parameter(torch.zeros(1))

    def forward_multi_task(self, x, **kwargs):  # type: ignore[override]
        self.last_kwargs = kwargs
        return {"log_rv": torch.zeros(1)}

    def forward(self, x, **kwargs):  # type: ignore[override]
        self.last_kwargs = kwargs
        return torch.zeros(1, 3)


def _seq_with_pooled_text(text_embedding_dim: int) -> list[FeatureVector]:
    pooled = [0.1 * i for i in range(text_embedding_dim)]
    vectors: list[FeatureVector] = []
    for idx in range(5):
        v = FeatureVector(
            date=f"2026-03-{10 + idx:02d}",
            sentiment_score=0.5,
            market_close=5000.0 + idx,
            market_volatility=0.012,
        )
        v.text_embedding_pooled = pooled
        v.text_embedding_missing = 0.0
        vectors.append(v)
    return vectors


def test_regime_card_threads_text_embedding_into_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``build_regime_classification_card`` must wire ``text_embedding`` +
    ``text_embedding_missing`` into ``forward_multi_task`` when the
    checkpoint's text path is active. The shapes must match the contract
    the loader emits at training time (``(1, dim)`` + ``(1, 1)``)."""

    fake = _FakeTextMountedModel(text_embedding_dim=4)
    monkeypatch.setattr(forecaster_module, "_get_model", lambda: fake)
    monkeypatch.setattr(forecaster_module, "_model_artifact_metadata", {})

    sequence = _seq_with_pooled_text(text_embedding_dim=4)
    card = forecaster_module.build_regime_classification_card(sequence)

    # Card may be None on a vol_regime_quantiles miss path, but the
    # important assertion is that the forward saw the text inputs.
    assert "text_embedding" in fake.last_kwargs
    assert "text_embedding_missing" in fake.last_kwargs
    te = fake.last_kwargs["text_embedding"]
    miss = fake.last_kwargs["text_embedding_missing"]
    assert te.shape == (1, 4)
    assert miss.shape == (1, 1)
    # The loader populates the pooled vector + missing=0 on the
    # target-row bar; the inference helper round-trips both.
    assert float(miss.item()) == pytest.approx(0.0)
    assert torch.allclose(
        te.flatten(),
        torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float32),
        atol=1e-6,
    )
    # Card-renders-or-not is a separate concern (#322 surface); the
    # /analyze integration test in test_analyze_regression_canonical
    # already covers the populated-predicted_set path.
    if card is not None:
        assert "predicted_set" in card


def test_regime_card_zero_fallback_when_pooled_vector_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the last bar has no ``text_embedding_pooled``, the helper
    emits zeros + ``missing=1`` so the adapter's keep-mask zeros the slot."""

    fake = _FakeTextMountedModel(text_embedding_dim=4)
    monkeypatch.setattr(forecaster_module, "_get_model", lambda: fake)
    monkeypatch.setattr(forecaster_module, "_model_artifact_metadata", {})

    sequence = [
        FeatureVector(
            date=f"2026-03-{10 + i:02d}",
            sentiment_score=0.0,
            market_close=5000.0 + i,
            market_volatility=0.012,
        )
        for i in range(5)
    ]
    forecaster_module.build_regime_classification_card(sequence)

    te = fake.last_kwargs["text_embedding"]
    miss = fake.last_kwargs["text_embedding_missing"]
    assert te.shape == (1, 4)
    assert torch.count_nonzero(te) == 0
    assert float(miss.item()) == pytest.approx(1.0)
