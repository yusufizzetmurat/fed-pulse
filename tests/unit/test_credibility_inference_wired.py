"""Verify ``/analyze`` threads the live credibility vector into the forecaster (#339).

Before this issue the inference path built ``credibility = torch.zeros(...)``
unconditionally, so the forecaster's credibility-feature input was
always the neutral vector even when the loader had real per-axis values
to surface. This test mocks the credibility loader to return a non-zero
tensor and asserts the model forward receives it verbatim through
``_predict_next_point`` and ``build_regime_classification_card``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from app.models.config import FeatureVector  # noqa: E402
from app.services import forecaster as forecaster_module  # noqa: E402


class _FakeCredibilityModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.credibility_features = True
        self.credibility_dim = 4
        self.input_size = 6
        self.output_mode = "regression"
        self._text_path_active = False
        self.text_embedding_dim = 0
        self.head_mode = "regression"
        self.regression_head = torch.nn.Identity()
        self.last_kwargs: dict[str, torch.Tensor] = {}
        self._param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, **kwargs):  # type: ignore[override]
        self.last_kwargs = kwargs
        return torch.zeros(1, 2)


def _seq() -> list[FeatureVector]:
    return [
        FeatureVector(
            date=f"2026-03-{10 + i:02d}",
            sentiment_score=0.0,
            market_close=5000.0 + i,
            market_volatility=0.012,
        )
        for i in range(5)
    ]


def test_predict_next_point_surfaces_real_credibility_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_predict_next_point`` must consult the loader for credibility
    instead of zero-defaulting. A non-zero loader response must reach
    the model forward unchanged."""

    fake = _FakeCredibilityModel()
    monkeypatch.setattr(forecaster_module, "_model_artifact_metadata", {})

    live = torch.tensor([[0.4, -0.1, 0.0, 3.0]], dtype=torch.float32)
    monkeypatch.setattr(
        forecaster_module,
        "compute_credibility_for_inference",
        lambda event_date: live,
    )

    sequence = _seq()
    forecaster_module._predict_next_point(fake, sequence)

    received = fake.last_kwargs.get("credibility")
    assert received is not None, "credibility kwarg must be threaded into the forward"
    assert received.shape == (1, 4)
    assert torch.allclose(received, live)
    # The vector is non-zero on at least one axis -- guarantees the
    # forward did not silently fall back to torch.zeros(...).
    assert float(received.abs().sum().item()) > 0.0


def test_predict_next_point_zero_fallback_when_loader_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``None`` from the loader (missing FRED cache, missing embeddings)
    must surface as the neutral zero vector so the forward still runs."""

    fake = _FakeCredibilityModel()
    monkeypatch.setattr(forecaster_module, "_model_artifact_metadata", {})
    monkeypatch.setattr(
        forecaster_module, "compute_credibility_for_inference", lambda event_date: None
    )

    forecaster_module._predict_next_point(fake, _seq())

    received = fake.last_kwargs.get("credibility")
    assert received is not None
    assert received.shape == (1, 4)
    assert torch.count_nonzero(received) == 0
