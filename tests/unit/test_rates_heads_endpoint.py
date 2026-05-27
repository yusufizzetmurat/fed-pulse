"""Endpoint test for POST /analyze/market (#293).

Verifies the response shape carries one :class:`RatesReactionCard`
per mounted rates head + a vol-regime card. Uses a stub forecaster
singleton so the test does not depend on a trained checkpoint being
on disk.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")

import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.main as main_mod  # noqa: E402
import app.services.forecaster as forecaster_service  # noqa: E402
from app.models.config import FeatureVector  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(main_mod.app)


def _stub_market_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock yfinance + sentiment so the endpoint stays deterministic."""

    def fake_market_history(*, target_date: str, symbol: str, history_length: int):
        return [
            {
                "date": (
                    _dt.date.fromisoformat(target_date)
                    - _dt.timedelta(days=history_length - i)
                ).isoformat(),
                "close": 100.0 + float(i),
                "volatility_5d": 0.01 + 0.0001 * i,
            }
            for i in range(history_length)
        ]

    monkeypatch.setattr(main_mod, "fetch_market_history", fake_market_history)
    monkeypatch.setattr(
        main_mod,
        "analyze_text",
        lambda _text: {"label": "neutral", "score": 0.0, "raw": []},
    )


def test_analyze_market_returns_empty_when_no_panel(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the panel builder returns None, the response stays valid."""

    _stub_market_history(monkeypatch)
    monkeypatch.setattr(
        main_mod, "build_market_reaction_panel", lambda _vectors: None
    )
    response = client.post(
        "/analyze/market",
        json={
            "text": "Inflation remains elevated.",
            "date": "2024-12-18",
            "symbol": "^GSPC",
            "horizon": "5d",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["rates"] == []
    assert payload["vol_regime"] is None


def test_analyze_market_renders_rates_cards(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A populated panel must serialise into one card per rates head."""

    _stub_market_history(monkeypatch)
    fake_payload: dict[str, Any] = {
        "rates": [
            {
                "head": "2y",
                "point_bps": 4.5,
                "lower_bps": 1.0,
                "upper_bps": 8.0,
                "coverage": 0.8,
                "directional_bucket": "tightening",
                "bucket_probabilities": {
                    "easing": 0.1,
                    "neutral": 0.3,
                    "tightening": 0.6,
                },
            },
            {
                "head": "5y",
                "point_bps": -2.0,
                "lower_bps": -6.0,
                "upper_bps": 2.0,
                "coverage": 0.8,
                "directional_bucket": "neutral",
                "bucket_probabilities": {
                    "easing": 0.2,
                    "neutral": 0.55,
                    "tightening": 0.25,
                },
            },
            {
                "head": "terminal",
                "point_bps": 10.0,
                "lower_bps": 5.0,
                "upper_bps": 15.0,
                "coverage": 0.8,
                "directional_bucket": "tightening",
                "bucket_probabilities": {
                    "easing": 0.05,
                    "neutral": 0.1,
                    "tightening": 0.85,
                },
            },
        ],
        "vol_regime": {
            "log_rv_point": -3.0,
            "log_rv_lower": None,
            "log_rv_upper": None,
            "regime_label": "high",
            "regime_probabilities": {"calm": 0.1, "normal": 0.3, "high": 0.6},
            "predicted_set": ["high"],
            "coverage": 0.8,
        },
        "encoder_alias": "finbert_fomc",
        "checkpoint_path": "/tmp/forecaster_best.pt",
    }
    monkeypatch.setattr(
        main_mod, "build_market_reaction_panel", lambda _vectors: fake_payload
    )
    response = client.post(
        "/analyze/market",
        json={
            "text": "Inflation remains elevated.",
            "date": "2024-12-18",
            "symbol": "^GSPC",
            "horizon": "5d",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    rates = payload["rates"]
    assert len(rates) == 3
    by_head = {row["head"]: row for row in rates}
    assert set(by_head.keys()) == {"2y", "5y", "terminal"}
    assert by_head["2y"]["directional_bucket"] == "tightening"
    assert by_head["2y"]["point_bps"] == pytest.approx(4.5)
    assert by_head["2y"]["lower_bps"] == pytest.approx(1.0)
    assert by_head["2y"]["upper_bps"] == pytest.approx(8.0)
    assert sum(by_head["2y"]["bucket_probabilities"].values()) == pytest.approx(
        1.0, abs=1e-6
    )
    vol = payload["vol_regime"]
    assert vol["regime_label"] == "high"
    assert "calm" in vol["regime_probabilities"]
    assert payload["encoder_alias"] == "finbert_fomc"


def test_analyze_market_handles_builder_exception(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_market_history(monkeypatch)

    def _raises(_vectors):
        raise RuntimeError("boom")

    monkeypatch.setattr(main_mod, "build_market_reaction_panel", _raises)
    response = client.post(
        "/analyze/market",
        json={
            "text": "Inflation remains elevated.",
            "date": "2024-12-18",
            "symbol": "^GSPC",
            "horizon": "5d",
        },
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "Market reaction panel unavailable"


def test_build_market_reaction_panel_returns_none_on_regression_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A regression-mode model has no rates / vol-regime cards to emit.

    #341 promoted the previous bare-None to a structured payload
    carrying ``status="not_classification_mode"`` so the operator can
    grep the response for the soft-degrade branch; the /analyze
    route handler collapses this to an empty MarketReactionPanel."""

    class _DummyModel:
        output_mode = "regression"
        rates_heads_active = ()

        def parameters(self):  # pragma: no cover -- not invoked on regression path
            return iter([torch.zeros(1)])

    monkeypatch.setattr(forecaster_service, "_get_model", lambda: _DummyModel())
    out = forecaster_service.build_market_reaction_panel(
        [
            FeatureVector(
                date="2024-12-18",
                sentiment_score=0.0,
                market_close=100.0,
                market_volatility=0.01,
            )
        ]
    )
    assert isinstance(out, dict)
    assert out["status"] == "not_classification_mode"


# ---------------------------------------------------------------------------
# #317 fix-up tests (#22 + #23)


def test_market_reaction_panel_no_aux_classifier(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the checkpoint has no aux classifier, directional_bucket is None (#317 finding #22).

    Mirrors the #10 fix: a missing aux classifier surfaces as
    ``directional_bucket=None`` / ``bucket_probabilities=None`` rather
    than fabricating a confident 'easing' on uniform probabilities.
    """

    _stub_market_history(monkeypatch)
    fake_payload: dict[str, Any] = {
        "rates": [
            {
                "head": "2y",
                "point_bps": 4.5,
                "lower_bps": None,
                "upper_bps": None,
                "coverage": None,
                "directional_bucket": None,
                "bucket_probabilities": None,
                "predicted_set": None,
            }
        ],
        "vol_regime": None,
        "encoder_alias": None,
        "checkpoint_path": "/tmp/forecaster_best.pt",
    }
    monkeypatch.setattr(
        main_mod, "build_market_reaction_panel", lambda _vectors: fake_payload
    )
    response = client.post(
        "/analyze/market",
        json={
            "text": "Hello world.",
            "date": "2024-12-18",
            "symbol": "^GSPC",
            "horizon": "5d",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["rates"][0]["directional_bucket"] is None
    assert payload["rates"][0]["bucket_probabilities"] is None


def test_market_reaction_panel_regression_only_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A regression-only checkpoint with rates heads emits rates cards (#317 finding #23 / #11)."""

    class _DummyModel:
        output_mode = "regression"
        rates_heads_active = ("2y",)
        _text_path_active = False
        use_chunk_attention = False
        use_llm_embeddings = False
        credibility_features = False

        def parameters(self):
            return iter([torch.zeros(1)])

        def forward_multi_task(self, _x, **_kwargs):  # noqa: D401
            return {"rates_2y_bps": torch.tensor([0.5])}

    monkeypatch.setattr(forecaster_service, "_get_model", lambda: _DummyModel())
    monkeypatch.setattr(
        forecaster_service, "build_lookback_sequence", lambda seq: seq
    )
    monkeypatch.setattr(
        forecaster_service,
        "_build_inference_tensor",
        lambda _w, _m, _d: torch.zeros((1, 5, 6)),
    )
    monkeypatch.setattr(
        forecaster_service, "_conformal_manifest_for", lambda _p: None
    )
    monkeypatch.setattr(
        forecaster_service,
        "_model_artifact_metadata",
        {"rates_scalers": {"2y": {"mean": 0.0, "std": 1.0}}},
    )
    out = forecaster_service.build_market_reaction_panel(
        [
            FeatureVector(
                date="2024-12-18",
                sentiment_score=0.0,
                market_close=100.0,
                market_volatility=0.01,
            )
        ]
    )
    assert out is not None
    assert len(out["rates"]) == 1
    assert out["rates"][0]["head"] == "2y"
    # No aux classifier mounted -- directional fields are None.
    assert out["rates"][0]["directional_bucket"] is None
    # Regression-only mode -- no vol_regime card.
    assert out["vol_regime"] is None
