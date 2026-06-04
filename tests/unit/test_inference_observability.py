"""Tests for the inference observability surfaces (#342).

Three surfaces:

1. ``/settings/checkpoints`` carries ``required_kwargs`` +
   ``supplied_at_inference`` per forecaster checkpoint, sourced from the
   ``<stem>.inference_contract.json`` sidecar + the live serving forward
   signature.
2. ``_bootstrap_cold_start`` invokes the canonical ``_get_model`` loader
   after writing the checkpoint, so a freshly written sidecar that
   declares an unknown kwarg trips a ``RuntimeError`` at boot rather
   than silently binding via ``_set_singleton_after_train``.
3. ``forecast_quantitative_series`` emits one structured INFO log line
   per request listing the kwargs the serving forward was called with.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")

import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.main as main_mod  # noqa: E402
import app.services.forecaster as forecaster_service  # noqa: E402
from app.models.config import FEATURE_SIZE, ModelConfig  # noqa: E402
from app.models.factory import build_serving_forecaster  # noqa: E402
from app.training.inference_contract import (  # noqa: E402
    SIDECAR_SCHEMA_VERSION,
    InferenceContract,
    sidecar_path_for,
    write_sidecar,
)


def _toy_serving_model():
    return build_serving_forecaster(
        ModelConfig(input_size=FEATURE_SIZE, architecture="lstm")
    )


def _write_toy_checkpoint(path: Path) -> None:
    model = _toy_serving_model()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_size": FEATURE_SIZE,
                "architecture": "lstm",
            },
        },
        path,
    )


def test_settings_checkpoints_surfaces_required_kwargs(tmp_path, monkeypatch):
    """The endpoint flags a forecaster checkpoint whose sidecar declares
    a kwarg the serving forward does not accept."""

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    forecaster_ckpt = models_dir / "forecaster_best.pt"
    _write_toy_checkpoint(forecaster_ckpt)
    # Synthetic mismatch: declare a kwarg the serving signature does
    # not accept. The endpoint must mark the kwarg as not supplied.
    contract = InferenceContract(
        schema_version=SIDECAR_SCHEMA_VERSION,
        model_class="ForecasterServingModel",
        required_kwargs=("text_embedding", "unknown_kwarg"),
    )
    write_sidecar(contract, forecaster_ckpt)

    # Redirect the endpoint at the temp models dir + checkpoint. The
    # endpoint resolves MODELS_DIR via a function-local import from
    # app.models.config, so patch the canonical home.
    import app.models.config as model_config_mod

    monkeypatch.setattr(model_config_mod, "MODELS_DIR", models_dir)
    monkeypatch.setattr(
        forecaster_service, "BEST_MODEL_PATH", forecaster_ckpt
    )
    monkeypatch.setattr(forecaster_service, "_model", None)
    monkeypatch.setattr(forecaster_service, "_model_artifact_metadata", None)

    client = TestClient(main_mod.app)
    resp = client.get("/settings/checkpoints")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    rows = [r for r in payload["checkpoints"] if r["filename"] == forecaster_ckpt.name]
    assert rows, payload
    row = rows[0]
    assert row["inference_contract_status"] == "present"
    assert row["required_kwargs"] == ["text_embedding", "unknown_kwarg"]
    # ``text_embedding`` IS in the serving forward signature; the
    # synthetic unknown kwarg is NOT.
    assert row["supplied_at_inference"]["text_embedding"] is True
    assert row["supplied_at_inference"]["unknown_kwarg"] is False


def test_settings_checkpoints_marks_legacy_absent_sidecar(tmp_path, monkeypatch):
    """A forecaster checkpoint with no sidecar surfaces as ``sidecar_absent``."""

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    forecaster_ckpt = models_dir / "forecaster_best.pt"
    _write_toy_checkpoint(forecaster_ckpt)
    # No sidecar written -- pre-#341 legacy artefact.

    import app.models.config as model_config_mod

    monkeypatch.setattr(model_config_mod, "MODELS_DIR", models_dir)
    monkeypatch.setattr(
        forecaster_service, "BEST_MODEL_PATH", forecaster_ckpt
    )
    monkeypatch.setattr(forecaster_service, "_model", None)
    monkeypatch.setattr(forecaster_service, "_model_artifact_metadata", None)

    client = TestClient(main_mod.app)
    resp = client.get("/settings/checkpoints")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    rows = [r for r in payload["checkpoints"] if r["filename"] == forecaster_ckpt.name]
    assert rows, payload
    row = rows[0]
    assert row["inference_contract_status"] == "sidecar_absent"
    assert row["required_kwargs"] == []
    assert row["supplied_at_inference"] == {}


def test_cold_start_invokes_get_model_on_contract_mismatch(tmp_path, monkeypatch):
    """``_bootstrap_cold_start`` re-validates the freshly written
    sidecar through the same loader ``/analyze`` uses. A bootstrap that
    lands a sidecar declaring an unknown kwarg must raise loudly."""

    forecaster_ckpt = tmp_path / "forecaster_best.pt"

    def _fake_bootstrap_checkpoint(**_kwargs):
        # Pretend training wrote a checkpoint + a bad sidecar.
        _write_toy_checkpoint(forecaster_ckpt)
        contract = InferenceContract(
            schema_version=SIDECAR_SCHEMA_VERSION,
            model_class="ForecasterServingModel",
            required_kwargs=("nonexistent_kwarg",),
        )
        write_sidecar(contract, forecaster_ckpt)
        return None

    monkeypatch.setattr(main_mod, "bootstrap_checkpoint", _fake_bootstrap_checkpoint)
    monkeypatch.setattr(main_mod, "analyze_text", lambda _: {"score": 0.0})
    monkeypatch.setattr(main_mod, "fetch_market_history", lambda **_: [])
    monkeypatch.setattr(main_mod, "build_feature_vectors", lambda *_a, **_k: [])

    monkeypatch.setattr(forecaster_service, "BEST_MODEL_PATH", forecaster_ckpt)
    monkeypatch.setattr(forecaster_service, "_model", None)
    monkeypatch.setattr(forecaster_service, "_model_artifact_metadata", None)

    payload = main_mod.AnalyzeRequest(
        text="warmup",
        date="2026-05-15",
        symbol="^GSPC",
        horizon="3d",
    )

    with pytest.raises(RuntimeError) as excinfo:
        main_mod._bootstrap_cold_start(payload)
    assert "inference contract" in str(excinfo.value).lower()
    # The structured surface picks up the synthetic mismatch.
    status = forecaster_service.get_serving_contract_status()
    assert status["status"] == "serving_signature_missing_kwargs"
    assert "nonexistent_kwarg" in status["missing_kwargs"]


def test_cold_start_succeeds_when_sidecar_matches(tmp_path, monkeypatch):
    """Sanity: a bootstrap that writes a sidecar declaring kwargs the
    serving signature accepts must NOT raise — the loud-fail path only
    fires on real drift."""

    forecaster_ckpt = tmp_path / "forecaster_best.pt"

    def _fake_bootstrap_checkpoint(**_kwargs):
        _write_toy_checkpoint(forecaster_ckpt)
        contract = InferenceContract(
            schema_version=SIDECAR_SCHEMA_VERSION,
            model_class="ForecasterServingModel",
            required_kwargs=("text_embedding",),
        )
        write_sidecar(contract, forecaster_ckpt)
        return None

    monkeypatch.setattr(main_mod, "bootstrap_checkpoint", _fake_bootstrap_checkpoint)
    monkeypatch.setattr(main_mod, "analyze_text", lambda _: {"score": 0.0})
    monkeypatch.setattr(main_mod, "fetch_market_history", lambda **_: [])
    monkeypatch.setattr(main_mod, "build_feature_vectors", lambda *_a, **_k: [])

    monkeypatch.setattr(forecaster_service, "BEST_MODEL_PATH", forecaster_ckpt)
    monkeypatch.setattr(forecaster_service, "_model", None)
    monkeypatch.setattr(forecaster_service, "_model_artifact_metadata", None)

    payload = main_mod.AnalyzeRequest(
        text="warmup",
        date="2026-05-15",
        symbol="^GSPC",
        horizon="3d",
    )

    # Must not raise.
    main_mod._bootstrap_cold_start(payload)
    status = forecaster_service.get_serving_contract_status()
    assert status["status"] == "ok"


def test_analyze_emits_structured_kwarg_log_line(monkeypatch, caplog):
    """``forecast_quantitative_series`` emits one structured INFO line
    per request listing the kwargs the serving forward was called with.
    Format: ``analyze_serving_forward kwargs=<...> checkpoint=<stem>``.
    """

    model = _toy_serving_model()

    # Force the singleton + bypass the heavy forward call so the test
    # exercises the log surface in isolation.
    monkeypatch.setattr(forecaster_service, "_model", model)
    monkeypatch.setattr(
        forecaster_service,
        "_get_model",
        lambda: model,
    )
    monkeypatch.setattr(
        forecaster_service,
        "_predict_next_point",
        lambda _model, _sequence, **_: (0.0, 0.0),
    )
    # Skip the conformal / get_model_artifact_metadata machinery by
    # exercising the function with a single-step horizon.
    monkeypatch.setattr(
        forecaster_service,
        "_conformal_manifest_for",
        lambda _path: None,
    )
    monkeypatch.setattr(
        forecaster_service,
        "get_model_artifact_metadata",
        lambda **_k: {},
    )

    from app.models.config import FeatureVector

    vectors = [
        FeatureVector(
            date=f"2026-05-{day:02d}",
            sentiment_score=0.0,
            market_close=100.0,
            market_volatility=0.01,
        )
        for day in range(1, 11)
    ]

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=forecaster_service.__name__):
        forecaster_service.forecast_quantitative_series(
            vectors=vectors,
            forecast_mode="fast",
            horizon="1d",
        )

    matching = [
        r for r in caplog.records if r.getMessage().startswith("analyze_serving_forward ")
    ]
    assert len(matching) == 1, [r.getMessage() for r in caplog.records]
    msg = matching[0].getMessage()
    assert "kwargs=" in msg
    assert "checkpoint=" in msg
    assert "mode=" in msg
    # Regression-only toy model: no text path active -- kwargs list is
    # empty (``kwargs= checkpoint=... mode=regression``).
    assert " kwargs= checkpoint=" in msg, msg
    assert "mode=regression" in msg, msg


def test_analyze_log_line_lists_kwargs_for_text_path(monkeypatch, caplog):
    """When the text path is active, the log line lists the text
    embedding kwargs."""

    model = _toy_serving_model()
    # Flip the runtime gates the helper inspects.
    model._text_path_active = True  # type: ignore[attr-defined]
    model.text_embedding_dim = 8  # type: ignore[attr-defined]
    model.credibility_features = True  # type: ignore[attr-defined]

    monkeypatch.setattr(forecaster_service, "_model", model)
    monkeypatch.setattr(forecaster_service, "_get_model", lambda: model)
    monkeypatch.setattr(
        forecaster_service,
        "_predict_next_point",
        lambda _model, _sequence, **_: (0.0, 0.0),
    )
    monkeypatch.setattr(
        forecaster_service,
        "_conformal_manifest_for",
        lambda _path: None,
    )
    monkeypatch.setattr(
        forecaster_service,
        "get_model_artifact_metadata",
        lambda **_k: {},
    )

    from app.models.config import FeatureVector

    vectors = [
        FeatureVector(
            date=f"2026-05-{day:02d}",
            sentiment_score=0.0,
            market_close=100.0,
            market_volatility=0.01,
        )
        for day in range(1, 11)
    ]

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=forecaster_service.__name__):
        forecaster_service.forecast_quantitative_series(
            vectors=vectors,
            forecast_mode="fast",
            horizon="1d",
        )

    matching = [
        r for r in caplog.records if r.getMessage().startswith("analyze_serving_forward ")
    ]
    assert len(matching) == 1
    msg = matching[0].getMessage()
    # Comma-separated kwarg list -- text + credibility.
    assert "credibility" in msg
    assert "text_embedding" in msg
    assert "text_embedding_missing" in msg


def test_analyze_log_line_carries_output_mode(monkeypatch, caplog):
    """The log line must include ``mode=<output_mode>`` so the operator
    can distinguish "kwargs declared, forward invoked" (regression mode)
    from "kwargs declared, forward short-circuited" (classification mode
    -- the ``_predict_next_point`` early-return path)."""

    model = _toy_serving_model()
    # Flip the model into classification mode -- _predict_next_point
    # will short-circuit to last-bar echo and NOT call the forward.
    model.output_mode = "classification"  # type: ignore[attr-defined]

    monkeypatch.setattr(forecaster_service, "_model", model)
    monkeypatch.setattr(forecaster_service, "_get_model", lambda: model)
    monkeypatch.setattr(
        forecaster_service,
        "_conformal_manifest_for",
        lambda _path: None,
    )
    monkeypatch.setattr(
        forecaster_service,
        "get_model_artifact_metadata",
        lambda **_k: {},
    )

    from app.models.config import FeatureVector

    vectors = [
        FeatureVector(
            date=f"2026-05-{day:02d}",
            sentiment_score=0.0,
            market_close=100.0,
            market_volatility=0.01,
        )
        for day in range(1, 11)
    ]

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=forecaster_service.__name__):
        forecaster_service.forecast_quantitative_series(
            vectors=vectors,
            forecast_mode="fast",
            horizon="1d",
        )

    matching = [
        r for r in caplog.records if r.getMessage().startswith("analyze_serving_forward ")
    ]
    assert len(matching) == 1
    msg = matching[0].getMessage()
    assert "mode=classification" in msg, msg
