"""Unit coverage for the inference contract sidecar (#341)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from app.models.factory import build_serving_forecaster
from app.models.config import ModelConfig, FEATURE_SIZE
from app.training.inference_contract import (
    SIDECAR_SCHEMA_VERSION,
    InferenceContract,
    collect_serving_forward_kwargs,
    derive_contract,
    read_sidecar,
    sidecar_path_for,
    validate_against_serving,
    write_sidecar,
)


def _toy_serving_model() -> torch.nn.Module:
    return build_serving_forecaster(
        ModelConfig(input_size=FEATURE_SIZE, architecture="lstm")
    )


def test_derive_contract_default_no_text_path() -> None:
    """A vanilla serving forecaster declares no required kwargs.

    The legacy 6-feature regression-output checkpoint has no
    credibility / text / chunk gates on; the contract is therefore
    empty on required-kwargs.
    """

    model = _toy_serving_model()
    contract = derive_contract(model, encoder_alias="bert_base")
    assert contract.required_kwargs == ()
    assert contract.encoder_alias == "bert_base"
    assert contract.schema_version == SIDECAR_SCHEMA_VERSION
    assert contract.model_class == "ForecasterServingModel"


def test_derive_contract_text_path_required() -> None:
    """When the text path is mounted, text_embedding becomes required."""

    model = _toy_serving_model()
    model._text_path_active = True  # type: ignore[attr-defined]
    model.text_embedding_dim = 768  # type: ignore[attr-defined]
    contract = derive_contract(model)
    assert "text_embedding" in contract.required_kwargs
    assert "text_embedding_missing" in contract.required_kwargs


def test_sidecar_roundtrip(tmp_path: Path) -> None:
    """Write -> read -> structured equality."""

    model = _toy_serving_model()
    model.credibility_features = True  # type: ignore[attr-defined]
    ckpt_path = tmp_path / "toy.pt"
    ckpt_path.write_bytes(b"\x00")  # the file just needs to exist
    contract = derive_contract(
        model,
        encoder_alias="finbert_fed_adjacent",
        inference_features=("text_embedding",),
    )
    written = write_sidecar(contract, ckpt_path)
    assert written == sidecar_path_for(ckpt_path)
    assert written.exists()
    raw = json.loads(written.read_text(encoding="utf-8"))
    assert raw["schema_version"] == SIDECAR_SCHEMA_VERSION
    assert raw["model_class"] == "ForecasterServingModel"
    loaded = read_sidecar(ckpt_path)
    assert loaded == contract


def test_sidecar_absent_returns_none(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "no_sidecar.pt"
    assert read_sidecar(ckpt_path) is None


def test_sidecar_malformed_degrades_to_none(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "bad.pt"
    sidecar_path_for(ckpt_path).write_text("{not json", encoding="utf-8")
    assert read_sidecar(ckpt_path) is None


def test_validate_against_serving_ok_when_subset() -> None:
    contract = InferenceContract(
        schema_version=SIDECAR_SCHEMA_VERSION,
        model_class="ForecasterServingModel",
        required_kwargs=("credibility", "text_embedding"),
    )
    out = validate_against_serving(contract)
    assert out.ok is True
    assert out.status == "ok"


def test_validate_against_serving_rejects_unknown_kwarg() -> None:
    contract = InferenceContract(
        schema_version=SIDECAR_SCHEMA_VERSION,
        model_class="ForecasterServingModel",
        required_kwargs=("not_a_real_kwarg",),
    )
    out = validate_against_serving(contract)
    assert out.ok is False
    assert out.status == "serving_signature_missing_kwargs"
    assert "not_a_real_kwarg" in out.missing_kwargs


def test_validate_against_registry_features_mismatch() -> None:
    """When the registry declares fewer features than the contract,
    the loader refuses to bind."""

    contract = InferenceContract(
        schema_version=SIDECAR_SCHEMA_VERSION,
        model_class="ForecasterServingModel",
        required_kwargs=(),
        inference_features=("text_embedding", "credibility"),
    )
    out = validate_against_serving(
        contract,
        registry_inference_features=("text_embedding",),
    )
    assert out.ok is False
    assert out.status == "registry_inference_features_mismatch"
    assert "credibility" in out.extra_kwargs


def test_collect_serving_forward_kwargs_includes_expected() -> None:
    from app.models.serving_model import ForecasterServingModel

    kwargs = collect_serving_forward_kwargs(ForecasterServingModel)
    for expected in (
        "text_embedding",
        "text_embedding_missing",
        "credibility",
        "chunks",
        "elapsed_days",
    ):
        assert expected in kwargs


def test_save_checkpoint_writes_sidecar(tmp_path: Path) -> None:
    """The checkpoint save path emits the sidecar next to the .pt file."""

    from app.evaluation.metrics import TrainingRunSummary
    from app.training.checkpoint import _save_model_checkpoint

    model = _toy_serving_model()
    ckpt_path = tmp_path / "forecaster.pt"
    summary = TrainingRunSummary(
        model_config=ModelConfig(input_size=FEATURE_SIZE, architecture="lstm"),
        device="cpu",
        epochs_requested=1,
        epochs_completed=1,
        batch_size=1,
        learning_rate=1e-3,
        validation_split=0.2,
        early_stopping_patience=1,
        sequence_groups=1,
        total_windows=1,
        train_windows=1,
        validation_windows=0,
        checkpoint_path=str(ckpt_path),
        checkpoint_saved=True,
        best_epoch=1,
        metrics=None,
    )
    _save_model_checkpoint(
        model,
        ckpt_path,
        summary,
        encoder_alias="bert_base",
        inference_features=("text_embedding",),
    )
    assert ckpt_path.exists()
    sidecar = sidecar_path_for(ckpt_path)
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["encoder_alias"] == "bert_base"
    assert payload["inference_features"] == ["text_embedding"]


def test_loader_refuses_bind_on_contract_mismatch(tmp_path: Path, monkeypatch) -> None:
    """``_get_model`` raises when the sidecar declares a kwarg the
    serving signature does not accept."""

    from app.evaluation.metrics import TrainingRunSummary
    from app.training.checkpoint import _save_model_checkpoint
    from app.services import forecaster as forecaster_service

    model = _toy_serving_model()
    ckpt_path = tmp_path / "forecaster.pt"
    summary = TrainingRunSummary(
        model_config=ModelConfig(input_size=FEATURE_SIZE, architecture="lstm"),
        device="cpu",
        epochs_requested=1,
        epochs_completed=1,
        batch_size=1,
        learning_rate=1e-3,
        validation_split=0.2,
        early_stopping_patience=1,
        sequence_groups=1,
        total_windows=1,
        train_windows=1,
        validation_windows=0,
        checkpoint_path=str(ckpt_path),
        checkpoint_saved=True,
        best_epoch=1,
        metrics=None,
    )
    _save_model_checkpoint(model, ckpt_path, summary)
    # Corrupt the sidecar to declare a kwarg that does not exist on
    # the serving signature; the loader must refuse to bind.
    sidecar = sidecar_path_for(ckpt_path)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["required_kwargs"] = ["nonexistent_kwarg"]
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(forecaster_service, "BEST_MODEL_PATH", ckpt_path)
    monkeypatch.setattr(forecaster_service, "_model", None)
    monkeypatch.setattr(forecaster_service, "_model_artifact_metadata", None)

    with pytest.raises(RuntimeError) as excinfo:
        forecaster_service._get_model()
    assert "inference contract" in str(excinfo.value).lower()

    status = forecaster_service.get_serving_contract_status()
    assert status["status"] == "serving_signature_missing_kwargs"
    assert "nonexistent_kwarg" in status["missing_kwargs"]
