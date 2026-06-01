"""Unit coverage for the non-forecaster inference contract sidecars (#393).

Extends the #341 sidecar machinery to the multi-axis classifier and
the trajectory bundle. Each artefact gets three assertions:

* Refusal on signature mismatch (hard-fail path).
* Soft sidecar_absent legacy path (pre-#393 artefacts keep binding).
* Match-on-correct-signature (positive bind path).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from app.training.inference_contract import (
    SIDECAR_SCHEMA_VERSION,
    InferenceContract,
    derive_multi_axis_contract,
    derive_trajectory_contract,
    read_sidecar,
    sidecar_path_for,
    write_sidecar,
)


# ---------------------------------------------------------------------------
# Multi-axis classifier
# ---------------------------------------------------------------------------


def test_derive_multi_axis_contract_declares_forward_kwargs() -> None:
    """The multi-axis contract requires the two tokeniser kwargs.

    The serving call site populates ``input_ids`` + ``attention_mask``
    from the HF tokeniser; both are required so a forward refactor
    that drops one trips at boot rather than silently degrading.
    """

    class _Stub(torch.nn.Module):
        pass

    contract = derive_multi_axis_contract(
        _Stub(), encoder_alias="finbert_fed_adjacent"
    )
    assert contract.required_kwargs == ("input_ids", "attention_mask")
    assert contract.encoder_alias == "finbert_fed_adjacent"
    assert contract.schema_version == SIDECAR_SCHEMA_VERSION


def test_multi_axis_sidecar_absent_soft_legacy(tmp_path: Path, monkeypatch) -> None:
    """A multi-axis checkpoint with no sidecar binds (legacy path)."""

    from app.services import multi_axis_classifier as service

    ckpt = tmp_path / "text_multi_axis_best.pt"
    ckpt.write_bytes(b"\x00")  # presence is what gates the path

    monkeypatch.setattr(service, "_state", None)
    monkeypatch.setenv("FED_PULSE_TEXT_MULTI_AXIS_CHECKPOINT", str(ckpt))

    ok, status = service._validate_contract(ckpt)
    assert ok is True
    assert status == "sidecar_absent"
    surface = service.get_serving_contract_status()
    assert surface["status"] == "sidecar_absent"


def test_multi_axis_loader_refuses_on_signature_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    """A sidecar declaring a kwarg the forward does not accept hard-refuses.

    The loader must raise ``RuntimeError`` with a structured status
    string (no ``str(exc)`` leak) and surface the reason on the
    contract-status surface.
    """

    from app.services import multi_axis_classifier as service

    ckpt = tmp_path / "text_multi_axis_best.pt"
    ckpt.write_bytes(b"\x00")
    # Hand-write a sidecar declaring a kwarg the classifier forward
    # does not accept. The legitimate kwargs are
    # ``input_ids`` + ``attention_mask`` -- ``not_a_real_kwarg`` is
    # the canonical drift case.
    bad = InferenceContract(
        schema_version=SIDECAR_SCHEMA_VERSION,
        model_class="TextMultiAxisClassifier",
        required_kwargs=("not_a_real_kwarg",),
    )
    write_sidecar(bad, ckpt)

    monkeypatch.setattr(service, "_state", None)
    monkeypatch.setenv("FED_PULSE_TEXT_MULTI_AXIS_CHECKPOINT", str(ckpt))

    ok, status = service._validate_contract(ckpt)
    assert ok is False
    assert status == "serving_signature_missing_kwargs"

    surface = service.get_serving_contract_status()
    assert surface["status"] == "serving_signature_missing_kwargs"
    assert "not_a_real_kwarg" in surface["missing_kwargs"]

    # ``_load_state`` is the public boot path; the structured status
    # must propagate as a ``RuntimeError`` rather than swallowing
    # silently into ``None``.
    with pytest.raises(RuntimeError) as excinfo:
        service._load_state()
    assert "inference contract" in str(excinfo.value).lower()


def test_multi_axis_loader_binds_on_matching_contract(
    tmp_path: Path, monkeypatch
) -> None:
    """A sidecar whose required kwargs match the forward signature binds."""

    from app.services import multi_axis_classifier as service

    ckpt = tmp_path / "text_multi_axis_best.pt"
    ckpt.write_bytes(b"\x00")

    contract = InferenceContract(
        schema_version=SIDECAR_SCHEMA_VERSION,
        model_class="TextMultiAxisClassifier",
        required_kwargs=("input_ids", "attention_mask"),
    )
    write_sidecar(contract, ckpt)

    monkeypatch.setattr(service, "_state", None)
    monkeypatch.setenv("FED_PULSE_TEXT_MULTI_AXIS_CHECKPOINT", str(ckpt))

    ok, status = service._validate_contract(ckpt)
    assert ok is True
    assert status == "ok"

    surface = service.get_serving_contract_status()
    assert surface["status"] == "ok"


def test_multi_axis_save_emits_sidecar(tmp_path: Path, monkeypatch) -> None:
    """The training save site emits an inference-contract sidecar.

    Drives ``_save_checkpoint`` from
    :mod:`app.data.train_text_multi_axis_classifier` with a stub
    model so the sidecar derivation is exercised end-to-end without
    needing a real transformer encoder.
    """

    import argparse

    from app.data.train_text_multi_axis_classifier import _save_checkpoint

    class _StubModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def metadata(self) -> dict[str, object]:
            return {"encoder_alias": "finbert_fed_adjacent"}

    args = argparse.Namespace(
        training_package_id="test_pkg",
        encoder_alias="finbert_fed_adjacent",
        epochs=1,
        seed=0,
        learning_rate=1e-3,
        batch_size=1,
        val_fraction=0.1,
        max_length=64,
        data_source="events_parquet",
        gtfintechlab_fed_only=False,
        cross_bank_supervision="off",
        cross_bank_stance_weight=0.25,
    )
    ckpt = tmp_path / "text_multi_axis_best.pt"

    _save_checkpoint(
        _StubModel(),  # type: ignore[arg-type]
        path=ckpt,
        metrics={"loss": 0.1},
        args=args,
        class_weights={},
    )

    sidecar = sidecar_path_for(ckpt)
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["encoder_alias"] == "finbert_fed_adjacent"
    assert "input_ids" in payload["required_kwargs"]
    assert "attention_mask" in payload["required_kwargs"]


# ---------------------------------------------------------------------------
# Trajectory bundle
# ---------------------------------------------------------------------------


def _toy_trajectory_model() -> tuple[object, object, Path]:
    """Build a minimal trajectory LSTM for the contract assertions."""

    from app.trajectory.model import TrajectoryConfig, build_model

    config = TrajectoryConfig(
        architecture="lstm", embedding_dim=4, history_length=3
    )
    model = build_model(config)
    return model, config, Path()


def test_derive_trajectory_contract_declares_forward_kwargs() -> None:
    """Trajectory contract requires ``inputs``; ``mask`` is optional."""

    model, _config, _ = _toy_trajectory_model()
    contract = derive_trajectory_contract(
        model, encoder_alias="finbert_fed_adjacent_xbank_dapt"
    )
    assert contract.required_kwargs == ("inputs",)
    assert contract.optional_kwargs == ("mask",)
    assert contract.encoder_alias == "finbert_fed_adjacent_xbank_dapt"


def test_trajectory_save_emits_sidecar(tmp_path: Path) -> None:
    """``trajectory.model.save_model`` writes the sidecar next to .pt."""

    from app.trajectory import model as traj_model

    model, config, _ = _toy_trajectory_model()
    ckpt = tmp_path / "model.pt"
    traj_model.save_model(
        model,
        config,
        ckpt,
        encoder_alias="finbert_fed_adjacent_xbank_dapt",
    )

    sidecar = sidecar_path_for(ckpt)
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["encoder_alias"] == "finbert_fed_adjacent_xbank_dapt"
    assert payload["required_kwargs"] == ["inputs"]
    assert "mask" in payload["optional_kwargs"]


def test_trajectory_sidecar_absent_soft_legacy(tmp_path: Path) -> None:
    """A trajectory bundle with no sidecar degrades to ``sidecar_absent``."""

    from app.services import trajectory as service

    model, _config, _ = _toy_trajectory_model()
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"\x00")  # presence is what gates the read

    ok, status = service._validate_contract(ckpt, model)
    assert ok is True
    assert status == "sidecar_absent"
    surface = service.get_serving_contract_status()
    assert surface["status"] == "sidecar_absent"


def test_trajectory_loader_refuses_on_signature_mismatch(tmp_path: Path) -> None:
    """A sidecar declaring an unknown kwarg refuses to bind."""

    from app.services import trajectory as service

    model, _config, _ = _toy_trajectory_model()
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"\x00")
    bad = InferenceContract(
        schema_version=SIDECAR_SCHEMA_VERSION,
        model_class=type(model).__name__,
        required_kwargs=("not_a_real_kwarg",),
    )
    write_sidecar(bad, ckpt)

    ok, status = service._validate_contract(ckpt, model)
    assert ok is False
    assert status == "serving_signature_missing_kwargs"

    surface = service.get_serving_contract_status()
    assert surface["status"] == "serving_signature_missing_kwargs"
    assert "not_a_real_kwarg" in surface["missing_kwargs"]


def test_trajectory_loader_binds_on_matching_contract(tmp_path: Path) -> None:
    """A sidecar whose required kwargs match the trajectory forward binds."""

    from app.services import trajectory as service

    model, _config, _ = _toy_trajectory_model()
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"\x00")
    contract = InferenceContract(
        schema_version=SIDECAR_SCHEMA_VERSION,
        model_class=type(model).__name__,
        required_kwargs=("inputs",),
        optional_kwargs=("mask",),
    )
    write_sidecar(contract, ckpt)

    ok, status = service._validate_contract(ckpt, model)
    assert ok is True
    assert status == "ok"

    surface = service.get_serving_contract_status()
    assert surface["status"] == "ok"


def test_trajectory_load_model_round_trip_carries_sidecar(tmp_path: Path) -> None:
    """End-to-end: save_model -> load_model -> sidecar exists alongside."""

    from app.trajectory import model as traj_model

    model, config, _ = _toy_trajectory_model()
    ckpt = tmp_path / "model.pt"
    traj_model.save_model(model, config, ckpt, encoder_alias="some_alias")
    reloaded, reloaded_config = traj_model.load_model(ckpt)
    assert reloaded_config.architecture == config.architecture
    assert sidecar_path_for(ckpt).exists()
    loaded_contract = read_sidecar(ckpt)
    assert loaded_contract is not None
    assert loaded_contract.encoder_alias == "some_alias"
    assert loaded_contract.required_kwargs == ("inputs",)
