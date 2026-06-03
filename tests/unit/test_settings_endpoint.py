"""Settings endpoint coverage for the two checkpoint sources.

The ``/settings/checkpoints`` handler reads two places: the host-mounted
``backend/models/`` directory and the ``huggingface_hub`` snapshot cache.
Files under MODELS_DIR carry ``source="models_dir"``; files reached via
``try_to_load_from_cache`` (e.g. the multi-axis classifier when no local
copy exists yet) carry ``source="hf_cache"`` plus HF Hub provenance
(``repo``, ``revision``, ``snapshot_path``).

Both sources are exercised here with a mocked filesystem so the test
does not depend on what is or is not in the developer's real HF cache.
"""

from __future__ import annotations

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
from app.models.registry import ArtefactRef  # noqa: E402


def _write_toy_checkpoint(path: Path) -> None:
    model = build_serving_forecaster(
        ModelConfig(input_size=FEATURE_SIZE, architecture="lstm")
    )
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


def _stub_artefacts() -> dict[str, ArtefactRef]:
    """Two registered artefacts: the forecaster (MODELS_DIR landing) and
    the multi-axis classifier (HF cache landing)."""

    return {
        "forecaster_canonical": ArtefactRef(
            name="forecaster_canonical",
            hf_uri="hf://yusufizzetmurat/fed-pulse-forecaster",
            revision="7ab0a87399336b67ca4e2423a40ed8ab6666530b",
            eager=True,
            description="",
            inference_features=(),
        ),
        "multi_axis_text_classifier": ArtefactRef(
            name="multi_axis_text_classifier",
            hf_uri="hf://yusufizzetmurat/fed-pulse-multi-axis-text-classifier",
            revision="c863f18753e87f2576b3609112a10efd85671e8f",
            eager=False,
            description="",
            inference_features=(),
        ),
    }


def test_settings_checkpoints_reports_both_sources(tmp_path, monkeypatch):
    """A MODELS_DIR entry and an HF-cache entry land in the response
    side by side, each tagged with the correct ``source`` value."""

    # --- MODELS_DIR side: write a real forecaster checkpoint --------
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    forecaster_ckpt = models_dir / "forecaster_best.pt"
    _write_toy_checkpoint(forecaster_ckpt)

    import app.models.config as model_config_mod

    monkeypatch.setattr(model_config_mod, "MODELS_DIR", models_dir)
    monkeypatch.setattr(forecaster_service, "BEST_MODEL_PATH", forecaster_ckpt)
    monkeypatch.setattr(forecaster_service, "_model", None)
    monkeypatch.setattr(forecaster_service, "_model_artifact_metadata", None)

    # --- HF cache side: write a fake .pt under tmp_path that the
    # ``try_to_load_from_cache`` stub will return for the multi-axis
    # repo + filename. Anything else returns None so we don't surface
    # phantom entries.
    hf_cache_dir = tmp_path / "hf_cache" / "snapshots" / "c863f18"
    hf_cache_dir.mkdir(parents=True)
    fake_multi_axis = hf_cache_dir / "text_multi_axis_best.pt"
    fake_multi_axis.write_bytes(b"\x00" * 128)

    def fake_try_to_load_from_cache(
        *,
        repo_id: str,
        filename: str,
        repo_type: str = "model",
        revision: str | None = None,
    ):
        if (
            repo_id == "yusufizzetmurat/fed-pulse-multi-axis-text-classifier"
            and filename == "text_multi_axis_best.pt"
        ):
            return str(fake_multi_axis)
        return None

    # Patch at the import site inside the helper. The helper does
    # ``from huggingface_hub import try_to_load_from_cache`` so we patch
    # the attribute on the module object.
    import huggingface_hub as hf_mod

    monkeypatch.setattr(
        hf_mod, "try_to_load_from_cache", fake_try_to_load_from_cache
    )
    monkeypatch.setattr(
        "app.models.registry.load_artefacts",
        _stub_artefacts,
        raising=True,
    )

    client = TestClient(main_mod.app)
    resp = client.get("/settings/checkpoints")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    by_filename = {row["filename"]: row for row in payload["checkpoints"]}

    # MODELS_DIR entry --------------------------------------------------
    assert "forecaster_best.pt" in by_filename, payload
    fc = by_filename["forecaster_best.pt"]
    assert fc["source"] == "models_dir"
    assert fc["repo"] is None
    assert fc["snapshot_path"] is None
    assert fc["role"] == "forecaster"

    # HF-cache entry ----------------------------------------------------
    assert "text_multi_axis_best.pt" in by_filename, payload
    ma = by_filename["text_multi_axis_best.pt"]
    assert ma["source"] == "hf_cache"
    assert ma["repo"] == "yusufizzetmurat/fed-pulse-multi-axis-text-classifier"
    assert ma["revision"] == "c863f18753e87f2576b3609112a10efd85671e8f"
    assert ma["snapshot_path"] == str(fake_multi_axis)
    assert ma["role"] == "multi_axis"
    assert ma["size_bytes"] == 128


def test_settings_checkpoints_skips_hf_cache_when_models_dir_has_file(
    tmp_path, monkeypatch
):
    """MODELS_DIR is authoritative: when ``text_multi_axis_best.pt``
    exists locally, the HF-cache walk must not surface a duplicate
    entry."""

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    local_multi_axis = models_dir / "text_multi_axis_best.pt"
    local_multi_axis.write_bytes(b"\x00" * 64)

    import app.models.config as model_config_mod

    monkeypatch.setattr(model_config_mod, "MODELS_DIR", models_dir)
    monkeypatch.setattr(forecaster_service, "_model", None)
    monkeypatch.setattr(forecaster_service, "_model_artifact_metadata", None)

    hf_cache_dir = tmp_path / "hf_cache" / "snapshots" / "c863f18"
    hf_cache_dir.mkdir(parents=True)
    phantom = hf_cache_dir / "text_multi_axis_best.pt"
    phantom.write_bytes(b"\xff" * 99)

    def fake_try_to_load_from_cache(
        *,
        repo_id: str,
        filename: str,
        repo_type: str = "model",
        revision: str | None = None,
    ):
        # Pretend the cache has every file we ask about — the dedupe
        # filter is what we're testing.
        return str(phantom) if filename == "text_multi_axis_best.pt" else None

    import huggingface_hub as hf_mod

    monkeypatch.setattr(
        hf_mod, "try_to_load_from_cache", fake_try_to_load_from_cache
    )
    monkeypatch.setattr(
        "app.models.registry.load_artefacts",
        _stub_artefacts,
        raising=True,
    )

    client = TestClient(main_mod.app)
    resp = client.get("/settings/checkpoints")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    multi_axis_rows = [
        row
        for row in payload["checkpoints"]
        if row["filename"] == "text_multi_axis_best.pt"
    ]
    assert len(multi_axis_rows) == 1, multi_axis_rows
    row = multi_axis_rows[0]
    # The local copy wins.
    assert row["source"] == "models_dir"
    assert row["size_bytes"] == 64
    assert row["snapshot_path"] is None


def test_settings_checkpoints_handles_empty_hf_cache(tmp_path, monkeypatch):
    """When the HF cache holds nothing for the registered artefacts,
    only MODELS_DIR rows surface."""

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    forecaster_ckpt = models_dir / "forecaster_best.pt"
    _write_toy_checkpoint(forecaster_ckpt)

    import app.models.config as model_config_mod

    monkeypatch.setattr(model_config_mod, "MODELS_DIR", models_dir)
    monkeypatch.setattr(forecaster_service, "BEST_MODEL_PATH", forecaster_ckpt)
    monkeypatch.setattr(forecaster_service, "_model", None)
    monkeypatch.setattr(forecaster_service, "_model_artifact_metadata", None)

    def fake_try_to_load_from_cache(**_kwargs):
        return None  # cache cold

    import huggingface_hub as hf_mod

    monkeypatch.setattr(
        hf_mod, "try_to_load_from_cache", fake_try_to_load_from_cache
    )
    monkeypatch.setattr(
        "app.models.registry.load_artefacts",
        _stub_artefacts,
        raising=True,
    )

    client = TestClient(main_mod.app)
    resp = client.get("/settings/checkpoints")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    sources = {row["source"] for row in payload["checkpoints"]}
    assert sources == {"models_dir"}, payload
