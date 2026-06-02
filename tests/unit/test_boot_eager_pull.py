"""Behavioural tests for the boot-time eager-pull shim.

The shim must:
- never raise out; failures degrade silently to the cold-start path
- never overwrite a file that already exists in ``MODELS_DIR``
- only touch artefacts mapped in ``_ARTEFACT_FILES`` (rates_heads et al.
  are intentionally left out to avoid clobbering the forecaster path)
- no-op cleanly when ``HF_TOKEN`` is unset
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.boot import eager_pull
from app.models.registry import ArtefactRef


@pytest.fixture()
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "models"
    target.mkdir()
    # The shim reads MODELS_DIR off ``app.models.config``; monkeypatch
    # the symbol the shim imports.
    monkeypatch.setattr(
        "app.models.config.MODELS_DIR", target, raising=True
    )
    return target


def _stub_artefact(name: str = "forecaster_canonical") -> ArtefactRef:
    return ArtefactRef(
        name=name,
        hf_uri="hf://yusufizzetmurat/fed-pulse-forecaster",
        revision="deadbeefcafe",
        eager=True,
        description="",
        inference_features=(),
    )


def test_hydrate_no_token_returns_silently(
    monkeypatch: pytest.MonkeyPatch, models_dir: Path
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # Should not raise, should not touch MODELS_DIR
    eager_pull.hydrate()
    assert list(models_dir.iterdir()) == []


def test_hydrate_skips_artefact_with_no_file_mapping(
    monkeypatch: pytest.MonkeyPatch, models_dir: Path
) -> None:
    monkeypatch.setenv("HF_TOKEN", "stub")

    called: dict[str, int] = {"snapshot_download": 0}

    def _fake_snapshot_download(**_kwargs: Any) -> str:
        called["snapshot_download"] += 1
        return str(models_dir)  # unused — should not get here

    monkeypatch.setattr(
        "app.boot.eager_pull.snapshot_download",
        _fake_snapshot_download,
        raising=False,
    )
    monkeypatch.setattr(
        "app.models.registry.eager_artefacts",
        lambda: [_stub_artefact("unknown_artefact")],
    )
    eager_pull.hydrate()
    assert called["snapshot_download"] == 0
    assert list(models_dir.iterdir()) == []


def test_hydrate_copies_missing_files_only(
    monkeypatch: pytest.MonkeyPatch, models_dir: Path, tmp_path: Path
) -> None:
    """Files absent on disk get copied; files already present are left alone."""

    monkeypatch.setenv("HF_TOKEN", "stub")
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    # Stage all five files the mapping expects, each with a unique
    # marker payload so the test can prove which side won.
    for fname in eager_pull._ARTEFACT_FILES["forecaster_canonical"]:
        (snapshot / fname).write_bytes(b"FROM_HF_" + fname.encode())

    # Pre-populate one file locally to prove the shim does not overwrite.
    pre_existing_name = "forecaster_best.pt"
    pre_existing = models_dir / pre_existing_name
    pre_existing.write_bytes(b"LOCAL_CANONICAL")

    monkeypatch.setattr(
        "app.boot.eager_pull.snapshot_download",
        lambda **kwargs: str(snapshot),
        raising=False,
    )
    monkeypatch.setattr(
        "app.models.registry.eager_artefacts",
        lambda: [_stub_artefact()],
    )

    eager_pull.hydrate()

    # Pre-existing local file is untouched.
    assert pre_existing.read_bytes() == b"LOCAL_CANONICAL"
    # Every other mapped file got pulled in.
    for fname in eager_pull._ARTEFACT_FILES["forecaster_canonical"]:
        if fname == pre_existing_name:
            continue
        assert (models_dir / fname).read_bytes() == b"FROM_HF_" + fname.encode()


def test_hydrate_snapshot_download_failure_does_not_raise(
    monkeypatch: pytest.MonkeyPatch, models_dir: Path
) -> None:
    monkeypatch.setenv("HF_TOKEN", "stub")

    def _explode(**_kwargs: Any) -> str:
        raise RuntimeError("simulated HF outage")

    monkeypatch.setattr(
        "app.boot.eager_pull.snapshot_download", _explode, raising=False
    )
    monkeypatch.setattr(
        "app.models.registry.eager_artefacts",
        lambda: [_stub_artefact()],
    )

    # The shim swallows the exception; MODELS_DIR stays empty and the
    # caller (entrypoint) continues to uvicorn.
    eager_pull.hydrate()
    assert list(models_dir.iterdir()) == []


def test_hydrate_missing_file_in_snapshot_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch,
    models_dir: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "stub")
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    # Only stage two of five files; the rest should log "missing".
    (snapshot / "forecaster_best.pt").write_bytes(b"x")
    (snapshot / "forecaster_best.conformal.json").write_bytes(b"y")

    monkeypatch.setattr(
        "app.boot.eager_pull.snapshot_download",
        lambda **kwargs: str(snapshot),
        raising=False,
    )
    monkeypatch.setattr(
        "app.models.registry.eager_artefacts",
        lambda: [_stub_artefact()],
    )

    with caplog.at_level("WARNING", logger="app.boot.eager_pull"):
        eager_pull.hydrate()

    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("missing from snapshot" in m for m in warnings)
    assert (models_dir / "forecaster_best.pt").exists()


def test_main_returns_zero_even_on_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(eager_pull, "hydrate", _explode)
    assert eager_pull.main() == 0
