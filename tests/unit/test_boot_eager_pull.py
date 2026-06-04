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


def test_hydrate_copies_missing_files_and_overwrites_drift(
    monkeypatch: pytest.MonkeyPatch, models_dir: Path, tmp_path: Path
) -> None:
    """Files absent on disk get copied; matching files are skipped;
    drift (existing local file whose content differs from the snapshot)
    is OVERWRITTEN so a stale checkpoint from a prior revision cannot
    mask the pinned artefact.
    """

    monkeypatch.setenv("HF_TOKEN", "stub")
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    # Stage all five files the mapping expects, each with a unique
    # marker payload so the test can prove which side won.
    for fname in eager_pull._ARTEFACT_FILES["forecaster_canonical"]:
        (snapshot / fname).write_bytes(b"FROM_HF_" + fname.encode())

    # Pre-populate one file locally with DIFFERENT content so the drift
    # branch overwrites it. Pre-populate a second file with byte-identical
    # content so the same-content branch keeps it untouched.
    drift_name = "forecaster_best.pt"
    drift_path = models_dir / drift_name
    drift_path.write_bytes(b"STALE_LOCAL")  # differs from the snapshot payload

    match_name = "forecaster_best.pt.inference_contract.json"
    match_path = models_dir / match_name
    match_payload = b"FROM_HF_" + match_name.encode()
    match_path.write_bytes(match_payload)

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

    # Drift was overwritten with the snapshot copy.
    assert drift_path.read_bytes() == b"FROM_HF_" + drift_name.encode()
    # Byte-identical local copy stayed in place.
    assert match_path.read_bytes() == match_payload
    # Every other mapped file got pulled in.
    for fname in eager_pull._ARTEFACT_FILES["forecaster_canonical"]:
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


def test_split_entry_plain_string_collapses_to_triple() -> None:
    """A flat filename maps to ``(name, name, "MODELS")`` — the legacy
    flat-file mapping defaults to the ``MODELS_DIR`` root.
    """

    assert eager_pull._split_entry("forecaster_best.pt") == (
        "forecaster_best.pt",
        "forecaster_best.pt",
        "MODELS",
    )


def test_split_entry_pair_form_defaults_to_models_root() -> None:
    """A ``(snapshot_name, dst_relpath)`` pair expands to a ``"MODELS"``
    root so the existing ``volume_har_canonical`` mapping is unchanged.
    """

    entry = ("volume_har_artifact.json", "volume_har/volume_har_artifact.json")
    assert eager_pull._split_entry(entry) == (
        "volume_har_artifact.json",
        "volume_har/volume_har_artifact.json",
        "MODELS",
    )


def test_split_entry_triple_form_returns_src_dst_root() -> None:
    """A ``(snapshot_name, dst_relpath, dst_root)`` triple is preserved as-is."""

    entry = ("model.pt", "artifacts/trajectory/trajectory_transformer/model.pt", "DATA")
    assert eager_pull._split_entry(entry) == entry


def test_hydrate_tuple_entry_lands_in_subdirectory(
    monkeypatch: pytest.MonkeyPatch, models_dir: Path, tmp_path: Path
) -> None:
    """The tuple-form mapping copies into ``MODELS_DIR / dst_relpath``.

    Exercises the ``volume_har_canonical`` path end-to-end: the file in
    the snapshot is named flat (``volume_har_artifact.json``), but the
    destination must land under ``models/volume_har/``. Guards against
    regressions in ``dst.parent.mkdir`` and the ``snapshot_names`` list
    using the wrong side of the tuple.
    """

    monkeypatch.setenv("HF_TOKEN", "stub")
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "volume_har_artifact.json").write_bytes(b"FROM_HF_volume")

    captured: dict[str, Any] = {}

    def _fake_snapshot_download(**kwargs: Any) -> str:
        captured.update(kwargs)
        return str(snapshot)

    monkeypatch.setattr(
        "app.boot.eager_pull.snapshot_download",
        _fake_snapshot_download,
        raising=False,
    )
    monkeypatch.setattr(
        "app.models.registry.eager_artefacts",
        lambda: [
            ArtefactRef(
                name="volume_har_canonical",
                hf_uri="hf://yusufizzetmurat/fomc-volume-har",
                revision="540b25a8d66c6f0110dd02b89f10e507139c80b8",
                eager=True,
                description="",
                inference_features=(),
            )
        ],
    )

    eager_pull.hydrate()

    # The destination sits under the sub-directory carved out of the
    # tuple's right-hand side; the snapshot path itself was flat.
    dst = models_dir / "volume_har" / "volume_har_artifact.json"
    assert dst.exists()
    assert dst.read_bytes() == b"FROM_HF_volume"
    # ``allow_patterns`` must use the snapshot-side (flat) name, not the
    # destination relpath, or HF would 404 on the prefix scan.
    assert captured.get("allow_patterns") == ["volume_har_artifact.json"]
