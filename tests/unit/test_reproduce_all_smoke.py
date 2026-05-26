"""Tests for ``make reproduce-all`` wiring (#302 Stage 5)."""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Discover the Makefile robustly. The dev container only mounts
# ``./backend:/app`` plus ``./scripts:/app/scripts:ro`` and
# ``./tests:/app/tests:ro``, so the repo-root Makefile is not visible
# from inside the container. Walk up from this test file and from the
# scripts dir to find it; the test that asserts on Makefile contents
# skips when neither path resolves (CI on the host repo always finds
# it; in-container runs skip the assertion).


def _find_makefile() -> Path | None:
    for candidate in (REPO_ROOT / "Makefile", SCRIPTS_DIR.parent / "Makefile"):
        if candidate.exists():
            return candidate
    return None


def test_makefile_declares_reproduce_all_target() -> None:
    makefile = _find_makefile()
    if makefile is None:
        pytest.skip(
            "Makefile not visible from this run (container does not mount the repo root)"
        )
    contents = makefile.read_text(encoding="utf-8")
    # PHONY declaration includes the target.
    assert "reproduce-all" in contents.split(".PHONY:", 1)[1].split("\n", 1)[0]
    # The recipe wires through to the python smoke script and threads the
    # HF token + the reproduce-smoke env flag — both are load-bearing.
    assert "python scripts/reproduce_all.py" in contents
    assert "HF_TOKEN=$$HF_TOKEN" in contents
    assert "FED_PULSE_REPRODUCE_SMOKE=1" in contents


def test_reproduce_script_imports() -> None:
    # Importing the script must not error — that catches broken paths /
    # missing dependencies in the registry resolver layer.
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import reproduce_all  # noqa: F401
    finally:
        sys.path.remove(str(SCRIPTS_DIR))


def test_reproduce_script_pulls_training_package_and_runs_smoke(
    monkeypatch, tmp_path: Path
) -> None:
    # End-to-end smoke against the script with the HF fetch + the
    # actual training subprocess both mocked out. We assert that the
    # script (a) calls the registry resolver for the canonical training
    # package URI and (b) invokes ``python -m app.train_forecaster``
    # with the smoke-shape arguments.

    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import reproduce_all
    finally:
        sys.path.remove(str(SCRIPTS_DIR))

    fake_snapshot = tmp_path / "snapshot"
    fake_snapshot.mkdir()
    (fake_snapshot / "events.parquet").write_bytes(b"")

    resolve_calls: list[str] = []

    def fake_resolve(uri: str, **_kwargs: object) -> Path:
        resolve_calls.append(uri)
        return fake_snapshot

    monkeypatch.setattr(reproduce_all, "resolve_hf_uri", fake_resolve)

    fake_data_dir = tmp_path / "data"
    monkeypatch.setattr(reproduce_all, "DATA_DIR", fake_data_dir)

    class _Result:
        returncode = 0

    subprocess_calls: list[list[str]] = []

    def fake_run(cmd, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        subprocess_calls.append(list(cmd))
        return _Result()

    monkeypatch.setattr(reproduce_all.subprocess, "run", fake_run)

    exit_code = reproduce_all.main()

    assert exit_code == 0
    assert len(resolve_calls) == 1
    assert resolve_calls[0].startswith("hf://datasets/yusufizzetmurat/fed-pulse-training-package")

    # The training package was copied into the canonical processed dir.
    target = fake_data_dir / "processed" / "canonical"
    assert (target / "events.parquet").exists()

    assert len(subprocess_calls) == 1
    cmd = subprocess_calls[0]
    assert "app.train_forecaster" in cmd
    assert "--training-package-id" in cmd
    assert "canonical" in cmd
    assert "--epochs" in cmd
    assert "1" in cmd
    assert "--seed" in cmd
    assert "11" in cmd


def test_reproduce_script_propagates_nonzero_exit(monkeypatch, tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import reproduce_all
    finally:
        sys.path.remove(str(SCRIPTS_DIR))

    fake_snapshot = tmp_path / "snap"
    fake_snapshot.mkdir()

    monkeypatch.setattr(reproduce_all, "resolve_hf_uri", lambda *_a, **_k: fake_snapshot)
    monkeypatch.setattr(reproduce_all, "DATA_DIR", tmp_path / "data")

    class _Fail:
        returncode = 7

    monkeypatch.setattr(reproduce_all.subprocess, "run", lambda *_a, **_k: _Fail())

    assert reproduce_all.main() == 7


def test_push_artefacts_dry_run_does_not_touch_hf(monkeypatch) -> None:
    # The push uploader must be safely runnable from a developer
    # machine in dry-run mode (no HF_TOKEN, no network).
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import push_artefacts_to_hub
    finally:
        sys.path.remove(str(SCRIPTS_DIR))

    import huggingface_hub  # type: ignore[import-not-found]

    def fail(**_kwargs: object) -> None:
        raise AssertionError("hf must not be touched in dry-run")

    # Guard every potential network entrypoint the script could hit.
    for attr in ("create_repo", "upload_folder", "upload_file"):
        if hasattr(huggingface_hub, attr):
            monkeypatch.setattr(huggingface_hub, attr, fail, raising=False)

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    rc = push_artefacts_to_hub.main(["--dry-run"])
    assert rc == 0
