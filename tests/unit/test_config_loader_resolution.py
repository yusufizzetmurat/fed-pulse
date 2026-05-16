"""Path-resolution tests for the ablation-config loader.

The loader resolves bare relative paths against a fixed list of
``configs/`` candidates (repo root + container ``/app/configs``) plus a
cwd walk fallback. These tests pin the resolution semantics so a
future refactor cannot silently re-introduce the original cwd-only
behaviour that broke two tests across the docker boundary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from app.training.config_loader import (
    _resolve_config_path,
    load_ablation_config,
)


def test_absolute_path_resolves_unchanged(tmp_path: Path) -> None:
    target = tmp_path / "abs.yaml"
    target.write_text("name: x\n", encoding="utf-8")
    resolved = _resolve_config_path(target)
    assert resolved == target


def test_existing_relative_path_resolves_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "configs" / "ablation_demo.yaml"
    target.parent.mkdir()
    target.write_text("name: demo\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    resolved = _resolve_config_path(Path("configs/ablation_demo.yaml"))
    assert resolved.resolve() == target.resolve()


def test_missing_path_returns_input_for_clean_filenotfound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # cwd has no ``configs/`` dir, repo-root candidates do not match;
    # the resolver returns the input path unchanged so the caller's
    # ``FileNotFoundError`` carries the original argument.
    monkeypatch.chdir(tmp_path)
    out = _resolve_config_path(Path("configs/nope.yaml"))
    assert out == Path("configs/nope.yaml")


def test_walk_upward_finds_configs_above_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Layout: tmp/repo/configs/foo.yaml ; cwd = tmp/repo/some/sub/dir
    repo = tmp_path / "repo"
    deep = repo / "some" / "sub" / "dir"
    deep.mkdir(parents=True)
    (repo / "configs").mkdir()
    target = repo / "configs" / "ablation_walk.yaml"
    target.write_text("name: walk\n", encoding="utf-8")
    monkeypatch.chdir(deep)
    resolved = _resolve_config_path(Path("configs/ablation_walk.yaml"))
    assert resolved.resolve() == target.resolve()


def test_load_ablation_config_raises_filenotfound_for_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_ablation_config("configs/does_not_exist.yaml")


def test_load_ablation_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    bad = tmp_path / "list.yaml"
    bad.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_ablation_config(bad)


def test_load_ablation_config_rejects_non_mapping_overrides(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: x\nfeature_overrides: [1, 2, 3]\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="feature_overrides must be a mapping"):
        load_ablation_config(bad)


def test_load_ablation_config_accepts_bare_filename_against_repo_configs() -> None:
    # The repo ships ``configs/ablation_no_text.yaml`` and
    # ``configs/ablation_calendar_only.yaml`` at the repo root and at
    # ``/app/configs`` inside the container. The loader resolves either
    # bare-filename or ``configs/<file>`` input to the same file.
    bare = load_ablation_config("ablation_no_text.yaml")
    nested = load_ablation_config("configs/ablation_no_text.yaml")
    assert bare.name == nested.name
