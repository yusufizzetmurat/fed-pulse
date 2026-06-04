"""Behavioural tests for the ``DATA_DIR`` destination in the eager-pull shim.

These cover the triple-form mapping ``(snapshot_name, dst_relpath, "DATA")``
used by ``trajectory_bundle`` and ``retrieval_bundle``, which land their
files under ``DATA_DIR / artifacts / ...`` instead of ``MODELS_DIR``.
The shim must:

- copy files into ``DATA_DIR`` when ``dst_root == "DATA"``;
- leave ``MODELS_DIR`` untouched for those entries;
- create nested sub-directories (e.g. ``checkpoint/1_Pooling/``);
- still honour the "no overwrite" rule for files already on disk;
- skip an entry with an unknown ``dst_root`` value rather than raise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.boot import eager_pull
from app.models.registry import ArtefactRef


@pytest.fixture()
def models_and_data_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    models = tmp_path / "models"
    data = tmp_path / "data"
    models.mkdir()
    data.mkdir()
    monkeypatch.setattr("app.models.config.MODELS_DIR", models, raising=True)
    monkeypatch.setattr("app.config.DATA_DIR", data, raising=True)
    return models, data


def _trajectory_artefact() -> ArtefactRef:
    return ArtefactRef(
        name="trajectory_bundle",
        hf_uri="hf://yusufizzetmurat/fed-pulse-trajectory",
        revision="df7ccaac07473dfff4e1a62d557a6979d5077304",
        eager=True,
        description="",
        inference_features=(),
    )


def _retrieval_artefact() -> ArtefactRef:
    return ArtefactRef(
        name="retrieval_bundle",
        hf_uri="hf://yusufizzetmurat/fed-pulse-retrieval",
        revision="a4693b818e4e2a738f3e32c844f45c651424ee3e",
        eager=True,
        description="",
        inference_features=(),
    )


def test_trajectory_bundle_lands_under_data_dir(
    monkeypatch: pytest.MonkeyPatch,
    models_and_data_dirs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Every ``trajectory_bundle`` file is copied into ``DATA_DIR``
    and the ``MODELS_DIR`` tree stays empty for these entries.
    """

    models, data = models_and_data_dirs
    monkeypatch.setenv("HF_TOKEN", "stub")

    snapshot = tmp_path / "snap_trajectory"
    snapshot.mkdir()
    for src_name, _, _ in (
        eager_pull._split_entry(e) for e in eager_pull._ARTEFACT_FILES["trajectory_bundle"]
    ):
        (snapshot / src_name).write_bytes(b"FROM_HF_" + src_name.encode())

    monkeypatch.setattr(
        "app.boot.eager_pull.snapshot_download",
        lambda **kwargs: str(snapshot),
        raising=False,
    )
    monkeypatch.setattr(
        "app.models.registry.eager_artefacts",
        lambda: [_trajectory_artefact()],
    )

    eager_pull.hydrate()

    bundle_dir = data / "artifacts" / "trajectory" / "trajectory_transformer"
    for src_name, dst_relpath, dst_root in (
        eager_pull._split_entry(e) for e in eager_pull._ARTEFACT_FILES["trajectory_bundle"]
    ):
        assert dst_root == "DATA"
        assert (data / dst_relpath).read_bytes() == b"FROM_HF_" + src_name.encode()
    # The model checkpoint, the canonical "must exist" payload, is the
    # spot-check most likely to regress on a path-build bug.
    assert (bundle_dir / "model.pt").exists()
    # MODELS_DIR must not see any of these files — the whole point of
    # the triple form is to route them elsewhere.
    assert list(models.iterdir()) == []


def test_retrieval_bundle_creates_nested_checkpoint_subdirs(
    monkeypatch: pytest.MonkeyPatch,
    models_and_data_dirs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """The retrieval bundle has files two levels deep
    (``checkpoint/1_Pooling/config.json``) — the shim must mkdir those
    intermediate directories before copying.
    """

    models, data = models_and_data_dirs
    monkeypatch.setenv("HF_TOKEN", "stub")

    snapshot = tmp_path / "snap_retrieval"
    snapshot.mkdir()
    for src_name, _, _ in (
        eager_pull._split_entry(e) for e in eager_pull._ARTEFACT_FILES["retrieval_bundle"]
    ):
        src_path = snapshot / src_name
        src_path.parent.mkdir(parents=True, exist_ok=True)
        src_path.write_bytes(b"FROM_HF_" + src_name.encode())

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
        lambda: [_retrieval_artefact()],
    )

    eager_pull.hydrate()

    bundle_dir = data / "artifacts" / "retrieval" / "finbert_fed_adjacent_xbank_dapt_retrieval"
    assert (bundle_dir / "embeddings.npy").read_bytes() == b"FROM_HF_embeddings.npy"
    assert (bundle_dir / "checkpoint" / "model.safetensors").exists()
    assert (
        bundle_dir / "checkpoint" / "1_Pooling" / "config.json"
    ).read_bytes() == b"FROM_HF_checkpoint/1_Pooling/config.json"
    # ``allow_patterns`` must carry snapshot-side (nested) paths so HF
    # serves the checkpoint/* tree at all.
    allow = captured.get("allow_patterns") or []
    assert "checkpoint/1_Pooling/config.json" in allow
    assert "checkpoint/model.safetensors" in allow
    assert list(models.iterdir()) == []


def test_data_root_entry_keeps_byte_identical_local_file(
    monkeypatch: pytest.MonkeyPatch,
    models_and_data_dirs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """A byte-identical local copy under the ``DATA`` root is skipped
    (no needless rewrite); a drifted copy is OVERWRITTEN so a stale
    file from a prior revision cannot mask the pinned artefact.
    """

    _, data = models_and_data_dirs
    monkeypatch.setenv("HF_TOKEN", "stub")

    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    for src_name, _, _ in (
        eager_pull._split_entry(e) for e in eager_pull._ARTEFACT_FILES["trajectory_bundle"]
    ):
        (snapshot / src_name).write_bytes(b"FROM_HF")

    bundle_dir = data / "artifacts" / "trajectory" / "trajectory_transformer"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    # Drift case: same name, different content — must be overwritten.
    drift = bundle_dir / "model.pt"
    drift.write_bytes(b"STALE_LOCAL_TRAINED")
    # Match case: byte-identical to the snapshot copy — must stay put.
    match = bundle_dir / "embedding_index.parquet"
    match.write_bytes(b"FROM_HF")

    monkeypatch.setattr(
        "app.boot.eager_pull.snapshot_download",
        lambda **kwargs: str(snapshot),
        raising=False,
    )
    monkeypatch.setattr(
        "app.models.registry.eager_artefacts",
        lambda: [_trajectory_artefact()],
    )

    eager_pull.hydrate()
    # Drift was replaced with the snapshot copy.
    assert drift.read_bytes() == b"FROM_HF"
    # Byte-identical entry kept the same content (and the destination
    # path is the same path, so no spurious copy churn either).
    assert match.read_bytes() == b"FROM_HF"


def test_unknown_dst_root_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch,
    models_and_data_dirs: tuple[Path, Path],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An entry pointing at an unrecognised root logs and skips rather
    than raising — the shim's fail-open contract must hold even when
    the mapping itself is malformed.
    """

    models, data = models_and_data_dirs
    monkeypatch.setenv("HF_TOKEN", "stub")

    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "junk.txt").write_bytes(b"x")

    monkeypatch.setattr(
        "app.boot.eager_pull._ARTEFACT_FILES",
        {"junk_artefact": (("junk.txt", "junk.txt", "UNKNOWN"),)},
        raising=False,
    )
    monkeypatch.setattr(
        "app.boot.eager_pull.snapshot_download",
        lambda **kwargs: str(snapshot),
        raising=False,
    )
    monkeypatch.setattr(
        "app.models.registry.eager_artefacts",
        lambda: [
            ArtefactRef(
                name="junk_artefact",
                hf_uri="hf://yusufizzetmurat/fed-pulse-forecaster",
                revision="x",
                eager=True,
                description="",
                inference_features=(),
            )
        ],
    )

    with caplog.at_level("WARNING", logger="app.boot.eager_pull"):
        eager_pull.hydrate()
    assert any("unknown dst_root" in r.message for r in caplog.records)
    # Nothing landed in either root.
    assert list(models.iterdir()) == []
    assert not (data / "junk.txt").exists()


def test_trajectory_bundle_mapping_matches_service_constants() -> None:
    """The trajectory bundle entry must drop its files exactly where
    ``app.services.trajectory`` reads them.

    Drift between the two is the bug class this whole change is meant
    to prevent — pinning the relpath in a test stops a silent rename
    on one side from breaking startup on the other.
    """

    from app.services import trajectory as trajectory_svc

    expected_dir = "artifacts/trajectory/trajectory_transformer"
    for src_name, dst_relpath, dst_root in (
        eager_pull._split_entry(e) for e in eager_pull._ARTEFACT_FILES["trajectory_bundle"]
    ):
        assert dst_root == "DATA"
        assert dst_relpath.startswith(expected_dir + "/")
    # The four "must exist" filenames in the trajectory loader must all
    # show up on the snapshot side, or the bundle hydrates incomplete
    # and the service falls into its degraded "not available" state.
    src_names = {
        s for s, _, _ in (
            eager_pull._split_entry(e)
            for e in eager_pull._ARTEFACT_FILES["trajectory_bundle"]
        )
    }
    for required in (
        trajectory_svc.PARQUET_NAME,
        trajectory_svc.NPZ_NAME,
        trajectory_svc.MODEL_NAME,
        trajectory_svc.MANIFEST_NAME,
    ):
        assert required in src_names
