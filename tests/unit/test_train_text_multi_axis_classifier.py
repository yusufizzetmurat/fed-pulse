"""Bundle A.1.5: per-run HF-format encoder save on the multi-axis trainer.

Pins that the trainer's best-checkpoint save path emits a HF
directory containing the encoder backbone (NOT the
``TextMultiAxisClassifier`` wrapper) plus the tokenizer, so the
registry / embedding-cache builder / forecaster can load it via
``AutoModel.from_pretrained`` regardless of which trainer produced
it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from app.data.train_text_multi_axis_classifier import (  # noqa: E402
    _save_hf_encoder_directory,
)


class _RecordingEncoder(torch.nn.Module):
    """Minimal stand-in for a HF encoder backbone.

    ``save_pretrained`` is the only surface ``_save_hf_encoder_directory``
    touches on the encoder; the recording variant writes a sentinel
    ``config.json`` and ``model.safetensors`` so the artifact directory
    looks like a real HF checkpoint to downstream callers (and to the
    standard-files assertion below). The recording also pins WHICH
    object got ``save_pretrained`` called on it, which the trainer's
    main-path test uses to assert the encoder backbone was saved
    rather than the ``TextMultiAxisClassifier`` wrapper.
    """

    save_calls: list[str]

    def __init__(self) -> None:
        super().__init__()
        self._param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.save_calls = []

    def save_pretrained(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"\x00")
        self.save_calls.append(str(path))


class _RecordingTokenizer:
    """Tokenizer stand-in with the same ``save_pretrained`` contract."""

    save_calls: list[str]

    def __init__(self) -> None:
        self.save_calls = []

    def save_pretrained(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        self.save_calls.append(str(path))


class _ClassifierWithEncoder(torch.nn.Module):
    """Mimics the ``TextMultiAxisClassifier.encoder`` attribute layout.

    The helper under test reads ``model.encoder.save_pretrained`` —
    NOT ``model.save_pretrained`` — so the head weights stay out of
    the HF directory the registry consumes.
    """

    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        # A dummy "head" parameter so any accidental save of the whole
        # wrapper would be detectable in the resulting state_dict.
        self.head = torch.nn.Linear(1, 1)

    def save_pretrained(self, directory: str) -> None:  # pragma: no cover
        # Wrapper-level save MUST NOT be called by
        # ``_save_hf_encoder_directory``. The body raises so any
        # regression that swaps ``model.encoder.save_pretrained`` for
        # ``model.save_pretrained`` fails the test loudly.
        raise AssertionError(
            "TextMultiAxisClassifier.save_pretrained called — the HF "
            "save path must save only the encoder backbone."
        )


def test_save_hf_encoder_directory_writes_standard_hf_files(
    tmp_path: Path,
) -> None:
    """The helper writes a directory containing the standard HF files
    the registry's ``AutoModel.from_pretrained`` / ``AutoTokenizer``
    pair expects."""

    encoder = _RecordingEncoder()
    tokenizer = _RecordingTokenizer()
    model = _ClassifierWithEncoder(encoder)
    checkpoint_dir = tmp_path / "hf_checkpoints"

    _save_hf_encoder_directory(model, tokenizer, checkpoint_dir)

    assert checkpoint_dir.is_dir()
    files = {p.name for p in checkpoint_dir.iterdir()}
    # Encoder side: config + one of the two HF weight conventions.
    assert "config.json" in files
    assert "model.safetensors" in files or "pytorch_model.bin" in files
    # Tokenizer side: at least one of the standard tokenizer files.
    assert "tokenizer.json" in files or "tokenizer_config.json" in files


def test_save_hf_encoder_directory_saves_encoder_backbone_not_wrapper(
    tmp_path: Path,
) -> None:
    """The helper must call ``save_pretrained`` on the encoder backbone,
    not on the ``TextMultiAxisClassifier`` wrapper — otherwise the
    multi-task head would land in the HF directory and the registry's
    ``AutoModel.from_pretrained`` would either fail or silently load
    a head-contaminated state dict.

    The wrapper's ``save_pretrained`` raises on call, so any regression
    that swaps the receiver flips this test from green to red.
    """

    encoder = _RecordingEncoder()
    tokenizer = _RecordingTokenizer()
    model = _ClassifierWithEncoder(encoder)
    checkpoint_dir = tmp_path / "hf_checkpoints"

    _save_hf_encoder_directory(model, tokenizer, checkpoint_dir)

    # The encoder backbone (NOT the wrapper) received the save call.
    assert encoder.save_calls == [str(checkpoint_dir)]
    # And the tokenizer was saved to the same directory.
    assert tokenizer.save_calls == [str(checkpoint_dir)]


def test_save_hf_encoder_directory_creates_parent_directories(
    tmp_path: Path,
) -> None:
    """The helper must ``mkdir(parents=True, exist_ok=True)`` so a
    caller can pass a nested artifact path without pre-creating the
    intermediate directories (the trainer's main path builds
    ``{artifact_root}/text_multi_axis_{run_token}/hf_checkpoints/``
    in one shot)."""

    encoder = _RecordingEncoder()
    tokenizer = _RecordingTokenizer()
    model = _ClassifierWithEncoder(encoder)
    checkpoint_dir = tmp_path / "artifacts" / "text_multi_axis_2026XYZ" / "hf_checkpoints"

    _save_hf_encoder_directory(model, tokenizer, checkpoint_dir)

    assert checkpoint_dir.is_dir()


def test_default_artifact_root_lives_under_data_dir() -> None:
    """The default artifact root mirrors ``finetune_pilot``'s convention
    of writing under ``DATA_DIR / "artifacts" / "<trainer-tag>"`` so
    future tooling can find HF dirs the same way regardless of which
    trainer produced them."""

    from app.config import DATA_DIR
    from app.data.train_text_multi_axis_classifier import DEFAULT_ARTIFACT_ROOT

    assert DEFAULT_ARTIFACT_ROOT == DATA_DIR / "artifacts" / "text_multi_axis"


def test_artifact_root_cli_flag_is_overridable() -> None:
    """The ``--artifact-root`` flag must be parseable and override the
    default — needed for CI / job-runner contexts that write to a
    sandboxed artifact directory."""

    from app.data.train_text_multi_axis_classifier import _parse_args

    args = _parse_args(
        [
            "--artifact-root",
            "/tmp/custom_artifact_root",
        ]
    )
    assert args.artifact_root == "/tmp/custom_artifact_root"
