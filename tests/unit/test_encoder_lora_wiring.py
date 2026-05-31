"""Round 5 (#244) source-level tests for the LoRA + in-loop FinBERT path.

The full LoRA training run needs ``peft`` + ``transformers`` + a GPU
and is too expensive for CI. These tests cover the wiring around the
LoRA path: the config field exists with the right default, the
``--encoder-lora`` CLI flag flows into the candidate ``ModelConfig``
construction, the factory drops the field before reaching
``ForecasterModel`` (which does not consume it), and the
``encoder_lora.py`` helper module exposes the documented surface.
The byte-identity guarantee for ``encoder_lora=False`` rides on the
existing determinism regression at
``tests/regression/test_forecaster_determinism.py`` (its synthetic
vectors carry no ``raw_text`` and the static-cache path runs
unchanged).
"""

from __future__ import annotations

from pathlib import Path


from app import __file__ as _app_init_path

_APP_DIR = Path(_app_init_path).parent


def _read(rel_path: str) -> str:
    # Accepts both "models/config.py" and the legacy "backend/app/models/config.py"
    # prefix. Resolves via the live ``app`` package so the same path works whether
    # the test runs from the host repo root or inside the backend container (where
    # ``backend/`` is the mount root and there is no nested ``backend/app/`` tree).
    if rel_path.startswith("backend/app/"):
        rel_path = rel_path[len("backend/app/"):]
    return (_APP_DIR / rel_path).read_text(encoding="utf-8")


def test_model_config_carries_encoder_lora_field() -> None:
    """``ModelConfig.encoder_lora`` must be a real field with default
    ``False`` so existing checkpoints deserialise into the legacy
    static-cache path."""

    source = _read("backend/app/models/config.py")
    assert "encoder_lora: bool = False" in source, (
        "ModelConfig is missing the encoder_lora field (default False)"
    )
    assert "encoder_lora=bool(getattr(model, \"encoder_lora\", False))" in source, (
        "ModelConfig.from_model does not round-trip encoder_lora"
    )


def test_feature_vector_carries_raw_text_field() -> None:
    """``FeatureVector.raw_text`` is populated by the loader on the
    target-row bar of each sequence when ``encoder_lora=True``. The
    LoRA tokeniser reads ``sequence[-1].raw_text`` per group."""

    source = _read("backend/app/models/config.py")
    assert "raw_text: str = \"\"" in source, (
        "FeatureVector is missing the raw_text field (default empty)"
    )


def test_factory_pops_encoder_lora_before_forecaster_model() -> None:
    """``ForecasterModel`` does not accept ``encoder_lora`` as a kwarg
    -- the factory strips it before constructing the model. The value
    is then stamped onto the built module so
    ``ModelConfig.from_model`` reads it back faithfully when the run
    summary is serialised (otherwise every Round 5 LoRA cell's
    persisted metadata lies about whether LoRA was active)."""

    source = _read("backend/app/models/factory.py")
    assert "kwargs.pop(\"encoder_lora\"" in source, (
        "factory does not pop encoder_lora before ForecasterModel(**kwargs)"
    )
    assert "model.encoder_lora =" in source, (
        "factory pops encoder_lora but does not re-stamp it on the model; "
        "from_model() will read back False"
    )


def test_train_forecaster_exposes_encoder_lora_cli_flag() -> None:
    """The CLI must expose ``--encoder-lora`` so the ceiling probe can
    be driven from a wrapper script."""

    source = _read("backend/app/train_forecaster.py")
    assert "\"--encoder-lora\"" in source
    assert "dest=\"encoder_lora\"" in source
    assert "parser.set_defaults(encoder_lora=False)" in source


def test_train_forecaster_threads_encoder_lora_into_model_config() -> None:
    """``encoder_lora`` rides on ``ModelConfig`` so the checkpoint
    persists it. The CLI flag must reach every ``ModelConfig(...)``
    construction site so the sweep + single-run paths honour it."""

    source = _read("backend/app/train_forecaster.py")
    occurrences = source.count(
        "encoder_lora=bool(getattr(args, \"encoder_lora\", False))"
    )
    assert occurrences >= 3, (
        "encoder_lora is not threaded into every ModelConfig construction "
        f"site (found {occurrences}, expected >= 3)"
    )


def test_loaders_thread_encoder_lora_into_metadata_pass() -> None:
    """The loader must accept ``encoder_lora`` and populate
    ``vector.raw_text`` on the target-row bar when the flag is set."""

    source = _read("backend/app/training/loaders.py")
    assert "encoder_lora: bool = False" in source
    assert "vectors[-1].raw_text = row_text" in source, (
        "loader does not populate raw_text on the target-row bar"
    )


def test_encoder_lora_module_exposes_documented_surface() -> None:
    """The helper module exports ``build_lora_encoder``,
    ``tokenize_sequence_texts``, ``encode_batch_pooled``, and the
    ``LoraEncoderBundle`` container. Each is called from
    ``train_model`` -- a rename in the helper without a corresponding
    update in the loop would break the LoRA training path silently."""

    source = _read("backend/app/training/encoder_lora.py")
    for symbol in (
        "def build_lora_encoder(",
        "def tokenize_sequence_texts(",
        "def encode_batch_pooled(",
        "class LoraEncoderBundle",
    ):
        assert symbol in source, f"encoder_lora module missing symbol: {symbol}"


def test_train_model_builds_lora_bundle_only_when_active() -> None:
    """``train_model`` builds the LoRA bundle once before the partition
    tensors and threads it everywhere. When ``encoder_lora`` is off
    the bundle stays ``None`` and the static-cache path runs."""

    source = _read("backend/app/training/loop.py")
    assert "encoder_lora_active = bool(" in source
    assert "encoder_lora_bundle: Any = None" in source
    assert "build_lora_encoder(str(text_encoder))" in source
    # Bundle is threaded into all three partition builders + every
    # _evaluate_model call site.
    assert source.count("lora_bundle=encoder_lora_bundle") >= 3
    assert source.count("encoder_lora_bundle=encoder_lora_bundle") >= 3
