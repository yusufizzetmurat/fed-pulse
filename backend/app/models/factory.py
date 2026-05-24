"""Forecaster architecture factory.

The six architectures (``lstm``, ``lstm_attn``, ``gru``, ``tcn``,
``transformer``, ``dlinear``) are all wired through the same
:class:`app.models.lstm.ForecasterModel` wrapper via its ``model_type``
constructor argument — the wrapper handles per-architecture core selection
plus the shared optional ``TimeDecayAttention`` /
``RecurrentSequenceAttention`` / ``ChunkAttentionPooler`` /
credibility-features paths. The factory exists so callers (sweep harness,
training loop, tests) get a single dispatch entry point that does not have
to know the wrapper's internal kwarg layout, and so the public
``ModelConfig.architecture`` field is the only thing they have to set.

The default ``architecture="lstm"`` path is byte-identical to the previous
``ForecasterModel()`` construction so the determinism regression at
``tests/regression/test_forecaster_determinism.py`` stays green.

When ``config.fusion_mode == "gated_infonce"`` (#235), the factory
dispatches to :class:`MultiModalForecasterModel` instead — the two
modalities stay separate until the gated fusion stage so the InfoNCE
alignment loss has separable representations to pull together.
"""

from __future__ import annotations

from typing import Any

from app.models.config import FORECASTER_ARCHITECTURES, ModelConfig
from app.models.lstm import ForecasterModel
from app.models.multimodal_forecaster import MultiModalForecasterModel


def build_forecaster(
    config: ModelConfig | dict[str, Any],
) -> ForecasterModel | MultiModalForecasterModel:
    """Build a forecaster module for ``config.architecture``.

    All architectures share the same input contract ``(batch, seq_len, 6)``
    and output contract ``(batch, 2)`` (close, volatility).  The wrapper
    dispatches the recurrent core internally via ``model_type``; this
    function exists to translate the public ``architecture`` field to the
    wrapper's internal ``model_type`` argument and to raise a clear error
    when the architecture string is unknown.
    """

    resolved = config if isinstance(config, ModelConfig) else ModelConfig(**config)
    architecture = resolved.architecture
    if architecture not in FORECASTER_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture: {architecture!r}. "
            f"Allowed: {sorted(FORECASTER_ARCHITECTURES)}"
        )

    if resolved.fusion_mode == "gated_infonce":
        if resolved.output_mode != "classification":
            raise ValueError(
                "fusion_mode='gated_infonce' requires output_mode='classification' "
                f"(got {resolved.output_mode!r}); InfoNCE-paired regression is out of scope."
            )
        text_dim = int(resolved.text_embedding_dim or 0)
        if text_dim <= 0:
            raise ValueError(
                "fusion_mode='gated_infonce' requires text_embedding_dim > 0; "
                "set --text-encoder so the loader emits pooled embeddings."
            )
        model = MultiModalForecasterModel(
            market_input_size=int(resolved.input_size),
            text_embedding_dim=text_dim,
            latent_dim=int(resolved.infonce_latent_dim),
            hidden_size=int(resolved.hidden_size),
            num_layers=int(resolved.num_layers),
            dropout=float(resolved.dropout),
            head_hidden_size=int(resolved.head_hidden_size),
            architecture=architecture,
            n_classes=int(resolved.n_classes),
        )
        # Stash the InfoNCE hyperparameters on the module so
        # ``ModelConfig.from_model`` round-trips them and the training
        # loop can read them without an extra config plumbing pass.
        model.fusion_mode = "gated_infonce"  # type: ignore[assignment]
        model.infonce_lambda = float(resolved.infonce_lambda)  # type: ignore[assignment]
        model.infonce_temperature = float(resolved.infonce_temperature)  # type: ignore[assignment]
        model.infonce_latent_dim = int(resolved.infonce_latent_dim)  # type: ignore[assignment]
        model.text_embedding_dim = text_dim  # type: ignore[assignment]
        return model

    kwargs = resolved.to_dict()
    kwargs.pop("architecture", None)
    # Multi-modal-only fields stay on the ModelConfig for round-tripping
    # but the legacy ForecasterModel constructor does not consume them.
    kwargs.pop("fusion_mode", None)
    kwargs.pop("infonce_lambda", None)
    kwargs.pop("infonce_temperature", None)
    kwargs.pop("infonce_latent_dim", None)
    # Phase 9 V2 (#195) fields all forwarded: ``output_mode`` /
    # ``n_classes`` drive the head shape; ``vol_regime_quantiles`` /
    # ``vol_regime_target`` ride on the module so the checkpoint
    # round-trips the per-fold boundaries via ``ModelConfig.from_model``.
    # Phase B (#227) fields ride on the ``ModelConfig`` so the checkpoint
    # carries the schedule + sequence-length choice into resume, but the
    # underlying ``ForecasterModel`` does not consume them on
    # construction (the training loop reads them from the config).
    kwargs.pop("lr_schedule", None)
    kwargs.pop("sequence_length", None)
    # Round 5 (#244) LoRA toggle is a training-loop concern; the
    # ``ForecasterModel`` consumes its text input from a per-batch
    # tensor regardless of whether the encoder ran statically (parquet
    # cache) or per-batch (LoRA-wrapped tower). The constructor does
    # not need to know about it, but we stash the value as a plain
    # attribute on the built module so ``ModelConfig.from_model``
    # (called when the run summary is serialised) reflects what
    # actually trained — without it, every persisted summary shows
    # ``encoder_lora=False`` even on an active LoRA cell.
    encoder_lora_flag = bool(kwargs.pop("encoder_lora", False))
    model = ForecasterModel(model_type=architecture, **kwargs)
    # mypy reads ``nn.Module`` attribute writes as ``Tensor | Module``;
    # the LoRA flag is a plain bool stashed for ``from_model`` to read
    # back, so suppress the noise rather than register a fake buffer.
    model.encoder_lora = encoder_lora_flag  # type: ignore[assignment]
    return model


__all__ = ["build_forecaster"]
