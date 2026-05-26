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
        multimodal = MultiModalForecasterModel(
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
        multimodal.fusion_mode = "gated_infonce"  # type: ignore[assignment]
        multimodal.infonce_lambda = float(resolved.infonce_lambda)  # type: ignore[assignment]
        multimodal.infonce_temperature = float(resolved.infonce_temperature)  # type: ignore[assignment]
        multimodal.infonce_latent_dim = int(resolved.infonce_latent_dim)  # type: ignore[assignment]
        return multimodal

    kwargs = resolved.to_dict()
    kwargs.pop("architecture", None)
    # Multi-modal-only fields stay on the ModelConfig for round-tripping
    # but the legacy ForecasterModel constructor does not consume them.
    kwargs.pop("fusion_mode", None)
    kwargs.pop("infonce_lambda", None)
    kwargs.pop("infonce_temperature", None)
    kwargs.pop("infonce_latent_dim", None)
    # Multi-task loss (#273) is a training-loop concern; the existing
    # MultiTaskHead is already wired on the ForecasterModel (since #272)
    # in classification mode. These knobs stash on the built module so
    # ``ModelConfig.from_model`` recovers them when the checkpoint is
    # loaded for resume / inference.
    multi_task_loss_flag = bool(kwargs.pop("multi_task_loss", False))
    multi_task_lambda_stance = float(kwargs.pop("multi_task_lambda_stance", 1.0))
    multi_task_lambda_factor = float(kwargs.pop("multi_task_lambda_factor", 0.3))
    multi_task_lambda_certainty = float(kwargs.pop("multi_task_lambda_certainty", 0.3))
    multi_task_lambda_topic = float(kwargs.pop("multi_task_lambda_topic", 0.3))
    class_weight_power = float(kwargs.pop("class_weight_power", 1.0))
    # #304 dual-head methodology. ``regression_alpha`` is a loss-side
    # knob (the training loop reads it from the model attribute);
    # ``head_mode`` is forwarded to the ForecasterModel constructor so
    # the regression_head mounts correctly. ``use_derived_text_features``
    # (#309) is a loader-side flag, only stashed on the module for
    # round-tripping through ModelConfig.from_model.
    regression_alpha_value = float(kwargs.pop("regression_alpha", 0.5))
    use_derived_text_features_flag = bool(
        kwargs.pop("use_derived_text_features", True)
    )
    # #292 rates heads. ``rates_heads`` is a tuple of head short-names
    # forwarded to the ForecasterModel constructor so the rates regression
    # + aux classification heads mount on the shared encoder. The mode +
    # alpha knobs are loss-side concerns (the training loop reads them
    # off the stashed module attribute); stash them so
    # ``ModelConfig.from_model`` round-trips them onto the persisted
    # summary regardless of whether the run also wires the loss.
    rates_heads_tuple = tuple(
        str(v) for v in kwargs.pop("rates_heads", ()) or ()
    )
    rates_head_mode_value = str(
        kwargs.pop("rates_head_mode", "regression") or "regression"
    )
    rates_alpha_value = float(kwargs.pop("rates_alpha", 0.5))
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
    # Bundle B LoRA freeze curriculum is a training-loop concern too;
    # the recurrent ``ForecasterModel`` constructor never sees it. The
    # value is stashed back on the built module so ``ModelConfig.from_model``
    # round-trips it onto the persisted run summary the same way
    # ``encoder_lora`` does.
    lora_curriculum_freeze_epoch_val = kwargs.pop(
        "lora_curriculum_freeze_epoch", None
    )
    model = ForecasterModel(
        model_type=architecture,
        rates_heads=rates_heads_tuple,
        **kwargs,
    )
    # mypy reads ``nn.Module`` attribute writes as ``Tensor | Module``;
    # the LoRA flag is a plain bool stashed for ``from_model`` to read
    # back, so suppress the noise rather than register a fake buffer.
    model.encoder_lora = encoder_lora_flag  # type: ignore[assignment]
    model.lora_curriculum_freeze_epoch = lora_curriculum_freeze_epoch_val
    model.multi_task_loss = multi_task_loss_flag  # type: ignore[assignment]
    model.multi_task_lambda_stance = multi_task_lambda_stance  # type: ignore[assignment]
    model.multi_task_lambda_factor = multi_task_lambda_factor  # type: ignore[assignment]
    model.multi_task_lambda_certainty = multi_task_lambda_certainty  # type: ignore[assignment]
    model.multi_task_lambda_topic = multi_task_lambda_topic  # type: ignore[assignment]
    model.class_weight_power = class_weight_power  # type: ignore[assignment]
    # #304 / #309 -- stash the loss + loader flags so
    # ``ModelConfig.from_model`` round-trips them onto the persisted
    # checkpoint payload. ``head_mode`` itself was already passed to
    # the ForecasterModel constructor above; the alpha + derived-text
    # flag are training-loop / loader concerns and stay as attributes.
    model.regression_alpha = regression_alpha_value  # type: ignore[assignment]
    model.use_derived_text_features = use_derived_text_features_flag  # type: ignore[assignment]
    # #292 stash the loss-side knobs so ``ModelConfig.from_model`` round-
    # trips them onto the persisted run summary.
    model.rates_heads = rates_heads_tuple  # type: ignore[assignment]
    model.rates_head_mode = rates_head_mode_value  # type: ignore[assignment]
    model.rates_alpha = rates_alpha_value  # type: ignore[assignment]
    return model


__all__ = ["build_forecaster"]
