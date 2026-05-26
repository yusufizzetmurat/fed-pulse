"""ModelConfig.head_mode default after the #322 canonical flip.

ADR 0015 makes regression the canonical training objective for the
vol-regime head; the dataclass default flips from ``"classification"``
to ``"regression"``. ``from_model`` must keep round-tripping explicit
``head_mode`` values off old checkpoints so the back-compat contract
holds. The rates-complex head's mode and the dual-head joint-loss alpha
are unrelated to #322 and stay at their existing defaults.
"""

from __future__ import annotations

from dataclasses import asdict, replace

from app.models.config import ModelConfig


class _StubModelWithHeadMode:
    """Minimal stand-in for a checkpoint-loaded torch module.

    Only the attributes ``ModelConfig.from_model`` reads need to be
    populated. ``head_mode`` carries the value the round-trip is being
    asserted on; everything else is set to the dataclass default so the
    test isolates the head-mode contract.
    """

    head_mode = "classification"

    # Architecture + core sizing.
    model_type = "lstm"
    input_size = 6
    hidden_size = 64
    num_layers = 2
    dropout = 0.15
    head_hidden_size = 32
    initial_decay_rate = 1.5

    # Text path + adapters.
    text_channel = "scalar"
    chunk_projection_dim = 128
    credibility_features = False
    text_embedding_dim = 0
    text_adapter_dim = 0

    # Phase 9 V2 classification target.
    output_mode = "regression"
    n_classes = 3
    vol_regime_quantiles: tuple[float, ...] = ()
    vol_regime_target = "forward_realized_vol_10d"

    # LR schedule, sequence length, decay toggle.
    lr_schedule = "plateau"
    sequence_length = 0
    use_time_decay = True

    # LoRA path.
    encoder_lora = False
    lora_curriculum_freeze_epoch = None

    # Fusion (#235).
    fusion_mode = "concat"
    infonce_lambda = 0.1
    infonce_temperature = 0.07
    infonce_latent_dim = 64

    # Multi-task head (#78, #273).
    multi_task_loss = False
    multi_task_lambda_stance = 1.0
    multi_task_lambda_factor = 0.3
    multi_task_lambda_certainty = 0.3
    multi_task_lambda_topic = 0.3
    class_weight_power = 1.0

    # Dual-head joint-loss alpha (#304).
    regression_alpha = 0.5

    # Derived-text-features ablation (#309).
    use_derived_text_features = True

    # Rates-complex (#292).
    rates_heads: tuple[str, ...] = ()
    rates_head_mode = "regression"
    rates_alpha = 0.5


class _StubModelWithoutHeadMode(_StubModelWithHeadMode):
    """Same as above but with ``head_mode`` removed entirely so
    ``from_model`` exercises the ``getattr`` fallback."""

    # Python's class attribute removal trick: shadow the parent with
    # ``__delattr__`` semantics by not redeclaring it and using a
    # ``__getattr__`` that raises AttributeError on access. Simpler:
    # rebuild the attribute list without head_mode.


def _strip_head_mode(stub_cls: type) -> type:
    """Return a new class identical to ``stub_cls`` but without a
    ``head_mode`` attribute, so ``getattr(model, "head_mode", default)``
    routes through the default branch."""

    fields = {
        name: value
        for name, value in vars(stub_cls).items()
        if not name.startswith("__") and name != "head_mode"
    }
    return type(stub_cls.__name__ + "_NoHeadMode", (), fields)


def test_default_modelconfig_head_mode_is_regression() -> None:
    cfg = ModelConfig()
    assert cfg.head_mode == "regression"


def test_classification_head_mode_round_trips_via_replace_and_asdict() -> None:
    """A caller that opts back into the legacy classification objective
    must be able to dataclass-replace cleanly and dump to a dict whose
    ``head_mode`` survives."""

    cfg = ModelConfig(head_mode="classification")
    assert cfg.head_mode == "classification"

    bumped = replace(cfg, head_mode="classification", regression_alpha=0.5)
    assert bumped.head_mode == "classification"

    dumped = asdict(cfg)
    assert dumped["head_mode"] == "classification"


def test_from_model_preserves_explicit_classification_head_mode() -> None:
    """Pre-#322 checkpoints carrying ``head_mode='classification'`` on the
    stashed module must round-trip back into the dataclass unchanged."""

    cfg = ModelConfig.from_model(_StubModelWithHeadMode())
    assert cfg.head_mode == "classification"


def test_from_model_falls_back_to_regression_when_attribute_missing() -> None:
    """A freshly built model (or a stub without the attribute) must
    surface the new ``regression`` default rather than the pre-#322
    ``classification`` fallback."""

    stub_cls = _strip_head_mode(_StubModelWithHeadMode)
    cfg = ModelConfig.from_model(stub_cls())
    assert cfg.head_mode == "regression"


def test_rates_head_mode_default_unchanged_by_322() -> None:
    """The rates-complex head's regression default predates #322 and is
    out of scope; the flip must not perturb it."""

    cfg = ModelConfig()
    assert cfg.rates_head_mode == "regression"


def test_regression_alpha_default_unchanged_by_322() -> None:
    """The dual-head joint-loss mixing weight stays at 0.5 (the balanced
    starting point for the comparison sweep). #322 only flips the
    objective default, not the loss-term blend."""

    cfg = ModelConfig()
    assert cfg.regression_alpha == 0.5
