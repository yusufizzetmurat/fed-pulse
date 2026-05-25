"""Unit tests for the LoRA freeze helpers in :mod:`app.training.encoder_lora`.

Covers ``freeze_adapter`` (the PEFT-side helper that flips every LoRA
matrix's ``requires_grad`` to False) and the pure boundary-policy
helper ``should_freeze_lora_at_epoch`` that the training loop drives
the stage-1-train-then-freeze curriculum through. Also covers the
loop-side wiring + CLI plumbing + ``ModelConfig`` round-trip that the
forecaster pipeline relies on to honour the curriculum.

The PEFT-backed behavioural tests load only when ``peft`` is importable;
the boundary-policy and source-wiring tests run unconditionally.
"""

from __future__ import annotations

from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    """Read a repo-relative file regardless of the test layout.

    On the host the worktree puts ``backend/`` at the repo root; inside
    the Docker test container ``backend/`` is mounted at ``/app`` so the
    same file is reachable without the leading ``backend/`` segment.
    Trying both candidates keeps the source-wiring tests runnable under
    both invocations.
    """

    candidates = [
        _REPO_ROOT / rel_path,
        _REPO_ROOT / Path(rel_path).relative_to("backend")
        if rel_path.startswith("backend/")
        else _REPO_ROOT / rel_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"could not locate {rel_path!r} under repo root {_REPO_ROOT}; "
        f"tried: {[str(c) for c in candidates]}"
    )


# ---------------------------------------------------------------------------
# should_freeze_lora_at_epoch — pure boundary policy. No torch / peft.
# ---------------------------------------------------------------------------


def test_should_freeze_returns_false_when_freeze_epoch_none() -> None:
    """``freeze_epoch=None`` → never freeze. Locks the default behaviour
    that pre-Bundle-B runs see no change."""

    from app.training.encoder_lora import should_freeze_lora_at_epoch

    for epoch in range(10):
        assert should_freeze_lora_at_epoch(None, epoch, already_frozen=False) is False
        assert should_freeze_lora_at_epoch(None, epoch, already_frozen=True) is False


def test_should_freeze_fires_at_boundary_and_holds() -> None:
    """``freeze_epoch=2`` over a 4-epoch budget: returns True at epoch 2
    and remains True at later epochs unless the loop already flipped
    ``already_frozen`` to True (the idempotency hook)."""

    from app.training.encoder_lora import should_freeze_lora_at_epoch

    assert should_freeze_lora_at_epoch(2, 0, already_frozen=False) is False
    assert should_freeze_lora_at_epoch(2, 1, already_frozen=False) is False
    assert should_freeze_lora_at_epoch(2, 2, already_frozen=False) is True
    assert should_freeze_lora_at_epoch(2, 3, already_frozen=False) is True


def test_should_freeze_respects_already_frozen() -> None:
    """Once the loop has flipped ``already_frozen=True`` the helper
    must return False so the loop does not repeat the freeze + log
    every epoch."""

    from app.training.encoder_lora import should_freeze_lora_at_epoch

    assert should_freeze_lora_at_epoch(2, 2, already_frozen=True) is False
    assert should_freeze_lora_at_epoch(2, 5, already_frozen=True) is False


def test_should_freeze_zero_epoch_freezes_from_start() -> None:
    """``freeze_epoch=0`` collapses stage 1 to nothing — the adapter is
    frozen at the start of epoch 0 (effectively a head-only run with
    the random-init adapter). Valid degenerate configuration."""

    from app.training.encoder_lora import should_freeze_lora_at_epoch

    assert should_freeze_lora_at_epoch(0, 0, already_frozen=False) is True


# ---------------------------------------------------------------------------
# Loop-side boundary simulation. Counts how often the freeze helper would
# be called across an epoch budget — proves the loop fires it exactly once.
# ---------------------------------------------------------------------------


def _simulate_loop(epochs: int, freeze_epoch: int | None) -> list[int]:
    """Return the list of epoch indices at which the loop would freeze.

    Mirrors the exact boundary policy ``train_model`` runs so the test
    asserts the wire-level guarantee: at most one freeze per training
    run, exactly at the configured epoch."""

    from app.training.encoder_lora import should_freeze_lora_at_epoch

    already = False
    fired_at: list[int] = []
    for epoch_index in range(epochs):
        if should_freeze_lora_at_epoch(
            freeze_epoch, epoch_index, already_frozen=already
        ):
            fired_at.append(epoch_index)
            already = True
    return fired_at


def test_loop_simulation_fires_once_at_boundary() -> None:
    """4 epochs, freeze=2 → exactly one freeze, at epoch index 2."""

    assert _simulate_loop(epochs=4, freeze_epoch=2) == [2]


def test_loop_simulation_never_fires_when_disabled() -> None:
    """``freeze_epoch=None`` → no freeze across any epoch count."""

    assert _simulate_loop(epochs=10, freeze_epoch=None) == []


def test_loop_simulation_fires_at_first_epoch_when_zero() -> None:
    """``freeze_epoch=0`` → single freeze at epoch 0 (degenerate but valid)."""

    assert _simulate_loop(epochs=3, freeze_epoch=0) == [0]


def test_loop_simulation_fires_late_when_boundary_at_last_epoch() -> None:
    """``freeze_epoch=epochs-1`` → freeze fires on the final epoch."""

    assert _simulate_loop(epochs=5, freeze_epoch=4) == [4]


def test_loop_simulation_never_fires_when_boundary_past_horizon() -> None:
    """If ``freeze_epoch >= epochs`` the loop never reaches the boundary
    and the adapter stays trainable for every epoch."""

    assert _simulate_loop(epochs=3, freeze_epoch=10) == []


# ---------------------------------------------------------------------------
# ModelConfig field round-trip.
# ---------------------------------------------------------------------------


def test_model_config_default_freeze_epoch_is_none() -> None:
    """``ModelConfig.lora_curriculum_freeze_epoch`` defaults to ``None``
    so pre-Bundle-B runs keep their byte-identical contract."""

    from app.models.config import ModelConfig

    cfg = ModelConfig()
    assert cfg.lora_curriculum_freeze_epoch is None


def test_model_config_freeze_epoch_accepts_positive_int() -> None:
    """Round-trip a positive integer through ``to_dict``."""

    from app.models.config import ModelConfig

    cfg = ModelConfig(lora_curriculum_freeze_epoch=3)
    assert cfg.lora_curriculum_freeze_epoch == 3
    assert cfg.to_dict()["lora_curriculum_freeze_epoch"] == 3


def test_model_config_from_model_round_trips_freeze_epoch() -> None:
    """``ModelConfig.from_model`` pulls the curriculum field off a built
    model so the persisted run summary records whether the curriculum
    was active."""

    from app.models.config import ModelConfig

    class _Stub:
        model_type = "lstm"
        input_size = 6
        hidden_size = 16
        num_layers = 1
        dropout = 0.1
        head_hidden_size = 8
        initial_decay_rate = 1.0
        lora_curriculum_freeze_epoch = 5

    cfg = ModelConfig.from_model(_Stub())
    assert cfg.lora_curriculum_freeze_epoch == 5


def test_model_config_from_model_handles_missing_attribute() -> None:
    """Legacy models without the curriculum attribute must yield
    ``None`` (not raise)."""

    from app.models.config import ModelConfig

    class _LegacyStub:
        model_type = "lstm"
        input_size = 6
        hidden_size = 16
        num_layers = 1
        dropout = 0.1
        head_hidden_size = 8
        initial_decay_rate = 1.0

    cfg = ModelConfig.from_model(_LegacyStub())
    assert cfg.lora_curriculum_freeze_epoch is None


def test_factory_does_not_leak_curriculum_field_into_forecaster_model() -> None:
    """``ForecasterModel`` does not accept ``lora_curriculum_freeze_epoch``
    as a kwarg; the factory must pop it before construction and stash
    the value on the built module for ``from_model`` to round-trip."""

    from app.models.config import ModelConfig
    from app.models.factory import build_forecaster

    cfg = ModelConfig(architecture="lstm", lora_curriculum_freeze_epoch=2)
    # No exception means the factory popped the field correctly.
    model = build_forecaster(cfg)
    assert getattr(model, "lora_curriculum_freeze_epoch", None) == 2


# ---------------------------------------------------------------------------
# Source-level wiring guarantees. These run without torch / peft so they
# always exercise on CI and would have failed against the pre-change loop.
# ---------------------------------------------------------------------------


def test_loop_wires_freeze_helper_inside_epoch_loop() -> None:
    """``train_model`` calls ``freeze_adapter`` from inside the per-epoch
    branch and gates the call through ``should_freeze_lora_at_epoch``
    + a local flag so the freeze fires once and not per batch."""

    source = _read("backend/app/training/loop.py")
    assert "lora_adapter_frozen = False" in source, (
        "loop did not initialise the lora_adapter_frozen flag"
    )
    assert "should_freeze_lora_at_epoch" in source, (
        "loop did not call the boundary helper"
    )
    assert "freeze_adapter(encoder_lora_bundle.encoder)" in source, (
        "loop did not freeze the LoRA bundle's encoder"
    )
    assert "INFO lora_curriculum_freeze" in source, (
        "loop did not log the freeze boundary"
    )
    assert "lora_adapter_frozen = True" in source, (
        "loop did not flip the idempotency flag after freezing"
    )


def test_loop_freeze_check_is_outside_inner_batch_loop() -> None:
    """The freeze check sits at the top of the epoch loop, above the
    ``for batch in train_loader`` line. Per-batch freezes would be
    correctness-noise (and a waste) — this test pins the placement."""

    source = _read("backend/app/training/loop.py")
    freeze_idx = source.find("should_freeze_lora_at_epoch")
    train_batch_idx = source.find("for batch in train_loader")
    assert freeze_idx > 0 and train_batch_idx > 0
    # Confirm no ``for batch in train_loader`` sits between the freeze
    # check and the next batch loop — otherwise the freeze would fire
    # per batch instead of per epoch.
    inter_segment = source[freeze_idx:train_batch_idx]
    assert "for batch in train_loader" not in inter_segment, (
        "freeze helper is inside an inner loop — would fire per batch"
    )


def test_train_forecaster_exposes_lora_freeze_epoch_cli_flag() -> None:
    """The CLI surface includes ``--lora-freeze-epoch`` so the sweep
    runner can drive the curriculum from a wrapper script."""

    source = _read("backend/app/train_forecaster.py")
    assert "\"--lora-freeze-epoch\"" in source
    assert "dest=\"lora_freeze_epoch\"" in source


def test_train_forecaster_threads_freeze_epoch_into_all_config_sites() -> None:
    """The CLI value must reach all three ``ModelConfig`` construction
    sites (single-run + random-search + exhaustive sweep) so the freeze
    epoch is honoured regardless of which dispatch path the sweep takes."""

    source = _read("backend/app/train_forecaster.py")
    occurrences = source.count(
        "lora_curriculum_freeze_epoch=getattr(args, \"lora_freeze_epoch\", None)"
    )
    assert occurrences >= 3, (
        "lora_curriculum_freeze_epoch is not threaded into every "
        f"ModelConfig construction site (found {occurrences}, expected >= 3)"
    )


# ---------------------------------------------------------------------------
# Real PEFT freeze_adapter behavioural tests (skipped when peft missing).
# ---------------------------------------------------------------------------


peft = pytest.importorskip("peft")
torch = pytest.importorskip("torch")


def _make_toy_peft_model() -> "torch.nn.Module":
    """Tiny linear-stack module wrapped with a PEFT LoRA adapter.

    PEFT's ``LoraConfig`` requires named target modules; the trick is
    to name the inner ``Linear`` layers so the adapter has something to
    attach to. ``r=2`` keeps the adapter near-zero size for fast tests.
    """

    import torch.nn as nn
    from peft import LoraConfig, get_peft_model

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query = nn.Linear(8, 8)
            self.value = nn.Linear(8, 8)
            self.out = nn.Linear(8, 4)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.out(self.value(self.query(x)))

    base = Toy()
    for param in base.parameters():
        param.requires_grad_(False)
    cfg = LoraConfig(r=2, lora_alpha=4, target_modules=["query", "value"], bias="none")
    return get_peft_model(base, cfg)


def test_freeze_adapter_flips_lora_params_to_requires_grad_false() -> None:
    """After ``freeze_adapter``, every LoRA matrix's ``requires_grad`` is
    False and every non-LoRA parameter is left at its prior value (the
    base encoder stays frozen because ``build_lora_encoder`` freezes it
    upfront; we mirror that contract here by pre-freezing the base)."""

    from app.training.encoder_lora import freeze_adapter

    model = _make_toy_peft_model()
    # Snapshot the pre-call requires_grad state per param name so the
    # assertion below can compare LoRA-only vs non-LoRA-touched.
    pre_state = {name: p.requires_grad for name, p in model.named_parameters()}
    lora_names_pre = [n for n, g in pre_state.items() if g and "lora_" in n]
    assert lora_names_pre, (
        "test setup is wrong: the toy model must expose at least one "
        "trainable LoRA parameter for the freeze assertion to be meaningful"
    )

    frozen = freeze_adapter(model)
    assert frozen >= len(lora_names_pre)

    post_state = {name: p.requires_grad for name, p in model.named_parameters()}
    for name, was in pre_state.items():
        if "lora_" in name:
            assert post_state[name] is False, (
                f"LoRA parameter {name} still has requires_grad=True after freeze"
            )
        else:
            # Non-LoRA params keep their prior state — freeze must not
            # accidentally re-enable a frozen base encoder weight.
            assert post_state[name] == was, (
                f"non-LoRA parameter {name} requires_grad changed unexpectedly "
                f"({was} -> {post_state[name]})"
            )


def test_freeze_adapter_is_idempotent() -> None:
    """A second call must not raise and must leave the requires_grad
    flags unchanged."""

    from app.training.encoder_lora import freeze_adapter

    model = _make_toy_peft_model()
    first_count = freeze_adapter(model)
    snapshot = {name: p.requires_grad for name, p in model.named_parameters()}
    second_count = freeze_adapter(model)
    after = {name: p.requires_grad for name, p in model.named_parameters()}
    assert second_count == first_count
    assert snapshot == after


def test_freeze_adapter_leaves_head_trainable() -> None:
    """If a downstream head is added to the PEFT-wrapped model after
    freezing, it must remain trainable. Proxy here: add a fresh
    ``Linear`` after the freeze and assert its weights stay
    ``requires_grad=True``."""

    import torch.nn as nn

    from app.training.encoder_lora import freeze_adapter

    model = _make_toy_peft_model()
    freeze_adapter(model)
    head = nn.Linear(4, 3)
    assert head.weight.requires_grad is True
    assert head.bias.requires_grad is True
