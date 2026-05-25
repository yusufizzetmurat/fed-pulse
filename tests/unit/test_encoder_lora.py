"""Unit tests for the LoRA freeze helpers in :mod:`app.training.encoder_lora`.

Covers ``freeze_adapter`` (the PEFT-side helper that flips every LoRA
matrix's ``requires_grad`` to False) and the pure boundary-policy
helper ``should_freeze_lora_at_epoch`` that the training loop drives
the stage-1-train-then-freeze curriculum through.

The PEFT-backed behavioural tests load only when ``peft`` is importable;
the boundary-policy tests run unconditionally.
"""

from __future__ import annotations

import pytest


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
