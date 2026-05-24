"""Cover the GatedInfoNCEFusion module (#235).

The module is small but load-bearing — every downstream training
step reads ``fused`` for classification and ``(r_t, t_t)`` for the
InfoNCE alignment loss. These tests pin the per-key shape contract
and the gate's per-row behaviour.
"""

from __future__ import annotations

import torch

from app.models.gated_infonce_fusion import GatedInfoNCEFusion


def test_fusion_emits_four_keys_with_expected_shapes() -> None:
    torch.manual_seed(0)
    fusion = GatedInfoNCEFusion(market_dim=128, text_dim=768, latent_dim=64)
    market = torch.randn(4, 128)
    text = torch.randn(4, 768)
    out = fusion(market, text)
    assert set(out.keys()) == {"r_t", "t_t", "fused", "gate"}
    assert out["r_t"].shape == (4, 64)
    assert out["t_t"].shape == (4, 64)
    assert out["fused"].shape == (4, 64)
    assert out["gate"].shape == (4, 64)


def test_gate_is_per_row_in_zero_to_one_range() -> None:
    """The gate is sigmoid-bounded so it must live in [0, 1] for
    every row + every latent dim; clamping is what makes the
    convex combination at the fused step well-defined."""

    torch.manual_seed(1)
    fusion = GatedInfoNCEFusion(market_dim=32, text_dim=64, latent_dim=16)
    market = torch.randn(8, 32) * 10.0  # large activations
    text = torch.randn(8, 64) * 10.0
    out = fusion(market, text)
    assert torch.all(out["gate"] >= 0.0)
    assert torch.all(out["gate"] <= 1.0)


def test_fused_is_convex_combination_of_projections() -> None:
    """Fused = gate * r_t + (1 - gate) * t_t. Verify the contract
    by reconstructing the fused tensor manually from the dict outputs."""

    torch.manual_seed(2)
    fusion = GatedInfoNCEFusion(market_dim=16, text_dim=32, latent_dim=8)
    market = torch.randn(3, 16)
    text = torch.randn(3, 32)
    out = fusion(market, text)
    reconstructed = out["gate"] * out["r_t"] + (1.0 - out["gate"]) * out["t_t"]
    assert torch.allclose(out["fused"], reconstructed, atol=1e-6)


def test_fusion_rejects_mismatched_market_dim() -> None:
    """Wrong market input dim must raise loudly — the training loop
    should never silently pass a tensor with a mismatched shape into
    the linear projection."""

    import pytest

    fusion = GatedInfoNCEFusion(market_dim=64, text_dim=768, latent_dim=64)
    with pytest.raises(ValueError, match="market_pooled last-dim"):
        fusion(torch.randn(4, 32), torch.randn(4, 768))


def test_fusion_rejects_batch_size_mismatch() -> None:
    import pytest

    fusion = GatedInfoNCEFusion(market_dim=16, text_dim=32, latent_dim=8)
    with pytest.raises(ValueError, match="same length"):
        fusion(torch.randn(4, 16), torch.randn(5, 32))


def test_gate_distinguishes_modalities_when_one_is_zero() -> None:
    """Push the gate to favour the market side by feeding text=0
    on a freshly initialised model; the fused output should track
    the market projection more closely than the text projection.
    This is a directional smoke test, not a strict pin."""

    torch.manual_seed(3)
    fusion = GatedInfoNCEFusion(market_dim=16, text_dim=16, latent_dim=8)
    market = torch.randn(8, 16)
    text = torch.zeros(8, 16)
    out = fusion(market, text)
    # When text is zero, t_t = bias. The market side has non-zero
    # projection; the fused output should not be identical to t_t
    # (which is the "text-only" collapse).
    assert not torch.allclose(out["fused"], out["t_t"], atol=1e-3)
