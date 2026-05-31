"""Tests for the gated fusion forecaster + InfoNCE loss (CPU, synthetic)."""

from __future__ import annotations
import torch
from app.data import gated_fusion as gf


def _model():
    torch.manual_seed(0)
    return gf.build_model(d_text=16, d_market=4, n_horizons=3, d_hidden=8)


def test_gate_floor_no_text_ignores_text_embedding() -> None:
    m = _model().eval()
    mkt = torch.randn(5, 4)
    mask = torch.zeros(5)  # no fresh text
    with torch.no_grad():
        p1 = m(torch.randn(5, 16), mkt, mask)
        p2 = m(torch.randn(5, 16), mkt, mask)  # different text emb
    assert torch.allclose(p1["pred"], p2["pred"], atol=1e-6)  # text cannot leak in
    assert torch.allclose(p1["gate"], torch.zeros(5), atol=1e-6)


def test_gate_active_with_text_uses_text() -> None:
    m = _model().eval()
    mkt = torch.randn(5, 4)
    mask = torch.ones(5)
    with torch.no_grad():
        p1 = m(torch.zeros(5, 16), mkt, mask)
        p2 = m(torch.ones(5, 16) * 3.0, mkt, mask)
    assert not torch.allclose(p1["pred"], p2["pred"], atol=1e-4)  # text changes output


def test_info_nce_aligned_low_vs_random_high() -> None:
    z = torch.randn(8, 8)
    mask = torch.ones(8)
    aligned = gf.info_nce_loss(z, z.clone(), mask, temperature=0.07)
    rand = gf.info_nce_loss(z, torch.randn(8, 8), mask, temperature=0.07)
    assert aligned < rand
    assert gf.info_nce_loss(z, z, torch.zeros(8), temperature=0.07).item() == 0.0  # <2 positives


def test_fusion_loss_trains_down() -> None:
    torch.manual_seed(1)
    m = gf.build_model(16, 4, 3, d_hidden=8)
    batch = {
        "text_emb": torch.randn(12, 16),
        "market_feat": torch.randn(12, 4),
        "text_mask": torch.cat([torch.ones(8), torch.zeros(4)]),
        "targets": torch.randn(12, 3),
    }
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    first = gf.fusion_loss(m, batch)["loss"].item()
    for _ in range(50):
        opt.zero_grad()
        out = gf.fusion_loss(m, batch)
        out["loss"].backward()
        opt.step()
    assert gf.fusion_loss(m, batch)["loss"].item() < first  # learns to overfit


def test_supcon_pulls_same_label_together() -> None:
    import torch
    from app.data import gated_fusion as gf

    torch.manual_seed(0)
    # two clusters by label; aligned-with-label embeddings → lower loss than scrambled
    z = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = torch.tensor([0, 0, 1, 1])
    mask = torch.ones(4)
    good = gf.supcon_loss(z, labels, mask, temperature=0.1)
    scrambled = gf.supcon_loss(z, torch.tensor([0, 1, 0, 1]), mask, temperature=0.1)
    assert good < scrambled
    assert gf.supcon_loss(z, labels, torch.zeros(4), temperature=0.1).item() == 0.0


def test_fusion_clf_loss_trains_down() -> None:
    import torch
    from app.data import gated_fusion as gf

    torch.manual_seed(1)
    m = gf.build_model(16, 4, 3, d_hidden=8)  # n_horizons=3 → 3-class logits
    batch = {
        "text_emb": torch.randn(12, 16),
        "market_feat": torch.randn(12, 4),
        "text_mask": torch.cat([torch.ones(8), torch.zeros(4)]),
        "labels": torch.randint(0, 3, (12,)),
    }
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    first = gf.fusion_clf_loss(m, batch)["loss"].item()
    for _ in range(60):
        opt.zero_grad()
        gf.fusion_clf_loss(m, batch)["loss"].backward()
        opt.step()
    assert gf.fusion_clf_loss(m, batch)["loss"].item() < first
