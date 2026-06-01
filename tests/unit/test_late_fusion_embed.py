"""Unit tests for the asserting late-fusion text embedder (no network)."""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from app.data import late_fusion_embed as lfe


class _FakeRef:
    def __init__(self, repo: str | None, revision: str | None = None) -> None:
        self.repo = repo
        self.revision = revision


def test_load_encoder_rejects_non_finbert_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    # The anti-fallback gate: a resolved repo that is not FinBERT-fed must raise
    # BEFORE any model is loaded (so distilbert can never be used silently).
    monkeypatch.setattr(lfe, "encoder_ref", lambda _alias: _FakeRef("distilbert-base-uncased"))
    with pytest.raises(RuntimeError, match="refusing silent fallback"):
        lfe.load_encoder()


def test_load_encoder_rejects_unresolved_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lfe, "encoder_ref", lambda _alias: None)
    with pytest.raises(RuntimeError, match="did not resolve"):
        lfe.load_encoder()

    monkeypatch.setattr(lfe, "encoder_ref", lambda _alias: _FakeRef(None))
    with pytest.raises(RuntimeError, match="did not resolve"):
        lfe.load_encoder()


class _StubTokenizer:
    """Returns one chunk: 3 tokens, last one padding."""

    def __call__(self, text: str, **_: object) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([[5, 6, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
        }


class _StubModel:
    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> object:
        # hidden per token: [[1,1],[2,2],[3,3]] for the single chunk
        hidden = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
        return types.SimpleNamespace(last_hidden_state=hidden)


def _stub_encoder() -> lfe.Encoder:
    return lfe.Encoder(
        tokenizer=_StubTokenizer(),
        model=_StubModel(),
        device=torch.device("cpu"),
        repo="yusufizzetmurat/finbert-fed-adjacent",
        revision="abc",
        dim=2,
    )


def test_embed_documents_mean_pools_over_nonpad_tokens() -> None:
    enc = _stub_encoder()
    out = lfe.embed_documents(enc, ["some text"])
    assert out.shape == (1, 2)
    # mean of the two non-pad tokens [1,1] and [2,2] = [1.5, 1.5]; pad token excluded
    assert out[0] == pytest.approx([1.5, 1.5])


def test_embed_documents_empty_text_is_zero_vector() -> None:
    enc = _stub_encoder()
    out = lfe.embed_documents(enc, ["", "   "])
    assert out.shape == (2, 2)
    assert np.allclose(out, 0.0)


def test_fingerprint_is_stable_and_sensitive() -> None:
    a = np.ones((3, 4), dtype=np.float32)
    b = np.ones((3, 4), dtype=np.float32)
    b[0, 0] = 2.0
    assert lfe._fingerprint(a) == lfe._fingerprint(a.copy())
    assert lfe._fingerprint(a) != lfe._fingerprint(b)
