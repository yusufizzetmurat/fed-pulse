"""Cover the TextMultiAxisClassifier module (#78 follow-up).

The classifier wires a transformer encoder onto the MultiTaskHead.
Tests use a tiny BERT-config stub so the suite runs in seconds
without pulling FinBERT weights from disk.
"""

from __future__ import annotations

import torch
from torch import nn

from app.models.text_multi_axis_classifier import TextMultiAxisClassifier


class _StubEncoder(nn.Module):
    """Minimal stand-in for a HF transformer encoder.

    Returns a namespace-like object exposing ``last_hidden_state`` so
    the classifier's ``[CLS]`` pooling step works without pulling the
    real FinBERT weights into the test process.
    """

    def __init__(self, hidden_size: int = 16, vocab_size: int = 30) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask=None):
        hidden = self.embed(input_ids)

        class _Out:
            pass

        out = _Out()
        out.last_hidden_state = hidden
        return out


def test_classifier_emits_three_axis_dict_from_stub_encoder() -> None:
    """Post-ADR-0044: topic branch retired; classifier emits stance / factor / certainty."""
    torch.manual_seed(0)
    encoder = _StubEncoder(hidden_size=16, vocab_size=30)
    model = TextMultiAxisClassifier(
        encoder,
        hidden_size=16,
        head_hidden_size=8,
        dropout=0.0,
        encoder_alias="stub",
        encoder_revision="rev",
    )
    input_ids = torch.randint(0, 30, (2, 12))
    attention_mask = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    assert set(out.keys()) == {"stance", "factor", "certainty"}
    assert out["stance"].shape == (2, 3)
    assert out["factor"].shape == (2,)
    assert out["certainty"].shape == (2, 3)


def test_metadata_round_trips_encoder_provenance() -> None:
    """The classifier records the encoder alias + revision so a
    checkpoint payload can re-resolve the same weights on load."""

    encoder = _StubEncoder(hidden_size=8, vocab_size=20)
    model = TextMultiAxisClassifier(
        encoder,
        hidden_size=8,
        head_hidden_size=4,
        dropout=0.1,
        encoder_alias="alias",
        encoder_revision="rev123",
    )
    meta = model.metadata()
    assert meta["encoder_alias"] == "alias"
    assert meta["encoder_revision"] == "rev123"
    assert meta["hidden_size"] == 8
    assert meta["head_hidden_size"] == 4
    assert meta["stance_classes"] == 3


def test_factor_branch_stays_in_minus_one_to_one_range() -> None:
    """Tanh bound on the factor head — same contract as the shared
    MultiTaskHead unit test, exercised here through the classifier
    wrapper so a future refactor that bypasses the head is caught."""

    torch.manual_seed(1)
    encoder = _StubEncoder(hidden_size=16, vocab_size=30)
    model = TextMultiAxisClassifier(
        encoder,
        hidden_size=16,
        head_hidden_size=8,
        dropout=0.0,
    )
    # Push activations through with extreme weights to provoke
    # saturation; without tanh the factor branch could exceed 1.
    with torch.no_grad():
        model.head.factor.weight.fill_(10.0)
        model.head.factor.bias.fill_(10.0)
    input_ids = torch.randint(0, 30, (4, 8))
    out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    assert torch.all(out["factor"] >= -1.0)
    assert torch.all(out["factor"] <= 1.0)
