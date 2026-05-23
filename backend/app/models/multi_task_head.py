"""Multi-task head for the forecaster (#78).

Replaces the legacy single-output classification head on
``ForecasterModel`` when ``output_mode=="classification"``. Emits four
branches from a shared pre-classifier stem:

- ``stance`` — 3-class logits over ``{hawkish, dovish, neutral}`` (the
  legacy 3-class head; the existing CrossEntropy loss reads from this
  branch on the training path so the headline macro-F1 stays
  comparable to the single-head baseline).
- ``factor`` — scalar regression in ``[-1, 1]`` (tanh-bounded).
- ``certainty`` — 3-class logits over ``{certain, uncertain, neutral}``.
- ``topic`` — K-class logits over ``MULTI_TASK_TOPIC_LABELS``
  (``{macro, forward_guidance, market_reaction, other}``).

The shared stem (LayerNorm + Linear + GELU + Dropout) mirrors the
existing single-head pre-classifier so the representation capacity
per branch is comparable to the baseline. Each branch is a single
linear projection from the stem output, which keeps the parameter
count small on top of the recurrent core.

Loss masking lives in :class:`app.training.loss.MultiTaskLoss`; the
head itself is mask-unaware (it always emits the same shape).
"""

from __future__ import annotations

import torch
from torch import nn

from app.models.config import (
    MULTI_TASK_CERTAINTY_CLASSES,
    MULTI_TASK_STANCE_CLASSES,
    MULTI_TASK_TOPIC_CLASSES,
)


class MultiTaskHead(nn.Module):
    """Per-axis output heads sharing a pre-classifier stem.

    Construction mirrors the legacy single-head Sequential at
    ``lstm.py:230-247`` (LayerNorm + Linear + GELU + Dropout) so the
    head capacity is comparable. The four output projections are
    independent linears applied to the stem output.
    """

    def __init__(
        self,
        hidden_size: int,
        head_hidden_size: int,
        dropout: float,
        *,
        stance_classes: int = MULTI_TASK_STANCE_CLASSES,
        certainty_classes: int = MULTI_TASK_CERTAINTY_CLASSES,
        topic_classes: int = MULTI_TASK_TOPIC_CLASSES,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.head_hidden_size = int(head_hidden_size)
        self.dropout = float(dropout)
        self.stance_classes = int(stance_classes)
        self.certainty_classes = int(certainty_classes)
        self.topic_classes = int(topic_classes)

        self.stem = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.head_hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.stance = nn.Linear(self.head_hidden_size, self.stance_classes)
        self.factor = nn.Linear(self.head_hidden_size, 1)
        self.certainty = nn.Linear(self.head_hidden_size, self.certainty_classes)
        self.topic = nn.Linear(self.head_hidden_size, self.topic_classes)

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        """Emit per-axis predictions from the pooled backbone output.

        Returns a dict with four keys:

        - ``stance`` — ``(B, stance_classes)`` raw logits
        - ``factor`` — ``(B,)`` tanh-bounded scalar in ``[-1, 1]``
        - ``certainty`` — ``(B, certainty_classes)`` raw logits
        - ``topic`` — ``(B, topic_classes)`` raw logits

        Classification branches return raw logits so CrossEntropy can
        apply log-softmax internally. The factor branch applies a tanh
        bound at emit time because the upstream label support is in
        ``[-1, 1]`` and an unconstrained linear regressor would drift
        outside that range on rows with no factor supervision.
        """

        stem = self.stem(pooled)
        return {
            "stance": self.stance(stem),
            "factor": torch.tanh(self.factor(stem).squeeze(-1)),
            "certainty": self.certainty(stem),
            "topic": self.topic(stem),
        }
