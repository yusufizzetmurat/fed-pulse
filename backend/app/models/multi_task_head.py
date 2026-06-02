"""Multi-task head for the forecaster (#78).

Replaces the legacy single-output classification head on
``ForecasterModel`` when ``output_mode=="classification"``. Emits three
branches from a shared pre-classifier stem:

- ``stance`` — 3-class logits over ``{hawkish, dovish, neutral}`` (the
  legacy 3-class head; the existing CrossEntropy loss reads from this
  branch on the training path so the headline macro-F1 stays
  comparable to the single-head baseline).
- ``certainty`` — 3-class logits over ``{certain, uncertain, neutral}``.
- ``time`` — 2-class logits over ``{forward looking, not forward
  looking}`` (gtfintechlab ``time_label``).

The shared stem (LayerNorm + Linear + GELU + Dropout) mirrors the
existing single-head pre-classifier so the representation capacity
per branch is comparable to the baseline. Each branch is a single
linear projection from the stem output, which keeps the parameter
count small on top of the recurrent core.

The factor branch (GSS market-derived regression target) was retired
because text cannot predict it and the training pool had 0% coverage;
the time branch replaces it. The topic branch was retired in ADR 0044
because no upstream FOMC corpus or cross-bank gtfintechlab dataset
ships topic labels.

Loss masking lives in :class:`app.training.loss.MultiTaskLoss`; the
head itself is mask-unaware (it always emits the same shape).
"""

from __future__ import annotations

import torch
from torch import nn

from app.models.config import (
    MULTI_TASK_CERTAINTY_CLASSES,
    MULTI_TASK_STANCE_CLASSES,
    MULTI_TASK_TIME_CLASSES,
)


class MultiTaskHead(nn.Module):
    """Per-axis output heads sharing a pre-classifier stem.

    Construction mirrors the legacy single-head Sequential at
    ``lstm.py:230-247`` (LayerNorm + Linear + GELU + Dropout) so the
    head capacity is comparable. The three output projections are
    independent linears applied to the stem output.
    """

    def __init__(  # noqa: PLR0913 — per-axis class-count kwargs surface by design
        self,
        hidden_size: int,
        head_hidden_size: int,
        dropout: float,
        *,
        stance_classes: int = MULTI_TASK_STANCE_CLASSES,
        certainty_classes: int = MULTI_TASK_CERTAINTY_CLASSES,
        time_classes: int = MULTI_TASK_TIME_CLASSES,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.head_hidden_size = int(head_hidden_size)
        self.dropout = float(dropout)
        self.stance_classes = int(stance_classes)
        self.certainty_classes = int(certainty_classes)
        self.time_classes = int(time_classes)

        self.stem = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.head_hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.stance = nn.Linear(self.head_hidden_size, self.stance_classes)
        self.certainty = nn.Linear(self.head_hidden_size, self.certainty_classes)
        self.time = nn.Linear(self.head_hidden_size, self.time_classes)

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        """Emit per-axis predictions from the pooled backbone output.

        Returns a dict with three keys:

        - ``stance`` — ``(B, stance_classes)`` raw logits
        - ``certainty`` — ``(B, certainty_classes)`` raw logits
        - ``time`` — ``(B, time_classes)`` raw logits

        All branches return raw logits so CrossEntropy can apply
        log-softmax internally.
        """

        stem = self.stem(pooled)
        return {
            "stance": self.stance(stem),
            "certainty": self.certainty(stem),
            "time": self.time(stem),
        }
