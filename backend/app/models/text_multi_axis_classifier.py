"""Text-only multi-axis classifier for the /analyze surface (#78 follow-up).

Wraps a pre-trained transformer encoder (default: finbert_fed_adjacent,
the project's continued-pretrained FinBERT on BIS speeches) with the
multi-task head shipped in #272. Emits per-axis predictions on stance,
certainty, and time from a single input text.

The classifier is trained on the supervised rows in events.parquet
that carry axis labels (primarily the gtfintechlab cross-bank rows
for stance / certainty / time), not on the volatility-regime target
the time-series forecaster uses. This is a parallel model to the
forecaster, not a replacement.

The encoder pools the [CLS] token from the last hidden state into a
single vector per text; the MultiTaskHead emits three branches from
that pooled representation. The shared encoder is fine-tuned end-to-
end during training; at inference the model is frozen behind the
service singleton at backend/app/services/multi_axis_classifier.py.

The topic axis was retired in ADR 0044 — no upstream corpus shipped
topic labels, so the topic branch always predicted on zero training
signal and the inference card always rendered the same fallback.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from app.models.multi_task_head import MultiTaskHead
from app.models.config import (
    MULTI_TASK_CERTAINTY_CLASSES,
    MULTI_TASK_STANCE_CLASSES,
    MULTI_TASK_TIME_CLASSES,
)


class TextMultiAxisClassifier(nn.Module):
    """Pre-trained transformer encoder + MultiTaskHead.

    Constructed via :meth:`from_encoder_alias` which resolves the
    encoder from the registry (pinned repo + revision) so every
    training + inference run loads the same weights deterministically.
    """

    def __init__(  # noqa: PLR0913 — per-axis class-count + provenance kwargs surface by design
        self,
        encoder: nn.Module,
        *,
        hidden_size: int,
        head_hidden_size: int = 128,
        dropout: float = 0.1,
        stance_classes: int = MULTI_TASK_STANCE_CLASSES,
        certainty_classes: int = MULTI_TASK_CERTAINTY_CLASSES,
        time_classes: int = MULTI_TASK_TIME_CLASSES,
        encoder_alias: str = "",
        encoder_revision: str = "",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.hidden_size = int(hidden_size)
        self.head = MultiTaskHead(
            hidden_size=self.hidden_size,
            head_hidden_size=head_hidden_size,
            dropout=dropout,
            stance_classes=stance_classes,
            certainty_classes=certainty_classes,
            time_classes=time_classes,
        )
        # Surface the encoder provenance on the module so the
        # checkpoint payload can persist the exact alias + revision
        # the training run consumed.
        self.encoder_alias = str(encoder_alias)
        self.encoder_revision = str(encoder_revision)

    @classmethod
    def from_encoder_alias(
        cls,
        encoder_alias: str = "finbert_fed_adjacent",
        *,
        head_hidden_size: int = 128,
        dropout: float = 0.1,
    ) -> "TextMultiAxisClassifier":
        """Build the classifier with the encoder resolved from the registry.

        Loads the encoder via ``AutoModel.from_pretrained`` using the
        pinned revision. Refuses unpinned aliases — every checkpoint
        the multi-axis classifier consumes must round-trip through a
        recorded revision so reproducibility is non-negotiable.
        """

        from transformers import AutoModel

        from app.models.registry import encoder_ref

        ref = encoder_ref(encoder_alias)
        if ref is None:
            raise ValueError(
                f"Encoder alias {encoder_alias!r} is not in models/registry.yaml. "
                "Add it before constructing the classifier."
            )
        if not ref.revision:
            raise ValueError(
                f"Encoder alias {encoder_alias!r} has an empty revision in the "
                "registry. Multi-axis classifier training refuses unpinned "
                "encoders to keep checkpoint provenance unambiguous."
            )
        encoder = AutoModel.from_pretrained(
            ref.repo,
            revision=ref.revision,
            trust_remote_code=bool(getattr(ref, "trust_remote_code", False)),
        )
        hidden_size = int(getattr(encoder.config, "hidden_size", 0))
        if hidden_size <= 0:
            raise ValueError(
                f"Encoder {encoder_alias!r} reported hidden_size={hidden_size}; "
                "expected a positive integer from the transformer config."
            )
        return cls(
            encoder,
            hidden_size=hidden_size,
            head_hidden_size=head_hidden_size,
            dropout=dropout,
            encoder_alias=encoder_alias,
            encoder_revision=ref.revision,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Emit per-axis predictions for a batch of tokenised inputs.

        Uses the [CLS] token pooling convention. Returns the same
        dict shape MultiTaskHead emits: ``{stance, certainty, time}``.
        """

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Last hidden state shape: (B, T, H). [CLS] is at position 0
        # for the BERT / FinBERT family the classifier targets.
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.head(pooled)  # type: ignore[no-any-return]

    def metadata(self) -> dict[str, Any]:
        """Round-trippable provenance for checkpoint payloads."""

        return {
            "encoder_alias": self.encoder_alias,
            "encoder_revision": self.encoder_revision,
            "hidden_size": self.hidden_size,
            "head_hidden_size": self.head.head_hidden_size,
            "dropout": self.head.dropout,
            "stance_classes": self.head.stance_classes,
            "certainty_classes": self.head.certainty_classes,
            "time_classes": self.head.time_classes,
        }
