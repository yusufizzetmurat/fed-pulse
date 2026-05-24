"""Multi-modal forecaster with gated InfoNCE fusion (#235).

Companion model to :class:`ForecasterModel` that keeps the market and
text modalities independent until the fusion stage so the InfoNCE
alignment loss has two separable representations to pull together.

The market side runs through the same recurrent_core families the
single-modality forecaster supports (lstm / gru / tcn / transformer /
informer / tft / dlinear); the text side reads the FinBERT pooled
embedding directly. A :class:`GatedInfoNCEFusion` block projects both
into a shared latent dim and emits a per-row gated fusion that feeds
the classification head.

Classification-only. The regression-target path stays on
``ForecasterModel`` — InfoNCE for a regression target needs a
different loss formulation that is out of scope here.
"""

from __future__ import annotations

import torch
from torch import nn

from app.models.config import (
    DEFAULT_DROPOUT,
    DEFAULT_HEAD_HIDDEN_SIZE,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_LAYERS,
    FEATURE_SIZE,
    SEQUENCE_LENGTH,
)
from app.models.dlinear import DLinear
from app.models.gated_infonce_fusion import GatedInfoNCEFusion
from app.models.tcn import TemporalConvNet
from app.models.transformer import SmallTransformer

_ATTENTION_POOL_MODELS = frozenset({"lstm_attn"})
_MEAN_POOL_MODELS = frozenset({"transformer", "tft", "informer"})
_ALLOWED_ARCHITECTURES = frozenset(
    {"lstm", "lstm_attn", "gru", "tcn", "transformer", "dlinear", "informer", "tft"}
)


class MultiModalForecasterModel(nn.Module):
    """Recurrent market encoder + FinBERT text + gated InfoNCE fusion.

    The forward emits the canonical 3-class logits the existing
    CrossEntropy training path consumes. :meth:`forward_with_modality_outputs`
    additionally returns the per-modality projections so the
    InfoNCE alignment loss can read them — the training loop calls
    the second method when ``fusion_mode == "gated_infonce"``.
    """

    def __init__(
        self,
        *,
        market_input_size: int = FEATURE_SIZE,
        text_embedding_dim: int = 768,
        latent_dim: int = 64,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        head_hidden_size: int = DEFAULT_HEAD_HIDDEN_SIZE,
        architecture: str = "lstm",
        n_classes: int = 3,
        fusion_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if architecture not in _ALLOWED_ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture {architecture!r}; allowed: {sorted(_ALLOWED_ARCHITECTURES)}"
            )
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2; got {n_classes}")
        self.architecture = architecture
        self.market_input_size = int(market_input_size)
        self.text_embedding_dim = int(text_embedding_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.head_hidden_size = int(head_hidden_size)
        self.n_classes = int(n_classes)
        # Compatibility shims so generic code that reads attributes on
        # the legacy ForecasterModel (e.g. ``model.output_mode``) does
        # not blow up when handed a MultiModalForecasterModel.
        self.output_mode = "classification"
        self.input_size = self.market_input_size
        self.initial_decay_rate = 0.0
        self.use_time_decay = False

        self.recurrent_core = self._build_recurrent_core(
            architecture=self.architecture,
            input_size=self.market_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.uses_attention_pool = self.architecture in _ATTENTION_POOL_MODELS
        self.uses_mean_pool = self.architecture in _MEAN_POOL_MODELS
        if self.uses_attention_pool:
            from app.models.attention import RecurrentSequenceAttention

            self.recurrent_attention = RecurrentSequenceAttention(
                hidden_size=self.hidden_size
            )
        else:
            self.recurrent_attention = None
        self.fusion = GatedInfoNCEFusion(
            market_dim=self.hidden_size,
            text_dim=self.text_embedding_dim,
            latent_dim=self.latent_dim,
            dropout=fusion_dropout,
        )
        self.classification_head = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.head_hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_hidden_size, self.n_classes),
        )

    def _pool_market(self, x: torch.Tensor) -> torch.Tensor:
        """Run the recurrent core and pool to a per-sequence vector."""

        if self.architecture == "dlinear":
            # DLinear emits the pooled state directly; no per-step
            # sequence to attend over.
            return self.recurrent_core(x)
        output, _ = self.recurrent_core(x)
        if self.uses_attention_pool:
            if self.recurrent_attention is None:
                raise RuntimeError(
                    "recurrent_attention not initialised but lstm_attn variant is active"
                )
            pooled, _ = self.recurrent_attention(output)
            return pooled
        if self.uses_mean_pool:
            return output.mean(dim=1)
        return output[:, -1, :]

    @staticmethod
    def _zero_out_missing_text(
        text_embedding: torch.Tensor,
        text_embedding_missing: torch.Tensor | None,
    ) -> torch.Tensor:
        """Mask the text embedding row-wise when its missing flag is set.

        The data loader emits a zero vector + missing=1 on rows
        without a prior statement; we multiply explicitly so a future
        loader path that emits non-zero placeholders cannot leak
        signal into the fusion.
        """

        if text_embedding_missing is None:
            return text_embedding
        flag = text_embedding_missing
        if flag.dim() == 1:
            flag = flag.unsqueeze(-1)
        keep = (1.0 - flag).clamp_(min=0.0, max=1.0)
        return text_embedding * keep

    def forward(
        self,
        x: torch.Tensor,
        text_embedding: torch.Tensor | None = None,
        text_embedding_missing: torch.Tensor | None = None,
        **_legacy_kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Return classification logits ``(B, n_classes)``.

        Convenience entry point for callers that don't need the
        per-modality projections (e.g. inference + eval). Training
        callers should use :meth:`forward_with_modality_outputs` so
        the InfoNCE loss has direct access to ``r_t`` and ``t_t``.

        Accepts and discards the legacy ``credibility`` / ``chunks`` /
        ``elapsed_days`` kwargs the single-modality forecaster takes,
        so the existing training-loop call site (which assembles a
        kwargs dict from several feature paths) doesn't need a
        model-class branch. The multi-modal path consumes only
        ``x`` (market features) + ``text_embedding`` (FinBERT pooled).
        """

        if text_embedding is None:
            raise ValueError(
                "MultiModalForecasterModel.forward requires text_embedding; "
                "set --text-encoder so the loader emits pooled embeddings."
            )
        return self.forward_with_modality_outputs(
            x,
            text_embedding=text_embedding,
            text_embedding_missing=text_embedding_missing,
        )["logits"]

    def forward_with_modality_outputs(
        self,
        x: torch.Tensor,
        text_embedding: torch.Tensor,
        text_embedding_missing: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return ``{logits, r_t, t_t, fused, gate}`` for InfoNCE training."""

        if x.dim() != 3:
            raise ValueError(
                f"market input x must be 3-D (B, T, F); got {x.dim()}-D"
            )
        if text_embedding.dim() == 1:
            text_embedding = text_embedding.unsqueeze(0)
        if text_embedding.dim() != 2:
            raise ValueError(
                f"text_embedding must be 2-D (B, D); got {text_embedding.dim()}-D"
            )

        market_pooled = self._pool_market(x)
        text_pooled = self._zero_out_missing_text(text_embedding, text_embedding_missing)
        fusion_out = self.fusion(market_pooled, text_pooled)
        logits = self.classification_head(fusion_out["fused"])
        return {
            "logits": logits,
            "r_t": fusion_out["r_t"],
            "t_t": fusion_out["t_t"],
            "fused": fusion_out["fused"],
            "gate": fusion_out["gate"],
        }

    @staticmethod
    def _build_recurrent_core(
        *,
        architecture: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> nn.Module:
        """Mirror :meth:`ForecasterModel._build_recurrent_core` for parity.

        Kept as a private static method so the multi-modal model can
        evolve its recurrent options without touching the single-modal
        path. Today it covers the same eight architectures the
        legacy model supports.
        """

        if architecture in {"lstm", "lstm_attn"}:
            return nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        if architecture == "gru":
            return nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        if architecture == "tcn":
            return TemporalConvNet(
                input_size=input_size,
                hidden_size=hidden_size,
                dropout=dropout,
            )
        if architecture == "transformer":
            return SmallTransformer(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        if architecture == "dlinear":
            return DLinear(
                input_size=input_size,
                hidden_size=hidden_size,
                sequence_length=SEQUENCE_LENGTH,
            )
        if architecture == "informer":
            from app.models.informer import InformerEncoder

            return InformerEncoder(
                input_size=input_size,
                hidden_size=hidden_size,
                dropout=dropout,
            )
        if architecture == "tft":
            from app.models.tft import TFTEncoder

            return TFTEncoder(
                input_size=input_size,
                hidden_size=hidden_size,
                dropout=dropout,
            )
        raise ValueError(f"Unknown architecture: {architecture!r}")
