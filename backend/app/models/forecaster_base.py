"""Shared forecaster backbone (issue #336).

The research and serving forecasters share a recurrent backbone, the
chunk/LLM pooler, the credibility / pooled-text adapters, and the
input-prep that broadcasts each per-event scalar into the sequence
axis. The pre-#336 ``ForecasterModel`` duplicated the input-prep across
``forward`` and ``forward_multi_task`` (~60 lines each); both copies
now live on :func:`prepare_recurrent_input` and are reused by both
forwards on the research class and by the single forward on the
serving class.
"""

from __future__ import annotations

import torch
from torch import nn

from app.models.attention import (
    ChunkAttentionPooler,
    RecurrentSequenceAttention,
    TimeDecayAttention,
)
from app.models.config import (
    CREDIBILITY_FEATURE_DIM,
    DEFAULT_CHUNK_DECAY_RATE,
    DEFAULT_CHUNK_EMBEDDING_SIZE,
    DEFAULT_CHUNK_PROJECTION_DIM,
    DEFAULT_DROPOUT,
    DEFAULT_HEAD_HIDDEN_SIZE,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_INITIAL_DECAY_RATE,
    DEFAULT_NUM_LAYERS,
    FEATURE_SIZE,
    SEQUENCE_LENGTH,
)
from app.models.dlinear import DLinear
from app.models.tcn import TemporalConvNet
from app.models.transformer import SmallTransformer

_ATTENTION_POOL_MODELS = frozenset({"lstm_attn"})
# Non-causal sequence cores need mean-pool: a transformer / TFT /
# informer encoder produces a contextualised token per timestep but does
# not accumulate global state into the final position, so reading
# ``output[:, -1, :]`` would drop the representations of timesteps
# 0..T-2. Mean-pool across the sequence axis recovers the full-sequence
# representation; equivalent in the limit to a learnable [CLS] token.
_MEAN_POOL_MODELS = frozenset({"transformer", "tft", "informer"})
_DLINEAR_MODELS = frozenset({"dlinear"})

_ALLOWED_MODEL_TYPES = frozenset({
    "lstm",
    "lstm_attn",
    "gru",
    "tcn",
    "transformer",
    "dlinear",
    "informer",
    "tft",
})


class ForecasterBase(nn.Module):
    """Shared backbone: recurrent core + optional text / credibility adapters."""

    def __init__(
        self,
        input_size: int = FEATURE_SIZE,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        head_hidden_size: int = DEFAULT_HEAD_HIDDEN_SIZE,
        initial_decay_rate: float = DEFAULT_INITIAL_DECAY_RATE,
        *,
        model_type: str = "lstm",
        use_time_decay: bool = True,
        use_chunk_attention: bool = False,
        use_llm_embeddings: bool = False,
        chunk_embedding_size: int = DEFAULT_CHUNK_EMBEDDING_SIZE,
        chunk_projection_dim: int = DEFAULT_CHUNK_PROJECTION_DIM,
        text_channel: str = "scalar",
        embedding_adapter_dim: int = 128,
        credibility_features: bool = False,
        text_embedding_dim: int = 0,
        text_adapter_dim: int = 0,
    ):
        super().__init__()
        if model_type not in _ALLOWED_MODEL_TYPES:
            raise ValueError(
                f"Unknown model_type: {model_type!r}. Allowed: {sorted(_ALLOWED_MODEL_TYPES)}"
            )
        if use_chunk_attention and use_llm_embeddings:
            raise ValueError(
                "use_chunk_attention and use_llm_embeddings are mutually exclusive. "
                "Set at most one to True."
            )
        if text_channel not in {"scalar", "embeddings"}:
            raise ValueError(
                f"Unknown text_channel: {text_channel!r}. Allowed: scalar, embeddings"
            )
        self.model_type = model_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.head_hidden_size = head_hidden_size
        self.initial_decay_rate = float(initial_decay_rate)
        self.use_time_decay = bool(use_time_decay)
        self.use_chunk_attention = bool(use_chunk_attention)
        self.use_llm_embeddings = bool(use_llm_embeddings)
        self.text_channel = text_channel
        self.credibility_features = bool(credibility_features)
        self.credibility_dim = CREDIBILITY_FEATURE_DIM if self.credibility_features else 0
        self.chunk_embedding_size = int(chunk_embedding_size)
        _uses_pooler = self.use_chunk_attention or self.use_llm_embeddings
        _adapter_active = _uses_pooler and text_channel == "embeddings"
        effective_dim = (
            int(embedding_adapter_dim) if _adapter_active else int(chunk_projection_dim)
        )
        self.chunk_projection_dim = effective_dim if _uses_pooler else 0
        self.time_decay = TimeDecayAttention(initial_decay_rate)
        if _uses_pooler:
            self.chunk_pooler: ChunkAttentionPooler | None = ChunkAttentionPooler(
                embedding_size=self.chunk_embedding_size,
                initial_decay_rate=DEFAULT_CHUNK_DECAY_RATE,
            )
            if _adapter_active:
                from app.models.embedding_adapter import EmbeddingAdapter

                self.chunk_projection: nn.Module | None = EmbeddingAdapter(
                    input_dim=self.chunk_embedding_size,
                    output_dim=self.chunk_projection_dim,
                    zero_init=True,
                )
            else:
                self.chunk_projection = nn.Linear(
                    self.chunk_embedding_size, self.chunk_projection_dim, bias=True
                )
                # Zero-init so Variant B/C starts equivalent to baseline; the
                # model only departs from the baseline subspace if the text
                # signal actually reduces loss. Avoids drowning 6 base
                # features under 8 dims of random-init noise on a small
                # training set.
                nn.init.zeros_(self.chunk_projection.weight)
                nn.init.zeros_(self.chunk_projection.bias)
        else:
            self.chunk_pooler = None
            self.chunk_projection = None
        # Pooled-text-embedding path (PR #176). Adapter projects the
        # encoder-native pooled vector to a fixed slot the recurrent
        # core broadcasts across every bar. The +1 (when active) is the
        # missing flag the loader emits when fewer than one prior
        # statement is available.
        self.text_embedding_dim = int(text_embedding_dim or 0)
        self.text_adapter_dim = int(text_adapter_dim or 0)
        self._text_path_active = self.text_embedding_dim > 0 and self.text_adapter_dim > 0
        if self._text_path_active:
            from app.models.text_embedding_adapter import TextEmbeddingAdapter

            self.text_adapter: nn.Module | None = TextEmbeddingAdapter(
                in_dim=self.text_embedding_dim,
                out_dim=self.text_adapter_dim,
                zero_init=True,
            )
            text_path_dim = self.text_adapter_dim + 1
        else:
            self.text_adapter = None
            text_path_dim = 0
        self.text_path_dim = text_path_dim

        lstm_input_size = (
            input_size
            + self.chunk_projection_dim
            + self.credibility_dim
            + text_path_dim
        )
        self.lstm_input_size = lstm_input_size
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.recurrent_core = self._build_recurrent_core(
            model_type=self.model_type,
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
        )
        self.lstm = self.recurrent_core  # alias for back-compat
        self.uses_attention_pool = self.model_type in _ATTENTION_POOL_MODELS
        self.uses_mean_pool = self.model_type in _MEAN_POOL_MODELS
        if self.uses_attention_pool:
            self.recurrent_attention: RecurrentSequenceAttention | None = (
                RecurrentSequenceAttention(hidden_size=hidden_size)
            )
        else:
            self.recurrent_attention = None

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run the recurrent core and pool the sequence.

        Last-step pool for causal cores (lstm, gru, tcn, dlinear),
        learned attention pool for ``lstm_attn``, mean pool for
        non-causal cores (transformer, tft, informer). Returns the
        pooled ``(B, hidden_size)`` representation downstream heads
        consume.
        """
        output, _ = self.lstm(x)
        if self.uses_attention_pool:
            if self.recurrent_attention is None:
                raise RuntimeError(
                    "recurrent_attention not initialised but lstm_attn variant is active"
                )
            pooled_step, _ = self.recurrent_attention(output)
            return pooled_step
        if self.uses_mean_pool:
            return output.mean(dim=1)
        return output[:, -1, :]

    @staticmethod
    def _build_recurrent_core(
        *,
        model_type: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> nn.Module:
        if model_type in {"lstm", "lstm_attn"}:
            return nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        if model_type == "gru":
            return nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        if model_type == "tcn":
            return TemporalConvNet(
                input_size=input_size,
                hidden_size=hidden_size,
                dropout=dropout,
            )
        if model_type == "transformer":
            return SmallTransformer(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        if model_type == "dlinear":
            return DLinear(
                input_size=input_size,
                hidden_size=hidden_size,
                sequence_length=SEQUENCE_LENGTH,
            )
        if model_type == "informer":
            from app.models.informer import InformerEncoder

            return InformerEncoder(
                input_size=input_size,
                hidden_size=hidden_size,
                dropout=dropout,
            )
        if model_type == "tft":
            from app.models.tft import TFTEncoder

            return TFTEncoder(
                input_size=input_size,
                hidden_size=hidden_size,
                dropout=dropout,
            )
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            "Allowed: lstm, lstm_attn, gru, tcn, transformer, dlinear, informer, tft"
        )


def prepare_recurrent_input(
    model: ForecasterBase,
    x: torch.Tensor,
    *,
    chunks: torch.Tensor | None = None,
    elapsed_days: torch.Tensor | None = None,
    chunk_mask: torch.Tensor | None = None,
    credibility: torch.Tensor | None = None,
    text_embedding: torch.Tensor | None = None,
    text_embedding_missing: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the optional input-side transforms before the recurrent core.

    Applies (in order) the learnable time-decay, the chunk/LLM pooler,
    the credibility broadcast, and the pooled-text adapter. Each step
    is a no-op when the corresponding flag is off.

    Extracted from the pre-#336 ``ForecasterModel`` where the same
    sequence appeared inline in both ``forward()`` and
    ``forward_multi_task()``. Both copies were verbatim minus the
    ``ForecasterModel`` self-reference in error messages, so a single
    helper is a strict refactor.
    """
    if model.use_time_decay:
        x = model.time_decay(x)
    _uses_pooler = model.use_chunk_attention or model.use_llm_embeddings
    if _uses_pooler:
        if model.chunk_pooler is None or model.chunk_projection is None:
            raise RuntimeError(
                "chunk_pooler not initialised but pooler variant is active"
            )
        if chunks is None or elapsed_days is None:
            variant = (
                "use_chunk_attention" if model.use_chunk_attention else "use_llm_embeddings"
            )
            raise ValueError(
                f"forecaster requires chunks/elapsed_days when {variant}=True"
            )
        pooled, _, _ = model.chunk_pooler(chunks, elapsed_days, mask=chunk_mask)
        projected = model.chunk_projection(pooled)
        if projected.dim() == 1:
            projected = projected.unsqueeze(0)
        seq_len = x.shape[1]
        broadcast = projected.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([x, broadcast], dim=-1)
    if model.credibility_features:
        if credibility is None:
            raise ValueError(
                "forecaster requires `credibility` tensor when credibility_features=True"
            )
        if credibility.dim() == 1:
            credibility = credibility.unsqueeze(0)
        if credibility.shape[-1] != model.credibility_dim:
            raise ValueError(
                f"credibility tensor must have shape (..., {model.credibility_dim}); "
                f"got {tuple(credibility.shape)}"
            )
        seq_len = x.shape[1]
        broadcast = credibility.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([x, broadcast], dim=-1)
    if model._text_path_active:
        if model.text_adapter is None:
            raise RuntimeError(
                "text_adapter not initialised but text-embedding path is active"
            )
        if text_embedding is None:
            raise ValueError(
                "forecaster requires `text_embedding` when text_adapter_dim > 0"
            )
        if text_embedding.dim() == 1:
            text_embedding = text_embedding.unsqueeze(0)
        if text_embedding.shape[-1] != model.text_embedding_dim:
            raise ValueError(
                f"text_embedding tensor must have shape (..., {model.text_embedding_dim}); "
                f"got {tuple(text_embedding.shape)}"
            )
        projected = model.text_adapter(text_embedding)
        if text_embedding_missing is None:
            missing_column = torch.zeros(
                (projected.shape[0], 1), dtype=projected.dtype, device=projected.device
            )
        else:
            missing_column = text_embedding_missing
            if missing_column.dim() == 1:
                missing_column = missing_column.unsqueeze(-1)
            missing_column = missing_column.to(
                dtype=projected.dtype, device=projected.device
            )
        # When the missing flag is on, the pooled embedding is zeros by
        # construction (loader emits zeros + flag=1 together). Multiply
        # the adapter output by (1 - missing) so the recurrent core
        # sees an unambiguous zero slot even if a future loader path
        # emits non-zero placeholders.
        keep_mask = (1.0 - missing_column).clamp_(min=0.0, max=1.0)
        projected = projected * keep_mask
        text_slot = torch.cat([projected, missing_column], dim=-1)
        seq_len = x.shape[1]
        broadcast = text_slot.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([x, broadcast], dim=-1)
    return x


__all__ = [
    "ForecasterBase",
    "prepare_recurrent_input",
]
