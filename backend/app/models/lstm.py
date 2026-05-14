from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from app.models.attention import ChunkAttentionPooler, TimeDecayAttention
from app.models.config import (
    DEFAULT_CHUNK_DECAY_RATE,
    DEFAULT_CHUNK_EMBEDDING_SIZE,
    DEFAULT_CHUNK_PROJECTION_DIM,
    DEFAULT_DROPOUT,
    DEFAULT_HEAD_HIDDEN_SIZE,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_INITIAL_DECAY_RATE,
    DEFAULT_NUM_LAYERS,
    FEATURE_SIZE,
)
from app.models.tcn import TemporalConvNet
from app.models.transformer import SmallTransformer


class ForecasterModel(nn.Module):
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
    ):
        """Forecaster LSTM with optional text-feature variants.

        Variant flags
        -------------
        use_time_decay : bool
            Variant A — dampens the sentiment feature by learned exponential
            time decay before the LSTM.
        use_chunk_attention : bool
            Variant B — attends over per-chunk embeddings from the chunk
            parquet (``embedding_source="chunk"``).
        use_llm_embeddings : bool
            Variant C — attends over per-document LLM embeddings from the
            llm_embeddings parquet (``embedding_source="llm"``).  Shares the
            same ``ChunkAttentionPooler`` and projection head as Variant B.

        Mutual exclusivity
        ------------------
        ``use_chunk_attention`` and ``use_llm_embeddings`` are mutually
        exclusive.  Both cannot be ``True`` simultaneously because they occupy
        the same LSTM input slot (the pooler output projection).  A
        ``ValueError`` is raised at construction time if both are set.  The
        eight-way ablation sweeps them as separate cells.
        """
        super().__init__()
        _allowed_model_types = {"lstm", "gru", "tcn", "transformer"}
        if model_type not in _allowed_model_types:
            raise ValueError(
                f"Unknown model_type: {model_type!r}. Allowed: lstm, gru, tcn, transformer"
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
        self.chunk_embedding_size = int(chunk_embedding_size)
        # Either Variant B or Variant C activates the pooler path.
        _uses_pooler = self.use_chunk_attention or self.use_llm_embeddings
        _adapter_active = _uses_pooler and text_channel == "embeddings"
        effective_dim = int(embedding_adapter_dim) if _adapter_active else int(chunk_projection_dim)
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
                # Zero-init so Variant B/C starts equivalent to baseline; the model
                # only departs from the baseline subspace if the text signal
                # actually reduces loss. Avoids drowning 6 base features under
                # 8 dims of random-init noise on a small training set.
                nn.init.zeros_(self.chunk_projection.weight)
                nn.init.zeros_(self.chunk_projection.bias)
        else:
            self.chunk_pooler = None
            self.chunk_projection = None
        lstm_input_size = input_size + self.chunk_projection_dim
        self.lstm_input_size = lstm_input_size
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.recurrent_core = self._build_recurrent_core(
            model_type=self.model_type,
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
        )
        self.lstm = self.recurrent_core  # alias for backward compatibility
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, head_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, 2),
        )

    def forward(
        self,
        x: torch.Tensor,
        chunks: torch.Tensor | None = None,
        elapsed_days: torch.Tensor | None = None,
        chunk_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_time_decay:
            x = self.time_decay(x)
        _uses_pooler = self.use_chunk_attention or self.use_llm_embeddings
        if _uses_pooler:
            if self.chunk_pooler is None or self.chunk_projection is None:
                raise RuntimeError("chunk_pooler not initialised but pooler variant is active")
            if chunks is None or elapsed_days is None:
                variant = "use_chunk_attention" if self.use_chunk_attention else "use_llm_embeddings"
                raise ValueError(
                    f"ForecasterModel requires chunks/elapsed_days when {variant}=True"
                )
            pooled, _, _ = self.chunk_pooler(chunks, elapsed_days, mask=chunk_mask)
            projected = self.chunk_projection(pooled)
            if projected.dim() == 1:
                projected = projected.unsqueeze(0)
            seq_len = x.shape[1]
            broadcast = projected.unsqueeze(1).expand(-1, seq_len, -1)
            x = torch.cat([x, broadcast], dim=-1)
        output, _ = self.lstm(x)
        last_step = output[:, -1, :]
        raw = self.head(last_step)
        close = raw[:, 0:1]
        # Volatility must stay non-negative, while close remains unconstrained.
        volatility = F.softplus(raw[:, 1:2])
        return torch.cat((close, volatility), dim=1)

    def attention_diagnostics(
        self,
        chunks: torch.Tensor,
        elapsed_days: torch.Tensor,
        chunk_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Return pooler diagnostics when either Variant B or C is active.

        Returns ``None`` when neither pooler variant is enabled.  The returned
        dict uses the key ``"pooled"`` / ``"weights"`` / ``"decay_coeffs"``
        regardless of which source (chunk or LLM) is active.
        """
        _uses_pooler = self.use_chunk_attention or self.use_llm_embeddings
        if not _uses_pooler or self.chunk_pooler is None:
            return None
        pooled, weights, decay_coeffs = self.chunk_pooler(chunks, elapsed_days, mask=chunk_mask)
        return {
            "pooled": pooled.detach(),
            "weights": weights.detach(),
            "decay_coeffs": decay_coeffs.detach(),
        }

    @staticmethod
    def _build_recurrent_core(
        *,
        model_type: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> nn.Module:
        if model_type == "lstm":
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
        raise ValueError(
            f"Unknown model_type: {model_type!r}. Allowed: lstm, gru, tcn, transformer"
        )
