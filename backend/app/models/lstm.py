from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

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
_DLINEAR_MODELS = frozenset({"dlinear"})


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
        credibility_features: bool = False,
        text_embedding_dim: int = 0,
        text_adapter_dim: int = 0,
        # Phase 9 V2 (#195) classification mode. Default "regression"
        # preserves the byte-identical 2-output (close, vol) head;
        # "classification" swaps in a 3-class head with CrossEntropy
        # loss dispatched in the training loop.
        output_mode: str = "regression",
        n_classes: int = 3,
        # Phase 9 V2 (#195) per-fold quantile cutoffs + target column.
        # Stored on the module so ``ModelConfig.from_model`` can round
        # them into the saved checkpoint payload. Inference + eval read
        # the same cutoffs back through ``ModelConfig.vol_regime_quantiles``
        # and apply ``vol_regime_class_for`` to keep the boundary
        # identical to training. Default ``()`` keeps the regression
        # path byte-identical.
        vol_regime_quantiles: tuple[float, ...] = (),
        vol_regime_target: str = "forward_realized_vol_10d",
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
        _allowed_model_types = {
            "lstm",
            "lstm_attn",
            "gru",
            "tcn",
            "transformer",
            "dlinear",
            "informer",
            "tft",
        }
        if model_type not in _allowed_model_types:
            raise ValueError(
                f"Unknown model_type: {model_type!r}. Allowed: {sorted(_allowed_model_types)}"
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
        if output_mode not in {"regression", "classification"}:
            raise ValueError(
                f"Unknown output_mode: {output_mode!r}. Allowed: regression, classification"
            )
        if output_mode == "classification" and int(n_classes) < 2:
            raise ValueError(
                f"output_mode='classification' requires n_classes >= 2; got {n_classes}"
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
        # Pooled-text-embedding path (PR #176 onward). The adapter
        # projects the encoder-native pooled vector (``text_embedding_dim``)
        # to a fixed ``text_adapter_dim`` slot that the recurrent core
        # broadcasts to every bar of the prior window plus the
        # event-day target frame. Both dims are 0 by default so a
        # checkpoint trained without the text path forwards
        # byte-identically. The trailing +1 (when the path is active)
        # is the missing flag the loader emits when fewer than one
        # prior statement is available.
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
        self.lstm = self.recurrent_core  # alias for backward compatibility
        self.uses_attention_pool = self.model_type in _ATTENTION_POOL_MODELS
        if self.uses_attention_pool:
            self.recurrent_attention: RecurrentSequenceAttention | None = (
                RecurrentSequenceAttention(hidden_size=hidden_size)
            )
        else:
            self.recurrent_attention = None
        # Phase 9 V2 (#195) head dispatch. ``regression`` keeps the
        # 2-output (close, vol) shape; ``classification`` switches the
        # final linear to emit ``n_classes`` logits for CrossEntropy.
        # The intermediate LayerNorm + Linear(hidden, head_hidden) +
        # GELU + Dropout stack is shared across modes so the
        # representation capacity stays comparable.
        self.output_mode = output_mode
        self.n_classes = int(n_classes)
        self.vol_regime_quantiles = tuple(float(v) for v in vol_regime_quantiles or ())
        self.vol_regime_target = str(vol_regime_target or "forward_realized_vol_10d")
        head_out = self.n_classes if output_mode == "classification" else 2
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, head_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, head_out),
        )

    def forward(
        self,
        x: torch.Tensor,
        chunks: torch.Tensor | None = None,
        elapsed_days: torch.Tensor | None = None,
        chunk_mask: torch.Tensor | None = None,
        credibility: torch.Tensor | None = None,
        text_embedding: torch.Tensor | None = None,
        text_embedding_missing: torch.Tensor | None = None,
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
        if self.credibility_features:
            if credibility is None:
                raise ValueError(
                    "ForecasterModel requires `credibility` tensor when credibility_features=True"
                )
            if credibility.dim() == 1:
                credibility = credibility.unsqueeze(0)
            if credibility.shape[-1] != self.credibility_dim:
                raise ValueError(
                    f"credibility tensor must have shape (..., {self.credibility_dim}); got {tuple(credibility.shape)}"
                )
            seq_len = x.shape[1]
            broadcast = credibility.unsqueeze(1).expand(-1, seq_len, -1)
            x = torch.cat([x, broadcast], dim=-1)
        if self._text_path_active:
            if self.text_adapter is None:
                raise RuntimeError(
                    "text_adapter not initialised but text-embedding path is active"
                )
            if text_embedding is None:
                raise ValueError(
                    "ForecasterModel requires `text_embedding` when text_adapter_dim > 0"
                )
            if text_embedding.dim() == 1:
                text_embedding = text_embedding.unsqueeze(0)
            if text_embedding.shape[-1] != self.text_embedding_dim:
                raise ValueError(
                    f"text_embedding tensor must have shape (..., {self.text_embedding_dim}); "
                    f"got {tuple(text_embedding.shape)}"
                )
            projected = self.text_adapter(text_embedding)
            if text_embedding_missing is None:
                missing_column = torch.zeros(
                    (projected.shape[0], 1), dtype=projected.dtype, device=projected.device
                )
            else:
                missing_column = text_embedding_missing
                if missing_column.dim() == 1:
                    missing_column = missing_column.unsqueeze(-1)
                missing_column = missing_column.to(dtype=projected.dtype, device=projected.device)
            # When the missing flag is on, the pooled embedding is
            # zeros by construction (the loader emits zeros + flag=1
            # together). Multiply the adapter output by (1 - missing)
            # so the recurrent core sees an unambiguous zero slot
            # even if a future loader path emits non-zero placeholders.
            keep_mask = (1.0 - missing_column).clamp_(min=0.0, max=1.0)
            projected = projected * keep_mask
            text_slot = torch.cat([projected, missing_column], dim=-1)
            seq_len = x.shape[1]
            broadcast = text_slot.unsqueeze(1).expand(-1, seq_len, -1)
            x = torch.cat([x, broadcast], dim=-1)
        output, _ = self.lstm(x)
        if self.uses_attention_pool:
            if self.recurrent_attention is None:
                raise RuntimeError(
                    "recurrent_attention not initialised but lstm_attn variant is active"
                )
            pooled_step, _attn_weights = self.recurrent_attention(output)
        else:
            pooled_step = output[:, -1, :]
        raw = self.head(pooled_step)
        # Phase 9 V2 (#195) dispatch. Classification mode returns the
        # raw class logits ``(batch, n_classes)`` so the training-loop
        # CrossEntropyLoss path can apply ``log_softmax`` itself. The
        # regression path keeps the existing softplus-on-volatility
        # post-processing (unconstrained close + non-negative vol).
        if self.output_mode == "classification":
            return raw
        close = raw[:, 0:1]
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
