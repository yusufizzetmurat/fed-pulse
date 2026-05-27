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

from typing import cast

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
    RICH_FEATURE_SIZE,
    RICH_MACRO_REGIME_DIM,
    RICH_MACRO_REGIME_MISSING_DIM,
    RICH_SEP_DIM,
    RICH_SEP_MISSING_DIM,
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
        use_regime_conditioning: bool = False,
        use_sep: bool = False,
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
        if text_channel not in {"scalar", "embeddings", "per_bar"}:
            raise ValueError(
                f"Unknown text_channel: {text_channel!r}. "
                "Allowed: scalar, embeddings, per_bar"
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
        # #307 macro-regime conditioning. The gate is a small linear
        # projection from the strict-prior regime block (3 scalars) onto
        # a per-position multiplicative mask over the rich-feature slice
        # ``[FEATURE_SIZE:RICH_FEATURE_SIZE]``. Initialised so the output
        # is identically 1.0 at the start of training (weight and bias
        # zero-init, ``2 * sigmoid(0) == 1.0``) so flipping the flag on
        # without a model re-init still produces a byte-identical
        # forward pass. The model only departs from the no-gate behaviour
        # if the regime signal actually reduces the supervised loss.
        # The block tail (positions past ``RICH_FEATURE_SIZE`` carrying
        # the regime indicators + missing flag) is forwarded unchanged
        # so the gate never gates its own conditioning input.
        self.use_regime_conditioning = bool(use_regime_conditioning)
        if self.use_regime_conditioning:
            rich_slice_dim = RICH_FEATURE_SIZE - FEATURE_SIZE
            self.regime_gate: nn.Linear | None = nn.Linear(
                RICH_MACRO_REGIME_DIM, rich_slice_dim, bias=True
            )
            nn.init.zeros_(self.regime_gate.weight)
            nn.init.zeros_(self.regime_gate.bias)
            # The loader appends ``RICH_MACRO_REGIME_DIM + RICH_MACRO_REGIME_MISSING_DIM``
            # extra scalars past ``RICH_FEATURE_SIZE`` on every per-bar
            # tensor when conditioning is on (see ``FeatureVector.as_rich_list``).
            # The gate modulates the legacy rich-feature slice in place,
            # but the regime tail itself flows past unchanged into the
            # recurrent core so the temporal dynamics still see the
            # indicator that triggered the modulation. The LSTM width
            # therefore widens by the regime tail; without this the core
            # would be built at ``RICH_FEATURE_SIZE`` and reject the
            # 91-wide tensor the loader actually produces.
            regime_tail_dim = RICH_MACRO_REGIME_DIM + RICH_MACRO_REGIME_MISSING_DIM
        else:
            self.regime_gate = None
            regime_tail_dim = 0
        self.regime_tail_dim = regime_tail_dim
        # #215 SEP dot-plot block. The loader appends
        # ``RICH_SEP_DIM + RICH_SEP_MISSING_DIM`` extra scalars past the
        # regime tail on every per-bar tensor when ``--use-sep`` is on
        # (see ``FeatureVector.as_rich_list``). The recurrent core must
        # absorb the widened input; the SEP block is a feature-only
        # contribution with no architectural gate so the only wiring is
        # the input-projection width here.
        self.use_sep = bool(use_sep)
        if self.use_sep:
            sep_tail_dim = RICH_SEP_DIM + RICH_SEP_MISSING_DIM
        else:
            sep_tail_dim = 0
        self.sep_tail_dim = sep_tail_dim
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
            + regime_tail_dim
            + sep_tail_dim
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
            return cast(torch.Tensor, pooled_step)
        if self.uses_mean_pool:
            return cast(torch.Tensor, output.mean(dim=1))
        return cast(torch.Tensor, output[:, -1, :])

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
    text_embedding_per_bar: torch.Tensor | None = None,
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
    # #307 macro-regime conditioning gate. When wired, the per-bar tensor
    # carries the regime block at its tail (3 scalars + 1 missing flag,
    # positions ``[RICH_FEATURE_SIZE:RICH_FEATURE_SIZE + 4]``) by the
    # loader's ``as_rich_list`` contract. The gate reads the 3 scalars,
    # produces a per-position multiplicative mask over the legacy rich-
    # feature slice ``[FEATURE_SIZE:RICH_FEATURE_SIZE]``, and applies it
    # in place so the recurrent core sees a modulated rich-feature view.
    # The mask is ``2 * sigmoid(linear(regime))``; the zero-init Linear
    # makes the mask identically 1.0 at start of training, so the
    # forward pass is byte-identical to the no-gate path until gradients
    # push the gate off identity. The conditioning tail itself is left
    # untouched and continues to flow into the recurrent core alongside
    # the gated rich block -- the recurrent core therefore still sees
    # which regime triggered the modulation, so its temporal dynamics can
    # exploit the regime indicator on top of the gated mask.
    regime_gate = getattr(model, "regime_gate", None)
    if regime_gate is not None:
        regime_dim_total = RICH_MACRO_REGIME_DIM + 1  # block + missing flag
        if x.shape[-1] >= RICH_FEATURE_SIZE + regime_dim_total:
            regime_input = x[..., -regime_dim_total:-1]
            # Broadcast the per-bar regime block through the linear +
            # sigmoid path. The bars within a sequence carry identical
            # regime values (the loader broadcasts per event), so the
            # gate output is constant across the time axis; we still
            # compute it per bar so a future per-bar regime-shift loader
            # path drops in without touching this branch.
            gate_logits = regime_gate(regime_input)
            gate = 2.0 * torch.sigmoid(gate_logits)
            modulated = x[..., FEATURE_SIZE:RICH_FEATURE_SIZE] * gate
            x = torch.cat(
                [
                    x[..., :FEATURE_SIZE],
                    modulated,
                    x[..., RICH_FEATURE_SIZE:],
                ],
                dim=-1,
            )
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
        use_per_bar = (
            model.text_channel == "per_bar" and text_embedding_per_bar is not None
        )
        if use_per_bar:
            # #327 Arm A. The per-bar layout carries one pooled vector
            # per lookback bar (``(B, T, in_dim)``); each bar is
            # projected through the same adapter so the recurrent core
            # consumes actual temporal text dynamics rather than the
            # broadcast-static replication of a single pooled vector.
            seq_len = x.shape[1]
            assert text_embedding_per_bar is not None  # narrowed via use_per_bar
            per_bar = text_embedding_per_bar
            if per_bar.dim() != 3:
                raise ValueError(
                    "text_embedding_per_bar must be a 3D tensor "
                    f"(B, T, in_dim); got {tuple(per_bar.shape)}"
                )
            if per_bar.shape[1] != seq_len:
                raise ValueError(
                    "text_embedding_per_bar sequence length must match the "
                    f"market input sequence ({seq_len}); got {per_bar.shape[1]}"
                )
            if per_bar.shape[-1] != model.text_embedding_dim:
                raise ValueError(
                    "text_embedding_per_bar last-dim must be "
                    f"{model.text_embedding_dim}; got {per_bar.shape[-1]}"
                )
            # Flatten (B, T, in_dim) -> (B*T, in_dim) so the adapter
            # forward path stays a single Linear+LN+GELU call regardless
            # of the per-bar broadcast.
            b, t, in_dim = per_bar.shape
            projected = model.text_adapter(per_bar.reshape(b * t, in_dim))
            projected = projected.reshape(b, t, -1)
            if text_embedding_missing is None:
                missing_column_bar = torch.zeros(
                    (b, t, 1), dtype=projected.dtype, device=projected.device
                )
            else:
                missing_column_bar = text_embedding_missing
                if missing_column_bar.dim() == 2:
                    # ``(B, T)`` broadcasts to a per-bar mask.
                    missing_column_bar = missing_column_bar.unsqueeze(-1)
                elif missing_column_bar.dim() == 1:
                    # ``(B,)`` -- tile to per-bar.
                    missing_column_bar = (
                        missing_column_bar.view(-1, 1, 1).expand(-1, t, 1).contiguous()
                    )
                missing_column_bar = missing_column_bar.to(
                    dtype=projected.dtype, device=projected.device
                )
            keep_mask_bar = (1.0 - missing_column_bar).clamp_(min=0.0, max=1.0)
            projected = projected * keep_mask_bar
            text_slot_bar = torch.cat([projected, missing_column_bar], dim=-1)
            x = torch.cat([x, text_slot_bar], dim=-1)
            return x
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
