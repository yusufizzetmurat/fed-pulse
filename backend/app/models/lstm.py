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
from app.models.multi_task_head import MultiTaskHead
from app.models.rates_heads import RATES_HEAD_N_CLASSES, RATES_HEAD_NAMES
from app.models.tcn import TemporalConvNet
from app.models.transformer import SmallTransformer

_ATTENTION_POOL_MODELS = frozenset({"lstm_attn"})
# Non-causal sequence cores: the encoder produces a contextualised
# token per timestep but does not accumulate global state into the
# final position the way an LSTM/GRU or a causal TCN does. Pooling the
# last timestep would discard the contextualised representation of
# every other position; the standard fix is to mean-pool across the
# sequence axis (or prepend a learnable [CLS] token, equivalent in the
# limit). Without this, ``output[:, -1, :]`` reads timestep T-1 only,
# losing the contextualised representations of timesteps 0..T-2.
_MEAN_POOL_MODELS = frozenset({"transformer", "tft", "informer"})
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
        # #304 dual-head methodology. ``classification`` (default) is the
        # pre-#304 byte-identical path -- only the MultiTaskHead mounts
        # and the CrossEntropy loss drives training. ``regression`` /
        # ``dual`` mount a second linear head off the same pooled
        # backbone output emitting a scalar log(RV) prediction; the
        # training loop drives it with MSE on
        # ``log(forward_realized_vol_10d)``. The classifier head still
        # mounts in all three modes so the checkpoint shape stays
        # stable and the existing conformal classification surface
        # keeps working -- only the loss contribution and the persisted
        # output dict differ across modes.
        head_mode: str = "classification",
        # #292 rates-complex heads. Tuple of head short-names
        # (``"2y"`` / ``"5y"`` / ``"terminal"``) to mount alongside the
        # existing MultiTaskHead and optional log_rv regression head.
        # Each name maps to one regression-head (``Linear -> 1`` scalar
        # predicting strict-forward 5-day yield change in basis points)
        # and one auxiliary 3-class classifier head (easing / neutral /
        # tightening). Default ``()`` keeps the pre-#292 path
        # byte-identical: no rates heads mount.
        rates_heads: tuple[str, ...] = (),
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
        if head_mode not in {"classification", "regression", "dual"}:
            raise ValueError(
                f"Unknown head_mode: {head_mode!r}. "
                "Allowed: classification, regression, dual"
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
        self.uses_mean_pool = self.model_type in _MEAN_POOL_MODELS
        if self.uses_attention_pool:
            self.recurrent_attention: RecurrentSequenceAttention | None = (
                RecurrentSequenceAttention(hidden_size=hidden_size)
            )
        else:
            self.recurrent_attention = None
        # Phase 9 V2 (#195) head dispatch. ``regression`` keeps the
        # 2-output (close, vol) shape; ``classification`` switches to
        # the multi-task head (#78) which emits four branches: stance
        # (the 3-class headline target), factor (scalar [-1, 1]),
        # certainty (3-class), and topic (4-class). The shared
        # pre-classifier stem mirrors the legacy LayerNorm + Linear +
        # GELU + Dropout stack so per-branch capacity matches the
        # baseline. Multi-task replaces single-head as the canonical
        # classification path (state_dict shape break documented in
        # ADR-0010); checkpoints trained on the legacy single head do
        # not load here, and cold-start /analyze retrains from scratch.
        self.output_mode = output_mode
        self.n_classes = int(n_classes)
        self.vol_regime_quantiles = tuple(float(v) for v in vol_regime_quantiles or ())
        self.vol_regime_target = str(vol_regime_target or "forward_realized_vol_10d")
        self.head_mode = str(head_mode)
        if output_mode == "classification":
            self.head: nn.Module = MultiTaskHead(
                hidden_size=hidden_size,
                head_hidden_size=head_hidden_size,
                dropout=dropout,
                stance_classes=self.n_classes,
            )
            # #304 dual-head methodology. Mount the log(RV) regression
            # head only when the operator explicitly opts into the
            # dual-head or regression-only path; the default
            # head_mode='classification' leaves it ``None`` so the
            # state_dict shape stays byte-identical to pre-#304
            # checkpoints. The head consumes the same pooled backbone
            # output the MultiTaskHead does so the comparison across
            # head_mode configurations measures only the head + loss
            # contribution, not the backbone capacity.
            if self.head_mode in {"regression", "dual"}:
                self.regression_head: nn.Module | None = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, head_hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden_size, 1),
                )
            else:
                self.regression_head = None
            # #292 rates-complex heads. Each mounted head gets two
            # parallel projections off the shared pooled backbone: one
            # scalar regression head (bps prediction) plus one 3-class
            # classifier (easing / neutral / tightening). Both share a
            # per-head LayerNorm + Linear + GELU + Dropout stem matching
            # the MultiTaskHead's per-axis stem so head capacity stays
            # comparable across the regime + log_rv + rates heads.
            self.rates_heads_active: tuple[str, ...] = tuple(
                str(name).lower() for name in rates_heads or ()
            )
            for name in self.rates_heads_active:
                if name not in RATES_HEAD_NAMES:
                    raise ValueError(
                        f"Unknown rates head: {name!r}. Allowed: "
                        f"{list(RATES_HEAD_NAMES)}"
                    )
            self.rates_regression_heads: nn.ModuleDict = nn.ModuleDict()
            self.rates_classification_heads: nn.ModuleDict = nn.ModuleDict()
            for name in self.rates_heads_active:
                self.rates_regression_heads[name] = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, head_hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden_size, 1),
                )
                self.rates_classification_heads[name] = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, head_hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden_size, RATES_HEAD_N_CLASSES),
                )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, head_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_size, 2),
            )
            # The #304 log(RV) head is only meaningful on the
            # classification branch (which carries the
            # forward_realized_vol_10d target); regression-output mode
            # already emits (close, vol) and ignores head_mode entirely.
            # #317 finding #8: the factory now raises at construction
            # time if rates_heads is non-empty alongside
            # output_mode='regression', so reaching this branch with
            # active rates heads is a programmer error. The empty
            # ModuleDict defaults stay so the regression-only forward
            # paths keep their attribute checks cheap.
            self.regression_head = None
            self.rates_heads_active = ()
            self.rates_regression_heads = nn.ModuleDict()
            self.rates_classification_heads = nn.ModuleDict()

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
        elif self.uses_mean_pool:
            # Non-causal cores (transformer / tft / informer): the
            # encoder does not accumulate global state into the final
            # token, so the last-step slice would only see the
            # contextualised representation of position T-1. Mean-pool
            # across the sequence axis recovers the full-sequence
            # representation.
            pooled_step = output.mean(dim=1)
        else:
            pooled_step = output[:, -1, :]
        # Phase 9 V2 (#195) + multi-task head (#78) dispatch.
        # Classification mode calls the MultiTaskHead which emits a
        # dict of per-axis logits; the stance branch is the canonical
        # 3-class output the training-loop CrossEntropyLoss reads, the
        # other three branches are stashed on ``self._last_multi_task``
        # for the multi-task loss + the /analyze response serialiser
        # to read. Regression mode keeps the existing 2-output
        # (close, vol) head and softplus post-processing.
        if self.output_mode == "classification":
            multi_task = self.head(pooled_step)
            stashed: dict[str, torch.Tensor] = {
                key: tensor.detach() for key, tensor in multi_task.items()
            }
            # #304 dual-head: when the regression head is mounted, the
            # standard forward path must emit its scalar log(RV)
            # prediction so the /analyze inference path and any caller
            # that uses ``model(x)`` rather than ``forward_multi_task``
            # can read it off ``_last_multi_task['log_rv']``. The
            # ``_skip_regression_head`` runtime flag opts the training
            # loop out of the regression head's forward at the dual +
            # alpha=0 boundary so the alpha=0 byte-identity contract
            # holds against pure classification.
            if (
                self.regression_head is not None
                and not bool(getattr(self, "_skip_regression_head", False))
            ):
                log_rv_pred = self.regression_head(pooled_step).squeeze(-1)
                stashed["log_rv"] = log_rv_pred.detach()
            # #292 rates heads. The per-head regression / classifier
            # output rides alongside the four classification axes so
            # the inference path can read both surfaces in one forward
            # call. ``rates_<name>_bps`` is the scalar bps prediction;
            # ``rates_<name>_cls_logits`` is the (B, 3) softmax-able
            # logits over (easing / neutral / tightening). Detached
            # because ``forward`` returns the stance branch alone for
            # back-compat with the CrossEntropy training path; the
            # gradient-tracked variant rides through
            # ``forward_multi_task`` below.
            for name in self.rates_heads_active:
                bps_pred = self.rates_regression_heads[name](pooled_step).squeeze(-1)
                cls_logits = self.rates_classification_heads[name](pooled_step)
                stashed[f"rates_{name}_bps"] = bps_pred.detach()
                stashed[f"rates_{name}_cls_logits"] = cls_logits.detach()
            self._last_multi_task = stashed
            return multi_task["stance"]  # type: ignore[no-any-return]
        raw = self.head(pooled_step)
        close = raw[:, 0:1]
        volatility = F.softplus(raw[:, 1:2])
        return torch.cat((close, volatility), dim=1)

    def forward_multi_task(
        self,
        x: torch.Tensor,
        chunks: torch.Tensor | None = None,
        elapsed_days: torch.Tensor | None = None,
        chunk_mask: torch.Tensor | None = None,
        credibility: torch.Tensor | None = None,
        text_embedding: torch.Tensor | None = None,
        text_embedding_missing: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the forward pass and return the full multi-task dict.

        Classification mode only; raises ``RuntimeError`` on a model
        configured for regression. Used by the multi-task training
        loss (which needs gradient-tracked logits for all four
        branches) and by the inference path (which serialises every
        branch into the ``/analyze`` response). The main
        :meth:`forward` returns only the stance logits so existing
        CrossEntropy callers stay byte-compatible.
        """

        if self.output_mode != "classification":
            raise RuntimeError(
                "forward_multi_task requires output_mode='classification'"
            )
        # Mirror the body of forward() up to the pooled-step but emit
        # the multi-task dict instead of the stance-only tensor.
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
            seq_len = x.shape[1]
            broadcast = credibility.unsqueeze(1).expand(-1, seq_len, -1)
            x = torch.cat([x, broadcast], dim=-1)
        if self._text_path_active:
            if self.text_adapter is None or text_embedding is None:
                raise RuntimeError("text path active but inputs missing")
            if text_embedding.dim() == 1:
                text_embedding = text_embedding.unsqueeze(0)
            projected = self.text_adapter(text_embedding)
            if text_embedding_missing is None:
                missing_column = torch.zeros(
                    (projected.shape[0], 1),
                    dtype=projected.dtype,
                    device=projected.device,
                )
            else:
                missing_column = text_embedding_missing
                if missing_column.dim() == 1:
                    missing_column = missing_column.unsqueeze(-1)
                missing_column = missing_column.to(
                    dtype=projected.dtype, device=projected.device
                )
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
            pooled_step, _ = self.recurrent_attention(output)
        elif self.uses_mean_pool:
            pooled_step = output.mean(dim=1)
        else:
            pooled_step = output[:, -1, :]
        multi_task: dict[str, torch.Tensor] = self.head(pooled_step)
        # #304 dual-head methodology. When the regression head is
        # mounted (head_mode in {regression, dual}) emit the log(RV)
        # scalar prediction alongside the four classification axes so
        # the training loop's dual-head loss can pick it up off the
        # same call. ``(B, 1)`` -> ``(B,)`` so downstream MSE / metric
        # helpers can index without a redundant squeeze, matching the
        # factor branch convention.
        #
        # The ``_skip_regression_head`` runtime flag lets the training
        # loop opt out of the regression head's forward computation
        # entirely (used when ``head_mode='dual'`` is paired with
        # ``regression_alpha=0.0`` so the regression head produces no
        # gradient and the dual + alpha=0 boundary is byte-identical
        # to the pure classification path; symmetric to the ``alpha=1``
        # case which skips the classifier branch on the loss side).
        if (
            self.regression_head is not None
            and not bool(getattr(self, "_skip_regression_head", False))
        ):
            log_rv_pred = self.regression_head(pooled_step).squeeze(-1)
            multi_task["log_rv"] = log_rv_pred
        # #292 rates heads -- gradient-tracked variants. ``rates_<name>_bps``
        # is the scalar bps prediction the rates MSE loss reads;
        # ``rates_<name>_cls_logits`` is the (B, 3) auxiliary classifier
        # the optional CE branch reads. Both ride alongside the four
        # classification axes so the dual-head joint loss can mix them.
        for name in self.rates_heads_active:
            bps_pred = self.rates_regression_heads[name](pooled_step).squeeze(-1)
            cls_logits = self.rates_classification_heads[name](pooled_step)
            multi_task[f"rates_{name}_bps"] = bps_pred
            multi_task[f"rates_{name}_cls_logits"] = cls_logits
        return multi_task

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
