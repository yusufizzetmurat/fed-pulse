"""Research forecaster (issue #336).

`ForecasterResearchModel` carries every research knob the sweep harness
and the training loop need: eight architecture types, three text
variants, two embedding channels, three head modes, two output modes,
the multi-task head, three rates heads, LoRA toggles, InfoNCE
fields, credibility features, and class-weight knobs. It is the
training entrypoint; checkpoints saved off this class are promoted to
the serving class via :func:`scripts.promote_checkpoint` before they
hit the `/analyze` path.

The two ``forward`` methods share input-prep through
:func:`app.models.forecaster_base.prepare_recurrent_input` so the
~60-line duplication that lived inline on the legacy class is now in
one place. The shared backbone + adapter wiring also lives on
:class:`ForecasterBase` so the serving class can re-use it with a
narrower forward surface.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from app.models.config import (
    CREDIBILITY_FEATURE_DIM,
    DEFAULT_CHUNK_EMBEDDING_SIZE,
    DEFAULT_CHUNK_PROJECTION_DIM,
    DEFAULT_DROPOUT,
    DEFAULT_HEAD_HIDDEN_SIZE,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_INITIAL_DECAY_RATE,
    DEFAULT_NUM_LAYERS,
    FEATURE_SIZE,
    N_SUPPORTED_SYMBOLS,
)
from app.models.forecaster_base import ForecasterBase, prepare_recurrent_input
from app.models.multi_task_head import MultiTaskHead
from app.models.rates_heads import RATES_HEAD_N_CLASSES, RATES_HEAD_NAMES


class ForecasterResearchModel(ForecasterBase):
    """Research-side forecaster carrying every knob."""

    def __init__(  # noqa: C901
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
        output_mode: str = "regression",
        n_classes: int = 3,
        vol_regime_quantiles: tuple[float, ...] = (),
        vol_regime_target: str = "forward_realized_vol_10d",
        head_mode: str = "classification",
        rates_heads: tuple[str, ...] = (),
        rates_aux_classification: bool = False,
        use_regime_conditioning: bool = False,
        use_sep: bool = False,
        use_press_conf: bool = False,
        use_statement_delta: bool = False,
        use_vote_features: bool = False,
        use_vix_features: bool = False,
        use_doc_length: bool = False,
        symbol_embedding_dim: int = 0,
        n_symbols: int = 0,
        aux_horizons: tuple[int, ...] = (),
    ):
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
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            head_hidden_size=head_hidden_size,
            initial_decay_rate=initial_decay_rate,
            model_type=model_type,
            use_time_decay=use_time_decay,
            use_chunk_attention=use_chunk_attention,
            use_llm_embeddings=use_llm_embeddings,
            chunk_embedding_size=chunk_embedding_size,
            chunk_projection_dim=chunk_projection_dim,
            text_channel=text_channel,
            embedding_adapter_dim=embedding_adapter_dim,
            credibility_features=credibility_features,
            text_embedding_dim=text_embedding_dim,
            text_adapter_dim=text_adapter_dim,
            use_regime_conditioning=use_regime_conditioning,
            use_sep=use_sep,
            use_press_conf=use_press_conf,
            use_statement_delta=use_statement_delta,
            use_vote_features=use_vote_features,
            use_vix_features=use_vix_features,
            use_doc_length=use_doc_length,
        )
        # Head dispatch -- classification mounts the MultiTaskHead, the
        # optional log(RV) regression head, and the per-rates-head pair
        # (regression + 3-class auxiliary classifier). Regression-output
        # mode keeps the 2-output (close, vol) head and ignores
        # head_mode entirely (see ADR 0015 for the canonical-objective
        # rationale on the classification branch).
        self.output_mode = output_mode
        self.n_classes = int(n_classes)
        self.vol_regime_quantiles = tuple(float(v) for v in vol_regime_quantiles or ())
        self.vol_regime_target = str(vol_regime_target or "forward_realized_vol_10d")
        self.head_mode = str(head_mode)
        # #480 symbol-conditioned regime head. ``symbol_embedding_dim=0``
        # (default) is the symbol-agnostic canonical: no embedding module
        # mounts, no widening of the regime / log-RV head input, no new
        # forward-time index lookup. ``> 0`` mounts the embedding and
        # widens the regime head + dual-head regression head inputs by
        # ``symbol_embedding_dim`` so the concatenated ``(pool, embed)``
        # vector lands cleanly on the head's first LayerNorm.
        self.symbol_embedding_dim = int(symbol_embedding_dim or 0)
        if self.symbol_embedding_dim < 0:
            raise ValueError(
                f"symbol_embedding_dim must be >= 0; got {symbol_embedding_dim}"
            )
        self.n_symbols = int(n_symbols) if n_symbols else N_SUPPORTED_SYMBOLS
        # The embedding only wires into the regime head + dual-head
        # log-RV regression head, both of which live on the
        # classification branch. Regression-output mode (close, vol) has
        # no regime head to condition, so the embedding mount is rejected
        # there to keep the head-construction graph one-knob-deep.
        if self.symbol_embedding_dim > 0 and output_mode != "classification":
            raise ValueError(
                "symbol_embedding_dim > 0 requires output_mode='classification' "
                "(the symbol-conditioned head is the regime classifier on the "
                "classification branch). Got "
                f"output_mode={output_mode!r}."
            )
        if self.symbol_embedding_dim > 0:
            self.symbol_embedding: nn.Embedding | None = nn.Embedding(
                self.n_symbols, self.symbol_embedding_dim
            )
            head_input_size = hidden_size + self.symbol_embedding_dim
        else:
            self.symbol_embedding = None
            head_input_size = hidden_size
        if output_mode == "classification":
            self.head: nn.Module = MultiTaskHead(
                hidden_size=head_input_size,
                head_hidden_size=head_hidden_size,
                dropout=dropout,
                stance_classes=self.n_classes,
            )
            if self.head_mode in {"regression", "dual"}:
                self.regression_head: nn.Module | None = nn.Sequential(
                    nn.LayerNorm(head_input_size),
                    nn.Linear(head_input_size, head_hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden_size, 1),
                )
            else:
                self.regression_head = None
            # #471 multi-horizon auxiliary regression heads. Mount only
            # when the primary log-RV regression head is also mounted --
            # the aux heads share the encoder + recurrent core and only
            # differ from the primary in their supervised target column
            # (``forward_realized_vol_<H>d`` for each H). Aux heads stay
            # empty when ``head_mode='classification'`` because the joint
            # loss has no primary regression branch for them to compose
            # with; the same head-mode + aux-horizons combination is
            # rejected at the factory to surface the misconfiguration
            # early. ``aux_horizons=()`` (default) leaves the ModuleDict
            # empty so the state_dict shape is byte-identical to the
            # pre-#471 path.
            self.aux_horizons: tuple[int, ...] = tuple(
                int(h) for h in aux_horizons or ()
            )
            self.aux_regression_heads: nn.ModuleDict = nn.ModuleDict()
            if self.aux_horizons and self.regression_head is not None:
                for horizon in self.aux_horizons:
                    self.aux_regression_heads[f"h{int(horizon)}"] = nn.Sequential(
                        nn.LayerNorm(head_input_size),
                        nn.Linear(head_input_size, head_hidden_size),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(head_hidden_size, 1),
                    )
            self.rates_heads_active: tuple[str, ...] = tuple(
                str(name).lower() for name in rates_heads or ()
            )
            for name in self.rates_heads_active:
                if name not in RATES_HEAD_NAMES:
                    raise ValueError(
                        f"Unknown rates head: {name!r}. Allowed: "
                        f"{list(RATES_HEAD_NAMES)}"
                    )
            # The aux 3-class direction classifier is opt-in per #292.
            # Default OFF mounts only the regression heads -- the product
            # surface emits the regression card with a None
            # directional_bucket, and the joint loss reduces to MSE-only
            # on the rates branch. Opt-in mounts the paired classifier
            # so the easing / neutral / tightening surface appears and
            # the CE term enters the rates joint loss.
            self.rates_aux_classification = bool(rates_aux_classification)
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
                if self.rates_aux_classification:
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
            self.regression_head = None
            self.rates_heads_active = ()
            self.rates_aux_classification = bool(rates_aux_classification)
            self.rates_regression_heads = nn.ModuleDict()
            self.rates_classification_heads = nn.ModuleDict()
            self.aux_horizons = ()
            self.aux_regression_heads = nn.ModuleDict()

    def _pool_with_symbol(
        self,
        pooled_step: torch.Tensor,
        symbol_id: torch.Tensor | None,
    ) -> torch.Tensor:
        """Concatenate the symbol embedding to the encoder pool (#480).

        Default-off contract: when ``symbol_embedding_dim == 0`` the
        method short-circuits and returns ``pooled_step`` unchanged, so
        the forward pass is byte-identical to the symbol-agnostic
        canonical. When ``> 0`` and ``symbol_id`` is ``None``, the
        symbol-agnostic regime head was wired but no id was supplied;
        index ``0`` (the canonical ``^GSPC`` slot) is used so the
        existing single-symbol training contract still trains cleanly.
        """

        if self.symbol_embedding is None:
            return pooled_step
        if symbol_id is None:
            symbol_id = pooled_step.new_zeros(
                pooled_step.shape[0], dtype=torch.long
            )
        symbol_id = symbol_id.to(device=pooled_step.device, dtype=torch.long)
        embed = self.symbol_embedding(symbol_id)
        return torch.cat([pooled_step, embed], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        chunks: torch.Tensor | None = None,
        elapsed_days: torch.Tensor | None = None,
        chunk_mask: torch.Tensor | None = None,
        credibility: torch.Tensor | None = None,
        text_embedding: torch.Tensor | None = None,
        text_embedding_missing: torch.Tensor | None = None,
        text_embedding_per_bar: torch.Tensor | None = None,
        symbol_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = prepare_recurrent_input(
            self,
            x,
            chunks=chunks,
            elapsed_days=elapsed_days,
            chunk_mask=chunk_mask,
            credibility=credibility,
            text_embedding=text_embedding,
            text_embedding_missing=text_embedding_missing,
            text_embedding_per_bar=text_embedding_per_bar,
        )
        pooled_step = self._encode(x)
        head_input = self._pool_with_symbol(pooled_step, symbol_id)
        if self.output_mode == "classification":
            multi_task = self.head(head_input)
            stashed: dict[str, torch.Tensor] = {
                key: tensor.detach() for key, tensor in multi_task.items()
            }
            if (
                self.regression_head is not None
                and not bool(getattr(self, "_skip_regression_head", False))
            ):
                log_rv_pred = self.regression_head(head_input).squeeze(-1)
                stashed["log_rv"] = log_rv_pred.detach()
                for horizon in self.aux_horizons:
                    key = f"h{int(horizon)}"
                    aux_pred = self.aux_regression_heads[key](head_input).squeeze(-1)
                    stashed[f"aux_log_rv_{int(horizon)}d"] = aux_pred.detach()
            for name in self.rates_heads_active:
                bps_pred = self.rates_regression_heads[name](pooled_step).squeeze(-1)
                stashed[f"rates_{name}_bps"] = bps_pred.detach()
                if name in self.rates_classification_heads:
                    cls_logits = self.rates_classification_heads[name](pooled_step)
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
        text_embedding_per_bar: torch.Tensor | None = None,
        symbol_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the forward pass and return the full multi-task dict.

        Classification mode only; raises ``RuntimeError`` on a model
        configured for regression. Drives the multi-task training loss
        (gradient-tracked logits across all four branches) and the
        /analyze regime + rates card surfaces.
        """

        if self.output_mode != "classification":
            raise RuntimeError(
                "forward_multi_task requires output_mode='classification'"
            )
        x = prepare_recurrent_input(
            self,
            x,
            chunks=chunks,
            elapsed_days=elapsed_days,
            chunk_mask=chunk_mask,
            credibility=credibility,
            text_embedding=text_embedding,
            text_embedding_missing=text_embedding_missing,
            text_embedding_per_bar=text_embedding_per_bar,
        )
        pooled_step = self._encode(x)
        head_input = self._pool_with_symbol(pooled_step, symbol_id)
        multi_task: dict[str, torch.Tensor] = self.head(head_input)
        if (
            self.regression_head is not None
            and not bool(getattr(self, "_skip_regression_head", False))
        ):
            log_rv_pred = self.regression_head(head_input).squeeze(-1)
            multi_task["log_rv"] = log_rv_pred
            for horizon in self.aux_horizons:
                key = f"h{int(horizon)}"
                aux_pred = self.aux_regression_heads[key](head_input).squeeze(-1)
                multi_task[f"aux_log_rv_{int(horizon)}d"] = aux_pred
        for name in self.rates_heads_active:
            bps_pred = self.rates_regression_heads[name](pooled_step).squeeze(-1)
            multi_task[f"rates_{name}_bps"] = bps_pred
            if name in self.rates_classification_heads:
                cls_logits = self.rates_classification_heads[name](pooled_step)
                multi_task[f"rates_{name}_cls_logits"] = cls_logits
        return multi_task

    def attention_diagnostics(
        self,
        chunks: torch.Tensor,
        elapsed_days: torch.Tensor,
        chunk_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Return pooler diagnostics when Variant B or C is active."""
        _uses_pooler = self.use_chunk_attention or self.use_llm_embeddings
        if not _uses_pooler or self.chunk_pooler is None:
            return None
        pooled, weights, decay_coeffs = self.chunk_pooler(
            chunks, elapsed_days, mask=chunk_mask
        )
        return {
            "pooled": pooled.detach(),
            "weights": weights.detach(),
            "decay_coeffs": decay_coeffs.detach(),
        }


__all__ = ["ForecasterResearchModel"]
