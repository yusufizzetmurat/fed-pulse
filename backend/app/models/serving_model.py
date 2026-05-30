"""Serving forecaster (issue #336).

`ForecasterServingModel` is the narrow inference-side class the
`/analyze` request flow imports. The shape is frozen: regression-output
default (close + softplus(vol)), regression-canonical head_mode (per
ADR 0015), no rates heads, no LoRA toggles, no InfoNCE fields, no
class-weight knobs. The class still supports the regime card via the
optional ``multi_task`` head (mounted when the promoted checkpoint
carries it) so the regime_classification card and the
build_market_reaction_panel surface keep working.

The class accepts research-shaped state_dicts under the same key names
the research class emits (the backbone + adapter weights are shared
verbatim through :class:`ForecasterBase`). Heads beyond the canonical
serving surface load via :meth:`load_state_dict(..., strict=False)`
when present so the same checkpoint can drive both classes -- the
serving path simply does not read them through its narrow forward.

The promotion contract lives in :mod:`scripts.promote_checkpoint`.
Calling code should not skip the promotion -- it version-bumps the
``model_version`` metadata so the served checkpoint is distinguishable
from the research-side artefact it came from.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from app.models.config import (
    DEFAULT_CHUNK_EMBEDDING_SIZE,
    DEFAULT_CHUNK_PROJECTION_DIM,
    DEFAULT_DROPOUT,
    DEFAULT_HEAD_HIDDEN_SIZE,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_INITIAL_DECAY_RATE,
    DEFAULT_NUM_LAYERS,
    FEATURE_SIZE,
)
from app.models.forecaster_base import ForecasterBase, prepare_recurrent_input
from app.models.multi_task_head import MultiTaskHead
from app.models.rates_heads import RATES_HEAD_N_CLASSES, RATES_HEAD_NAMES


class ForecasterServingModel(ForecasterBase):
    """Narrow inference forecaster.

    Frozen ctor surface for /analyze callers. Supports both
    output_mode='regression' (close/vol forecast series) and
    output_mode='classification' (regime + multi-task heads for the
    regime / market-reaction cards). The classification branch is
    needed because the regime card under ADR 0015 lifts ``log_rv``
    from the regression head off ``forward_multi_task``; the serving
    class therefore still carries the multi-task head when the
    promoted checkpoint was trained with it.
    """

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
        output_mode: str = "regression",
        n_classes: int = 3,
        vol_regime_quantiles: tuple[float, ...] = (),
        vol_regime_target: str = "forward_realized_vol_10d",
        head_mode: str = "regression",
        rates_heads: tuple[str, ...] = (),
        rates_aux_classification: bool = False,
        use_regime_conditioning: bool = False,
        use_sep: bool = False,
        use_press_conf: bool = False,
        use_statement_delta: bool = False,
        use_vote_features: bool = False,
        use_vix_features: bool = False,
    ):
        if output_mode not in {"regression", "classification"}:
            raise ValueError(
                f"Unknown output_mode: {output_mode!r}. Allowed: regression, classification"
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
        )
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
            self.rates_heads_active: tuple[str, ...] = tuple(
                str(name).lower() for name in rates_heads or ()
            )
            for name in self.rates_heads_active:
                if name not in RATES_HEAD_NAMES:
                    raise ValueError(
                        f"Unknown rates head: {name!r}. Allowed: "
                        f"{list(RATES_HEAD_NAMES)}"
                    )
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
    ) -> torch.Tensor:
        """Narrow serving forward.

        Regression-output mode: returns the (close, softplus(vol))
        forecast pair the close/vol time series consumes. Classification
        mode: returns the stance logits the regime card surface
        consumes; ``forward_multi_task`` then drives the multi-axis
        cards via the same backbone.
        """
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
        if self.output_mode == "classification":
            multi_task = self.head(pooled_step)
            stashed: dict[str, torch.Tensor] = {
                key: tensor.detach() for key, tensor in multi_task.items()
            }
            if self.regression_head is not None:
                log_rv_pred = self.regression_head(pooled_step).squeeze(-1)
                stashed["log_rv"] = log_rv_pred.detach()
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
    ) -> dict[str, torch.Tensor]:
        """Emit the full multi-task / rates / regime dict.

        Used by ``build_market_reaction_panel`` and
        ``build_regime_classification_card`` to pull the regime
        log_rv + bucket distribution and the rates cards off one
        forward pass. Classification mode only.
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
        multi_task: dict[str, torch.Tensor] = self.head(pooled_step)
        if self.regression_head is not None:
            log_rv_pred = self.regression_head(pooled_step).squeeze(-1)
            multi_task["log_rv"] = log_rv_pred
        for name in self.rates_heads_active:
            bps_pred = self.rates_regression_heads[name](pooled_step).squeeze(-1)
            multi_task[f"rates_{name}_bps"] = bps_pred
            if name in self.rates_classification_heads:
                cls_logits = self.rates_classification_heads[name](pooled_step)
                multi_task[f"rates_{name}_cls_logits"] = cls_logits
        return multi_task


__all__ = ["ForecasterServingModel"]
