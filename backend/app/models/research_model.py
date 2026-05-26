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
)
from app.models.forecaster_base import ForecasterBase, prepare_recurrent_input
from app.models.multi_task_head import MultiTaskHead
from app.models.rates_heads import RATES_HEAD_N_CLASSES, RATES_HEAD_NAMES


class ForecasterResearchModel(ForecasterBase):
    """Research-side forecaster carrying every knob."""

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
        head_mode: str = "classification",
        rates_heads: tuple[str, ...] = (),
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
        x = prepare_recurrent_input(
            self,
            x,
            chunks=chunks,
            elapsed_days=elapsed_days,
            chunk_mask=chunk_mask,
            credibility=credibility,
            text_embedding=text_embedding,
            text_embedding_missing=text_embedding_missing,
        )
        pooled_step = self._encode(x)
        if self.output_mode == "classification":
            multi_task = self.head(pooled_step)
            stashed: dict[str, torch.Tensor] = {
                key: tensor.detach() for key, tensor in multi_task.items()
            }
            if (
                self.regression_head is not None
                and not bool(getattr(self, "_skip_regression_head", False))
            ):
                log_rv_pred = self.regression_head(pooled_step).squeeze(-1)
                stashed["log_rv"] = log_rv_pred.detach()
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
        )
        pooled_step = self._encode(x)
        multi_task: dict[str, torch.Tensor] = self.head(pooled_step)
        if (
            self.regression_head is not None
            and not bool(getattr(self, "_skip_regression_head", False))
        ):
            log_rv_pred = self.regression_head(pooled_step).squeeze(-1)
            multi_task["log_rv"] = log_rv_pred
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
