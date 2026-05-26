"""Flat-MLP forecaster (issue #327 Arm B).

The recurrent forecaster broadcasts the pooled text vector across every
input bar of the lookback window, so the sequence-model capacity is
unused for the text signal. Arm B drops the sequence wrap entirely:
``ForecasterFlatMLP`` mean-pools the market window across the sequence
axis and feeds the resulting flat vector ``[pooled_market
|| pooled_text_adapter || rich_static]`` through a two-layer MLP into
the same head shapes the recurrent forecaster mounts (regression head
``(close, softplus(vol))``, multi-task head with optional log-RV
regression branch).

This is the honest comparator for the broadcast-static framing.  If
this module's metrics match or exceed the recurrent forecaster on the
canonical fold protocol, the sequence-model framing of the text path
retires (per issue #327 acceptance).

Shape contract
---------------

Input ``x`` matches the recurrent forecaster: ``(B, T, input_size)``
where ``input_size`` is the per-bar scalar feature width. The model
mean-pools the sequence axis to ``(B, input_size)`` before
concatenating the text adapter output and the (optionally) zero-padded
credibility broadcast. The forward signature accepts the same optional
text / chunk / credibility kwargs as the recurrent class so the
training loop dispatches identically; the recurrent-only kwargs
(``chunks``, ``elapsed_days``, ``chunk_mask``) are accepted-and-ignored
to keep the call-site narrow.
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
from app.models.multi_task_head import MultiTaskHead
from app.models.rates_heads import RATES_HEAD_N_CLASSES, RATES_HEAD_NAMES
from app.models.text_embedding_adapter import TextEmbeddingAdapter


class ForecasterFlatMLP(nn.Module):
    """Sequence-wrap-free comparator for the broadcast-static text path."""

    def __init__(
        self,
        input_size: int = FEATURE_SIZE,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        head_hidden_size: int = DEFAULT_HEAD_HIDDEN_SIZE,
        initial_decay_rate: float = DEFAULT_INITIAL_DECAY_RATE,
        *,
        model_type: str = "flat_mlp",
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
    ) -> None:
        super().__init__()
        if output_mode not in {"regression", "classification"}:
            raise ValueError(
                f"Unknown output_mode: {output_mode!r}. "
                "Allowed: regression, classification"
            )
        if head_mode not in {"classification", "regression", "dual"}:
            raise ValueError(
                f"Unknown head_mode: {head_mode!r}. "
                "Allowed: classification, regression, dual"
            )
        # The chunk / LLM pooler is intentionally NOT mounted here:
        # ``flat_mlp`` is the honest comparator for the broadcast-static
        # text path, not the chunk/LLM variant. Callers wiring the
        # gated-InfoNCE / chunk pooler should stay on the recurrent
        # forecaster.
        if use_chunk_attention or use_llm_embeddings:
            raise ValueError(
                "ForecasterFlatMLP does not support the chunk / LLM pooler. "
                "Disable use_chunk_attention and use_llm_embeddings."
            )
        # Arm B's whole point is "no sequence wrap"; ``text_channel
        # == 'per_bar'`` is meaningless because there is no per-bar
        # axis to project against. Reject so a misconfigured run fails
        # fast rather than silently collapsing to the broadcast path.
        if text_channel == "per_bar":
            raise ValueError(
                "ForecasterFlatMLP rejects text_channel='per_bar'; "
                "use --architecture lstm --text-channel per_bar for Arm A."
            )
        self.model_type = model_type
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.head_hidden_size = int(head_hidden_size)
        self.initial_decay_rate = float(initial_decay_rate)
        self.use_time_decay = bool(use_time_decay)
        self.use_chunk_attention = False
        self.use_llm_embeddings = False
        self.text_channel = text_channel
        self.credibility_features = bool(credibility_features)
        self.credibility_dim = CREDIBILITY_FEATURE_DIM if self.credibility_features else 0
        self.chunk_embedding_size = int(chunk_embedding_size)
        self.chunk_projection_dim = 0
        self.text_embedding_dim = int(text_embedding_dim or 0)
        self.text_adapter_dim = int(text_adapter_dim or 0)
        self._text_path_active = (
            self.text_embedding_dim > 0 and self.text_adapter_dim > 0
        )
        if self._text_path_active:
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
        # Backbone -- two-layer MLP on the flat input vector. The hidden
        # width mirrors the recurrent forecaster's ``hidden_size`` so a
        # parameter-count comparison stays roughly comparable.
        flat_input_size = self.input_size + self.credibility_dim + self.text_path_dim
        self.flat_input_size = flat_input_size
        self.backbone = nn.Sequential(
            nn.Linear(flat_input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.output_mode = output_mode
        self.n_classes = int(n_classes)
        self.vol_regime_quantiles = tuple(float(v) for v in vol_regime_quantiles or ())
        self.vol_regime_target = str(vol_regime_target or "forward_realized_vol_10d")
        self.head_mode = str(head_mode)
        # Head dispatch -- mirrors the research class so the runner /
        # training loop can swap architectures without re-pluming the
        # downstream metrics.
        if output_mode == "classification":
            self.head: nn.Module = MultiTaskHead(
                hidden_size=self.hidden_size,
                head_hidden_size=self.head_hidden_size,
                dropout=self.dropout,
                stance_classes=self.n_classes,
            )
            if self.head_mode in {"regression", "dual"}:
                self.regression_head: nn.Module | None = nn.Sequential(
                    nn.LayerNorm(self.hidden_size),
                    nn.Linear(self.hidden_size, self.head_hidden_size),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.head_hidden_size, 1),
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
                    nn.LayerNorm(self.hidden_size),
                    nn.Linear(self.hidden_size, self.head_hidden_size),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.head_hidden_size, 1),
                )
                self.rates_classification_heads[name] = nn.Sequential(
                    nn.LayerNorm(self.hidden_size),
                    nn.Linear(self.hidden_size, self.head_hidden_size),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.head_hidden_size, RATES_HEAD_N_CLASSES),
                )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, self.head_hidden_size),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.head_hidden_size, 2),
            )
            self.regression_head = None
            self.rates_heads_active = ()
            self.rates_regression_heads = nn.ModuleDict()
            self.rates_classification_heads = nn.ModuleDict()

    def _flatten(
        self,
        x: torch.Tensor,
        credibility: torch.Tensor | None,
        text_embedding: torch.Tensor | None,
        text_embedding_missing: torch.Tensor | None,
    ) -> torch.Tensor:
        # Mean-pool over the sequence axis so the recurrence dimension
        # is collapsed. The recurrent forecaster's last-step pool would
        # give the model lookahead access; mean-pool stays even-handed
        # across the lookback window.
        if x.dim() != 3:
            raise ValueError(
                f"ForecasterFlatMLP expects a 3D market tensor (B, T, F); got {tuple(x.shape)}"
            )
        pooled_market = x.mean(dim=1)
        parts: list[torch.Tensor] = [pooled_market]
        if self.credibility_features:
            if credibility is None:
                raise ValueError(
                    "ForecasterFlatMLP requires `credibility` when credibility_features=True"
                )
            if credibility.dim() == 1:
                credibility = credibility.unsqueeze(0)
            if credibility.shape[-1] != self.credibility_dim:
                raise ValueError(
                    f"credibility tensor must have shape (..., {self.credibility_dim}); "
                    f"got {tuple(credibility.shape)}"
                )
            parts.append(credibility.to(pooled_market.dtype))
        if self._text_path_active:
            if self.text_adapter is None:
                raise RuntimeError(
                    "text_adapter not initialised but text-embedding path is active"
                )
            if text_embedding is None:
                raise ValueError(
                    "ForecasterFlatMLP requires `text_embedding` when text_adapter_dim > 0"
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
            parts.append(torch.cat([projected, missing_column], dim=-1))
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        chunks: torch.Tensor | None = None,  # noqa: ARG002 -- accepted for call-site parity
        elapsed_days: torch.Tensor | None = None,  # noqa: ARG002 -- accepted for call-site parity
        chunk_mask: torch.Tensor | None = None,  # noqa: ARG002 -- accepted for call-site parity
        credibility: torch.Tensor | None = None,
        text_embedding: torch.Tensor | None = None,
        text_embedding_missing: torch.Tensor | None = None,
        text_embedding_per_bar: torch.Tensor | None = None,  # noqa: ARG002 -- rejected at ctor
    ) -> torch.Tensor:
        flat = self._flatten(x, credibility, text_embedding, text_embedding_missing)
        pooled_step = self.backbone(flat)
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
        chunks: torch.Tensor | None = None,  # noqa: ARG002
        elapsed_days: torch.Tensor | None = None,  # noqa: ARG002
        chunk_mask: torch.Tensor | None = None,  # noqa: ARG002
        credibility: torch.Tensor | None = None,
        text_embedding: torch.Tensor | None = None,
        text_embedding_missing: torch.Tensor | None = None,
        text_embedding_per_bar: torch.Tensor | None = None,  # noqa: ARG002
    ) -> dict[str, torch.Tensor]:
        """Return the full multi-task / regression / rates dict."""
        if self.output_mode != "classification":
            raise RuntimeError(
                "forward_multi_task requires output_mode='classification'"
            )
        flat = self._flatten(x, credibility, text_embedding, text_embedding_missing)
        pooled_step = self.backbone(flat)
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


__all__ = ["ForecasterFlatMLP"]
