from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from app.models.config import (
    DEFAULT_CHUNK_DECAY_RATE,
    DEFAULT_INITIAL_DECAY_RATE,
    ELAPSED_TIME_FEATURE_INDEX,
    SENTIMENT_FEATURE_INDEX,
)


class TimeDecayAttention(nn.Module):
    """Dampens the sentiment feature by exp(-lambda * |elapsed_time|) per timestep.

    lambda = softplus(raw_lambda) so it stays non-negative while remaining
    smoothly differentiable. elapsed_time is read from feature index 5 of the
    input tensor (days between the FOMC document and the record, normalized by
    30 upstream). The absolute value makes decay symmetric in time so past bars
    (elapsed < 0) attenuate rather than amplify.
    """

    def __init__(self, initial_decay_rate: float = DEFAULT_INITIAL_DECAY_RATE):
        super().__init__()
        raw_init = math.log(math.expm1(max(float(initial_decay_rate), 1e-6)))
        self.raw_lambda = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

    @property
    def decay_rate(self) -> torch.Tensor:
        return F.softplus(self.raw_lambda)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        elapsed = x[..., ELAPSED_TIME_FEATURE_INDEX].abs()
        decay = torch.exp(-self.decay_rate * elapsed).unsqueeze(-1)
        feature_mask = torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)
        feature_mask[SENTIMENT_FEATURE_INDEX] = 1.0
        scale = (1.0 - feature_mask) + decay * feature_mask
        return x * scale


class ChunkAttentionPooler(nn.Module):
    """Variant B from the Phase 4 attention plan: attention over chunk embeddings
    with values multiplicatively damped by exp(-lambda * |elapsed_days|).

    A single learnable query token attends to the chunk embeddings (keys), and
    the values are decayed by chunk-elapsed-time before the weighted sum. Output
    is a single pooled vector per document plus the attention weights and decay
    coefficients for downstream UI/visualization.
    """

    def __init__(
        self,
        embedding_size: int,
        initial_decay_rate: float = DEFAULT_CHUNK_DECAY_RATE,
    ):
        super().__init__()
        self.embedding_size = int(embedding_size)
        raw_init = math.log(math.expm1(max(float(initial_decay_rate), 1e-6)))
        self.raw_lambda = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        self.q_proj = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.k_proj = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.v_proj = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.query_token = nn.Parameter(torch.zeros(self.embedding_size))
        nn.init.normal_(self.query_token, std=0.02)

    @property
    def decay_rate(self) -> torch.Tensor:
        return F.softplus(self.raw_lambda)

    def forward(
        self,
        embeddings: torch.Tensor,
        elapsed_days: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unbatched = embeddings.dim() == 2
        if unbatched:
            embeddings = embeddings.unsqueeze(0)
            elapsed_days = elapsed_days.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        batch_size, num_chunks, dim = embeddings.shape
        if dim != self.embedding_size:
            raise ValueError(
                f"ChunkAttentionPooler embedding_size={self.embedding_size} "
                f"does not match input dim={dim}"
            )

        q = self.q_proj(self.query_token).expand(batch_size, dim)
        k = self.k_proj(embeddings)
        v = self.v_proj(embeddings)

        decay_coeffs = torch.exp(-self.decay_rate * elapsed_days.abs())
        if mask is not None:
            decay_coeffs = decay_coeffs * mask
        v_decayed = v * decay_coeffs.unsqueeze(-1)

        scores = torch.einsum("bd,bnd->bn", q, k) / math.sqrt(dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        # If a row was fully masked, softmax produces NaNs — zero them out.
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = torch.einsum("bn,bnd->bd", weights, v_decayed)

        if unbatched:
            return pooled.squeeze(0), weights.squeeze(0), decay_coeffs.squeeze(0)
        return pooled, weights, decay_coeffs
