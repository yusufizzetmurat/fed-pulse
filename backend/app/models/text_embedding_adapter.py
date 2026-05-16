"""Encoder-agnostic adapter for pooled FOMC statement embeddings.

The forecaster consumes pooled text embeddings as a 5th feature family on
top of the 35-dim rich-feature scalar input. The pooling logic (time-
decay weighted mean over the prior-4 statement embeddings) lives in
``app.training.loaders`` and produces a single ``in_dim``-vector per
event. This module projects that vector to a fixed ``out_dim`` so the
recurrent core sees the same per-bar feature size regardless of which
encoder (FinBERT 768, voyage-finance-2 1024, BGE 1024, etc.) emitted
the pooled embedding.

Layer order is ``Linear -> LayerNorm -> GELU``. The linear stage is
zero-initialised so a freshly activated text-embedding path forwards to
the same point in feature space as the rich-features-only baseline; the
adapter only departs from that subspace if gradients show the text
signal reduces the loss. On ``in_dim`` mismatch the adapter projects
zeros (the loader will have set the missing flag to ``1.0``) and the
recurrent core sees the same zero-broadcast slot it gets when the
encoder is disabled.
"""

from __future__ import annotations

from torch import Tensor, nn


class TextEmbeddingAdapter(nn.Module):
    """Project a pooled text embedding from ``in_dim`` to ``out_dim``.

    Parameters
    ----------
    in_dim:
        Source embedding dim. FinBERT / FinBERT-FOMC / FinBERT-Fed-adjacent
        emit 768; BGE-large / nomic-embed / voyage-finance-2 emit 1024.
        The loader resolves the source dim from the embedding parquet
        itself; the model constructor pins ``in_dim`` so a mismatched
        run fails fast at the linear forward rather than silently
        truncating.
    out_dim:
        Projection target. Default 64. The forecaster sweep iterates over
        ``{32, 64, 128}`` so the diminishing-returns curve is visible in
        the aggregator table.
    zero_init:
        When ``True`` (default) the linear weight and bias are zeroed
        so the model starts byte-identical to the no-text-embedding
        baseline and only departs when gradients lower the loss.

    Forward contract
    ----------------
    Input  ``(batch, in_dim)``  -- the pooled embedding per event.
    Output ``(batch, out_dim)`` -- the projected vector the recurrent
    core broadcasts to every bar of the prior window + the event-day
    target frame.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 64,
        *,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be a positive integer; got {in_dim}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be a positive integer; got {out_dim}")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.linear = nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.norm = nn.LayerNorm(self.out_dim)
        self.activation = nn.GELU()
        if zero_init:
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    @property
    def out_features(self) -> int:
        return self.out_dim

    def forward(self, pooled: Tensor) -> Tensor:
        if pooled.dim() == 1:
            pooled = pooled.unsqueeze(0)
        if pooled.shape[-1] != self.in_dim:
            raise ValueError(
                f"TextEmbeddingAdapter expected input of shape (..., {self.in_dim}); "
                f"got {tuple(pooled.shape)}"
            )
        projected = self.linear(pooled)
        normalised = self.norm(projected)
        activated: Tensor = self.activation(normalised)
        return activated


__all__ = ["TextEmbeddingAdapter"]
