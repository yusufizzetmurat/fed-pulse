from __future__ import annotations

from torch import Tensor, nn


class EmbeddingAdapter(nn.Module):
    """Adapter that lifts pooled chunk embeddings into a 128-d projection.

    Replaces the legacy `nn.Linear(768, 8)` bottleneck used in earlier ablation
    runs. Layer order is Linear → LayerNorm → GELU. The linear stage is
    zero-initialised so the v1 baseline forward pass is recovered exactly when
    the text channel is freshly activated; the adapter only departs from that
    subspace if gradients show the text signal lowers loss.
    """

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 128,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.linear = nn.Linear(self.input_dim, self.output_dim, bias=True)
        self.norm = nn.LayerNorm(self.output_dim)
        self.activation = nn.GELU()
        if zero_init:
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    @property
    def out_features(self) -> int:
        return self.output_dim

    def forward(self, pooled: Tensor) -> Tensor:
        projected = self.linear(pooled)
        normalised = self.norm(projected)
        return self.activation(normalised)
