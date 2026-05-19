"""Forecaster architecture factory.

The six architectures (``lstm``, ``lstm_attn``, ``gru``, ``tcn``,
``transformer``, ``dlinear``) are all wired through the same
:class:`app.models.lstm.ForecasterModel` wrapper via its ``model_type``
constructor argument — the wrapper handles per-architecture core selection
plus the shared optional ``TimeDecayAttention`` /
``RecurrentSequenceAttention`` / ``ChunkAttentionPooler`` /
credibility-features paths. The factory exists so callers (sweep harness,
training loop, tests) get a single dispatch entry point that does not have
to know the wrapper's internal kwarg layout, and so the public
``ModelConfig.architecture`` field is the only thing they have to set.

The default ``architecture="lstm"`` path is byte-identical to the previous
``ForecasterModel()`` construction so the determinism regression at
``tests/regression/test_forecaster_determinism.py`` stays green.
"""

from __future__ import annotations

from typing import Any

from app.models.config import FORECASTER_ARCHITECTURES, ModelConfig
from app.models.lstm import ForecasterModel


def build_forecaster(config: ModelConfig | dict[str, Any]) -> ForecasterModel:
    """Build a forecaster module for ``config.architecture``.

    All architectures share the same input contract ``(batch, seq_len, 6)``
    and output contract ``(batch, 2)`` (close, volatility).  The wrapper
    dispatches the recurrent core internally via ``model_type``; this
    function exists to translate the public ``architecture`` field to the
    wrapper's internal ``model_type`` argument and to raise a clear error
    when the architecture string is unknown.
    """

    resolved = config if isinstance(config, ModelConfig) else ModelConfig(**config)
    architecture = resolved.architecture
    if architecture not in FORECASTER_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture: {architecture!r}. "
            f"Allowed: {sorted(FORECASTER_ARCHITECTURES)}"
        )

    kwargs = resolved.to_dict()
    kwargs.pop("architecture", None)
    # Phase 9 V2 (#195) loader-side fields are consumed by the
    # training-loop loss dispatch + the per-fold quantile cutoff
    # fitter, not by ``ForecasterModel.__init__``. Strip them before
    # forwarding so the wrapper's ``**kwargs`` contract stays
    # unchanged. ``output_mode`` and ``n_classes`` ARE forwarded;
    # the wrapper now dispatches the output-head shape on them.
    for loader_only_field in ("vol_regime_quantiles", "vol_regime_target"):
        kwargs.pop(loader_only_field, None)
    return ForecasterModel(model_type=architecture, **kwargs)


__all__ = ["build_forecaster"]
