"""Trajectory model architectures + persistence (#296).

Two architectures share the same input / output contract so the train
script can A/B them on the same fold without bespoke wiring per side:

* :class:`TrajectoryLSTM` — single-direction LSTM baseline. Two layers,
  64-dim hidden. The conservative baseline against which the Transformer
  arm is graded.
* :class:`TrajectoryTransformer` — small encoder-only Transformer.
  Four layers, 64-dim model, 4 attention heads. Learned positional
  embeddings because the sequence is short (12 meetings) and a
  sinusoidal table buys little at that scale.

Both ingest a ``(B, T, D_in)`` float tensor where ``T`` is the meeting
history length and ``D_in = embedding_dim + market_feature_dim``. The
per-meeting market block carries the two #291 pre-meeting columns
(``pre_meeting_trailing_2y_yield_change_5d_bps`` and a VIX proxy off
``cross_asset.vix_close``) plus a leading bias term, so the dim is
fixed at three for callers that follow :func:`market_feature_vector`.

Padding is handled via a companion ``(B, T)`` bool ``mask`` where
``True`` marks a real meeting and ``False`` marks a left-pad slot.

The forward pass returns ``(logits, hidden_state)``:

* ``logits`` is ``(B, n_classes)`` — log-odds over the next meeting's
  stance, decoded from the final non-padded position.
* ``hidden_state`` is ``(B, hidden_dim)`` — the pooled context vector
  the train loop also exposes as the projected-next semantic anchor.

Persistence is plain ``torch.save`` of a state-dict + an architecture
tag string so :func:`load_model` can rebuild without inspecting the
checkpoint. Manifests are written by the trainer (see
:mod:`app.trajectory.train`); this module deliberately does not touch
the filesystem.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

# --- Constants --------------------------------------------------------------

# 3-class stance head: matches the canonical {hawkish, dovish, neutral}
# label set documented in docs/data-and-training-contracts.md so the
# trajectory head can be A/B'd against the per-statement classifier
# without label-space surgery.
STANCE_CLASSES: tuple[str, str, str] = ("hawkish", "dovish", "neutral")
N_STANCE_CLASSES: int = len(STANCE_CLASSES)

# Default history length per the issue (~1.5 years of FOMC meetings,
# which is 8 scheduled + ~4 intermeeting in the modern era).
DEFAULT_HISTORY_LENGTH: int = 12

# Default architecture knobs. The Transformer side intentionally stays
# small — the corpus has ~250 historical statements, so widening the
# model trades training stability for a parameter count the data
# cannot warrant.
DEFAULT_LSTM_HIDDEN: int = 64
DEFAULT_LSTM_LAYERS: int = 2
DEFAULT_TRANSFORMER_LAYERS: int = 4
DEFAULT_TRANSFORMER_D_MODEL: int = 64
DEFAULT_TRANSFORMER_N_HEADS: int = 4
DEFAULT_DROPOUT: float = 0.1

# Per-meeting market feature dim: [bias, trailing_2y_yield_change_5d_bps,
# vix_close_z]. Fixed so callers that build inputs via
# :func:`market_feature_vector` stay schema-stable across rebuilds.
MARKET_FEATURE_DIM: int = 3

Architecture = Literal["lstm", "transformer"]


@dataclass(frozen=True)
class TrajectoryConfig:
    """Hyperparameter envelope for both architectures."""

    architecture: Architecture
    embedding_dim: int
    market_feature_dim: int = MARKET_FEATURE_DIM
    history_length: int = DEFAULT_HISTORY_LENGTH
    n_classes: int = N_STANCE_CLASSES
    dropout: float = DEFAULT_DROPOUT
    # LSTM-only knobs.
    lstm_hidden: int = DEFAULT_LSTM_HIDDEN
    lstm_layers: int = DEFAULT_LSTM_LAYERS
    # Transformer-only knobs.
    transformer_layers: int = DEFAULT_TRANSFORMER_LAYERS
    transformer_d_model: int = DEFAULT_TRANSFORMER_D_MODEL
    transformer_n_heads: int = DEFAULT_TRANSFORMER_N_HEADS

    @property
    def input_dim(self) -> int:
        return int(self.embedding_dim + self.market_feature_dim)

    @property
    def hidden_dim(self) -> int:
        """Width of the pooled context vector returned by ``forward``.

        Equal to ``lstm_hidden`` for the LSTM arm and
        ``transformer_d_model`` for the Transformer arm so a
        downstream consumer (e.g. the projection layer in the runtime
        singleton) does not have to branch on architecture.
        """

        if self.architecture == "lstm":
            return int(self.lstm_hidden)
        return int(self.transformer_d_model)

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "embedding_dim": int(self.embedding_dim),
            "market_feature_dim": int(self.market_feature_dim),
            "history_length": int(self.history_length),
            "n_classes": int(self.n_classes),
            "dropout": float(self.dropout),
            "lstm_hidden": int(self.lstm_hidden),
            "lstm_layers": int(self.lstm_layers),
            "transformer_layers": int(self.transformer_layers),
            "transformer_d_model": int(self.transformer_d_model),
            "transformer_n_heads": int(self.transformer_n_heads),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrajectoryConfig":
        arch = str(payload.get("architecture", "lstm"))
        if arch not in ("lstm", "transformer"):
            raise ValueError(
                f"architecture must be 'lstm' or 'transformer', got {arch!r}"
            )
        return cls(
            architecture=arch,  # type: ignore[arg-type]
            embedding_dim=int(payload["embedding_dim"]),
            market_feature_dim=int(payload.get("market_feature_dim", MARKET_FEATURE_DIM)),
            history_length=int(payload.get("history_length", DEFAULT_HISTORY_LENGTH)),
            n_classes=int(payload.get("n_classes", N_STANCE_CLASSES)),
            dropout=float(payload.get("dropout", DEFAULT_DROPOUT)),
            lstm_hidden=int(payload.get("lstm_hidden", DEFAULT_LSTM_HIDDEN)),
            lstm_layers=int(payload.get("lstm_layers", DEFAULT_LSTM_LAYERS)),
            transformer_layers=int(
                payload.get("transformer_layers", DEFAULT_TRANSFORMER_LAYERS)
            ),
            transformer_d_model=int(
                payload.get("transformer_d_model", DEFAULT_TRANSFORMER_D_MODEL)
            ),
            transformer_n_heads=int(
                payload.get("transformer_n_heads", DEFAULT_TRANSFORMER_N_HEADS)
            ),
        )


# --- Feature builders -------------------------------------------------------


def market_feature_vector(
    *,
    trailing_2y_yield_change_5d_bps: float | None,
    vix_close: float | None,
    vix_mean: float = 20.0,
    vix_std: float = 8.0,
) -> np.ndarray:
    """Build the per-meeting market block as a ``(MARKET_FEATURE_DIM,)`` vector.

    The block is fixed-width by design so the train + inference paths
    can concatenate it onto the encoder embedding without reasoning
    about which #291 columns are present. ``None`` inputs degrade to
    zero so a missing pre-meeting row contributes a no-signal block
    rather than crashing the build.

    ``vix_close`` is z-scored with a pinned mean / std (default 20 / 8
    matches the long-run distribution of ^VIX on the train slice). The
    pinned moments mean the inference path does not have to ship a
    scaler artifact alongside the model checkpoint — the same constants
    apply at train + inference time.
    """

    def _finite(value: float | None) -> float:
        try:
            if value is None:
                return 0.0
            scalar = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(scalar):
            return 0.0
        return scalar

    bias = 1.0
    yield_bp = _finite(trailing_2y_yield_change_5d_bps)
    vix_raw = _finite(vix_close)
    denom = float(vix_std) if vix_std and vix_std > 1e-9 else 1.0
    vix_z = (vix_raw - float(vix_mean)) / denom if vix_raw > 0 else 0.0
    return np.asarray([bias, yield_bp, vix_z], dtype=np.float32)


def pad_sequence(
    embeddings: Sequence[np.ndarray],
    market_blocks: Sequence[np.ndarray],
    *,
    history_length: int = DEFAULT_HISTORY_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Left-pad two parallel meeting sequences into ``(T, D)`` + ``(T,)``.

    ``embeddings`` and ``market_blocks`` must align row-by-row; the
    function asserts equal length and returns the padded tensor stack
    plus a boolean mask that marks real meetings as ``True`` and the
    leading pad slots as ``False``. Padding goes at the front (most
    recent meeting last) so the model's final-position decoder always
    reads the most recent real meeting.

    The function clips sequences longer than ``history_length`` to the
    most recent ``history_length`` meetings — older meetings carry the
    least predictive value for next-meeting stance and dropping them
    keeps the input tensor dense.
    """

    if len(embeddings) != len(market_blocks):
        raise ValueError(
            "embeddings and market_blocks must align in length; "
            f"got {len(embeddings)} vs {len(market_blocks)}"
        )
    if history_length <= 0:
        raise ValueError(f"history_length must be positive; got {history_length}")
    if not embeddings:
        emb_dim = 1
        mkt_dim = MARKET_FEATURE_DIM
        zero_input = np.zeros(
            (history_length, emb_dim + mkt_dim), dtype=np.float32
        )
        zero_mask = np.zeros(history_length, dtype=bool)
        return zero_input, zero_mask

    embedding_dim = int(np.asarray(embeddings[0]).shape[-1])
    market_dim = int(np.asarray(market_blocks[0]).shape[-1])
    rows = list(zip(embeddings, market_blocks))
    if len(rows) > history_length:
        rows = rows[-history_length:]
    real_count = len(rows)
    pad_count = history_length - real_count
    input_dim = embedding_dim + market_dim
    padded = np.zeros((history_length, input_dim), dtype=np.float32)
    mask = np.zeros(history_length, dtype=bool)
    for offset, (emb, mkt) in enumerate(rows):
        slot = pad_count + offset
        emb_arr = np.asarray(emb, dtype=np.float32).reshape(-1)
        mkt_arr = np.asarray(mkt, dtype=np.float32).reshape(-1)
        if emb_arr.shape[0] != embedding_dim:
            raise ValueError(
                "inconsistent embedding dim across meetings; "
                f"expected {embedding_dim}, got {emb_arr.shape[0]}"
            )
        if mkt_arr.shape[0] != market_dim:
            raise ValueError(
                "inconsistent market block dim across meetings; "
                f"expected {market_dim}, got {mkt_arr.shape[0]}"
            )
        padded[slot, :embedding_dim] = emb_arr
        padded[slot, embedding_dim:] = mkt_arr
        mask[slot] = True
    return padded, mask


# --- Architectures ----------------------------------------------------------
#
# The torch import is deferred so the module loads cleanly under any
# import-time path that does not need to run a forward pass (typical
# pytest collection of unrelated tests).


def _import_torch() -> Any:
    import torch  # type: ignore[import-not-found,unused-ignore]

    return torch


def build_model(config: TrajectoryConfig) -> Any:
    """Construct a torch module matching ``config.architecture``.

    Returns the module in evaluation mode by default; the trainer
    flips it to train mode explicitly. The lazy torch import keeps the
    rest of the module usable in environments where torch is mocked
    out (some unit tests).
    """

    torch = _import_torch()

    if config.architecture == "lstm":
        return _LSTMTrajectoryModel(config, torch)
    return _TransformerTrajectoryModel(config, torch)


def _LSTMTrajectoryModel(config: TrajectoryConfig, torch_mod: Any) -> Any:
    nn = torch_mod.nn

    class _LSTM(nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.input_proj = nn.Linear(config.input_dim, config.lstm_hidden)
            self.lstm = nn.LSTM(
                input_size=config.lstm_hidden,
                hidden_size=config.lstm_hidden,
                num_layers=config.lstm_layers,
                dropout=config.dropout if config.lstm_layers > 1 else 0.0,
                batch_first=True,
            )
            self.dropout = nn.Dropout(config.dropout)
            self.head = nn.Linear(config.lstm_hidden, config.n_classes)

        def forward(
            self, inputs: Any, mask: Any | None = None
        ) -> tuple[Any, Any]:
            projected = self.input_proj(inputs)
            outputs, _ = self.lstm(projected)
            pooled = _final_real_position(outputs, mask, torch_mod)
            pooled = self.dropout(pooled)
            logits = self.head(pooled)
            return logits, pooled

    model = _LSTM()
    model.eval()
    return model


def _TransformerTrajectoryModel(config: TrajectoryConfig, torch_mod: Any) -> Any:
    nn = torch_mod.nn

    class _Transformer(nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.input_proj = nn.Linear(
                config.input_dim, config.transformer_d_model
            )
            # Learned positional embeddings — sequence is short so a
            # full table is cheaper than a sin/cos one and easier to
            # introspect.
            self.position_embedding = nn.Embedding(
                config.history_length, config.transformer_d_model
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.transformer_d_model,
                nhead=config.transformer_n_heads,
                dim_feedforward=config.transformer_d_model * 4,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=config.transformer_layers
            )
            self.dropout = nn.Dropout(config.dropout)
            self.head = nn.Linear(config.transformer_d_model, config.n_classes)

        def forward(
            self, inputs: Any, mask: Any | None = None
        ) -> tuple[Any, Any]:
            batch_size, seq_len, _ = inputs.shape
            projected = self.input_proj(inputs)
            positions = torch_mod.arange(seq_len, device=inputs.device)
            pos_embed = self.position_embedding(positions).unsqueeze(0)
            projected = projected + pos_embed
            key_padding_mask: Any | None = None
            if mask is not None:
                # ``TransformerEncoder`` expects ``True`` for positions to
                # IGNORE; our ``mask`` marks real positions as ``True``,
                # so flip the polarity here.
                key_padding_mask = ~mask.bool()
            encoded = self.encoder(
                projected, src_key_padding_mask=key_padding_mask
            )
            pooled = _final_real_position(encoded, mask, torch_mod)
            pooled = self.dropout(pooled)
            logits = self.head(pooled)
            return logits, pooled

    model = _Transformer()
    model.eval()
    return model


def _final_real_position(outputs: Any, mask: Any | None, torch_mod: Any) -> Any:
    """Gather the last non-pad position of each sequence in ``outputs``.

    Falls back to the last absolute position when ``mask`` is None
    (interpret-as-fully-real input). The mask convention matches
    :func:`pad_sequence`: ``True`` = real, ``False`` = pad.
    """

    if mask is None:
        return outputs[:, -1, :]
    counts = mask.long().sum(dim=1).clamp(min=1)
    last_idx = (counts - 1).long()
    batch_idx = torch_mod.arange(outputs.size(0), device=outputs.device)
    return outputs[batch_idx, last_idx, :]


# --- Persistence ------------------------------------------------------------


def save_model(model: Any, config: TrajectoryConfig, path: Path) -> None:
    """Persist a model + its config to a single ``.pt`` file.

    Writes via ``torch.save`` to a sibling ``.tmp`` path and renames so
    a crash mid-write never leaves a truncated checkpoint on disk.
    """

    import os

    torch = _import_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "config": config.to_dict(),
            "state_dict": model.state_dict(),
        },
        tmp,
    )
    os.replace(tmp, path)


def load_model(path: Path) -> tuple[Any, TrajectoryConfig]:
    """Reconstruct a model + config from a checkpoint written by ``save_model``."""

    torch = _import_torch()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"trajectory checkpoint not found: {path}")
    # weights_only=False so the optimizer / config dict round-trips,
    # but the file is project-internal so the trust boundary is the
    # same as for the encoder checkpoints.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = TrajectoryConfig.from_dict(payload["config"])
    model = build_model(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config


__all__ = [
    "Architecture",
    "DEFAULT_HISTORY_LENGTH",
    "DEFAULT_LSTM_HIDDEN",
    "DEFAULT_LSTM_LAYERS",
    "DEFAULT_TRANSFORMER_D_MODEL",
    "DEFAULT_TRANSFORMER_LAYERS",
    "DEFAULT_TRANSFORMER_N_HEADS",
    "MARKET_FEATURE_DIM",
    "N_STANCE_CLASSES",
    "STANCE_CLASSES",
    "TrajectoryConfig",
    "build_model",
    "load_model",
    "market_feature_vector",
    "pad_sequence",
    "save_model",
]
