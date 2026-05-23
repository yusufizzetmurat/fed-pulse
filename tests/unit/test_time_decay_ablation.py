"""Round 4 (#243) ablation: ``--no-time-decay`` short-circuits the
``TimeDecayAttention`` path.

The advisor mandated the elapsed-time decay mechanism but it has never
been measured against a clean post-embargo baseline; the existing
``use_time_decay`` kwarg on ``ForecasterModel`` is plumbed but has no
end-to-end CLI surface. This test pins the CLI + config wiring so the
ablation can be flipped from a sweep harness without code edits.
"""

from __future__ import annotations

from pathlib import Path

import pytest


_TRAIN_FORECASTER_PATH = (
    Path(__file__).resolve().parents[2] / "backend" / "app" / "train_forecaster.py"
)
_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "backend" / "app" / "models" / "config.py"
)
_LSTM_PATH = (
    Path(__file__).resolve().parents[2] / "backend" / "app" / "models" / "lstm.py"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_model_config_carries_use_time_decay_field() -> None:
    """``ModelConfig.use_time_decay`` must be a real field with default
    ``True`` so existing checkpoints deserialise into the legacy
    time-decay-on forward path."""

    source = _read(_CONFIG_PATH)
    assert "use_time_decay: bool = True" in source, (
        "ModelConfig is missing the use_time_decay field (default True)"
    )
    # ``from_model`` must round-trip the field from the constructed module.
    assert "use_time_decay=bool(getattr(model, \"use_time_decay\", True))" in source, (
        "ModelConfig.from_model does not round-trip use_time_decay"
    )


def test_forecaster_model_still_owns_the_time_decay_kwarg() -> None:
    """``ForecasterModel`` already accepts ``use_time_decay`` (predates
    this round). Pin the kwarg + guarded forward-path branch so the
    plumbed config field actually does something."""

    source = _read(_LSTM_PATH)
    assert "use_time_decay: bool = True" in source
    assert "self.use_time_decay = bool(use_time_decay)" in source
    assert "if self.use_time_decay:" in source


def test_train_forecaster_exposes_no_time_decay_flag() -> None:
    """The CLI must expose ``--no-time-decay`` so the ablation flips
    from the sweep harness without source edits."""

    source = _read(_TRAIN_FORECASTER_PATH)
    assert "\"--no-time-decay\"" in source, "missing --no-time-decay CLI flag"
    assert "dest=\"use_time_decay\"" in source, (
        "--no-time-decay does not write into args.use_time_decay"
    )
    assert "parser.set_defaults(use_time_decay=True)" in source, (
        "default for --no-time-decay is not 'time decay on'"
    )


def test_model_config_construction_threads_use_time_decay() -> None:
    """Every ``ModelConfig(`` construction site in train_forecaster must
    thread ``use_time_decay`` so the sweep + single-run paths honour
    the CLI flag."""

    source = _read(_TRAIN_FORECASTER_PATH)
    # Three construction sites: _build_model_config, the random-search
    # candidate builder, and the exhaustive candidate builder.
    occurrences = source.count(
        "use_time_decay=bool(getattr(args, \"use_time_decay\", True))"
    )
    assert occurrences >= 3, (
        "use_time_decay is not threaded into every ModelConfig construction "
        f"site (found {occurrences}, expected >= 3)"
    )
