"""Unit tests for the Phase B ``--sequence-length`` CLI knob (#227).

The flag is persisted on ``ModelConfig.sequence_length`` so a future
data-prep step (re-built events.parquet with a wider prior-bars window)
can opt in. Today the flag is recorded on the checkpoint but not yet
consumed by the loader's window slicer — that follow-up PR rebuilds the
prior-bars window beyond 20 bars.
"""

from __future__ import annotations

import sys
from typing import Iterator

import pytest

from app.models.config import ModelConfig
from app.train_forecaster import _parse_args


@pytest.fixture
def parse_argv() -> Iterator:
    """Drive ``_parse_args()`` by patching ``sys.argv`` since the helper
    reads its arguments from the global module state, not from a
    parameter."""

    original = sys.argv

    def _do_parse(argv: list[str]):
        sys.argv = ["train_forecaster", *argv]
        try:
            return _parse_args()
        finally:
            sys.argv = original

    yield _do_parse
    sys.argv = original


def test_default_sequence_length_is_zero_meaning_module_constant() -> None:
    """``0`` is the documented sentinel for ``use the SEQUENCE_LENGTH
    module constant`` so the regression test stays byte-identical."""

    cfg = ModelConfig()
    assert cfg.sequence_length == 0


def test_model_config_serialises_sequence_length() -> None:
    cfg = ModelConfig(sequence_length=60)
    assert cfg.to_dict()["sequence_length"] == 60


def test_cli_parses_single_sequence_length_flag(parse_argv) -> None:
    args = parse_argv(
        [
            "--training-package-id",
            "demo",
            "--sequence-length",
            "60",
        ]
    )
    assert args.sequence_length == 60


def test_cli_parses_sweep_grid(parse_argv) -> None:
    args = parse_argv(
        [
            "--training-package-id",
            "demo",
            "--sequence-lengths",
            "20",
            "40",
            "60",
        ]
    )
    assert args.sequence_lengths == [20, 40, 60]


def test_cli_parses_lr_schedule(parse_argv) -> None:
    args = parse_argv(
        [
            "--training-package-id",
            "demo",
            "--lr-schedule",
            "cosine_warmup",
        ]
    )
    assert args.lr_schedule == "cosine_warmup"


def test_cli_parses_lr_schedules_grid(parse_argv) -> None:
    args = parse_argv(
        [
            "--training-package-id",
            "demo",
            "--lr-schedules",
            "plateau",
            "cosine_warmup",
        ]
    )
    assert args.lr_schedules == ["plateau", "cosine_warmup"]
