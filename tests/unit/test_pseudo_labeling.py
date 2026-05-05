from __future__ import annotations

import pytest

from app.data import pseudo_labeling


def test_parse_args_requires_teacher_checkpoint() -> None:
    with pytest.raises(SystemExit):
        pseudo_labeling._parse_args([])


def test_parse_args_accepts_threshold_flag() -> None:
    args = pseudo_labeling._parse_args(
        [
            "--teacher-checkpoint",
            "/some/path",
            "--threshold",
            "0.95",
        ]
    )
    assert args.teacher_checkpoint == "/some/path"
    assert args.threshold == 0.95


def test_parse_args_default_threshold_is_0_85() -> None:
    args = pseudo_labeling._parse_args(
        ["--teacher-checkpoint", "/some/path"]
    )
    assert args.threshold == 0.85
    assert args.teacher_model_id == "fomc_roberta_s71"
    assert args.teacher_model_version == "phase4_finetune_v1"
