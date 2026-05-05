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


class _StubPipeline:
    """Stand-in for transformers.pipeline that the teacher loader returns."""

    def __init__(self, label_to_score: list[list[dict[str, float]]]):
        self._batches = label_to_score

    def __call__(self, texts, **kwargs):
        # Mirror the transformers `text-classification` shape with
        # return_all_scores=True: list of [list of {label, score}].
        out = []
        for _ in texts:
            out.append(self._batches.pop(0))
        return out


def test_score_passages_returns_one_prediction_per_passage_with_label_and_confidence() -> None:
    pipeline = _StubPipeline(
        [
            [
                {"label": "hawkish", "score": 0.92},
                {"label": "dovish", "score": 0.05},
                {"label": "neutral", "score": 0.03},
            ],
            [
                {"label": "hawkish", "score": 0.30},
                {"label": "dovish", "score": 0.40},
                {"label": "neutral", "score": 0.30},
            ],
        ]
    )

    predictions = pseudo_labeling.score_passages(
        ["Strong tightening signal.", "Mixed signals on the labor market."],
        pipeline=pipeline,
    )

    assert len(predictions) == 2
    assert predictions[0]["predicted_label"] == "hawkish"
    assert predictions[0]["max_score"] == pytest.approx(0.92)
    assert predictions[0]["scores"] == {"hawkish": 0.92, "dovish": 0.05, "neutral": 0.03}
    assert predictions[1]["predicted_label"] == "dovish"
    assert predictions[1]["max_score"] == pytest.approx(0.40)
