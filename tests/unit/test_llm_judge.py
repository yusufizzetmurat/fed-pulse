from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.data import llm_judge


class _StubGeminiModel:
    """Returns a list of response texts in order."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)

    def generate_content(self, prompt, **kwargs):
        class _R:
            def __init__(self, text):
                self.text = text

        return _R(self._responses.pop(0))


def _write_pseudo_fixture(path: Path) -> None:
    rows = [
        {
            "record_id": "r1",
            "source": "scraped_fed",
            "source_type": "fomc_minutes",
            "event_date": "2024-01-31",
            "title": "FOMC Minutes",
            "text": "Hawkish passage about tightening.",
            "label": "hawkish",
            "label_origin": "pseudo",
            "teacher_model_id": "fomc_roberta_s71",
            "teacher_model_version": "phase4_finetune_v1",
            "teacher_max_score": 0.78,
            "teacher_scores": {"hawkish": 0.78, "dovish": 0.12, "neutral": 0.10},
        },
        {
            "record_id": "r2",
            "source": "scraped_fed",
            "source_type": "fomc_minutes",
            "event_date": "2024-03-20",
            "title": "FOMC Minutes",
            "text": "Mixed signals on growth.",
            "label": "neutral",
            "label_origin": "pseudo",
            "teacher_model_id": "fomc_roberta_s71",
            "teacher_model_version": "phase4_finetune_v1",
            "teacher_max_score": 0.81,
            "teacher_scores": {"hawkish": 0.10, "dovish": 0.09, "neutral": 0.81},
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_run_judge_persists_judge_label_and_confidence_per_row(tmp_path: Path) -> None:
    input_path = tmp_path / "registry_pseudo.jsonl"
    output_path = tmp_path / "registry_pseudo_judged.jsonl"
    _write_pseudo_fixture(input_path)

    model = _StubGeminiModel(
        [
            '{"label": "hawkish", "confidence": 0.95}',
            '{"label": "neutral", "confidence": 0.62}',
        ]
    )

    written = llm_judge.run_judge(
        input_path=input_path,
        output_path=output_path,
        gemini_model=model,
        judge_model_id="gemini-2.5-pro",
        judge_model_version="20250505_v1",
    )

    assert written == 2
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["judge_label"] == "hawkish"
    assert rows[0]["judge_confidence"] == pytest.approx(0.95)
    assert rows[0]["judge_model_id"] == "gemini-2.5-pro"
    assert rows[0]["judge_model_version"] == "20250505_v1"
    # Original teacher fields are preserved
    assert rows[0]["label"] == "hawkish"
    assert rows[0]["teacher_model_id"] == "fomc_roberta_s71"


def test_parse_args_requires_input() -> None:
    with pytest.raises(SystemExit):
        llm_judge._parse_args([])


def test_parse_args_default_judge_model_is_gemini_2_5_pro() -> None:
    args = llm_judge._parse_args(["--input", "/some/path"])
    assert args.judge_model == "gemini-2.5-pro"
