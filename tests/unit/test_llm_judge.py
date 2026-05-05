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


def _judged_row(label, max_score, judge_label, judge_conf, **rest):
    base = {
        "label": label,
        "teacher_max_score": max_score,
        "judge_label": judge_label,
        "judge_confidence": judge_conf,
    }
    base.update(rest)
    return base


def test_gating_policy_confidence_only_keeps_above_tau() -> None:
    rows = [
        _judged_row("hawkish", 0.92, "neutral", 0.50),
        _judged_row("dovish", 0.40, "dovish", 0.95),
    ]
    kept = llm_judge.apply_gating_policy(rows, policy="confidence_only", tau=0.85)
    assert len(kept) == 1
    assert kept[0]["label"] == "hawkish"


def test_gating_policy_confidence_and_judge_requires_agreement() -> None:
    rows = [
        _judged_row("hawkish", 0.92, "hawkish", 0.50),  # agree at tau=0.85 -> keep
        _judged_row("hawkish", 0.92, "neutral", 0.99),  # disagree -> drop
        _judged_row("dovish", 0.40, "dovish", 0.99),    # below tau -> drop
    ]
    kept = llm_judge.apply_gating_policy(rows, policy="confidence_and_judge", tau=0.85)
    assert len(kept) == 1
    assert kept[0]["judge_label"] == "hawkish"


def test_gating_policy_judge_only_keeps_when_judge_label_present() -> None:
    rows = [
        _judged_row("hawkish", 0.40, "hawkish", 0.99),  # judge confident -> keep
        _judged_row("hawkish", 0.92, "neutral", 0.99),  # judge says neutral but valid -> keep
        _judged_row("hawkish", 0.92, "", 0.0),          # judge empty -> drop
    ]
    kept = llm_judge.apply_gating_policy(rows, policy="judge_only", tau=0.85)
    assert len(kept) == 2


def test_gating_policy_unknown_raises() -> None:
    with pytest.raises(ValueError):
        llm_judge.apply_gating_policy([], policy="bogus", tau=0.85)


def test_sample_audit_set_is_stratified_by_teacher_label_and_size_n(tmp_path: Path) -> None:
    rows = []
    for label, count in [("hawkish", 100), ("dovish", 30), ("neutral", 30)]:
        for idx in range(count):
            rows.append(
                _judged_row(
                    label,
                    0.9,
                    label,
                    0.9,
                    record_id=f"{label}_{idx}",
                    text=f"text {label} {idx}",
                )
            )

    audit = llm_judge.sample_audit_set(rows, n=60, seed=11)

    assert len(audit) == 60
    counts = {label: sum(1 for r in audit if r["label"] == label) for label in ("hawkish", "dovish", "neutral")}
    assert all(c > 0 for c in counts.values())
    assert sum(counts.values()) == 60


def test_audit_metrics_returns_teacher_and_judge_precision_against_human(tmp_path: Path) -> None:
    audit_rows = [
        # human, teacher, judge
        {"human_label": "hawkish", "label": "hawkish", "judge_label": "hawkish"},
        {"human_label": "hawkish", "label": "hawkish", "judge_label": "neutral"},
        {"human_label": "dovish", "label": "neutral", "judge_label": "dovish"},
        {"human_label": "neutral", "label": "neutral", "judge_label": "neutral"},
    ]
    metrics = llm_judge.audit_metrics(audit_rows)

    assert metrics["teacher_accuracy"] == pytest.approx(0.75)
    assert metrics["judge_accuracy"] == pytest.approx(0.75)
    assert "cohen_kappa" in metrics
    assert "teacher_per_class" in metrics
    assert "judge_per_class" in metrics
    assert "hawkish" in metrics["teacher_per_class"]


def test_audit_metrics_handles_empty_human_labels(tmp_path: Path) -> None:
    """Until the human pass is done, audit rows have empty human_label;
    the metrics computer must skip those rows rather than raise."""

    audit_rows = [
        {"human_label": "", "label": "hawkish", "judge_label": "hawkish"},
        {"human_label": "hawkish", "label": "hawkish", "judge_label": "hawkish"},
    ]
    metrics = llm_judge.audit_metrics(audit_rows)
    assert metrics["audit_size"] == 1
    assert metrics["teacher_accuracy"] == pytest.approx(1.0)


def test_write_audit_csv_includes_empty_human_label_column(tmp_path: Path) -> None:
    sample = [
        {"record_id": "r1", "label": "hawkish", "judge_label": "hawkish", "text": "Some text"},
    ]
    out = tmp_path / "audit_set.csv"
    llm_judge.write_audit_csv(sample, out)
    content = out.read_text(encoding="utf-8")
    assert "human_label" in content.splitlines()[0]
    assert "Some text" in content


def test_summarise_gating_policies_returns_yield_per_policy() -> None:
    rows = [
        _judged_row("hawkish", 0.92, "hawkish", 0.95),
        _judged_row("hawkish", 0.92, "neutral", 0.95),
        _judged_row("dovish", 0.40, "dovish", 0.95),
    ]
    summary = llm_judge.summarise_gating_policies(rows, tau=0.85)
    assert summary["confidence_only"]["kept"] == 2
    assert summary["confidence_and_judge"]["kept"] == 1
    assert summary["judge_only"]["kept"] == 3
    assert "label_distribution" in summary["confidence_only"]
