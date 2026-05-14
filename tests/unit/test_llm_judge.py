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


class _FlakyGeminiModel:
    """Stub model that raises on a configurable subset of calls.

    `fail_on_indices` is a set of call-indexes that raise; everything
    else returns the next canned response. Used to test the run_judge
    per-row try/except behaviour when the Gemini API returns 503/429
    mid-batch.
    """

    def __init__(self, responses: list[str], fail_on_indices: set[int]):
        self._responses = list(responses)
        self._fail_on = set(fail_on_indices)
        self._index = -1

    def generate_content(self, _prompt):  # pragma: no cover - matches SDK shape
        self._index += 1
        if self._index in self._fail_on:
            raise RuntimeError(f"503 UNAVAILABLE on call {self._index}")
        text = self._responses.pop(0)

        class _Response:
            def __init__(self, txt):
                self.text = txt

        return _Response(text)


def test_run_judge_writes_one_progress_line_per_row(tmp_path: Path) -> None:
    """A row-by-row progress writer collects one line per input row."""

    input_path = tmp_path / "registry_pseudo.jsonl"
    output_path = tmp_path / "judged.jsonl"
    _write_pseudo_fixture(input_path)

    model = _StubGeminiModel(
        [
            '{"label": "hawkish", "confidence": 0.95}',
            '{"label": "neutral", "confidence": 0.62}',
        ]
    )
    lines: list[str] = []
    written = llm_judge.run_judge(
        input_path=input_path,
        output_path=output_path,
        gemini_model=model,
        judge_model_id="gemini-2.5-pro",
        judge_model_version="20260514_v1",
        progress_writer=lines.append,
    )
    assert written == 2
    assert len(lines) == 2
    assert "1/2" in lines[0] and "hawkish" in lines[0]
    assert "2/2" in lines[1] and "neutral" in lines[1]


def test_run_judge_writes_incrementally_one_row_per_line(tmp_path: Path) -> None:
    """Each completed row hits disk before the next call — verified by
    flushing-after-write semantics. We check the file is non-empty
    immediately after a single-row run + the JSONL parses cleanly."""

    input_path = tmp_path / "registry_pseudo.jsonl"
    output_path = tmp_path / "judged.jsonl"
    _write_pseudo_fixture(input_path)

    model = _StubGeminiModel(
        [
            '{"label": "hawkish", "confidence": 0.95}',
            '{"label": "dovish", "confidence": 0.81}',
        ]
    )
    llm_judge.run_judge(
        input_path=input_path,
        output_path=output_path,
        gemini_model=model,
        judge_model_id="gemini-2.5-pro",
        judge_model_version="v1",
        progress_writer=lambda _line: None,
    )
    raw_lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 2
    rows = [json.loads(ln) for ln in raw_lines]
    assert rows[0]["judge_label"] == "hawkish"
    assert rows[1]["judge_label"] == "dovish"


def test_run_judge_continues_after_api_error_marks_row_with_blank_judge_label(tmp_path: Path) -> None:
    """A transient API failure on row 1 must NOT crash the run; the
    failing row lands with judge_label='' + judge_error=type-name, and
    row 2 still runs."""

    input_path = tmp_path / "registry_pseudo.jsonl"
    output_path = tmp_path / "judged.jsonl"
    _write_pseudo_fixture(input_path)

    model = _FlakyGeminiModel(
        responses=['{"label": "dovish", "confidence": 0.77}'],
        fail_on_indices={0},  # first call raises
    )
    lines: list[str] = []
    written = llm_judge.run_judge(
        input_path=input_path,
        output_path=output_path,
        gemini_model=model,
        judge_model_id="gemini-2.5-pro",
        judge_model_version="v1",
        progress_writer=lines.append,
    )
    assert written == 2  # both rows persisted (the first as a blank)
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["judge_label"] == ""
    assert rows[0]["judge_confidence"] == 0.0
    assert rows[0]["judge_error"] == "RuntimeError"
    assert rows[1]["judge_label"] == "dovish"
    # Progress writer reported the error
    assert any("ERROR" in line for line in lines)


def test_run_judge_resume_skips_record_ids_already_in_output(tmp_path: Path) -> None:
    """When resume=True and the output file already contains row 'r1',
    only the second row is scored on the re-run."""

    input_path = tmp_path / "registry_pseudo.jsonl"
    output_path = tmp_path / "judged.jsonl"
    _write_pseudo_fixture(input_path)
    # Pre-seed the output with the first row already judged.
    output_path.write_text(
        json.dumps(
            {
                "record_id": "r1",
                "label": "hawkish",
                "text": "first",
                "judge_label": "hawkish",
                "judge_confidence": 0.99,
                "judge_model_id": "gemini-2.5-pro",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    model = _StubGeminiModel(['{"label": "neutral", "confidence": 0.60}'])
    lines: list[str] = []
    written = llm_judge.run_judge(
        input_path=input_path,
        output_path=output_path,
        gemini_model=model,
        judge_model_id="gemini-2.5-pro",
        judge_model_version="v1",
        resume=True,
        progress_writer=lines.append,
    )
    # Only r2 ran in this invocation (return value excludes resumed rows)
    assert written == 1
    raw = output_path.read_text(encoding="utf-8").splitlines()
    assert len(raw) == 2
    rows = [json.loads(ln) for ln in raw]
    record_ids = {r["record_id"] for r in rows}
    assert record_ids == {"r1", "r2"}
    # Progress writer mentioned the skip on r1
    skip_lines = [ln for ln in lines if "skip" in ln]
    assert skip_lines, lines


def test_run_judge_resume_with_no_existing_output_runs_everything(tmp_path: Path) -> None:
    """resume=True with a missing output JSONL should behave like a
    fresh run (no skips)."""

    input_path = tmp_path / "registry_pseudo.jsonl"
    output_path = tmp_path / "judged.jsonl"  # does not exist
    _write_pseudo_fixture(input_path)

    model = _StubGeminiModel(
        [
            '{"label": "hawkish", "confidence": 0.95}',
            '{"label": "neutral", "confidence": 0.50}',
        ]
    )
    written = llm_judge.run_judge(
        input_path=input_path,
        output_path=output_path,
        gemini_model=model,
        judge_model_id="gemini-2.5-pro",
        judge_model_version="v1",
        resume=True,
        progress_writer=lambda _: None,
    )
    assert written == 2


def test_parse_args_resume_flag_default_is_false() -> None:
    args = llm_judge._parse_args(["--input", "/some/path"])
    assert args.resume is False


def test_parse_args_resume_flag_set_true() -> None:
    args = llm_judge._parse_args(["--input", "/some/path", "--resume"])
    assert args.resume is True


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


def test_audit_metrics_judge_only_uses_judge_as_gold_and_computes_per_class_precision() -> None:
    """Judge-only audit shape: gold is the LLM judge's label per row;
    teacher precision per class is computed against that gold."""

    rows = [
        # teacher hawkish, judge hawkish → TP for hawkish
        {"label": "hawkish", "judge_label": "hawkish"},
        {"label": "hawkish", "judge_label": "hawkish"},
        # teacher hawkish, judge neutral → FP for hawkish; FN for neutral
        {"label": "hawkish", "judge_label": "neutral"},
        # teacher neutral, judge neutral → TP for neutral
        {"label": "neutral", "judge_label": "neutral"},
        {"label": "neutral", "judge_label": "neutral"},
        # teacher dovish, judge dovish → TP for dovish
        {"label": "dovish", "judge_label": "dovish"},
    ]
    metrics = llm_judge.audit_metrics_judge_only(rows)
    assert metrics["audit_size"] == 6
    assert metrics["gold_source"] == "judge_only"
    # Teacher predicted hawkish 3 times; 2 agreed with judge → precision 2/3
    assert metrics["teacher_per_class"]["hawkish"]["precision"] == pytest.approx(2 / 3)
    assert metrics["teacher_per_class"]["neutral"]["precision"] == pytest.approx(1.0)
    assert metrics["teacher_per_class"]["dovish"]["precision"] == pytest.approx(1.0)
    # Per-class support (in gold) — what judge marked as each class
    assert metrics["teacher_per_class"]["hawkish"]["support_in_gold"] == 2
    assert metrics["teacher_per_class"]["neutral"]["support_in_gold"] == 3
    assert metrics["teacher_per_class"]["dovish"]["support_in_gold"] == 1


def test_audit_metrics_judge_only_marks_gate_passed_when_all_classes_above_threshold() -> None:
    rows = [
        {"label": "hawkish", "judge_label": "hawkish"},
        {"label": "dovish", "judge_label": "dovish"},
        {"label": "neutral", "judge_label": "neutral"},
    ]
    metrics = llm_judge.audit_metrics_judge_only(rows)
    assert metrics["audit_gate_passed"] is True
    for label in ("hawkish", "dovish", "neutral"):
        assert metrics["audit_gate_per_class"][label] is True


def test_audit_metrics_judge_only_fails_gate_when_one_class_below_threshold() -> None:
    # Teacher labels 5 rows hawkish; judge agrees on 4 of them → precision 0.80
    rows = [
        {"label": "hawkish", "judge_label": "hawkish"},
        {"label": "hawkish", "judge_label": "hawkish"},
        {"label": "hawkish", "judge_label": "hawkish"},
        {"label": "hawkish", "judge_label": "hawkish"},
        {"label": "hawkish", "judge_label": "neutral"},
        {"label": "dovish", "judge_label": "dovish"},
        {"label": "neutral", "judge_label": "neutral"},
    ]
    metrics = llm_judge.audit_metrics_judge_only(rows)
    assert metrics["audit_gate_passed"] is False
    assert metrics["audit_gate_per_class"]["hawkish"] is False
    assert metrics["audit_gate_per_class"]["dovish"] is True


def test_audit_metrics_judge_only_drops_rows_with_missing_labels() -> None:
    rows = [
        {"label": "hawkish", "judge_label": "hawkish"},
        {"label": "", "judge_label": "neutral"},  # teacher abstain → drop
        {"label": "dovish", "judge_label": ""},  # judge parse failure → drop
    ]
    metrics = llm_judge.audit_metrics_judge_only(rows)
    assert metrics["audit_size"] == 1


def test_audit_metrics_judge_only_returns_zero_on_empty_input() -> None:
    metrics = llm_judge.audit_metrics_judge_only([])
    assert metrics["audit_size"] == 0
    assert metrics["gold_source"] == "judge_only"
    assert metrics["audit_gate_per_class"] == {}


def test_audit_metrics_judge_only_tracks_judge_model_id_distribution() -> None:
    rows = [
        {"label": "hawkish", "judge_label": "hawkish", "judge_model_id": "gemini-2.5-pro"},
        {"label": "dovish", "judge_label": "dovish", "judge_model_id": "gemini-2.5-pro"},
        {"label": "neutral", "judge_label": "neutral", "judge_model_id": "gemini-2.5-flash"},
    ]
    metrics = llm_judge.audit_metrics_judge_only(rows)
    assert metrics["judge_model_id_distribution"] == {
        "gemini-2.5-pro": 2,
        "gemini-2.5-flash": 1,
    }


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


def test_parse_args_default_request_interval_is_zero() -> None:
    args = llm_judge._parse_args(["--input", "/some/path"])
    assert args.request_interval_seconds == 0.0


def test_parse_args_accepts_request_interval_flag() -> None:
    args = llm_judge._parse_args(
        ["--input", "/some/path", "--request-interval-seconds", "35"]
    )
    assert args.request_interval_seconds == 35.0


def test_run_judge_sleeps_between_calls_when_interval_set(tmp_path: Path, monkeypatch) -> None:
    """request_interval_seconds calls time.sleep between scoring rows.

    Three input rows -> two between-call sleeps (one after row 1, one
    after row 2; no sleep after the last row).
    """

    input_path = tmp_path / "registry_pseudo.jsonl"
    output_path = tmp_path / "registry_pseudo_judged.jsonl"
    rows = [
        {
            "record_id": f"r{i}",
            "source": "scraped_fed",
            "text": f"text {i}",
            "label": "hawkish",
            "label_origin": "pseudo",
            "teacher_max_score": 0.8,
            "teacher_scores": {"hawkish": 0.8, "dovish": 0.1, "neutral": 0.1},
        }
        for i in range(3)
    ]
    input_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    sleep_calls: list[float] = []

    def fake_sleep(secs: float) -> None:
        sleep_calls.append(secs)

    monkeypatch.setattr("app.data.llm_judge.time.sleep", fake_sleep)

    model = _StubGeminiModel(
        ['{"label": "hawkish", "confidence": 0.9}'] * 3
    )

    written = llm_judge.run_judge(
        input_path=input_path,
        output_path=output_path,
        gemini_model=model,
        judge_model_id="gemini-2.5-flash",
        judge_model_version="v0",
        request_interval_seconds=2.5,
    )

    assert written == 3
    assert sleep_calls == [2.5, 2.5]  # n_rows - 1 sleeps


def test_run_judge_does_not_sleep_when_interval_zero(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "registry_pseudo.jsonl"
    output_path = tmp_path / "registry_pseudo_judged.jsonl"
    rows = [
        {
            "record_id": "r0",
            "source": "scraped_fed",
            "text": "text",
            "label": "hawkish",
            "label_origin": "pseudo",
        }
    ]
    input_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    sleep_calls: list[float] = []
    monkeypatch.setattr("app.data.llm_judge.time.sleep", lambda s: sleep_calls.append(s))

    model = _StubGeminiModel(['{"label": "hawkish", "confidence": 0.9}'])
    llm_judge.run_judge(
        input_path=input_path,
        output_path=output_path,
        gemini_model=model,
        judge_model_id="gemini-2.5-flash",
        judge_model_version="v0",
        request_interval_seconds=0.0,
    )

    assert sleep_calls == []
