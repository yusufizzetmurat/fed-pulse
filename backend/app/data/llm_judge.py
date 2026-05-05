"""LLM-as-judge augmentation of the pseudo-labelling pipeline.

Reads the teacher's pseudo set produced by app.data.pseudo_labeling,
scores every row with an architecturally-distinct LLM (default: Gemini
2.5 Pro), persists judge_label / judge_confidence / judge_model_id /
judge_model_version per row, and exposes three gating policies plus a
stratified audit-set sampler and audit-metrics computer.

This module is the implementation of issue #37 and the audit half of
#31. Tests stub the Gemini model directly so they require no creds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from app.services.gemini_client import score_passage


BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path("/data") if Path("/data").exists() else BACKEND_ROOT.parent / "data"
DEFAULT_INPUT = DEFAULT_DATA_DIR / "interim" / "phase2" / "registry_pseudo.jsonl"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "interim" / "phase2" / "registry_pseudo_judged.jsonl"
DEFAULT_AUDIT_DIR = DEFAULT_DATA_DIR / "artifacts" / "pseudo_label_audits"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the teacher's pseudo set with an LLM judge and emit gated output + audit set."
    )
    parser.add_argument("--input", required=True, help="Pseudo-set JSONL produced by pseudo_labeling.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Judged pseudo-set JSONL output.")
    parser.add_argument("--judge-model", default="gemini-2.5-pro", help="Gemini model name.")
    parser.add_argument(
        "--judge-model-version",
        default="20260505_v1",
        help="Provenance version string for the judge.",
    )
    parser.add_argument("--audit-dir", default=str(DEFAULT_AUDIT_DIR), help="Audit artefact directory.")
    parser.add_argument("--max-rows", type=int, default=0, help="Score at most N rows; 0 = no limit.")
    return parser.parse_args(argv)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def run_judge(
    *,
    input_path: Path,
    output_path: Path,
    gemini_model,
    judge_model_id: str,
    judge_model_version: str,
    max_rows: int = 0,
) -> int:
    """Score every row in input_path, write judged rows to output_path.

    Each output row preserves all input fields and adds judge_label,
    judge_confidence, judge_model_id, judge_model_version. Returns the
    number of rows written.
    """

    rows = _read_jsonl(input_path)
    if max_rows > 0:
        rows = rows[:max_rows]

    judged: list[dict[str, Any]] = []
    for row in rows:
        prediction = score_passage(row.get("text", ""), model=gemini_model)
        out = dict(row)
        out["judge_label"] = prediction["label"]
        out["judge_confidence"] = float(prediction["confidence"])
        out["judge_model_id"] = judge_model_id
        out["judge_model_version"] = judge_model_version
        judged.append(out)

    _write_jsonl(output_path, judged)
    return len(judged)


GATING_POLICIES = ("confidence_only", "confidence_and_judge", "judge_only")
ALLOWED_LABELS = ("hawkish", "dovish", "neutral")


def apply_gating_policy(
    rows: list[dict[str, Any]], *, policy: str, tau: float
) -> list[dict[str, Any]]:
    """Filter judged pseudo rows under one of three gating policies.

    - confidence_only: teacher_max_score >= tau.
    - confidence_and_judge: teacher_max_score >= tau AND judge_label == teacher label.
    - judge_only: judge_label is in {hawkish, dovish, neutral}.
    """

    if policy not in GATING_POLICIES:
        raise ValueError(f"Unknown gating policy: {policy!r}. Allowed: {GATING_POLICIES}")

    kept: list[dict[str, Any]] = []
    for row in rows:
        if policy == "confidence_only":
            if float(row.get("teacher_max_score", 0.0)) >= tau:
                kept.append(row)
            continue
        if policy == "confidence_and_judge":
            if float(row.get("teacher_max_score", 0.0)) < tau:
                continue
            if str(row.get("judge_label", "")) != str(row.get("label", "")):
                continue
            kept.append(row)
            continue
        # judge_only
        if str(row.get("judge_label", "")) in ALLOWED_LABELS:
            kept.append(row)
    return kept


import random
from collections import defaultdict


def sample_audit_set(
    rows: list[dict[str, Any]], *, n: int, seed: int = 11
) -> list[dict[str, Any]]:
    """Stratified sample of size n from rows by teacher label.

    The sample is proportional to each class's count, with at least one
    row per non-empty class when n is large enough. The seed makes the
    sample reproducible.
    """

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row.get("label", ""))].append(row)

    rng = random.Random(seed)
    total = sum(len(v) for v in by_label.values())
    if total == 0:
        return []

    quotas: dict[str, int] = {}
    for label, items in by_label.items():
        share = round(n * len(items) / total)
        quotas[label] = max(1, share) if items else 0

    overflow = sum(quotas.values()) - n
    if overflow > 0:
        for label in sorted(quotas, key=lambda k: -quotas[k]):
            while overflow > 0 and quotas[label] > 1:
                quotas[label] -= 1
                overflow -= 1
            if overflow <= 0:
                break
    elif overflow < 0:
        for label in sorted(quotas, key=lambda k: -len(by_label[k])):
            while overflow < 0 and quotas[label] < len(by_label[label]):
                quotas[label] += 1
                overflow += 1
            if overflow >= 0:
                break

    sample: list[dict[str, Any]] = []
    for label, items in by_label.items():
        k = min(quotas.get(label, 0), len(items))
        sample.extend(rng.sample(items, k))
    return sample


def audit_metrics(audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute teacher / judge precision against the human gold label.

    Rows where human_label is empty are skipped (the pass is in progress).
    Returns aggregate accuracy, Cohen's κ between (teacher, human),
    (judge, human), (teacher, judge), and per-class precision tables.
    """

    labelled = [r for r in audit_rows if str(r.get("human_label", "")).strip()]
    if not labelled:
        return {
            "audit_size": 0,
            "teacher_accuracy": 0.0,
            "judge_accuracy": 0.0,
            "cohen_kappa": {},
            "teacher_per_class": {},
            "judge_per_class": {},
        }

    humans = [str(r["human_label"]).strip().lower() for r in labelled]
    teachers = [str(r.get("label", "")).strip().lower() for r in labelled]
    judges = [str(r.get("judge_label", "")).strip().lower() for r in labelled]

    teacher_acc = sum(t == h for t, h in zip(teachers, humans)) / len(labelled)
    judge_acc = sum(j == h for j, h in zip(judges, humans)) / len(labelled)

    try:
        from sklearn.metrics import cohen_kappa_score  # type: ignore

        kappa = {
            "teacher_human": float(cohen_kappa_score(teachers, humans)),
            "judge_human": float(cohen_kappa_score(judges, humans)),
            "teacher_judge": float(cohen_kappa_score(teachers, judges)),
        }
    except Exception:  # pragma: no cover - import guard
        kappa = {}

    def _per_class(predictions: list[str], gold: list[str]) -> dict[str, float]:
        per: dict[str, float] = {}
        for label in ALLOWED_LABELS:
            tp = sum(1 for p, g in zip(predictions, gold) if p == label and g == label)
            fp = sum(1 for p, g in zip(predictions, gold) if p == label and g != label)
            denom = tp + fp
            per[label] = tp / denom if denom else 0.0
        return per

    return {
        "audit_size": len(labelled),
        "teacher_accuracy": teacher_acc,
        "judge_accuracy": judge_acc,
        "cohen_kappa": kappa,
        "teacher_per_class": _per_class(teachers, humans),
        "judge_per_class": _per_class(judges, humans),
    }


def main() -> int:
    args = _parse_args()
    raise NotImplementedError("Wire main() in Task 7 once gating and audit are in place.")


if __name__ == "__main__":
    raise SystemExit(main())
