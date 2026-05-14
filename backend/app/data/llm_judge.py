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
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from app.services.gemini_client import score_passage


from app.config import DATA_DIR as DEFAULT_DATA_DIR
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
    parser.add_argument(
        "--request-interval-seconds",
        type=float,
        default=0.0,
        help="Sleep this many seconds between Gemini calls to respect rate limits. 0 = no sleep.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip rows whose record_id already appears in the output JSONL. "
            "Combined with append-mode incremental writes, re-running this "
            "command after a crash picks up where the previous run died."
        ),
    )
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


# Gemini error signatures that warrant an in-row retry. The SDK raises
# concrete ServerError / ClientError subclasses whose names include the
# status — easiest classifier is to grep the string for known transient
# tokens. Conservative on what counts as transient; everything else
# fails fast and gets recorded as an error.
_TRANSIENT_ERROR_TOKENS = (
    "429",
    "503",
    "UNAVAILABLE",
    "RESOURCE_EXHAUSTED",
    "DEADLINE_EXCEEDED",
    "INTERNAL",
    "ConnectionError",
    "Timeout",
    "RemoteDisconnect",
)
_RETRY_ATTEMPTS_DEFAULT = 4
_RETRY_BASE_DELAY_DEFAULT = 1.0
_RETRY_MAX_DELAY_DEFAULT = 64.0


def _is_transient_error(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    haystack = f"{type(exc).__name__}: {exc!s}"
    return any(token in haystack for token in _TRANSIENT_ERROR_TOKENS)


def _score_with_retry(
    text: str,
    model: Any,
    *,
    max_attempts: int = _RETRY_ATTEMPTS_DEFAULT,
    base_delay: float = _RETRY_BASE_DELAY_DEFAULT,
    max_delay: float = _RETRY_MAX_DELAY_DEFAULT,
    on_retry=None,
    sleep_fn=time.sleep,
) -> tuple[dict[str, Any] | None, BaseException | None]:
    """Call score_passage with exponential backoff on transient Gemini errors.

    Returns (prediction, None) on success. Returns (None, exception) when
    the call fails after ``max_attempts`` or hits a non-transient error
    (which is not retried — fails fast). ``on_retry`` receives
    ``(attempt_index, exc, delay)`` so the caller can log each backoff.
    """

    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            prediction = score_passage(text, model=model)
            return prediction, None
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_transient_error(exc) or attempt == max_attempts - 1:
                return None, exc
            delay = min(base_delay * (4 ** attempt), max_delay)
            if on_retry is not None:
                on_retry(attempt + 1, exc, delay)
            sleep_fn(delay)
    return None, last_exc


def run_judge(
    *,
    input_path: Path,
    output_path: Path,
    gemini_model,
    judge_model_id: str,
    judge_model_version: str,
    max_rows: int = 0,
    request_interval_seconds: float = 0.0,
    resume: bool = False,
    progress_writer=None,
) -> int:
    """Score every row in ``input_path`` and write judged rows incrementally
    to ``output_path``. Returns the number of rows written *by this
    invocation* (resumed rows are not counted).

    Per-row behaviour:

    - Prints one line per row to stdout so the operator can see progress
      ("[judge] 17/91  hawkish  ...").
    - Catches any exception from the Gemini SDK (transient 429/503 are
      common at scale) and persists the row with ``judge_label=""``
      rather than crashing the run. The audit step counts blank rows as
      parse / API failures.
    - Writes one row at a time in append mode and flushes after each
      write, so an interrupted run keeps every completed row on disk.

    ``resume`` rereads ``output_path`` and skips rows whose
    ``record_id`` is present AND has a non-empty ``judge_label``.
    Rows previously recorded with a ``judge_error`` (transient API
    failure) get retried on the next invocation — the only rows
    treated as durably done are the ones with a real judge label.

    ``progress_writer`` is an optional callable that receives each
    progress line (defaults to the built-in ``print``). Tests inject a
    list-appender to avoid noisy stdout.

    ``request_interval_seconds`` spaces successive Gemini calls; use
    ``35.0`` for the free-tier flash 2-req/min cap, ``0.0`` for paid.
    """

    if progress_writer is None:
        def progress_writer(line: str) -> None:
            print(line, flush=True)

    rows = _read_jsonl(input_path)
    if max_rows > 0:
        rows = rows[:max_rows]
    total = len(rows)

    successful_record_ids: set[str] = set()
    error_record_ids: set[str] = set()
    if resume and output_path.exists():
        for existing in _read_jsonl(output_path):
            rid = str(existing.get("record_id", "")).strip()
            if not rid:
                continue
            label_present = str(existing.get("judge_label", "")).strip()
            if label_present:
                successful_record_ids.add(rid)
            else:
                error_record_ids.add(rid)

    # When resuming we keep the existing successful rows by appending the
    # new content. If we are NOT resuming (or the output is empty) the
    # old content gets overwritten. Errored rows always get re-tried;
    # they will be re-written into the file as new rows, so on resume we
    # also need to drop the previous errored rows from the file. The
    # simplest implementation: when resuming, rewrite the file from
    # scratch with only the successful rows, then append.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and error_record_ids:
        kept = [
            existing
            for existing in _read_jsonl(output_path)
            if str(existing.get("judge_label", "")).strip()
        ]
        with output_path.open("w", encoding="utf-8") as handle:
            for kept_row in kept:
                handle.write(json.dumps(kept_row) + "\n")
        open_mode = "a"
    else:
        open_mode = "a" if (resume and successful_record_ids) else "w"

    completed_record_ids = successful_record_ids  # alias to keep the loop readable

    written = 0
    with output_path.open(open_mode, encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            record_id = str(row.get("record_id", "")).strip()
            short_id = record_id[:16] if record_id else "(no-id)"

            if record_id and record_id in completed_record_ids:
                progress_writer(f"[judge] {index + 1}/{total}  skip      {short_id}  (resumed)")
                continue

            def _log_retry(attempt: int, exc: BaseException, delay: float) -> None:
                progress_writer(
                    f"[judge] {index + 1}/{total}  retry {attempt}/{_RETRY_ATTEMPTS_DEFAULT - 1}  "
                    f"{short_id}  ({type(exc).__name__} — backoff {delay:.1f}s)"
                )

            prediction, exc = _score_with_retry(
                row.get("text", ""),
                gemini_model,
                on_retry=_log_retry,
            )
            if prediction is not None:
                label = str(prediction.get("label", "") or "")
                confidence = float(prediction.get("confidence", 0.0) or 0.0)
                error_kind = ""
            else:
                label = ""
                confidence = 0.0
                error_kind = type(exc).__name__ if exc is not None else "Unknown"
                msg = str(exc)[:160] if exc is not None else ""
                progress_writer(
                    f"[judge] {index + 1}/{total}  ERROR     {short_id}  "
                    f"({error_kind}): {msg}"
                )

            out = dict(row)
            out["judge_label"] = label
            out["judge_confidence"] = confidence
            out["judge_model_id"] = judge_model_id
            out["judge_model_version"] = judge_model_version
            if error_kind:
                out["judge_error"] = error_kind

            handle.write(json.dumps(out) + "\n")
            handle.flush()
            written += 1

            if not error_kind:
                progress_writer(
                    f"[judge] {index + 1}/{total}  {label or 'blank':<8}  {short_id}"
                )

            if request_interval_seconds > 0 and index < total - 1:
                time.sleep(request_interval_seconds)

    return written


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


def audit_metrics_judge_only(judged_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute teacher precision against the LLM judge as gold annotator.

    Per-class teacher precision must clear 0.90 for the pseudo set to
    enter `source_registry.jsonl`. Rows where either label is empty are
    skipped (judge parse failure or teacher abstain). Classes the judge
    never assigns are excluded from the gate (no support, no signal).
    """

    usable = [
        r
        for r in judged_rows
        if str(r.get("label", "")).strip() and str(r.get("judge_label", "")).strip()
    ]
    if not usable:
        return {
            "audit_size": 0,
            "gold_source": "judge_only",
            "judge_model_id_distribution": {},
            "teacher_judge_accuracy": 0.0,
            "cohen_kappa_teacher_judge": None,
            "teacher_per_class": {},
            "teacher_label_distribution": {},
            "judge_label_distribution": {},
            "audit_gate_per_class": {},
            "audit_gate_passed": False,
        }

    teachers = [str(r["label"]).strip().lower() for r in usable]
    judges = [str(r["judge_label"]).strip().lower() for r in usable]

    teacher_judge_acc = sum(t == j for t, j in zip(teachers, judges)) / len(usable)

    kappa: float | None
    try:
        from sklearn.metrics import cohen_kappa_score  # type: ignore

        kappa = float(cohen_kappa_score(teachers, judges))
    except Exception:  # pragma: no cover
        kappa = None

    def _per_class(predictions: list[str], gold: list[str]) -> dict[str, dict[str, float | int]]:
        per: dict[str, dict[str, float | int]] = {}
        for label in ALLOWED_LABELS:
            tp = sum(1 for p, g in zip(predictions, gold) if p == label and g == label)
            fp = sum(1 for p, g in zip(predictions, gold) if p == label and g != label)
            fn = sum(1 for p, g in zip(predictions, gold) if p != label and g == label)
            denom = tp + fp
            precision = tp / denom if denom else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            per[label] = {
                "precision": precision,
                "recall": recall,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support_in_gold": tp + fn,
            }
        return per

    judge_label_distribution = {
        label: sum(1 for j in judges if j == label) for label in ALLOWED_LABELS
    }
    teacher_label_distribution = {
        label: sum(1 for t in teachers if t == label) for label in ALLOWED_LABELS
    }

    teacher_per_class = _per_class(teachers, judges)
    gate_per_class = {
        label: teacher_per_class[label]["precision"] >= 0.90
        for label in ALLOWED_LABELS
        if teacher_per_class[label]["support_in_gold"] > 0
    }
    gate_overall = bool(gate_per_class) and all(gate_per_class.values())

    return {
        "audit_size": len(usable),
        "gold_source": "judge_only",
        "judge_model_id_distribution": _judge_model_distribution(usable),
        "teacher_judge_accuracy": teacher_judge_acc,
        "cohen_kappa_teacher_judge": kappa,
        "teacher_per_class": teacher_per_class,
        "teacher_label_distribution": teacher_label_distribution,
        "judge_label_distribution": judge_label_distribution,
        "audit_gate_per_class": gate_per_class,
        "audit_gate_passed": gate_overall,
    }


def _judge_model_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Track which judge model produced each row so we can audit
    consistency when the run is split across multiple Gemini variants
    (e.g. 2.5-pro vs 2.5-flash fallback)."""

    counts: dict[str, int] = {}
    for row in rows:
        model = str(row.get("judge_model_id") or "unknown")
        counts[model] = counts.get(model, 0) + 1
    return counts


def write_audit_csv(sample: list[dict[str, Any]], output_path: Path) -> None:
    """Write the audit set as a CSV with a `human_label` column the
    offline labelling pass fills in."""

    fieldnames = [
        "record_id",
        "event_date",
        "source_type",
        "title",
        "text",
        "label",  # teacher label
        "judge_label",
        "human_label",  # empty
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sample:
            writer.writerow(
                {
                    "record_id": row.get("record_id", ""),
                    "event_date": row.get("event_date", ""),
                    "source_type": row.get("source_type", ""),
                    "title": row.get("title", ""),
                    "text": row.get("text", ""),
                    "label": row.get("label", ""),
                    "judge_label": row.get("judge_label", ""),
                    "human_label": "",
                }
            )


def summarise_gating_policies(
    rows: list[dict[str, Any]], *, tau: float
) -> dict[str, Any]:
    """Summarize kept rows and label distribution for each gating policy."""
    summary: dict[str, Any] = {}
    for policy in GATING_POLICIES:
        kept = apply_gating_policy(rows, policy=policy, tau=tau)
        labels = Counter(r.get("label", "") for r in kept)
        summary[policy] = {
            "kept": len(kept),
            "label_distribution": dict(labels),
        }
    return summary


def main() -> int:
    args = _parse_args()
    from app.services.gemini_client import load_model

    model = load_model(args.judge_model)

    judged_path = Path(args.output)
    written = run_judge(
        input_path=Path(args.input),
        output_path=judged_path,
        gemini_model=model,
        judge_model_id=args.judge_model,
        judge_model_version=args.judge_model_version,
        max_rows=args.max_rows,
        request_interval_seconds=args.request_interval_seconds,
        resume=args.resume,
    )
    print(f"Judged rows written: {written}")

    judged_rows = _read_jsonl(judged_path)

    audit_dir = Path(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)

    sweep: dict[str, Any] = {}
    for tau in (0.75, 0.85, 0.95):
        sweep[f"{tau}"] = summarise_gating_policies(judged_rows, tau=tau)
    (audit_dir / "policy_sweep.json").write_text(
        json.dumps(sweep, indent=2), encoding="utf-8"
    )
    print(f"Policy sweep written to {audit_dir / 'policy_sweep.json'}")

    audit_sample = sample_audit_set(judged_rows, n=100, seed=11)
    _write_jsonl(audit_dir / "audit_set.jsonl", audit_sample)
    write_audit_csv(audit_sample, audit_dir / "audit_set.csv")
    print(f"Audit set written ({len(audit_sample)} rows) to {audit_dir}")

    filled_path = audit_dir / "audit_set_filled.jsonl"
    if filled_path.exists():
        filled_rows = _read_jsonl(filled_path)
        metrics = audit_metrics(filled_rows)
        (audit_dir / "audit_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        print(f"Audit metrics written to {audit_dir / 'audit_metrics.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
