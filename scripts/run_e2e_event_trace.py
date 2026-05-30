"""End-to-end event trace smoke (#505 A.1.c).

Loads one walk-forward fold with every rich-feature flag ON, finds the
target event's prior-window sequence, calls ``as_rich_list`` on the
event-day bar, and walks every documented core slice from
``backend/app/models/config.py``. Reports slice-by-slice population
and asserts:

- vector length matches the loader output
- every non-missing-flag slice carries at least one finite non-zero
  value (the missing-flag scalars can legitimately be 0 or 1)

Usage:

    python -m scripts.run_e2e_event_trace \\
        --training-package-id <tp_id> \\
        --event-date 2024-09-18 \\
        --output backend/artifacts/audits/pre_sweep_<tp_id>/e2e_trace.json

Exit 0 on success, 1 if any slice fails the non-zero check.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_FOLD = "wf_fold_3"
DEFAULT_EVENT = "2024-09-18"

# Slice constants we walk. These are the CORE rich slices inside
# [0, RICH_FEATURE_SIZE]. Missing-flag slots are flagged so the
# non-zero check skips them (a 0 there means "value is present").
CORE_SLICES: tuple[tuple[str, str, bool], ...] = (
    # (logical_name, RICH_*_SLICE constant name, is_missing_flag)
    ("market", "RICH_MARKET_SLICE", False),
    ("credibility", "RICH_CREDIBILITY_SLICE", False),
    ("linguistic", "RICH_LINGUISTIC_SLICE", False),
    ("mp_surprise", "RICH_MP_SURPRISE_SLICE", False),
    ("multi_axis", "RICH_MULTI_AXIS_SLICE", False),
    ("realized_vol", "RICH_REALIZED_VOL_SLICE", False),
    ("cross_asset", "RICH_CROSS_ASSET_SLICE", False),
    ("llm_feature", "RICH_LLM_FEATURE_SLICE", False),
    ("llm_feature_missing", "RICH_LLM_FEATURE_MISSING_SLICE", True),
    ("retrieval_analog", "RICH_RETRIEVAL_ANALOG_SLICE", False),
    ("retrieval_analog_missing", "RICH_RETRIEVAL_ANALOG_MISSING_SLICE", True),
)


@dataclass(frozen=True)
class SliceReport:
    name: str
    start: int
    stop: int
    dim: int
    is_missing_flag: bool
    finite: int
    non_zero: int
    sample_values: list[float]
    ok: bool
    note: str


def _summarise_slice(
    name: str,
    sl: slice,
    is_missing_flag: bool,
    vector: list[float],
) -> SliceReport:
    block = vector[sl]
    finite = sum(1 for v in block if math.isfinite(v))
    non_zero = sum(1 for v in block if math.isfinite(v) and v != 0.0)
    sample = block[: min(8, len(block))]
    if is_missing_flag:
        ok = finite == len(block)
        note = (
            "missing flag, zero is a valid value (means feature present)"
            if ok
            else "missing flag has non-finite value"
        )
    else:
        ok = non_zero > 0
        note = (
            "ok"
            if ok
            else "every position is zero or non-finite (possible silent zeros)"
        )
    return SliceReport(
        name=name,
        start=sl.start,
        stop=sl.stop,
        dim=len(block),
        is_missing_flag=is_missing_flag,
        finite=finite,
        non_zero=non_zero,
        sample_values=sample,
        ok=ok,
        note=note,
    )


def _trace_event(
    training_package_id: str,
    event_date: str,
    fold_id: str,
) -> dict[str, Any]:
    """Run the loader, find the target event, build the slice report."""

    from app.models import config as model_config
    from app.training.loaders import load_walk_forward_split

    split = load_walk_forward_split(
        training_package_id=training_package_id,
        fold_id=fold_id,
        rich_features=True,
        use_credibility=True,
        use_linguistic=True,
        use_mp_surprise=True,
        use_multi_axis=True,
        use_llm_features=True,
        use_retrieval_analogs=True,
        use_regime_conditioning=False,
        use_sep=False,
        use_press_conf=False,
        use_statement_delta=False,
        use_vote_features=False,
    )

    # Search every partition for the event_date.
    target_seq: list | None = None
    found_partition: str = ""
    for partition_name, sequences, event_dates in (
        ("train", split.train, split.train_event_dates),
        ("val", split.val, split.val_event_dates),
        ("test", split.test, split.test_event_dates),
    ):
        for idx, evdate in enumerate(event_dates):
            if str(evdate)[:10] == event_date:
                target_seq = sequences[idx]
                found_partition = partition_name
                break
        if target_seq is not None:
            break

    if target_seq is None:
        raise SystemExit(
            f"event_date {event_date} not found in any partition of "
            f"fold {fold_id}. Pick a different event or a different fold."
        )
    if not target_seq:
        raise SystemExit(
            f"event_date {event_date} resolved to an empty sequence; "
            "the loader dropped this event's prior window."
        )

    event_bar = target_seq[-1]  # event-day bar (last of the prior window)
    rich = event_bar.as_rich_list()

    slice_reports: list[SliceReport] = []
    for name, const_name, is_missing_flag in CORE_SLICES:
        sl = getattr(model_config, const_name)
        slice_reports.append(
            _summarise_slice(name, sl, is_missing_flag, rich)
        )

    rich_feature_size = int(model_config.RICH_FEATURE_SIZE)

    failures = [r for r in slice_reports if not r.ok]
    return {
        "training_package_id": training_package_id,
        "event_date": event_date,
        "fold_id": fold_id,
        "partition_found_in": found_partition,
        "rich_feature_size_const": rich_feature_size,
        "rich_list_length": len(rich),
        "length_matches_const": len(rich) == rich_feature_size,
        "slices": [r.__dict__ for r in slice_reports],
        "failures": [r.name for r in failures],
        "pass": len(failures) == 0 and len(rich) == rich_feature_size,
    }


def _render_summary_md(report: dict[str, Any]) -> str:
    verdict = "PASS" if report["pass"] else "FAIL"
    lines: list[str] = []
    lines.append(f"# E2E event trace ({verdict})")
    lines.append("")
    lines.append(f"- tp: `{report['training_package_id']}`")
    lines.append(f"- event_date: `{report['event_date']}`")
    lines.append(f"- fold: `{report['fold_id']}`")
    lines.append(f"- partition: `{report['partition_found_in']}`")
    lines.append(
        f"- rich_list length: {report['rich_list_length']} "
        f"(const RICH_FEATURE_SIZE = {report['rich_feature_size_const']})"
    )
    lines.append("")
    lines.append("| slice | range | dim | finite | non_zero | ok |")
    lines.append("|---|---|---|---|---|---|")
    for sl in report["slices"]:
        mark = "ok" if sl["ok"] else "FAIL"
        lines.append(
            f"| {sl['name']} | "
            f"{sl['start']}:{sl['stop']} | "
            f"{sl['dim']} | {sl['finite']} | {sl['non_zero']} | {mark} |"
        )
    lines.append("")
    if report["failures"]:
        lines.append("## Failed slices")
        for sl in report["slices"]:
            if sl["ok"]:
                continue
            lines.append(
                f"- `{sl['name']}` ({sl['start']}:{sl['stop']}): "
                f"{sl['note']}; sample: {sl['sample_values']}"
            )
    else:
        lines.append("All slices populated with at least one finite non-zero value.")
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2E event trace (#505 A.1.c)")
    parser.add_argument("--training-package-id", required=True)
    parser.add_argument("--event-date", default=DEFAULT_EVENT)
    parser.add_argument("--fold-id", default=DEFAULT_FOLD)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = _trace_event(
        training_package_id=args.training_package_id,
        event_date=args.event_date,
        fold_id=args.fold_id,
    )
    summary_md = _render_summary_md(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        md_path = args.output.with_suffix(".md")
        md_path.write_text(summary_md, encoding="utf-8")
        print(f"[e2e-trace] wrote {args.output} + {md_path}")
    else:
        sys.stdout.write(summary_md)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
