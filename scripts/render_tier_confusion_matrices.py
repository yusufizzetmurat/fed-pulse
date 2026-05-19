"""Render per-tier confusion-matrix heatmaps for the appendix + UI.

Reads every ``forecaster_sweep_results.json`` under
``data/artifacts/regime_baseline_tiers/<package>/<tier>/`` and writes:

- ``aggregated_confusion_matrix.png`` per tier -- the elementwise sum
  across every (seed, fold) trial, framed as the population-level
  headline visual the report appendix can lift directly.
- ``per_trial/<seed>_<fold>.png`` per (seed, fold) -- one heatmap per
  trial for the appendix or for the UI's drill-in.

The class labels default to ``calm`` / ``normal`` / ``high`` to match
the operational regime semantics; override via ``--class-labels``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.evaluation.confusion_matrix_render import (
    aggregate_confusion_matrices,
    render_confusion_matrix_png,
)


_DEFAULT_REPORT_ROOT = Path("data/artifacts/regime_baseline_tiers")


def _resolve_root(report_root: Path, training_package_id: str) -> Path:
    candidates = [
        report_root / training_package_id,
        Path("backend") / report_root / training_package_id,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"could not locate tier artefacts for {training_package_id}; tried: "
        + ", ".join(str(c) for c in candidates)
    )


def _iter_trials(report_path: Path):
    payload = json.loads(report_path.read_text())
    if isinstance(payload, list):
        trials = payload
    elif isinstance(payload, dict):
        trials = payload.get("trials") or payload.get("results") or []
    else:
        trials = []
    for t in trials:
        summary = t.get("summary") or t
        metrics = summary.get("test_metrics") or {}
        breakdown = metrics.get("classification_breakdown")
        if not isinstance(breakdown, dict):
            continue
        cm = breakdown.get("confusion_matrix")
        if not isinstance(cm, list) or not cm:
            continue
        yield {
            "trial": t,
            "summary": summary,
            "confusion_matrix": cm,
            "seed": summary.get("model_config", {}).get("seed")
            or t.get("seed"),
            "fold": summary.get("fold_id") or t.get("fold_id"),
        }


def _render_tier(tier_dir: Path, labels: list[str]) -> dict[str, int]:
    report_path = tier_dir / "forecaster_sweep_results.json"
    if not report_path.exists():
        return {"trials": 0, "rendered": 0}

    per_trial_dir = tier_dir / "per_trial"
    per_trial_dir.mkdir(parents=True, exist_ok=True)
    trials = list(_iter_trials(report_path))
    if not trials:
        return {"trials": 0, "rendered": 0}

    for t in trials:
        out = per_trial_dir / f"seed_{t['seed']}_{t['fold']}.png"
        render_confusion_matrix_png(
            t["confusion_matrix"],
            out,
            class_labels=labels,
            title=f"{tier_dir.name} · seed {t['seed']} · {t['fold']}",
        )

    aggregated = aggregate_confusion_matrices([t["confusion_matrix"] for t in trials])
    render_confusion_matrix_png(
        aggregated,
        tier_dir / "aggregated_confusion_matrix.png",
        class_labels=labels,
        title=f"{tier_dir.name} · aggregated across {len(trials)} trials",
    )
    return {"trials": len(trials), "rendered": len(trials) + 1}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-package-id", required=True)
    p.add_argument(
        "--report-root", type=Path, default=_DEFAULT_REPORT_ROOT
    )
    p.add_argument(
        "--class-labels",
        nargs="+",
        default=["calm", "normal", "high"],
        help="Class labels in order of class index (defaults to the "
        "operational vol-regime labels).",
    )
    args = p.parse_args(argv)

    root = _resolve_root(args.report_root, args.training_package_id)
    print(f"[render_cm] scanning {root}", flush=True)
    overall: dict[str, dict[str, int]] = {}
    for tier_dir in sorted(root.iterdir()):
        if not tier_dir.is_dir():
            continue
        # Tier-3 capacity sweep nests one encoder per dir.
        if (tier_dir / "forecaster_sweep_results.json").exists():
            stats = _render_tier(tier_dir, args.class_labels)
            print(f"[render_cm] {tier_dir.name}: {stats}", flush=True)
            overall[tier_dir.name] = stats
            continue
        for sub in sorted(tier_dir.iterdir()):
            if sub.is_dir() and (sub / "forecaster_sweep_results.json").exists():
                stats = _render_tier(sub, args.class_labels)
                print(
                    f"[render_cm] {tier_dir.name}/{sub.name}: {stats}",
                    flush=True,
                )
                overall[f"{tier_dir.name}/{sub.name}"] = stats
    if not overall:
        print("[render_cm] no tier artefacts found", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
