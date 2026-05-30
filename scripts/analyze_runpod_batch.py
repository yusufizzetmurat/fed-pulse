"""Cross-arm paired analysis on the runpod batch artefacts.

For every opt-in arm vs canonical (matched by seed × fold), reports the
paired Wilcoxon signed-rank statistic + Holm-Bonferroni-corrected
p-value on the dual head_mode regime_f1_macro deltas. Also emits the
arm × head_mode mean F1 table consumed by wiki §6.

The cross-arm comparison sits outside :mod:`app.eval.paired_comparisons`
because that helper compares head_modes within one sweep file; this
script joins across sweep files.

CLI::

    python scripts/analyze_runpod_batch.py \\
        --artefact-dir backend/artifacts/experiments/runpod_20260530 \\
        --output backend/artifacts/experiments/runpod_20260530/cross_arm_paired.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scipy.stats import wilcoxon

_DEFAULT_DIR = Path("backend/artifacts/experiments/runpod_20260530")

ARMS: dict[str, str] = {
    "canonical": "runpod_canonical_20260530T120517Z.json",
    "statement_delta": "runpod_statement_delta_20260530T134347Z.json",
    "vote_tally": "runpod_vote_tally_20260530T135114Z.json",
    "press_conf": "runpod_press_conf_20260530T141754Z.json",
    "surprise": "runpod_surprise_20260530T143043Z.json",
    "retrieval": "runpod_retrieval_20260530T124410Z.json",
    "regime": "runpod_regime_20260530T124847Z.json",
}


def _extract_cells(
    path: Path, head_mode: str, metric: str = "regime_f1_macro"
) -> dict[tuple[int, str], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    trials = payload.get("trials") or {}
    cells: dict[tuple[int, str], float] = {}
    for trial in trials.get(head_mode, []) or []:
        seed = int(trial.get("seed", -1))
        for fold in trial.get("folds", []) or []:
            fid = str(fold.get("fold_id", ""))
            v = (fold.get("metrics") or {}).get(metric)
            if v is not None and seed >= 0:
                cells[(seed, fid)] = float(v)
    return cells


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(
        enumerate(p_values), key=lambda kv: (math.isnan(kv[1]), kv[1])
    )
    out = [0.0] * n
    running = 0.0
    for rank, (idx, p) in enumerate(indexed):
        adj = 1.0 if math.isnan(p) else min(1.0, p * (n - rank))
        adj = max(adj, running)
        running = adj
        out[idx] = adj
    return out


@dataclass(frozen=True)
class PairedResult:
    arm: str
    head_mode: str
    n_pairs: int
    mean_delta: float
    std_delta: float
    wilcoxon_stat: float
    p_value: float
    p_value_holm: float


def cross_arm_paired(
    artefact_dir: Path,
    head_mode: str = "dual",
    reference_arm: str = "canonical",
    arms: dict[str, str] | None = None,
) -> list[PairedResult]:
    arms = arms or ARMS
    ref_cells = _extract_cells(artefact_dir / arms[reference_arm], head_mode)
    raw: list[PairedResult] = []
    for name, fname in arms.items():
        if name == reference_arm:
            continue
        arm_cells = _extract_cells(artefact_dir / fname, head_mode)
        shared = sorted(
            set(ref_cells) & set(arm_cells), key=lambda k: (k[1], k[0])
        )
        deltas = [arm_cells[k] - ref_cells[k] for k in shared]
        n = len(deltas)
        if n == 0:
            raw.append(PairedResult(
                arm=name, head_mode=head_mode, n_pairs=0,
                mean_delta=float("nan"), std_delta=float("nan"),
                wilcoxon_stat=float("nan"), p_value=float("nan"),
                p_value_holm=float("nan"),
            ))
            continue
        mu = statistics.fmean(deltas)
        sd = statistics.pstdev(deltas) if n > 1 else 0.0
        nonzero = [d for d in deltas if d != 0.0]
        if len(nonzero) >= 2:
            stat, pval = wilcoxon(nonzero, alternative="two-sided")
        else:
            stat, pval = float("nan"), float("nan")
        raw.append(PairedResult(
            arm=name, head_mode=head_mode, n_pairs=n,
            mean_delta=mu, std_delta=sd,
            wilcoxon_stat=float(stat), p_value=float(pval),
            p_value_holm=float("nan"),
        ))
    holm = _holm_bonferroni([r.p_value for r in raw])
    return [
        PairedResult(
            arm=r.arm, head_mode=r.head_mode, n_pairs=r.n_pairs,
            mean_delta=r.mean_delta, std_delta=r.std_delta,
            wilcoxon_stat=r.wilcoxon_stat, p_value=r.p_value,
            p_value_holm=h,
        )
        for r, h in zip(raw, holm, strict=False)
    ]


def headline_means(
    artefact_dir: Path,
    arms: dict[str, str] | None = None,
    metric: str = "regime_f1_macro",
) -> list[dict[str, Any]]:
    arms = arms or ARMS
    rows: list[dict[str, Any]] = []
    for name, fname in arms.items():
        d = json.loads((artefact_dir / fname).read_text(encoding="utf-8"))
        row: dict[str, Any] = {"arm": name}
        for mode in ("classification", "dual", "regression"):
            vals = []
            for trial in (d.get("trials") or {}).get(mode, []) or []:
                for fold in trial.get("folds", []) or []:
                    v = (fold.get("metrics") or {}).get(metric)
                    if v is not None:
                        vals.append(float(v))
            row[mode] = {
                "n": len(vals),
                "mean": statistics.fmean(vals) if vals else None,
                "std": statistics.pstdev(vals) if len(vals) > 1 else (0.0 if vals else None),
            }
        rows.append(row)
    return rows


def render_markdown(
    headline: list[dict[str, Any]],
    paired_dual: list[PairedResult],
    paired_class: list[PairedResult],
    *,
    reference_arm: str = "canonical",
) -> str:
    lines = [
        "### Headline table: per-arm × head_mode mean F1 (n=25 unless noted)",
        "",
        "| Arm | class | dual | regression |",
        "|---|---:|---:|---:|",
    ]
    for row in headline:
        cls = row["classification"]
        dl = row["dual"]
        rg = row["regression"]
        def _f(b: dict[str, Any]) -> str:
            return "—" if b["mean"] is None else f"{b['mean']:.4f} ± {b['std']:.4f}"
        lines.append(f"| {row['arm']} | {_f(cls)} | {_f(dl)} | {_f(rg)} |")
    lines.extend([
        "",
        f"### Dual-head paired Wilcoxon vs {reference_arm} (Holm-corrected)",
        "",
        "| Arm | n | mean Δ | W | p | p (Holm) | verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for r in paired_dual:
        verdict = (
            "—" if math.isnan(r.p_value_holm)
            else "significant ↓" if r.p_value_holm < 0.05 and r.mean_delta < 0
            else "significant ↑" if r.p_value_holm < 0.05 and r.mean_delta > 0
            else "null"
        )
        lines.append(
            f"| {r.arm} | {r.n_pairs} | {r.mean_delta:+.4f}"
            f" | {r.wilcoxon_stat:.1f} | {r.p_value:.4f}"
            f" | {r.p_value_holm:.4f} | {verdict} |"
        )
    lines.extend([
        "",
        f"### Classification-only paired Wilcoxon vs {reference_arm} (Holm-corrected)",
        "",
        "| Arm | n | mean Δ | W | p | p (Holm) | verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for r in paired_class:
        verdict = (
            "—" if math.isnan(r.p_value_holm)
            else "significant ↓" if r.p_value_holm < 0.05 and r.mean_delta < 0
            else "significant ↑" if r.p_value_holm < 0.05 and r.mean_delta > 0
            else "null"
        )
        lines.append(
            f"| {r.arm} | {r.n_pairs} | {r.mean_delta:+.4f}"
            f" | {r.wilcoxon_stat:.1f} | {r.p_value:.4f}"
            f" | {r.p_value_holm:.4f} | {verdict} |"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artefact-dir", default=str(_DEFAULT_DIR))
    p.add_argument(
        "--output",
        default=str(_DEFAULT_DIR / "cross_arm_paired.json"),
    )
    p.add_argument("--reference-arm", default="canonical")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    art = Path(args.artefact_dir)
    paired_dual = cross_arm_paired(art, head_mode="dual", reference_arm=args.reference_arm)
    paired_class = cross_arm_paired(art, head_mode="classification", reference_arm=args.reference_arm)
    headline = headline_means(art)
    md = render_markdown(headline, paired_dual, paired_class, reference_arm=args.reference_arm)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "reference_arm": args.reference_arm,
        "headline": headline,
        "paired_dual": [asdict(r) for r in paired_dual],
        "paired_classification": [asdict(r) for r in paired_class],
        "markdown": md,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out}\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
