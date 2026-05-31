"""Moving-block bootstrap on Run B dual-residual.

For each arm (vixon_baseline, vixon_bge) extract per-(seed, fold) predictions
and targets from the dual head. Pool. Compute pooled macro-F1 and the
moving-block bootstrap CI with block_size=10 (matching the 10d forward
realized vol horizon). Then compute (fused - baseline) gap and its CI.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import f1_score

OUTDIR = Path("backend/artifacts/experiments/jobB_vixon_bootstrap_20260531T171043Z")

def pool_arm(path: Path, head_mode: str = "dual"):
    d = json.load(open(path))
    trials = d["trials"].get(head_mode) or []
    cells = []
    for t in trials:
        seed = t["seed"]
        for fold in t["folds"]:
            fid = fold["fold_id"]
            m = fold["metrics"]
            preds = m.get("predictions") or []
            targs = m.get("targets") or []
            if not preds or not targs or len(preds) != len(targs):
                continue
            cells.append({"seed": seed, "fold": fid, "preds": preds, "targs": targs})
    return cells

def block_bootstrap_f1(preds, targs, n_boot=2000, block_size=10, seed=42):
    n = len(preds)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    f1s = []
    preds_a = np.asarray(preds); targs_a = np.asarray(targs)
    for _ in range(n_boot):
        start_idx = rng.integers(0, n - block_size + 1, size=n_blocks)
        block_starts = start_idx[:, None] + np.arange(block_size)[None, :]
        flat = block_starts.flatten()[:n] % n
        b_preds = preds_a[flat]
        b_targs = targs_a[flat]
        f1s.append(f1_score(b_targs, b_preds, labels=[0,1,2], average="macro", zero_division=0))
    return float(np.mean(f1s)), float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))

def gap_bootstrap(pred_a, targ_a, pred_b, targ_b, n_boot=2000, block_size=10, seed=42):
    """Bootstrap (F1_a - F1_b) by resampling block indices applied to BOTH series
    on the same indices when the rows align. The targets must be identical
    across arms for the gap to be meaningful per row; here we assume the
    per-(seed,fold) cells are matched by index so the pooled order is the
    same per arm."""
    n = min(len(pred_a), len(pred_b))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    gaps = []
    pa = np.asarray(pred_a[:n]); ta = np.asarray(targ_a[:n])
    pb = np.asarray(pred_b[:n]); tb = np.asarray(targ_b[:n])
    for _ in range(n_boot):
        start_idx = rng.integers(0, n - block_size + 1, size=n_blocks)
        block_starts = start_idx[:, None] + np.arange(block_size)[None, :]
        flat = block_starts.flatten()[:n] % n
        f_a = f1_score(ta[flat], pa[flat], labels=[0,1,2], average="macro", zero_division=0)
        f_b = f1_score(tb[flat], pb[flat], labels=[0,1,2], average="macro", zero_division=0)
        gaps.append(f_a - f_b)
    return float(np.mean(gaps)), float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))

def main():
    arms = {}
    for arm_name, fname in [("baseline", "vixon_baseline.json"), ("bge", "vixon_bge.json")]:
        cells = pool_arm(OUTDIR / fname, head_mode="dual")
        # Sort by (seed, fold) for deterministic pooling order matching across arms
        cells.sort(key=lambda c: (c["seed"], c["fold"]))
        pooled_preds = [p for c in cells for p in c["preds"]]
        pooled_targs = [t for c in cells for t in c["targs"]]
        arms[arm_name] = {"cells": cells, "preds": pooled_preds, "targs": pooled_targs}
        print(f"=== arm {arm_name}: {len(cells)} cells, {len(pooled_preds)} pooled rows ===")

    print()
    print("--- Per-arm pooled dual-head macro-F1 ---")
    for arm, d in arms.items():
        mean, lo, hi = block_bootstrap_f1(d["preds"], d["targs"], n_boot=2000, block_size=10)
        # also report point estimate
        point = f1_score(d["targs"], d["preds"], labels=[0,1,2], average="macro", zero_division=0)
        print(f"  {arm:10s}  point={point:.4f}  block-bootstrap mean={mean:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")

    # Gap bootstrap (bge - baseline)
    print()
    print("--- Gap (bge - baseline) block-bootstrap, block=10 ---")
    gap_mean, gap_lo, gap_hi = gap_bootstrap(arms["bge"]["preds"], arms["bge"]["targs"],
                                              arms["baseline"]["preds"], arms["baseline"]["targs"],
                                              n_boot=2000, block_size=10)
    print(f"  Δ(bge-baseline) point = {arms['bge']['preds'] and (f1_score(arms['bge']['targs'], arms['bge']['preds'], labels=[0,1,2], average='macro', zero_division=0) - f1_score(arms['baseline']['targs'], arms['baseline']['preds'], labels=[0,1,2], average='macro', zero_division=0)):.4f}")
    print(f"  block-bootstrap mean Δ = {gap_mean:.4f}")
    print(f"  95% CI               = [{gap_lo:.4f}, {gap_hi:.4f}]")
    print(f"  Clears 0?            : {'NO (CI straddles 0)' if gap_lo <= 0 <= gap_hi else 'YES (positive)' if gap_lo > 0 else 'YES (negative)'}")

if __name__ == "__main__":
    main()
