"""Run A analysis: transformer + VIX-on + text-off pooled F1 + bootstrap.

Compares to JOB 2 transformer (text-on, no VIX) which scored 0.5147 cls pooled.
The right comparison for #217 ensemble decision was already done; for the
text-vs-architecture isolation question, the right comparison is:

  transformer + VIX-off + text-on  (JOB 2)  -> 0.5147 pooled cls
  transformer + VIX-on  + text-off (RUN A)  -> ?
  transformer + VIX-on  + text-on            -> NOT RUN

If RUN A is similar to or above 0.5147, the lift is mostly transformer
expressivity + vol persistence, not text. If RUN A is well below, the 0.5147
needed text.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

PATH = Path("backend/artifacts/experiments/jobA_transformer_vixon_textoff_20260531T171043Z/arch_sweep_raw/canonical_vix/transformer/forecaster_sweep_results.json")

def main():
    d = json.load(open(PATH))
    print(f"trials: {len(d['trials'])} (5 HP x 5 seeds x 5 folds = 125 expected)")
    trials = d['trials']
    
    # Group by (seed, fold), pick best HP by val_metrics.regime_f1_macro
    groups = defaultdict(list)
    for t in trials:
        key = (t['seed'], t['fold_id'])
        groups[key].append(t)
    
    pooled_preds = []
    pooled_targets = []
    n_cells = 0
    for ts in groups.values():
        def val_f1(t):
            return (t['summary'].get('val_metrics') or {}).get('regime_f1_macro') or -1
        best = max(ts, key=val_f1)
        tm = best['summary'].get('test_metrics') or {}
        preds = tm.get('predictions') or []
        targs = tm.get('targets') or []
        if preds and targs and len(preds) == len(targs):
            pooled_preds.extend(preds)
            pooled_targets.extend(targs)
            n_cells += 1
    
    print(f"cells with predictions: {n_cells}")
    print(f"pooled rows: {len(pooled_preds)}")
    if not pooled_preds:
        print("no predictions; cannot compute")
        return
    
    point = f1_score(pooled_targets, pooled_preds, labels=[0,1,2], average='macro', zero_division=0)
    print(f"\nRun A transformer + VIX-on + text-off pooled cls F1 = {point:.4f}")
    
    # Block bootstrap CI (block=10, matching horizon)
    rng = np.random.default_rng(42)
    n = len(pooled_preds)
    pa = np.asarray(pooled_preds); ta = np.asarray(pooled_targets)
    block_size = 10
    n_blocks = int(np.ceil(n / block_size))
    f1s = []
    for _ in range(2000):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        flat = (starts[:, None] + np.arange(block_size)[None, :]).flatten()[:n] % n
        f1s.append(f1_score(ta[flat], pa[flat], labels=[0,1,2], average='macro', zero_division=0))
    lo, hi = np.percentile(f1s, [2.5, 97.5])
    print(f"block-bootstrap mean = {float(np.mean(f1s)):.4f}, 95% CI = [{lo:.4f}, {hi:.4f}]")
    
    print()
    print("--- Comparison to JOB 2 transformer (text-on, no VIX) ---")
    print(f"  JOB 2 transformer text-on / VIX-off pooled cls = 0.5147 [0.4950, 0.5376]")
    print(f"  RUN A transformer text-off / VIX-on  pooled cls = {point:.4f} [{lo:.4f}, {hi:.4f}]")
    delta = point - 0.5147
    sign = '+' if delta >= 0 else '-'
    print(f"  Δ (Run A - JOB 2) = {sign}{abs(delta):.4f}")
    
    if lo > 0.5376:
        verdict = "RUN A clearly EXCEEDS JOB 2 text-on -> text didn't help (VIX + transformer alone is stronger)"
    elif hi < 0.4950:
        verdict = "RUN A clearly BELOW JOB 2 text-on -> text was contributing the lift"
    else:
        verdict = "CIs overlap -> text contribution is statistically indistinguishable from VIX+architecture alone"
    print(f"\nVerdict: {verdict}")

if __name__ == "__main__":
    main()
