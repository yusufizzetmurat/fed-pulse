"""Chance-level null table for the absolute-label regime task (§6.29-style).

Majority-class and stratified-random nulls computed against the SAME absolute
label set the wave-3 absolute_regime arm used, with the SAME macro-F1 convention
(mean over classes present in the test fold). Train-prior derived from
events.parquet via the runner's own vol_regime_absolute_class_for; test counts
taken from the arm artefact's per-fold confusion matrices (exact).
CPU only.
"""
import json
import numpy as np
import pandas as pd
from app.training.loaders import vol_regime_absolute_class_for

ART = "backend/artifacts/experiments/runpod_wave3_absolute_regime_20260530T211214Z.json"
EVENTS = "data/processed/canonical/events.parquet"
MANIFEST = "data/processed/canonical/fold_manifest_expanding_walk_forward.json"
SEEDS = [11, 29, 47, 71, 97]

d = json.load(open(ART))
THR = tuple(d["absolute_vol_thresholds"])           # exact per-period cutoffs the arm used
MODEL_CLS = d["summary"]["classification"]["regime_f1_macro"]["mean"]

def macro_f1_present(y_true, y_pred, n_classes=3):
    """Macro-F1 averaged over classes PRESENT in y_true (matches the arm)."""
    present = [c for c in range(n_classes) if np.any(y_true == c)]
    f1s = []
    for c in present:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0

# --- test counts per fold from the artefact (seed-independent true supports) ---
test_counts = {}
for t in d["trials"]["classification"]:
    for f in t["folds"]:
        cb = f["metrics"].get("classification_breakdown")
        if not cb:
            continue
        sup = tuple(c["support"] for c in cb["per_class"])
        test_counts.setdefault(f["fold_id"], sup)

# --- train prior per fold from events.parquet, labeled with the runner's fn ---
ev = pd.read_parquet(EVENTS, columns=["event_date", "forward_realized_vol_10d"])
ev["lab"] = ev["forward_realized_vol_10d"].apply(lambda v: vol_regime_absolute_class_for(v, THR))
manifest = json.load(open(MANIFEST))["folds"]
folds = {f["fold_id"]: f for f in manifest}

print(f"absolute thresholds (per-period): {THR}")
print(f"model cls macro-F1 (arm)        : {MODEL_CLS:.4f}\n")
print(f"{'fold':12s} {'test_counts':>14s} {'train_maj':>9s} {'major_f1':>8s} {'strat_f1':>8s}")

maj_all, strat_all = [], []
for fid, sup in test_counts.items():
    fo = folds[fid]
    tr = ev[(ev.event_date >= fo["train_start"]) & (ev.event_date <= fo["train_end"]) & (ev.lab >= 0)]
    tr_counts = np.array([int((tr.lab == c).sum()) for c in range(3)], dtype=float)
    prior = tr_counts / tr_counts.sum()
    maj = int(np.argmax(tr_counts))
    # exact test label vector from artefact supports
    y_true = np.concatenate([np.full(n, c) for c, n in enumerate(sup)]).astype(int)
    # majority null (deterministic): predict train-majority class everywhere
    maj_f1 = macro_f1_present(y_true, np.full(y_true.shape, maj))
    # stratified null: sample test preds ~ train prior, average over the seed grid
    s_vals = []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        y_pred = rng.choice(3, size=y_true.shape, p=prior)
        s_vals.append(macro_f1_present(y_true, y_pred))
    strat_f1 = float(np.mean(s_vals))
    maj_all.append(maj_f1); strat_all.append(strat_f1)
    print(f"{fid:12s} {str(sup):>14s} {maj:>9d} {maj_f1:>8.4f} {strat_f1:>8.4f}")

print(f"\n{'':12s} {'MEAN over folds':>14s}  ->  majority={np.mean(maj_all):.4f}  stratified={np.mean(strat_all):.4f}")
print(f"model (arm)                                      = {MODEL_CLS:.4f}")
print(f"model - majority_null = {MODEL_CLS-np.mean(maj_all):+.4f}   model - stratified_null = {MODEL_CLS-np.mean(strat_all):+.4f}")

out = {
    "label_set": "absolute", "thresholds_per_period": list(THR),
    "calm_max_annualized": d.get("absolute_calm_max_annualized"),
    "high_min_annualized": d.get("absolute_high_min_annualized"),
    "macro_f1_convention": "mean over classes present in test fold (matches arm)",
    "model_cls_macro_f1": MODEL_CLS,
    "majority_class_null_macro_f1": float(np.mean(maj_all)),
    "stratified_random_null_macro_f1": float(np.mean(strat_all)),
    "model_minus_majority": MODEL_CLS - float(np.mean(maj_all)),
    "model_minus_stratified": MODEL_CLS - float(np.mean(strat_all)),
    "per_fold": [
        {"fold_id": fid, "test_counts": list(test_counts[fid])} for fid in test_counts
    ],
}
json.dump(out, open("backend/artifacts/experiments/absolute_label_nulls.json", "w"), indent=2)
print("\nwrote backend/artifacts/experiments/absolute_label_nulls.json")
