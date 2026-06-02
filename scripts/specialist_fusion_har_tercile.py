"""Per-horizon LSTM specialist on the fusion TP for HAR-tercile classification.

Specialist arm for the HAR-tercile improvement bake-off. For each horizon
h in {1, 5, 22} an LSTM with asymmetric (class-balanced focal) loss is
trained over a 20-step sequence window of fusion daily features. The
target is the train-slice tercile bucket of rv_fwd_h. The evaluation
protocol matches the recovered baseline exactly: pooled macro-F1 across
5 walk-forward expanding folds with embargo=23.

Output: docs/research/har-tercile-fusion-specialist_fusion-result.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


_N_CLASSES = 3
_HORIZONS = (1, 5, 22)
_N_FOLDS = 5
_EMBARGO = max(_HORIZONS) + 1
_SEQ_LEN = 20
_BATCH = 128
_EPOCHS = 40
_LR = 1e-3
_HIDDEN = 64
_DROPOUT = 0.2
_PATIENCE = 6
_VAL_FRAC = 0.15
_FOCAL_GAMMA = 2.0
_CB_BETA = 0.999
_SEEDS = (11, 29, 47, 71, 97)

_FEATURE_COLS = [
    "rv_daily",
    "rv_weekly",
    "rv_monthly",
    "volume_daily",
    "volume_weekly",
    "volume_monthly",
    "downside_daily",
    "downside_weekly",
    "downside_monthly",
    "jump_daily",
    "jump_weekly",
    "jump_monthly",
    "days_since_stmt",
    "days_to_stmt",
    "surprise_level",
    "surprise_path",
    "surprise_info",
    "doc_age_days",
]


def _walk_forward_splits(n: int, *, n_folds: int = _N_FOLDS, embargo: int = _EMBARGO):
    if n < n_folds * (embargo + 2):
        raise ValueError(f"too few rows ({n}) for {n_folds} folds with embargo {embargo}")
    test_size = n // (n_folds + 1)
    start = n - test_size * n_folds
    folds = []
    cursor = start
    for _ in range(n_folds):
        train_idx = list(range(0, cursor - embargo))
        test_idx = list(range(cursor, cursor + test_size))
        folds.append((train_idx, test_idx))
        cursor += test_size
    return folds


def _labels(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(values, thresholds).astype(np.int64)


def _macro_f1(true: np.ndarray, pred: np.ndarray, n_classes: int = _N_CLASSES) -> float:
    f1s = []
    for c in range(n_classes):
        tp = float(np.sum((pred == c) & (true == c)))
        fp = float(np.sum((pred == c) & (true != c)))
        fn = float(np.sum((pred != c) & (true == c)))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


class LSTMSpecialist(nn.Module):
    def __init__(self, n_features: int, hidden: int = _HIDDEN, dropout: float = _DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, _N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


def _focal_class_balanced_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_counts: np.ndarray,
    *,
    beta: float = _CB_BETA,
    gamma: float = _FOCAL_GAMMA,
) -> torch.Tensor:
    """Class-balanced focal loss. Asymmetric across class supports."""
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / np.where(effective_num > 0, effective_num, 1.0)
    weights = weights / weights.sum() * len(class_counts)
    w_tensor = torch.tensor(weights, dtype=logits.dtype, device=logits.device)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    tgt_one = nn.functional.one_hot(target, num_classes=_N_CLASSES).to(logits.dtype)
    pt = (probs * tgt_one).sum(dim=-1).clamp(min=1e-12)
    focal = -((1.0 - pt) ** gamma) * torch.log(pt)
    per_class_w = (w_tensor.unsqueeze(0) * tgt_one).sum(dim=-1)
    return (focal * per_class_w).mean()


def _build_sequences(
    features: np.ndarray, idx_subset: np.ndarray, seq_len: int = _SEQ_LEN
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sequence_array, target_idx_array). Each row i in idx_subset
    yields a sequence ending at i (inclusive). If i < seq_len-1 the sequence
    is left-padded with the earliest available row.
    """
    n_feat = features.shape[1]
    out = np.zeros((len(idx_subset), seq_len, n_feat), dtype=np.float32)
    valid_target_idx = []
    for k, i in enumerate(idx_subset):
        lo = max(0, i - seq_len + 1)
        window = features[lo : i + 1]
        if len(window) < seq_len:
            pad = np.tile(window[0:1], (seq_len - len(window), 1))
            window = np.concatenate([pad, window], axis=0)
        out[k] = window
        valid_target_idx.append(i)
    return out, np.array(valid_target_idx, dtype=np.int64)


def _fit_horizon_seed(
    seed: int,
    *,
    features: np.ndarray,
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    target: np.ndarray,
    tr_global_idx: np.ndarray,
    te_global_idx: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Fit one (fold, horizon, seed) and return test logits."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Standardize features using train slice only.
    feats_std = (features - feature_means) / np.where(feature_stds > 0, feature_stds, 1.0)
    feats_std = feats_std.astype(np.float32)

    # Tercile thresholds from train-slice targets.
    y_tr_raw = target[tr_global_idx]
    thr = np.quantile(y_tr_raw, [1.0 / 3.0, 2.0 / 3.0])
    y_tr = _labels(y_tr_raw, thr)
    y_te = _labels(target[te_global_idx], thr)

    # Hold out the tail of train as validation for early stopping.
    n_tr = len(tr_global_idx)
    n_val = max(1, int(n_tr * _VAL_FRAC))
    tr_main_idx = tr_global_idx[: n_tr - n_val]
    val_idx = tr_global_idx[n_tr - n_val :]
    y_main = y_tr[: n_tr - n_val]
    y_val = y_tr[n_tr - n_val :]

    x_tr, _ = _build_sequences(feats_std, tr_main_idx)
    x_val, _ = _build_sequences(feats_std, val_idx)
    x_te, _ = _build_sequences(feats_std, te_global_idx)

    class_counts = np.bincount(y_main, minlength=_N_CLASSES).astype(np.float64)

    x_tr_t = torch.from_numpy(x_tr)
    y_tr_t = torch.from_numpy(y_main)
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    x_te_t = torch.from_numpy(x_te).to(device)

    train_loader = DataLoader(
        TensorDataset(x_tr_t, y_tr_t),
        batch_size=_BATCH,
        shuffle=True,
        drop_last=False,
    )

    model = LSTMSpecialist(features.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=_EPOCHS)

    best_val_f1 = -1.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad = 0
    for ep in range(_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = _focal_class_balanced_loss(logits, yb, class_counts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            vl = model(x_val_t).argmax(dim=-1).cpu().numpy()
        val_f1 = _macro_f1(y_val, vl)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= _PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_logits = model(x_te_t).cpu().numpy()
    return te_logits, thr, y_te


def main() -> int:
    fusion_path = Path(
        "/home/yusuf/SWE599/fed-pulse/data/processed/"
        "tp_intraday_fomc_text_volatility/fusion/daily_fusion.parquet"
    )
    out_path = Path(
        "/home/yusuf/SWE599/fed-pulse/docs/research/"
        "har-tercile-fusion-specialist_fusion-result.json"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    daily = pd.read_parquet(fusion_path).sort_values("date").reset_index(drop=True)

    # Build feature matrix: imputed for non-HAR cols with median to keep rows.
    feats = daily[_FEATURE_COLS].to_numpy(dtype=np.float64)
    # Median impute missing values column-wise (computed on the full
    # series here for sequence construction; per-fold standardization
    # below uses train-only stats so test info is not leaked into the
    # model. The presence/absence pattern itself contains no future
    # information beyond the column being NaN at that calendar row).
    for j in range(feats.shape[1]):
        col = feats[:, j]
        if not np.isfinite(col).all():
            med = np.nanmedian(col)
            feats[~np.isfinite(col), j] = med

    har = daily[["rv_daily", "rv_weekly", "rv_monthly"]].to_numpy(dtype=np.float64)
    targets = np.column_stack(
        [daily[f"rv_fwd_{h}"].to_numpy(dtype=np.float64) for h in _HORIZONS]
    )

    valid = np.isfinite(targets).all(axis=1) & np.isfinite(har).all(axis=1)
    idx_all = np.where(valid)[0]
    folds = _walk_forward_splits(len(idx_all))

    per_horizon = {}
    for k, h in enumerate(_HORIZONS):
        target_h = targets[:, k]
        pooled_true = []
        pooled_pred = []
        per_fold = []
        for fi, (tr_l, te_l) in enumerate(folds, start=1):
            tr_global = idx_all[np.array(tr_l)]
            te_global = idx_all[np.array(te_l)]

            mean = feats[tr_global].mean(axis=0)
            std = feats[tr_global].std(axis=0)

            # Ensemble across seeds via averaged logits.
            te_logits_acc = None
            thr_used = None
            y_te = None
            for seed in _SEEDS:
                te_logits, thr, y_te_seed = _fit_horizon_seed(
                    seed,
                    features=feats,
                    feature_means=mean,
                    feature_stds=std,
                    target=target_h,
                    tr_global_idx=tr_global,
                    te_global_idx=te_global,
                    device=device,
                )
                te_logits_acc = te_logits if te_logits_acc is None else te_logits_acc + te_logits
                if thr_used is None:
                    thr_used = thr
                    y_te = y_te_seed
            te_logits_avg = te_logits_acc / len(_SEEDS)
            preds = te_logits_avg.argmax(axis=-1)

            f1 = _macro_f1(y_te, preds)
            per_fold.append(
                {
                    "fold": fi,
                    "n_train": int(len(tr_global)),
                    "n_test": int(len(te_global)),
                    "macro_f1": f1,
                    "q33": float(thr_used[0]),
                    "q67": float(thr_used[1]),
                }
            )
            pooled_true.append(y_te)
            pooled_pred.append(preds)
            print(f"  h={h} fold={fi} macro_f1={f1:.4f}", flush=True)

        pooled_t = np.concatenate(pooled_true)
        pooled_p = np.concatenate(pooled_pred)
        f1s = np.array([row["macro_f1"] for row in per_fold])
        per_horizon[f"h{h}"] = {
            "per_fold": per_fold,
            "fold_macro_f1_mean": float(f1s.mean()),
            "fold_macro_f1_std": float(f1s.std(ddof=0)),
            "pooled_macro_f1": _macro_f1(pooled_t, pooled_p),
            "n_pooled": int(len(pooled_t)),
        }
        print(
            f"h={h} pooled_macro_f1={per_horizon[f'h{h}']['pooled_macro_f1']:.4f}"
            f" fold_mean={per_horizon[f'h{h}']['fold_macro_f1_mean']:.4f}"
            f" +/- {per_horizon[f'h{h}']['fold_macro_f1_std']:.4f}",
            flush=True,
        )

    # Recovered baseline comparator.
    baseline = {
        "h1": {"pooled": 0.6872512285221816, "fold_mean": 0.6292101379398904, "fold_std": 0.04248175764239546},
        "h5": {"pooled": 0.6850235551635501, "fold_mean": 0.6178191494707697, "fold_std": 0.03422667311197767},
        "h22": {"pooled": 0.6541869390643578, "fold_mean": 0.5541906851615217, "fold_std": 0.04649472175893584},
    }

    delta = {}
    beats_outside_ci = False
    for hk in ("h1", "h5", "h22"):
        spec_mean = per_horizon[hk]["fold_macro_f1_mean"]
        base_mean = baseline[hk]["fold_mean"]
        base_std = baseline[hk]["fold_std"]
        d_pooled = per_horizon[hk]["pooled_macro_f1"] - baseline[hk]["pooled"]
        d_mean = spec_mean - base_mean
        beats = (spec_mean - base_mean) > base_std
        if beats:
            beats_outside_ci = True
        delta[hk] = {
            "specialist_pooled": per_horizon[hk]["pooled_macro_f1"],
            "baseline_pooled": baseline[hk]["pooled"],
            "delta_pooled": d_pooled,
            "specialist_fold_mean": spec_mean,
            "baseline_fold_mean": base_mean,
            "baseline_fold_std": base_std,
            "delta_fold_mean": d_mean,
            "beats_baseline_by_more_than_1sigma": bool(beats),
        }

    result = {
        "arm_key": "specialist_fusion",
        "arm_name": "Per-horizon specialist on fusion TP",
        "approach": (
            "2-layer LSTM (hidden=64, dropout=0.2) over a 20-step sequence window "
            "of fusion HAR + cross-feature inputs, trained per-horizon with "
            "class-balanced focal loss (beta=0.999, gamma=2.0). Ensemble of 5 "
            "seeds (11, 29, 47, 71, 97), averaged logits. Tercile thresholds taken "
            "from the train slice (q33/q67). Early stopping on a 15% tail-of-train "
            "validation split."
        ),
        "protocol": {
            "source_parquet": str(fusion_path),
            "feature_cols": _FEATURE_COLS,
            "har_lags": "rv_daily / rv_weekly / rv_monthly (log-space)",
            "forward_targets": "rv_fwd_1 / rv_fwd_5 / rv_fwd_22",
            "folds": f"walk_forward_splits n_folds={_N_FOLDS} embargo={_EMBARGO}",
            "valid_mask": "finite HAR lags + finite forward targets",
            "tercile_thresholds": "q33/q67 of train-slice forward targets",
            "macro_f1": "pooled across folds (canonical); also per-fold mean+/-std",
            "horizons": list(_HORIZONS),
            "seeds": list(_SEEDS),
            "seq_len": _SEQ_LEN,
            "epochs": _EPOCHS,
            "patience": _PATIENCE,
            "batch_size": _BATCH,
            "lr": _LR,
            "hidden": _HIDDEN,
            "dropout": _DROPOUT,
            "focal_gamma": _FOCAL_GAMMA,
            "class_balanced_beta": _CB_BETA,
        },
        "n_rows_total": int(len(daily)),
        "n_rows_valid": int(len(idx_all)),
        "by_horizon": per_horizon,
        "recovered_baseline": baseline,
        "delta_vs_recovered_baseline": delta,
        "beats_baseline_outside_1sigma": beats_outside_ci,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
