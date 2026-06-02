"""Heterogeneous QLIKE-RV ensemble: MLP + sigma-LSTM (pre-registered).

Tests whether adding architecturally-diverse sigma-LSTM members to the
production MLP-only ensemble lowers walk-forward QLIKE. Pre-registration:
docs/research/qlike-heterogeneous-ensemble-preregistration.md

Arms (mean of member predictions in log-RV space):
  A MLP-5    : production baseline (MLP seeds 11,22,33,44,55)
  B MLP-8    : count-matched control (MLP seeds 11..88)
  C Hetero-8 : MLP{11,22,33,44,55} + sigmaLSTM{11,22,33}

Same data/protocol as the canonical eval (5-fold expanding walk-forward, embargo
h+1, 300 epochs), scored on the sigma-LSTM's scorable origin set so the two
families are directly averageable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from app.data.dense_daily_dataset import walk_forward_splits
from app.data.dense_forecast_train import _fit_predict_ols
from app.data.intraday_rv_arch import _feature_matrix, build_sequences, _train_arch_fold
from app.data.intraday_rv_forecast import (
    _forward_log_rv,
    _qlike,
    _qlike_pointwise,
    _train_fold_qlike,
)

RV_PATH = "data/external/alphavantage_bars/spx_5min_daily_rv.parquet"
OUT = "data/artifacts/qlike_hetero_ensemble/result.json"
HORIZONS = (1, 5, 22)
N_FOLDS = 5
EPOCHS = int(os.environ.get("QLIKE_HETERO_EPOCHS", "300"))
MLP_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
SLSTM_SEEDS = [11, 22, 33]
ARMS = {
    "A_mlp5": {"mlp": [11, 22, 33, 44, 55], "slstm": []},
    "B_mlp8": {"mlp": MLP_SEEDS, "slstm": []},
    "C_hetero8": {"mlp": [11, 22, 33, 44, 55], "slstm": SLSTM_SEEDS},
}


def _ens(preds: dict[str, np.ndarray], spec: dict[str, list[int]]) -> np.ndarray:
    members = [preds[f"mlp_{s}"] for s in spec["mlp"]] + [
        preds[f"slstm_{s}"] for s in spec["slstm"]
    ]
    return np.mean(np.stack(members, axis=0), axis=0)  # mean in log-RV space


def _gain_ci(
    better: np.ndarray, worse: np.ndarray, true: np.ndarray, *, block: int, alpha: float, seed: int = 11
) -> list[float]:
    """Two-sided (1-alpha) CI of mean QLIKE gain (worse - better); >0 ⇒ better wins."""

    gain = _qlike_pointwise(worse, true) - _qlike_pointwise(better, true)
    rng = np.random.default_rng(seed)
    n = len(gain)
    if n <= block:
        return [float("nan"), float("nan")]
    n_blocks = int(np.ceil(n / block))
    boots = []
    for _ in range(1000):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        boots.append(float(gain[idx].mean()))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return [float(np.percentile(boots, lo)), float(np.percentile(boots, hi))]


def main() -> None:
    data = _feature_matrix(RV_PATH)
    rv, har, full = data["rv"], data["har"], data["full"]
    seqs = build_sequences(full, np.ones(len(rv), dtype=bool))
    origin, seq_all = seqs["origin"], seqs["seq"]

    out: dict[str, Any] = {"epochs": EPOCHS, "n_days": int(len(rv)), "by_horizon": {}}
    for h in HORIZONS:
        y = _forward_log_rv(rv, h)
        ok = ~np.isnan(y[origin])
        idx = np.where(ok)[0]  # positions into origin
        folds = walk_forward_splits(len(idx), n_folds=N_FOLDS, embargo=h + 1)

        pools: dict[str, list[float]] = {f"mlp_{s}": [] for s in MLP_SEEDS}
        pools.update({f"slstm_{s}": [] for s in SLSTM_SEEDS})
        pools["har"] = []
        pools["true"] = []
        for tr_l, te_l in folds:
            tr_pos, te_pos = idx[np.array(tr_l)], idx[np.array(te_l)]
            d_tr, d_te = origin[tr_pos], origin[te_pos]
            ytr, yte = y[d_tr], y[d_te]
            har_pred = _fit_predict_ols(har[d_tr], ytr, har[d_te])
            har_fit_tr = _fit_predict_ols(har[d_tr], ytr, har[d_tr])
            resid = (ytr - har_fit_tr).reshape(-1, 1)
            for s in MLP_SEEDS:
                pred = _train_fold_qlike(
                    har_fit_tr, full[d_tr], resid, full[d_te], har_pred,
                    seed=s, epochs=EPOCHS, device="cpu",
                )
                pools[f"mlp_{s}"].extend(pred.tolist())
            for s in SLSTM_SEEDS:
                pred = _train_arch_fold(
                    "sigma_lstm", seq_all[tr_pos], seq_all[te_pos],
                    har[d_tr], har[d_te], ytr, seed=s, epochs=EPOCHS, device="cpu",
                )
                pools[f"slstm_{s}"].extend(pred.tolist())
            pools["har"].extend(har_pred.tolist())
            pools["true"].extend(yte.tolist())

        p = {k: np.asarray(v) for k, v in pools.items()}
        true = p["true"]
        ens = {a: _ens(p, spec) for a, spec in ARMS.items()}
        row: dict[str, Any] = {
            "n_eval": int(len(true)),
            "qlike": {"har": _qlike(p["har"], true), **{a: _qlike(e, true) for a, e in ens.items()}},
            "seed_qlike": {
                **{f"mlp_{s}": _qlike(p[f"mlp_{s}"], true) for s in MLP_SEEDS},
                **{f"slstm_{s}": _qlike(p[f"slstm_{s}"], true) for s in SLSTM_SEEDS},
            },
            # primary: C over A, Bonferroni 96.7% (alpha=0.10/3) AND standard 90%
            "C_vs_A_ci967": _gain_ci(ens["C_hetero8"], ens["A_mlp5"], true, block=h, alpha=0.10 / 3),
            "C_vs_A_ci90": _gain_ci(ens["C_hetero8"], ens["A_mlp5"], true, block=h, alpha=0.10),
            # secondary: C over B (diversity beyond count), 90%
            "C_vs_B_ci90": _gain_ci(ens["C_hetero8"], ens["B_mlp8"], true, block=h, alpha=0.10),
            # context: each arm vs HAR, 90%
            "vs_har_ci90": {
                a: _gain_ci(e, p["har"], true, block=h, alpha=0.10) for a, e in ens.items()
            },
        }
        out["by_horizon"][f"h{h}"] = row
        c = row["C_vs_A_ci967"]
        print(
            f"h{h}: HAR {row['qlike']['har']:.4f} | A {row['qlike']['A_mlp5']:.4f} "
            f"B {row['qlike']['B_mlp8']:.4f} C {row['qlike']['C_hetero8']:.4f} | "
            f"C-vs-A 96.7%CI [{c[0]:+.4f},{c[1]:+.4f}]"
        )

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(out, indent=2))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
