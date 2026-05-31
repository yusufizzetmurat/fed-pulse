"""Train + walk-forward-evaluate the intraday reaction-MAGNITUDE regressor.

Tests whether FOMC text + pre-announcement bars predict the SIZE of the
announcement-window reaction (|return|) — the volatility/magnitude claim
of the CVJ/Lucca-Moench literature, distinct from the (nulled) direction
target. Reuses the direction harness's pure feature/fold/embedding units;
the model is MultiModalForecasterModel(n_classes=1) trained with MSE.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np

from app.data.intraday_direction_train import (
    _bar_features,
    _embed_text,
    _standardize_per_fold,
    _walk_forward_folds,
)


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def _oos_r2(pred: np.ndarray, true: np.ndarray, *, baseline_pred: float) -> float:
    sse = float(np.sum((true - pred) ** 2))
    sst = float(np.sum((true - baseline_pred) ** 2))
    return 1.0 - sse / sst if sst > 0 else float("nan")


def _spearman(pred: np.ndarray, true: np.ndarray) -> float:
    if len(pred) < 2:
        return float("nan")
    pr = np.argsort(np.argsort(pred)).astype(float)
    tr = np.argsort(np.argsort(true)).astype(float)
    if pr.std() == 0 or tr.std() == 0:
        return float("nan")
    return float(np.corrcoef(pr, tr)[0, 1])


def build_magnitude_arrays(
    df: Any, window: str, *, embed_fn: Callable[[str], np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = df.sort_values("event_date").reset_index(drop=True)
    x = np.stack([_bar_features(r["pre_close"], r["pre_volume"]) for _, r in df.iterrows()])
    text = np.stack([embed_fn(str(t)) for t in df["text"]])
    y = df[f"mag_{window}"].astype(float).to_numpy()
    return x.astype(np.float32), text.astype(np.float32), y.astype(np.float32)


def _oos_r2_pointwise(pred: np.ndarray, true: np.ndarray, base: np.ndarray) -> float:
    """R2 where each test point's baseline is its own fold's train-mean."""

    sse = float(np.sum((true - pred) ** 2))
    sst = float(np.sum((true - base) ** 2))
    return 1.0 - sse / sst if sst > 0 else float("nan")


def _r2_bootstrap_ci(
    pred: np.ndarray,
    true: np.ndarray,
    base: np.ndarray,
    *,
    seed: int = 11,
    n_boot: int = 1000,
    alpha: float = 0.1,
) -> tuple[float, float, float]:
    point = _oos_r2_pointwise(pred, true, base)
    rng = np.random.default_rng(seed)
    n = len(true)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        val = _oos_r2_pointwise(pred[idx], true[idx], base[idx])
        if not np.isnan(val):
            boots.append(val)
    lo = float(np.quantile(boots, alpha / 2)) if boots else float("nan")
    hi = float(np.quantile(boots, 1 - alpha / 2)) if boots else float("nan")
    return lo, point, hi


# --------------------------------------------------------------------------
# Integration: torch regression training (exercised by the live run).
# --------------------------------------------------------------------------


def _train_one_fold_reg(
    x_tr: np.ndarray,
    t_tr: np.ndarray,
    y_tr_std: np.ndarray,
    x_te: np.ndarray,
    t_te: np.ndarray,
    *,
    seed: int,
    epochs: int,
    device: str,
    market_only: bool,
) -> np.ndarray:
    import torch
    from torch import nn

    from app.determinism import enable_deterministic_mode
    from app.models.multimodal_forecaster import MultiModalForecasterModel

    enable_deterministic_mode(seed)
    dev = torch.device(device)
    # The multimodal model enforces n_classes >= 2 (it is classification-first);
    # we use it as a regressor by supervising only the FIRST output unit with
    # MSE and reading that column as the scalar prediction. No model change.
    model = MultiModalForecasterModel(
        architecture="gru",
        market_input_size=x_tr.shape[-1],
        text_embedding_dim=t_tr.shape[-1],
        n_classes=2,
        hidden_size=48,
        num_layers=1,
        dropout=0.1,
    ).to(dev)

    n_tr = x_tr.shape[0]
    n_val = max(1, n_tr // 5)
    tr_sl, val_sl = slice(0, n_tr - n_val), slice(n_tr - n_val, n_tr)

    def _miss(n: int) -> Any:
        return torch.ones((n, 1), device=dev) if market_only else torch.zeros((n, 1), device=dev)

    def _t(a: np.ndarray) -> Any:
        return torch.tensor(a, dtype=torch.float32, device=dev)

    xb_tr, tb_tr = _t(x_tr[tr_sl]), _t(t_tr[tr_sl])
    yb_tr = _t(y_tr_std[tr_sl]).unsqueeze(-1)
    xb_val, tb_val = _t(x_tr[val_sl]), _t(t_tr[val_sl])
    yb_val = _t(y_tr_std[val_sl]).unsqueeze(-1)
    xb_te, tb_te = _t(x_te), _t(t_te)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_val, best_state, bad, patience = float("inf"), None, 0, 40
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(xb_tr, text_embedding=tb_tr, text_embedding_missing=_miss(xb_tr.shape[0]))
        loss = loss_fn(out[:, 0:1], yb_tr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vout = model(
                xb_val, text_embedding=tb_val, text_embedding_missing=_miss(xb_val.shape[0])
            )
            vloss = float(loss_fn(vout[:, 0:1], yb_val))
        if vloss < best_val - 1e-6:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(xb_te, text_embedding=tb_te, text_embedding_missing=_miss(xb_te.shape[0]))
        pred = out[:, 0]
        return cast(np.ndarray, pred.cpu().numpy().astype(np.float64))


def run(
    df: Any,
    window: str,
    *,
    embed_fn: Callable[[str], np.ndarray] | None = None,
    seed: int = 11,
    epochs: int = 200,
    device: str = "cpu",
    n_folds: int = 4,
    market_only: bool = False,
) -> dict[str, Any]:
    embed_fn = embed_fn or _embed_text
    x, text, y = build_magnitude_arrays(df, window, embed_fn=embed_fn)
    folds = _walk_forward_folds(x.shape[0], n_folds=n_folds)
    pred_all: list[float] = []
    true_all: list[float] = []
    base_all: list[float] = []
    per_fold: list[dict[str, Any]] = []
    for fi, (tr, te) in enumerate(folds):
        xs = _standardize_per_fold(x, tr)
        mu = float(y[tr].mean())
        sd = float(y[tr].std()) or 1.0
        y_tr_std = (y[tr] - mu) / sd
        pred_std = _train_one_fold_reg(
            xs[tr],
            text[tr],
            y_tr_std.astype(np.float32),
            xs[te],
            text[te],
            seed=seed,
            epochs=epochs,
            device=device,
            market_only=market_only,
        )
        pred = pred_std * sd + mu
        true = y[te].astype(np.float64)
        pred_all.extend(pred.tolist())
        true_all.extend(true.tolist())
        base_all.extend([mu] * len(te))
        per_fold.append(
            {"fold": fi, "n_test": len(te), "fold_r2": _oos_r2(pred, true, baseline_pred=mu)}
        )
    p = np.asarray(pred_all)
    t = np.asarray(true_all)
    b = np.asarray(base_all)
    lo, r2, hi = _r2_bootstrap_ci(p, t, b, seed=seed)
    return {
        "window": window,
        "market_only": market_only,
        "n_events": int(x.shape[0]),
        "n_folds": n_folds,
        "oos_r2": r2,
        "r2_ci90": [lo, hi],
        "rmse": _rmse(p, t),
        "baseline_rmse": _rmse(b, t),
        "spearman": _spearman(p, t),
        "per_fold": per_fold,
        "config": {"arch": "gru", "hidden": 48, "layers": 1, "epochs": epochs, "seed": seed},
    }


def main() -> int:
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Train + walk-forward eval the magnitude regressor."
    )
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    df = pd.read_parquet(args.events)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    built_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for window in ("immediate", "delayed"):
        full = run(df, window, seed=args.seed, epochs=args.epochs, device=args.device)
        mkt = run(
            df, window, seed=args.seed, epochs=args.epochs, device=args.device, market_only=True
        )
        (args.out_dir / f"result_{window}.json").write_text(
            json.dumps({"built_at_utc": built_at, "full": full, "market_only": mkt}, indent=2),
            encoding="utf-8",
        )
        print(
            f"[{window}] R2={full['oos_r2']:.3f} CI90={full['r2_ci90'][0]:.3f}..{full['r2_ci90'][1]:.3f} "
            f"rmse={full['rmse']:.5f} vs base={full['baseline_rmse']:.5f} "
            f"spearman={full['spearman']:.3f} | market_only R2={mkt['oos_r2']:.3f} (n={full['n_events']})"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
