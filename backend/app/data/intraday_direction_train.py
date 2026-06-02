"""Train + walk-forward-evaluate the intraday direction model.

Self-contained harness: bypasses the training-package loader and feeds
tensors straight to MultiModalForecasterModel (pre-announcement 1-min bar
sequence + FinBERT pooled text -> 2-class reaction direction). Reports
pooled out-of-fold directional accuracy with a bootstrap CI vs the
majority-class and market-only baselines, for both target windows.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np


def _bar_features(close: list[float], volume: list[float]) -> np.ndarray:
    """(T, 3) scale-free features: log_return, cum_log_return, volume_z."""

    c = np.asarray(close, dtype=np.float64)
    v = np.asarray(volume, dtype=np.float64)
    log_ret = np.zeros_like(c)
    log_ret[1:] = np.log(c[1:] / c[:-1])
    cum = np.cumsum(log_ret)
    vstd = v.std()
    vol_z = (v - v.mean()) / vstd if vstd > 0 else np.zeros_like(v)
    return np.stack([log_ret, cum, vol_z], axis=1)


def _walk_forward_folds(n: int, n_folds: int = 4) -> list[tuple[list[int], list[int]]]:
    """Expanding walk-forward: growing train head, next contiguous test block.

    Splits the tail into ``n_folds`` test blocks; fold k trains on
    everything before its block. Requires >= 2*n_folds events.
    """

    if n < 2 * n_folds:
        raise ValueError(f"too few events ({n}) for {n_folds} walk-forward folds")
    test_size = n // (n_folds + 1)
    start = n - test_size * n_folds
    folds: list[tuple[list[int], list[int]]] = []
    cursor = start
    for _ in range(n_folds):
        train_idx = list(range(cursor))
        test_idx = list(range(cursor, cursor + test_size))
        folds.append((train_idx, test_idx))
        cursor += test_size
    return folds


def _accuracy(pred: list[int], true: list[int]) -> float:
    if not true:
        return float("nan")
    return float(np.mean([int(p == t) for p, t in zip(pred, true)]))


def _majority_baseline_accuracy(train_y: list[int], test_y: list[int]) -> float:
    maj = int(round(float(np.mean(train_y)))) if train_y else 0
    return _accuracy([maj] * len(test_y), test_y)


def _bootstrap_ci(
    correct: list[int], *, n_boot: int = 1000, seed: int = 11, alpha: float = 0.1
) -> tuple[float, float, float]:
    """Percentile bootstrap CI over a 0/1 correctness vector."""

    arr = np.asarray(correct, dtype=np.float64)
    point = float(arr.mean()) if arr.size else float("nan")
    if arr.size == 0:
        return float("nan"), point, float("nan")
    rng = np.random.default_rng(seed)
    boots = [float(arr[rng.integers(0, arr.size, arr.size)].mean()) for _ in range(n_boot)]
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, point, hi


def _standardize_per_fold(x: np.ndarray, train_idx: list[int]) -> np.ndarray:
    """Z-score each feature using stats from train events' bars only."""

    train_bars = x[train_idx].reshape(-1, x.shape[-1])
    mean = train_bars.mean(axis=0)
    std = train_bars.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return cast(np.ndarray, (x - mean) / std)


def build_arrays(
    df: Any, window: str, *, embed_fn: Callable[[str], np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x[N,T,F], text_emb[N,768], y[N]) for the chosen window.

    ``window`` is "immediate" or "delayed"; ``embed_fn(text)->np.ndarray[768]``
    is injected so tests can stub the FinBERT call.
    """

    df = df.sort_values("event_date").reset_index(drop=True)
    x = np.stack([_bar_features(r["pre_close"], r["pre_volume"]) for _, r in df.iterrows()])
    text = np.stack([embed_fn(str(t)) for t in df["text"]])
    y = df[f"dir_{window}"].astype(int).to_numpy()
    return x.astype(np.float32), text.astype(np.float32), y


# --------------------------------------------------------------------------
# Integration: text embedding + torch training (exercised by the live run).
# --------------------------------------------------------------------------


def _embed_text(text: str) -> np.ndarray:
    """Mean-pool FinBERT per-chunk CLS embeddings -> (768,)."""

    from app.services.text_encoder import encode_chunks

    encs = encode_chunks(text)
    vecs = [np.asarray(e.embedding, dtype=np.float64) for e in encs if e.embedding]
    if not vecs:
        return np.zeros(768, dtype=np.float32)
    return cast(np.ndarray, np.mean(vecs, axis=0).astype(np.float32))


def _train_one_fold(
    x_tr: np.ndarray,
    t_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    t_te: np.ndarray,
    *,
    seed: int,
    epochs: int,
    device: str,
    market_only: bool,
) -> list[int]:
    import torch
    from torch import nn

    from app.determinism import enable_deterministic_mode
    from app.models.multimodal_forecaster import MultiModalForecasterModel

    enable_deterministic_mode(seed)
    dev = torch.device(device)
    model = MultiModalForecasterModel(
        architecture="gru",
        market_input_size=x_tr.shape[-1],
        text_embedding_dim=t_tr.shape[-1],
        n_classes=2,
        hidden_size=48,
        num_layers=1,
        dropout=0.1,
    ).to(dev)

    # 20% tail of the (chronological) train head held out for early stopping.
    n_tr = x_tr.shape[0]
    n_val = max(1, n_tr // 5)
    tr_sl, val_sl = slice(0, n_tr - n_val), slice(n_tr - n_val, n_tr)

    def _tensors(x: np.ndarray, t: np.ndarray) -> tuple[Any, Any, Any]:
        xb = torch.tensor(x, dtype=torch.float32, device=dev)
        tb = torch.tensor(t, dtype=torch.float32, device=dev)
        miss = (
            torch.ones((x.shape[0], 1), device=dev)
            if market_only
            else torch.zeros((x.shape[0], 1), device=dev)
        )
        return xb, tb, miss

    xb_tr, tb_tr, miss_tr = _tensors(x_tr[tr_sl], t_tr[tr_sl])
    yb_tr = torch.tensor(y_tr[tr_sl], dtype=torch.long, device=dev)
    xb_val, tb_val, miss_val = _tensors(x_tr[val_sl], t_tr[val_sl])
    yb_val = torch.tensor(y_tr[val_sl], dtype=torch.long, device=dev)
    xb_te, tb_te, miss_te = _tensors(x_te, t_te)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    patience, bad = 30, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(xb_tr, text_embedding=tb_tr, text_embedding_missing=miss_tr)
        loss = loss_fn(logits, yb_tr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vlogits = model(xb_val, text_embedding=tb_val, text_embedding_missing=miss_val)
            vloss = float(loss_fn(vlogits, yb_val))
        if vloss < best_val - 1e-5:
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
        te_logits = model(xb_te, text_embedding=tb_te, text_embedding_missing=miss_te)
        preds = te_logits.argmax(dim=1).cpu().tolist()
    return [int(p) for p in preds]


def run(
    df: Any,
    window: str,
    *,
    embed_fn: Callable[[str], np.ndarray] | None = None,
    seed: int = 11,
    epochs: int = 150,
    device: str = "cpu",
    n_folds: int = 4,
    market_only: bool = False,
) -> dict[str, Any]:
    embed_fn = embed_fn or _embed_text
    x, text, y = build_arrays(df, window, embed_fn=embed_fn)
    folds = _walk_forward_folds(x.shape[0], n_folds=n_folds)
    pooled_correct: list[int] = []
    pooled_true: list[int] = []
    pooled_majority_correct: list[int] = []
    per_fold: list[dict[str, Any]] = []
    for fi, (tr, te) in enumerate(folds):
        xs = _standardize_per_fold(x, tr)
        preds = _train_one_fold(
            xs[tr],
            text[tr],
            y[tr],
            xs[te],
            text[te],
            seed=seed,
            epochs=epochs,
            device=device,
            market_only=market_only,
        )
        true = [int(v) for v in y[te]]
        maj = int(round(float(np.mean(y[tr]))))
        pooled_correct += [int(p == t) for p, t in zip(preds, true)]
        pooled_majority_correct += [int(maj == t) for t in true]
        pooled_true += true
        per_fold.append({"fold": fi, "n_test": len(te), "acc": _accuracy(preds, true)})
    lo, point, hi = _bootstrap_ci(pooled_correct, seed=seed)
    return {
        "window": window,
        "market_only": market_only,
        "n_events": int(x.shape[0]),
        "n_folds": n_folds,
        "pooled_accuracy": point,
        "ci90": [lo, hi],
        "majority_baseline": float(np.mean(pooled_majority_correct))
        if pooled_majority_correct
        else float("nan"),
        "per_fold": per_fold,
        "config": {"arch": "gru", "hidden": 48, "layers": 1, "epochs": epochs, "seed": seed},
    }


def main() -> int:
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Train + walk-forward eval the intraday direction model."
    )
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=150)
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
        result = {"built_at_utc": built_at, "full": full, "market_only": mkt}
        out = args.out_dir / f"result_{window}.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(
            f"[{window}] full={full['pooled_accuracy']:.3f} "
            f"CI90={full['ci90'][0]:.3f}-{full['ci90'][1]:.3f} | "
            f"market_only={mkt['pooled_accuracy']:.3f} | "
            f"majority={full['majority_baseline']:.3f} (n={full['n_events']}) -> {out}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
