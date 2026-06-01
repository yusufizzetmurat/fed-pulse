"""Volume head — forward daily volume forecaster.

Unlike text, volume is genuinely predictive: daily log-volume is highly persistent
(lag-1 autocorrelation ~0.99) and clusters, so a HAR-style model forecasts it well.
This builds the deployable forward-volume head (1d/1w/1mo), parallel to the QLIKE-RV
volatility forecaster: a HAR baseline (Corsi lags on log-volume) vs a small DL head,
walk-forward with embargo, moving-block bootstrap on the R^2 gain, and a saved
artifact the serving layer can load for an "expected volume" card.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from app.config import DATA_DIR
from app.data.late_fusion_experiment import _standardize, walk_forward_splits

logger = logging.getLogger(__name__)
_HORIZONS = (1, 5, 22)
_MAX_LAG = 22


def load_log_volume(path: Path, symbol: str = "^GSPC") -> pd.DataFrame:
    """Daily log-volume series, sorted, positive volumes only."""
    frame = pd.read_parquet(path)
    if "symbol" in frame.columns and frame["symbol"].nunique() > 1:
        frame = frame[frame["symbol"] == symbol]
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"].astype(str).str[:10])
    frame = frame.sort_values("date").drop_duplicates("date")
    frame = frame[frame["volume"].astype(float) > 0]
    frame["log_vol"] = np.log(frame["volume"].astype(float))
    return frame[["date", "log_vol"]].reset_index(drop=True)


def _har_matrix(log_vol: np.ndarray) -> np.ndarray:
    """Corsi HAR features on log-volume: [lag1, mean over 5, mean over 22]."""
    n = len(log_vol)
    feats = np.full((n, 3), np.nan)
    for t in range(_MAX_LAG, n):
        feats[t, 0] = log_vol[t - 1]
        feats[t, 1] = log_vol[t - 5 : t].mean()
        feats[t, 2] = log_vol[t - 22 : t].mean()
    return feats


def _forward_target(log_vol: np.ndarray, h: int) -> np.ndarray:
    """Mean log-volume over t+1 .. t+h (forward, strictly future)."""
    n = len(log_vol)
    out = np.full(n, np.nan)
    for t in range(n - h):
        out[t] = log_vol[t + 1 : t + 1 + h].mean()
    return out


def _ols(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray) -> np.ndarray:
    aug_tr = np.hstack([x_tr, np.ones((len(x_tr), 1))])
    beta, *_ = np.linalg.lstsq(aug_tr, y_tr, rcond=None)
    aug_te = np.hstack([x_te, np.ones((len(x_te), 1))])
    return np.asarray(aug_te @ beta, dtype=float)


def _mlp_fit_predict(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, seed: int, epochs: int = 200
) -> np.ndarray:
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(x_tr.shape[1], 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 1)
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xt, yt = torch.from_numpy(x_tr).float(), torch.from_numpy(y_tr).float().unsqueeze(1)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(xt), yt)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
    model.eval()
    with torch.no_grad():
        return np.asarray(model(torch.from_numpy(x_te).float()).squeeze(1).numpy(), dtype=float)


def _r2(y: np.ndarray, pred: np.ndarray, baseline: float) -> float:
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - baseline) ** 2))
    return 1.0 - sse / sst if sst > 0 else float("nan")


def _block_boot_gain(
    y: np.ndarray, p_dl: np.ndarray, p_har: np.ndarray, base: float, block: int, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    n_blocks = int(np.ceil(n / block))
    vals = []
    for _ in range(2000):
        starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        vals.append(_r2(y[idx], p_dl[idx], base) - _r2(y[idx], p_har[idx], base))
    return float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


def run(
    volume_path: Path, out_path: Path, n_folds: int = 5, seeds: tuple[int, ...] = (11, 22, 33)
) -> dict[str, object]:
    data = load_log_volume(volume_path)
    log_vol = data["log_vol"].to_numpy()
    har = _har_matrix(log_vol)

    results: dict[str, object] = {"model": "volume-head", "n_days": int(len(log_vol)), "by_horizon": {}}
    by_h: dict[str, dict[str, object]] = {}
    for h in _HORIZONS:
        target = _forward_target(log_vol, h)
        valid = ~(np.isnan(har).any(axis=1) | np.isnan(target))
        X, y = har[valid], target[valid]
        splits = walk_forward_splits(len(y), n_folds, embargo=h)
        oy, oh, od = [], [], []
        for tr, te in splits:
            xtr, xte = _standardize(X[tr], X[te])
            # HAR (OLS) is scale-robust; the MLP needs a standardized target
            # (log-volume ~22) or it cannot fit, so train on z-scored y and invert.
            ym, ys = float(y[tr].mean()), float(y[tr].std()) or 1.0
            oh.append(_ols(xtr, y[tr], xte))
            dl_std = np.mean(
                [_mlp_fit_predict(xtr, (y[tr] - ym) / ys, xte, s) for s in seeds], axis=0
            )
            od.append(dl_std * ys + ym)
            oy.append(y[te])
        yy, ph, pd_ = np.concatenate(oy), np.concatenate(oh), np.concatenate(od)
        base = float(yy.mean())
        gain = _block_boot_gain(yy, pd_, ph, base, block=h)
        by_h[f"h{h}"] = {
            "r2_har": round(_r2(yy, ph, base), 4),
            "r2_dl": round(_r2(yy, pd_, base), 4),
            "dl_minus_har_ci90": [round(gain[0], 4), round(gain[1], 4)],
            "n_oos": int(len(yy)),
        }
        logger.info("h%d: HAR R2 %.4f | DL R2 %.4f | gain CI %s", h, by_h[f"h{h}"]["r2_har"], by_h[f"h{h}"]["r2_dl"], by_h[f"h{h}"]["dl_minus_har_ci90"])
    results["by_horizon"] = by_h
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    return results


def run_event_window(event_frame_path: Path, n_folds: int = 5) -> dict[str, object]:
    """Event-window volume head: forecast the FOMC announcement-window volume from
    pre-window features (market-based), walk-forward. Honest predictive feature."""
    frame = pd.read_parquet(event_frame_path)
    frame = frame[(frame["imm_volume"] > 0) & (frame["pre_volume"] > 0)].reset_index(drop=True)
    y = np.log(frame["imm_volume"].to_numpy())
    feats = np.column_stack(
        [
            np.log(frame["pre_volume"].to_numpy()),
            frame["pre_rv"].fillna(frame["pre_rv"].median()).to_numpy(),
        ]
    )
    splits = walk_forward_splits(len(y), n_folds, embargo=1)
    oy, op = [], []
    for tr, te in splits:
        xtr, xte = _standardize(feats[tr], feats[te])
        op.append(_ols(xtr, y[tr], xte))
        oy.append(y[te])
    yy, pp = np.concatenate(oy), np.concatenate(op)
    base = float(yy.mean())
    return {
        "target": "log announcement-window volume",
        "n_oos": int(len(yy)),
        "r2_market": round(_r2(yy, pp, base), 4),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Forward daily volume head.")
    parser.add_argument(
        "--volume", type=Path,
        default=DATA_DIR / "processed" / "tp_v3_full_rebuild_2026_05_30" / "_market_cache" / "GSPC.parquet",
    )
    parser.add_argument(
        "--out", type=Path,
        default=DATA_DIR / "processed" / "late_fusion" / "volume_head_eval.json",
    )
    parser.add_argument(
        "--event-frame", type=Path,
        default=DATA_DIR / "processed" / "late_fusion" / "event_frame.parquet",
    )
    args = parser.parse_args()
    result = run(args.volume, args.out)
    print(json.dumps(result, indent=2))
    if args.event_frame.exists():
        ev = run_event_window(args.event_frame)
        print("event-window volume head:", json.dumps(ev))


if __name__ == "__main__":
    main()
