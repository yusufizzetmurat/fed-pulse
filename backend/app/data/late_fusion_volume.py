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


def _calendar_features(dates: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Volume-seasonality features HAR's 3 lags miss: day-of-week and month/
    quarter-end (rebalancing / triple-witching volume spikes). Computable from the
    date alone, so leak-free."""
    d = pd.to_datetime(dates)
    dow = d.dt.dayofweek.to_numpy()
    dom = d.dt.day.to_numpy()
    month = d.dt.month.to_numpy()
    cols = [(dow == k).astype(float) for k in range(4)]  # Mon..Thu (Fri baseline)
    names = [f"dow_{k}" for k in range(4)]
    cols.append((dom >= 25).astype(float))
    names.append("month_end")
    cols.append(((dom >= 25) & np.isin(month, [3, 6, 9, 12])).astype(float))
    names.append("quarter_end")
    return np.column_stack(cols), names


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
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, seed: int, epochs: int = 400
) -> np.ndarray:
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(x_tr.shape[1], 64), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
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
    cal, _ = _calendar_features(data["date"])

    results: dict[str, object] = {"model": "volume-head", "n_days": int(len(log_vol)), "by_horizon": {}}
    by_h: dict[str, dict[str, object]] = {}
    for h in _HORIZONS:
        target = _forward_target(log_vol, h)
        valid = ~(np.isnan(har).any(axis=1) | np.isnan(target))
        # HAR-3 baseline vs a rich feature set (HAR + calendar seasonality). The
        # rich-LINEAR model isolates the value of the features; DL-rich then tests
        # whether nonlinearity adds anything beyond that.
        x_har, x_rich, y = har[valid], np.hstack([har[valid], cal[valid]]), target[valid]
        splits = walk_forward_splits(len(y), n_folds, embargo=h)
        oy, o_har, o_rich, o_dl = [], [], [], []
        for tr, te in splits:
            xh_tr, xh_te = _standardize(x_har[tr], x_har[te])
            xr_tr, xr_te = _standardize(x_rich[tr], x_rich[te])
            ym, ys = float(y[tr].mean()), float(y[tr].std()) or 1.0
            o_har.append(_ols(xh_tr, y[tr], xh_te))
            o_rich.append(_ols(xr_tr, y[tr], xr_te))
            dl_std = np.mean(
                [_mlp_fit_predict(xr_tr, (y[tr] - ym) / ys, xr_te, s) for s in seeds], axis=0
            )
            o_dl.append(dl_std * ys + ym)
            oy.append(y[te])
        yy = np.concatenate(oy)
        ph, pr, pdl = np.concatenate(o_har), np.concatenate(o_rich), np.concatenate(o_dl)
        base = float(yy.mean())
        dl_gain = _block_boot_gain(yy, pdl, ph, base, block=h)
        rich_gain = _block_boot_gain(yy, pr, ph, base, block=h)
        by_h[f"h{h}"] = {
            "r2_har": round(_r2(yy, ph, base), 4),
            "r2_rich_linear": round(_r2(yy, pr, base), 4),
            "r2_dl": round(_r2(yy, pdl, base), 4),
            "dl_minus_har_ci90": [round(dl_gain[0], 4), round(dl_gain[1], 4)],
            "richlin_minus_har_ci90": [round(rich_gain[0], 4), round(rich_gain[1], 4)],
            "n_oos": int(len(yy)),
        }
        logger.info(
            "h%d: HAR %.4f | rich-lin %.4f | DL %.4f | DL-HAR %s | richlin-HAR %s",
            h, by_h[f"h{h}"]["r2_har"], by_h[f"h{h}"]["r2_rich_linear"], by_h[f"h{h}"]["r2_dl"],
            by_h[f"h{h}"]["dl_minus_har_ci90"], by_h[f"h{h}"]["richlin_minus_har_ci90"],
        )
    results["by_horizon"] = by_h
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    return results


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """One-sided ``1-alpha`` quantile of absolute residuals (log space)."""
    if scores.size == 0:
        return 0.0
    return float(np.quantile(np.abs(scores), 1.0 - alpha))


def fit_production_artifact(
    volume_path: Path,
    out_path: Path,
    *,
    n_folds: int = 5,
    seeds: tuple[int, ...] = (11, 22, 33),
    alphas: tuple[float, ...] = (0.10, 0.20),
    symbol: str = "^GSPC",
) -> dict[str, object]:
    """Build the deployable HAR-volume serving artifact.

    Distinct from :func:`run`, which is an evaluation-only routine that
    reports R^2 / DL-minus-HAR CIs. This builder fits the HAR baseline on
    ALL valid-target history (no test split), collects pooled walk-forward
    OOS residuals to derive the prospective conformal quantiles, and
    saves the per-horizon serving spec the
    :mod:`app.services.volume_forecaster` layer consumes:

    - ``har_coef``: ``[intercept, daily, weekly, monthly]`` on the
      Corsi triple (lag1, mean over 5, mean over 22), full-history fit.
    - ``calendar_dummy_names`` / ``calendar_dummy_coef``: weekday +
      month-end / quarter-end seasonality block, full-history fit.
    - ``conformal_quantiles``: ``{"0.10": q90, "0.20": q80}`` half-widths
      from pooled walk-forward OOS HAR residuals.
    - ``r2_har``: pooled walk-forward HAR R^2 for the calibration chip.
    """

    data = load_log_volume(volume_path, symbol=symbol)
    log_vol = data["log_vol"].to_numpy()
    har = _har_matrix(log_vol)
    cal, cal_names = _calendar_features(data["date"])

    spec: dict[str, object] = {
        "model": "volume_har",
        "symbol": symbol,
        "n_days": int(len(log_vol)),
        "seeds": list(seeds),
        "n_folds": n_folds,
        "alphas": list(alphas),
        "date_first": str(data["date"].iloc[0].date()),
        "date_last": str(data["date"].iloc[-1].date()),
        "by_horizon": {},
    }
    by_h: dict[str, dict[str, object]] = {}

    for h in _HORIZONS:
        target = _forward_target(log_vol, h)
        valid = ~(np.isnan(har).any(axis=1) | np.isnan(target))
        x_har, y = har[valid], target[valid]
        x_cal = cal[valid]

        # Full-history HAR coefficient fit (no train/test split — the
        # serving layer needs the all-data estimate).
        a_all = np.column_stack([np.ones(len(x_har)), x_har])
        har_coef, *_ = np.linalg.lstsq(a_all, y, rcond=None)

        # Full-history seasonality coefficient fit on the HAR residuals,
        # so dot(cal, cal_coef) captures the calendar contribution
        # additively on log space.
        har_fit = a_all @ har_coef
        cal_resid = y - har_fit
        cal_coef, *_ = np.linalg.lstsq(x_cal, cal_resid, rcond=None)

        # Pooled walk-forward HAR OOS residuals → prospective conformal
        # widths. Uses the same walk-forward scaffolding as ``run``.
        splits = walk_forward_splits(len(y), n_folds, embargo=h)
        oy, op = [], []
        for tr, te in splits:
            xh_tr, xh_te = _standardize(x_har[tr], x_har[te])
            op.append(_ols(xh_tr, y[tr], xh_te))
            oy.append(y[te])
        yy, pp = np.concatenate(oy), np.concatenate(op)
        base = float(yy.mean())
        scores = yy - pp
        quantiles = {f"{a:.2f}": _conformal_quantile(scores, a) for a in alphas}

        by_h[f"h{h}"] = {
            "har_coef": [float(c) for c in har_coef.tolist()],
            "calendar_dummy_names": list(cal_names),
            "calendar_dummy_coef": [float(c) for c in cal_coef.tolist()],
            "conformal_quantiles": quantiles,
            "r2_har": round(_r2(yy, pp, base), 4),
            "n_oos": int(len(yy)),
        }
        logger.info(
            "h%d serving artifact: HAR R^2 %.4f | q80 %.4f | q90 %.4f",
            h,
            by_h[f"h{h}"]["r2_har"],
            quantiles.get("0.20", 0.0),
            quantiles.get("0.10", 0.0),
        )

    spec["by_horizon"] = by_h
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, indent=2))
    return spec


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
    parser.add_argument(
        "--serving-artifact", type=Path, default=None,
        help=(
            "Path for the deployable serving JSON consumed by "
            "app.services.volume_forecaster (har_coef + calendar block + "
            "conformal_quantiles). Omit to skip the production fit."
        ),
    )
    args = parser.parse_args()
    result = run(args.volume, args.out)
    print(json.dumps(result, indent=2))
    if args.serving_artifact is not None:
        artifact = fit_production_artifact(args.volume, args.serving_artifact)
        print("serving artifact horizons:", list(artifact["by_horizon"]))  # type: ignore[arg-type]
    if args.event_frame.exists():
        ev = run_event_window(args.event_frame)
        print("event-window volume head:", json.dumps(ev))


if __name__ == "__main__":
    main()
