"""Phase 4 — the honest late-fusion experiment.

Answers the question the rebuild was built to answer: does FOMC/Fed text carry a
market-reaction signal once the data, encoder, and model are correct? Compares
three configurations per frame and head with walk-forward, seed-averaged, and a
moving-block bootstrap on the text marginal value (full minus market-only):

* market-only  — structured (market + SEP) features
* text-only    — FinBERT-fed embedding only
* full fusion  — late concat of both

Leak discipline: per fold, the standardizer, the text PCA, and the optional
text-on-market residualizer are ALL fit on the train split only; walk-forward
uses an embargo. The residualizer is the fault-class-#4 ablation: running with
and without it measures whether the prior pipeline's leak controls were
suppressing real text signal.
"""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from app.config import DATA_DIR
from app.data.late_fusion_model import LateFusionModel, joint_loss

logger = logging.getLogger(__name__)

_BASE = DATA_DIR / "processed" / "late_fusion"

# Compact, economically-motivated SEP feature set (dense relative horizons).
_SEP_FEATURES = [
    "sep_point_ffr_h0",
    "sep_point_ffr_h1",
    "sep_point_ffr_LR",
    "sep_disp_ffr_h0",
    "sep_point_gdp_h0",
    "sep_point_unemployment_h0",
    "sep_point_pce_h0",
    "sep_available",
]


@dataclass
class FrameSpec:
    frame: str
    emb: str
    join_key: str
    market_features: list[str]
    sep_features: list[str]
    dir_col: str
    mag_col: str


_SPECS = {
    "event": FrameSpec(
        frame="event_frame.parquet",
        emb="event_text_emb.parquet",
        join_key="event_date",
        market_features=["pre_ret", "pre_rv", "log_pre_volume"],
        sep_features=_SEP_FEATURES,
        dir_col="dir_immediate",
        mag_col="mag_immediate",
    ),
    "daily": FrameSpec(
        frame="daily_frame.parquet",
        emb="daily_text_emb.parquet",
        join_key="row_hash",
        market_features=["ret_5d", "ret_22d", "rv_22"],
        sep_features=[],
        dir_col="dir_nextday",
        mag_col="mag_nextday",
    ),
}


@dataclass
class Config:
    n_folds: int = 5
    embargo: int = 5
    pca_k: int = 12
    seeds: tuple[int, ...] = (11, 22, 33)
    epochs: int = 150
    lr: float = 1e-3
    weight_decay: float = 1e-3
    n_boot: int = 2000
    block: int = 5
    extras: dict[str, object] = field(default_factory=dict)


@dataclass
class Loaded:
    text: np.ndarray
    struct: np.ndarray
    y_dir: np.ndarray
    y_mag: np.ndarray
    struct_names: list[str]
    doc_type: np.ndarray


def load_frame(spec: FrameSpec) -> Loaded:
    frame = pd.read_parquet(_BASE / spec.frame)
    emb = pd.read_parquet(_BASE / spec.emb)
    emb_cols = [c for c in emb.columns if c.startswith("emb_")]
    # join embeddings to the frame by the stable key (NOT by position)
    merged = frame.merge(emb[[spec.join_key, *emb_cols]], on=spec.join_key, how="inner")
    if len(merged) != len(frame):
        logger.warning("join kept %d of %d frame rows", len(merged), len(frame))

    if "pre_volume" in merged.columns:
        merged["log_pre_volume"] = np.log1p(merged["pre_volume"].fillna(0.0))

    struct_names = [c for c in spec.market_features if c in merged.columns]
    sep_names = [c for c in spec.sep_features if c in merged.columns]
    struct_names = struct_names + sep_names

    text = merged[emb_cols].to_numpy(dtype=np.float32)
    struct = merged[struct_names].to_numpy(dtype=np.float32)
    y_dir = merged[spec.dir_col].to_numpy(dtype=np.float32)
    y_mag = merged[spec.mag_col].to_numpy(dtype=np.float32)
    doc_type = (
        merged["doc_type"].to_numpy()
        if "doc_type" in merged.columns
        else np.array(["statement"] * len(merged))
    )
    # keep only rows with a defined target
    ok = ~(np.isnan(y_dir) | np.isnan(y_mag))
    return Loaded(text[ok], struct[ok], y_dir[ok], y_mag[ok], struct_names, doc_type[ok])


def walk_forward_splits(n: int, n_folds: int, embargo: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window splits; train excludes the last ``embargo`` rows before test."""
    fold = n // (n_folds + 1)
    splits = []
    for i in range(1, n_folds + 1):
        test_start = i * fold
        test_end = (i + 1) * fold if i < n_folds else n
        train_end = max(0, test_start - embargo)
        if train_end < 20 or test_start >= test_end:
            continue
        splits.append((np.arange(train_end), np.arange(test_start, test_end)))
    return splits


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    std[std == 0] = 1.0
    tr = np.nan_to_num((train - mean) / std, nan=0.0)
    te = np.nan_to_num((test - mean) / std, nan=0.0)
    return tr.astype(np.float32), te.astype(np.float32)


def _pca(train: np.ndarray, test: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    u, s, vt = np.linalg.svd(train - mean, full_matrices=False)
    comp = vt[:k]
    return ((train - mean) @ comp.T).astype(np.float32), ((test - mean) @ comp.T).astype(np.float32)


def _residualize(
    text_tr: np.ndarray, struct_tr: np.ndarray, text_te: np.ndarray, struct_te: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the part of text linearly predictable from struct (train-fit OLS)."""
    aug_tr = np.hstack([struct_tr, np.ones((len(struct_tr), 1), dtype=np.float32)])
    beta, *_ = np.linalg.lstsq(aug_tr, text_tr, rcond=None)
    aug_te = np.hstack([struct_te, np.ones((len(struct_te), 1), dtype=np.float32)])
    return (text_tr - aug_tr @ beta).astype(np.float32), (text_te - aug_te @ beta).astype(
        np.float32
    )


def _train_predict(
    text_tr: np.ndarray,
    struct_tr: np.ndarray,
    ydir_tr: np.ndarray,
    ymag_tr: np.ndarray,
    text_te: np.ndarray,
    struct_te: np.ndarray,
    use_text: bool,
    use_struct: bool,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Seed-averaged predictions (dir probability, magnitude) on the test split."""
    dir_probs, mags = [], []
    tt = torch.from_numpy(text_tr)
    st = torch.from_numpy(struct_tr)
    yd = torch.from_numpy(ydir_tr)
    ym = torch.from_numpy(ymag_tr)
    tte, ste = torch.from_numpy(text_te), torch.from_numpy(struct_te)
    for seed in cfg.seeds:
        torch.manual_seed(seed)
        model = LateFusionModel(
            text_dim=text_tr.shape[1],
            struct_dim=struct_tr.shape[1],
            use_text=use_text,
            use_struct=use_struct,
        )
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        model.train()
        for _ in range(cfg.epochs):
            opt.zero_grad()
            dl, mg = model(tt, st)
            loss = joint_loss(dl, mg, yd, ym)
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
        model.eval()
        with torch.no_grad():
            dl, mg = model(tte, ste)
            dir_probs.append(torch.sigmoid(dl).numpy())
            mags.append(mg.numpy())
    return np.mean(dir_probs, axis=0), np.mean(mags, axis=0)


def _r2(y: np.ndarray, pred: np.ndarray, baseline: float) -> float:
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - baseline) ** 2))
    return 1.0 - sse / sst if sst > 0 else float("nan")


@dataclass
class OOS:
    y_dir: np.ndarray
    y_mag: np.ndarray
    mag_baseline: np.ndarray
    market: dict[str, np.ndarray]
    text: dict[str, np.ndarray]
    full: dict[str, np.ndarray]
    doc_type: np.ndarray


def run_frame(spec: FrameSpec, cfg: Config, residualize: bool) -> OOS:
    data = load_frame(spec)
    n = len(data.y_dir)
    splits = walk_forward_splits(n, cfg.n_folds, cfg.embargo)
    first_oos = int(splits[0][1].min()) if splits else -1
    logger.info(
        "%s: %d/%d folds usable (n=%d); OOS starts at row %d (rows 0-%d are train-only)",
        spec.frame,
        len(splits),
        cfg.n_folds,
        n,
        first_oos,
        max(first_oos - 1, 0),
    )
    acc: dict[str, list[np.ndarray]] = {
        "y_dir": [],
        "y_mag": [],
        "mag_baseline": [],
        "doc_type": [],
        "market_dir": [],
        "market_mag": [],
        "text_dir": [],
        "text_mag": [],
        "full_dir": [],
        "full_mag": [],
    }
    for train_idx, test_idx in splits:
        s_tr, s_te = _standardize(data.struct[train_idx], data.struct[test_idx])
        t_tr, t_te = _pca(data.text[train_idx], data.text[test_idx], cfg.pca_k)
        if residualize:
            t_tr, t_te = _residualize(t_tr, s_tr, t_te, s_te)
        yd_tr = data.y_dir[train_idx]
        # Standardize the magnitude target on TRAIN so the linear head predicts a
        # z-scored value; tiny raw |return| targets are otherwise unlearnable.
        mag_mean = float(data.y_mag[train_idx].mean())
        mag_std = float(data.y_mag[train_idx].std()) or 1.0
        ym_tr = ((data.y_mag[train_idx] - mag_mean) / mag_std).astype(np.float32)
        ym_te_s = ((data.y_mag[test_idx] - mag_mean) / mag_std).astype(np.float32)

        configs = {
            "market": (False, True),
            "text": (True, False),
            "full": (True, True),
        }
        for name, (use_text, use_struct) in configs.items():
            dprob, mag = _train_predict(
                t_tr, s_tr, yd_tr, ym_tr, t_te, s_te, use_text, use_struct, cfg
            )
            acc[f"{name}_dir"].append(dprob)
            acc[f"{name}_mag"].append(mag)
        acc["y_dir"].append(data.y_dir[test_idx])
        acc["y_mag"].append(ym_te_s)  # standardized magnitude target
        # baseline in standardized space is the train mean -> 0
        acc["mag_baseline"].append(np.zeros(len(test_idx), dtype=np.float32))
        acc["doc_type"].append(data.doc_type[test_idx])

    cat = {k: np.concatenate(v) for k, v in acc.items()}
    return OOS(
        y_dir=cat["y_dir"],
        y_mag=cat["y_mag"],
        mag_baseline=cat["mag_baseline"],
        market={"dir": cat["market_dir"], "mag": cat["market_mag"]},
        text={"dir": cat["text_dir"], "mag": cat["text_mag"]},
        full={"dir": cat["full_dir"], "mag": cat["full_mag"]},
        doc_type=cat["doc_type"],
    )


def _acc(y: np.ndarray, prob: np.ndarray) -> float:
    return float(((prob > 0.5).astype(np.float32) == y).mean())


def _mcnemar(
    y: np.ndarray, prob_full: np.ndarray, prob_market: np.ndarray
) -> tuple[int, int, float]:
    """Paired McNemar test (continuity-corrected) for full vs market direction.

    b = full correct & market wrong; c = full wrong & market correct. Returns
    (b, c, two-sided p-value). Unlike the bootstrap this is a proper paired test
    and does not depend on resampling assumptions.
    """
    yi = y.astype(int)
    cf = (prob_full > 0.5).astype(int) == yi
    cm = (prob_market > 0.5).astype(int) == yi
    b = int((cf & ~cm).sum())
    c = int((~cf & cm).sum())
    if b + c == 0:
        return b, c, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)  # chi-square, 1 df
    p = math.erfc(math.sqrt(stat / 2.0))
    return b, c, float(p)


def _block_boot_ci(fn: Callable[[np.ndarray], float], oos: OOS, cfg: Config) -> tuple[float, float]:
    """Moving-block bootstrap CI on a scalar statistic ``fn(idx)`` over OOS rows."""
    n = len(oos.y_dir)
    rng = np.random.default_rng(0)
    n_blocks = int(np.ceil(n / cfg.block))
    vals = []
    for _ in range(cfg.n_boot):
        starts = rng.integers(0, max(1, n - cfg.block + 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + cfg.block, n)) for s in starts])[:n]
        vals.append(fn(idx))
    lo, hi = np.percentile(vals, [5, 95])
    return float(lo), float(hi)


def summarize(spec: FrameSpec, oos: OOS, cfg: Config, residualize: bool) -> dict[str, object]:
    majority = max(float(oos.y_dir.mean()), 1.0 - float(oos.y_dir.mean()))
    acc_market = _acc(oos.y_dir, oos.market["dir"])
    acc_text = _acc(oos.y_dir, oos.text["dir"])
    acc_full = _acc(oos.y_dir, oos.full["dir"])
    r2_market = _r2(oos.y_mag, oos.market["mag"], float(oos.mag_baseline.mean()))
    r2_text = _r2(oos.y_mag, oos.text["mag"], float(oos.mag_baseline.mean()))
    r2_full = _r2(oos.y_mag, oos.full["mag"], float(oos.mag_baseline.mean()))

    dir_gain = _block_boot_ci(
        lambda i: _acc(oos.y_dir[i], oos.full["dir"][i]) - _acc(oos.y_dir[i], oos.market["dir"][i]),
        oos,
        cfg,
    )
    base = float(oos.mag_baseline.mean())
    mag_gain = _block_boot_ci(
        lambda i: _r2(oos.y_mag[i], oos.full["mag"][i], base)
        - _r2(oos.y_mag[i], oos.market["mag"][i], base),
        oos,
        cfg,
    )
    b, c, mcnemar_p = _mcnemar(oos.y_dir, oos.full["dir"], oos.market["dir"])
    return {
        "frame": spec.frame,
        "residualize": residualize,
        "n": int(len(oos.y_dir)),
        "majority": round(majority, 4),
        "dir_acc": {
            "market": round(acc_market, 4),
            "text": round(acc_text, 4),
            "full": round(acc_full, 4),
        },
        "dir_text_lift_ci90": [round(dir_gain[0], 4), round(dir_gain[1], 4)],
        "dir_mcnemar": {"full_better": b, "market_better": c, "p": round(mcnemar_p, 4)},
        "mag_r2": {
            "market": round(r2_market, 4),
            "text": round(r2_text, 4),
            "full": round(r2_full, 4),
        },
        "mag_text_lift_ci90": [round(mag_gain[0], 4), round(mag_gain[1], 4)],
        "note": "bootstrap CI is conditional on fitted models (may be narrow); mcnemar is the paired test",
    }


def per_doctype(oos: OOS) -> list[dict[str, object]]:
    """Per-doc-type direction breakdown (which communications carry the signal)."""
    rows: list[dict[str, object]] = []
    for dt in sorted(set(oos.doc_type.tolist())):
        mask = oos.doc_type == dt
        if int(mask.sum()) < 30:
            continue
        b, c, p = _mcnemar(oos.y_dir[mask], oos.full["dir"][mask], oos.market["dir"][mask])
        rows.append(
            {
                "doc_type": dt,
                "n": int(mask.sum()),
                "acc_market": round(_acc(oos.y_dir[mask], oos.market["dir"][mask]), 4),
                "acc_full": round(_acc(oos.y_dir[mask], oos.full["dir"][mask]), 4),
                "mcnemar": {"full_better": b, "market_better": c, "p": round(p, 4)},
            }
        )
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run the late-fusion experiment.")
    parser.add_argument("--frames", nargs="+", default=["event", "daily"])
    parser.add_argument("--pca-k", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    cfg = Config(n_folds=args.n_folds, pca_k=args.pca_k, epochs=args.epochs)
    for frame_name in args.frames:
        spec = _SPECS[frame_name]
        cfg.block = 5 if frame_name == "event" else 22
        for residualize in (False, True):
            oos = run_frame(spec, cfg, residualize)
            result = summarize(spec, oos, cfg, residualize)
            tag = "WITH leak-control (residualized)" if residualize else "WITHOUT leak-control"
            print(f"\n=== {frame_name.upper()} | {tag} ===")
            for key, val in result.items():
                print(f"  {key}: {val}")
            if frame_name == "daily" and not residualize:
                print("  per_doc_type:")
                for row in per_doctype(oos):
                    print(f"    {row}")


if __name__ == "__main__":
    main()
