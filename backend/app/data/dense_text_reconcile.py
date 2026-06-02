"""Reconcile the event-based vol-regime lift with the dense regression null.

Runs BOTH tasks in one framework, on the FOMC-event sample, matching the
event-based architecture (LSTM-attn over a pre-announcement market
sequence + an encoder text embedding):

  - CLASSIFICATION: 3-class realized-vol regime (terciles) — macro-F1.
  - REGRESSION:     realized-vol magnitude (log rv_5) — out-of-sample R².

For each encoder, compares market-only (text zeroed) vs +text, walk-forward
with bootstrap CIs. If text lifts the classification F1 but not the
regression R², the two analyses reconcile: text helps the coarse regime
bucket, not the magnitude.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from app.data.dense_daily_dataset import build_dataset, walk_forward_splits

_EPS = 1e-8
_SEQ_LEN = 22


def _regime_thresholds(y_train: np.ndarray) -> tuple[float, float]:
    """Tercile cut points fit on TRAIN realized vol (no leakage)."""

    return float(np.quantile(y_train, 1 / 3)), float(np.quantile(y_train, 2 / 3))


def _to_regime(y: np.ndarray, thresholds: tuple[float, float]) -> np.ndarray:
    lo, hi = thresholds
    return cast(np.ndarray, np.digitize(y, [lo, hi]))  # 0 / 1 / 2


def _macro_f1(pred: np.ndarray, true: np.ndarray, n_classes: int = 3) -> float:
    f1s = []
    for c in range(n_classes):
        tp = float(np.sum((pred == c) & (true == c)))
        fp = float(np.sum((pred == c) & (true != c)))
        fn = float(np.sum((pred != c) & (true == c)))
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
    return float(np.mean(f1s))


def _oos_r2(pred: np.ndarray, true: np.ndarray, base: float) -> float:
    sse = float(np.sum((true - pred) ** 2))
    sst = float(np.sum((true - base) ** 2))
    return 1.0 - sse / sst if sst > 0 else float("nan")


# All vol-informative market features (HAR realized-vol lags + the VIX block
# + the volume ratio). Dropping the lot yields a genuinely WEAK vol baseline
# (returns/rates/calendar only) — to test whether the event-based text lift
# is just compensating for a market baseline that lacks vol features.
_HAR_FEATURES = ("rv_lag_1", "rv_lag_5", "rv_lag_22", "vix", "vix_chg_5", "vol_ratio_30")


def build_event_sequences(
    cache_dir: Path | str,
    emb_parquet: Path | str,
    *,
    seq_len: int = _SEQ_LEN,
    drop_cols: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (seq[N,T,F] market windows, text[N,D], rv[N] log realized vol) for FOMC events."""

    import pandas as pd

    X, Y, dates = build_dataset(cache_dir)
    if drop_cols:
        X = X.drop(columns=list(drop_cols))
    Xv = X.to_numpy(dtype=np.float64)
    rv = np.log(Y["rv_5"].to_numpy(dtype=np.float64) + _EPS)
    date_str = dates.astype(str).str[:10].tolist()

    emb_df = pd.read_parquet(emb_parquet)
    ecols = [c for c in emb_df.columns if c.startswith("emb_")]
    emap = {str(r["date"]): r[ecols].to_numpy(dtype=np.float64) for _, r in emb_df.iterrows()}

    seqs, texts, rvs = [], [], []
    for i, d in enumerate(date_str):
        if d in emap and i >= seq_len - 1:
            seqs.append(Xv[i - seq_len + 1 : i + 1])
            texts.append(emap[d])
            rvs.append(rv[i])
    return (
        np.asarray(seqs, dtype=np.float32),
        np.asarray(texts, dtype=np.float32),
        np.asarray(rvs, dtype=np.float64),
    )


def _train_predict(
    seq_tr: np.ndarray,
    txt_tr: np.ndarray,
    y_tr: np.ndarray,
    seq_te: np.ndarray,
    txt_te: np.ndarray,
    *,
    task: str,
    n_classes: int,
    market_only: bool,
    seed: int,
    epochs: int,
    device: str = "cpu",
) -> np.ndarray:
    import torch
    from torch import nn

    from app.determinism import enable_deterministic_mode
    from app.models.multimodal_forecaster import MultiModalForecasterModel

    enable_deterministic_mode(seed)
    dev = torch.device(device)
    # standardize sequence features (over all train timesteps) + text
    sm = seq_tr.reshape(-1, seq_tr.shape[-1]).mean(0)
    ss = seq_tr.reshape(-1, seq_tr.shape[-1]).std(0)
    ss = np.where(ss > 0, ss, 1.0)
    tm, tsd = txt_tr.mean(0), txt_tr.std(0)
    tsd = np.where(tsd > 0, tsd, 1.0)

    def _t(a: np.ndarray) -> Any:
        return torch.tensor(a, dtype=torch.float32, device=dev)

    Xtr = _t((seq_tr - sm) / ss)
    Xte = _t((seq_te - sm) / ss)
    Ttr = _t((txt_tr - tm) / tsd)
    Tte = _t((txt_te - tm) / tsd)

    def miss(n: int) -> Any:
        return torch.ones((n, 1), device=dev) if market_only else torch.zeros((n, 1), device=dev)

    model = MultiModalForecasterModel(
        architecture="lstm_attn",
        market_input_size=seq_tr.shape[-1],
        text_embedding_dim=txt_tr.shape[-1],
        n_classes=n_classes,
        hidden_size=48,
        num_layers=1,
        dropout=0.1,
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    if task == "classification":
        yb = torch.tensor(y_tr, dtype=torch.long, device=dev)
        loss_fn: Any = nn.CrossEntropyLoss()
    else:
        ym, ysd = float(y_tr.mean()), float(y_tr.std()) or 1.0
        yb = _t((y_tr - ym) / ysd).unsqueeze(-1)
        loss_fn = nn.MSELoss()

    n_val = max(1, len(Xtr) // 5)
    tr, val = slice(0, len(Xtr) - n_val), slice(len(Xtr) - n_val, len(Xtr))
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(
            Xtr[tr], text_embedding=Ttr[tr], text_embedding_missing=miss(tr.stop - tr.start)
        )
        loss = loss_fn(out if task == "classification" else out[:, 0:1], yb[tr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vo = model(
                Xtr[val], text_embedding=Ttr[val], text_embedding_missing=miss(val.stop - val.start)
            )
            vl = float(loss_fn(vo if task == "classification" else vo[:, 0:1], yb[val]))
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 30:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(Xte, text_embedding=Tte, text_embedding_missing=miss(len(Xte)))
        if task == "classification":
            return cast(np.ndarray, out.argmax(1).cpu().numpy())
        ym, ysd = float(y_tr.mean()), float(y_tr.std()) or 1.0
        return cast(np.ndarray, out[:, 0].cpu().numpy() * ysd + ym)


def run_reconcile(
    cache_dir: Path | str,
    emb_parquet: Path | str,
    *,
    seed: int = 11,
    n_folds: int = 5,
    epochs: int = 200,
    weak_market: bool = False,
) -> dict[str, Any]:
    drop = _HAR_FEATURES if weak_market else ()
    seq, txt, rv = build_event_sequences(cache_dir, emb_parquet, drop_cols=drop)
    folds = walk_forward_splits(len(seq), n_folds=n_folds, embargo=2)
    out: dict[str, Any] = {"n_events": len(seq), "classification": {}, "regression": {}}
    pools: dict[str, dict[str, list[float]]] = {
        "cls": {"mkt": [], "txt": [], "true": []},
        "reg": {"mkt": [], "txt": [], "true": [], "base": []},
    }
    for tr_list, te_list in folds:
        tr, te = np.array(tr_list), np.array(te_list)
        thr = _regime_thresholds(rv[tr])
        reg_lab_tr, reg_lab_true = _to_regime(rv[tr], thr), _to_regime(rv[te], thr)
        for mo, key in ((True, "mkt"), (False, "txt")):
            cls = _train_predict(
                seq[tr],
                txt[tr],
                reg_lab_tr,
                seq[te],
                txt[te],
                task="classification",
                n_classes=3,
                market_only=mo,
                seed=seed,
                epochs=epochs,
            )
            pools["cls"][key].extend(cls.tolist())
            reg = _train_predict(
                seq[tr],
                txt[tr],
                rv[tr],
                seq[te],
                txt[te],
                task="regression",
                n_classes=2,
                market_only=mo,
                seed=seed,
                epochs=epochs,
            )
            pools["reg"][key].extend(reg.tolist())
        pools["cls"]["true"].extend(reg_lab_true.tolist())
        pools["reg"]["true"].extend(rv[te].tolist())
        pools["reg"]["base"].extend([float(rv[tr].mean())] * len(te))

    clsp = {k: np.asarray(v) for k, v in pools["cls"].items()}
    regp = {k: np.asarray(v) for k, v in pools["reg"].items()}
    f1_m, f1_t = _macro_f1(clsp["mkt"], clsp["true"]), _macro_f1(clsp["txt"], clsp["true"])
    r2_m = _oos_r2(regp["mkt"], regp["true"], float(regp["base"].mean()))
    r2_t = _oos_r2(regp["txt"], regp["true"], float(regp["base"].mean()))
    # bootstrap CI on the deltas
    rng = np.random.default_rng(seed)
    n = len(clsp["true"])
    df1, dr2 = [], []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        df1.append(
            _macro_f1(clsp["txt"][idx], clsp["true"][idx])
            - _macro_f1(clsp["mkt"][idx], clsp["true"][idx])
        )
        b = float(regp["base"][idx].mean())
        dr2.append(
            _oos_r2(regp["txt"][idx], regp["true"][idx], b)
            - _oos_r2(regp["mkt"][idx], regp["true"][idx], b)
        )
    out["classification"] = {
        "f1_market": f1_m,
        "f1_text": f1_t,
        "delta": f1_t - f1_m,
        "delta_ci90": [float(np.quantile(df1, 0.05)), float(np.quantile(df1, 0.95))],
    }
    out["regression"] = {
        "r2_market": r2_m,
        "r2_text": r2_t,
        "delta": r2_t - r2_m,
        "delta_ci90": [float(np.quantile(dr2, 0.05)), float(np.quantile(dr2, 0.95))],
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile text lift: classification vs regression, lstm_attn."
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--embeddings", nargs="+", required=True, help="name=path pairs")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--weak-market",
        action="store_true",
        help="Drop HAR vol-persistence features (replicate a weak market baseline).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = "WEAK market baseline (no HAR lags)" if args.weak_market else "strong market baseline"
    print(f"=== {tag} ===")
    print(f"{'encoder':<14}{'task':<14}{'market':>8}{'+text':>8}{'delta':>8}{'delta_CI90':>18}")
    allres = {}
    for pair in args.embeddings:
        name, path = pair.split("=", 1)
        r = run_reconcile(args.cache_dir, path, seed=args.seed, weak_market=args.weak_market)
        allres[name] = r
        c, g = r["classification"], r["regression"]
        cci = f"[{c['delta_ci90'][0]:+.3f},{c['delta_ci90'][1]:+.3f}]"
        gci = f"[{g['delta_ci90'][0]:+.3f},{g['delta_ci90'][1]:+.3f}]"
        print(
            f"{name:<14}{'cls macroF1':<14}{c['f1_market']:>8.3f}{c['f1_text']:>8.3f}{c['delta']:>+8.3f}{cci:>18}"
        )
        print(
            f"{name:<14}{'reg R2':<14}{g['r2_market']:>8.3f}{g['r2_text']:>8.3f}{g['delta']:>+8.3f}{gci:>18}"
        )
    (args.out_dir / "reconcile.json").write_text(json.dumps(allres, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
