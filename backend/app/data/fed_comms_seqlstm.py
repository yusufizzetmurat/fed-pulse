"""Late-fusion sequence-LSTM forecaster on the DENSE daily fed-comms fusion design.

Ports the parallel track's recurrent dual-head architecture (which they ran on
event-sampled data) onto our dense daily fusion arrays, so the two tracks are
comparable on identical machinery save for the sampling. An LSTM rolls the last
L=22 days of market features into a market representation; the origin-day text
embedding is projected and gated by its freshness mask (floored: no fresh text →
text path zeroed → pure market path). The two representations are concatenated
into a shared trunk feeding TWO heads — a multi-horizon RV-residual regressor
(stacked on HAR, so HAR is the floor) and a 3-class vol-regime classifier
(train-fold terciles of the per-horizon forward RV). The joint loss is
Huber(residual) + CE(regime).

Reported per horizon, on the REGRESSION head: HAR R², LSTM-market-only R²,
LSTM-fused R², and a text-vs-market block-bootstrap CI; and on the
CLASSIFICATION head: majority-F1, market-only macro-F1, fused macro-F1, and a
text-vs-market macro-F1 block CI. The classification surface is the whole point
— the parallel track's only surviving text residual lived there, so we replicate
both surfaces rather than collapse to R² alone.

Text isolation is the same trained weights run twice: once with the text input
present (fused) and once with it zeroed (market-only). The gap is purely the text
contribution; a CI clearing zero on either surface is the result that would
overturn the text-is-null finding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from app.data.dense_daily_dataset import walk_forward_splits
from app.data.dense_forecast_train import _fit_predict_ols, _oos_r2
from app.data.fed_comms_dataset import DEFAULT_HORIZONS, MEASURES
from app.data.fed_comms_regime import _block_f1_gap_ci, _labels, _macro_f1
from app.data.fed_comms_train import (
    DEFAULT_FUSION_DIR,
    _assemble,
    _block_bootstrap_r2_ci,
)

_SEQ_LEN = 22  # market-sequence window length (one trading month)
_N_CLASSES = 3


def build_sequences(
    market_feat: np.ndarray,
    text_emb: np.ndarray,
    text_mask: np.ndarray,
    valid: np.ndarray,
    *,
    seq_len: int = _SEQ_LEN,
) -> dict[str, np.ndarray]:
    """Stack a leak-safe L×d_market window per valid origin day t (late text fusion).

    For each origin index t the market sequence is rows ``[t-L+1 .. t]`` of
    ``market_feat`` — strictly rows ≤ t, so it never sees the future. The text
    inputs are the origin-day embedding and mask (both known by close of t). The
    forward targets are NOT touched here; they stay the caller's ``targets``
    (rows t+1..t+h), so the only leak surface — the sequence — is closed by
    construction.

    Origins with fewer than ``seq_len`` prior rows (t < L-1) or flagged invalid
    are dropped. Returns the kept origin indices alongside the stacked arrays so
    the caller can align targets/HAR/dates by ``origin``.
    """

    n = market_feat.shape[0]
    origins = np.array(
        [t for t in range(n) if t >= seq_len - 1 and bool(valid[t])], dtype=np.int64
    )
    seqs = np.stack([market_feat[t - seq_len + 1 : t + 1] for t in origins], axis=0)
    return {
        "origin": origins,
        "seq": seqs,  # (M, L, d_market)
        "text_emb": text_emb[origins],  # (M, d_text)
        "text_mask": text_mask[origins],  # (M,)
    }


def _standardize_seq(
    seq_tr: np.ndarray, seq_te: np.ndarray, fit: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize L×d sequences per feature using stats from `fit` (train-core rows)."""

    flat = fit.reshape(-1, fit.shape[-1])
    m, s = flat.mean(0), flat.std(0)
    s = np.where(s > 0, s, 1.0)
    return (seq_tr - m) / s, (seq_te - m) / s


def _build_model(d_market: int, d_text: int, n_horizons: int) -> Any:
    """Construct the late-fusion dual-head LSTM (torch imported lazily)."""

    import torch
    from torch import nn

    class _SeqLstmFusion(nn.Module):
        """LSTM(market seq) ⊕ gated text → shared trunk → (RV regressor, regime clf)."""

        def __init__(self) -> None:
            super().__init__()
            self.d_hidden = 128
            self.lstm = nn.LSTM(
                d_market, self.d_hidden, num_layers=1, batch_first=True
            )
            self.text_proj = nn.Sequential(
                nn.Linear(d_text, self.d_hidden), nn.GELU(), nn.LayerNorm(self.d_hidden)
            )
            self.trunk = nn.Sequential(
                nn.Linear(2 * self.d_hidden, self.d_hidden),
                nn.GELU(),
                nn.LayerNorm(self.d_hidden),
            )
            self.reg_head = nn.Linear(self.d_hidden, n_horizons)
            self.clf_head = nn.Linear(self.d_hidden, _N_CLASSES)

        def forward(
            self, seq: torch.Tensor, text_emb: torch.Tensor, text_mask: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            _, (h_n, _) = self.lstm(seq)
            market_rep = h_n[-1]  # (B, d_hidden) — final hidden state
            mask = text_mask.float().unsqueeze(1)
            text_rep = self.text_proj(text_emb) * mask  # FLOOR: no fresh text → zeroed
            fused = self.trunk(torch.cat([market_rep, text_rep], dim=1))
            return {"reg": self.reg_head(fused), "clf": self.clf_head(fused)}

    return _SeqLstmFusion()


def _train_seqlstm_fold(
    data: dict[str, np.ndarray],
    seqs: dict[str, np.ndarray],
    tr: np.ndarray,
    te: np.ndarray,
    *,
    seed: int,
    epochs: int,
    patience: int = 12,
) -> dict[str, np.ndarray]:
    """Train one walk-forward fold; return per-horizon RV preds + regime preds.

    `tr`/`te` index into the sequence arrays (origins). The RV head predicts the
    HAR residual (standardized) and HAR OLS is added back, flooring at HAR. The
    regime label is the train-fold tercile of each horizon's forward RV. Early
    stopping watches a time-ordered val split on the joint loss.
    """

    import torch
    from torch import nn

    from app.determinism import enable_deterministic_mode

    enable_deterministic_mode(seed)
    origin = seqs["origin"]
    har_all, tgt_all = data["har"], data["targets"]
    har_tr, har_te = har_all[origin[tr]], har_all[origin[te]]
    tgt_tr, tgt_te = tgt_all[origin[tr]], tgt_all[origin[te]]
    n_h = tgt_tr.shape[1]
    n = len(tr)
    n_val = max(1, n // 5)
    core, val = slice(0, n - n_val), slice(n - n_val, n)

    # HAR OLS floor. The test baseline is fit on the full train split (standard,
    # no test leak); the residual TRAINING signal uses a core-only fit so the
    # early-stopping val residuals are not self-imputed by val-period targets.
    har_te_pred = np.column_stack(
        [_fit_predict_ols(har_tr, tgt_tr[:, k], har_te) for k in range(n_h)]
    )
    har_core_pred = np.column_stack(
        [_fit_predict_ols(har_tr[core], tgt_tr[core, k], har_tr) for k in range(n_h)]
    )
    resid = tgt_tr - har_core_pred

    # tercile regime labels per horizon, thresholds from TRAIN origins only
    thr = [np.quantile(tgt_tr[:, k], [1 / 3, 2 / 3]) for k in range(n_h)]
    y_tr = np.column_stack([_labels(tgt_tr[:, k], thr[k]) for k in range(n_h)])
    y_te = np.column_stack([_labels(tgt_te[:, k], thr[k]) for k in range(n_h)])
    maj = np.array([int(np.bincount(y_tr[:, k], minlength=_N_CLASSES).argmax()) for k in range(n_h)])

    seq_tr_raw, seq_te_raw = seqs["seq"][tr], seqs["seq"][te]
    seq_tr, seq_te = _standardize_seq(seq_tr_raw, seq_te_raw, seq_tr_raw[core])
    emb_tr_raw, emb_te_raw = seqs["text_emb"][tr], seqs["text_emb"][te]
    mask_tr, mask_te = seqs["text_mask"][tr], seqs["text_mask"][te]
    present = mask_tr[core] > 0
    ref = emb_tr_raw[core][present] if present.any() else emb_tr_raw[core]
    em, es = ref.mean(0), ref.std(0)
    es = np.where(es > 0, es, 1.0)
    emb_tr = ((emb_tr_raw - em) / es) * mask_tr[:, None]
    emb_te = ((emb_te_raw - em) / es) * mask_te[:, None]
    ym, ys = resid[core].mean(0), resid[core].std(0)
    ys = np.where(ys > 0, ys, 1.0)
    resid_std = (resid - ym) / ys

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(seq_tr.shape[-1], emb_tr.shape[1], n_h).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def tt(a: np.ndarray, *, long: bool = False) -> "torch.Tensor":
        return torch.tensor(a, dtype=torch.long if long else torch.float32, device=dev)

    seq_t, emb_t, mask_t = tt(seq_tr), tt(emb_tr), tt(mask_tr)
    y_reg_t, y_clf_t = tt(resid_std), tt(y_tr[:, n_h - 1], long=True)
    core_pos = np.arange(0, n - n_val)
    rng = np.random.default_rng(seed)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(core_pos))
        for s in range(0, len(order), 256):
            b = core_pos[order[s : s + 256]]
            opt.zero_grad()
            out = model(seq_t[b], emb_t[b], mask_t[b])
            huber = nn.functional.huber_loss(out["reg"], y_reg_t[b])
            ce = nn.functional.cross_entropy(out["clf"], y_clf_t[b])
            loss = huber + ce
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
        model.eval()
        with torch.no_grad():
            vo = model(seq_t[val], emb_t[val], mask_t[val])
            vloss = float(
                nn.functional.huber_loss(vo["reg"], y_reg_t[val])
                + nn.functional.cross_entropy(vo["clf"], y_clf_t[val])
            )
        if vloss < best - 1e-6:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        seq_e, emb_e, msk_e = tt(seq_te), tt(emb_te), tt(mask_te)
        fused = model(seq_e, emb_e, msk_e)
        mkt = model(seq_e, emb_e, torch.zeros_like(msk_e))  # text zeroed
    fused_reg = har_te_pred + (cast(np.ndarray, fused["reg"].cpu().numpy()) * ys + ym)
    mkt_reg = har_te_pred + (cast(np.ndarray, mkt["reg"].cpu().numpy()) * ys + ym)
    return {
        "fused_reg": fused_reg,
        "mkt_reg": mkt_reg,
        "har_reg": har_te_pred,
        "true_reg": tgt_te,
        "fused_clf": cast(np.ndarray, fused["clf"].cpu().numpy()).argmax(1),
        "mkt_clf": cast(np.ndarray, mkt["clf"].cpu().numpy()).argmax(1),
        "true_clf": y_te[:, n_h - 1],
        "maj_clf": np.full(len(te), maj[n_h - 1], dtype=np.int64),
        "mask": mask_te,
    }


def run(
    fusion_dir: Path | str = DEFAULT_FUSION_DIR,
    *,
    market_cache_dir: Path | str,
    corpus_path: Path | str,
    emb_path: Path | str,
    seed: int = 11,
    epochs: int = 80,
    n_folds: int = 5,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    measure: str = "rv",
) -> dict[str, Any]:
    import pandas as pd

    daily = pd.read_parquet(Path(fusion_dir) / "daily_fusion.parquet")
    corpus = pd.read_parquet(corpus_path)
    emb_df = pd.read_parquet(emb_path)
    data = _assemble(daily, corpus, emb_df, market_cache_dir, horizons, measure=measure)
    seqs = build_sequences(
        data["market_feat"], data["text_emb"], data["text_mask"], data["valid"]
    )

    n_seq = len(seqs["origin"])
    folds = walk_forward_splits(n_seq, n_folds=n_folds, embargo=max(horizons) + 1)
    pools: dict[str, list[np.ndarray]] = {
        k: []
        for k in (
            "fused_reg", "mkt_reg", "har_reg", "true_reg",
            "fused_clf", "mkt_clf", "true_clf", "maj_clf", "mask",
        )
    }
    for tr_l, te_l in folds:
        out = _train_seqlstm_fold(
            data, seqs, np.array(tr_l), np.array(te_l), seed=seed, epochs=epochs
        )
        for key in pools:
            pools[key].append(out[key])
    cat = {key: np.concatenate(v) for key, v in pools.items()}

    fused = cat["fused_reg"]
    mkt = cat["mkt_reg"]
    har = cat["har_reg"]
    true = cat["true_reg"]
    base = np.tile(true.mean(0), (true.shape[0], 1))

    results: dict[str, Any] = {
        "n_eval": int(true.shape[0]),
        "measure": measure,
        "seq_len": _SEQ_LEN,
        "by_horizon": {},
    }
    for k, h in enumerate(horizons):
        row: dict[str, Any] = {
            "har_r2": _oos_r2(har[:, k], true[:, k], base[:, k]),
            "mkt_only_r2": _oos_r2(mkt[:, k], true[:, k], base[:, k]),
            "fused_r2": _oos_r2(fused[:, k], true[:, k], base[:, k]),
            "text_vs_mkt_block_ci90": list(
                _block_bootstrap_r2_ci(
                    fused[:, k], true[:, k], mkt[:, k], block=max(h, 1), seed=seed
                )
            ),
        }
        results["by_horizon"][f"h{h}"] = row

    # classification surface: the per-horizon regime head is trained on the
    # longest horizon's terciles (the parallel track's dual head); report its
    # majority floor, market-only, fused macro-F1 and a text-vs-market F1 block CI.
    block = max(horizons)
    results["regime"] = {
        "trained_on": f"h{max(horizons)}",
        "majority_f1": _macro_f1(cat["true_clf"], cat["maj_clf"]),
        "mkt_only_f1": _macro_f1(cat["true_clf"], cat["mkt_clf"]),
        "fused_f1": _macro_f1(cat["true_clf"], cat["fused_clf"]),
        "text_vs_mkt_f1_block_ci90": list(
            _block_f1_gap_ci(
                cat["true_clf"], cat["fused_clf"], cat["mkt_clf"], block=block, seed=seed
            )
        ),
    }
    active = cat["mask"] > 0
    if active.sum() > 5:
        results["regime"]["fused_f1_active"] = _macro_f1(
            cat["true_clf"][active], cat["fused_clf"][active]
        )
        results["regime"]["mkt_f1_active"] = _macro_f1(
            cat["true_clf"][active], cat["mkt_clf"][active]
        )
    results["text_active_frac"] = float(active.mean())
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Late-fusion sequence-LSTM dual-head forecaster (dense fed-comms)."
    )
    parser.add_argument("--fusion-dir", type=Path, default=DEFAULT_FUSION_DIR)
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--emb-path", type=Path, required=True)
    parser.add_argument("--market-cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target", default="rv", choices=list(MEASURES))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    res = run(
        args.fusion_dir,
        market_cache_dir=args.market_cache_dir,
        corpus_path=args.corpus_path,
        emb_path=args.emb_path,
        seed=args.seed,
        epochs=args.epochs,
        measure=args.target,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "seqlstm_bakeoff.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8"
    )
    print(
        f"target={res['measure']}  n_eval={res['n_eval']}  L={res['seq_len']}  "
        f"text_active_frac={res['text_active_frac']:.3f}"
    )
    print(f"{'horizon':<8}{'HAR':>8}{'mkt':>8}{'fused':>8}{'txt-vs-mkt_R2_block':>24}")
    for hk, r in res["by_horizon"].items():
        c = r["text_vs_mkt_block_ci90"]
        print(
            f"{hk:<8}{r['har_r2']:>8.3f}{r['mkt_only_r2']:>8.3f}{r['fused_r2']:>8.3f}"
            f"{f'[{c[0]:+.3f},{c[1]:+.3f}]':>24}"
        )
    g = res["regime"]
    cf = g["text_vs_mkt_f1_block_ci90"]
    print(f"regime ({g['trained_on']} terciles, macro-F1; floor ≈ 0.33):")
    print(
        f"{'  major':<8}{g['majority_f1']:>8.3f}{'  mkt':<6}{g['mkt_only_f1']:>8.3f}"
        f"{'  fused':<8}{g['fused_f1']:>8.3f}"
        f"  txt-vs-mkt_F1_block=[{cf[0]:+.3f},{cf[1]:+.3f}]"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
