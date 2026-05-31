"""Intraday RV architecture sweep — does a TCN or a leverage-LSTM beat HAR by
more than the QLIKE-trained MLP (DLq)?

The bake-off (`intraday_rv_forecast`) already showed a QLIKE-trained residual MLP
(DLq) edging HAR on the daily realized-measure series. This module asks the next
question: does *sequence* structure over the last L=22 trading days buy anything
the flat MLP cannot — either through a causal convolution stack (TCN) or through
a recurrent cell with an explicit leverage-effect inductive bias.

Both contenders consume the same per-day `full` feature matrix the bake-off
builds (HAR daily/weekly/monthly lags + rs_pos, rs_neg, bv, rq, rskew, rkurt,
parkinson, log rvol), windowed into leak-safe L-day sequences (window ending at
t uses rows ≤ t only). The target is forward log-RV at h ∈ {1,5,22}, residual-
stacked on HAR exactly as DLq: HAR is fit on the train split, the net learns the
log-residual, HAR is added back at test, and the QLIKE loss is computed on the
reconstructed variance σ² = exp(HAR + residual) — the econometric vol loss, not
MSE on the log-residual.

  - TCN          : a small causal Temporal Convolutional Network — stacked
                   dilated causal conv1d (dilations 1,2,4) with residual
                   connections, last-step pooled into a residual head.
  - σLSTM        : a leverage-LSTM. The signed downside/upside semivariance
                   asymmetry (rs_neg vs rs_pos) is split into an explicit signed
                   channel and a learned gate weights downside vs upside before
                   the LSTM rolls the window — a pragmatic leverage cell that
                   lets a volatility spike driven by losses register differently
                   from one driven by gains.

Reported per horizon: QLIKE for HAR / TCN / σLSTM (lower better), OOS-R² in
log-RV space, and a moving-block bootstrap QLIKE-gain CI (block = horizon) for
TCN-vs-HAR and σLSTM-vs-HAR. The DLq numbers from the bake-off are the reference
bar both sequence models are trying to clear by a wider margin.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from app.data.dense_daily_dataset import walk_forward_splits
from app.data.dense_forecast_train import _fit_predict_ols, _oos_r2
from app.data.intraday_rv_forecast import (
    _bootstrap_qlike_gain_ci,
    _EPS,
    _forward_log_rv,
    _har_lags,
    _LOGV_CLAMP,
    _qlike,
)

_SEQ_LEN = 22  # sequence window length (one trading month), matching the seq-LSTM track

# Column layout of the per-day `full` matrix (see `_feature_matrix`). The σLSTM
# leverage gate needs to find rs_pos / rs_neg by position, so the layout is fixed.
_RS_POS_COL = 3
_RS_NEG_COL = 4

# Bound the model's deviation from the HAR floor (in log-RV units) at both train
# and inference, so an exp-loss can't blow up on an early/divergent prediction.
# True residuals are well under 1; ±6 is generous yet prevents var=exp(−80)≈0.
_RESID_CLAMP = 6.0


def _feature_matrix(rv_path: Path | str) -> dict[str, np.ndarray]:
    """Build the bake-off's `full` per-day feature matrix + HAR lags + targets.

    Columns of `full`: [HAR daily, HAR weekly, HAR monthly, rs_pos, rs_neg, bv,
    rq, rskew, rkurt, parkinson, log(rvol)] — identical to the matrix the DLq
    contender trains on, so the architecture comparison is feature-for-feature.
    """

    import pandas as pd

    df = pd.read_parquet(rv_path).sort_values("date").reset_index(drop=True)
    rv = df["rv"].to_numpy(dtype=np.float64)
    log_rv = np.log(rv + _EPS)
    har = _har_lags(log_rv)
    feat_cols = ["rs_pos", "rs_neg", "bv", "rq", "rskew", "rkurt", "parkinson"]
    extra = np.column_stack([df[c].to_numpy(dtype=np.float64) for c in feat_cols])
    extra = np.column_stack([extra, np.log(df["rvol"].to_numpy(dtype=np.float64) + 1.0)])
    full = np.column_stack([har, extra])
    return {"rv": rv, "har": har, "full": full, "dates": df["date"].astype(str).to_numpy()}


def build_sequences(
    feat: np.ndarray, valid: np.ndarray, *, seq_len: int = _SEQ_LEN
) -> dict[str, np.ndarray]:
    """Stack a leak-safe L×d window per valid origin day t.

    For each origin index t the window is rows ``[t-L+1 .. t]`` of ``feat`` —
    strictly rows ≤ t, so it never sees the future. Forward targets are NOT
    touched here (they stay rows t+1..t+h in the caller), so the only leak
    surface — the window — is closed by construction. Origins with fewer than
    ``seq_len`` prior rows (t < L-1) or flagged invalid are dropped. Returns the
    kept origin indices so the caller aligns targets / HAR / dates by ``origin``.
    """

    n = feat.shape[0]
    origins = np.array([t for t in range(n) if t >= seq_len - 1 and bool(valid[t])], dtype=np.int64)
    seqs = np.stack([feat[t - seq_len + 1 : t + 1] for t in origins], axis=0)
    return {"origin": origins, "seq": seqs}  # seq: (M, L, d)


def _standardize_seq(
    seq_tr: np.ndarray, seq_te: np.ndarray, fit: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize L×d windows per feature using stats from `fit` (train-core rows)."""

    flat = fit.reshape(-1, fit.shape[-1])
    m, s = flat.mean(0), flat.std(0)
    s = np.where(s > 0, s, 1.0)
    return (seq_tr - m) / s, (seq_te - m) / s


def _build_tcn(d_in: int) -> Any:
    """Small causal TCN: dilated causal conv1d stack (dilations 1,2,4) → residual.

    Each block left-pads by (kernel-1)*dilation and trims the right tail, so the
    output at step t depends only on inputs ≤ t (strict causality — verified in
    the unit test). Two residual blocks keep parameter count tiny for ~5k rows.
    """

    import torch
    from torch import nn

    class _CausalConv1d(nn.Module):
        def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int) -> None:
            super().__init__()
            self.pad = (kernel - 1) * dilation
            self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, L)
            x = nn.functional.pad(x, (self.pad, 0))  # left-pad only → causal
            return cast(torch.Tensor, self.conv(x))

    class _TCNBlock(nn.Module):
        def __init__(self, c_in: int, c_out: int, dilation: int) -> None:
            super().__init__()
            self.conv1 = _CausalConv1d(c_in, c_out, 3, dilation)
            self.conv2 = _CausalConv1d(c_out, c_out, 3, dilation)
            self.act = nn.GELU()
            self.drop = nn.Dropout(0.1)
            self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.drop(self.act(self.conv1(x)))
            y = self.drop(self.act(self.conv2(y)))
            res = x if self.down is None else self.down(x)
            return cast(torch.Tensor, y + res)

    class _TCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            ch = 32
            self.blocks = nn.ModuleList(
                [
                    _TCNBlock(d_in, ch, dilation=1),
                    _TCNBlock(ch, ch, dilation=2),
                    _TCNBlock(ch, ch, dilation=4),
                ]
            )
            self.head = nn.Sequential(nn.Linear(ch, ch), nn.GELU(), nn.Linear(ch, 1))

        def forward(self, seq: torch.Tensor) -> torch.Tensor:  # seq: (B, L, d)
            x = seq.transpose(1, 2)  # (B, d, L)
            for blk in self.blocks:
                x = blk(x)
            last = x[:, :, -1]  # last-step pooling (output at origin t)
            return cast(torch.Tensor, self.head(last))

    return _TCN()


def _build_sigma_lstm(d_in: int) -> Any:
    """Leverage-LSTM: a learned gate weights downside vs upside before the LSTM.

    The leverage inductive bias: an extra signed channel rs_neg − rs_pos is
    appended (negative when the day's variance is upside-driven, positive when
    downside-driven), and a per-step scalar gate g = σ(w·[rs_pos, rs_neg] + b)
    multiplies the rs_neg channel. This lets a downside-driven volatility burst
    enter the recurrence with a different effective weight from an upside one — a
    pragmatic leverage cell that gates the semivariance channels rather than
    rewiring the LSTM's internal gates.
    """

    import torch
    from torch import nn

    class _SigmaLstm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.d_hidden = 64
            # gate reads the two semivariance channels → scalar downside weight
            self.lev_gate = nn.Linear(2, 1)
            # input = original d features + signed (rs_neg − rs_pos) leverage channel
            self.lstm = nn.LSTM(d_in + 1, self.d_hidden, num_layers=1, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(self.d_hidden, self.d_hidden), nn.GELU(), nn.Linear(self.d_hidden, 1)
            )

        def forward(self, seq: torch.Tensor) -> torch.Tensor:  # seq: (B, L, d)
            rs_pos = seq[:, :, _RS_POS_COL : _RS_POS_COL + 1]
            rs_neg = seq[:, :, _RS_NEG_COL : _RS_NEG_COL + 1]
            gate = torch.sigmoid(self.lev_gate(torch.cat([rs_pos, rs_neg], dim=-1)))
            signed = gate * rs_neg - (1.0 - gate) * rs_pos  # gated downside-minus-upside
            x = torch.cat([seq, signed], dim=-1)
            _, (h_n, _) = self.lstm(x)
            return cast(torch.Tensor, self.head(h_n[-1]))

    return _SigmaLstm()


def _train_arch_fold(
    arch: str,
    seq_tr: np.ndarray,
    seq_te: np.ndarray,
    har_tr: np.ndarray,
    har_te: np.ndarray,
    tgt_tr: np.ndarray,
    *,
    seed: int,
    epochs: int,
    device: str,
) -> np.ndarray:
    """Train one walk-forward fold of a sequence arch; QLIKE-trained, HAR-stacked.

    Mirrors `_train_fold_qlike`: HAR is fit on the train split (core-only for the
    residual target so the early-stopping val residuals are not self-imputed),
    the net learns the standardized log-residual, and the loss is QLIKE on the
    reconstructed variance σ² = exp(HAR_pred + residual). Returns test predictions
    in log-RV space (HAR_te + residual), so the caller scores it identically to
    HAR. Grad-clipped (exp-loss → bound grads), deterministic, early-stopped.
    """

    import torch

    from app.determinism import enable_deterministic_mode

    enable_deterministic_mode(seed)
    dev = torch.device(device)
    n = len(seq_tr)
    n_val = max(1, n // 5)
    core, val = slice(0, n - n_val), slice(n - n_val, n)

    # HAR floor: test prediction fit on the full train split (no leak); the
    # residual TRAINING target uses a core-only HAR fit so val residuals are honest.
    har_te_pred = _fit_predict_ols(har_tr, tgt_tr, har_te)
    har_core_te = _fit_predict_ols(
        har_tr[core], tgt_tr[core], har_tr
    )  # in-sample on all train rows
    resid = (tgt_tr - har_core_te).reshape(-1, 1)

    seq_tr_s, seq_te_s = _standardize_seq(seq_tr, seq_te, seq_tr[core])
    rm, rs = resid[core].mean(), resid[core].std()
    rs = rs if rs > 0 else 1.0

    xtr = torch.tensor(seq_tr_s, dtype=torch.float32, device=dev)
    xte = torch.tensor(seq_te_s, dtype=torch.float32, device=dev)
    har_t = torch.tensor(har_core_te, dtype=torch.float32, device=dev).reshape(-1, 1)
    log_true_t = torch.tensor(tgt_tr.reshape(-1, 1), dtype=torch.float32, device=dev)

    def qlike_loss(resid_std: torch.Tensor, har: torch.Tensor, log_true: torch.Tensor) -> Any:
        resid = torch.clamp(resid_std * rs + rm, -_RESID_CLAMP, _RESID_CLAMP)
        log_pred = torch.clamp(har + resid, -_LOGV_CLAMP, _LOGV_CLAMP)
        log_true_c = torch.clamp(log_true, -_LOGV_CLAMP, _LOGV_CLAMP)
        var_pred = torch.exp(log_pred) + _EPS
        var_true = torch.exp(log_true_c)
        ratio = var_true / var_pred
        return torch.mean(ratio - torch.log(ratio) - 1.0)

    d_in = seq_tr.shape[-1]
    model = (_build_tcn(d_in) if arch == "tcn" else _build_sigma_lstm(d_in)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = float("inf"), None, 0
    rng = np.random.default_rng(seed)
    core_pos = np.arange(0, n - n_val)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(core_pos))
        for s in range(0, len(order), 256):
            b = core_pos[order[s : s + 256]]
            opt.zero_grad()
            loss = qlike_loss(model(xtr[b]), har_t[b], log_true_t[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # exp-loss → bound grads
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(qlike_loss(model(xtr[val]), har_t[val], log_true_t[val]))
        if vloss < best - 1e-6:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 25:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        resid_std = model(xte).cpu().numpy()
    resid_pred = np.clip(resid_std[:, 0] * rs + rm, -_RESID_CLAMP, _RESID_CLAMP)
    return cast(np.ndarray, har_te_pred + resid_pred)


def run(
    rv_path: Path | str,
    *,
    seed: int = 11,
    epochs: int = 120,
    n_folds: int = 5,
    horizons: tuple[int, ...] = (1, 5, 22),
    device: str = "cpu",
) -> dict[str, Any]:
    data = _feature_matrix(rv_path)
    rv, har, full = data["rv"], data["har"], data["full"]
    valid = np.ones(len(rv), dtype=bool)
    seqs = build_sequences(full, valid)
    origin = seqs["origin"]
    seq_all = seqs["seq"]

    results: dict[str, Any] = {"n_days": int(len(rv)), "seq_len": _SEQ_LEN, "by_horizon": {}}
    for h in horizons:
        y = _forward_log_rv(rv, h)
        # an origin is scorable only if its forward target window stays in range
        ok = ~np.isnan(y[origin])
        idx = np.where(ok)[0]
        folds = walk_forward_splits(len(idx), n_folds=n_folds, embargo=h + 1)
        pools: dict[str, list[float]] = {
            k: [] for k in ("har", "tcn", "sigma_lstm", "true", "base")
        }
        for tr_l, te_l in folds:
            tr, te = idx[np.array(tr_l)], idx[np.array(te_l)]
            o_tr, o_te = origin[tr], origin[te]
            ytr, yte = y[o_tr], y[o_te]
            har_pred = _fit_predict_ols(har[o_tr], ytr, har[o_te])
            pools["har"].extend(har_pred.tolist())
            pools["base"].extend([float(ytr.mean())] * len(te))  # per-fold train mean, no leak
            for arch, key in (("tcn", "tcn"), ("sigma_lstm", "sigma_lstm")):
                pred = _train_arch_fold(
                    arch,
                    seq_all[tr],
                    seq_all[te],
                    har[o_tr],
                    har[o_te],
                    ytr,
                    seed=seed,
                    epochs=epochs,
                    device=device,
                )
                pools[key].extend(pred.tolist())
            pools["true"].extend(yte.tolist())
        p = {k: np.asarray(v) for k, v in pools.items()}
        base = p["base"]  # per-fold train mean (no cross-fold leakage into the R² baseline)
        row: dict[str, Any] = {
            "qlike": {k: _qlike(p[k], p["true"]) for k in ("har", "tcn", "sigma_lstm")},
            "r2": {k: _oos_r2(p[k], p["true"], base) for k in ("har", "tcn", "sigma_lstm")},
            "tcn_vs_har_qlike_ci90": _bootstrap_qlike_gain_ci(
                p["tcn"], p["har"], p["true"], seed=seed, block=max(h, 1)
            ),
            "sigma_lstm_vs_har_qlike_ci90": _bootstrap_qlike_gain_ci(
                p["sigma_lstm"], p["har"], p["true"], seed=seed, block=max(h, 1)
            ),
        }
        results["by_horizon"][f"h{h}"] = row
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Intraday RV architecture sweep: HAR vs TCN vs leverage-LSTM (QLIKE-trained)."
    )
    parser.add_argument("--rv-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    res = run(args.rv_path, seed=args.seed, epochs=args.epochs, device=args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "intraday_rv_arch.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"n_days={res['n_days']}  L={res['seq_len']}")
    print("QLIKE (lower better) + block-bootstrap QLIKE-gain CI90 vs HAR (>0 ⇒ beats HAR)")
    head = (
        f"{'horizon':<8}{'HAR':>10}{'TCN':>10}{'sLSTM':>10}"
        f"{'TCNgain_CI90':>22}{'sLSTMgain_CI90':>22}"
    )
    print(head)
    for hk, r in res["by_horizon"].items():
        q = r["qlike"]
        gt = r["tcn_vs_har_qlike_ci90"]
        gl = r["sigma_lstm_vs_har_qlike_ci90"]
        print(
            f"{hk:<8}{q['har']:>10.4f}{q['tcn']:>10.4f}{q['sigma_lstm']:>10.4f}"
            f"{f'[{gt[0]:+.4f},{gt[1]:+.4f}]':>22}{f'[{gl[0]:+.4f},{gl[1]:+.4f}]':>22}"
        )
    print("OOS-R2 (log-RV space, vs mean, higher better)")
    print(f"{'horizon':<8}{'HAR':>10}{'TCN':>10}{'sLSTM':>10}")
    for hk, r in res["by_horizon"].items():
        r2 = r["r2"]
        print(f"{hk:<8}{r2['har']:>10.3f}{r2['tcn']:>10.3f}{r2['sigma_lstm']:>10.3f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
