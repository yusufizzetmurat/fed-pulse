"""LoRA fine-tune test for the late-fusion rebuild.

The frozen-embedding result is weak and hyperparameter-fragile. This module asks
the decisive question for fault class #3: if the FinBERT-fed encoder is unfrozen
(LoRA adapters trained end-to-end inside the fusion model, so text gradients
actually reach the encoder), does a ROBUST direction signal emerge?

It fine-tunes per walk-forward fold (train only), compares LoRA-text+market vs
market-only with a paired McNemar test, on the daily frame (where the frozen
signal concentrated). Texts are truncated to the first 512 tokens for the
fine-tune (chunking a trainable encoder per doc is prohibitively expensive); this
is noted as a simplification. Requires a GPU in practice.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from app.config import DATA_DIR
from app.data.late_fusion_embed import MAX_TOKENS, load_encoder
from app.data.late_fusion_experiment import (
    _SPECS,
    _acc,
    _mcnemar,
    _standardize,
    walk_forward_splits,
)

logger = logging.getLogger(__name__)
_BASE = DATA_DIR / "processed" / "late_fusion"


@dataclass
class LoraConfigArgs:
    n_folds: int = 5
    embargo: int = 5
    seeds: tuple[int, ...] = (11, 22)
    epochs: int = 4
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 1e-3


class LoraFusion(nn.Module):
    def __init__(
        self, encoder: nn.Module | None, struct_dim: int,
        text_latent: int = 16, struct_latent: int = 16, trunk_dim: int = 32, dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.use_text = encoder is not None
        self.encoder = encoder
        fused = struct_latent
        if self.use_text:
            self.text_proj = nn.Linear(768, text_latent)
            fused += text_latent
        self.struct_branch = nn.Sequential(
            nn.Linear(struct_dim, struct_latent), nn.GELU(), nn.Dropout(dropout)
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(fused), nn.Linear(fused, trunk_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.dir_head = nn.Linear(trunk_dim, 1)

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        assert self.encoder is not None
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = torch.as_tensor(out.last_hidden_state)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled  # type: ignore[no-any-return]

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, struct: torch.Tensor
    ) -> torch.Tensor:
        parts = []
        if self.use_text:
            parts.append(self.text_proj(self._encode(input_ids, attention_mask)))
        parts.append(self.struct_branch(struct))
        hidden = self.trunk(torch.cat(parts, dim=-1))
        return self.dir_head(hidden).squeeze(-1)


def _lora_encoder(base: nn.Module) -> nn.Module:
    cfg = LoraConfig(
        r=8, lora_alpha=16, target_modules=["query", "value"],
        lora_dropout=0.1, bias="none", task_type="FEATURE_EXTRACTION",
    )
    return get_peft_model(base, cfg)  # type: ignore[arg-type]


def _train_predict_dir(
    ids_tr: torch.Tensor, mask_tr: torch.Tensor, struct_tr: torch.Tensor, y_tr: torch.Tensor,
    ids_te: torch.Tensor, mask_te: torch.Tensor, struct_te: torch.Tensor,
    base_encoder: nn.Module | None, device: torch.device, cfg: LoraConfigArgs, seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    encoder = _lora_encoder(base_encoder) if base_encoder is not None else None
    model = LoraFusion(encoder, struct_dim=struct_tr.shape[1]).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    n = len(y_tr)
    model.train()
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            opt.zero_grad()
            logit = model(ids_tr[idx].to(device), mask_tr[idx].to(device), struct_tr[idx].to(device))
            loss = bce(logit, y_tr[idx].to(device))
            loss.backward()
            opt.step()

    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(struct_te), cfg.batch_size):
            sl = slice(start, start + cfg.batch_size)
            logit = model(ids_te[sl].to(device), mask_te[sl].to(device), struct_te[sl].to(device))
            probs.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(probs)


def run(frame: str, cfg: LoraConfigArgs) -> dict[str, object]:
    spec = _SPECS[frame]
    enc = load_encoder()  # asserting loader: real FinBERT-fed, never distilbert
    frame_df = pd.read_parquet(_BASE / spec.frame)
    if "pre_volume" in frame_df.columns:
        frame_df["log_pre_volume"] = np.log1p(frame_df["pre_volume"].fillna(0.0))
    struct_names = [c for c in spec.market_features if c in frame_df.columns]
    struct_names += [c for c in spec.sep_features if c in frame_df.columns]

    texts = frame_df["text"].astype(str).tolist()
    tok = enc.tokenizer(
        texts, max_length=MAX_TOKENS, truncation=True, padding="max_length", return_tensors="pt"
    )
    ids_all, mask_all = tok["input_ids"], tok["attention_mask"]
    struct_all = frame_df[struct_names].to_numpy(dtype=np.float32)
    y_all = frame_df[spec.dir_col].to_numpy(dtype=np.float32)
    ok = ~np.isnan(y_all)
    ids_all, mask_all, struct_all, y_all = ids_all[ok], mask_all[ok], struct_all[ok], y_all[ok]

    splits = walk_forward_splits(len(y_all), cfg.n_folds, cfg.embargo)
    oos_y, oos_full, oos_market = [], [], []
    for train_idx, test_idx in splits:
        s_tr, s_te = _standardize(struct_all[train_idx], struct_all[test_idx])
        st_tr, st_te = torch.from_numpy(s_tr), torch.from_numpy(s_te)
        yt = torch.from_numpy(y_all[train_idx])
        full_seeds, market_seeds = [], []
        for seed in cfg.seeds:
            full_seeds.append(_train_predict_dir(
                ids_all[train_idx], mask_all[train_idx], st_tr, yt,
                ids_all[test_idx], mask_all[test_idx], st_te, enc.model, enc.device, cfg, seed,
            ))
            market_seeds.append(_train_predict_dir(
                ids_all[train_idx], mask_all[train_idx], st_tr, yt,
                ids_all[test_idx], mask_all[test_idx], st_te, None, enc.device, cfg, seed,
            ))
        oos_full.append(np.mean(full_seeds, axis=0))
        oos_market.append(np.mean(market_seeds, axis=0))
        oos_y.append(y_all[test_idx])
        logger.info("fold done: test=%d", len(test_idx))

    y = np.concatenate(oos_y)
    full = np.concatenate(oos_full)
    market = np.concatenate(oos_market)
    b, c, p = _mcnemar(y, full, market)
    return {
        "frame": frame, "n_oos": int(len(y)),
        "acc_market": round(_acc(y, market), 4),
        "acc_full_lora": round(_acc(y, full), 4),
        "majority": round(max(float(y.mean()), 1 - float(y.mean())), 4),
        "mcnemar": {"full_better": b, "market_better": c, "p": round(p, 4)},
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="LoRA fine-tune late-fusion test.")
    parser.add_argument("--frame", default="daily")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22])
    args = parser.parse_args()
    cfg = LoraConfigArgs(epochs=args.epochs, seeds=tuple(args.seeds))
    result = run(args.frame, cfg)
    print("\n=== LoRA fine-tune result ===")
    for key, val in result.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
