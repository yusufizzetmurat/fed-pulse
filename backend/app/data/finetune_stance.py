"""Fine-tune an own hawkish/dovish/neutral FOMC stance classifier — clean eval.

Replaces the third-party (gated) FOMC-RoBERTa fallback with an own, published,
ungated model: a 3-class head on the project's finbert-fed-adjacent encoder.

Fair comparison design: FOMC-RoBERTa was trained on the Trillion-Dollar-Words
FOMC set (source ``hf_fomc_communication``). We train OURS on the same TDW data,
and evaluate BOTH on a held-out Fed-stance set neither model trained on
(``gtfintechlab_federal_reserve_system`` + ``op_fed`` — verified zero text-hash
overlap with TDW). Both models face the same domain/scheme shift, so the
head-to-head is apples-to-apples. We only swap in our model if it genuinely wins.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import DATA_DIR

logger = logging.getLogger(__name__)

LABELS = ["hawkish", "dovish", "neutral"]
_L2I = {lab: i for i, lab in enumerate(LABELS)}
BASE_ENCODER = "yusufizzetmurat/finbert-fed-adjacent"
_ROBERTA_MAP = {"LABEL_0": "dovish", "LABEL_1": "hawkish", "LABEL_2": "neutral"}

# FOMC-RoBERTa's training data — train ours on the same, exclude from the test.
TRAIN_SOURCES = ["hf_fomc_communication"]
# Fed-stance sentences neither model trained on (zero TDW overlap) — clean test.
TEST_SOURCES = ["gtfintechlab_federal_reserve_system", "op_fed"]


def load_labeled(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path)
    d = d[d["mapped_label"].astype(str).isin(LABELS)].dropna(subset=["text"]).copy()
    d = d.drop_duplicates("text_hash") if "text_hash" in d.columns else d.drop_duplicates("text")
    d["y"] = d["mapped_label"].astype(str).map(_L2I)
    return d[["text", "y", "mapped_label", "source"]].reset_index(drop=True)


def _val_carve(
    df: pd.DataFrame, seed: int = 11, val: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for _, grp in df.groupby("y"):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        n_va = int(len(idx) * val)
        va.append(grp.loc[idx[:n_va]])
        tr.append(grp.loc[idx[n_va:]])
    return pd.concat(tr).reset_index(drop=True), pd.concat(va).reset_index(drop=True)


@torch.no_grad()
def _predict(
    model: Any, tok: Any, texts: list[str], device: torch.device, bs: int = 32
) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(texts), bs):
        enc = tok(
            texts[i : i + bs], return_tensors="pt", truncation=True, max_length=256, padding=True
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        preds.append(model(**enc).logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def train(
    df_tr: pd.DataFrame, df_va: pd.DataFrame, device: torch.device, epochs: int, lr: float
) -> tuple[Any, Any]:
    tok = AutoTokenizer.from_pretrained(BASE_ENCODER)  # type: ignore[no-untyped-call]
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_ENCODER,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=_L2I,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # class weights (inverse frequency) — the stance classes are imbalanced.
    counts = df_tr["y"].value_counts().reindex(range(len(LABELS))).fillna(1).to_numpy()
    cw = torch.tensor(counts.sum() / (len(LABELS) * counts), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=cw)

    texts, ys = df_tr["text"].tolist(), torch.tensor(df_tr["y"].to_numpy())
    n, bs = len(texts), 16
    best_f1, best_state, patience, bad = -1.0, None, 3, 0
    for ep in range(epochs):
        model.train()
        order = torch.randperm(n)
        for s in range(0, n, bs):
            idx = order[s : s + bs].tolist()
            enc = tok(
                [texts[i] for i in idx],
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            opt.zero_grad()
            logits = model(**enc).logits
            loss_fn(logits, ys[idx].to(device)).backward()
            opt.step()
        f1 = f1_score(
            df_va["y"], _predict(model, tok, df_va["text"].tolist(), device), average="macro"
        )
        logger.info("epoch %d: val macro-F1 %.4f", ep + 1, f1)
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                logger.info("early stop at epoch %d (best val-F1 %.4f)", ep + 1, best_f1)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, tok


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "acc": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Fine-tune own FOMC stance classifier (clean eval)."
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DATA_DIR
        / "processed"
        / "tp_v3_full_rebuild_2026_05_30"
        / "registry_normalized.parquet",
    )
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "processed" / "stance_finetune")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_labeled(args.labels)
    train_pool = df[df["source"].isin(TRAIN_SOURCES)].reset_index(drop=True)
    test_df = df[df["source"].isin(TEST_SOURCES)].reset_index(drop=True)
    logger.info("train pool (TDW): %d | clean test: %d", len(train_pool), len(test_df))
    logger.info("test by class: %s", dict(test_df["mapped_label"].value_counts()))
    tr, va = _val_carve(train_pool)

    model, tok = train(tr, va, device, args.epochs, args.lr)
    ours = _predict(model, tok, test_df["text"].tolist(), device)

    tok_b = AutoTokenizer.from_pretrained(
        "gtfintechlab/FOMC-RoBERTa", token=os.environ.get("HF_TOKEN")
    )  # type: ignore[no-untyped-call]
    mdl_b = AutoModelForSequenceClassification.from_pretrained(
        "gtfintechlab/FOMC-RoBERTa", token=os.environ.get("HF_TOKEN")
    ).to(device)
    raw_b = _predict(mdl_b, tok_b, test_df["text"].tolist(), device)
    base = np.array([_L2I[_ROBERTA_MAP[f"LABEL_{int(p)}"]] for p in raw_b])

    y_true = test_df["y"].to_numpy()
    report = {
        "design": "both trained on TDW; tested on Fed-stance held-out neither trained on",
        "n_test": int(len(test_df)),
        "ours": _scores(y_true, ours),
        "fomc_roberta": _scores(y_true, base),
        "ours_per_class": classification_report(
            y_true, ours, target_names=LABELS, output_dict=True, zero_division=0
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    (args.out_dir / "stance_eval.json").write_text(json.dumps(report, indent=2, default=str))
    print(
        json.dumps({k: report[k] for k in ("design", "n_test", "ours", "fomc_roberta")}, indent=2)
    )


if __name__ == "__main__":
    main()
