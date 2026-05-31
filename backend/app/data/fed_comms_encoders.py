"""General AutoModel text-embedding zoo for the encoder sweep.

`fed_comms_train.build_corpus_embeddings` covers classification-style encoders
(FinBERT) via the repo's `encode_chunks`. This module adds the embedding-native
encoders that path can't drive — bge / e5 / gte — through a direct
transformers `AutoModel` with explicit pooling, so the encoder sweep can compare
representations on equal footing. Output matches the FinBERT cache schema
(`url`, `emb_0…emb_{d-1}`), so the trainer consumes any of them unchanged.

Long documents (minutes / press-conference transcripts run 50–90K chars) are
tokenized, split into ≤`max_len`-token chunks, each chunk pooled to a vector,
then mean-pooled across chunks — the same reduction the FinBERT path uses, so
the only variable across the sweep is the encoder itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

# encoder spec: hf id → (pooling, per-document text prefix)
ENCODERS: dict[str, tuple[str, str]] = {
    "bge-large": ("cls", ""),  # BAAI/bge-large-en-v1.5
    "e5-large": ("mean", "passage: "),  # intfloat/e5-large-v2
    "gte-large": ("mean", ""),  # thenlper/gte-large
}
_HF_IDS = {
    "bge-large": "BAAI/bge-large-en-v1.5",
    "e5-large": "intfloat/e5-large-v2",
    "gte-large": "thenlper/gte-large",
}


def _pool(hidden: Any, attn: Any, mode: str) -> Any:
    """CLS or attention-masked mean pooling of a (chunk, token, d) batch."""

    import torch

    if mode == "cls":
        return hidden[:, 0]
    mask = attn.unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def _embed_one(
    text: str,
    tok: Any,
    model: Any,
    *,
    pooling: str,
    prefix: str,
    max_len: int,
    max_chunks: int = 64,
) -> Any:
    """Chunk a document, encode all chunks in one forward, mean-pool chunk vectors.

    Chunk tensors are built directly from special-token ids (robust across BERT /
    RoBERTa tokenizers, no `prepare_for_model`). Capped at `max_chunks` (~32K
    tokens) to bound memory on the longest minutes; this rarely truncates since
    most documents are well under that.
    """

    import torch

    ids = tok(prefix + text, add_special_tokens=False)["input_ids"]
    body = max_len - 2  # room for [CLS]/[SEP]
    chunks = [ids[i : i + body] for i in range(0, max(len(ids), 1), body)][:max_chunks] or [[]]
    cls, sep = tok.cls_token_id, tok.sep_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    seqs, masks = [], []
    for chunk in chunks:
        seq = [cls, *chunk, sep]
        attn = [1] * len(seq)
        if len(seq) < max_len:
            n = max_len - len(seq)
            seq, attn = seq + [pad] * n, attn + [0] * n
        seqs.append(seq)
        masks.append(attn)
    input_ids = torch.tensor(seqs, device=model.device)
    attn_t = torch.tensor(masks, device=model.device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_t)
    return _pool(out.last_hidden_state, attn_t, pooling).mean(0).cpu().numpy()


def build_corpus_embeddings_automodel(
    corpus_path: Path | str,
    out_path: Path | str,
    *,
    encoder: str,
    max_len: int = 512,
    force: bool = False,
) -> Path:
    """Embed the corpus with an AutoModel encoder; write the trainer's cache schema."""

    import pandas as pd
    import torch
    from transformers import AutoModel, AutoTokenizer

    out_path = Path(out_path)
    if out_path.exists() and not force:
        return out_path
    if encoder not in ENCODERS:
        raise ValueError(f"unknown encoder {encoder!r}; choices: {sorted(ENCODERS)}")
    pooling, prefix = ENCODERS[encoder]
    hf_id = _HF_IDS[encoder]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id).to(dev).eval()

    corpus = pd.read_parquet(corpus_path)
    rows: list[dict[str, Any]] = []
    dim = 0
    for _, doc in corpus.iterrows():
        emb = _embed_one(
            str(doc["text"]), tok, model, pooling=pooling, prefix=prefix, max_len=max_len
        )
        dim = len(emb)
        rows.append({"url": doc["url"], **{f"emb_{i}": float(emb[i]) for i in range(dim)}})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"[fed_comms_encoders] {encoder} ({hf_id}) wrote {len(rows)} embeddings (dim={dim})")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed the Fed corpus with an AutoModel encoder.")
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--encoder", required=True, choices=sorted(ENCODERS))
    parser.add_argument("--max-len", type=int, default=512)
    args = parser.parse_args()
    build_corpus_embeddings_automodel(
        args.corpus_path, args.out_path, encoder=args.encoder, max_len=args.max_len
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
