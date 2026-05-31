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


def _embed_one(text: str, tok: Any, model: Any, *, pooling: str, prefix: str, max_len: int) -> Any:
    """Chunk a document, encode each chunk, mean-pool chunk vectors → one vector."""

    import torch

    ids = tok(prefix + text, add_special_tokens=False)["input_ids"]
    body = max_len - 2  # room for [CLS]/[SEP]
    chunks = [ids[i : i + body] for i in range(0, max(len(ids), 1), body)] or [[]]
    vecs = []
    for chunk in chunks:
        enc = tok.prepare_for_model(
            chunk, return_tensors="pt", padding="max_length", max_length=max_len, truncation=True
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        vecs.append(_pool(out.last_hidden_state, enc["attention_mask"], pooling)[0])
    return torch.stack(vecs).mean(0).cpu().numpy()


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
