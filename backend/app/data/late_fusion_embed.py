"""Clean-room text embedding for the late-fusion rebuild — with a hard
anti-fallback gate.

Fault class #2 in this rebuild is the known silent fallback: the serving encoder
loader logs but does not raise when the FinBERT-fed model is unavailable, quietly
substituting a generic ``distilbert-sst-2`` model and producing meaningless
sentiment-axis vectors. This module refuses that: it resolves the encoder via the
registry, then HARD-ASSERTS the resolved repo is the expected FinBERT-fed model
(by name AND by ``model_type`` — distilbert shares the 768 hidden size, so a
dimension check alone is insufficient) and the expected embedding dim. Any
mismatch raises; there is no silent path.

It also embeds the WHOLE document (chunk + mean-pool + average across chunks)
rather than truncating to the first 512 tokens, so a full FOMC statement is not
discarded after its first paragraph.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from app.config import DATA_DIR
from app.models.registry import encoder_ref

logger = logging.getLogger(__name__)

DEFAULT_ALIAS = "finbert_fed_adjacent"
EXPECTED_REPO_SUBSTR = "finbert-fed"
EXPECTED_DIM = 768
MAX_TOKENS = 512
MAX_CHUNKS = 16  # cap very long speeches at ~8k tokens


@dataclass(frozen=True)
class Encoder:
    tokenizer: Any
    model: Any
    device: torch.device
    repo: str
    revision: str | None
    dim: int


def load_encoder(
    alias: str = DEFAULT_ALIAS,
    expected_dim: int = EXPECTED_DIM,
    expected_repo_substr: str = EXPECTED_REPO_SUBSTR,
) -> Encoder:
    """Resolve + load the encoder, hard-failing on any non-expected model.

    Raises RuntimeError rather than silently falling back, so a wrong/generic
    encoder can never be used unnoticed (fault class #2).
    """
    ref = encoder_ref(alias)
    repo = getattr(ref, "repo", None) if ref is not None else None
    if not repo:
        raise RuntimeError(f"encoder alias {alias!r} did not resolve to a repo")
    if expected_repo_substr not in repo.lower():
        raise RuntimeError(
            f"resolved encoder {repo!r} is not the expected "
            f"{expected_repo_substr!r}; refusing silent fallback"
        )
    revision = getattr(ref, "revision", None)
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)  # type: ignore[no-untyped-call]
    model = AutoModel.from_pretrained(repo, revision=revision)
    if model.config.model_type == "distilbert":
        raise RuntimeError(f"resolved encoder {repo!r} is a distilbert model; refusing")
    if model.config.hidden_size != expected_dim:
        raise RuntimeError(
            f"encoder {repo!r} hidden_size {model.config.hidden_size} != {expected_dim}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    logger.info(
        "loaded encoder repo=%s rev=%s type=%s dim=%d device=%s",
        repo, revision, model.config.model_type, model.config.hidden_size, device,
    )
    return Encoder(tokenizer, model, device, repo, revision, model.config.hidden_size)


@torch.no_grad()
def embed_documents(enc: Encoder, texts: list[str]) -> np.ndarray:
    """Embed each document to a (N, dim) matrix via chunk + mean-pool + average.

    Empty text -> zero vector (the caller should track an availability mask).
    """
    out = np.zeros((len(texts), enc.dim), dtype=np.float32)
    for i, text in enumerate(texts):
        cleaned = (text or "").strip()
        if not cleaned:
            continue
        enc_tok = enc.tokenizer(
            cleaned,
            max_length=MAX_TOKENS,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=True,
            stride=0,
            return_tensors="pt",
        )
        n_chunks = enc_tok["input_ids"].shape[0]
        if n_chunks > MAX_CHUNKS:
            logger.warning(
                "doc %d: %d chunks available, capping at %d (tail dropped)",
                i, n_chunks, MAX_CHUNKS,
            )
        ids = enc_tok["input_ids"][:MAX_CHUNKS].to(enc.device)
        mask = enc_tok["attention_mask"][:MAX_CHUNKS].to(enc.device)
        hidden = enc.model(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # (chunks, dim)
        out[i] = pooled.mean(dim=0).cpu().numpy()
    return out


def _fingerprint(embeddings: np.ndarray) -> str:
    """Stable short hash of the embedding matrix, for provenance logging."""
    rounded = np.round(embeddings, 4).astype(np.float32).tobytes()
    return hashlib.sha256(rounded).hexdigest()[:16]


def embed_frame(
    enc: Encoder, frame_path: Path, text_col: str, id_col: str, out_path: Path
) -> pd.DataFrame:
    """Embed a frame's text column; write doc_id + id_col + emb_* columns.

    doc_id is the positional row index in ``frame_path`` (read without re-sorting),
    so the modelling phase aligns embeddings to the frame by position.
    """
    frame = pd.read_parquet(frame_path)
    embeddings = embed_documents(enc, frame[text_col].astype(str).tolist())
    cols = {f"emb_{j:03d}": embeddings[:, j] for j in range(embeddings.shape[1])}
    out = pd.DataFrame(cols)
    out.insert(0, "doc_id", np.arange(len(frame)))
    out.insert(1, id_col, frame[id_col].to_numpy())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    norms = np.linalg.norm(embeddings, axis=1)
    logger.info(
        "embedded %d docs from %s | dim=%d repo=%s rev=%s | norm mean=%.4f std=%.4f | fp=%s",
        len(frame), frame_path.name, enc.dim, enc.repo, enc.revision,
        float(norms.mean()), float(norms.std()), _fingerprint(embeddings),
    )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Embed late-fusion frames (asserting encoder).")
    base = DATA_DIR / "processed" / "late_fusion"
    parser.add_argument("--event-frame", type=Path, default=base / "event_frame.parquet")
    parser.add_argument("--daily-frame", type=Path, default=base / "daily_frame.parquet")
    parser.add_argument("--event-out", type=Path, default=base / "event_text_emb.parquet")
    parser.add_argument("--daily-out", type=Path, default=base / "daily_text_emb.parquet")
    parser.add_argument("--alias", type=str, default=DEFAULT_ALIAS)
    args = parser.parse_args()

    enc = load_encoder(args.alias)
    if args.event_frame.exists():
        embed_frame(enc, args.event_frame, "text", "event_date", args.event_out)
    if args.daily_frame.exists():
        # row_hash is the stable join key (comm_date is not unique).
        embed_frame(enc, args.daily_frame, "text", "row_hash", args.daily_out)
    print(f"done: encoder={enc.repo}@{enc.revision} dim={enc.dim}")


if __name__ == "__main__":
    main()
