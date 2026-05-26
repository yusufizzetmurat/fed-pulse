"""Fine-tune the historical-analog retrieval encoder (#294).

Layers a sentence-transformer head on top of the cross-bank DAPT
checkpoint (``finbert_fed_adjacent_xbank_dapt``) and trains with
contrastive ``MultipleNegativesRankingLoss`` (MNRL — Henderson et al.,
2017) over same-meeting positive pairs:

* anchor   = an FOMC statement
* positive = the minutes or press conference released by the same
  meeting (rows in ``events.parquet`` sharing the anchor's
  ``event_date``)

In-batch contrastive learning treats every other positive in the batch
as a negative, so the loss is the standard cross-entropy over the
``batch_size x batch_size`` similarity matrix. The recipe is the SBERT
default and reproduces the same training signal as the asymmetric
search-retrieval MS MARCO setup, just on a vastly smaller corpus.

Outputs:
* ``out_dir / "checkpoint"`` — the sentence-transformer save directory
  (HF-style: ``config.json``, ``pytorch_model.bin``, tokenizer files,
  plus the SBERT ``1_Pooling/`` config). The runtime singleton at
  ``app.services.analogs`` reads this directory with
  ``AutoTokenizer`` + ``AutoModel`` so the inference path does not
  pull in sentence-transformers at startup.
* ``out_dir / "index.parquet"`` + ``embeddings.npy`` + ``manifest.json``
  — the historical-statement retrieval index produced by
  ``app.retrieval.index.build_index_from_events`` using the freshly
  trained encoder.
* ``out_dir / "training_args.json"`` — full run provenance.

Usage::

    python -m app.retrieval.train \\
        --events-parquet /data/processed/<pkg>/events.parquet \\
        --base-encoder-alias finbert_fed_adjacent_xbank_dapt \\
        --epochs 1 --batch-size 16 --seed 11 \\
        --fold-id wf_fold_3

The ``--train-end`` / ``--fold-id`` flags enforce a strict-backward
walk-forward boundary at training time: rows with
``event_date >= train_end`` are dropped BEFORE pair construction, so
the encoder's weights never see future text relative to the
walk-forward train slice. Pass either flag (not both); the resolved
boundary is persisted into the retrieval manifest so the runtime
query path can enforce the same cut.

Sentence-transformers is loaded lazily so the import-time surface stays
free of the heavy dependency — the runtime FastAPI worker never has to
touch it. Tests for the index + endpoint operate on faked encoders;
this script is only exercised at training time.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import DATA_DIR
from app.models.registry import encoder_ref
from app.retrieval.index import (
    EXCERPT_CHARS,
    MAX_TEXT_CHARS,
    build_index_from_events,
)

_logger = logging.getLogger(__name__)

DEFAULT_BASE_ENCODER_ALIAS = "finbert_fed_adjacent_xbank_dapt"
DEFAULT_OUTPUT_ROOT = DATA_DIR / "artifacts" / "retrieval"
DEFAULT_RUN_NAME = "finbert_fed_adjacent_xbank_dapt_retrieval"
DEFAULT_MAX_LENGTH = 256
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WARMUP_STEPS = 100
DEFAULT_SEED = 11

FOLD_MANIFEST_FILENAME = "fold_manifest_expanding_walk_forward.json"

# The pair-build path treats statements as anchors and pairs each
# statement with every other doc released on the same event_date.
# Limiting the kinds we accept as positives to minutes /
# press_conference avoids polluting the contrastive signal with
# orthogonal macro_release rows that share a date but carry no FOMC
# content. Adding "speech" / "testimony" stays a follow-up — they are
# released by individual speakers, not the committee.
POSITIVE_KINDS = ("minutes", "press_conference")


@dataclass(frozen=True)
class TrainingPair:
    """One (anchor, positive) example for MNRL contrastive training."""

    anchor: str
    positive: str
    anchor_date: str
    positive_kind: str


def _set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover — torch is a hard dep but keep the import lazy
        pass


def _clean_text(text: Any) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    return s[:MAX_TEXT_CHARS]


def _validate_train_end(train_end: str | None) -> str | None:
    """Normalise a train_end string to ISO ``YYYY-MM-DD`` form or ``None``."""

    if train_end is None or str(train_end).strip() == "":
        return None
    text = str(train_end).strip()
    try:
        return date_type.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(
            f"train_end {train_end!r} is not a valid ISO date (YYYY-MM-DD)"
        ) from exc


def resolve_train_end_from_fold(
    *,
    events_parquet: Path,
    fold_id: str,
) -> str:
    """Look up a fold's ``train_end`` from the sibling fold manifest.

    The fold manifest is expected at
    ``events_parquet.parent / fold_manifest_expanding_walk_forward.json``
    — the canonical training-package layout produced by
    :mod:`app.data.training_package_builder`. Raises ``ValueError`` if
    the manifest is missing or the fold_id is unknown.
    """

    manifest_path = events_parquet.parent / FOLD_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise ValueError(
            f"fold manifest not found at {manifest_path}; pass --train-end explicitly"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    folds = payload.get("folds") or []
    for fold in folds:
        if fold.get("fold_id") == fold_id:
            train_end = fold.get("train_end")
            if not train_end:
                raise ValueError(
                    f"fold {fold_id!r} in {manifest_path} carries no train_end"
                )
            return str(train_end)
    available = sorted(str(f.get("fold_id", "")) for f in folds)
    raise ValueError(
        f"fold_id {fold_id!r} not found in {manifest_path}; available: {available}"
    )


def _filter_events_by_train_end(events: pd.DataFrame, train_end: str) -> pd.DataFrame:
    """Drop rows with ``event_date >= train_end`` — strict walk-forward cut."""

    if "event_date" not in events.columns:
        raise KeyError("events parquet missing 'event_date' column")
    cutoff = pd.Timestamp(train_end).date().isoformat()
    dates = events["event_date"].astype(str)
    return events.loc[dates < cutoff].copy()


def build_training_pairs(  # noqa: C901 — flat column-validation guards keep the data contract greppable at the top of the function.
    events: pd.DataFrame,
    *,
    train_end: str | None = None,
) -> list[TrainingPair]:
    """Materialise same-meeting (statement, sibling) pairs from events.parquet.

    Statements are the anchor population; their sibling minutes /
    press_conference rows become positives. Meetings with no sibling
    are dropped entirely — the pre-review build emitted a degenerate
    ``(statement, statement)`` self-pair fallback, which teaches MNRL
    to maximise self-similarity and amplifies the self-match problem
    at retrieval time. Better to skip the meeting than to ship that
    signal.

    ``train_end`` (ISO date) enforces the walk-forward boundary at the
    earliest possible step: rows with ``event_date >= train_end`` are
    dropped BEFORE pair construction so the encoder never sees future
    text. Pass ``None`` for unbounded training (smoke / debug only).
    """

    if "event_kind" not in events.columns:
        raise KeyError("events parquet missing 'event_kind' column")
    if "text" not in events.columns:
        raise KeyError("events parquet missing 'text' column")
    if "event_date" not in events.columns:
        raise KeyError("events parquet missing 'event_date' column")
    if "text_hash" not in events.columns:
        raise KeyError("events parquet missing 'text_hash' column")

    df = events.copy()
    if train_end is not None:
        df = _filter_events_by_train_end(df, train_end)
    df["event_kind"] = df["event_kind"].astype(str).str.lower()
    df["event_date"] = df["event_date"].astype(str)
    if "horizon" in df.columns:
        df = df.sort_values(["event_date", "event_kind", "horizon"])
    df = df.drop_duplicates(subset=["text_hash"], keep="first").reset_index(drop=True)

    statements = df[df["event_kind"] == "statement"]
    pairs: list[TrainingPair] = []
    for _, anchor_row in statements.iterrows():
        anchor_text = _clean_text(anchor_row.get("text"))
        if not anchor_text:
            continue
        anchor_date = str(anchor_row.get("event_date", ""))
        siblings = df[
            (df["event_date"] == anchor_date)
            & (df["event_kind"].isin(POSITIVE_KINDS))
        ]
        for _, sibling in siblings.iterrows():
            sibling_text = _clean_text(sibling.get("text"))
            if not sibling_text or sibling_text == anchor_text:
                continue
            pairs.append(
                TrainingPair(
                    anchor=anchor_text,
                    positive=sibling_text,
                    anchor_date=anchor_date,
                    positive_kind=str(sibling.get("event_kind", "")),
                )
            )
        # Meetings whose only document is the statement contribute
        # nothing — silently dropped. MNRL cannot learn from a
        # (statement, statement) pair without ingraining a self-match
        # bias at retrieval time.
    return pairs


def _resolve_base_repo(alias: str) -> tuple[str, str]:
    """Resolve a registry alias to (repo, revision) for sentence-transformers.

    Local-path encoders (e.g. the DAPT checkpoint at
    ``/data/artifacts/continued_pretraining/...``) are handed back
    verbatim — sentence-transformers accepts a directory path the same
    way HF transformers does.
    """

    ref = encoder_ref(alias)
    if ref is None:
        raise ValueError(
            f"Unknown encoder alias {alias!r}. Add it to backend/app/models/registry.yaml."
        )
    return ref.repo, ref.revision or ""


def _build_sbert_model(repo: str, revision: str, *, max_seq_length: int):
    """Lazy sentence-transformers builder.

    Wraps the encoder in a Transformer + mean-pooling module so the
    saved checkpoint folder is a plain SBERT directory the runtime
    inference path can read with ``AutoModel`` + manual mean-pool.
    """

    from sentence_transformers import SentenceTransformer, models  # type: ignore[import-not-found,attr-defined,unused-ignore]

    word_embedding = models.Transformer(
        repo, max_seq_length=max_seq_length, model_args={"revision": revision or None}
    )
    pooling = models.Pooling(
        word_embedding.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    return SentenceTransformer(modules=[word_embedding, pooling])


def _build_input_examples(pairs: list[TrainingPair]):
    from sentence_transformers import InputExample  # type: ignore[import-not-found,unused-ignore]

    return [InputExample(texts=[pair.anchor, pair.positive]) for pair in pairs]


def _embed_with_sbert(model, texts: list[str]) -> np.ndarray:
    """Adapter so :mod:`app.retrieval.index` can call SBERT's encode path.

    SBERT returns numpy by default when ``convert_to_numpy=True``; we
    ask for unnormalised vectors and let ``index.build_index_from_events``
    apply the L2 normalisation so the on-disk matrix matches the
    cosine-similarity invariant.
    """

    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def fine_tune_and_index(  # noqa: PLR0913 — keyword-only training-script knobs; grouping would obscure CLI parity.
    *,
    events_parquet: Path,
    base_encoder_alias: str = DEFAULT_BASE_ENCODER_ALIAS,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_name: str = DEFAULT_RUN_NAME,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    max_length: int = DEFAULT_MAX_LENGTH,
    seed: int = DEFAULT_SEED,
    training_package_id: str | None = None,
    train_end: str | None = None,
    fold_id: str | None = None,
) -> Path:
    """Train the retrieval encoder and persist the index alongside it.

    Returns the path to the saved checkpoint directory.
    """

    if batch_size < 2:
        # MNRL builds its negatives from the rest of the in-batch
        # positives; a batch of 1 means there are no negatives and the
        # loss silently degenerates to zero. Fail fast so a broken
        # encoder never ships.
        raise ValueError(
            "MultipleNegativesRankingLoss requires batch_size >= 2; "
            f"got batch_size={batch_size}"
        )
    if train_end is not None and fold_id is not None:
        raise ValueError(
            "--train-end and --fold-id are mutually exclusive; pass only one"
        )

    resolved_train_end: str | None
    if fold_id is not None:
        resolved_train_end = resolve_train_end_from_fold(
            events_parquet=Path(events_parquet), fold_id=fold_id
        )
    else:
        resolved_train_end = _validate_train_end(train_end)

    _set_all_seeds(seed)
    repo, revision = _resolve_base_repo(base_encoder_alias)

    events = pd.read_parquet(events_parquet)
    pairs = build_training_pairs(events, train_end=resolved_train_end)
    if not pairs:
        raise RuntimeError(
            f"events_parquet {events_parquet} yielded zero (statement, sibling) pairs; "
            "verify the parquet carries statement + minutes rows and the "
            "train_end cutoff is not stripping every meeting."
        )

    import torch  # type: ignore[import-not-found,unused-ignore]
    from torch.utils.data import DataLoader  # type: ignore[import-not-found,unused-ignore]
    from sentence_transformers import losses  # type: ignore[import-not-found,attr-defined,unused-ignore]

    model = _build_sbert_model(repo, revision, max_seq_length=max_length)
    train_examples = _build_input_examples(pairs)
    # Explicit generator so the shuffle order is reproducible — the
    # ``--seed`` knob is otherwise effectively meaningless for batch
    # ordering, which is the dominant source of contrastive-loss
    # variance.
    shuffle_generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
        generator=shuffle_generator,
    )
    loss = losses.MultipleNegativesRankingLoss(model)

    out_dir = Path(output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoint"

    _logger.info(
        "retrieval_train_start base=%s pairs=%d epochs=%d batch_size=%d train_end=%s",
        base_encoder_alias,
        len(pairs),
        epochs,
        batch_size,
        resolved_train_end,
    )
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=min(warmup_steps, max(1, len(loader) // 2)),
        optimizer_params={"lr": learning_rate},
        show_progress_bar=False,
        output_path=str(checkpoint_dir),
        save_best_model=True,
    )

    # Build the historical-statement index alongside the checkpoint so
    # ``/analyze/analogs`` can mount the bundle as a single directory.
    def _embed(texts: list[str]) -> np.ndarray:
        return _embed_with_sbert(model, texts)

    loaded_index = build_index_from_events(
        events_parquet=Path(events_parquet),
        encoder_alias="finbert_fed_adjacent_xbank_dapt_retrieval",
        encoder_revision="",  # filled in once the registry pins this run
        embed_fn=_embed,
        training_package_id=training_package_id,
        out_dir=out_dir,
        train_end=resolved_train_end,
    )
    _logger.info(
        "retrieval_train_done index_rows=%d dim=%d out_dir=%s",
        loaded_index.size,
        loaded_index.embedding_dim,
        out_dir,
    )

    training_args = {
        "base_encoder_alias": base_encoder_alias,
        "base_encoder_repo": repo,
        "base_encoder_revision": revision,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "max_length": max_length,
        "seed": seed,
        "training_package_id": training_package_id,
        "pair_count": len(pairs),
        "excerpt_chars": EXCERPT_CHARS,
        "train_end": resolved_train_end,
        "fold_id": fold_id,
        "saved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out_dir / "training_args.json").write_text(
        json.dumps(training_args, indent=2, sort_keys=True), encoding="utf-8"
    )
    return checkpoint_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the historical-analog retrieval encoder (#294)."
    )
    parser.add_argument("--events-parquet", required=True, type=Path)
    parser.add_argument("--base-encoder-alias", default=DEFAULT_BASE_ENCODER_ALIAS)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), type=Path)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--training-package-id", default=None)
    parser.add_argument(
        "--train-end",
        default=None,
        help=(
            "ISO date (YYYY-MM-DD). Drop every event with event_date >= "
            "train_end BEFORE pair construction so the encoder never sees "
            "future text relative to the walk-forward train slice. Mutually "
            "exclusive with --fold-id."
        ),
    )
    parser.add_argument(
        "--fold-id",
        default=None,
        help=(
            "Resolve train_end from the sibling fold manifest "
            "(fold_manifest_expanding_walk_forward.json) by fold_id "
            "(e.g. wf_fold_3). Mutually exclusive with --train-end."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    checkpoint = fine_tune_and_index(
        events_parquet=args.events_parquet,
        base_encoder_alias=args.base_encoder_alias,
        output_root=args.output_root,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_length=args.max_length,
        seed=args.seed,
        training_package_id=args.training_package_id,
        train_end=args.train_end,
        fold_id=args.fold_id,
    )
    print(f"[retrieval.train] saved checkpoint to {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
