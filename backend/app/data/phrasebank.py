"""Financial PhraseBank loader for the B2 auxiliary-task fine-tune (#33).

PhraseBank (Malo et al. 2014) is a 4 840-sentence financial-sentiment
corpus with 3-way labels (positive / negative / neutral). Path A (DAPT
substrate) was rejected as noise-level against the 909k BIS NSP pair
pool; this module supports Path B — PhraseBank rows feed an auxiliary
3-way classification head during the encoder fine-tune so the encoder
sees in-domain sentiment supervision alongside the FOMC vol-regime
target.

Loader contract:

- :func:`load_phrasebank_rows` returns a list of
  :class:`PhraseBankRow` (id, sentence, 3-class label index). Labels
  follow the canonical order ``("negative", "neutral", "positive")``
  so the auxiliary head's softmax index is reproducible across runs.
- The on-wire source is the public HF mirror
  ``takala/financial_phrasebank`` (``sentences_allagree`` subset by
  default — 2 264 sentences with 100% annotator agreement). The
  ``sentences_50agree`` subset is wired through as an opt-in for the
  full 4 840-sentence pool.
- Reads are cached under ``data/external/phrasebank/`` as a parquet
  artefact so the loader is offline after the first hit. The cache
  key includes the subset name + dataset revision (when pinned).
- No write path. HF write tokens are revoked on this account; the
  loader is read-only.

The auxiliary head consumes the rows via
:func:`app.data.finetune_pilot_b2._train_and_eval_one_cell` when
``--enable-phrasebank-aux`` is set.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.config import DATA_DIR

_logger = logging.getLogger(__name__)

DEFAULT_DATASET_ID = "takala/financial_phrasebank"
# `sentences_allagree` is the strict 100%-agreement subset (2 264
# rows); the wider 50%-agreement subset (`sentences_50agree`, 4 840
# rows) is opt-in via `subset=` for runs that prefer the higher row
# count over the label-noise floor.
DEFAULT_SUBSET = "sentences_allagree"
VALID_SUBSETS: tuple[str, ...] = (
    "sentences_allagree",
    "sentences_75agree",
    "sentences_66agree",
    "sentences_50agree",
)

# Canonical label order. Pin the order so the auxiliary head's softmax
# index is reproducible — HF's `ClassLabel` happens to expose the
# labels in the same order today, but the harness must not depend on
# that. The mapping below is the contract.
PHRASEBANK_LABELS: tuple[str, ...] = ("negative", "neutral", "positive")
LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(PHRASEBANK_LABELS)}
ID2LABEL: dict[int, str] = dict(enumerate(PHRASEBANK_LABELS))
N_CLASSES = len(PHRASEBANK_LABELS)

# HF dataset revision pin (#425). Left None until the first canonical
# sweep resolves a 40-char SHA -- a named ref like "main" would add a
# cache-filename change with zero reproducibility upside (both `None`
# and `"main"` resolve to the mutable HEAD of takala/financial_phrasebank
# main). Mirrors the BIS-MLM pin discipline in
# ``continued_pretraining.py``: the SHA is the only pin that protects
# the aux artefact from upstream branch rewrites.
# TODO(#425): pin to the SHA the next canonical sweep prints out.
DEFAULT_REVISION: str | None = None

DEFAULT_CACHE_ROOT = DATA_DIR / "external" / "phrasebank"


@dataclass(frozen=True)
class PhraseBankRow:
    """One PhraseBank sentence + its 3-class label index.

    ``label_idx`` follows :data:`PHRASEBANK_LABELS` ordering (0 =
    negative, 1 = neutral, 2 = positive). ``row_id`` is the loader's
    monotonically-assigned index per (subset, revision) so two loads
    of the same subset round-trip to the same id.
    """

    row_id: str
    sentence: str
    label_idx: int

    @property
    def label(self) -> str:
        return PHRASEBANK_LABELS[self.label_idx]


def _coerce_label_idx(label: Any) -> int | None:
    """Normalise an HF row's label field to the canonical index.

    The HF mirror exposes the label either as an int (0/1/2 keyed on
    its own `ClassLabel` order — which happens to be
    `negative / neutral / positive` today) or as a string after a
    `dataset.cast(...)` pass. Accept both; reject anything else.
    """

    if isinstance(label, str):
        normalised = label.strip().lower()
        if normalised in LABEL2ID:
            return LABEL2ID[normalised]
        return None
    if isinstance(label, int) and not isinstance(label, bool):
        if 0 <= int(label) < N_CLASSES:
            return int(label)
        return None
    try:
        idx = int(label)
    except (TypeError, ValueError):
        return None
    if 0 <= idx < N_CLASSES:
        return idx
    return None


def _iter_local_rows(rows: Iterable[dict[str, Any]]) -> list[PhraseBankRow]:
    """Adapt iter-of-dict rows into :class:`PhraseBankRow` objects.

    Drops rows whose sentence is empty or whose label does not
    normalise to a canonical 3-class index. Splits cleanly from the
    HF read path so a local fixture (parquet or JSONL) can replay the
    same parse contract during tests.
    """

    out: list[PhraseBankRow] = []
    for idx, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        sentence = str(item.get("sentence") or item.get("text") or "").strip()
        if not sentence:
            continue
        label_idx = _coerce_label_idx(item.get("label"))
        if label_idx is None:
            continue
        out.append(
            PhraseBankRow(
                row_id=str(item.get("row_id") or f"pb_{idx:05d}"),
                sentence=sentence,
                label_idx=label_idx,
            )
        )
    return out


def _cache_path(cache_root: Path, subset: str, revision: str | None) -> Path:
    rev_tag = (revision or "head")[:12]
    return cache_root / f"{subset}__{rev_tag}.parquet"


def _read_cache(path: Path) -> list[PhraseBankRow] | None:
    if not path.exists():
        return None
    try:
        import pandas as pd
    except ImportError:
        return None
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 -- cache miss is recoverable
        _logger.warning("PhraseBank cache read failed at %s: %s", path, exc)
        return None
    return _iter_local_rows(frame.to_dict("records"))


def _write_cache(path: Path, rows: list[PhraseBankRow]) -> None:
    try:
        import pandas as pd
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [{"row_id": r.row_id, "sentence": r.sentence, "label": r.label_idx} for r in rows]
    )
    frame.to_parquet(path)


def _download_from_hub(dataset_id: str, subset: str, revision: str | None) -> list[PhraseBankRow]:
    """Pull PhraseBank from the HF mirror — read-only.

    Lazy import of :mod:`datasets` so the module is importable in CI
    without the HF datasets dependency. Raises :class:`RuntimeError`
    when the dependency is missing so the caller surfaces a clean
    skip in tests.
    """

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PhraseBank loader requires the `datasets` package; "
            "install it via the backend requirements lock."
        ) from exc

    kwargs: dict[str, Any] = {"split": "train"}
    if revision:
        kwargs["revision"] = revision
    # `trust_remote_code=True` is the HF default for legacy script-
    # based loaders (PhraseBank ships one). The mirror is public so
    # no token is needed; HF_TOKEN, when set, is forwarded transparently
    # by the datasets library itself.
    kwargs["trust_remote_code"] = True
    ds = load_dataset(dataset_id, subset, **kwargs)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        rows.append(
            {
                "row_id": f"pb_{subset}_{idx:05d}",
                "sentence": row.get("sentence"),
                "label": row.get("label"),
            }
        )
    return _iter_local_rows(rows)


def load_phrasebank_rows(
    *,
    subset: str = DEFAULT_SUBSET,
    revision: str | None = DEFAULT_REVISION,
    dataset_id: str = DEFAULT_DATASET_ID,
    cache_root: Path | None = None,
    local_jsonl: Path | None = None,
) -> list[PhraseBankRow]:
    """Load PhraseBank rows for the auxiliary head.

    Parameters
    ----------
    subset
        PhraseBank subset name — one of :data:`VALID_SUBSETS`.
        Defaults to the strict 100%-agreement subset.
    revision
        Optional HF dataset revision pin. When ``None`` the loader
        fetches HEAD; when pinned the cache key changes so two
        revisions are stored side by side.
    dataset_id
        HF dataset id. Defaults to :data:`DEFAULT_DATASET_ID`.
    cache_root
        Optional override for the on-disk cache root. Defaults to
        :data:`DEFAULT_CACHE_ROOT` under the configured data dir.
    local_jsonl
        Optional path to a local JSONL fixture. When supplied the
        loader skips the HF read path entirely and parses the file
        in place — used by tests + air-gapped reproductions.
    """

    if subset not in VALID_SUBSETS:
        raise ValueError(f"PhraseBank subset {subset!r} not in {VALID_SUBSETS!r}")

    if local_jsonl is not None:
        if not local_jsonl.exists():
            raise FileNotFoundError(f"PhraseBank fixture missing: {local_jsonl}")
        rows = []
        for line in local_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return _iter_local_rows(rows)

    root = cache_root if cache_root is not None else DEFAULT_CACHE_ROOT
    cache_path = _cache_path(root, subset, revision)
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached

    rows = _download_from_hub(dataset_id, subset, revision)
    if rows:
        _write_cache(cache_path, rows)
    return rows


def class_counts(rows: Iterable[PhraseBankRow]) -> list[int]:
    """Return the per-class row counts in canonical label order."""

    counts = [0] * N_CLASSES
    for row in rows:
        if 0 <= row.label_idx < N_CLASSES:
            counts[row.label_idx] += 1
    return counts
