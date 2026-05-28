"""Loughran-McDonald financial sentiment lexicon loader (#445).

The L-M Master Dictionary is the most-cited classical financial sentiment
resource (Loughran & McDonald, 2011; updated annually by McDonald at
https://sraf.nd.edu/loughranmcdonald-master-dictionary/). Each entry tags
a word with one or more sentiment categories (Negative, Positive,
Uncertainty, Litigious, Strong_Modal, Weak_Modal, Constraining) by the
year the word was first added; a non-zero cell means the word belongs
to that category in the published lexicon.

#445 wires the lexicon as an ablation baseline against the canonical
FinBERT-Fed-Adjacent encoder (ADR 0019). The loader reads the cached
CSV under ``data/external/loughran_mcdonald/<sha>__master_dictionary.csv``
and returns per-category word sets keyed on lowercase tokens; the
``compute_lm_features`` helper turns a document into six category-count
percentages that slot into the same FeatureVector position the pooled
encoder vector occupies on the canonical path.

The lexicon is public but is not re-downloaded in CI -- the SHA pin
locks the on-disk artefact, tests work off a small fixture passed via
``local_csv=``, and the HF / network paths are out of scope here. If the
cached file is missing, the loader raises ``FileNotFoundError`` rather
than reaching for the McDonald site at runtime.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from app.config import DATA_DIR

_logger = logging.getLogger(__name__)


# Canonical category names. ``modal`` is the union of ``Strong_Modal`` and
# ``Weak_Modal`` per the issue body -- the published lexicon splits modal
# strength into two columns; the ablation collapses them into a single
# modality count so the feature vector stays at six dims.
LM_CATEGORIES: tuple[str, ...] = (
    "positive",
    "negative",
    "uncertainty",
    "litigious",
    "constraining",
    "modal",
)

# CSV column names in the published Master Dictionary. The dictionary's
# schema has been stable since the 2014 release; the columns below are
# the ones the lexicon-as-feature path reads.
_LM_CSV_COLUMNS: dict[str, tuple[str, ...]] = {
    "positive": ("Positive",),
    "negative": ("Negative",),
    "uncertainty": ("Uncertainty",),
    "litigious": ("Litigious",),
    "constraining": ("Constraining",),
    # Modal: union of Strong_Modal + Weak_Modal so the count is "any
    # modal word" rather than a strength-weighted blend.
    "modal": ("Strong_Modal", "Weak_Modal"),
}

# Pinned SHA of the cached Master Dictionary CSV. The on-disk file is
# named ``<sha>__master_dictionary.csv`` so multiple vintages can sit
# side by side; the loader resolves the active vintage off the pinned
# constant rather than walking the directory. Updating the pin is a
# deliberate methodology decision -- a new vintage may shift the
# per-category counts and the ablation row's framing should be re-run
# against the new SHA before the wiki cell is updated.
LM_LEXICON_SHA: str = "lm_master_2024_q4"

DEFAULT_CACHE_ROOT = DATA_DIR / "external" / "loughran_mcdonald"


def _default_cache_path() -> Path:
    return DEFAULT_CACHE_ROOT / f"{LM_LEXICON_SHA}__master_dictionary.csv"


@dataclass(frozen=True)
class LoughranMcDonaldLexicon:
    """Per-category word sets keyed on lowercase tokens.

    Each set holds the tokens flagged in the named category on the
    cached Master Dictionary CSV. The ``source_sha`` field records the
    vintage tag the loader resolved against so a downstream artefact
    can be traced back to a specific lexicon pin.
    """

    source_sha: str
    categories: dict[str, frozenset[str]] = field(default_factory=dict)

    def words(self, category: str) -> frozenset[str]:
        if category not in LM_CATEGORIES:
            raise KeyError(
                f"Unknown L-M category: {category!r}. "
                f"Known: {LM_CATEGORIES}"
            )
        return self.categories.get(category, frozenset())

    @property
    def total_words(self) -> int:
        """Distinct tokens across every category (no double counting)."""

        merged: set[str] = set()
        for words in self.categories.values():
            merged.update(words)
        return len(merged)


def _is_flagged(cell: str | int | float | None) -> bool:
    """Return True when the lexicon cell marks a word as in-category.

    The published cells encode the year the word was first added to that
    category (e.g. ``2009``); an unmapped word has ``0`` (or empty).
    Treat any non-zero numeric (or non-empty truthy string) as flagged.
    """

    if cell is None:
        return False
    if isinstance(cell, bool):
        return bool(cell)
    if isinstance(cell, int | float):
        return cell != 0
    text = str(cell).strip()
    if not text:
        return False
    try:
        return float(text) != 0.0
    except ValueError:
        # Non-numeric truthy strings (e.g. "yes") are not part of the
        # published schema but we accept them defensively.
        return text.lower() not in {"0", "false", "no"}


def _normalise_token(word: str) -> str:
    return word.strip().lower()


def load_loughran_mcdonald(
    *,
    cache_root: Path | None = None,
    local_csv: Path | None = None,
    source_sha: str | None = None,
) -> LoughranMcDonaldLexicon:
    """Load the Loughran-McDonald Master Dictionary from cache.

    Parameters
    ----------
    cache_root
        Optional override for the cache root. Defaults to
        :data:`DEFAULT_CACHE_ROOT`.
    local_csv
        Optional path to a CSV fixture. When supplied the loader reads
        from this path directly and ignores ``cache_root`` -- this is
        the air-gapped path tests use so CI never reaches the network.
    source_sha
        Tag stored on the returned lexicon. Defaults to
        :data:`LM_LEXICON_SHA` (or the stem of ``local_csv`` when that
        path is supplied without an explicit tag).

    Raises
    ------
    FileNotFoundError
        When the resolved CSV does not exist on disk. The loader does
        not fetch the lexicon at runtime -- the SHA-pinned cache is the
        contract.
    """

    if local_csv is not None:
        csv_path = Path(local_csv)
        sha = source_sha or csv_path.stem
    else:
        root = cache_root if cache_root is not None else DEFAULT_CACHE_ROOT
        sha = source_sha or LM_LEXICON_SHA
        csv_path = root / f"{sha}__master_dictionary.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Loughran-McDonald cache miss at {csv_path}. The lexicon "
            "is public but not auto-downloaded; place the Master "
            "Dictionary CSV at the expected path or pass local_csv= "
            "(see docs/adr/0036-loughran-mcdonald-baseline.md)."
        )

    buckets: dict[str, set[str]] = {cat: set() for cat in LM_CATEGORIES}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(
                f"Loughran-McDonald CSV at {csv_path} is empty or has no header."
            )
        word_column = _resolve_word_column(reader.fieldnames)
        for row in reader:
            token = _normalise_token(str(row.get(word_column, "")))
            if not token:
                continue
            for category, columns in _LM_CSV_COLUMNS.items():
                if any(_is_flagged(row.get(col)) for col in columns):
                    buckets[category].add(token)

    return LoughranMcDonaldLexicon(
        source_sha=sha,
        categories={cat: frozenset(words) for cat, words in buckets.items()},
    )


def _resolve_word_column(fieldnames: Iterable[str]) -> str:
    """Pick the word column off the CSV header.

    The published schema uses ``Word``; older mirrors capitalise as
    ``WORD`` or ship a leading ``word`` lowercase variant. Accept any
    of them but raise when none are present so an upstream rename does
    not silently emit empty word sets.
    """

    field_list = list(fieldnames)
    for candidate in ("Word", "WORD", "word"):
        if candidate in field_list:
            return candidate
    raise ValueError(
        f"Loughran-McDonald CSV header missing the Word column; "
        f"saw {field_list[:8]}..."
    )


_TOKEN_RE = re.compile(r"[A-Za-z']+")


def _tokenise(text: str) -> list[str]:
    """Lowercase + alphabetic-only tokeniser.

    The L-M lexicon is alphabetic; the tokeniser strips punctuation and
    digits so the per-category counts measure word-level matches rather
    than the noise from numeric mentions or quoted symbols. Apostrophes
    are kept (the lexicon includes contractions / possessives in a few
    rows) but stripped at the token boundaries.
    """

    return [match.group(0).lower().strip("'") for match in _TOKEN_RE.finditer(text)]


def compute_lm_features(
    text: str,
    lexicon: LoughranMcDonaldLexicon,
) -> dict[str, float]:
    """Return the six per-category L-M percentages for ``text``.

    The keys follow the issue body's naming: ``lm_positive_pct``,
    ``lm_negative_pct``, ``lm_uncertainty_pct``, ``lm_litigious_pct``,
    ``lm_constraining_pct``, ``lm_modal_pct``. Each value is the share
    of document tokens (after lowercase + alphabetic-only tokenisation)
    that match the L-M category, expressed as a percentage in
    ``[0.0, 100.0]``. An empty document returns all zeros so the
    feature vector is well-defined on the missing-text branch.
    """

    tokens = _tokenise(text)
    if not tokens:
        return {f"lm_{cat}_pct": 0.0 for cat in LM_CATEGORIES}

    counts = dict.fromkeys(LM_CATEGORIES, 0)
    for token in tokens:
        for category in LM_CATEGORIES:
            if token in lexicon.categories.get(category, frozenset()):
                counts[category] += 1

    denom = float(len(tokens))
    return {
        f"lm_{cat}_pct": 100.0 * counts[cat] / denom for cat in LM_CATEGORIES
    }


def compute_lm_feature_vector(
    text: str,
    lexicon: LoughranMcDonaldLexicon,
) -> list[float]:
    """Return the six L-M percentages as an ordered list.

    Order follows :data:`LM_CATEGORIES`. The list is what the ablation
    runner writes into ``FeatureVector.text_embedding_pooled`` when the
    L-M arm replaces the canonical pooled-encoder block.
    """

    features = compute_lm_features(text, lexicon)
    return [features[f"lm_{cat}_pct"] for cat in LM_CATEGORIES]
