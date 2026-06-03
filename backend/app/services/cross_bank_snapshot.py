"""Cross-bank stance + vol-regime snapshot service.

Backs ``GET /cross-bank/snapshot``. For each of the six central banks the
xbank-DAPT multi-axis classifier was trained on (Fed, ECB, BoE, BoC, BoJ,
RBA), this module:

1. Pulls the most recent annotated sentences from the ingested
   ``source_registry.jsonl`` (the gtfintechlab corpus is rev-pinned at
   ingest time; event_date is year-rounded).
2. Runs the multi-axis classifier (``score_text``) over a small sample and
   averages the stance/certainty/time distributions to produce one card
   per bank. The classifier weights are bank-aware because the xbank-DAPT
   continued-pretraining substrate already saw all six banks' corpora.
3. Looks up the bank's flagship equity index (^GSPC, ^STOXX50E, ^FTSE,
   ^GSPTSE, ^N225, ^AXJO) and bins the trailing 5-day realised volatility
   into ``calm`` / ``normal`` / ``high`` against a per-symbol band. This
   is a coarse heuristic that does not depend on the Fed-trained
   vol-regime classifier (which was calibrated on ^GSPC FOMC-day windows
   and is not transferable cross-asset).

Results are cached in-process for ``_CACHE_TTL_SECONDS`` (default 1h) — a
Fed/ECB/BoE statement does not change minute to minute, and the
classifier cold-start is the most expensive part of the path.

When the corpus or the classifier checkpoint is missing for a bank, the
card is still emitted with ``status="corpus_missing"`` /
``status="classifier_unavailable"`` so the frontend can render a
"Coming soon" placeholder instead of a 500.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.config import DATA_DIR
from app.models.config import MULTI_TASK_STANCE_LABELS

_logger = logging.getLogger(__name__)

# Cache TTL: an hour balances "fresh enough for the dashboard" against
# the cold-start cost of the multi-axis classifier checkpoint load.
_CACHE_TTL_SECONDS = 3600

# Sentence sample size per bank. The classifier is ~30 ms/sentence on
# CPU after warm-up; 32 keeps the worst-case panel build under ~6 s
# across all six banks while still smoothing per-sentence noise.
_SAMPLE_SIZE = 32


@dataclass(frozen=True)
class BankSpec:
    key: str
    source: str
    display_name: str
    flag: str  # ISO country code; frontend maps to flag emoji
    symbol: str
    short_code: str  # FED / ECB / BOE / BOC / BOJ / RBA


BANK_SPECS: tuple[BankSpec, ...] = (
    BankSpec(
        key="fed",
        source="gtfintechlab_federal_reserve_system",
        display_name="Federal Reserve",
        flag="US",
        symbol="^GSPC",
        short_code="FED",
    ),
    BankSpec(
        key="ecb",
        source="gtfintechlab_european_central_bank",
        display_name="European Central Bank",
        flag="EU",
        symbol="^STOXX50E",
        short_code="ECB",
    ),
    BankSpec(
        key="boe",
        source="gtfintechlab_bank_of_england",
        display_name="Bank of England",
        flag="GB",
        symbol="^FTSE",
        short_code="BOE",
    ),
    BankSpec(
        key="boc",
        source="gtfintechlab_bank_of_canada",
        display_name="Bank of Canada",
        flag="CA",
        symbol="^GSPTSE",
        short_code="BOC",
    ),
    BankSpec(
        key="boj",
        source="gtfintechlab_bank_of_japan",
        display_name="Bank of Japan",
        flag="JP",
        symbol="^N225",
        short_code="BOJ",
    ),
    BankSpec(
        key="rba",
        source="gtfintechlab_reserve_bank_of_australia",
        display_name="Reserve Bank of Australia",
        flag="AU",
        symbol="^AXJO",
        short_code="RBA",
    ),
)


@dataclass(frozen=True)
class _CacheEntry:
    payload: dict[str, Any]
    expires_at: float


_cache: dict[str, _CacheEntry] = {}
_cache_lock = threading.Lock()


def _registry_path() -> Path:
    return DATA_DIR / "raw" / "phase2" / "source_registry.jsonl"


def _load_recent_sentences(  # noqa: C901 - branching is per-line defensive parsing
    source: str,
    *,
    limit: int = _SAMPLE_SIZE,
    registry_path: Path | None = None,
) -> tuple[list[str], str | None]:
    """Return ``(sentences, latest_event_date_iso)`` for the given bank.

    Streams ``source_registry.jsonl`` so the full registry never sits
    in memory (it tops 600 MB at full ingest). The gtfintechlab corpora
    use a year-rounded ``event_date``; we still walk the file and pick
    the rows with the maximum date, sampling up to ``limit`` from that
    set. Returns an empty list + ``None`` when the registry or the
    source is absent.
    """

    path = registry_path or _registry_path()
    if not path.exists():
        _logger.warning("cross_bank_snapshot_registry_missing path=%s", path)
        return [], None

    latest_date = ""
    bucket: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("source") != source:
                    continue
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                event_date = row.get("event_date") or ""
                if event_date > latest_date:
                    latest_date = event_date
                    bucket = [text]
                elif event_date == latest_date:
                    bucket.append(text)
    except OSError as exc:
        _logger.warning(
            "cross_bank_snapshot_registry_read_failed source=%s err=%s",
            source,
            exc,
        )
        return [], None

    if not bucket:
        return [], None

    # Deterministic sample: alphabetical sort + head ``limit`` so a
    # cache rebuild on a fresh process produces the same card.
    bucket.sort()
    sample = bucket[:limit]
    return sample, (latest_date or None)


def _aggregate_distribution(
    scores: list[dict[str, Any]],
    block: str,
    labels: tuple[str, ...],
) -> dict[str, float] | None:
    """Average the per-class softmax across the sentence scores.

    Returns ``None`` when no usable score is present. Renormalises
    after averaging to absorb floating-point drift.
    """

    sums: dict[str, float] = {label: 0.0 for label in labels}
    n = 0
    for score in scores:
        if not score:
            continue
        dist = (score.get(block) or {}).get("distribution") or {}
        if not dist:
            continue
        for label in labels:
            sums[label] += float(dist.get(label, 0.0))
        n += 1
    if n == 0:
        return None
    averaged = {label: sums[label] / n for label in labels}
    total = sum(averaged.values())
    if total <= 0:
        return None
    return {label: value / total for label, value in averaged.items()}


def _argmax_label(
    dist: dict[str, float] | None,
) -> tuple[str | None, float | None]:
    if not dist:
        return None, None
    label = max(dist, key=lambda k: dist[k])
    return label, float(dist[label])


def _build_vol_regime(symbol: str) -> dict[str, Any]:
    """Return ``{label, confidence, close, vol_5d_annualised, as_of, status}``.

    Uses a coarse 5-day realised-vol band: annualised vol < 12% -> calm,
    > 22% -> high, else normal. Cuts are intentionally the same across
    all six indices — this surfaces a quick cross-bank read, not a
    calibrated regime forecast.
    """

    try:
        from app.services.market_data import fetch_market_snapshot
    except Exception as exc:  # pragma: no cover -- import-time only
        _logger.warning("cross_bank_snapshot_market_data_import_failed err=%s", exc)
        return {
            "label": None,
            "confidence": None,
            "close": None,
            "vol_5d_annualised": None,
            "as_of": None,
            "status": "market_data_unavailable",
        }

    today_iso = date.today().isoformat()
    try:
        snapshot = fetch_market_snapshot(
            target_date=today_iso,
            symbol=symbol,
            lookback_days=14,
            volatility_window=5,
        )
    except Exception as exc:  # noqa: BLE001 -- graceful degradation
        _logger.warning(
            "cross_bank_snapshot_market_fetch_failed symbol=%s err=%s",
            symbol,
            exc,
        )
        return {
            "label": None,
            "confidence": None,
            "close": None,
            "vol_5d_annualised": None,
            "as_of": None,
            "status": "market_data_unavailable",
        }

    vol_5d_raw = float(snapshot.get("volatility_5d") or 0.0)
    annualised = vol_5d_raw * (252**0.5)
    if annualised < 0.12:
        label = "calm"
    elif annualised > 0.22:
        label = "high"
    else:
        label = "normal"

    # A crude "confidence": distance from the nearer band edge, clamped
    # into [0.55, 0.95]. Rendered as a confidence chip on the card.
    band_edges = (0.12, 0.22)
    nearest_edge = min(band_edges, key=lambda edge: abs(annualised - edge))
    distance = abs(annualised - nearest_edge)
    confidence = max(0.55, min(0.95, 0.55 + 4.0 * distance))

    return {
        "label": label,
        "confidence": float(round(confidence, 4)),
        "close": float(snapshot.get("close") or 0.0),
        "vol_5d_annualised": float(round(annualised, 6)),
        "as_of": str(snapshot.get("date_used") or today_iso),
        "status": "ok",
    }


def build_bank_card(
    spec: BankSpec,
    *,
    score_text: Callable[[str], dict[str, Any] | None] | None = None,
    market_lookup: Callable[[str], dict[str, Any]] | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Build one bank card. Pure function over its injected dependencies.

    ``score_text`` defaults to the lazy singleton in
    :mod:`app.services.multi_axis_classifier`; injecting a stub keeps
    the unit test offline. ``market_lookup`` is a callable
    ``symbol -> vol_regime_dict``; defaults to ``_build_vol_regime``.
    """

    if score_text is None:
        from app.services.multi_axis_classifier import score_text as _score_text

        score_text = _score_text
    if market_lookup is None:
        market_lookup = _build_vol_regime

    sentences, latest = _load_recent_sentences(spec.source, registry_path=registry_path)
    scores: list[dict[str, Any]] = []
    if sentences:
        for sentence in sentences:
            try:
                result = score_text(sentence)
            except Exception as exc:  # noqa: BLE001 -- per-sentence guard
                _logger.debug(
                    "cross_bank_snapshot_score_failed bank=%s err=%s",
                    spec.key,
                    exc,
                )
                continue
            if result:
                scores.append(result)

    stance_dist = _aggregate_distribution(scores, "stance", MULTI_TASK_STANCE_LABELS)
    stance_label, stance_conf = _argmax_label(stance_dist)

    certainty_dist = _aggregate_distribution(
        scores, "certainty", ("certain", "uncertain", "neutral")
    )
    certainty_label, certainty_conf = _argmax_label(certainty_dist)

    time_dist = _aggregate_distribution(scores, "time", ("forward looking", "not forward looking"))
    time_label, _ = _argmax_label(time_dist)

    vol_regime = market_lookup(spec.symbol)

    if not sentences:
        status = "corpus_missing"
    elif stance_dist is None:
        status = "classifier_unavailable"
    else:
        status = "ok"

    return {
        "bank": spec.key,
        "short_code": spec.short_code,
        "display_name": spec.display_name,
        "flag": spec.flag,
        "symbol": spec.symbol,
        "latest_statement_date": latest,
        "stance": stance_dist,
        "stance_label": stance_label,
        "stance_confidence": stance_conf,
        "certainty_label": certainty_label,
        "certainty_confidence": certainty_conf,
        "time_axis": time_label,
        "vol_regime_label": vol_regime.get("label"),
        "vol_regime_confidence": vol_regime.get("confidence"),
        "vol_regime_as_of": vol_regime.get("as_of"),
        "vol_regime_status": vol_regime.get("status"),
        "sample_size": len(scores),
        "status": status,
    }


def build_snapshot(
    *,
    score_text: Callable[[str], dict[str, Any] | None] | None = None,
    market_lookup: Callable[[str], dict[str, Any]] | None = None,
    registry_path: Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Public entrypoint. Builds the six-bank panel with TTL caching."""

    cache_key = "default"
    now = time.time()
    if use_cache:
        with _cache_lock:
            entry = _cache.get(cache_key)
            if entry is not None and entry.expires_at > now:
                return entry.payload

    # Double-checked locking: hold the lock across the full rebuild so a
    # post-TTL burst of concurrent requests does not fan out N parallel
    # classifier runs. Waiters re-check the cache after acquiring the
    # lock and return the freshly built payload instead of rebuilding.
    # The rebuild itself takes ~6 s cold which is acceptable to serialise;
    # subsequent hits in the same TTL window are O(ms).
    if use_cache:
        with _cache_lock:
            entry = _cache.get(cache_key)
            now = time.time()
            if entry is not None and entry.expires_at > now:
                return entry.payload
            banks = [
                build_bank_card(
                    spec,
                    score_text=score_text,
                    market_lookup=market_lookup,
                    registry_path=registry_path,
                )
                for spec in BANK_SPECS
            ]
            payload = {
                "banks": banks,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "cache_ttl_seconds": _CACHE_TTL_SECONDS,
            }
            _cache[cache_key] = _CacheEntry(payload=payload, expires_at=now + _CACHE_TTL_SECONDS)
            return payload

    # Cache disabled path (tests, ad-hoc calls): build without touching
    # the cache so a test that monkeypatches ``score_text`` cannot
    # poison the production cache for the next non-mocked request.
    banks = [
        build_bank_card(
            spec,
            score_text=score_text,
            market_lookup=market_lookup,
            registry_path=registry_path,
        )
        for spec in BANK_SPECS
    ]
    return {
        "banks": banks,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache_ttl_seconds": _CACHE_TTL_SECONDS,
    }


def reset_cache() -> None:
    """Drop the in-process snapshot cache (test hook)."""

    with _cache_lock:
        _cache.clear()
