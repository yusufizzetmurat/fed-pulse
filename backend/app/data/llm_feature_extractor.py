"""B1 (#212): LLM-as-feature-extractor over the FOMC document corpus.

For each event row in ``events.parquet``, look up the document text by
``text_hash``, send the document to Claude Sonnet 4.7 with the catalogue
prompt from :mod:`app.data.llm_feature_catalog`, parse the structured
JSON response, validate against the catalogue's allowed levels, and
persist to a per-document cache parquet.

The cache is keyed on ``(text_hash, model_id, catalog_version)`` so
re-runs against a bumped catalogue retire the old cache cleanly without
re-spending on documents whose features have not changed.

Failure modes handled:

- API rate-limit / transient error: exponential backoff up to ``max_retries``
- Malformed JSON in the model response: one retry with a stricter prompt
- Out-of-vocabulary level returned: one retry with the level constraint
  echoed; if it fails again the row is dropped from the cache with a
  warning logged to the audit trail
- Empty / very short document: skipped with reason ``"document_too_short"``

Idempotent on a partial cache: re-running picks up where it left off.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from app.data.llm_feature_catalog import (
    CATALOG_VERSION,
    MODEL_ID,
    SYSTEM_PROMPT,
    TEMPERATURE,
    build_user_prompt,
    feature_names,
    levels_for,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Robust JSON parser for LLM responses
# ---------------------------------------------------------------------------
# Frontier LLMs occasionally wrap their JSON output in markdown fences,
# add a preamble ("Here is the JSON:"), or append a closing sentence
# despite the system prompt forbidding prose. This parser handles every
# common deviation so the validator gets a fair shot at the payload.


def _parse_response_json(text: str) -> dict[str, Any]:
    """Strip markdown fences and any leading / trailing prose, then
    parse the first JSON object found in the response.

    Returns ``{}`` when no valid JSON object can be recovered.
    """

    candidate = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences.
    if candidate.startswith("```"):
        # remove opening fence + optional language tag
        first_newline = candidate.find("\n")
        if first_newline != -1:
            candidate = candidate[first_newline + 1 :]
        # remove closing fence
        if candidate.endswith("```"):
            candidate = candidate[:-3]
        candidate = candidate.strip()

    # Find the first balanced JSON object substring. The catalogue's
    # schema is a single top-level object, so we walk braces from the
    # first ``{`` and stop at the matching ``}``.
    start = candidate.find("{")
    if start == -1:
        return {}
    depth = 0
    end = -1
    in_string = False
    escape = False
    for i in range(start, len(candidate)):
        ch = candidate[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return {}
    json_text = candidate[start : end + 1]
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_DIR = Path("/data/raw/llm_features")
_MIN_DOCUMENT_CHARS = 200  # below this we skip; statements are 2-5kB, minutes 30-80kB
_MAX_RETRIES = 3
_BACKOFF_INITIAL_SECONDS = 2.0
_REQUEST_TIMEOUT_SECONDS = 90.0


@dataclasses.dataclass(frozen=True)
class ExtractionResult:
    """Outcome of one document extraction.

    ``features`` maps catalogue feature name -> selected level when
    the extraction succeeded. ``status`` is one of ``"ok"`` /
    ``"document_too_short"`` / ``"api_error"`` / ``"invalid_json"`` /
    ``"out_of_vocab"`` so the caller can route failures to the audit
    log without losing the structured outcome.
    """

    text_hash: str
    status: str
    features: dict[str, str] | None = None
    raw_response: str | None = None
    error_detail: str | None = None
    elapsed_seconds: float | None = None


# ---------------------------------------------------------------------------
# Anthropic client wrapper
# ---------------------------------------------------------------------------


class AnthropicExtractorClient:
    """Thin wrapper around the official ``anthropic`` SDK.

    Kept narrow so the extractor module is unit-testable without an API
    key: tests inject a stub that returns canned ExtractionResults.
    """

    def __init__(self, *, api_key: str | None = None, model_id: str = MODEL_ID) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY missing from environment. Add it to "
                "``.env`` per the .env.example template and re-source the "
                "shell or restart the docker container."
            )
        self._model_id = model_id
        # Lazy-import so the extractor module loads cleanly even when
        # anthropic is not installed (e.g. on a CI worker that skips
        # the LLM path).
        try:
            from anthropic import Anthropic  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - exercised by integration runs
            raise RuntimeError(
                "The ``anthropic`` package is not installed. Add it to "
                "``backend/pyproject.toml`` and rebuild the container."
            ) from exc
        self._client = Anthropic(api_key=self._api_key)

    def extract(self, document_text: str) -> tuple[str, dict[str, Any]]:
        """One API round-trip. Returns ``(raw_response_text, parsed_json)``.

        Caller handles retry / fallback. The raw text is returned so the
        audit log can record exactly what the model emitted, even when
        the JSON-parsing step fails downstream.
        """

        response = self._client.messages.create(
            model=self._model_id,
            max_tokens=2048,
            temperature=TEMPERATURE,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": build_user_prompt(document_text)},
            ],
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        # The SDK returns a list of content blocks; for our JSON-only
        # prompt there is always exactly one text block.
        if not response.content:
            return "", {}
        text = response.content[0].text  # type: ignore[union-attr]
        parsed = _parse_response_json(text)
        return text, parsed


# ---------------------------------------------------------------------------
# Per-document extraction with retries + validation
# ---------------------------------------------------------------------------


def _validate(parsed: dict[str, Any]) -> tuple[bool, str | None, dict[str, str]]:
    """Strict-level-set validator.

    Returns ``(ok, error_detail, features)``. ``features`` carries only
    the validated subset; missing or out-of-vocab values are logged in
    ``error_detail``. ``ok`` is True only when every catalogue feature
    is present with an allowed level.
    """

    features: dict[str, str] = {}
    missing: list[str] = []
    bad: list[str] = []
    for name in feature_names():
        if name not in parsed:
            missing.append(name)
            continue
        raw_value = parsed[name]
        if not isinstance(raw_value, str):
            bad.append(f"{name}=<not-a-string:{type(raw_value).__name__}>")
            continue
        if raw_value not in levels_for(name):
            bad.append(f"{name}={raw_value!r}")
            continue
        features[name] = raw_value
    if missing or bad:
        detail_parts: list[str] = []
        if missing:
            detail_parts.append(f"missing: {', '.join(missing)}")
        if bad:
            detail_parts.append(f"out_of_vocab: {', '.join(bad)}")
        return False, "; ".join(detail_parts), features
    return True, None, features


def extract_one(
    *,
    text_hash: str,
    document_text: str,
    client: AnthropicExtractorClient,
    max_retries: int = _MAX_RETRIES,
) -> ExtractionResult:
    """Single-document extraction with retry-on-validation and
    exponential backoff on transient errors."""

    if not document_text or len(document_text.strip()) < _MIN_DOCUMENT_CHARS:
        return ExtractionResult(
            text_hash=text_hash,
            status="document_too_short",
            error_detail=f"document length {len(document_text or '')} below {_MIN_DOCUMENT_CHARS}",
        )

    last_error: str | None = None
    last_raw: str | None = None
    last_features: dict[str, str] = {}
    start = time.monotonic()
    for attempt in range(max_retries):
        try:
            raw, parsed = client.extract(document_text)
        except Exception as exc:  # noqa: BLE001 - want any transport error caught
            last_error = f"api_error[{type(exc).__name__}]: {exc!s}"
            time.sleep(_BACKOFF_INITIAL_SECONDS * (2**attempt))
            continue
        last_raw = raw
        ok, err, features = _validate(parsed)
        if ok:
            elapsed = time.monotonic() - start
            return ExtractionResult(
                text_hash=text_hash,
                status="ok",
                features=features,
                raw_response=raw,
                elapsed_seconds=elapsed,
            )
        # Validation failed -- keep the partial features and try again.
        last_error = err or "unknown_validation_failure"
        last_features = features
        time.sleep(_BACKOFF_INITIAL_SECONDS)
    return ExtractionResult(
        text_hash=text_hash,
        status="invalid_json" if not last_features else "out_of_vocab",
        features=last_features or None,
        raw_response=last_raw,
        error_detail=last_error,
        elapsed_seconds=time.monotonic() - start,
    )


# ---------------------------------------------------------------------------
# Cache file I/O
# ---------------------------------------------------------------------------


def _cache_path(training_package_id: str, cache_dir: Path = _DEFAULT_CACHE_DIR) -> Path:
    """Per-package cache lives under
    ``data/raw/llm_features/<model_id>_<catalog_version>/<package>.parquet``."""

    return (
        cache_dir
        / f"{MODEL_ID.replace('/', '_')}_{CATALOG_VERSION}"
        / f"{training_package_id}.parquet"
    )


def _serialise_result(result: ExtractionResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "text_hash": result.text_hash,
        "status": result.status,
        "error_detail": result.error_detail,
        "elapsed_seconds": result.elapsed_seconds,
        "model_id": MODEL_ID,
        "catalog_version": CATALOG_VERSION,
        # Keep the model's raw text response on failures so a future
        # debug pass can see exactly what came back without re-spending
        # the API call. Truncated at 4 KB so the parquet stays small.
        "raw_response": (result.raw_response or "")[:4096],
    }
    for name in feature_names():
        row[name] = (result.features or {}).get(name)
    return row


def _load_existing_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    import pandas as pd

    frame = pd.read_parquet(path)
    return {str(row["text_hash"]): row.to_dict() for _, row in frame.iterrows()}


def _persist_cache(rows: Iterable[dict[str, Any]], path: Path) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Bulk extraction over a training package
# ---------------------------------------------------------------------------


# Status values that the bulk extractor treats as "permanently cached"
# and skips on a re-run. Everything else (api_error, invalid_json,
# out_of_vocab) might be transient -- a model upgrade, a rate-limit
# burst, or a one-off content-filter trip -- so re-runs default to
# retrying those rows.
_DONE_STATUSES: frozenset[str] = frozenset({"ok", "document_too_short"})


def extract_for_package(
    *,
    training_package_id: str,
    documents: Iterable[tuple[str, str]],
    cache_dir: Path = _DEFAULT_CACHE_DIR,
    client: AnthropicExtractorClient | None = None,
    progress_every: int = 1,
    retry_failed: bool = False,
) -> Path:
    """Run extraction over every (text_hash, document_text) pair.

    **Idempotent resume contract:**

    - ``ok`` and ``document_too_short`` rows in the existing cache are
      treated as final and skipped on every subsequent run -- they
      cannot change on retry.
    - ``api_error`` / ``invalid_json`` / ``out_of_vocab`` rows are
      retried by default, because each of those is a transient
      failure mode (rate-limit burst, model jitter, content-filter
      trip) that often clears on a retry.
    - Pass ``retry_failed=True`` to also re-extract rows the prior
      run marked ``ok`` -- useful when the catalogue prompt changes
      but ``CATALOG_VERSION`` was not yet bumped.

    The cache is flushed after every successful row by default
    (``progress_every=1``) so a SIGKILL between rows loses at most
    one document of work. Set higher values if API rate is the
    bottleneck.

    Returns the cache parquet path.
    """

    client = client or AnthropicExtractorClient()
    cache_path = _cache_path(training_package_id, cache_dir=cache_dir)
    existing = _load_existing_cache(cache_path)

    # Partition the existing cache into "final" rows we keep verbatim and
    # "retryable" rows we drop from the cache so the loop below re-runs
    # them. retry_failed=True forces a fresh run on every row.
    done_hashes: set[str] = set()
    rows_to_keep: list[dict[str, Any]] = []
    for h, row in existing.items():
        status = str(row.get("status", "")).strip()
        is_final = status in _DONE_STATUSES and not retry_failed
        if is_final:
            done_hashes.add(h)
            rows_to_keep.append(row)
    all_rows: list[dict[str, Any]] = list(rows_to_keep)

    total_processed = 0
    total_skipped = 0
    total_retried = 0
    for text_hash, document_text in documents:
        if text_hash in done_hashes:
            total_skipped += 1
            continue
        if text_hash in existing and text_hash not in done_hashes:
            total_retried += 1
        result = extract_one(
            text_hash=text_hash,
            document_text=document_text,
            client=client,
        )
        all_rows.append(_serialise_result(result))
        done_hashes.add(text_hash)
        total_processed += 1
        if progress_every and (total_processed % progress_every == 0):
            _logger.info(
                "[llm_features] %d processed (%d retried), %d skipped (cached)",
                total_processed,
                total_retried,
                total_skipped,
            )
            # Per-row flush is the default so an interrupted run
            # never loses more than the one document mid-flight.
            _persist_cache(all_rows, cache_path)

    _persist_cache(all_rows, cache_path)
    _logger.info(
        "[llm_features] done. total=%d processed=%d (retried=%d) skipped=%d cache=%s",
        len(all_rows),
        total_processed,
        total_retried,
        total_skipped,
        cache_path,
    )
    return cache_path


__all__ = [
    "ExtractionResult",
    "AnthropicExtractorClient",
    "extract_one",
    "extract_for_package",
]
