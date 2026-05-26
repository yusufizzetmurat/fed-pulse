"""Sealed post-cutoff holdout loader (R-14, issue #333).

The canonical training package's `event_date` cutoff is 2024-12-31. Any
FOMC event after that date sits under `data/external/sealed_holdout/`
and has never been part of any sweep / val / test partition. The
sealed-once protocol queries the slice EXACTLY ONCE at final-report
time and writes the consumption event to `AUDIT_TOKEN` so the integrity
contract is reviewable on disk.

Contract:
- `load_sealed_holdout(*, audit_caller)` reads the JSONL, increments
  `AUDIT_TOKEN.usage_count`, and flips `seal_status` from `sealed` to
  `broken_by:<audit_caller>` on the first successful read. Subsequent
  calls raise `SealedHoldoutAlreadyConsumedError` unless `force=True`,
  which logs a hard warning and still increments the counter.
- `audit_status()` returns the current AUDIT_TOKEN contents as a dict.
  Read-only; calling it does not break the seal.
- Stub rows (those whose `text` starts with `# pragma: stub`) emit a
  hard warning on load so the sealed-eval headline is never silently
  published against placeholder text.

No production code in `backend/app/` outside this module is permitted
to import `load_sealed_holdout`; the audit regression test enforces
this.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import warnings
from pathlib import Path
from typing import Any

from app.config import DATA_DIR

_logger = logging.getLogger(__name__)

_SEALED_HOLDOUT_DIR = Path(DATA_DIR) / "external" / "sealed_holdout"
_AUDIT_TOKEN_PATH = _SEALED_HOLDOUT_DIR / "AUDIT_TOKEN"
_DEFAULT_JSONL = _SEALED_HOLDOUT_DIR / "fomc_2025.jsonl"

_STUB_MARKER = "# pragma: stub"


class SealedHoldoutAlreadyConsumedError(RuntimeError):
    """Raised when `load_sealed_holdout` is called after the seal has been broken.

    The sealed-once protocol allows exactly one consumption of the
    reserve slice. Subsequent calls must either pass `force=True` (which
    logs a hard warning, increments the counter, and surfaces the
    repeat in the audit trail) or be rejected outright.
    """


def _read_audit_token(path: Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else _AUDIT_TOKEN_PATH
    if not target.exists():
        # Fail-closed default: treat a missing token as a sealed slice
        # so a deleted token cannot silently unlock reads.
        return {
            "seal_status": "sealed",
            "usage_count": 0,
            "last_accessed_utc": None,
        }
    with target.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_audit_token(payload: dict[str, Any], path: Path | None = None) -> None:
    target = Path(path) if path is not None else _AUDIT_TOKEN_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def audit_status() -> dict[str, Any]:
    """Return the current AUDIT_TOKEN contents.

    Read-only. Callable from anywhere (tests, CI audit, reporting
    scripts) without breaking the seal. This is the only public hook
    safe to import from production code outside this module.
    """
    return dict(_read_audit_token())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"malformed JSON in sealed holdout file {path} at line {line_no}: {exc}"
                ) from exc
    return rows


def load_sealed_holdout(
    *,
    audit_caller: str,
    jsonl_path: Path | None = None,
    audit_token_path: Path | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Read the sealed holdout slice exactly once.

    Parameters
    ----------
    audit_caller:
        Free-form string that identifies the caller in the audit
        trail (e.g. `"final-report-eval-2026-05-27"`). Persisted to
        `AUDIT_TOKEN.seal_status` on the first successful read.
    jsonl_path / audit_token_path:
        Override hooks for tests. Defaults read from
        `data/external/sealed_holdout/`.
    force:
        Permit reads after the seal has been broken. Logs a hard
        warning and increments the counter so the repeat is visible in
        the audit trail. Used only by the one-shot break-the-seal
        operator after the integrity review has signed off.

    Raises
    ------
    SealedHoldoutAlreadyConsumedError
        When the seal has already been broken and `force=False`.
    """
    if not audit_caller or not isinstance(audit_caller, str):
        raise ValueError("audit_caller must be a non-empty string identifying the caller")

    path_jsonl = Path(jsonl_path) if jsonl_path is not None else _DEFAULT_JSONL
    path_token = Path(audit_token_path) if audit_token_path is not None else _AUDIT_TOKEN_PATH

    token = _read_audit_token(path_token)
    already_consumed = str(token.get("seal_status", "sealed")) != "sealed"

    if already_consumed and not force:
        raise SealedHoldoutAlreadyConsumedError(
            "sealed holdout has already been consumed: "
            f"seal_status={token.get('seal_status')!r}, "
            f"usage_count={token.get('usage_count')}. Pass force=True "
            "only after the break-the-seal integrity review has signed off."
        )

    if already_consumed and force:
        _logger.warning(
            "[sealed_holdout] FORCE read after seal already broken: prior=%s usage_count=%s",
            token.get("seal_status"),
            token.get("usage_count"),
        )
        warnings.warn(
            "sealed holdout read with force=True after seal already broken",
            stacklevel=2,
        )

    rows = _read_jsonl(path_jsonl)
    stub_count = sum(1 for r in rows if str(r.get("text", "")).lstrip().startswith(_STUB_MARKER))
    if stub_count:
        _logger.warning(
            "[sealed_holdout] STUB DATA: %d of %d rows carry the `%s` marker — "
            "do NOT publish a sealed-eval headline against placeholder text",
            stub_count,
            len(rows),
            _STUB_MARKER,
        )
        warnings.warn(
            f"[sealed_holdout] STUB DATA: {stub_count}/{len(rows)} rows are placeholder stubs",
            stacklevel=2,
        )

    now_utc = _dt.datetime.now(_dt.timezone.utc).isoformat()
    new_token = {
        "seal_status": f"broken_by:{audit_caller}",
        "usage_count": int(token.get("usage_count", 0)) + 1,
        "last_accessed_utc": now_utc,
    }
    _write_audit_token(new_token, path_token)
    return rows


__all__ = [
    "SealedHoldoutAlreadyConsumedError",
    "audit_status",
    "load_sealed_holdout",
]
