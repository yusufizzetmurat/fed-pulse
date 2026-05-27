"""Regression: sealed post-cutoff holdout integrity (R-14, issue #333).

Three contracts:

1. `AUDIT_TOKEN.seal_status` is `sealed` on import-time inspection. The
   committed token reflects the unsealed-CI state of the repo; if a
   merge ever ships a `broken_by:*` token the build fails so a stale
   one-shot read cannot reach `main` unnoticed.
2. `audit_status()` is callable from anywhere and does not mutate
   on-disk state — the audit hook is safe to import from production
   code (CI dashboards, reporting scripts) without breaking the seal.
3. No production code under `backend/app/` (outside
   `sealed_holdout_loader.py` itself) imports `load_sealed_holdout`.
   The grep-style assertion fails the build the moment a non-test
   module reaches for the sealed reader.

A fourth coverage layer exercises the loader end-to-end against an
isolated tmp_path AUDIT_TOKEN: first call returns rows + flips the
seal, second call raises `SealedHoldoutAlreadyConsumedError`, `force=True`
permits a logged repeat. This locks the contract the doc + ADR
narrative point at.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.data.sealed_holdout_loader import (
    SealedHoldoutAlreadyConsumedError,
    audit_status,
    load_sealed_holdout,
)

pytestmark = pytest.mark.regression


REPO_ROOT = Path(__file__).resolve().parents[2]
SEALED_DIR = REPO_ROOT / "data" / "external" / "sealed_holdout"
AUDIT_TOKEN_PATH = SEALED_DIR / "AUDIT_TOKEN"
DEFAULT_JSONL = SEALED_DIR / "fomc_2025.jsonl"


def test_committed_audit_token_is_sealed() -> None:
    assert AUDIT_TOKEN_PATH.exists(), "AUDIT_TOKEN file must ship with the repo"
    payload = json.loads(AUDIT_TOKEN_PATH.read_text(encoding="utf-8"))
    assert payload["seal_status"] == "sealed", (
        f"committed AUDIT_TOKEN must read seal_status=sealed; got {payload['seal_status']!r}. "
        "A broken_by:* token in main would mean the one-shot has already fired against this repo."
    )
    assert payload["usage_count"] == 0
    assert payload["last_accessed_utc"] is None


def test_committed_holdout_jsonl_has_min_entries() -> None:
    assert DEFAULT_JSONL.exists(), "fomc_2025.jsonl must ship with the repo"
    rows = [
        json.loads(line) for line in DEFAULT_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(rows) >= 4, f"sealed holdout must carry >= 4 entries; got {len(rows)}"
    required_fields = {"event_date", "event_type", "text", "url", "scraped_at_utc"}
    for row in rows:
        missing = required_fields - set(row.keys())
        assert not missing, f"row {row.get('event_date')!r} missing fields: {missing}"


def test_audit_status_is_side_effect_free(tmp_path: Path) -> None:
    """Calling `audit_status()` must not mutate any committed state."""
    before = AUDIT_TOKEN_PATH.read_text(encoding="utf-8")
    snapshot = audit_status()
    after = AUDIT_TOKEN_PATH.read_text(encoding="utf-8")

    assert before == after, "audit_status() must not write to AUDIT_TOKEN"
    assert snapshot["seal_status"] == "sealed"
    assert snapshot["usage_count"] == 0


def _write_isolated_fixture(tmp_path: Path) -> tuple[Path, Path]:
    jsonl = tmp_path / "fomc_test.jsonl"
    token = tmp_path / "AUDIT_TOKEN"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_date": f"2025-01-{day:02d}",
                        "event_type": "statement",
                        "text": f"Sealed test row {day}",
                        "url": f"https://example.invalid/{day}",
                        "scraped_at_utc": "2026-05-27T00:00:00Z",
                    }
                )
                for day in (1, 2, 3, 4)
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    token.write_text(
        json.dumps({"seal_status": "sealed", "usage_count": 0, "last_accessed_utc": None}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return jsonl, token


def test_load_flips_seal_and_increments_counter(tmp_path: Path) -> None:
    jsonl, token = _write_isolated_fixture(tmp_path)

    rows = load_sealed_holdout(
        audit_caller="regression-test-first-call",
        jsonl_path=jsonl,
        audit_token_path=token,
    )
    assert len(rows) == 4
    payload = json.loads(token.read_text(encoding="utf-8"))
    assert payload["seal_status"] == "broken_by:regression-test-first-call"
    assert payload["usage_count"] == 1
    assert payload["last_accessed_utc"] is not None


def test_second_load_raises_already_consumed(tmp_path: Path) -> None:
    jsonl, token = _write_isolated_fixture(tmp_path)

    load_sealed_holdout(
        audit_caller="regression-test-first",
        jsonl_path=jsonl,
        audit_token_path=token,
    )
    with pytest.raises(SealedHoldoutAlreadyConsumedError):
        load_sealed_holdout(
            audit_caller="regression-test-second",
            jsonl_path=jsonl,
            audit_token_path=token,
        )

    payload = json.loads(token.read_text(encoding="utf-8"))
    assert payload["usage_count"] == 1, "rejected re-read must NOT bump the counter"


def test_force_repeat_logs_warning_and_increments(tmp_path: Path, recwarn) -> None:
    jsonl, token = _write_isolated_fixture(tmp_path)

    load_sealed_holdout(
        audit_caller="first",
        jsonl_path=jsonl,
        audit_token_path=token,
    )
    rows = load_sealed_holdout(
        audit_caller="force-repeat",
        jsonl_path=jsonl,
        audit_token_path=token,
        force=True,
    )
    assert len(rows) == 4
    payload = json.loads(token.read_text(encoding="utf-8"))
    assert payload["usage_count"] == 2
    assert payload["seal_status"] == "broken_by:force-repeat"
    # warnings module emitted at least one warn — the loader's hard force warning
    assert any("force=True" in str(w.message) for w in recwarn)


def test_empty_audit_caller_rejected(tmp_path: Path) -> None:
    jsonl, token = _write_isolated_fixture(tmp_path)
    with pytest.raises(ValueError, match="audit_caller"):
        load_sealed_holdout(audit_caller="", jsonl_path=jsonl, audit_token_path=token)


def test_no_production_code_imports_load_sealed_holdout() -> None:
    """Grep-style: only `sealed_holdout_loader.py` may import `load_sealed_holdout`.

    Production code under `backend/app/` must reach the sealed slice
    through `audit_status()` (read-only) or via an explicit, audited
    one-shot script outside `backend/app/`. The moment any non-loader
    module imports the loader's read symbol, this test fires.
    """
    backend_app = REPO_ROOT / "backend" / "app"
    assert backend_app.exists(), f"backend/app/ not found at {backend_app}"

    pattern = re.compile(r"\bload_sealed_holdout\b")
    offenders: list[Path] = []
    for py_file in backend_app.rglob("*.py"):
        if py_file.name == "sealed_holdout_loader.py":
            continue
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        if pattern.search(text):
            offenders.append(py_file.relative_to(REPO_ROOT))

    assert not offenders, (
        "no production code under backend/app/ may import load_sealed_holdout; "
        f"offenders: {[str(p) for p in offenders]}"
    )
