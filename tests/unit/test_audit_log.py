from __future__ import annotations

from pathlib import Path

from app.audit import append_audit_entry, hash_file, read_audit_entries


def test_append_audit_entry_writes_jsonl(tmp_path: Path):
    audit_path = tmp_path / "audit.log"
    entry = append_audit_entry(
        "checkpoint_saved",
        run_id="abc",
        before_hash="aaa",
        after_hash="bbb",
        metadata={"path": "models/forecaster_best.pt"},
        audit_path=audit_path,
    )
    assert entry["action"] == "checkpoint_saved"
    rows = read_audit_entries(audit_path)
    assert len(rows) == 1
    assert rows[0]["action"] == "checkpoint_saved"
    assert rows[0]["metadata"]["path"] == "models/forecaster_best.pt"


def test_append_audit_entry_appends_multiple_lines(tmp_path: Path):
    audit_path = tmp_path / "audit.log"
    append_audit_entry("one", audit_path=audit_path, run_id="r1")
    append_audit_entry("two", audit_path=audit_path, run_id="r2")
    rows = read_audit_entries(audit_path)
    assert [row["action"] for row in rows] == ["one", "two"]


def test_hash_file_returns_sha256(tmp_path: Path):
    target = tmp_path / "x.bin"
    target.write_bytes(b"hello world")
    digest = hash_file(target)
    assert digest == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_hash_file_returns_none_for_missing(tmp_path: Path):
    assert hash_file(tmp_path / "nope") is None
