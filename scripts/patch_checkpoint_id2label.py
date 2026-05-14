"""One-shot post-hoc fix for a fine-tuned checkpoint whose ``id2label`` was
written with the inverse of the Trillion Dollar Words canonical class
ordering (the 2026-05-14 root-cause bug — see
``backend/app/data/normalize_labels.py`` for the audit notes and TDW
sample verification).

The model on disk learned the correct CLASS DISTINCTIONS but with the
wrong NAMES attached to class indices 1 and 2. After running this
script the model's output indices are unchanged; only the human-
readable label names attached to those indices are corrected.

Usage:

    docker compose --profile gpu run --rm backend-gpu \\
        python -m scripts.patch_checkpoint_id2label \\
        /data/artifacts/phase3/pilot_finetune_20260505T142652Z/hf_checkpoints

The script edits ``config.json`` in place. It refuses to act if the
config doesn't match the known inverted shape, so re-running is a
no-op.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

INVERTED_ID2LABEL = {"0": "dovish", "1": "neutral", "2": "hawkish"}
CORRECT_ID2LABEL = {"0": "dovish", "1": "hawkish", "2": "neutral"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        help="Path to the HF checkpoint directory containing config.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the diff without writing.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.checkpoint_dir / "config.json"
    if not config_path.exists():
        print(f"config.json not found at {config_path}", file=sys.stderr)
        return 1

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    current_id2label = {str(k): v for k, v in (payload.get("id2label") or {}).items()}

    if current_id2label == CORRECT_ID2LABEL:
        print(f"[patch] {config_path} already canonical — no changes needed.")
        return 0
    if current_id2label != INVERTED_ID2LABEL:
        print(
            f"[patch] refusing to act: current id2label={current_id2label!r} "
            f"does not match the known-inverted shape {INVERTED_ID2LABEL!r} "
            f"or canonical {CORRECT_ID2LABEL!r}.",
            file=sys.stderr,
        )
        return 2

    payload["id2label"] = CORRECT_ID2LABEL
    payload["label2id"] = {label: int(idx) for idx, label in CORRECT_ID2LABEL.items()}

    if args.dry_run:
        print("[patch] dry-run — would write:")
        print(f"  id2label: {INVERTED_ID2LABEL} -> {CORRECT_ID2LABEL}")
        print(f"  label2id: {payload['label2id']}")
        return 0

    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[patch] wrote canonical id2label to {config_path}")
    print(f"  id2label: {CORRECT_ID2LABEL}")
    print(f"  label2id: {payload['label2id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
