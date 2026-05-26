from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "interim" / "toy_snapshot"


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def build_toy_snapshot(
    *,
    n_events: int,
    out_dir: Path = DEFAULT_OUT,
    statements_path: Path = REPO_ROOT / "data" / "fomc_statements.json",
    minutes_path: Path = REPO_ROOT / "data" / "fomc_minutes.json",
) -> Path:
    statements = _load_json_array(statements_path)
    minutes = _load_json_array(minutes_path)
    pool = statements + minutes
    pool.sort(key=lambda item: str(item.get("date", "")))
    slice_ = pool[: max(1, n_events)]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fomc_statements.json").write_text(
        json.dumps(slice_, indent=2), encoding="utf-8"
    )
    (out_dir / "fomc_minutes.json").write_text("[]", encoding="utf-8")
    print(f"[toy_snapshot] wrote {len(slice_)} rows to {out_dir}")
    return out_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tiny FOMC snapshot for make verify.")
    parser.add_argument("--n-events", type=int, default=50)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    build_toy_snapshot(n_events=args.n_events, out_dir=Path(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
