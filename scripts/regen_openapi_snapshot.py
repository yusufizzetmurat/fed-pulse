from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _resolve_backend_root() -> Path:
    here = Path(__file__).resolve()
    container_app = Path("/app")
    if (container_app / "main.py").exists():
        return container_app
    return here.parents[1] / "backend"


def _resolve_snapshot_path() -> Path:
    here = Path(__file__).resolve()
    container_tests = Path("/app/tests")
    if container_tests.exists() and (Path("/app/main.py").exists()):
        return container_tests / "snapshots" / "openapi.json"
    return here.parents[1] / "tests" / "snapshots" / "openapi.json"


def dump_schema() -> str:
    backend_root = _resolve_backend_root()
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))
    from app.main import app

    schema = app.openapi()
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump app.main:app.openapi() as JSON.")
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Write the schema to stdout instead of tests/snapshots/openapi.json.",
    )
    args = parser.parse_args(argv)

    payload = dump_schema()
    if args.stdout:
        sys.stdout.write(payload)
        return 0
    out = _resolve_snapshot_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
