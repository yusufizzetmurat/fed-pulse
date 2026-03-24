from __future__ import annotations

"""
Canonical capability-first entrypoint for source ingestion.

This module intentionally delegates to the existing implementation
to keep behavior identical while standardizing command names.
"""

from app.data.ingest_sources import main


if __name__ == "__main__":
    raise SystemExit(main())
