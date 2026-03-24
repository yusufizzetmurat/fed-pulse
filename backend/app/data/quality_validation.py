from __future__ import annotations

"""
Canonical capability-first entrypoint for data quality validation.
"""

from app.data.quality_checks import main


if __name__ == "__main__":
    raise SystemExit(main())
