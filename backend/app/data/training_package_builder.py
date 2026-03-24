from __future__ import annotations

"""
Canonical capability-first entrypoint for building training packages.
"""

from app.data.build_training_package import main


if __name__ == "__main__":
    raise SystemExit(main())
