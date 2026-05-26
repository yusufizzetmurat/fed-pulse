from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Repo root so ``from scripts.<module> import …`` resolves; ``scripts/``
# carries an ``__init__.py`` marker so it imports as a regular package.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (skip via -m 'not slow')",
    )
    config.addinivalue_line(
        "markers",
        "regression: marks tests that lock in the canonical training / serving contracts (run via -m regression)",
    )
