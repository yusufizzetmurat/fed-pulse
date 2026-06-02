"""Build ``stance_daily.parquet`` for the corner-B / validity loops.

The harness scripts (``stance_instrument_validity.py``,
``reverse_market_predicts_fed.py``, ``reverse_directional_followup.py``,
``corner_b_text_rates.py``) all read a per-meeting stance series at
``data/artifacts/corner_b_text_rates/stance_daily.parquet``. The
parquet carries one row per FOMC statement (``date``, ``s``) where
``s = P(hawkish) - P(dovish)`` from the multi-axis classifier.

This builder rebuilds that parquet from scratch by scoring every
statement in ``data/fomc_statements.json`` through the currently-loaded
``services.multi_axis_classifier.score_text``. Re-run after retraining
the stance head to close the improve → re-validate loop:

    python scripts/build_stance_daily.py
    python scripts/stance_instrument_validity.py

The output is deterministic up to the classifier checkpoint; a clean
run uses the active ``text_multi_axis_best.pt`` and produces 130-ish
rows over 2011-2026.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# The backend container mounts ./data at /data; on the host the same
# tree lives under <repo>/data. Resolve through app.config.DATA_DIR so
# the script works under both ``docker compose run`` (REPO_ROOT == /app)
# and host-side invocation (REPO_ROOT == repo root). The output parquet
# always lands under ``data/artifacts/`` relative to the resolved data
# directory so the consumer scripts find it without configuration.
from app.config import DATA_DIR  # noqa: E402  (sys.path mutation above)

STATEMENTS = DATA_DIR / "fomc_statements.json"
OUT = DATA_DIR / "artifacts" / "corner_b_text_rates" / "stance_daily.parquet"


def _score_one(text: str) -> float | None:
    """``s = P(hawkish) - P(dovish)`` for one statement, or None on failure."""

    from app.services.multi_axis_classifier import score_text

    block = score_text(text)
    if block is None:
        return None
    stance = block.get("stance")
    if not isinstance(stance, dict):
        return None
    distribution = stance.get("distribution")
    if not isinstance(distribution, dict):
        return None
    hawk = distribution.get("hawkish")
    dove = distribution.get("dovish")
    if not isinstance(hawk, int | float) and not isinstance(dove, int | float):
        return None
    h = float(hawk) if isinstance(hawk, int | float) else 0.0
    d = float(dove) if isinstance(dove, int | float) else 0.0
    return h - d


def main() -> int:
    import pandas as pd

    docs = json.loads(STATEMENTS.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for doc in docs:
        date = doc.get("date")
        text = doc.get("text")
        if not isinstance(date, str) or not isinstance(text, str) or not text:
            continue
        s = _score_one(text)
        if s is None:
            print(f"[skip] {date}: classifier returned no stance distribution")
            continue
        rows.append({"date": date, "s": float(s)})
        print(f"[ok]   {date}: s = {s:+.4f}")

    if not rows:
        print("[error] no rows scored; check the multi-axis classifier is loaded")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    print(f"\nwrote {len(df)} rows -> {OUT}")
    print(f"date range {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    print(f"s stats: mean {df['s'].mean():+.3f} std {df['s'].std():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
