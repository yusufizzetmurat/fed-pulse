"""Hawkish/dovish trajectory model (#296).

Sequence-of-meetings model over per-meeting encoder embeddings plus
market context features. Distinct from the per-statement multi-axis
classifier — different time scale (one decision per FOMC meeting rather
than one decision per sentence) and a different mechanism (sequence
model over the last N meetings rather than per-statement projection).

Layout mirrors :mod:`app.retrieval`:

* :mod:`app.trajectory.model` — two architectures (LSTM baseline +
  small Transformer) plus the persistence helpers. The architecture
  comparison is the methodological point per §3 Panel 4 of the
  finalization roadmap.
* :mod:`app.trajectory.train` — walk-forward training entry point that
  respects the ``fold_manifest_expanding_walk_forward.json`` boundary
  and emits a bundle under ``data/artifacts/trajectory/<run_name>/``.

The runtime singleton at :mod:`app.services.trajectory` mounts the
bundle for the FastAPI ``/analyze/trajectory`` handler.
"""

from __future__ import annotations
