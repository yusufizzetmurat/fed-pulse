"""Historical analog retrieval (#294).

Holds the sentence-transformer fine-tune driver
(:mod:`app.retrieval.train`) plus the on-disk index used by
``/analyze/analogs`` (:mod:`app.retrieval.index`). The runtime singleton
that wires the two together for the FastAPI handler lives at
:mod:`app.services.analogs`.
"""

from __future__ import annotations
