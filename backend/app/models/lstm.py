"""Legacy module shim (issue #336).

Pre-#336 this module hosted the 712-line ``ForecasterModel`` that
carried every research knob and every serving back-compat path on the
same class. Issue #336 split that into
:class:`app.models.research_model.ForecasterResearchModel` (research)
and :class:`app.models.serving_model.ForecasterServingModel` (frozen
serving surface). The shared backbone + input-prep lives on
:class:`app.models.forecaster_base.ForecasterBase`.

This module survives only as an import shim so external callers that
imported ``app.models.lstm.ForecasterModel`` keep working through a
deprecation window. New code should import directly from
``app.models.research_model`` or ``app.models.serving_model`` -- the
shim will be removed in a follow-up issue once the in-repo importers
have been migrated.
"""

from __future__ import annotations

from app.models.forecaster_base import (
    ForecasterBase,
    prepare_recurrent_input,
)
from app.models.research_model import ForecasterResearchModel

# Deprecation alias. ``ForecasterModel`` is now a re-export of the
# research class; existing checkpoints serialise the same state_dict
# key names (the backbone keys come from ForecasterBase and the head
# keys are identical to the legacy layout) so torch.load against the
# alias still loads pre-#336 artefacts. New code should import
# ``ForecasterResearchModel`` directly.
ForecasterModel = ForecasterResearchModel

__all__ = [
    "ForecasterBase",
    "ForecasterModel",
    "ForecasterResearchModel",
    "prepare_recurrent_input",
]
