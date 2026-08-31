"""Explicit v8 owning feature modules.

Production modules here name ontology classes and own faces.  Structural
analysis that owns nothing — side-wall band discovery, Open Sweep,
transition evidence, endpoint concavity — lives in ``_shared`` and is
consumed by these modules.

``SlotStepCandidate`` remains importable only as a deprecated compatibility
module for historical focused tests and old audit readers.  It is not called
by the formal orchestrator; ownership is now provided by ``Slot`` and
``Step``.
"""

from features import (  # noqa: F401
    BlindHole,
    Boss,
    Chamfer,
    Counterbore,
    Countersink,
    Fillet,
    FreeformFace,
    PlanarFace,
    Pocket,
    Slot,
    SlotStepCandidate,
    Step,
    StepBore,
    ThroughCut,
    ThroughHole,
)
