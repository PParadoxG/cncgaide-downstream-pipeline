"""Explicit v8 owning feature modules.

Production modules here name ontology classes and own faces.  Structural
analysis that owns nothing — side-wall band discovery, Open Sweep,
transition evidence, endpoint concavity — lives in ``_shared`` and is
consumed by these modules.

``SlotStepCandidate`` - the temporary audit feature that registered raw Open
Sweep candidates before there was a Slot/Step resolver - was retired in node
28.  Ownership of an open sweep is ``Slot``'s and ``Step``'s, and the gate it
demonstrated is ``slot_step_resolution.resolve_open_sweep_candidates``.
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
    Step,
    StepBore,
    ThroughCut,
    ThroughHole,
)
