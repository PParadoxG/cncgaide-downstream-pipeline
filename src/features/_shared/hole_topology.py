"""Compatibility surface: the rim walk now lives in ``topology``.

Node 28 moved these functions out of the v5.1 rule set, because the NX line
reads them too and importing one of them dragged every v5.1 leaf module into
NX's bundled Python (Node 26 finding C-10).  The v5.1 leaf modules keep their
existing import lines - rewriting four hundred of them buys nothing - so this
module stays as the name they know.

New code imports from ``topology.hole_topology`` / ``topology.signatures`` /
``topology.vectors`` directly.
"""

from __future__ import annotations

from topology.hole_topology import (  # noqa: F401
    HoleEnd,
    RimGroup,
    classify_rim_end,
    cylindrical_wall_axial_extent,
    is_reversed_cylindrical_wall,
    wall_rims,
)
from topology.signatures import (  # noqa: F401
    SignatureView,
    carries_length as _carries_length,
    edge_signature as _edge_signature,
)
from topology.vectors import dot, unit_vector, valid_vector  # noqa: F401

__all__ = [
    "HoleEnd",
    "RimGroup",
    "SignatureView",
    "classify_rim_end",
    "cylindrical_wall_axial_extent",
    "dot",
    "is_reversed_cylindrical_wall",
    "unit_vector",
    "valid_vector",
    "wall_rims",
]
