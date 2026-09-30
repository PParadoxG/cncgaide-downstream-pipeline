"""Compatibility surface: these helpers now live in ``topology``.

They read a PartModel and standard-library arithmetic and nothing else, and
both lines use them (``nxline.derive`` and both relation dialects call
``cad_face_ids``), so Node 28 moved them out of the v5.1 rule set.  The leaf
modules keep importing them from here.
"""

from __future__ import annotations

from topology.faces import (  # noqa: F401
    cad_face_ids,
    first_face_bbox_center,
    serialize_cad_face_ids,
    valid_bbox,
)
from topology.vectors import canonical_axis, rounded_scalar  # noqa: F401

__all__ = [
    "cad_face_ids",
    "canonical_axis",
    "first_face_bbox_center",
    "rounded_scalar",
    "serialize_cad_face_ids",
    "valid_bbox",
]
