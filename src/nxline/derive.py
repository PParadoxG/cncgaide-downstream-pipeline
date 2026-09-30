"""Tier-E measurements: the ``nxd:`` derived block of an NX-line node.

Node 19' froze the shape — a node's top level stays pure NX (name and value
exactly as published), and anything the *pipeline* measured about the feature
goes in a separate ``derived`` block, one entry per quantity, each carrying
``derivedBy: nx_kernel``.  ``ontology/nx_v1/source/derived.yaml`` declares the
three quantities; this module is what materialises them, inside the journal
process where the kernel is alive.  The seconds-tier heuristic recompute then
reads the block back off disk and never needs a session.

Which nodes are owed a measurement is the ontology's answer, not this
module's: a quantity's domain is the union of the ``applies_to`` patterns of
the heuristics that read it (``derived_domain``).  Widening a domain in the
YAML therefore *declares* a debt this code only repays on the next full
rebuild of the part — until then the evaluation gate reports the missing
measurement as a named NOT_EVALUABLE, which is the honest state.

Refusal is a first-class answer everywhere: no axis, no floor face, no
kernel, or the kernel's own ``None`` each produce an entry whose ``value`` is
null and whose ``reason`` names the gap, so a verdict downstream can say
exactly why it could not be reached.
"""

from __future__ import annotations

from fnmatch import fnmatchcase

from tbox import nx_v1 as source
from topology.faces import cad_face_ids
from relations.nx._shared import axis, floor_faces, lateral_faces

DERIVED_BY = "nx_kernel"

FLOOR = "derived_floor_thickness"
MIN_WALL = "derived_min_wall_thickness"
MIN_CLEARANCE = "derived_min_clearance"


def _in_domain(feature_type: str, patterns) -> bool:
    return any(fnmatchcase(feature_type, pattern) for pattern in patterns)


def _entry(value, *, method=None, evidence=None, reason=None) -> dict:
    entry = {"value": value, "derivedBy": DERIVED_BY}
    if method:
        entry["method"] = method
    if evidence:
        entry["evidence"] = evidence
    if reason:
        entry["reason"] = reason
    return entry


def _refusal(reason: str) -> dict:
    return _entry(None, reason=reason)


def _provenance(kernel, key: str) -> str | None:
    """Which kernel route produced *key*'s value.

    Not wrapped in a ``try``: ``KernelServices.provenance`` is a named seam
    method whose refusal is ``None``, so an exception out of it is a bug in
    the implementation, not a way of saying "unsupported" (node 26E finding
    E-16).  Letting it out makes the part a named error with a traceback
    instead of a ``derived`` entry that quietly lost its ``method``.
    """
    return kernel.provenance(key)


def _floor_thickness(model, kernel, node, direction) -> dict:
    floors = floor_faces(model, node, direction)
    if not floors:
        return _refusal("no face of this feature faces back toward the entry - "
                        "nothing qualifies as its floor")
    value = kernel.remaining_floor_thickness(
        cad_face_ids(model, list(node.partmodelFaceIds)),
        cad_face_ids(model, floors),
        direction,
    )
    if value is None:
        return _refusal("no reliable first opposite-surface hit was found below the floor")
    return _entry(round(float(value), 6),
                  method=_provenance(kernel, "remaining_floor_thickness"),
                  evidence={"floorFaceIds": floors,
                            "accessDirection": [round(v, 6) for v in direction]})


def _wall_thickness(model, kernel, node, direction, *, material_side: bool,
                    missing_reason: str, unreached_reason: str) -> dict:
    walls = lateral_faces(model, node, direction)
    if not walls:
        return _refusal(missing_reason)
    diagnostics: dict = {}
    value = kernel.min_wall_thickness(
        cad_face_ids(model, list(node.partmodelFaceIds)),
        cad_face_ids(model, walls),
        material_side=material_side,
        diagnostics=diagnostics,
    )
    if value is None:
        return _refusal(unreached_reason)
    return _entry(round(float(value), 6),
                  method=_provenance(kernel, "min_wall_thickness"),
                  evidence={"wallFaceIds": walls, **diagnostics})


def derive_nodes(model, nodes, kernel) -> dict:
    """Fill each in-domain node's ``derived`` block; return a small summary.

    Mutates ``node.derived`` in place (the block travels with the node into
    ``result.json``) and answers ``{quantity: {measured, refused}}`` for the
    projection diagnostics.
    """
    domains = {name: source.derived_domain(name)
               for name in (FLOOR, MIN_WALL, MIN_CLEARANCE)}
    summary = {name: {"measured": 0, "refused": 0} for name in domains}

    for node in nodes:
        owed = [name for name, patterns in domains.items()
                if _in_domain(node.featureType, patterns)]
        if not owed:
            continue
        direction = axis(node)
        for name in owed:
            if kernel is None:
                entry = _refusal("no geometry kernel was available in this process")
            elif direction is None:
                entry = _refusal("the feature publishes no usable access direction "
                                 "(X/Y/Z_ORIENTATION_D)")
            elif name == FLOOR:
                entry = _floor_thickness(model, kernel, node, direction)
            elif name == MIN_WALL:
                entry = _wall_thickness(
                    model, kernel, node, direction, material_side=True,
                    missing_reason="no face of this feature runs along its axis - "
                                   "nothing qualifies as a side wall",
                    unreached_reason="no non-self material surface was reached from "
                                     "sampled side-wall normals")
            else:
                entry = _wall_thickness(
                    model, kernel, node, direction, material_side=False,
                    missing_reason="no face of this feature runs along its axis - "
                                   "nothing qualifies as a side wall",
                    unreached_reason="no other solid surface was reached from sampled "
                                     "outward wall normals")
            node.derived[name] = entry
            summary[name]["measured" if entry["value"] is not None else "refused"] += 1

    return summary
