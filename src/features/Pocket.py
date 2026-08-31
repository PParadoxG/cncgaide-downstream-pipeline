"""Pocket recognition — the ontology ``Pocket`` class.

This module is deliberately thin.  Every hard part of the problem is a shared
capability consumed from ``_shared``:

- ``closed_wall_band``  — where the closed side-wall bands are, and the
  root-anchored lineage that tells incoming from outgoing;
- ``transition_evidence`` — whether a face, or a whole band, is a chamfer /
  fillet transition;
- ``band_endpoints`` — what each end of a band runs into once transition
  regions are crossed, and the effective material-side concavity of that
  crossing.

All this module adds is the Pocket rule itself:

    a side-wall band that is not itself a transition ring,
    entered CONVEX from its support and leaving CONCAVE into a terminal
    region, is a Pocket.

The direction matters and is the whole point: the mirror-image sign pattern
(CONCAVE in, CONVEX out) is a Boss, which is a later node.  ``incoming`` is
the band's ``rim_a`` — the rim it was reached by, walking away from the
inner-loop root — and ``outgoing`` is ``rim_b``.  The lineage supplies that
direction; whether a given rim still happens to coincide with somebody's
inner loop is not used and would be wrong on propagated bands.

Mouth transitions are folded in, not stopped on.  ``support -> chamfer SWB ->
pocket wall SWB`` registers one Pocket, on the *wall* band: the chamfer band
is skipped as a principal-wall candidate because it scores as a transition
region, and it is then crossed and absorbed as the wall band's incoming
transition.  So the chamfer's own faces are neither a second Pocket nor part
of this one.

Face ownership: the principal side walls plus the terminal floor region.
Transition faces are recorded but deliberately **not** claimed — chamfer and
fillet become their own features in a later node, and claiming them here
would take that away.

Attribute extraction
--------------------
Every property is measured in the band's own frame (``_shared/band_frame.py``),
never against global XYZ — a pocket tilted in space with a textbook flat floor
still has one fixed depth.  ``feature_axis`` comes from the opening rim plane,
signed opening → bottom by the SWB lineage, and everything else is derived
against it.

The two concave corner properties are not measured here at all, and they are
not opening measurements: a corner belongs to the cavity, not to the end the
SWB traversal started from, so ``band_corners`` minimises over **both**
principal rims — a drafted pocket carries its smallest radii at the floor — and
``ThroughCut`` reads the identical numbers from the same functions.  Everything
else on this node keeps the two stations explicit: ``feature_axis`` and
``machining_direction`` come from the opening-to-bottom frame, ``posX/Y/Z``
comes from the ``rim_b`` bottom footprint, and opening width remains a
``rim_a`` measurement.

Recognition and measurement are deliberately independent: a Pocket that cannot
be measured is still a Pocket.  When the axis cannot be established, or the
bottom region cannot be trusted, the affected properties stay ``null`` with a
recorded reason rather than being guessed.  In particular ``depth`` is only
emitted for a single fixed axial level — never a deepest-point maximum, which
in an intersecting cavity usually belongs to a hole rather than to the pocket.

The attribute set is the ontology's Pocket property set, minus two legacy
ratio aliases.  ``derived_ratio_depth_to_width`` and
``derived_ratio_corner_radius_to_depth`` appear in the ontology only as
undeclared instance triples carrying values identical to their ``min_*``
counterparts; the ``min_*`` names are the declared ``DatatypeProperty``
entries with domains and value spaces, and the ones every DFM rule references.
This module therefore emits the ``min_*`` ratios only.  The redundancy is left
standing in the ontology file for a separate ontology maintenance pass.
"""

from __future__ import annotations

import math
from typing import Any

from contract.partmodel import PartModel, SurfaceType

from features._shared.measurement import min_wall_thickness, remaining_floor_thickness
from core.models import FeatureNode
from features._shared.attributes import canonical_axis, cad_face_ids, rounded_scalar, serialize_cad_face_ids
from features._shared.band_endpoints import (
    CONCAVE,
    CONVEX,
    EndpointResolution,
    SolidPointClassifier,
    resolve_endpoint,
)
from features._shared.band_corners import (
    RimFootprint,
    band_corner_angle,
    band_corner_radius,
    principal_rim_footprints,
)
from features._shared.band_frame import (
    BandFrame,
    build_band_frame,
    max_draft_angle,
    outward_normals,
    parallel,
    rectangular_width,
    rectangular_shape_width,
    footprint_centroid,
    ordered_rim_loop,
)
from features._shared.closed_wall_band import (
    WallBandAnalysis,
    WallBandCandidate,
    build_edge_index,
    candidate_diagnostics,
    face_sort_key,
)
from features._shared.transition_evidence import TransitionEvidence

RECTANGULAR = "RectangularPocket"
GENERAL = "GeneralPocket"
_COMMON_FIELDS = ("cadFaceIds", "feature_axis", "machining_direction", "depth", "pocket_bottom_type", "max_wall_draft_angle", "posX", "posY", "posZ", "derived_floor_thickness", "derived_min_wall_thickness")
# Every closed-wall cavity has corners, so both leaves carry the corner
# properties.  The width stations do not follow: "the perpendicular distance
# between opposite walls" is a measurement only where there are opposite walls,
# and on a general outline that is undefined rather than unmeasured.
_CORNER_FIELDS = ("min_corner_radius", "min_corner_angle", "derived_ratio_min_corner_radius_to_depth")
_ATTRIBUTE_FIELDS = {
    RECTANGULAR: (*_COMMON_FIELDS, "min_opening_width", "min_bottom_width", "derived_ratio_depth_to_min_width", *_CORNER_FIELDS),
    GENERAL: (*_COMMON_FIELDS, *_CORNER_FIELDS),
}
# A bottom face must actually face back toward the opening.  Anything within
# this margin of parallel to the axis is a riser or a wall the rim grazed, and
# its presence means the terminal region is not a trustworthy bottom.
_BOTTOM_FACING_MARGIN_DEG = 5.0
# Axial agreement required before several floor faces count as one depth level.
_DEPTH_LEVEL_TOL = 1.0e-4


def recognize_and_build(
    model: PartModel,
    evidence_document: dict | None,
    claimed_face_ids: set[str] | None = None,
    *,
    bands: WallBandAnalysis,
    transition: TransitionEvidence,
    kernel: Any | None = None,
    start_index: int = 1,
) -> tuple[list[FeatureNode], set[str], dict[str, Any]]:
    """Register a Pocket for every CONVEX-in / CONCAVE-out side-wall band."""
    claimed = claimed_face_ids if claimed_face_ids is not None else set()
    index = build_edge_index(model)
    classifier = SolidPointClassifier(kernel)

    nodes: list[FeatureNode] = []
    rejected: list[dict[str, Any]] = []
    owned_signatures: set[frozenset] = set()
    counters: dict[str, int] = {RECTANGULAR: 0, GENERAL: 0}

    for band in bands.candidates:
        walls = set(band.wall_face_ids)

        band_verdict = transition.region_verdict(band.wall_face_ids)
        if band_verdict.is_transition:
            rejected.append(_rejection(band, "band_is_transition_region", band_verdict.reason))
            continue
        already = sorted(walls & claimed, key=face_sort_key)
        if already:
            rejected.append(_rejection(band, "wall_already_claimed", ",".join(already)))
            continue

        incoming = resolve_endpoint(model, index, transition, classifier, walls, band.rim_a_sigs)
        outgoing = resolve_endpoint(model, index, transition, classifier, walls, band.rim_b_sigs)

        verdict = _pocket_verdict(incoming, outgoing)
        if verdict is not None:
            rejected.append(_rejection(band, verdict, f"incoming={incoming.effective_concavity}/{incoming.status} outgoing={outgoing.effective_concavity}/{outgoing.status}"))
            continue

        floor = [face_id for face_id in outgoing.terminal_face_ids if face_id not in claimed]
        owned = sorted(set(band.wall_face_ids) | set(floor), key=face_sort_key)
        signature = frozenset(owned)
        if signature in owned_signatures:
            rejected.append(_rejection(band, "duplicate_face_set", ""))
            continue

        owned_signatures.add(signature)
        frame = build_band_frame(model, index, band)
        footprints = principal_rim_footprints(model, index, frame, band)
        feature_type = _classify_pocket(footprints)
        counters[feature_type] += 1
        nodes.append(
            _build_node(
                model, index, band, owned, floor, incoming, outgoing,
                band_verdict, kernel,
                start_index + counters[feature_type] - 1,
                frame=frame,
                footprints=footprints,
                feature_type=feature_type,
            )
        )
        claimed.update(owned)

    diagnostics = {
        "featureModule": "Pocket",
        "method": "swb_transition_endpoint_concavity_v1",
        "rule": "incoming CONVEX -> outgoing CONCAVE across a non-transition closed side-wall band",
        "directionSource": "root_anchored_swb_propagation_lineage",
        "aagnetUsed": "transition_face_hint_only",
        "solidClassifierAvailable": classifier.available,
        "bandCount": len(bands.candidates),
        "acceptedCount": len(nodes),
        "acceptedCountByLeafClass": {key: value for key, value in sorted(counters.items()) if value},
        "claimedFaceCount": sum(len(node.partmodelFaceIds) for node in nodes),
        "transitionEvidence": transition.diagnostics(),
        "rejectionSummary": _rejection_summary(rejected),
        "rejectedBands": rejected,
    }
    return nodes, claimed, diagnostics


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

def _pocket_verdict(incoming: EndpointResolution, outgoing: EndpointResolution) -> str | None:
    """Return a rejection reason, or ``None`` when the band is a Pocket."""
    if not incoming.resolved:
        return "incoming_endpoint_unresolved"
    if not outgoing.resolved:
        return "outgoing_endpoint_unresolved"
    if incoming.effective_concavity != CONVEX:
        return "incoming_not_convex"
    if outgoing.effective_concavity != CONCAVE:
        return "outgoing_not_concave"
    if not outgoing.terminal_face_ids:
        return "no_terminal_region"
    if set(outgoing.terminal_face_ids) & set(incoming.terminal_face_ids):
        return "terminal_regions_overlap"
    return None


# ---------------------------------------------------------------------------
# Node construction
# ---------------------------------------------------------------------------

def _build_node(
    model: PartModel,
    edge_index,
    band: WallBandCandidate,
    owned: list[str],
    floor: list[str],
    incoming: EndpointResolution,
    outgoing: EndpointResolution,
    band_verdict,
    kernel,
    index: int,
    *,
    frame: BandFrame,
    footprints: list[RimFootprint],
    feature_type: str,
) -> FeatureNode:
    bottom = _bottom_analysis(model, kernel, frame, floor)
    # Both principal rims: the corner properties are minima over the whole
    # cavity, while min_width stays an *opening* measurement and reads rim_a.
    width = _min_width(footprints[0])
    bottom_width = _min_width(footprints[1])
    angle_value, angle_diagnostics = band_corner_angle(footprints)
    radius_value, radius_diagnostics = band_corner_radius(model, frame, band.wall_face_ids, footprints)
    draft, draft_reason = max_draft_angle(frame) if frame.resolved else (None, f"no feature axis: {frame.detail or frame.status}")
    thickness, thickness_reason = _floor_thickness(model, kernel, frame, owned, floor, bottom)
    classification_diagnostics = _pocket_classification_diagnostics(footprints, feature_type)
    fields = _ATTRIBUTE_FIELDS[feature_type]
    attributes: dict[str, Any] = {field: None for field in fields}
    attributes["cadFaceIds"] = serialize_cad_face_ids(model, owned)
    attributes["feature_axis"] = canonical_axis(frame.axis) if frame.resolved else None
    attributes["machining_direction"] = [round(v, 6) + 0.0 for v in frame.axis] if frame.resolved else None
    attributes["depth"] = bottom["depth"]
    attributes["pocket_bottom_type"] = bottom["pocket_bottom_type"]
    attributes["max_wall_draft_angle"] = draft
    bottom_loop = ordered_rim_loop(model, edge_index, band.rim_b_sigs)
    centre = _bottom_reference_point(frame, bottom_loop)
    attributes["posX"] = rounded_scalar(centre[0]) if centre else None
    attributes["posY"] = rounded_scalar(centre[1]) if centre else None
    attributes["posZ"] = rounded_scalar(centre[2]) if centre else None
    attributes["derived_floor_thickness"] = thickness
    wall_diagnostics: dict[str, Any] = {}
    attributes["derived_min_wall_thickness"] = _wall_thickness(
        model,
        kernel,
        owned,
        band.wall_face_ids,
        diagnostics=wall_diagnostics,
    )
    attributes["min_corner_radius"] = rounded_scalar(radius_value) if radius_value is not None else None
    attributes["min_corner_angle"] = rounded_scalar(angle_value) if angle_value is not None else None
    attributes["derived_ratio_min_corner_radius_to_depth"] = _ratio(attributes["min_corner_radius"], attributes["depth"])
    if feature_type == RECTANGULAR:
        attributes["min_opening_width"] = width["min_width"]
        attributes["min_bottom_width"] = bottom_width["min_width"]
        attributes["derived_ratio_depth_to_min_width"] = _ratio(
            attributes["depth"], _min_positive(attributes["min_opening_width"], attributes["min_bottom_width"])
        )

    reasons = {
        "cadFaceIds": "no source CAD face ids could be recovered from the recognized faces",
        "feature_axis": frame.detail or frame.status,
        "machining_direction": frame.detail or frame.status,
        "depth": bottom["depth_reason"],
        "min_opening_width": width["reason"],
        "min_bottom_width": bottom_width["reason"],
        "min_corner_radius": radius_diagnostics.get("reason", ""),
        "min_corner_angle": angle_diagnostics.get("reason", ""),
        "pocket_bottom_type": bottom["type_reason"],
        "max_wall_draft_angle": draft_reason,
        "posX": "the bottom rim carries no reliable footprint centroid",
        "posY": "the bottom rim carries no reliable footprint centroid",
        "posZ": "the bottom rim carries no reliable footprint centroid",
        "derived_floor_thickness": thickness_reason,
        "derived_min_wall_thickness": "no non-self material surface was reached from sampled pocket-wall normals",
        "derived_ratio_depth_to_min_width": "depth and two positive station widths were not both available",
        "derived_ratio_min_corner_radius_to_depth": "min_corner_radius and a positive depth were not both available",
    }

    return FeatureNode(
        nodeId=f"{feature_type}_{index:03d}",
        featureType=feature_type,
        partmodelFaceIds=list(owned),
        attributes=attributes,
        missingAttributes={f: reasons.get(f, "property was not reliably measurable") for f in fields if attributes.get(f) is None},
        diagnostics={
            "method": "swb_transition_endpoint_concavity_v1",
            "attributeMethod": "band_local_frame_v1",
            "leafClassification": feature_type,
            "classification": classification_diagnostics,
            "sideWallFaceIds": list(band.wall_face_ids),
            "floorFaceIds": list(floor),
            "mouthTransitionFaceIds": list(incoming.transition_face_ids),
            "floorTransitionFaceIds": list(outgoing.transition_face_ids),
            "supportFaceIds": list(incoming.terminal_face_ids),
            "concavity": {
                "incoming": incoming.effective_concavity,
                "outgoing": outgoing.effective_concavity,
                "pattern": f"{incoming.effective_concavity}->{outgoing.effective_concavity}",
            },
            "featureFrame": frame.to_dict(),
            "bottomAnalysis": bottom["diagnostics"],
            "wallThickness": wall_diagnostics,
            "crossSection": width["diagnostics"],
            "bottomCrossSection": bottom_width["diagnostics"],
            "cornerRadii": radius_diagnostics,
            "footprintCorners": angle_diagnostics,
            "incomingEndpoint": incoming.to_dict(),
            "outgoingEndpoint": outgoing.to_dict(),
            "bandTransitionVerdict": band_verdict.to_dict(),
            "sideWallBand": candidate_diagnostics(band),
        },
    )


# ---------------------------------------------------------------------------
# Bottom region: type, and a single fixed depth when one exists
# ---------------------------------------------------------------------------

def _bottom_analysis(model: PartModel, kernel, frame: BandFrame, floor: list[str]) -> dict[str, Any]:
    """Classify the bottom region and derive a depth only if it is unique.

    Reliability comes first: every terminal face must actually face back
    toward the opening.  A face lying parallel to the axis is a riser or a
    wall the rim grazed on its way past, and its presence means the terminal
    region was not cleanly separated from neighbouring geometry — in which
    case nothing about the bottom is asserted.
    """
    out: dict[str, Any] = {
        "depth": None,
        "depth_reason": "",
        "pocket_bottom_type": None,
        "type_reason": "",
        "diagnostics": {},
    }
    if not floor:
        out["depth_reason"] = out["type_reason"] = "the pocket has no terminal bottom region"
        return out
    if not frame.resolved:
        out["depth_reason"] = out["type_reason"] = f"no feature axis: {frame.detail or frame.status}"
        return out

    threshold = -math.sin(math.radians(_BOTTOM_FACING_MARGIN_DEG))
    records, unreliable = [], []
    for face_id in sorted(floor, key=face_sort_key):
        face = model.faces.get(face_id)
        normals = outward_normals(model, face_id)
        dots = [sum(n[i] * frame.axis[i] for i in range(3)) for n in normals]
        facing = bool(dots) and max(dots) <= threshold
        records.append({
            "faceId": face_id,
            "surfaceType": face.surface_type.value if face else "UNKNOWN",
            "maxAxisDot": round(max(dots), 6) if dots else None,
            "facesOpening": facing,
        })
        if not facing:
            unreliable.append(face_id)
    out["diagnostics"] = {"faces": records, "facingMarginDeg": _BOTTOM_FACING_MARGIN_DEG}

    if unreliable:
        reason = (
            "the terminal region is not a clean bottom: "
            + ",".join(unreliable)
            + " do not face the opening, so the bottom could not be separated from adjoining geometry"
        )
        out["depth_reason"] = out["type_reason"] = reason
        return out

    planar = [r["faceId"] for r in records if r["surfaceType"] == SurfaceType.PLANE.value]
    curved = [r["faceId"] for r in records if r["surfaceType"] != SurfaceType.PLANE.value]
    out["pocket_bottom_type"] = "flat" if not curved else ("non_flat" if not planar else "mixed")
    out["diagnostics"]["planarFaceIds"] = planar
    out["diagnostics"]["nonPlanarFaceIds"] = curved

    if curved:
        out["depth_reason"] = "the bottom region is not planar, so it has no single axial depth"
        return out

    levels = []
    for face_id in planar:
        face = model.faces[face_id]
        normal = face.parameters.get("normal")
        location = face.parameters.get("location")
        if not parallel(normal, frame.axis) or not isinstance(location, (list, tuple)):
            out["depth_reason"] = f"floor face {face_id} is not perpendicular to the feature axis"
            return out
        levels.append(frame.depth_of(location))
    out["diagnostics"]["axialLevels"] = [round(v, 6) for v in sorted(levels)]
    if max(levels) - min(levels) > _DEPTH_LEVEL_TOL:
        out["depth_reason"] = (
            f"the floor is stepped: {len(set(round(v, 4) for v in levels))} distinct axial levels "
            f"spanning {max(levels) - min(levels):.4f} mm"
        )
        return out
    out["depth"] = rounded_scalar(sum(levels) / len(levels))
    return out


# ---------------------------------------------------------------------------
# Cross-section and corners
# ---------------------------------------------------------------------------

def _min_width(opening: RimFootprint) -> dict[str, Any]:
    """Wall-to-wall width of the **opening**, for a right-angled quadrilateral.

    Restricting this to squares and rectangles is deliberate: those are the
    outlines where "the width" has one defensible meaning — the shorter
    wall-to-wall distance, or the side length when the two are equal.  On a
    pentagon, a trapezoid or a re-entrant outline any single number would be a
    choice rather than a measurement, so none is reported.

    Unlike the two corner properties this stays an opening measurement, read
    from ``rim_a`` alone: the ontology pairs it with ``depth``, which is
    measured from the same opening plane, and a drafted pocket has a different
    width at every level.
    """
    if not opening.resolved:
        return {"min_width": None, "reason": opening.detail, "diagnostics": {}}
    value, diagnostics = rectangular_width(opening.profile)
    diagnostics["method"] = "perpendicular_distance_between_opposite_straight_walls"
    if value is None:
        return {
            "min_width": None,
            "reason": f"min_width is defined only for a right-angled quadrilateral footprint: {diagnostics.get('reason', 'unsupported outline')}",
            "diagnostics": diagnostics,
        }
    return {"min_width": rounded_scalar(value), "reason": "", "diagnostics": diagnostics}


def _classify_pocket(footprints: list[RimFootprint]) -> str:
    """Split the closed-wall family by both principal rim profiles."""
    if len(footprints) != 2 or any(not item.resolved for item in footprints):
        return GENERAL
    for item in footprints:
        width, _ = rectangular_shape_width(item.profile)
        if width is None:
            return GENERAL
    return RECTANGULAR


def _pocket_classification_diagnostics(
    footprints: list[RimFootprint], feature_type: str
) -> dict[str, Any]:
    """Explain the two-rim shape gate used for Pocket leaves.

    Keep both the per-rim evidence and the final reason in the node
    diagnostics so a reviewer can tell a non-quadrilateral outline from an
    unresolved one without inferring it from nulls.  The per-rim record keeps
    ``blendedCornerCount`` even though a blend no longer decides the leaf: it
    is how a reader sees which corners carry the radius the node publishes.
    """
    rims: list[dict[str, Any]] = []
    for footprint in footprints:
        record: dict[str, Any] = {
            "rim": footprint.label,
            "status": footprint.status,
            "detail": footprint.detail,
        }
        if footprint.resolved and footprint.profile is not None:
            width, shape = rectangular_shape_width(footprint.profile)
            record["rectangularShapeWidth"] = rounded_scalar(width)
            record["shapeDiagnostics"] = shape
        else:
            record["rectangularShapeWidth"] = None
            record["shapeDiagnostics"] = {
                "reason": footprint.detail or "rim profile was not resolved",
            }
        rims.append(record)

    failures = []
    for item in rims:
        if item.get("rectangularShapeWidth") is not None:
            continue
        shape = item.get("shapeDiagnostics")
        detail = shape.get("reason") if isinstance(shape, dict) else None
        failures.append(f"{item.get('rim', 'unknown rim')}: {detail or 'rectangular_shape_width failed'}")
    if feature_type == RECTANGULAR:
        reason = "both principal rims passed rectangular_shape_width"
    else:
        reason = "; ".join(failures) + "; classified as GeneralPocket"
    return {
        "method": "rectangular_shape_width_two_principal_rims_v1",
        "leaf": feature_type,
        "reason": reason,
        "rims": rims,
    }


def _min_positive(*values: float | None) -> float | None:
    usable = [float(value) for value in values if isinstance(value, (int, float)) and float(value) > 0.0]
    return min(usable) if len(usable) == len(values) and usable else None


def _wall_thickness(
    model: PartModel,
    kernel,
    owned: list[str],
    wall_face_ids: list[str],
    *,
    diagnostics: dict[str, Any] | None = None,
) -> float | None:
    return min_wall_thickness(
        kernel,
        cad_face_ids(model, owned),
        cad_face_ids(model, wall_face_ids),
        diagnostics=diagnostics,
    )


def _bottom_reference_point(frame: BandFrame, bottom_loop: list[list[float]] | None) -> list[float] | None:
    """Return the ``rim_b`` bottom-footprint centroid, without a rim-a fallback."""
    if not frame.resolved or not bottom_loop:
        return None
    return footprint_centroid(frame, bottom_loop)


# ---------------------------------------------------------------------------
# Remaining material below the floor
# ---------------------------------------------------------------------------

def _floor_thickness(model: PartModel, kernel, frame: BandFrame, owned: list[str], floor: list[str], bottom: dict) -> tuple[float | None, str]:
    """Reuse the shared ray cast, but only on a trustworthy flat bottom."""
    if kernel is None:
        return None, "no geometry kernel was available"
    if not frame.resolved:
        return None, f"no feature axis: {frame.detail or frame.status}"
    if bottom["depth"] is None:
        return None, "floor thickness is only measured below a bottom with a single fixed depth: " + (bottom["depth_reason"] or "no depth")
    value = remaining_floor_thickness(
        kernel,
        cad_face_ids(model, owned),
        cad_face_ids(model, floor),
        frame.axis,
    )
    if value is None:
        return None, "no material was hit below the floor along the feature axis"
    return value, ""


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or float(denominator) <= 0.0:
        return None
    return rounded_scalar(float(numerator) / float(denominator))


def _rejection(band: WallBandCandidate, reason: str, detail: str) -> dict[str, Any]:
    return {
        "candidateKey": band.candidate_key,
        "discoveryDepth": band.discovery_depth,
        "wallFaceIds": list(band.wall_face_ids),
        "reason": reason,
        "detail": detail,
    }


def _rejection_summary(rejected: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for item in rejected:
        summary[item["reason"]] = summary.get(item["reason"], 0) + 1
    return dict(sorted(summary.items()))
