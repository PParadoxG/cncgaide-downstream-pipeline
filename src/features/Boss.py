"""Boss recognition — the ontology leaf classes ``CircularBoss``,
``RectangularBoss`` and ``GeneralBoss``.

This module is the mirror image of ``Pocket.py`` and deliberately reuses the
same shared chain, unchanged:

- ``closed_wall_band``   — where the closed side-wall bands are, and the
  root-anchored lineage that tells incoming from outgoing;
- ``transition_evidence``— whether a face, or a whole band, is a chamfer /
  fillet transition;
- ``band_endpoints``     — what each end of a band runs into once transition
  regions are crossed, and the effective material-side concavity of that
  crossing;
- ``band_frame``         — the feature-local axis, the footprint decomposition
  and the round geometry the leaf classification needs.

All this module adds is the Boss rule and the footprint classification:

    a side-wall band that is not itself a transition ring,
    entered CONCAVE from its support and leaving CONVEX into a terminal
    region, is a Boss.

The sign pattern is exactly Pocket's reversed, and the direction comes from
the SWB lineage: ``incoming`` is ``rim_a``, the rim the band was reached by
walking away from the inner-loop root, ``outgoing`` is ``rim_b``.  For a Boss
that makes ``rim_a`` the *footprint* on the support and ``rim_b`` the top rim,
so the feature axis recovered from ``rim_a`` runs base → top.  The v5 output
publishes this canonical ``feature_axis`` together with the signed
``machining_direction`` that points from the boss top into the support.

Face ownership: the principal side walls plus the top terminal region.  The
support face the boss stands on is **not** claimed — it belongs to the part,
not to the protrusion — and neither are the chamfer / fillet faces crossed on
the way in or out, which become their own features in a later node.

Leaf classification
-------------------
The ontology partitions Boss by footprint shape, so exactly one leaf class is
emitted per boss; there is no intermediate ``Boss`` node.

``CircularBoss`` is restricted to a *regular complete cylindrical* boss: the
principal wall must be one coaxial cylindrical surface (one merged face, or
several unmerged seam fragments of it) and both principal rims must be
complete closed circles of that same radius, on that same axis.  Corner
decomposition is explicitly not used for this: a circle, an ellipse and any
smooth blob all decompose to "no corner and no straight", so counting corners
would classify an oval boss as circular.  The ontology allows a tapered
(conical) circular boss and this implementation deliberately does not cover
it — a taper is left unclassified rather than admitted by loosening the
cylinder test, and it is *not* swept into ``GeneralBoss`` either, because a
round footprint is not "neither circular nor rectangular".

``RectangularBoss`` is a *rectangular* footprint: four straight walls meeting
at four 90-degree corners, with opposite walls parallel and equal in length,
which is what the ontology asks for in so many words ("gegenueberliegende
Seiten gleich lang").  The corners may be blended — a corner reads its angle
from the turn accumulated over its whole run, so a filleted right angle is
still a right angle, and requiring sharpness would have named a shape a cutter
cannot leave.  A rounded rectangle is a rectangle here, and lands in this
class rather than in ``GeneralBoss``.

``GeneralBoss`` is the catch-all, and is only reached when the footprint was
successfully decomposed *and* shown to be neither of the above.  A band whose
geometry could not be resolved at all — no feature axis, no closed rim loop,
no simple projected profile — produces no node and is reported under
``unclassifiedBands`` instead.  Guessing ``GeneralBoss`` there would silently
turn "we could not measure this" into a positive ontological claim.

Attribute extraction
--------------------
Everything is measured in the band's own frame, never against global XYZ.
``height`` is the axial distance from the footprint plane to the top rim, and
is emitted only when the top rim sits at one fixed axial level — a boss whose
top runs uphill has no single height and gets ``null`` with a reason rather
than an average.  ``posX/Y/Z`` is the footprint area centroid, per the
ontology's Boss reference-point convention.

The attribute set is each leaf class's v5 property set and nothing more:
the common ``cadFaceIds`` / ``feature_axis`` / ``machining_direction`` /
``height`` / ``posX,Y,Z``, plus ``base_diameter`` and ``top_diameter`` for
``CircularBoss`` and ``base_thickness`` / ``top_thickness`` with
``derived_ratio_height_to_thickness`` for ``RectangularBoss``.  ``GeneralBoss``
gets no invented scalar — the ontology states outright that its outline lives
in the CAD geometry referenced by ``cadFaceIds``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from contract.partmodel import PartModel, SurfaceType

from features._shared.measurement import min_wall_thickness
from core.models import FeatureNode
from features._shared.attributes import canonical_axis, cad_face_ids, rounded_scalar, serialize_cad_face_ids
from features._shared.band_endpoints import (
    CONCAVE,
    CONVEX,
    EndpointResolution,
    SolidPointClassifier,
    resolve_endpoint,
)
from features._shared.band_frame import (
    AxialLevel,
    BandFrame,
    FootprintProfile,
    RimCircle,
    axial_level,
    build_band_frame,
    cylindrical_wall_radius,
    footprint_centroid,
    footprint_profile,
    lateral_offset,
    max_draft_angle,
    ordered_rim_loop,
    outward_normals,
    rim_circle,
    rim_points,
    rectangular_shape_width,
)
from features._shared.closed_wall_band import (
    EdgeIndex,
    WallBandAnalysis,
    WallBandCandidate,
    build_edge_index,
    candidate_diagnostics,
    face_sort_key,
)
from features._shared.transition_evidence import TransitionEvidence

CIRCULAR = "CircularBoss"
RECTANGULAR = "RectangularBoss"
GENERAL = "GeneralBoss"

_COMMON_FIELDS = ("cadFaceIds", "feature_axis", "machining_direction", "height", "posX", "posY", "posZ")
_ATTRIBUTE_FIELDS: dict[str, tuple[str, ...]] = {
    CIRCULAR: (*_COMMON_FIELDS, "base_diameter", "top_diameter", "derived_wall_draft_angle", "derived_min_clearance"),
    RECTANGULAR: (*_COMMON_FIELDS, "base_thickness", "top_thickness", "max_wall_draft_angle", "derived_min_clearance", "derived_ratio_height_to_thickness"),
    GENERAL: (*_COMMON_FIELDS, "max_wall_draft_angle", "derived_min_clearance"),
}

# A top face must actually face away from the support.  Anything within this
# margin of parallel to the axis is a riser the top rim grazed rather than part
# of the cap; this is reported, never used to reject.
_TOP_FACING_MARGIN_DEG = 5.0


def recognize_and_build(
    model: PartModel,
    evidence_document: dict | None,
    claimed_face_ids: set[str] | None = None,
    *,
    bands: WallBandAnalysis,
    transition: TransitionEvidence,
    kernel: Any | None = None,
    start_index: int = 1,
    edge_index: EdgeIndex | None = None,
) -> tuple[list[FeatureNode], set[str], dict[str, Any]]:
    """Register a Boss leaf class for every CONCAVE-in / CONVEX-out band."""
    claimed = set(claimed_face_ids or ())
    # The orchestrator builds one index per part and hands it down; building
    # our own is the standalone-call path, not the production one.
    index = edge_index if edge_index is not None else build_edge_index(model)
    classifier = SolidPointClassifier(kernel)

    nodes: list[FeatureNode] = []
    rejected: list[dict[str, Any]] = []
    unclassified: list[dict[str, Any]] = []
    owned_signatures: set[frozenset] = set()
    counters: dict[str, int] = {CIRCULAR: 0, RECTANGULAR: 0, GENERAL: 0}

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

        verdict = _boss_verdict(incoming, outgoing)
        if verdict is not None:
            rejected.append(_rejection(band, verdict, f"incoming={incoming.effective_concavity}/{incoming.status} outgoing={outgoing.effective_concavity}/{outgoing.status}"))
            continue

        top = [face_id for face_id in outgoing.terminal_face_ids if face_id not in claimed]
        owned = sorted(walls | set(top), key=face_sort_key)
        signature = frozenset(owned)
        if signature in owned_signatures:
            rejected.append(_rejection(band, "duplicate_face_set", ""))
            continue

        shape = _classify(model, index, band)
        if shape.feature_type is None:
            # A boss we cannot type is reported, not invented: the ontology's
            # Boss partition is by footprint, so an unmeasurable footprint
            # names no leaf class and must not be defaulted into GeneralBoss.
            unclassified.append(_rejection(band, "footprint_not_classifiable", shape.reason))
            continue

        owned_signatures.add(signature)
        counters[shape.feature_type] += 1
        nodes.append(
            _build_node(
                model, index, band, owned, top, incoming, outgoing,
                band_verdict, shape, kernel, start_index + counters[shape.feature_type] - 1,
            )
        )
        claimed.update(owned)

    diagnostics = {
        "featureModule": "Boss",
        "method": "swb_transition_endpoint_concavity_v1",
        "rule": "incoming CONCAVE -> outgoing CONVEX across a non-transition closed side-wall band",
        "directionSource": "root_anchored_swb_propagation_lineage",
        "leafClassification": "cylindrical wall + two complete rim circles -> CircularBoss; four 90-degree corners between four matching straight walls, blended or sharp -> RectangularBoss; otherwise, only when the footprint resolved and is neither -> GeneralBoss",
        "taperedCircularBossCovered": False,
        "aagnetUsed": "transition_face_hint_only",
        "solidClassifierAvailable": classifier.available,
        "bandCount": len(bands.candidates),
        "acceptedCount": len(nodes),
        "acceptedCountByLeafClass": {key: value for key, value in sorted(counters.items()) if value},
        "unclassifiedCount": len(unclassified),
        "claimedFaceCount": sum(len(node.partmodelFaceIds) for node in nodes),
        "rejectionSummary": _rejection_summary(rejected),
        "rejectedBands": rejected,
        "unclassifiedBands": unclassified,
    }
    return nodes, claimed, diagnostics


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

def _boss_verdict(incoming: EndpointResolution, outgoing: EndpointResolution) -> str | None:
    """Return a rejection reason, or ``None`` when the band is a Boss."""
    if not incoming.resolved:
        return "incoming_endpoint_unresolved"
    if not outgoing.resolved:
        return "outgoing_endpoint_unresolved"
    if incoming.effective_concavity != CONCAVE:
        return "incoming_not_concave"
    if outgoing.effective_concavity != CONVEX:
        return "outgoing_not_convex"
    if not outgoing.terminal_face_ids:
        return "no_terminal_region"
    if set(outgoing.terminal_face_ids) & set(incoming.terminal_face_ids):
        return "terminal_regions_overlap"
    return None


# ---------------------------------------------------------------------------
# Leaf classification
# ---------------------------------------------------------------------------

@dataclass
class Shape:
    """Which Boss leaf class the band's footprint names, and the evidence."""

    feature_type: str | None
    reason: str = ""
    frame: BandFrame | None = None
    base_loop: list[list[float]] | None = None
    top_loop: list[list[float]] | None = None
    profile: FootprintProfile | None = None
    base_circle: RimCircle | None = None
    top_circle: RimCircle | None = None
    cylinder_radius: float | None = None
    cylinder: dict = field(default_factory=dict)
    width: float | None = None
    base_thickness: float | None = None
    top_thickness: float | None = None
    rectangle: dict = field(default_factory=dict)
    basis: str = ""

    def to_dict(self) -> dict:
        return {
            "featureType": self.feature_type,
            "basis": self.basis,
            "reason": self.reason,
            "cylindricalWall": self.cylinder,
            "baseRimCircle": self.base_circle.to_dict() if self.base_circle else None,
            "topRimCircle": self.top_circle.to_dict() if self.top_circle else None,
            "rectangularFootprint": self.rectangle,
            "baseThickness": self.base_thickness,
            "topThickness": self.top_thickness,
            "footprint": {
                "cornerCount": len(self.profile.corners) if self.profile else None,
                "straightCount": len(self.profile.straights) if self.profile else None,
                "excursionCount": len(self.profile.excursions) if self.profile else None,
                "corners": self.profile.corners if self.profile else [],
                "excursions": self.profile.excursions if self.profile else [],
            },
        }


def _classify(model: PartModel, edge_index, band: WallBandCandidate) -> Shape:
    """Name the ontology leaf class, or explain why the footprint names none."""
    frame = build_band_frame(model, edge_index, band)
    if not frame.resolved:
        return Shape(None, f"no feature axis: {frame.detail or frame.status}", frame=frame)

    base_loop = ordered_rim_loop(model, edge_index, band.rim_a_sigs)
    if base_loop is None:
        return Shape(None, "the footprint rim edges do not chain into a single closed loop", frame=frame)
    top_loop = ordered_rim_loop(model, edge_index, band.rim_b_sigs)

    base_circle = rim_circle(frame, base_loop)
    top_circle = rim_circle(frame, top_loop)
    radius, cylinder = cylindrical_wall_radius(model, frame, band.wall_face_ids)
    shape = Shape(
        None,
        frame=frame,
        base_loop=base_loop,
        base_circle=base_circle,
        top_circle=top_circle,
        cylinder_radius=radius,
        cylinder=cylinder,
    )

    circular, circular_reason = _is_regular_round(model, frame, band.wall_face_ids, radius, base_circle, top_circle)
    if circular:
        shape.feature_type = CIRCULAR
        shape.basis = "complete cylindrical wall between two coaxial closed rim circles"
        return shape
    cylinder["circularBossReason"] = circular_reason

    shape.profile = footprint_profile(frame, base_loop)
    if shape.profile is None:
        shape.reason = "the projected footprint is not a simple closed profile (total turn is not 360 degrees)"
        return shape

    shape.width, shape.rectangle = rectangular_shape_width(shape.profile)
    shape.top_loop = top_loop
    top_profile = footprint_profile(frame, top_loop)
    shape.base_thickness = shape.width
    shape.top_thickness = rectangular_shape_width(top_profile)[0] if top_profile is not None else None
    if shape.width is not None and shape.top_thickness is not None:
        shape.feature_type = RECTANGULAR
        shape.basis = "four right-angled corners between four matching straight walls"
        return shape

    if base_circle is not None and base_circle.is_closed_circle:
        # Round in plan, but the wall is not a plain cylinder — a tapered or
        # otherwise non-cylindrical circular boss.  It is not GeneralBoss
        # (the footprint *is* circular) and it is not CircularBoss under the
        # strict rule, so no leaf class is claimed.
        shape.reason = (
            "the footprint is a closed circle but the wall is not a complete cylindrical surface "
            f"({cylinder.get('reason', 'no cylinder')}); tapered circular bosses are not covered"
        )
        return shape

    shape.feature_type = GENERAL
    shape.basis = "footprint resolved and shown to be neither circular nor rectangular"
    return shape


def _is_regular_round(
    model: PartModel,
    frame: BandFrame,
    wall_face_ids,
    radius: float | None,
    base: RimCircle | None,
    top: RimCircle | None,
) -> tuple[bool, str]:
    """Is the boss bounded by two complete coaxial circular stations?"""
    if base is None or not base.is_closed_circle:
        return False, "the footprint rim is not one complete closed circle"
    if top is None or not top.is_closed_circle:
        return False, "the top rim is not one complete closed circle"
    budget = max(base.budget, top.budget)
    if radius is not None and (abs(base.radius - radius) > budget or abs(top.radius - radius) > budget):
        return False, f"rim radii {base.radius:.6f}/{top.radius:.6f} disagree with the wall radius {radius:.6f}"
    surfaces = {model.faces[face_id].surface_type for face_id in wall_face_ids if face_id in model.faces}
    if not surfaces or not surfaces <= {SurfaceType.CYLINDER, SurfaceType.CONE}:
        return False, "the principal wall is neither cylindrical nor conical"
    offset = lateral_offset(frame, base.center, top.center)
    if offset is None or offset > budget:
        return False, f"the two rim circles are not coaxial ({offset:.6g} mm apart across the axis)"
    return True, ""


# ---------------------------------------------------------------------------
# Node construction
# ---------------------------------------------------------------------------

def _build_node(
    model: PartModel,
    edge_index,
    band: WallBandCandidate,
    owned: list[str],
    top: list[str],
    incoming: EndpointResolution,
    outgoing: EndpointResolution,
    band_verdict,
    shape: Shape,
    kernel,
    index: int,
) -> FeatureNode:
    frame = shape.frame
    feature_type = shape.feature_type
    fields = _ATTRIBUTE_FIELDS[feature_type]

    height, height_level, height_reason = _height(model, edge_index, frame, band)
    centre = footprint_centroid(frame, shape.base_loop)
    centre_reason = "the footprint rim encloses no measurable area, so it has no centroid"
    clearance_diagnostics: dict[str, Any] = {}
    clearance = _clearance(model, kernel, owned, band.wall_face_ids, diagnostics=clearance_diagnostics)

    attributes: dict[str, Any] = {name: None for name in fields}
    reasons: dict[str, str] = {
        "cadFaceIds": "no source CAD face ids could be recovered from the recognized faces",
        "feature_axis": (frame.detail or frame.status) if frame else "the band has no feature frame",
        "machining_direction": (frame.detail or frame.status) if frame else "the band has no feature frame",
        "height": height_reason,
        "posX": centre_reason,
        "posY": centre_reason,
        "posZ": centre_reason,
    }
    attributes["cadFaceIds"] = serialize_cad_face_ids(model, owned)
    attributes["feature_axis"] = canonical_axis(frame.axis) if frame and frame.resolved else None
    attributes["machining_direction"] = [round(-value, 6) + 0.0 for value in frame.axis] if frame and frame.resolved else None
    attributes["height"] = height
    attributes["posX"] = _position(centre, 0)
    attributes["posY"] = _position(centre, 1)
    attributes["posZ"] = _position(centre, 2)

    if feature_type == CIRCULAR:
        attributes["base_diameter"] = rounded_scalar(2.0 * shape.base_circle.radius) if shape.base_circle else None
        attributes["top_diameter"] = rounded_scalar(2.0 * shape.top_circle.radius) if shape.top_circle else None
        attributes["derived_wall_draft_angle"] = _ratio_wall_draft(
            attributes.get("base_diameter"), attributes.get("top_diameter"), attributes.get("height")
        )
        attributes["derived_min_clearance"] = clearance
        reasons["base_diameter"] = "the base rim is not a complete circular station"
        reasons["top_diameter"] = "the top rim is not a complete circular station"
        reasons["derived_wall_draft_angle"] = "base/top diameter and height did not establish a finite derived draft"
        reasons["derived_min_clearance"] = "no other solid surface was reached from sampled boss-wall outward normals"
    elif feature_type == RECTANGULAR:
        attributes["base_thickness"] = rounded_scalar(shape.base_thickness)
        attributes["top_thickness"] = rounded_scalar(shape.top_thickness)
        draft, draft_reason = max_draft_angle(frame) if frame and frame.resolved else (None, "the boss frame has no measurable wall draft")
        attributes["max_wall_draft_angle"] = draft
        attributes["derived_min_clearance"] = clearance
        attributes["derived_ratio_height_to_thickness"] = _ratio(
            attributes["height"], _min_positive(attributes["base_thickness"], attributes["top_thickness"])
        )
        reasons["base_thickness"] = f"the base outline has no rectangular wall-to-wall thickness: {shape.rectangle.get('reason', 'unsupported outline')}"
        reasons["top_thickness"] = "the top outline has no rectangular wall-to-wall thickness"
        reasons["max_wall_draft_angle"] = draft_reason
        reasons["derived_min_clearance"] = "no other solid surface was reached from sampled boss-wall outward normals"
        reasons["derived_ratio_height_to_thickness"] = "height and two positive station thicknesses were not both available"
    elif feature_type == GENERAL:
        draft, draft_reason = max_draft_angle(frame) if frame and frame.resolved else (None, "the boss frame has no measurable wall draft")
        attributes["max_wall_draft_angle"] = draft
        attributes["derived_min_clearance"] = clearance
        reasons["max_wall_draft_angle"] = draft_reason
        reasons["derived_min_clearance"] = "no other solid surface was reached from sampled boss-wall outward normals"

    return FeatureNode(
        nodeId=f"{feature_type}_{index:03d}",
        featureType=feature_type,
        partmodelFaceIds=list(owned),
        attributes=attributes,
        missingAttributes={name: reasons[name] for name in fields if attributes.get(name) is None},
        diagnostics={
            "method": "swb_transition_endpoint_concavity_v1",
            "attributeMethod": "band_local_frame_v1",
            "sideWallFaceIds": list(band.wall_face_ids),
            "topFaceIds": list(top),
            "supportFaceIds": list(incoming.terminal_face_ids),
            "supportTransitionFaceIds": list(incoming.transition_face_ids),
            "topTransitionFaceIds": list(outgoing.transition_face_ids),
            "concavity": {
                "incoming": incoming.effective_concavity,
                "outgoing": outgoing.effective_concavity,
                "pattern": f"{incoming.effective_concavity}->{outgoing.effective_concavity}",
            },
            "featureFrame": frame.to_dict() if frame else None,
            "leafClassification": shape.to_dict(),
            "heightMeasurement": height_level.to_dict() if height_level else None,
            "topAnalysis": _top_analysis(model, frame, top),
            "clearance": clearance_diagnostics,
            "incomingEndpoint": incoming.to_dict(),
            "outgoingEndpoint": outgoing.to_dict(),
            "bandTransitionVerdict": band_verdict.to_dict(),
            "sideWallBand": candidate_diagnostics(band),
        },
    )


# ---------------------------------------------------------------------------
# Height, and what the top region looks like
# ---------------------------------------------------------------------------

def _height(
    model: PartModel,
    edge_index,
    frame: BandFrame,
    band: WallBandCandidate,
) -> tuple[float | None, AxialLevel | None, str]:
    """Axial rise from the footprint plane to the top rim.

    Measured rim to rim rather than from the wall's extent, so it is the same
    quantity on a cylinder and on a prismatic wall.  A top rim spread over
    several axial levels — a boss grown on a slanted support, or one whose top
    runs into other geometry — has no single height, and none is reported.
    """
    if frame is None or not frame.resolved:
        return None, None, f"no feature axis: {frame.detail if frame else 'no frame'}"
    points = rim_points(model, edge_index, band.rim_b_sigs)
    level = axial_level(frame, points)
    if level is None:
        return None, None, "the top rim carries no sampled points"
    if not level.flat:
        return None, level, (
            f"the top rim is not at one axial level: it spans {level.spread:.6g} mm along the axis "
            f"(budget {level.budget:.6g})"
        )
    if level.level <= 0.0:
        return None, level, f"the top rim does not rise above the footprint plane (axial level {level.level:.6g} mm)"
    return rounded_scalar(level.level), level, ""


def _top_analysis(model: PartModel, frame: BandFrame, top: list[str]) -> dict[str, Any]:
    """Does the claimed top region actually face away from the support?

    Recorded for audit only.  Nothing about a Boss is derived from the top
    faces — height comes from the rims — so a top region that also contains a
    riser is a reporting matter, not a reason to refuse the feature.
    """
    if not top or frame is None or not frame.resolved:
        return {"faces": [], "facingMarginDeg": _TOP_FACING_MARGIN_DEG}
    threshold = math.sin(math.radians(_TOP_FACING_MARGIN_DEG))
    records = []
    for face_id in sorted(top, key=face_sort_key):
        face = model.faces.get(face_id)
        normals = outward_normals(model, face_id)
        dots = [sum(normal[i] * frame.axis[i] for i in range(3)) for normal in normals]
        records.append({
            "faceId": face_id,
            "surfaceType": face.surface_type.value if face else "UNKNOWN",
            "minAxisDot": round(min(dots), 6) if dots else None,
            "facesAwayFromSupport": bool(dots) and min(dots) >= threshold,
        })
    return {
        "faces": records,
        "facingMarginDeg": _TOP_FACING_MARGIN_DEG,
        "planarFaceIds": [item["faceId"] for item in records if item["surfaceType"] == SurfaceType.PLANE.value],
    }


def _position(centre: list[float] | None, axis: int) -> float | None:
    """One coordinate of the footprint centroid, without a signed zero."""
    if centre is None:
        return None
    value = rounded_scalar(centre[axis])
    return None if value is None else value + 0.0


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or float(denominator) <= 0.0:
        return None
    return rounded_scalar(float(numerator) / float(denominator))


def _min_positive(*values: float | None) -> float | None:
    usable = [float(value) for value in values if isinstance(value, (int, float)) and float(value) > 0.0]
    return min(usable) if len(usable) == len(values) and usable else None


def _ratio_wall_draft(base_diameter: float | None, top_diameter: float | None, height: float | None) -> float | None:
    if base_diameter is None or top_diameter is None or height is None or float(height) <= 0.0:
        return None
    value = math.degrees(math.atan((float(base_diameter) - float(top_diameter)) / 2.0 / float(height)))
    return rounded_scalar(value) if math.isfinite(value) else None


def _clearance(
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
        material_side=False,
        diagnostics=diagnostics,
    )


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
