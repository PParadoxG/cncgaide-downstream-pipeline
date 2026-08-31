"""ThroughCut recognition — the ontology ``ThroughCut`` class.

The third consumer of the shared side-wall-band (SWB) chain, and like the other
two it adds nothing but its own rule:

    Pocket:      incoming CONVEX  -> outgoing CONCAVE
    Boss:        incoming CONCAVE -> outgoing CONVEX
    ThroughCut:  incoming CONVEX  -> outgoing CONVEX, with a real opening at
                 both ends

The sign pattern alone is deliberately **not** the rule.  ``CONVEX -> CONVEX``
says only that the material turns outward at both ends of the band, which is
equally true of the part's own outer wall between its top and bottom faces, and
of a shoulder or step band that happens to close.  What separates a window from
those is that a window *opens*: at each end the band runs out through a face of
the part and leaves a hole in it.  That is asked directly, and it is a
topological question, not a threshold:

    the rim where the band (after any chamfer / fillet is folded in) meets its
    terminal region is exactly an **inner loop** of a face of that region.

An inner loop is a hole bounded by surrounding material — precisely the
ontology's "passes through two faces of the part".  The outer wall of a plate
fails it at both ends, because the plate's silhouette is an *outer* loop of the
top and bottom faces; a step or an open slot fails earlier still, since its wall
band has no second closed rim at all and never reaches this module.

Geometric confirmation
----------------------
Topology says the two ends are openings; it does not say the volume between
them was removed.  So the passage is also confirmed geometrically: an interior
point of the entry footprint is swept along the feature axis and classified
against the solid at several levels strictly between the two openings.  Every
sample must lie **outside** the material.  A band whose passage is obstructed —
a cross rib, a blind recess that happens to open twice — is rejected on that
evidence rather than on a heuristic.

This makes the feature axis a *recognition* prerequisite here, which it is not
for Pocket or Boss.  That is the honest consequence of the class definition:
"through" is a claim about a passage, and a band with no axis, no closed rim
loop or no simple footprint offers nothing to check the claim against.  Such a
band produces no node and is reported under ``unconfirmedBands`` with the
specific reason — never guessed into a ThroughCut.

Round cross-sections belong to the Hole family
----------------------------------------------
``ThroughCut`` is the *window* class (AP224 Cutout).  A cut whose footprint is
one complete closed circle is a round hole and belongs to ``ThroughHole``,
which runs first and claims its walls, so such a band normally never reaches
here.  The guard is kept anyway and is explicit: a closed-circle footprint is
refused with a reason, so a hole that the hole modules failed to seed shows up
as a reported gap instead of being quietly re-typed as a window.

Face ownership: the principal side walls, and nothing else.  The faces the cut
breaks out through at either end belong to the part, not to the cut, and the
chamfer / fillet faces crossed on the way in or out become their own features
in a later node.

Attributes
----------
The ontology's ThroughCut property set and nothing more.  ``feature_axis`` is
canonicalized and ``posX/Y/Z`` are the midpoint of the two opening footprints.
The two concave corner properties are not: a window is open at both ends and
neither is privileged, so ``band_corners`` minimises them over **both**
principal rims, and Pocket reads the identical numbers from the same functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contract.partmodel import PartModel
from features._shared.measurement import min_wall_thickness

from core.models import FeatureNode
from features._shared.attributes import canonical_axis, cad_face_ids, rounded_scalar, serialize_cad_face_ids
from features._shared.band_endpoints import (
    CONVEX,
    EndpointResolution,
    SolidPointClassifier,
    resolve_endpoint,
)
from features._shared.band_corners import (
    band_corner_angle,
    band_corner_radius,
    principal_rim_footprints,
)
from features._shared.band_frame import (
    AxialLevel,
    BandFrame,
    FootprintProfile,
    RimCircle,
    axial_level,
    build_band_frame,
    footprint_centroid,
    footprint_interior_point,
    footprint_profile,
    ordered_rim_loop,
    max_draft_angle,
    rim_circle,
    rim_points,
)
from features._shared.closed_wall_band import (
    WallBandAnalysis,
    WallBandCandidate,
    build_edge_index,
    candidate_diagnostics,
    face_sort_key,
    match_inner_loop,
)
from features._shared.transition_evidence import TransitionEvidence

_ATTRIBUTE_FIELDS = (
    "cadFaceIds",
    "feature_axis",
    "min_corner_angle",
    "min_corner_radius",
    "max_wall_draft_angle",
    "derived_min_wall_thickness",
    "posX",
    "posY",
    "posZ",
)

# Fractions of the axial span between the two openings at which the passage is
# sampled.  Strictly interior, so a sample never lands on the terminal faces
# themselves, and spread out so a mid-height obstruction cannot be stepped over.
_PASSAGE_FRACTIONS = (0.1, 0.25, 0.5, 0.75, 0.9)
# Geometry status for a band whose footprint is a complete circle: a round
# cross-section is a Hole, and this module refuses it rather than re-typing it.
_CIRCULAR = "circular_footprint"


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
    """Register a ThroughCut for every band that opens at both of its ends."""
    claimed = claimed_face_ids if claimed_face_ids is not None else set()
    index = build_edge_index(model)
    classifier = SolidPointClassifier(kernel)

    nodes: list[FeatureNode] = []
    rejected: list[dict[str, Any]] = []
    unconfirmed: list[dict[str, Any]] = []
    owned_signatures: set[frozenset] = set()

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

        verdict = _concavity_verdict(incoming, outgoing)
        if verdict is not None:
            rejected.append(_rejection(band, verdict, f"incoming={incoming.effective_concavity}/{incoming.status} outgoing={outgoing.effective_concavity}/{outgoing.status}"))
            continue

        entry = _opening(model, index, walls, incoming)
        if entry.match is None:
            rejected.append(_rejection(band, "entry_end_is_not_an_opening", entry.reason))
            continue
        exit_ = _opening(model, index, walls, outgoing)
        if exit_.match is None:
            rejected.append(_rejection(band, "exit_end_is_not_an_opening", exit_.reason))
            continue

        geometry = _geometry(model, index, band, kernel)
        if geometry.status == _CIRCULAR:
            rejected.append(_rejection(band, "circular_footprint_belongs_to_the_hole_family", geometry.reason))
            continue
        if geometry.status != "resolved":
            unconfirmed.append(_rejection(band, geometry.status, geometry.reason))
            continue

        passage = _passage(geometry, classifier)
        if not passage["confirmed"]:
            if passage["blockedAt"]:
                rejected.append(_rejection(band, "passage_blocked", passage["reason"]))
            else:
                unconfirmed.append(_rejection(band, "passage_not_verifiable", passage["reason"]))
            continue

        owned = sorted(walls, key=face_sort_key)
        signature = frozenset(owned)
        if signature in owned_signatures:
            rejected.append(_rejection(band, "duplicate_face_set", ""))
            continue

        owned_signatures.add(signature)
        nodes.append(
            _build_node(
                model, index, band, owned, incoming, outgoing, entry, exit_,
                geometry, passage, band_verdict, kernel, start_index + len(nodes),
            )
        )
        claimed.update(owned)

    diagnostics = {
        "featureModule": "ThroughCut",
        "method": "swb_transition_endpoint_concavity_v1",
        "rule": (
            "incoming CONVEX -> outgoing CONVEX across a non-transition closed side-wall band, "
            "with the terminal rim at each end an exact inner loop of its terminal region, and "
            "the passage between the two openings confirmed void"
        ),
        "directionSource": "root_anchored_swb_propagation_lineage",
        "openingTest": "terminal_rim_equals_an_inner_loop_of_the_terminal_region",
        "passageTest": "footprint interior point classified against the solid at interior axial levels",
        "roundCrossSectionPolicy": "closed_circle_footprint_is_a_hole_and_is_refused_here",
        "aagnetUsed": "transition_face_hint_only",
        "solidClassifierAvailable": classifier.available,
        "bandCount": len(bands.candidates),
        "acceptedCount": len(nodes),
        "unconfirmedCount": len(unconfirmed),
        "claimedFaceCount": sum(len(node.partmodelFaceIds) for node in nodes),
        "rejectionSummary": _rejection_summary(rejected),
        "rejectedBands": rejected,
        "unconfirmedBands": unconfirmed,
    }
    return nodes, claimed, diagnostics


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

def _concavity_verdict(incoming: EndpointResolution, outgoing: EndpointResolution) -> str | None:
    """The sign pattern half of the rule: a rejection reason, or ``None``."""
    if not incoming.resolved:
        return "incoming_endpoint_unresolved"
    if not outgoing.resolved:
        return "outgoing_endpoint_unresolved"
    if incoming.effective_concavity != CONVEX:
        return "incoming_not_convex"
    if outgoing.effective_concavity != CONVEX:
        return "outgoing_not_convex"
    if not incoming.terminal_face_ids or not outgoing.terminal_face_ids:
        return "no_terminal_region"
    if set(outgoing.terminal_face_ids) & set(incoming.terminal_face_ids):
        return "terminal_regions_overlap"
    return None


@dataclass
class _Opening:
    """One end of the band, and the hole it leaves in the part surface."""

    resolution: EndpointResolution
    match: Any = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "terminalFaceIds": list(self.resolution.terminal_face_ids),
            "transitionFaceIds": list(self.resolution.transition_face_ids),
            "innerLoopMatch": (
                {"faceId": self.match.face_id, "loopId": self.match.loop_id, "method": self.match.method}
                if self.match is not None
                else None
            ),
            "reason": self.reason,
        }


def _opening(model: PartModel, index, walls: set[str], resolution: EndpointResolution) -> _Opening:
    """Does this end of the band leave a hole in the face it breaks out of?

    Asked of the rim that actually meets the terminal region — after any
    chamfer or fillet has been folded in — because with a mouth treatment the
    band's own rim bounds the transition ring, not the part surface.
    """
    rim = resolution.terminal_rim_sigs
    if not rim:
        return _Opening(resolution, None, "the endpoint traversal recorded no rim")
    # Everything the traversal has already been through is the near side: the
    # walls plus any transition region crossed on the way out.
    near = set(walls) | set(resolution.transition_face_ids)
    match = match_inner_loop(model, index, rim, near)
    if match is None:
        return _Opening(
            resolution,
            None,
            "the terminal rim is not an inner loop of any terminal face, so this end runs along a surface "
            "rather than breaking through it",
        )
    return _Opening(resolution, match, "")


# ---------------------------------------------------------------------------
# Frame, footprint and the passage between the two openings
# ---------------------------------------------------------------------------

@dataclass
class _Geometry:
    """The band's own frame plus everything measured in it."""

    status: str
    reason: str = ""
    frame: BandFrame | None = None
    entry_loop: list[list[float]] | None = None
    entry_circle: RimCircle | None = None
    profile: FootprintProfile | None = None
    interior_uv: tuple[float, float] | None = None
    exit_level: AxialLevel | None = None


def _geometry(model: PartModel, index, band: WallBandCandidate, kernel) -> _Geometry:
    """Recover the frame and footprint, or say exactly what was missing."""
    frame = build_band_frame(model, index, band)
    if not frame.resolved:
        return _Geometry("no_feature_axis", frame.detail or frame.status, frame=frame)

    entry_loop = ordered_rim_loop(model, index, band.rim_a_sigs)
    if entry_loop is None:
        return _Geometry(
            "entry_rim_not_a_single_loop",
            "the entry rim edges do not chain into a single closed loop",
            frame=frame,
        )

    entry_circle = rim_circle(frame, entry_loop)
    if entry_circle is not None and entry_circle.is_closed_circle:
        return _Geometry(
            _CIRCULAR,
            f"the entry footprint is one complete closed circle of radius {entry_circle.radius:.6f}",
            frame=frame,
            entry_loop=entry_loop,
            entry_circle=entry_circle,
        )

    profile = footprint_profile(frame, entry_loop)
    if profile is None:
        return _Geometry(
            "footprint_not_a_simple_profile",
            "the projected footprint is not a simple closed profile (total turn is not 360 degrees)",
            frame=frame,
            entry_loop=entry_loop,
        )

    interior = footprint_interior_point(frame, entry_loop)
    if interior is None:
        return _Geometry(
            "footprint_has_no_interior_point",
            "the projected footprint encloses no area, so the passage through it cannot be sampled",
            frame=frame,
            entry_loop=entry_loop,
            profile=profile,
        )

    level = axial_level(frame, rim_points(model, index, band.rim_b_sigs))
    if level is None or level.minimum <= 0.0:
        return _Geometry(
            "openings_not_separated_along_the_axis",
            "the exit rim does not lie wholly clear of the entry opening plane, so there is no passage to sample",
            frame=frame,
            entry_loop=entry_loop,
            profile=profile,
            interior_uv=interior,
        )
    return _Geometry(
        "resolved",
        frame=frame,
        entry_loop=entry_loop,
        entry_circle=entry_circle,
        profile=profile,
        interior_uv=interior,
        exit_level=level,
    )


def _passage(geometry: _Geometry, classifier: SolidPointClassifier | None) -> dict[str, Any]:
    """Is the volume between the two openings actually removed?

    The entry footprint's interior point is swept along the feature axis and
    classified against the solid at levels strictly between the two openings.
    ``in`` at any level means material still stands in the way, and the band is
    not a through cut whatever its ends look like.  ``on`` is treated as
    inconclusive rather than as material: it means the sample grazed a surface,
    which a wall-hugging interior point can do on a thin cut.
    """
    frame, interior = geometry.frame, geometry.interior_uv
    samples: list[dict[str, Any]] = []
    result = {"confirmed": False, "blockedAt": None, "reason": "", "samples": samples, "method": "solid_classification_along_the_feature_axis"}
    if classifier is None or not classifier.available:
        result["reason"] = "no solid classifier was available to confirm the passage"
        return result

    span = float(geometry.exit_level.minimum)
    tolerance = max(1.0e-7, abs(span) * 1.0e-6)
    inside = 0
    for fraction in _PASSAGE_FRACTIONS:
        depth = span * fraction
        point = frame.point_at(interior[0], interior[1], depth)
        state = classifier.state(point, tolerance)
        samples.append({"fraction": fraction, "depth": round(depth, 6), "state": state})
        if state == "in":
            inside += 1
    if inside:
        result["blockedAt"] = [item["fraction"] for item in samples if item["state"] == "in"]
        result["reason"] = f"{inside} of {len(samples)} axial samples between the two openings lie inside the material"
        return result
    if not any(item["state"] == "out" for item in samples):
        result["reason"] = "no axial sample between the two openings could be classified as outside the material"
        return result
    result["confirmed"] = True
    return result


# ---------------------------------------------------------------------------
# Node construction
# ---------------------------------------------------------------------------

def _build_node(
    model: PartModel,
    edge_index,
    band: WallBandCandidate,
    owned: list[str],
    incoming: EndpointResolution,
    outgoing: EndpointResolution,
    entry: _Opening,
    exit_: _Opening,
    geometry: _Geometry,
    passage: dict[str, Any],
    band_verdict,
    kernel,
    index: int,
) -> FeatureNode:
    frame = geometry.frame
    # The corner properties are minima over *both* principal rims — a window is
    # open at both ends and neither is privileged, so neither may be the only
    # one measured.  Position is the midpoint of the two opening centroids.
    footprints = principal_rim_footprints(model, edge_index, frame, band)
    angle, angle_diagnostics = band_corner_angle(footprints)
    radius, radius_diagnostics = band_corner_radius(model, frame, band.wall_face_ids, footprints)
    entry_centre = footprint_centroid(frame, geometry.entry_loop)
    exit_loop = ordered_rim_loop(model, edge_index, band.rim_b_sigs)
    exit_centre = footprint_centroid(frame, exit_loop)
    centre = _opening_midpoint(entry_centre, exit_centre)
    draft, draft_reason = max_draft_angle(frame)
    wall_diagnostics: dict[str, Any] = {}
    wall_thickness = min_wall_thickness(
        kernel,
        cad_face_ids(model, owned),
        cad_face_ids(model, band.wall_face_ids),
        material_side=True,
        diagnostics=wall_diagnostics,
    )

    attributes: dict[str, Any] = {name: None for name in _ATTRIBUTE_FIELDS}
    attributes["cadFaceIds"] = serialize_cad_face_ids(model, owned)
    attributes["feature_axis"] = canonical_axis(frame.axis)
    attributes["min_corner_angle"] = rounded_scalar(angle) if angle is not None else None
    attributes["min_corner_radius"] = rounded_scalar(radius) if radius is not None else None
    attributes["max_wall_draft_angle"] = draft
    attributes["derived_min_wall_thickness"] = wall_thickness
    attributes["posX"] = _position(centre, 0)
    attributes["posY"] = _position(centre, 1)
    attributes["posZ"] = _position(centre, 2)

    centre_reason = "both opening footprints need reliable area centroids to publish their midpoint"
    reasons = {
        "cadFaceIds": "no source CAD face ids could be recovered from the recognized faces",
        "feature_axis": frame.detail or frame.status,
        "min_corner_angle": angle_diagnostics.get("reason", ""),
        "min_corner_radius": radius_diagnostics.get("reason", ""),
        "max_wall_draft_angle": draft_reason,
        "derived_min_wall_thickness": "no non-self material surface was reached from sampled wall normals" if wall_thickness is None else "",
        "posX": centre_reason,
        "posY": centre_reason,
        "posZ": centre_reason,
    }

    return FeatureNode(
        nodeId=f"ThroughCut_{index:03d}",
        featureType="ThroughCut",
        partmodelFaceIds=list(owned),
        attributes=attributes,
        missingAttributes={name: reasons[name] for name in _ATTRIBUTE_FIELDS if attributes.get(name) is None},
        diagnostics={
            "method": "swb_transition_endpoint_concavity_v1",
            "attributeMethod": "band_local_frame_v1",
            "sideWallFaceIds": list(band.wall_face_ids),
            "entryOpening": entry.to_dict(),
            "exitOpening": exit_.to_dict(),
            "concavity": {
                "incoming": incoming.effective_concavity,
                "outgoing": outgoing.effective_concavity,
                "pattern": f"{incoming.effective_concavity}->{outgoing.effective_concavity}",
            },
            "featureFrame": frame.to_dict(),
            "wallThickness": wall_diagnostics,
            "passageCheck": passage,
            "openingSeparation": geometry.exit_level.to_dict() if geometry.exit_level else None,
            "entryRimCircle": geometry.entry_circle.to_dict() if geometry.entry_circle else None,
            "footprintCorners": angle_diagnostics,
            "cornerRadii": radius_diagnostics,
            "incomingEndpoint": incoming.to_dict(),
            "outgoingEndpoint": outgoing.to_dict(),
            "bandTransitionVerdict": band_verdict.to_dict(),
            "sideWallBand": candidate_diagnostics(band),
        },
    )


def _position(centre: list[float] | None, axis: int) -> float | None:
    """One coordinate of the opening centroid, without a signed zero."""
    if centre is None:
        return None
    value = rounded_scalar(centre[axis])
    return None if value is None else value + 0.0


def _opening_midpoint(entry_centre: list[float] | None, exit_centre: list[float] | None) -> list[float] | None:
    """Return a ThroughCut position only when both opening centroids exist."""
    if entry_centre is None or exit_centre is None:
        return None
    return [0.5 * (entry_centre[index] + exit_centre[index]) for index in range(3)]


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
