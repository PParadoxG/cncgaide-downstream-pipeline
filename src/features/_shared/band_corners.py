"""The ontology's two concave corner properties, measured on **both**
principal rims of a side-wall band.

``min_corner_radius`` and ``min_corner_angle`` are the same two questions for a
Pocket and for a ThroughCut — one cavity cut all the way through is the other
with its floor removed — so they are answered once, here, and each feature
module only supplies its band and renames the result.

Why both rims
-------------
A side-wall band has two principal rims and the cavity has a cross-section at
each of them.  Reading only one is wrong twice over:

- **it misses corners.**  A drafted pocket is narrower at the floor than at the
  mouth, so its floor corners carry the smaller radii — exactly the ones a
  minimum is supposed to find.  A cavity whose mouth is fully rounded but whose
  floor outline has a sharp corner would report "no corner at all".
- **it depends on which way the SWB traversal happened to arrive.**  ``rim_a``
  is the rim the band was entered by, walking away from the inner-loop root.
  That direction is right for the machining axis, the opening centroid and the
  depth, and it is *meaningless* for a corner: a corner belongs to the cavity,
  not to the end the traversal started from.  Both properties are now invariant
  under swapping the two rims, which is what makes them well defined.

So each rim is projected along the band's single feature axis, decomposed by
``band_frame.footprint_profile``, and the property is the minimum over the
**union** of both rims' concave corners.  One rim contributing nothing is fine —
a fully rounded mouth above a square floor is a real shape.  One rim failing to
*resolve* is not: minimising over what is left would silently report a "global"
minimum that was never taken over the whole feature, so the property is null
with the reason naming the rim that could not be read.

Only the principal band's own rims take part.  A mouth chamfer or a floor
fillet outside the band is a transition, is not claimed by the feature, and
contributes no corner here.

What is *not* a corner angle
----------------------------
``footprint_profile`` already refuses one degenerate run: an **excursion**,
whose turns cancel so the boundary leaves in the direction it arrived (a
scallop, a tangent bump).  Its interior angle would be 180 degrees, which says
nothing.

The mirror-image degeneracy is a **full reversal**: a run that turns through
180 degrees, so the boundary leaves *antiparallel* to the way it arrived.  That
is the end of an obround slot — two parallel walls joined by a semicircle — and
its interior angle comes out as 0.  Zero is outside the ontology's value space
(``0 < min_corner_angle < 180``) and, read as a corner, it claims a maximally
acute internal corner where the geometry has none: the walls never converge,
nothing sits in the corner that a cutter cannot reach, and what actually limits
the cutter there is the end *radius*.  A full reversal is therefore excluded
from the angle minimum.

It is deliberately **not** excluded from the radius: the semicircular end has a
real radius which really does bound the tool, and reporting it is the whole
point of ``min_corner_radius``.  The two properties disagree about that corner
because they are asking different questions about it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from contract.partmodel import PartModel, SurfaceType

from features._shared.band_frame import (
    BandFrame,
    FootprintProfile,
    footprint_profile,
    material_is_outside,
    ordered_rim_loop,
    parallel,
)
from features._shared.closed_wall_band import EdgeIndex, face_sort_key

RIM_A = "rimA"
RIM_B = "rimB"

# How close to the ends of the ontology value space a corner may sit before it
# stops being a discrete corner angle.  The lower end is the full reversal
# described in the module docstring; the upper end mirrors the excursion rule
# in ``footprint_profile`` and is belt-and-braces, since a corner at 180 is not
# concave in the first place.  Same magnitude as that rule's turn epsilon: nine
# orders of magnitude above the summed-``atan2`` noise floor, and far below any
# angle a machinist would argue about.
_ANGLE_MARGIN_DEG = 1.0e-3


@dataclass
class RimFootprint:
    """One principal rim, projected along the feature axis and decomposed."""

    label: str
    status: str
    detail: str = ""
    profile: FootprintProfile | None = None
    # A point at this rim's own axial station, needed where a corner radius
    # varies along the axis (a conical corner wall).
    station: list[float] | None = None

    @property
    def resolved(self) -> bool:
        return self.status == "resolved" and self.profile is not None

    @property
    def concave_corners(self) -> list[dict]:
        return self.profile.concave_corners if self.profile is not None else []

    def to_dict(self) -> dict:
        profile = self.profile
        return {
            "rim": self.label,
            "status": self.status,
            "detail": self.detail,
            "cornerCount": len(profile.corners) if profile else None,
            "concaveCornerCount": len(profile.concave_corners) if profile else None,
            "straightCount": len(profile.straights) if profile else None,
            "excursionCount": len(profile.excursions) if profile else None,
            "corners": profile.corners if profile else [],
            "excursions": profile.excursions if profile else [],
        }


def rim_footprint(
    model: PartModel,
    index: EdgeIndex,
    frame: BandFrame,
    rim_sigs,
    label: str,
) -> RimFootprint:
    """Project one rim into the band frame and decompose it, or say why not."""
    if not frame.resolved:
        return RimFootprint(label, "no_feature_axis", f"no feature axis: {frame.detail or frame.status}")
    loop = ordered_rim_loop(model, index, rim_sigs)
    if loop is None:
        return RimFootprint(label, "rim_not_a_single_loop", f"the {label} edges do not chain into a single closed loop")
    station = [sum(point[axis] for point in loop) / len(loop) for axis in range(3)]
    profile = footprint_profile(frame, loop)
    if profile is None:
        return RimFootprint(
            label,
            "footprint_not_a_simple_profile",
            f"the {label} footprint is not a simple closed profile (total turn is not 360 degrees)",
            station=station,
        )
    return RimFootprint(label, "resolved", profile=profile, station=station)


def principal_rim_footprints(model: PartModel, index: EdgeIndex, frame: BandFrame, band) -> list[RimFootprint]:
    """Both principal rims of a band, projected into its one feature axis.

    Both rims are projected into the *same* frame on purpose: that projection is
    the plan view the cutter sees, the axis is a property of the band rather
    than of either end, and using one frame is what makes the result identical
    when the two rims change places.
    """
    return [
        rim_footprint(model, index, frame, band.rim_a_sigs, RIM_A),
        rim_footprint(model, index, frame, band.rim_b_sigs, RIM_B),
    ]


# ---------------------------------------------------------------------------
# min_corner_angle
# ---------------------------------------------------------------------------

def band_corner_angle(footprints: list[RimFootprint]) -> tuple[float | None, dict]:
    """Smallest concave interior angle over both rims, in degrees.

    Each corner's angle is the turn accumulated over its whole run, measured
    between the straight stretches that flank it, so a corner reports the same
    angle whether it is modelled sharp or carries a fillet — which is what H41
    ("internal corners not acute") is about, and why a blend does not change
    the answer.

    Corners outside the ontology's ``0 < angle < 180`` value space are not
    reported as angles: a corner at 180 is convex from inside the cavity (the
    cutter sweeps past it), and a corner at 0 is a full reversal, an obround
    end whose walls never converge (see the module docstring).
    """
    diagnostics: dict = {
        "method": "minimum over the concave corners of both principal rim footprints",
        "cornerAngleMethod": "accumulated_turn_between_straight_stretches_of_the_axial_projection",
        "valueSpace": "0 < min_corner_angle < 180",
        "rims": [item.to_dict() for item in footprints],
    }
    blocked = _unresolved(footprints, diagnostics)
    if blocked is not None:
        return None, blocked

    candidates: list[dict] = []
    reversals: list[dict] = []
    for item in footprints:
        for corner in item.concave_corners:
            angle = float(corner["interiorAngleDeg"])
            record = {"rim": item.label, "interiorAngleDeg": angle, "blended": corner["blended"]}
            if _ANGLE_MARGIN_DEG < angle < 180.0 - _ANGLE_MARGIN_DEG:
                candidates.append(record)
            else:
                reversals.append(record)
    diagnostics["angleCandidates"] = candidates
    diagnostics["fullReversals"] = reversals

    if candidates:
        return min(item["interiorAngleDeg"] for item in candidates), diagnostics
    if reversals:
        diagnostics["reason"] = (
            f"every concave corner on both principal rims is a full reversal ({len(reversals)} of them): the "
            f"boundary leaves antiparallel to the way it arrived, as at the end of an obround slot, so the "
            f"walls never converge and there is no internal angle to report. The end radius is what bounds "
            f"the tool there, and min_corner_radius carries it."
        )
        return None, diagnostics
    diagnostics["reason"] = (
        "neither principal rim has a concave internal corner: no direction change bounded by straight "
        "stretches, or every corner is convex from inside the cavity"
    )
    return None, diagnostics


# ---------------------------------------------------------------------------
# min_corner_radius
# ---------------------------------------------------------------------------

def band_corner_radius(
    model: PartModel,
    frame: BandFrame,
    wall_face_ids,
    footprints: list[RimFootprint],
) -> tuple[float | None, dict]:
    """Smallest concave internal corner radius over both rims.

    The ontology's value space is explicit: ``>= 0``, zero means sharp, and
    absence is reserved for "no meaningful discrete concave corner exists".
    The footprints separate those cases:

    - no discrete concave corner on either rim → ``None``;
    - any concave corner on either rim is *sharp* → ``0``, because a sharp
      corner has radius zero and the property is a minimum over corners;
    - every concave corner is blended → measure the radii, which come from
      cylinder or cone walls whose axis runs along the feature axis and whose
      material lies *outside* the surface.  Convex corners of a re-entrant
      outline have their material inside and are not what the corner-radius
      rule is about.

    A conical corner wall has a different radius at each end of the band, so it
    is measured at both rim stations and both values compete in the minimum.
    A cylindrical one has a single radius and is measured once.
    """
    diagnostics: dict = {
        "method": "minimum over the concave corners of both principal rim footprints",
        "rims": [item.to_dict() for item in footprints],
    }
    blocked = _unresolved(footprints, diagnostics)
    if blocked is not None:
        return None, blocked

    concave = [(item, corner) for item in footprints for corner in item.concave_corners]
    diagnostics["concaveCornerCount"] = len(concave)
    diagnostics["concaveCornerCountByRim"] = {item.label: len(item.concave_corners) for item in footprints}
    if not concave:
        diagnostics["reason"] = (
            "neither principal rim has a discrete concave corner, so no corner radius exists to report"
        )
        return None, diagnostics

    sharp = [(item, corner) for item, corner in concave if not corner["blended"]]
    if sharp:
        diagnostics["basis"] = "sharp_concave_corner"
        diagnostics["sharpConcaveCornerCount"] = len(sharp)
        diagnostics["sharpConcaveCornerRims"] = sorted({item.label for item, _ in sharp})
        return 0.0, diagnostics

    corner_faces, convex = corner_wall_faces(model, frame, wall_face_ids)
    candidates = corner_wall_radii(model, corner_faces, footprints)
    diagnostics["candidates"] = candidates
    diagnostics["nonInternalCornerFaceIds"] = convex
    if not candidates:
        detail = f"only convex/outward corner faces are present ({','.join(convex)})" if convex else "no corner face is in the band at all"
        diagnostics["basis"] = "blended_corner_without_band_face"
        diagnostics["reason"] = (
            f"every concave corner on both principal rims is blended, but the radius is not carried by the "
            f"principal wall band: {detail}. A mouth-only fillet produces this — the blend face spans the "
            f"opening rather than the wall, so it is outside the band by construction."
        )
        return None, diagnostics
    diagnostics["basis"] = "band_corner_face"
    return min(item["radius"] for item in candidates), diagnostics


def corner_wall_faces(
    model: PartModel,
    frame: BandFrame,
    wall_face_ids,
) -> tuple[list[str], list[str]]:
    """Split the band walls into those that carry an internal corner radius,
    and those that turn the other way.

    A corner wall is a cylinder or cone revolved about the feature axis with
    the material *outside* its surface.  The convex corner of a re-entrant
    outline has its material inside and is not what the corner-radius rule is
    about; it is returned separately so the refusal can name it.
    """
    corners: list[str] = []
    convex: list[str] = []
    for face_id in sorted(set(wall_face_ids), key=face_sort_key):
        face = model.faces.get(face_id)
        if face is None or face.surface_type not in (SurfaceType.CYLINDER, SurfaceType.CONE):
            continue
        if not parallel(face.parameters.get("axis_direction"), frame.axis):
            continue
        if material_is_outside(model, face_id) is not True:
            convex.append(face_id)
            continue
        corners.append(face_id)
    return corners, convex


def corner_wall_radii(model: PartModel, face_ids, footprints: list[RimFootprint]) -> list[dict]:
    """The radius each corner wall carries, at every rim station.

    A cylinder has one radius wherever it is measured.  A cone does not, so it
    is measured once per rim and both values compete in the minimum — which is
    what makes a drafted corner report the smaller of its two ends.
    """
    stations = [item for item in footprints if item.station is not None]
    candidates: list[dict] = []
    for face_id in face_ids:
        face = model.faces.get(face_id)
        if face is None:
            continue
        params = face.parameters
        if face.surface_type == SurfaceType.CYLINDER:
            measured = [(None, params.get("radius"))]
        else:
            measured = [(item.label, cone_radius_at(params, item.station)) for item in stations]
        for label, radius in measured:
            if not isinstance(radius, (int, float)) or not math.isfinite(float(radius)) or float(radius) <= 0.0:
                continue
            record = {"faceId": face_id, "surfaceType": face.surface_type.value, "radius": round(float(radius), 6)}
            if label is not None:
                record["rim"] = label
            candidates.append(record)
    return candidates


def cone_radius_at(params: dict, point) -> float | None:
    """Radius of a conical surface at the axial station of ``point``."""
    apex, semi_angle = params.get("apex"), params.get("semi_angle")
    axis = params.get("axis_direction")
    if not isinstance(apex, (list, tuple)) or not isinstance(semi_angle, (int, float)) or not isinstance(axis, (list, tuple)):
        return None
    if not isinstance(point, (list, tuple)) or len(point) != 3:
        return None
    along = sum((float(point[i]) - float(apex[i])) * float(axis[i]) for i in range(3))
    return abs(along) * math.tan(float(semi_angle))


# ---------------------------------------------------------------------------
# Shared refusal
# ---------------------------------------------------------------------------

def _unresolved(footprints: list[RimFootprint], diagnostics: dict) -> dict | None:
    """Refuse the whole measurement when a rim that must take part cannot.

    Both rims bound the same cavity, so a minimum that quietly skipped one of
    them would be a minimum over part of the feature reported as if it were the
    whole.  The property is null instead, naming the rim and the reason.
    """
    blocked = [item for item in footprints if not item.resolved]
    if not blocked:
        return None
    diagnostics["unresolvedRims"] = [{"rim": item.label, "status": item.status, "detail": item.detail} for item in blocked]
    diagnostics["reason"] = (
        "a principal rim that takes part in this property could not be read, so no minimum over the whole "
        "feature can be stated: " + "; ".join(f"{item.label}: {item.detail}" for item in blocked)
    )
    return diagnostics
