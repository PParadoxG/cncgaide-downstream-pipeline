"""Feature-local frame for a closed side-wall band, and the geometry that
depends on it.

Every band-derived property has to be expressed against the feature's own
axis, never against global XYZ: a pocket tilted in space with a textbook flat
floor still has one fixed depth, and a global-Z rule would miss it.  This
module recovers that frame once and exposes the measurements that live in it.
It is Pocket-agnostic — Boss and Slot need exactly the same frame.

Axis recovery
-------------
The axis is the normal of the band's **opening rim plane**, signed from the
opening toward the bottom using the band's own incoming → outgoing lineage
(``rim_a`` → ``rim_b``).  A prismatic band and a uniformly drafted band both
have a planar opening rim, so this covers the supported cases and naturally
refuses the unsupported ones: a band on a curved support, or one whose mouth
is not planar, has no axis and gets none.

Wall admissibility is then checked *through* the axis rather than by surface
type.  For every wall face, outward normals are sampled across the face and
converted to a local draft angle

    draft = asin(-(n · axis))

which is 0 for a wall parallel to the axis, the semi-angle for a drafted
wall, and — the point of doing it this way — *varies across the face* for a
twisted or freeform wall.  Requiring the per-face spread to be small is
therefore a direct test of "does this wall admit a single stable machining
axis", and it does not need a surface-type whitelist: a B-spline that happens
to be a clean ruled draft surface passes, and a nominally simple surface that
twists does not.

Footprint
---------
The opening rim projected onto the plane perpendicular to the axis is the
feature's footprint, and ``footprint_profile`` decomposes it into **corners**,
**straight stretches** and **excursions**.  A corner's interior angle is the
turn accumulated over its whole run, so a corner reads the same angle whether
it is modelled sharp or carries a fillet — and whether it was blended is
recorded, which is how a corner *radius* is later known to be zero or not.  A
run whose turns cancel is not a corner but an excursion: a scallop or a tangent
bump that leaves the boundary running the way it arrived.

``rectangular_width`` then reports a width only for a right-angled
quadrilateral, as the perpendicular distance between opposite straight walls.
Restricting it that far is deliberate: for a square or a rectangle "the width"
has one defensible meaning, while on a pentagon or a re-entrant outline any
single number would be a choice rather than a measurement.  Measuring wall to
wall also makes corner fillets irrelevant — a rounded rectangle keeps the full
width its straight walls define.  ``rectangular_shape_width`` is the sibling
for the other question, *shape classification* rather than dimension: it adds
what a rectangle owes beyond four angles each right to within a tolerance,
namely opposite walls that are parallel and equal.  It asks nothing about
whether the corners are sharp — that is a question about the tool, not about
the shape, and a leaf class keyed on it names a shape no cutter can leave.

The ontology's two concave corner properties are *not* here: they are minima
over the corners of **both** principal rims of a band, not of one outline, so
they live in ``band_corners`` which composes this module's decomposition twice.
``footprint_interior_point`` does stay here — it answers "where is the inside of
this outline", which is what a passage-through test must know before it can
sample anything.

Round geometry
--------------
Corner decomposition cannot answer "is this outline a circle": a circle, an
ellipse and any smooth blob all decompose to *no corner and no straight*, so
"zero corners" is not a circularity test.  ``rim_circle`` answers the question
directly — a least-squares circle through the rim in the band's own plane,
reported together with the residuals that decide whether the rim really is one
complete closed circle (radial deviation, out-of-plane deviation, and the
winding number that separates a full turn from an arc).
``cylindrical_wall_radius`` is its wall-side counterpart: the single radius of
a band whose faces are all one coaxial cylindrical surface.  Together they are
what a *regular cylindrical* feature is recognized by; neither is enough alone.

``axial_level`` and ``footprint_centroid`` are the two remaining frame-local
measurements shared across features: how far a rim sits above or below the
opening plane (and whether it sits at one level at all), and where the
footprint's area centroid is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from contract.partmodel import PartModel, SurfaceType

from features._shared.closed_wall_band import EdgeIndex, face_sort_key

# Opening-rim planarity budget, relative to the rim's own extent.
_PLANARITY_RATIO = 1.0e-4
# Per-face draft spread allowed before a wall counts as twisted / freeform.
_DRAFT_SPREAD_TOL_DEG = 1.0
# A wall leaning further than this is not a side wall in any useful sense.
_MAX_DRAFT_DEG = 60.0
# Parallel / perpendicular direction tolerance.
_ANGLE_TOL_DEG = 1.0
# Endpoint merge rounding for chaining rim edges into a cycle — the same
# equivalence ``closed_wall_band`` uses when it builds rim cycles.
_VERTEX_MERGE_ROUND = 4
# Circle-fit budget: an absolute floor plus a size-relative term.  STEP stores
# circles analytically and the parser samples them on the exact curve, so a
# genuine circular rim lands orders of magnitude inside this; the budget is
# sized to survive B-spline rim samples, not to admit ellipses.
_CIRCLE_ABS_TOL = 1.0e-4
_CIRCLE_REL_TOL = 1.0e-4
# How far the accumulated turn of a rim may miss a full revolution.
_FULL_TURN_TOL = 0.01
# Coaxiality budget for cylinder axis lines, matching the parser's merge rule.
_AXIS_OFFSET_TOL = 1.0e-4
# Cylinder radius agreement, matching the parser's seam-merge radius rule.
_RADIUS_TOL = 1.0e-6
# Net turn a run must accumulate before it counts as a corner rather than an
# excursion.  Nine orders of magnitude above the ~1e-12 degree noise floor of
# the summed ``atan2`` turns, and far below any direction change a machinist
# would call a corner.
_CORNER_MIN_TURN_DEG = 1.0e-3


@dataclass
class WallDraft:
    """Per-wall draft measurement in the band frame."""

    face_id: str
    surface_type: str
    draft_deg: float | None
    spread_deg: float | None
    sample_count: int

    def to_dict(self) -> dict:
        return {
            "faceId": self.face_id,
            "surfaceType": self.surface_type,
            "draftDeg": round(self.draft_deg, 4) if self.draft_deg is not None else None,
            "spreadDeg": round(self.spread_deg, 4) if self.spread_deg is not None else None,
            "sampleCount": self.sample_count,
        }


@dataclass
class BandFrame:
    """Feature-local frame: unique axis from opening toward bottom."""

    status: str
    detail: str = ""
    axis: list[float] | None = None
    origin: list[float] | None = None
    basis_u: list[float] | None = None
    basis_v: list[float] | None = None
    opening_level: float | None = None
    opening_planarity: float | None = None
    wall_drafts: list[WallDraft] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.status == "resolved" and self.axis is not None

    def axial(self, point) -> float:
        return sum(float(point[i]) * self.axis[i] for i in range(3))

    def depth_of(self, point) -> float:
        """Axial distance below the opening plane (positive into the pocket)."""
        return self.axial(point) - float(self.opening_level)

    def plane_uv(self, point) -> tuple[float, float]:
        rel = [float(point[i]) - self.origin[i] for i in range(3)]
        return (
            sum(rel[i] * self.basis_u[i] for i in range(3)),
            sum(rel[i] * self.basis_v[i] for i in range(3)),
        )

    def point_at(self, u: float, v: float, depth: float = 0.0) -> list[float]:
        """The inverse of ``plane_uv``: frame coordinates back to a 3D point."""
        return [
            self.origin[i] + u * self.basis_u[i] + v * self.basis_v[i] + depth * self.axis[i]
            for i in range(3)
        ]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "detail": self.detail,
            "axis": [round(v, 6) for v in self.axis] if self.axis else None,
            "origin": [round(v, 6) for v in self.origin] if self.origin else None,
            "openingPlanarity": round(self.opening_planarity, 8) if self.opening_planarity is not None else None,
            "wallDrafts": [item.to_dict() for item in self.wall_drafts],
        }


# ---------------------------------------------------------------------------
# Axis recovery
# ---------------------------------------------------------------------------

def build_band_frame(model: PartModel, index: EdgeIndex, band) -> BandFrame:
    """Recover the band's feature-local frame, or say why there is none."""
    opening = rim_points(model, index, band.rim_a_sigs)
    if len(opening) < 3:
        return BandFrame("no_opening_rim_points", "the opening rim carries too few sampled points")

    centroid, normal, residual, extent = _fit_plane(opening)
    if normal is None:
        return BandFrame("opening_rim_degenerate", "the opening rim points are collinear")
    budget = max(1.0e-9, extent * _PLANARITY_RATIO)
    if residual > budget:
        return BandFrame(
            "opening_rim_not_planar",
            f"opening rim deviates {residual:.6g} mm from its best-fit plane (budget {budget:.6g})",
        )

    bottom = rim_points(model, index, band.rim_b_sigs)
    if not bottom:
        return BandFrame("no_bottom_rim_points", "the outgoing rim carries no sampled points")
    bottom_centroid = [sum(p[i] for p in bottom) / len(bottom) for i in range(3)]
    if sum((bottom_centroid[i] - centroid[i]) * normal[i] for i in range(3)) < 0.0:
        normal = [-value for value in normal]

    frame = BandFrame(
        status="resolved",
        axis=normal,
        origin=centroid,
        opening_level=sum(centroid[i] * normal[i] for i in range(3)),
        opening_planarity=residual,
    )
    frame.basis_u, frame.basis_v = _orthonormal_basis(normal)

    drafts: list[WallDraft] = []
    for face_id in sorted(band.wall_face_ids, key=face_sort_key):
        drafts.append(_wall_draft(model, face_id, normal))
    frame.wall_drafts = drafts

    unmeasured = [d.face_id for d in drafts if d.draft_deg is None]
    if unmeasured:
        return BandFrame("wall_normal_unavailable", "no outward normal could be sampled on: " + ",".join(unmeasured))
    twisted = [d.face_id for d in drafts if (d.spread_deg or 0.0) > _DRAFT_SPREAD_TOL_DEG]
    if twisted:
        return BandFrame(
            "wall_not_axis_ruled",
            "draft varies across the face (twisted or freeform wall): " + ",".join(twisted),
            wall_drafts=drafts,
        )
    steep = [d.face_id for d in drafts if abs(d.draft_deg) > _MAX_DRAFT_DEG]
    if steep:
        return BandFrame("wall_draft_out_of_range", "wall leans beyond the supported range: " + ",".join(steep), wall_drafts=drafts)
    return frame


def max_draft_angle(frame: BandFrame) -> tuple[float | None, str]:
    """Maximum absolute draft over the band's walls, per the ontology.

    ``max_wall_draft_angle`` is defined as the *maximum absolute* draft angle among
    all relevant wall faces, with zero meaning genuinely vertical walls — not a
    band-wide average, and not conditional on the walls agreeing.  Walls of
    0/0/1/2 degrees give 2.  A wall too twisted for a stable draft is already
    excluded earlier: it makes the whole frame unresolvable, so a resolved
    frame has a measurable draft on every wall.
    """
    values = [abs(d.draft_deg) for d in frame.wall_drafts if d.draft_deg is not None]
    if not values:
        return None, "no wall draft could be measured on the principal wall band"
    return round(max(values), 6) + 0.0, ""


def _wall_draft(model: PartModel, face_id: str, axis: list[float]) -> WallDraft:
    face = model.faces.get(face_id)
    surface_type = face.surface_type.value if face is not None else "UNKNOWN"
    normals = outward_normals(model, face_id)
    if not normals:
        return WallDraft(face_id, surface_type, None, None, 0)
    angles = [math.degrees(math.asin(max(-1.0, min(1.0, -_dot(n, axis))))) for n in normals]
    return WallDraft(
        face_id=face_id,
        surface_type=surface_type,
        draft_deg=sum(angles) / len(angles),
        spread_deg=max(angles) - min(angles),
        sample_count=len(angles),
    )


# ---------------------------------------------------------------------------
# Outward normals
# ---------------------------------------------------------------------------

def outward_normals(model: PartModel, face_id: str) -> list[list[float]]:
    """Sampled true outward normals of an active face.

    Planes are answered from the persisted parameters; everything else reads
    the stored interior samples so cones, cylinders and blends all report the
    variation that matters for the axis test.
    """
    face = model.faces.get(face_id)
    if face is None:
        return []
    if face.surface_type == SurfaceType.PLANE:
        unit = _unit(face.parameters.get("normal"))
        if unit is not None:
            return [[-v for v in unit] if face.is_reversed else unit]
    return [normal for _, normal in surface_samples(model, face_id)]


def surface_samples(model: PartModel, face_id: str):
    """``(point, outward_normal)`` pairs sampled inside an active face.

    Read from ``FaceData.interior_samples``, which the kernel line filled at
    parse time (merged members pooled by ascending raw face id, so the draft
    test is about the wall, not about how the kernel happened to split it).
    An old ``partmodel.json`` without the field yields no samples.
    """
    face = model.faces.get(face_id)
    if face is None:
        return []
    samples: list[tuple[list[float], list[float]]] = []
    for sample in getattr(face, "interior_samples", []) or []:
        point = sample.get("point")
        normal = sample.get("normal")
        if isinstance(point, list) and isinstance(normal, list):
            samples.append((point, normal))
    return samples


def material_is_outside(model: PartModel, face_id: str) -> bool | None:
    """Does the solid lie *outside* this face's revolved surface?

    True for a pocket's internal corner fillet (a concave-in-plan corner, the
    one the DFM corner-radius rule is about), False for the convex corner of a
    boss or a re-entrant outline.  Decided from the sampled outward normal
    against the radial direction — the face's ``is_reversed`` flag is not a
    substitute, as the corpus contains cavity corner cylinders with both
    orientations.
    """
    face = model.faces.get(face_id)
    if face is None:
        return None
    axis = _unit(face.parameters.get("axis_direction"))
    origin = face.parameters.get("axis_location") or face.parameters.get("apex")
    if axis is None or not isinstance(origin, (list, tuple)):
        return None
    votes: list[bool] = []
    for point, normal in surface_samples(model, face_id):
        rel = [float(point[i]) - float(origin[i]) for i in range(3)]
        along = _dot(rel, axis)
        radial = [rel[i] - along * axis[i] for i in range(3)]
        unit_radial = _unit(radial)
        if unit_radial is None:
            continue
        votes.append(_dot(unit_radial, normal) < 0.0)
    if not votes:
        return None
    if all(votes):
        return True
    if not any(votes):
        return False
    return None  # samples disagree — do not guess a corner sense


# ---------------------------------------------------------------------------
# Rim sampling and plane fitting
# ---------------------------------------------------------------------------

def rim_points(model: PartModel, index: EdgeIndex, rim_sigs) -> list[list[float]]:
    """Dense 3D sample points along a rim, in canonical-edge order."""
    points: list[list[float]] = []
    for sig in sorted(rim_sigs):
        edge = model.edges.get(index.representative(sig))
        if edge is None:
            continue
        samples = [[float(c) for c in point] for point in (getattr(edge, "sample_points", None) or [])]
        if not samples:
            samples = [
                [float(c) for c in model.vertices[vid].coordinates]
                for vid in (getattr(edge, "vertex_ids", None) or [])
                if vid in model.vertices
            ]
        points.extend(samples)
    return points


def _fit_plane(points: list[list[float]]):
    """Total-least-squares plane through the rim points.

    ``numpy.linalg.eigh`` rather than a hand-rolled iteration: the covariance
    spectrum of a real rim is badly conditioned for power iteration — an
    elongated rectangular mouth gives a second/first shifted-eigenvalue ratio
    around 0.89, so even 64 sweeps leave a ~5e-4 rad tilt in the normal.  That
    is invisible in the normal itself and shows up only as a fabricated
    planarity residual, which then rejects perfectly planar openings.

    numpy is imported here rather than at module level: the NX-line chain
    imports this module without ever fitting a band frame, and NX's bundled
    Python has no numpy - the dependency belongs to the code that uses it.
    """
    import numpy as np

    array = np.asarray(points, dtype=float)
    centroid = array.mean(axis=0)
    relative = array - centroid
    extent = float(np.linalg.norm(relative, axis=1).max()) if len(relative) else 0.0
    eigenvalues, eigenvectors = np.linalg.eigh(relative.T @ relative)
    # A rim that spans only one direction has two near-zero eigenvalues and no
    # well-defined plane; refuse rather than pick an arbitrary one.
    if eigenvalues[1] <= max(1.0e-12, eigenvalues[2] * 1.0e-9):
        return centroid.tolist(), None, 0.0, extent
    normal = _unit(eigenvectors[:, 0].tolist())
    if normal is None:
        return centroid.tolist(), None, 0.0, extent
    residual = float(np.abs(relative @ np.asarray(normal)).max())
    return centroid.tolist(), normal, residual, extent


def _orthonormal_basis(axis: list[float]):
    seed = [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0]
    u = _unit(_cross(seed, axis))
    v = _unit(_cross(axis, u))
    return u, v


# ---------------------------------------------------------------------------
# Footprint: corners, straights, width
# ---------------------------------------------------------------------------

def ordered_rim_loop(model: PartModel, index: EdgeIndex, rim_sigs) -> list[list[float]] | None:
    """The rim as one closed 3D polyline, edges walked in cycle order.

    Rim signature sets carry no ordering, and the footprint decomposition needs
    the actual walk, so edges are chained through their endpoints (merged on
    rounded coordinates, matching the equivalence the rest of the module uses
    for duplicate entities).  Returns ``None``
    if the rim does not close into a single cycle.
    """
    pieces: list[list[list[float]]] = []
    for sig in sorted(rim_sigs):
        edge = model.edges.get(index.representative(sig))
        if edge is None:
            continue
        points = [[float(c) for c in p] for p in (getattr(edge, "sample_points", None) or [])]
        if len(points) < 2:
            points = [
                [float(c) for c in model.vertices[vid].coordinates]
                for vid in (getattr(edge, "vertex_ids", None) or [])
                if vid in model.vertices
            ]
        if len(points) >= 2:
            pieces.append(points)
    if not pieces:
        return None

    def key(point):
        return tuple(round(value, _VERTEX_MERGE_ROUND) for value in point)

    incident: dict[tuple, list[int]] = {}
    for idx, points in enumerate(pieces):
        incident.setdefault(key(points[0]), []).append(idx)
        incident.setdefault(key(points[-1]), []).append(idx)

    used = [False] * len(pieces)
    loop = list(pieces[0])
    used[0] = True
    start_key, cursor = key(pieces[0][0]), key(pieces[0][-1])
    while cursor != start_key:
        nxt = None
        for idx in incident.get(cursor, ()):
            if not used[idx]:
                nxt = idx
                break
        if nxt is None:
            return None
        points = pieces[nxt]
        if key(points[0]) != cursor:
            points = list(reversed(points))
        used[nxt] = True
        loop.extend(points[1:])
        cursor = key(points[-1])
    if not all(used):
        return None
    return loop


@dataclass
class FootprintProfile:
    """The projected outline decomposed into corners, straights and excursions.

    An **excursion** is a curved run that returns the boundary to the direction
    it arrived in — a scallop, a semicircular lobe, a tangent bump on a wall.
    It is kept apart from the corners because it is not one: its net turn is
    zero, so calling it a corner would give it an interior angle of 180 degrees,
    which says nothing, and would leave its concavity to be decided by
    floating-point noise on the ``< 180`` comparison.  It is not a straight
    stretch either, so it is reported in its own right.
    """

    corners: list[dict] = field(default_factory=list)
    straights: list[dict] = field(default_factory=list)
    excursions: list[dict] = field(default_factory=list)

    @property
    def concave_corners(self) -> list[dict]:
        return [item for item in self.corners if item["concave"]]


def footprint_profile(frame: BandFrame, loop: list[list[float]], turn_epsilon_deg: float = 1.0e-4) -> FootprintProfile | None:
    """Decompose the axial projection of the rim into corners and straights.

    Walking the footprint, each vertex contributes a signed turn.  A **corner**
    is a maximal run of consecutive turning vertices, and its interior angle is
    ``180 - orientation * total turn of the run``; a **straight** is a maximal
    run between corners.

    Accumulating over a run rather than reading single vertices is what makes a
    filleted corner and a sharp corner report the same angle: a fillet spreads
    the same total turn over its arc samples, so a 90-degree corner reads 90
    whether or not it carries a radius.  ``blended`` records which of the two it
    was, which is how the corner *radius* is later known to be non-zero.

    A turning run whose turns *cancel* is not a corner: the boundary leaves in
    the direction it arrived, which is what a scallop or a tangent bump on a
    wall does.  Such a run is reported as an **excursion** instead, because
    admitting it as a corner would hand it an interior angle of 180 degrees and
    let floating-point noise decide, through ``interior < 180``, whether it
    counts as concave.  Every corner that survives is therefore a genuine
    direction change, and its concavity is decided with real margin.

    A boundary with no straight stretch at all — a circular pocket — has no
    corners, and none are invented.  ``None`` means the projection is not a
    simple closed profile and nothing about it should be interpreted.
    """
    planar = [frame.plane_uv(point) for point in loop]
    cleaned: list[tuple[float, float]] = []
    for point in planar:
        if not cleaned or _distance2(cleaned[-1], point) > 1.0e-18:
            cleaned.append(point)
    if len(cleaned) > 1 and _distance2(cleaned[0], cleaned[-1]) <= 1.0e-18:
        cleaned.pop()
    count = len(cleaned)
    if count < 3:
        return None

    turns = [_signed_turn(cleaned[i - 1], cleaned[i], cleaned[(i + 1) % count]) for i in range(count)]
    total = sum(turns)
    if abs(abs(total) - 360.0) > 1.0:
        return None  # not a simple closed footprint
    orientation = 1.0 if total > 0.0 else -1.0

    turning = [abs(value) > turn_epsilon_deg for value in turns]
    if all(turning):
        return FootprintProfile()  # smooth closed profile: no corner, no straight
    start = next(i for i in range(count) if not turning[i])

    groups: list[tuple[bool, list[int]]] = []
    for offset in range(count):
        i = (start + offset) % count
        if groups and groups[-1][0] == turning[i]:
            groups[-1][1].append(i)
        else:
            groups.append((turning[i], [i]))
    # The walk starts on a straight vertex, so a trailing straight group is the
    # same physical stretch as the leading one, wrapped.
    if len(groups) > 1 and not groups[-1][0]:
        groups[0] = (False, groups[-1][1] + groups[0][1])
        groups.pop()

    profile = FootprintProfile()
    for is_turning, indices in groups:
        if is_turning:
            net_turn = sum(turns[j] for j in indices)
            if abs(net_turn) <= _CORNER_MIN_TURN_DEG:
                profile.excursions.append({
                    "netTurnDeg": round(net_turn, 6),
                    "vertexCount": len(indices),
                })
                continue
            interior = 180.0 - orientation * net_turn
            profile.corners.append({
                # ``+ 0.0`` normalises the signed zero a corner whose flanking
                # walls are exactly antiparallel produces (the end of an
                # obround outline turns the full 180 degrees).
                "interiorAngleDeg": round(interior, 4) + 0.0,
                "concave": interior < 180.0,
                "vertexCount": len(indices),
                "blended": len(indices) > 1,
            })
        else:
            head = cleaned[(indices[0] - 1) % count]
            tail = cleaned[(indices[-1] + 1) % count]
            direction = _unit2((tail[0] - head[0], tail[1] - head[1]))
            if direction is None:
                continue
            profile.straights.append({
                "point": head,
                "direction": direction,
                "length": round(math.hypot(tail[0] - head[0], tail[1] - head[1]), 6),
            })
    return profile


def rectangular_width(profile: FootprintProfile, angle_tolerance_deg: float = 0.5) -> tuple[float | None, dict]:
    """Width of a right-angled quadrilateral footprint, measured wall to wall.

    Restricted to four concave 90-degree corners on purpose: for a square or a
    rectangle "the width" is unambiguous — the shorter of the two wall-to-wall
    distances, and the side length when the two are equal.  On any other
    outline (pentagon, trapezoid, re-entrant profile, rounded blob) there is no
    single defensible width, so none is reported.

    Measured as the perpendicular distance between opposite straight stretches,
    which is exact and unaffected by corner fillets — a rounded rectangle keeps
    the full wall-to-wall width its straight walls define.
    """
    corners, straights = profile.corners, profile.straights
    diagnostics: dict = {
        "cornerCount": len(corners),
        "straightCount": len(straights),
        "interiorAngles": [item["interiorAngleDeg"] for item in corners],
    }
    if len(corners) != 4 or len(straights) != 4:
        diagnostics["reason"] = "footprint is not a quadrilateral"
        return None, diagnostics
    if any(not item["concave"] or abs(item["interiorAngleDeg"] - 90.0) > angle_tolerance_deg for item in corners):
        diagnostics["reason"] = "footprint corners are not all 90 degrees"
        return None, diagnostics

    spans = []
    for offset in (0, 1):
        first, third = straights[offset], straights[offset + 2]
        normal = (-first["direction"][1], first["direction"][0])
        delta = (third["point"][0] - first["point"][0], third["point"][1] - first["point"][1])
        spans.append(abs(delta[0] * normal[0] + delta[1] * normal[1]))
    diagnostics["wallToWallSpans"] = [round(value, 6) for value in spans]
    diagnostics["shape"] = "square" if abs(spans[0] - spans[1]) <= 1.0e-6 else "rectangle"
    return min(spans), diagnostics


def rectangular_shape_width(
    profile: FootprintProfile,
    angle_tolerance_deg: float = 0.5,
) -> tuple[float | None, dict]:
    """Width of a footprint that is a rectangle, not merely right-angled.

    ``rectangular_width`` asks only for a right-angled quadrilateral, which is
    all a *dimension* needs.  This is the **shape classification** sibling: it
    adds what the ontology's rectangular leaves ask for beyond four right
    angles ("rechteckiger Grundriss, gegenueberliegende Seiten gleich lang"),
    namely opposite walls that are parallel.  For an exact quadrilateral that
    is a corollary of the four right angles; it is asserted explicitly because
    each corner is only right to within ``angle_tolerance_deg``, and two
    corners leaning the same way put opposite walls out by twice that, which
    is a skewed quadrilateral rather than a rectangle.

    Parallelism, and not the sides being *equal in length*, is what the
    ontology's phrase turns into once blended corners are admitted: a straight
    stretch is shortened by whatever radius its two neighbours carry, so on a
    rectangle whose corner radii differ the opposite stretches are legitimately
    unequal while the rectangle is perfectly good.  The wall lengths stay in
    the diagnostics, where they inform without deciding.

    A **blended corner is accepted**.  The decomposition accumulates the turn
    over a corner's whole run, so a filleted 90-degree corner reads 90 degrees
    exactly as a sharp one does: it is the same rectangle, drawn by a tool that
    has a radius.  Keying the leaf class on sharpness instead named a shape
    that never reaches the shop — a rotating cutter cannot leave a zero-radius
    internal corner, so every real rectangular pocket landed in the general
    leaf, and the rules written for a rectangle never met one.  Nothing is lost
    by admitting it: the corner *radius* is published as an attribute, where a
    zero is a measurement rather than a class definition.  How deep a blend may
    cut into a side is not asked either — a blend that swallows a side leaves
    fewer than four straight stretches, and the quadrilateral test already
    refuses that.
    """
    width, diagnostics = rectangular_width(profile, angle_tolerance_deg)
    diagnostics["blendedCornerCount"] = sum(1 for corner in profile.corners if corner["blended"])
    diagnostics["excursionCount"] = len(profile.excursions)
    if width is None:
        return None, diagnostics

    diagnostics["wallLengths"] = [item["length"] for item in profile.straights]
    for offset in (0, 1):
        first, third = profile.straights[offset], profile.straights[offset + 2]
        alignment = abs(first["direction"][0] * third["direction"][0] + first["direction"][1] * third["direction"][1])
        if alignment < math.cos(math.radians(angle_tolerance_deg)):
            diagnostics["reason"] = f"opposite walls {offset} and {offset + 2} are not parallel"
            return None, diagnostics
    return width, diagnostics


def footprint_interior_point(frame: BandFrame, loop: list[list[float]] | None) -> tuple[float, float] | None:
    """One point strictly inside the axial projection of a rim, in frame ``uv``.

    The area centroid is the natural candidate and is used whenever it lands
    inside, which covers every convex outline.  A re-entrant outline (an
    L-shaped window, a C-shaped cut) can put its centroid in the material, so
    the fallback is a scan line: the widest interior span found on horizontal
    cuts taken between consecutive distinct vertex heights, whose midpoint is
    inside by construction.  ``None`` means the projection encloses no area and
    nothing can be said about its interior.
    """
    planar = _planar_polygon(frame, loop)
    if planar is None:
        return None

    centroid = _polygon_centroid(planar)
    if centroid is not None and _point_in_polygon(planar, centroid):
        return centroid

    heights = sorted({v for _, v in planar})
    best: tuple[float, tuple[float, float]] | None = None
    for lower, upper in zip(heights, heights[1:]):
        height = 0.5 * (lower + upper)
        crossings = sorted(_scanline_crossings(planar, height))
        for left, right in zip(crossings[0::2], crossings[1::2]):
            width = right - left
            if width > 0.0 and (best is None or width > best[0]):
                best = (width, (0.5 * (left + right), height))
    return best[1] if best is not None else None


def _planar_polygon(frame: BandFrame, loop: list[list[float]] | None) -> list[tuple[float, float]] | None:
    """The rim projected into the frame plane, deduplicated and open."""
    if not frame.resolved or not loop:
        return None
    planar: list[tuple[float, float]] = []
    for point in loop:
        uv = frame.plane_uv(point)
        if not planar or _distance2(planar[-1], uv) > 1.0e-18:
            planar.append(uv)
    if len(planar) > 1 and _distance2(planar[0], planar[-1]) <= 1.0e-18:
        planar.pop()
    return planar if len(planar) >= 3 else None


def _polygon_centroid(planar: list[tuple[float, float]]) -> tuple[float, float] | None:
    twice_area = accumulated_u = accumulated_v = 0.0
    for position in range(len(planar)):
        u0, v0 = planar[position]
        u1, v1 = planar[(position + 1) % len(planar)]
        cross = u0 * v1 - u1 * v0
        twice_area += cross
        accumulated_u += (u0 + u1) * cross
        accumulated_v += (v0 + v1) * cross
    if abs(twice_area) <= 1.0e-12:
        return None
    return (accumulated_u / (3.0 * twice_area), accumulated_v / (3.0 * twice_area))


def _point_in_polygon(planar: list[tuple[float, float]], point: tuple[float, float]) -> bool:
    inside = False
    for position in range(len(planar)):
        u0, v0 = planar[position]
        u1, v1 = planar[(position + 1) % len(planar)]
        if (v0 > point[1]) == (v1 > point[1]):
            continue
        crossing = u0 + (point[1] - v0) * (u1 - u0) / (v1 - v0)
        if crossing > point[0]:
            inside = not inside
    return inside


def _scanline_crossings(planar: list[tuple[float, float]], height: float) -> list[float]:
    crossings: list[float] = []
    for position in range(len(planar)):
        u0, v0 = planar[position]
        u1, v1 = planar[(position + 1) % len(planar)]
        if (v0 > height) == (v1 > height):
            continue
        crossings.append(u0 + (height - v0) * (u1 - u0) / (v1 - v0))
    return crossings


# ---------------------------------------------------------------------------
# Round geometry: rim circles and cylindrical walls
# ---------------------------------------------------------------------------

@dataclass
class RimCircle:
    """Least-squares circle through one rim, measured in the band frame."""

    center: list[float]
    center_uv: tuple[float, float]
    radius: float
    axial_level: float
    max_radial_deviation: float
    max_axial_deviation: float
    winding: float
    sample_count: int

    @property
    def budget(self) -> float:
        return max(_CIRCLE_ABS_TOL, self.radius * _CIRCLE_REL_TOL)

    @property
    def is_closed_circle(self) -> bool:
        """One complete circle: round, flat, and closing exactly one revolution.

        Radial deviation is the load-bearing test — it is what rejects an
        ellipse, a rounded rectangle, or a D-shape whose straight side is
        sampled along its length.  Out-of-plane deviation rejects a rim that
        steps or spirals, and the winding number rejects a walk that encircles
        the centre more than once or doubles back on itself.
        """
        return (
            self.radius > 0.0
            and self.max_radial_deviation <= self.budget
            and self.max_axial_deviation <= self.budget
            and abs(abs(self.winding) - 1.0) <= _FULL_TURN_TOL
        )

    def to_dict(self) -> dict:
        return {
            "center": [round(value, 6) for value in self.center],
            "radius": round(self.radius, 6),
            "axialLevel": round(self.axial_level, 6),
            "maxRadialDeviation": round(self.max_radial_deviation, 9),
            "maxAxialDeviation": round(self.max_axial_deviation, 9),
            "budget": round(self.budget, 9),
            "winding": round(self.winding, 6),
            "sampleCount": self.sample_count,
            "isClosedCircle": self.is_closed_circle,
        }


def rim_circle(frame: BandFrame, loop: list[list[float]] | None) -> RimCircle | None:
    """Fit a circle to a rim in the plane perpendicular to the feature axis.

    An algebraic (Kasa) fit rather than a centroid-and-mean-radius estimate:
    rim sample points are distributed per edge, not uniformly around the rim, so
    a centroid is biased by however the exporter split the curve while the
    least-squares centre is not.  ``None`` means the rim carries too few
    distinct points for any circle to be spoken about.
    """
    planar = _planar_polygon(frame, loop)
    if planar is None:
        return None
    import numpy as np

    array = np.asarray(planar, dtype=float)
    lhs = np.column_stack([2.0 * array[:, 0], 2.0 * array[:, 1], np.ones(len(array))])
    rhs = (array ** 2).sum(axis=1)
    try:
        solution = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    center_u, center_v, offset = (float(value) for value in solution)
    squared = offset + center_u * center_u + center_v * center_v
    if not math.isfinite(squared) or squared <= 0.0:
        return None
    radius = math.sqrt(squared)
    radii = np.hypot(array[:, 0] - center_u, array[:, 1] - center_v)

    depths = [frame.depth_of(point) for point in loop]
    level = sum(depths) / len(depths)
    return RimCircle(
        center=frame.point_at(center_u, center_v, level),
        center_uv=(center_u, center_v),
        radius=radius,
        axial_level=level,
        max_radial_deviation=float(np.abs(radii - radius).max()),
        max_axial_deviation=max(abs(value - level) for value in depths),
        winding=_winding(array, center_u, center_v),
        sample_count=len(planar),
    )


def cylindrical_wall_radius(model: PartModel, frame: BandFrame, face_ids) -> tuple[float | None, dict]:
    """The single radius of a band that is one coaxial cylindrical surface.

    A cylindrical wall reaches the band either as one merged face or, when the
    parser conservatively skipped the seam merge, as several coaxial fragments
    of the same surface.  Both are the same wall, so the test is on the geometry
    (surface type, axis line, radius) rather than on the face count.  Any
    non-cylindrical face, any radius disagreement and any axis that is not the
    feature axis makes the band something other than a cylindrical wall.
    """
    ordered = sorted(set(face_ids), key=face_sort_key)
    records: list[dict] = []
    diagnostics: dict = {"faces": records}
    if not frame.resolved:
        diagnostics["reason"] = "the band has no feature axis"
        return None, diagnostics
    if not ordered:
        diagnostics["reason"] = "the band has no wall face"
        return None, diagnostics

    radii: list[float] = []
    anchor: tuple[list[float], list[float]] | None = None
    for face_id in ordered:
        face = model.faces.get(face_id)
        surface = face.surface_type.value if face is not None else "UNKNOWN"
        record = {"faceId": face_id, "surfaceType": surface}
        records.append(record)
        if face is None or face.surface_type != SurfaceType.CYLINDER:
            diagnostics["reason"] = f"wall face {face_id} is {surface}, not a cylinder"
            return None, diagnostics
        axis = _unit(face.parameters.get("axis_direction"))
        location = face.parameters.get("axis_location")
        radius = face.parameters.get("radius")
        if axis is None or not isinstance(location, (list, tuple)) or not isinstance(radius, (int, float)) or float(radius) <= 0.0:
            diagnostics["reason"] = f"wall face {face_id} carries no usable cylinder parameters"
            return None, diagnostics
        record["radius"] = round(float(radius), 6)
        if not parallel(axis, frame.axis):
            diagnostics["reason"] = f"wall face {face_id} is a cylinder about a different axis than the feature axis"
            return None, diagnostics
        if anchor is None:
            anchor = (axis, [float(value) for value in location])
        else:
            offset = _lateral_offset(anchor[1], [float(value) for value in location], anchor[0])
            record["axisOffset"] = round(offset, 9)
            if offset > _AXIS_OFFSET_TOL:
                diagnostics["reason"] = f"wall face {face_id} is not coaxial with the rest of the band ({offset:.6g} mm off axis)"
                return None, diagnostics
        radii.append(float(radius))

    if max(radii) - min(radii) > _RADIUS_TOL:
        diagnostics["reason"] = f"wall radii disagree: {min(radii):.6f} .. {max(radii):.6f}"
        return None, diagnostics
    radius = sum(radii) / len(radii)
    diagnostics["radius"] = round(radius, 6)
    diagnostics["axisLocation"] = [round(value, 6) for value in anchor[1]] if anchor else None
    return radius, diagnostics


def lateral_offset(frame: BandFrame, left, right) -> float | None:
    """Distance between two points measured in the plane across the axis."""
    if not frame.resolved or not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return None
    return _lateral_offset([float(value) for value in left], [float(value) for value in right], frame.axis)


def _lateral_offset(origin: list[float], point: list[float], axis: list[float]) -> float:
    delta = [point[i] - origin[i] for i in range(3)]
    along = _dot(delta, axis)
    return math.sqrt(sum((delta[i] - along * axis[i]) ** 2 for i in range(3)))


def _winding(array, center_u: float, center_v: float) -> float:
    """Turns the closed sample sequence makes around the fitted centre."""
    angles = [math.atan2(float(v) - center_v, float(u) - center_u) for u, v in array]
    total = 0.0
    for position in range(len(angles)):
        delta = angles[(position + 1) % len(angles)] - angles[position]
        while delta > math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi
        total += delta
    return total / (2.0 * math.pi)


# ---------------------------------------------------------------------------
# Axial level and footprint centroid
# ---------------------------------------------------------------------------

@dataclass
class AxialLevel:
    """Where a point set sits along the axis, and whether it sits at one level."""

    level: float
    minimum: float
    maximum: float
    spread: float
    extent: float
    sample_count: int

    @property
    def budget(self) -> float:
        return max(1.0e-9, self.extent * _PLANARITY_RATIO)

    @property
    def flat(self) -> bool:
        return self.spread <= self.budget

    def to_dict(self) -> dict:
        return {
            "level": round(self.level, 6),
            "minimum": round(self.minimum, 6),
            "maximum": round(self.maximum, 6),
            "spread": round(self.spread, 9),
            "budget": round(self.budget, 9),
            "flat": self.flat,
            "sampleCount": self.sample_count,
        }


def axial_level(frame: BandFrame, points: list[list[float]] | None) -> AxialLevel | None:
    """Axial position of a rim relative to the opening plane.

    The spread is what makes the value trustworthy or not: a rim that runs at
    one fixed level has a single axial position, and one that does not (a boss
    grown on a stepped or slanted support, a rim following an intersection
    curve) has none.  ``flat`` is judged against the rim's own size, the same
    relative budget the opening-plane fit uses.
    """
    if not frame.resolved or not points:
        return None
    depths = [frame.depth_of(point) for point in points]
    centroid = [sum(point[axis] for point in points) / len(points) for axis in range(3)]
    extent = max(math.dist([float(v) for v in point], centroid) for point in points)
    level = sum(depths) / len(depths)
    return AxialLevel(
        level=level,
        minimum=min(depths),
        maximum=max(depths),
        spread=max(depths) - min(depths),
        extent=extent,
        sample_count=len(depths),
    )


def footprint_centroid(frame: BandFrame, loop: list[list[float]] | None) -> list[float] | None:
    """Area centroid of the axial projection of a rim, returned in 3D.

    The *area* centroid, not the mean of the sample points: rim samples are
    distributed per edge, so on a rectangle with one long and one short wall the
    sample mean drifts toward the short wall while the area centroid does not.
    Returned at the rim's own mean axial level, so it lies on the footprint
    rather than somewhere along the axis.
    """
    planar = _planar_polygon(frame, loop)
    if planar is None:
        return None
    centroid = _polygon_centroid(planar)
    if centroid is None:
        return None
    level = sum(frame.depth_of(point) for point in loop) / len(loop)
    return frame.point_at(centroid[0], centroid[1], level)


def _unit2(vector):
    norm = math.hypot(vector[0], vector[1])
    if norm <= 1.0e-15:
        return None
    return (vector[0] / norm, vector[1] / norm)


def _signed_turn(prev, here, nxt) -> float:
    ax, ay = here[0] - prev[0], here[1] - prev[1]
    bx, by = nxt[0] - here[0], nxt[1] - here[1]
    if (ax * ax + ay * ay) <= 0.0 or (bx * bx + by * by) <= 0.0:
        return 0.0
    return math.degrees(math.atan2(ax * by - ay * bx, ax * bx + ay * by))


def _distance2(a, b) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


# ---------------------------------------------------------------------------
# Small vector helpers
# ---------------------------------------------------------------------------

def _dot(a, b) -> float:
    return sum(float(a[i]) * float(b[i]) for i in range(3))


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _unit(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        components = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in components):
        return None
    norm = math.sqrt(sum(v * v for v in components))
    if norm <= 1.0e-12:
        return None
    return [v / norm for v in components]


def parallel(a, b, tolerance_deg: float = _ANGLE_TOL_DEG) -> bool:
    ua, ub = _unit(a), _unit(b)
    if ua is None or ub is None:
        return False
    return abs(_dot(ua, ub)) >= math.cos(math.radians(tolerance_deg))
