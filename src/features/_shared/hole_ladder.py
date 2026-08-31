"""Coaxial hole ladders: the shared structural reading the D-7 grammar dispatches on.

This module answers one question about the part's geometry — *which cylindrical faces form
one coaxial stepped bore, in what order, and what sits on its two ends* — and answers it once.
It registers no node, names no ontology class and claims no face, exactly like
``closed_wall_band.discover_wall_bands``.  Three feature modules read the result:

``Countersink``   a cone joins the compound only on a **single-cylinder** run, at its entry end
``Counterbore``   a run of **exactly two** cylinders
``StepBore``      a run of **three or more**: x-1 step bores plus the terminal hole

Why this is one analysis instead of three matchers.  The previous resolver scanned for pairs:
every adjacent (smaller, larger) pair of coaxial cylinders was a Counterbore candidate on its
own.  On a four-step bore that yields three overlapping Counterbores — the middle cylinders end
up owned by two features each and the innermost segment is owned by none.  A pair is simply not
enough context to decide what a segment *is*; the whole run is.  Deciding once, on the run,
makes the three dispatch outcomes mutually exclusive by construction.

The orientation axiom (D-7 (0)).  Axial material removal requires the diameter not to increase
along the tool's feed, so the widest end of a run is the only end a tool can enter from.  That
is what fixes ``machining_direction`` for every segment and what "entry" means below.  A run
whose diameters are not monotone from a single widest end is a bore machined from both sides:
there is no single entry, the axiom does not pick one, and this module refuses to guess — the
run is reported invalid with a reason and its cylinders keep whatever the low-level hole layer
made of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from contract.partmodel import PartModel, SurfaceType

from features._shared.compound_resolution import (
    _AXIAL_TOL,
    _AXIS_DISTANCE_TOL,
    _AXIS_TOL,
    _RADIUS_TOL,
    _axis_angle,
    _cone_circles,
    _complete_cylindrical_section,
    _direct_rim_junction,
    _face_interval,
    _line_distance,
    _positive,
    _validate_counterbore_shoulder,
)
from features._shared.hole_topology import unit_vector, valid_vector


@dataclass(frozen=True)
class LadderSegment:
    """One cylindrical section of a run, with the transitions closing its two ends.

    Two kinds of transition step a bore down: a planar annular shoulder, and a coaxial cone.
    D-7 (3) dispatches on *the ladder left after the cones are stripped*, so both have to chain
    two cylinders into one run; the difference is only who owns the transition face afterwards.
    A shoulder belongs to the section it caps, a cone becomes a Chamfer atom of its own.
    """

    faceId: str
    radius: float
    interval: tuple[float, float]
    # Where this section is closed toward the next, narrower one, and by what.  The terminal
    # section has neither: its minor end is the bore's bottom, or the far opening of a through
    # bore.
    minorCoordinate: float | None = None
    transitionFaceId: str | None = None
    transitionKind: str | None = None          # "shoulder" | "cone"
    transitionProof: dict[str, Any] | None = None
    # Where the previous, wider section closes onto this one.  The entry section has none.
    majorCoordinate: float | None = None

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def length(self) -> float:
        return abs(self.interval[1] - self.interval[0])

    @property
    def shoulderFaceId(self) -> str | None:
        """The transition face this section *owns* - only a planar shoulder qualifies."""
        return self.transitionFaceId if self.transitionKind == "shoulder" else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "faceId": self.faceId,
            "radius": round(self.radius, 6),
            "interval": [round(self.interval[0], 6), round(self.interval[1], 6)],
            "transitionFaceId": self.transitionFaceId,
            "transitionKind": self.transitionKind,
            "minorCoordinate": (round(self.minorCoordinate, 6)
                                if self.minorCoordinate is not None else None),
            "majorCoordinate": (round(self.majorCoordinate, 6)
                                if self.majorCoordinate is not None else None),
        }


@dataclass(frozen=True)
class ConeAttachment:
    """A conical face joined to one end of a run by a proven rim junction."""

    faceId: str
    end: str                      # "entry" | "far" — which end of the run it sits on
    cylinderFaceId: str
    smallRadius: float
    largeRadius: float
    interval: tuple[float, float]
    smallBoundary: dict[str, Any]
    largeBoundary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "faceId": self.faceId,
            "end": self.end,
            "cylinderFaceId": self.cylinderFaceId,
            "smallRadius": round(self.smallRadius, 6),
            "largeRadius": round(self.largeRadius, 6),
        }


@dataclass
class HoleRun:
    """One maximal coaxial run of cylinders joined end to end by annular shoulders.

    ``segments`` is ordered from the entry (widest) end inward, so ``segments[0]`` is the
    section a tool reaches first and ``segments[-1]`` is the terminal one.
    """

    segments: list[LadderSegment]
    axis: list[float]
    origin: list[float]
    entryCoordinate: float
    inwardSign: float
    cones: list[ConeAttachment] = field(default_factory=list)
    valid: bool = True
    reason: str | None = None

    @property
    def size(self) -> int:
        return len(self.segments)

    @property
    def faceIds(self) -> list[str]:
        out: list[str] = []
        for segment in self.segments:
            out.append(segment.faceId)
            if segment.shoulderFaceId:
                out.append(segment.shoulderFaceId)
        return out

    @property
    def machiningDirection(self) -> list[float]:
        """Entry -> terminal, i.e. the direction a tool feeds along this run."""
        return [value * self.inwardSign for value in self.axis]

    def cone_at(self, end: str) -> ConeAttachment | None:
        for cone in self.cones:
            if cone.end == end:
                return cone
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "valid": self.valid,
            "reason": self.reason,
            "axis": [round(value, 6) for value in self.axis],
            "entryCoordinate": round(self.entryCoordinate, 6),
            "diameters": [round(segment.diameter, 6) for segment in self.segments],
            "segments": [segment.to_dict() for segment in self.segments],
            "cones": [cone.to_dict() for cone in self.cones],
        }


@dataclass
class LadderAnalysis:
    """Every run found in the part, with the lookups the three consumers need."""

    runs: list[HoleRun] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    _by_face: dict[str, HoleRun] = field(default_factory=dict)

    # -- lookups ----------------------------------------------------------
    def run_for_face(self, face_id: str) -> HoleRun | None:
        """The run a cylindrical face belongs to, valid or not."""
        return self._by_face.get(str(face_id))

    def valid_run_for_face(self, face_id: str) -> HoleRun | None:
        run = self.run_for_face(face_id)
        return run if run is not None and run.valid else None

    def runs_of_size(self, size: int) -> list[HoleRun]:
        return [run for run in self.runs if run.valid and run.size == size]

    def runs_of_size_at_least(self, size: int) -> list[HoleRun]:
        return [run for run in self.runs if run.valid and run.size >= size]

    def countersink_cone(self, cylinder_face_id: str) -> ConeAttachment | None:
        """The one cone allowed to form a Countersink with this cylinder, if any.

        D-7 (1): a cone joins the compound only when the cylinder stands alone — a lone section,
        not a rung of a ladder — and only at the entry end, which by the orientation axiom is the
        end where the cone widens outward.  Every other cone is an atom.
        """
        run = self.valid_run_for_face(cylinder_face_id)
        if run is None or run.size != 1:
            return None
        cone = run.cone_at("entry")
        return cone if cone is not None and cone.cylinderFaceId == str(cylinder_face_id) else None

    @property
    def released_cone_face_ids(self) -> frozenset[str]:
        """Cones the grammar declined to absorb, which are therefore Chamfer atoms.

        The Chamfer module registers transition regions AAGNet proposed.  A cone released here
        may not be among them — the vote is a hint, and a large screw seat does not look like a
        chamfer to a classifier — so the grammar hands over its own geometric proof instead.
        Without this the released cone would be owned by nobody: the residual passes register
        planes and freeform surfaces, never cones.
        """
        released: set[str] = set()
        for run in self.runs:
            for cone in run.cones:
                if not run.valid or run.size != 1 or cone.end != "entry":
                    released.add(cone.faceId)
            if not run.valid:
                # An unproven run keeps whatever the low-level layer made of it, transitions
                # included; nothing here has the standing to release them.
                continue
            for segment in run.segments:
                if segment.transitionKind == "cone" and segment.transitionFaceId:
                    released.add(segment.transitionFaceId)
        return frozenset(released)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "method": "coaxial_cylinder_run_with_proven_annular_shoulders_v1",
            "runCount": len(self.runs),
            "runsBySize": {str(size): len([r for r in self.runs if r.valid and r.size == size])
                           for size in sorted({r.size for r in self.runs if r.valid})},
            "invalidRunCount": len([r for r in self.runs if not r.valid]),
            "releasedConeFaceIds": sorted(self.released_cone_face_ids),
            "runs": [run.to_dict() for run in self.runs],
            "rejected": self.rejected,
        }


# ---------------------------------------------------------------- discovery

def discover_hole_ladders(model: PartModel) -> LadderAnalysis:
    """Find every coaxial cylinder run in the part.  Owns nothing, decides nothing."""
    analysis = LadderAnalysis()
    cylinders = [
        face_id for face_id, face in model.faces.items()
        if face.surface_type == SurfaceType.CYLINDER and _positive(face.parameters.get("radius")) is not None
        and unit_vector(face.parameters.get("axis_direction")) is not None
        and valid_vector(face.parameters.get("axis_location"))
    ]
    for group in _group_by_axis_line(model, cylinders):
        for run in _runs_in_group(model, group, analysis.rejected):
            analysis.runs.append(run)
            for segment in run.segments:
                analysis._by_face[segment.faceId] = run
    return analysis


def _group_by_axis_line(model: PartModel, cylinders: list[str]) -> list[list[str]]:
    """Partition cylinders into sets sharing one infinite axis line."""
    groups: list[list[str]] = []
    for face_id in sorted(cylinders, key=_face_sort_key):
        face = model.faces[face_id]
        axis = unit_vector(face.parameters.get("axis_direction"))
        origin = face.parameters.get("axis_location")
        for group in groups:
            other = model.faces[group[0]]
            other_axis = unit_vector(other.parameters.get("axis_direction"))
            other_origin = other.parameters.get("axis_location")
            if (_axis_angle(axis, other_axis) <= _AXIS_TOL
                    and _line_distance(other_origin, origin, other_axis) <= _AXIS_DISTANCE_TOL):
                group.append(face_id)
                break
        else:
            groups.append([face_id])
    return [group for group in groups if group]


def _runs_in_group(model: PartModel, group: list[str], rejected: list[dict[str, Any]]) -> list[HoleRun]:
    """Chain one coaxial group into maximal runs and orient each by the axiom."""
    reference = model.faces[group[0]]
    axis = unit_vector(reference.parameters.get("axis_direction"))
    origin = list(reference.parameters.get("axis_location"))

    members: dict[str, dict[str, Any]] = {}
    for face_id in group:
        face = model.faces[face_id]
        interval = _face_interval(model, face, axis)
        radius = _positive(face.parameters.get("radius"))
        if interval is None or radius is None or abs(interval[1] - interval[0]) <= _AXIAL_TOL:
            continue
        members[face_id] = {"radius": radius, "interval": (min(interval), max(interval))}
    if not members:
        return []

    # Adjacency: an unordered link between two cylinders that meet at one axial coordinate
    # through a *proven* internal annular shoulder.  Touching intervals alone are not enough;
    # a radius step with no real shoulder face is not a bore step.
    links: dict[str, dict[str, dict[str, Any]]] = {face_id: {} for face_id in members}
    ordered = sorted(members, key=_face_sort_key)
    for index, left_id in enumerate(ordered):
        for right_id in ordered[index + 1:]:
            join = _transition_between(model, left_id, right_id, members, axis)
            if join is None:
                continue
            links[left_id][right_id] = join
            links[right_id][left_id] = join

    runs: list[HoleRun] = []
    seen: set[str] = set()
    for start in ordered:
        if start in seen:
            continue
        component = _component(start, links)
        seen |= component
        if len(component) == 1:
            face_id = next(iter(component))
            runs.append(_single_segment_run(model, face_id, members[face_id], axis, origin))
            continue
        run = _chain_run(model, component, links, members, axis, origin, rejected)
        if run is not None:
            runs.append(run)
    for run in runs:
        _attach_cones(model, run, members)
    return runs


def _transition_between(model, left_id: str, right_id: str, members: dict, axis: list[float]) -> dict[str, Any] | None:
    """Prove the step joining two coaxial cylinders - annular shoulder or cone - or return None.

    ``baseCoordinate`` is where the narrower cylinder starts and ``sectionCoordinate`` where the
    wider one ends.  They coincide on a shoulder and differ by the cone's axial extent on a cone.
    """
    left, right = members[left_id], members[right_id]
    if abs(left["radius"] - right["radius"]) <= _RADIUS_TOL:
        return None
    base_id, section_id = ((left_id, right_id) if left["radius"] < right["radius"]
                           else (right_id, left_id))
    base, section = members[base_id], members[section_id]
    contact = _serial_contact(base["interval"], section["interval"])
    if contact is not None:
        proof = _validate_counterbore_shoulder(
            model, base_id, section_id, axis, contact, base["radius"], section["radius"]
        )
        if proof is None:
            return None
        return {"kind": "shoulder", "faceId": proof["supportFaceId"],
                "baseFaceId": base_id, "sectionFaceId": section_id,
                "baseCoordinate": contact, "sectionCoordinate": contact, "proof": proof}
    return _cone_transition(model, base_id, section_id, base, section, axis)


def _cone_transition(model, base_id: str, section_id: str, base: dict, section: dict,
                     axis: list[float]) -> dict[str, Any] | None:
    """A cone stepping one cylinder down to the next, with a cylinder proven on *both* sides.

    Requiring a cylinder at each end is what keeps this apart from a countersink: a countersink
    cone opens to the outside of the part, so it never has a second cylinder to chain to, and a
    lone cylinder with an entry cone stays a run of one.
    """
    for cone_id, cone in model.faces.items():
        if cone.surface_type != SurfaceType.CONE:
            continue
        cone_axis = unit_vector(cone.parameters.get("axis_direction"))
        apex = cone.parameters.get("apex")
        if cone_axis is None or not valid_vector(apex) or _axis_angle(axis, cone_axis) > _AXIS_TOL:
            continue
        if _line_distance(model.faces[base_id].parameters.get("axis_location"), apex, axis) > _AXIS_DISTANCE_TOL:
            continue
        circles = _cone_circles(model, cone_id, axis)
        if len(circles) != 2 or any(not item.get("complete") for item in circles):
            continue
        small = min(circles, key=lambda item: item["radius"])
        large = max(circles, key=lambda item: item["radius"])
        if abs(float(small["radius"]) - base["radius"]) > max(_RADIUS_TOL, base["radius"] * 1.0e-3):
            continue
        if abs(float(large["radius"]) - section["radius"]) > max(_RADIUS_TOL, section["radius"] * 1.0e-3):
            continue
        small_at, large_at = float(small["axisPosition"]), float(large["axisPosition"])
        if not _touches(base["interval"], small_at) or not _touches(section["interval"], large_at):
            continue
        if not _direct_rim_junction(model, base_id, cone_id, small, axis, small_at):
            continue
        if not _direct_rim_junction(model, section_id, cone_id, large, axis, large_at):
            continue
        return {"kind": "cone", "faceId": cone_id,
                "baseFaceId": base_id, "sectionFaceId": section_id,
                "baseCoordinate": small_at, "sectionCoordinate": large_at,
                "proof": {"valid": True, "coneFaceId": cone_id,
                          "smallRadius": round(float(small["radius"]), 6),
                          "largeRadius": round(float(large["radius"]), 6),
                          "smallAxisPosition": round(small_at, 6),
                          "largeAxisPosition": round(large_at, 6)}}
    return None


def _touches(interval: tuple[float, float], value: float) -> bool:
    tolerance = max(_AXIAL_TOL, 1.0e-3 * max(abs(interval[0]), abs(interval[1]), 1.0))
    return abs(interval[0] - value) <= tolerance or abs(interval[1] - value) <= tolerance


def _serial_contact(a: tuple[float, float], b: tuple[float, float]) -> float | None:
    """The single axial coordinate where two intervals meet end to end, if they do.

    Same rule ``_ordered_section_boundaries`` applies to a compound: the only allowed
    intersection is a tolerance-sized endpoint contact, so an embedded interval cannot pass
    as a step.
    """
    scale = max(a[1] - a[0], b[1] - b[0], 1.0)
    tolerance = max(_AXIAL_TOL, 1.0e-3 * scale)
    if max(0.0, min(a[1], b[1]) - max(a[0], b[0])) > tolerance:
        return None
    if abs(b[0] - a[1]) <= tolerance:
        return 0.5 * (b[0] + a[1])
    if abs(a[0] - b[1]) <= tolerance:
        return 0.5 * (a[0] + b[1])
    return None


def _component(start: str, links: dict[str, dict[str, dict]]) -> set[str]:
    out, queue = {start}, [start]
    while queue:
        for neighbour in links[queue.pop()]:
            if neighbour not in out:
                out.add(neighbour)
                queue.append(neighbour)
    return out


def _single_segment_run(model, face_id: str, member: dict, axis: list[float], origin: list[float]) -> HoleRun:
    """A lone cylinder is a run of one; its entry end is decided when a cone is attached."""
    interval = member["interval"]
    return HoleRun(
        segments=[LadderSegment(faceId=face_id, radius=member["radius"], interval=interval)],
        axis=list(axis), origin=list(origin),
        entryCoordinate=interval[1], inwardSign=-1.0,
    )


def _chain_run(model, component: set[str], links, members, axis, origin,
               rejected: list[dict[str, Any]]) -> HoleRun | None:
    """Order a connected component into a path and orient it by the widest end."""
    degrees = {face_id: len(links[face_id]) for face_id in component}
    ends = [face_id for face_id, degree in degrees.items() if degree == 1]
    if any(degree > 2 for degree in degrees.values()) or len(ends) != 2:
        # A branch means one cylinder steps to two different cylinders at the same end, which
        # is not a one-dimensional bore.  Nothing here can be ordered; say so and move on.
        rejected.append({
            "faceIds": sorted(component, key=_face_sort_key),
            "reason": "coaxial_cylinder_group_is_not_a_simple_chain",
            "degrees": {face_id: degrees[face_id] for face_id in sorted(component, key=_face_sort_key)},
        })
        return None

    path = [sorted(ends, key=_face_sort_key)[0]]
    while len(path) < len(component):
        current = path[-1]
        nxt = [face_id for face_id in links[current] if face_id not in path]
        if not nxt:
            break
        path.append(nxt[0])
    if len(path) != len(component):
        rejected.append({"faceIds": sorted(component, key=_face_sort_key),
                         "reason": "coaxial_cylinder_chain_could_not_be_ordered"})
        return None

    # Orientation axiom: enter from the widest end.  Both ends being equally wide, or the
    # diameters not falling monotonically from the widest end, means the bore was reached from
    # two sides; a single entry does not exist and is not invented.
    radii = [members[face_id]["radius"] for face_id in path]
    if radii[-1] > radii[0]:
        path, radii = path[::-1], radii[::-1]
    monotone = all(radii[i] >= radii[i + 1] - _RADIUS_TOL for i in range(len(radii) - 1))

    segments: list[LadderSegment] = []
    for index, face_id in enumerate(path):
        # path runs entry -> terminal, so the next member is always the narrower one: this
        # section is the join's ``section`` (wider) side and the next one is its ``base``.
        onward = links[face_id].get(path[index + 1]) if index + 1 < len(path) else None
        incoming = links[face_id].get(path[index - 1]) if index > 0 else None
        segments.append(LadderSegment(
            faceId=face_id,
            radius=members[face_id]["radius"],
            interval=members[face_id]["interval"],
            minorCoordinate=onward["sectionCoordinate"] if onward else None,
            transitionFaceId=onward["faceId"] if onward else None,
            transitionKind=onward["kind"] if onward else None,
            transitionProof=onward["proof"] if onward else None,
            majorCoordinate=incoming["baseCoordinate"] if incoming else None,
        ))

    entry_interval = segments[0].interval
    far_interval = segments[-1].interval
    inward = 1.0 if far_interval[1] > entry_interval[1] else -1.0
    entry_coordinate = entry_interval[0] if inward > 0 else entry_interval[1]

    run = HoleRun(segments=segments, axis=list(axis), origin=list(origin),
                  entryCoordinate=entry_coordinate, inwardSign=inward)
    if not monotone:
        run.valid = False
        run.reason = "coaxial_run_is_not_monotone_from_a_single_widest_end"
        rejected.append({
            "faceIds": [segment.faceId for segment in segments],
            "diameters": [round(segment.diameter, 6) for segment in segments],
            "reason": run.reason,
        })
    return run


def _free_ends(run: HoleRun) -> list[tuple[LadderSegment, float]]:
    """The two axial coordinates where the run opens: one per terminal segment.

    Every interior rim is taken by a shoulder, so a run has exactly two free ends.  On a run of
    one they are the two ends of the same cylinder, which is why they are addressed by
    coordinate rather than by segment.
    """
    first, last = run.segments[0], run.segments[-1]
    return [(first, _other_end(first.interval, first.minorCoordinate)),
            (last, _other_end(last.interval, last.majorCoordinate))]


def _other_end(interval: tuple[float, float], taken: float | None) -> float:
    if taken is None:
        return interval[1]
    tolerance = max(_AXIAL_TOL, 1.0e-3 * max(abs(taken), 1.0))
    return interval[0] if abs(interval[1] - taken) <= tolerance else interval[1]


def _attach_cones(model: PartModel, run: HoleRun, members: dict) -> None:
    """Find the cones sitting on the two ends of a run and label which end each is on."""
    ends = _free_ends(run)
    if run.size == 1:
        # One cylinder, two distinct free coordinates.  ``_free_ends`` returns the same
        # (segment, coordinate) twice here, so address the two interval endpoints directly.
        segment = run.segments[0]
        ends = [(segment, segment.interval[0]), (segment, segment.interval[1])]

    found: list[ConeAttachment] = []
    for segment, coordinate in ends:
        cone = _cone_at(model, run, segment, coordinate)
        if cone is not None and all(cone.faceId != other.faceId for other in found):
            found.append(cone)

    if run.size > 1:
        # A rung of a ladder never absorbs a cone (D-7 (2)); the labels only record where each
        # cone sat, so the released set and the diagnostics can name it.
        entry_coordinate = _free_ends(run)[0][1]
        run.cones = [_relabel(cone, "entry" if cone.cylinderFaceId == run.segments[0].faceId
                              and abs(cone.smallBoundary["axisPosition"] - entry_coordinate) <= _AXIAL_TOL
                              else "far")
                     for cone in found]
        return

    if len(found) == 2:
        # A lone cylinder with a cone at both ends: the axiom picks the wider cone's end as the
        # entry and the other cone is an atom.  Equal cones make the bore symmetric - no end is
        # distinguished, so neither cone is a countersink.
        a, b = found
        if abs(a.largeRadius - b.largeRadius) <= _RADIUS_TOL:
            run.cones = [_relabel(cone, "far") for cone in found]
            return
        wider, narrower = (a, b) if a.largeRadius > b.largeRadius else (b, a)
        run.cones = [_relabel(wider, "entry"), _relabel(narrower, "far")]
        _orient_single_run_to_cone(run, wider)
        return

    if len(found) == 1:
        # The cone widens outward, so its end is the wide end: by the axiom, the entry.
        run.cones = [_relabel(found[0], "entry")]
        _orient_single_run_to_cone(run, run.cones[0])
        return
    run.cones = []


def _relabel(cone: ConeAttachment, end: str) -> ConeAttachment:
    return ConeAttachment(**{**cone.__dict__, "end": end})


def _orient_single_run_to_cone(run: HoleRun, cone: ConeAttachment) -> None:
    """Point a lone cylinder's feed direction away from its countersink cone."""
    segment = run.segments[0]
    cone_is_above = cone.interval[0] >= segment.interval[1] - _AXIAL_TOL
    run.inwardSign = -1.0 if cone_is_above else 1.0
    run.entryCoordinate = segment.interval[1] if cone_is_above else segment.interval[0]


def _cone_at(model: PartModel, run: HoleRun, segment: LadderSegment, coordinate: float) -> ConeAttachment | None:
    """A cone whose small rim is exactly this cylinder's rim at *coordinate*.

    The small rim must match the cylinder's radius, the cone must widen away from it, and the
    two must share a real rim junction - the same three proofs the countersink matcher applies,
    so what the grammar sees here and what the resolver would accept cannot drift apart.
    """
    axis = run.axis
    cylinder = model.faces[segment.faceId]
    origin = cylinder.parameters.get("axis_location")
    best: ConeAttachment | None = None
    for cone_id, cone in model.faces.items():
        if cone.surface_type != SurfaceType.CONE:
            continue
        cone_axis = unit_vector(cone.parameters.get("axis_direction"))
        apex = cone.parameters.get("apex")
        if cone_axis is None or not valid_vector(apex) or _axis_angle(axis, cone_axis) > _AXIS_TOL:
            continue
        if _line_distance(origin, apex, axis) > _AXIS_DISTANCE_TOL:
            continue
        circles = _cone_circles(model, cone_id, axis)
        if len(circles) != 2 or any(not item.get("complete") for item in circles):
            continue
        small = min(circles, key=lambda item: item["radius"])
        large = max(circles, key=lambda item: item["radius"])
        if abs(float(small["radius"]) - segment.radius) > max(_RADIUS_TOL, segment.radius * 1.0e-3):
            continue
        if float(large["radius"]) <= segment.radius + _RADIUS_TOL:
            continue
        if bool(cone.is_reversed) != bool(cylinder.is_reversed):
            continue
        junction = float(small["axisPosition"])
        if abs(junction - coordinate) > max(_AXIAL_TOL, 1.0e-3 * max(abs(coordinate), 1.0)):
            continue
        if not _direct_rim_junction(model, segment.faceId, cone_id, small, axis, junction):
            continue
        positions = sorted((float(small["axisPosition"]), float(large["axisPosition"])))
        if positions[1] - positions[0] <= _AXIAL_TOL:
            continue
        candidate = ConeAttachment(
            faceId=cone_id, end="far", cylinderFaceId=segment.faceId,
            smallRadius=float(small["radius"]), largeRadius=float(large["radius"]),
            interval=(positions[0], positions[1]), smallBoundary=small, largeBoundary=large,
        )
        # Two cones cannot share one rim; if the geometry offers several, the widest wins and
        # the ambiguity would have shown up as an incomplete circle above.
        if best is None or candidate.largeRadius > best.largeRadius:
            best = candidate
    return best


def _face_sort_key(face_id: str) -> tuple[int, str]:
    text = str(face_id)
    digits = "".join(ch for ch in text if ch.isdigit())
    return (int(digits) if digits else math.inf, text)


def analysis_diagnostics(analysis: LadderAnalysis) -> dict[str, Any]:
    """Diagnostics block for the orchestrator's report."""
    return analysis.diagnostics()


__all__ = [
    "ConeAttachment",
    "HoleRun",
    "LadderAnalysis",
    "LadderSegment",
    "analysis_diagnostics",
    "discover_hole_ladders",
]
