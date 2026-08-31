"""Shared SWB endpoint resolution: cross transition regions, recover the
*effective* material-side concavity of the whole crossing.

A side-wall band (SWB) says what the walls are.  A manufacturing feature also
needs to know what each end of the band *runs into*, and which way the
material turns there:

    Pocket:  incoming CONVEX  -> outgoing CONCAVE
    Boss:    incoming CONCAVE -> outgoing CONVEX

Both questions are answered here, once, for every feature module — none of
this belongs in ``Pocket.py``.

Why a plain edge lookup is not enough
-------------------------------------
``PartModel`` already carries a validated material-side concavity per shared
edge (``FaceData.adjacencies``; see ``NODE_REPORT/MaterialSideConcavityAudit_T0.md``).
Reading the rim's own edges works only when the band runs straight into its
neighbour.  Real parts interpose a *transition region* — a chamfer or a
fillet — between the wall and whatever it actually meets, and that region can
itself be a complete legal side-wall band sitting directly against the real
one.  So the traversal is:

    band -> rim -> [transition region] -> [transition region] -> terminal

crossing as many transition regions as the evidence supports, and folding
them into the result rather than stopping on them.

Recovering the concavity of the crossing
----------------------------------------
Each crossing of a rim has a length-weighted verdict from the per-edge
concavities.  The path verdict then follows three rules, in order:

1. any crossing that is itself internally inconsistent → ``MIXED``;
2. otherwise, if some crossing is not ``SMOOTH``, every non-``SMOOTH``
   crossing must agree, and that value is the answer.  A chamfer gives
   ``CONVEX`` on both of its edges at a convex mouth, so agreement holds and
   no geometry is needed;
3. otherwise every crossing is ``SMOOTH`` — a fillet — and the sign has to
   come from geometry, because tangential continuity carries no sign.

Rule 3 uses a **chord test**, which needs no orientation convention at all
and works for cylinder, torus and B-spline blends alike.  Take a point
``A`` on the entry rim and the nearest point ``B`` on the exit rim, and
classify the midpoint of the straight chord ``AB`` against the solid:

- a convex blend (a rounded outer corner) bulges outward, so the chord cuts
  through material → midpoint **inside** → ``CONVEX``;
- a concave blend (a wall-to-floor radius) is dished away from the corner, so
  the chord passes through the cavity → midpoint **outside** → ``CONCAVE``.

Several chords are sampled around the rim and must agree.  A midpoint that
lands *on* the surface means there is no turn to measure, and the crossing is
reported as genuinely smooth rather than guessed at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from contract.kernel import POINT_UNAVAILABLE
from contract.partmodel import ConcavityType, CurveType, PartModel

from features._shared.closed_wall_band import EdgeIndex, face_sort_key
from features._shared.transition_evidence import TransitionEvidence

CONCAVE = ConcavityType.CONCAVE.value
CONVEX = ConcavityType.CONVEX.value
SMOOTH = ConcavityType.SMOOTH.value
UNKNOWN = ConcavityType.UNKNOWN.value
MIXED = "MIXED"

# A rim crossing is only given a definite verdict when one concavity holds
# this share of the crossed rim length.  Rims routinely carry a stray
# tangential edge where a corner blend meets the rim; a hard "all edges must
# agree" rule would throw those crossings away.
_CROSSING_DOMINANCE = 0.8
# How many transition regions may be folded into one endpoint before the
# traversal gives up.  Two is enough for chamfer+fillet stacking; the bound
# exists so a mis-scored region cannot walk across the whole part.
_MAX_TRANSITION_STEPS = 3
_CHORD_SAMPLES = 5
_CHORD_MIN_LENGTH = 1.0e-6
_CONTINUATION_PARALLEL_COS = math.cos(1.0e-4)
_CONTINUATION_MIN_LENGTH = 1.0e-6


@dataclass
class EndpointStep:
    """One rim crossing: from a face region, across a rim, into the next."""

    rim_edge_ids: list[str]
    from_face_ids: list[str]
    to_face_ids: list[str]
    concavity: str
    concavity_shares: dict[str, float]
    to_region_is_transition: bool
    transition_reason: str
    # The crossed rim as canonical signatures.  Kept alongside the reported
    # edge ids so a consumer can ask further topological questions about the
    # rim actually crossed — the last step's rim is the one that meets the
    # terminal region, which is not the band's own rim once a transition has
    # been folded in.  Diagnostics stay on the edge ids.
    rim_sigs: frozenset = field(default_factory=frozenset)

    def to_dict(self) -> dict:
        return {
            "rimEdgeIds": list(self.rim_edge_ids),
            "fromFaceIds": list(self.from_face_ids),
            "toFaceIds": list(self.to_face_ids),
            "concavity": self.concavity,
            "concavityShares": {key: round(value, 4) for key, value in sorted(self.concavity_shares.items())},
            "toRegionIsTransition": self.to_region_is_transition,
            "transitionReason": self.transition_reason,
        }


@dataclass
class EndpointResolution:
    """What one end of a wall band runs into, and how the material turns."""

    status: str
    effective_concavity: str
    method: str
    terminal_face_ids: list[str] = field(default_factory=list)
    transition_face_ids: list[str] = field(default_factory=list)
    steps: list[EndpointStep] = field(default_factory=list)
    detail: str = ""
    # Only OpenSweep consumers opt into a course-aware transition filter.
    # Closed-SWB callers leave this empty and retain the original traversal.
    continuation_constraints: list[dict] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"

    @property
    def terminal_rim_sigs(self) -> frozenset:
        """The rim that separates the terminal region from what precedes it.

        Equal to the band's own rim when nothing was crossed, and to the far
        boundary of the last transition region otherwise.  This is the rim a
        consumer has to ask its questions of: the opening of a through cut is
        bounded by *this* rim, not by the band rim behind the chamfer.
        """
        return self.steps[-1].rim_sigs if self.steps else frozenset()

    def to_dict(self) -> dict:
        result = {
            "status": self.status,
            "effectiveConcavity": self.effective_concavity,
            "method": self.method,
            "terminalFaceIds": list(self.terminal_face_ids),
            "transitionFaceIds": list(self.transition_face_ids),
            "transitionRegionCount": sum(1 for step in self.steps if step.to_region_is_transition),
            "steps": [step.to_dict() for step in self.steps],
            "detail": self.detail,
        }
        if self.continuation_constraints:
            result["continuationConstraints"] = list(self.continuation_constraints)
        return result


class SolidPointClassifier:
    """``inside / outside`` test against the solid, or unavailable.

    A thin adapter over the kernel, kept as its own object because the
    traversal passes it down several levels and because *whether it can
    answer at all* is asked once, up front: a chord test that finds no
    classifier reports a different reason than one whose midpoints land on
    the surface, and the two must not be confused.
    """

    def __init__(self, kernel) -> None:
        self._kernel = kernel if kernel is not None and kernel.supports_point_classification() else None

    @property
    def available(self) -> bool:
        return self._kernel is not None

    def state(self, point: list[float], tolerance: float) -> str:
        if self._kernel is None:
            return POINT_UNAVAILABLE
        return self._kernel.classify_point(point, tolerance)


def resolve_endpoint(
    model: PartModel,
    index: EdgeIndex,
    transition: TransitionEvidence,
    classifier: SolidPointClassifier | None,
    incoming_face_ids,
    rim_sigs,
    *,
    max_transition_steps: int = _MAX_TRANSITION_STEPS,
    continuation_context: dict | None = None,
) -> EndpointResolution:
    """Walk outward from ``rim_sigs``, crossing transition regions.

    ``incoming_face_ids`` is the side the traversal starts on (the wall band),
    so the walk always crosses the rim rather than turning back into it.
    """
    current_from = set(incoming_face_ids)
    current_rim = set(rim_sigs)
    visited: set[str] = set(current_from)
    transition_faces: list[str] = []
    steps: list[EndpointStep] = []
    continuation_constraints: list[dict] = []

    for _ in range(max_transition_steps + 1):
        if not current_rim:
            return _unresolved("open", steps, transition_faces, "rim became empty before a terminal region was reached")
        far = _far_faces(index, current_rim, visited)
        if not far:
            return _unresolved("open", steps, transition_faces, "rim has no face on its far side")

        concavity, shares = _crossing_concavity(model, index, current_from, far, current_rim)
        verdict = transition.region_verdict(far)
        steps.append(
            EndpointStep(
                rim_edge_ids=sorted(index.representative(sig) for sig in current_rim),
                from_face_ids=sorted(current_from, key=face_sort_key),
                to_face_ids=sorted(far, key=face_sort_key),
                concavity=concavity,
                concavity_shares=shares,
                to_region_is_transition=verdict.is_transition,
                transition_reason=verdict.reason,
                rim_sigs=frozenset(current_rim),
            )
        )
        if not verdict.is_transition:
            effective, method, detail = _effective_concavity(model, index, classifier, steps, current_rim)
            return EndpointResolution(
                status=_status_for(effective),
                effective_concavity=effective,
                method=method,
                terminal_face_ids=sorted(far, key=face_sort_key),
                transition_face_ids=sorted(set(transition_faces), key=face_sort_key),
                steps=steps,
                detail=detail,
                continuation_constraints=continuation_constraints,
            )

        transition_faces.extend(far)
        visited.update(far)
        current_from = far
        boundary = _region_far_boundary(index, far, current_rim)
        if continuation_context is not None:
            current_rim, constraint = _continuation_boundary(
                model,
                index,
                boundary,
                continuation_context,
                transition_face_ids=far,
                entry_rim=current_rim,
            )
            continuation_constraints.append(constraint)
            if constraint["status"] != "unique":
                return _unresolved(
                    "unresolved",
                    steps,
                    transition_faces,
                    constraint["reason"],
                    continuation_constraints=continuation_constraints,
                )
        else:
            current_rim = boundary

    return _unresolved(
        "transition_limit",
        steps,
        transition_faces,
        f"more than {max_transition_steps} consecutive transition regions",
        continuation_constraints=continuation_constraints,
    )


# ---------------------------------------------------------------------------
# Traversal helpers
# ---------------------------------------------------------------------------

def _far_faces(index: EdgeIndex, rim_sigs: set, visited: set[str]) -> set[str]:
    faces: set[str] = set()
    for sig in rim_sigs:
        faces.update(index.sig_incidence.get(sig, ()))
    return faces - visited


def _region_far_boundary(index: EdgeIndex, region: set[str], entry_rim: set) -> set:
    """External boundary of ``region`` excluding the rim it was entered by."""
    boundary: set = set()
    for sig, faces in index.sig_incidence.items():
        if sig in entry_rim or sig in index.self_seam_sigs:
            continue
        if len(faces & region) == 1:
            boundary.add(sig)
    return boundary


def _continuation_boundary(
    model: PartModel,
    index: EdgeIndex,
    boundary: set,
    context: dict,
    *,
    transition_face_ids: set[str],
    entry_rim: set,
) -> tuple[set, dict]:
    """Filter one open-mouth transition boundary to a unique full-course rail.

    Closed SWB traversal intentionally keeps every external far-boundary
    signature.  An OpenSweep mouth is different: a transition strip exposes
    its longitudinal start/end transverse boundaries alongside the one rail
    which continues to the true transverse support.  The OS consumer opts
    into this strict filter and must get exactly one canonical rail.
    """

    direction = _unit(context.get("course_direction"))
    start = _finite_float(context.get("course_start"))
    end = _finite_float(context.get("course_end"))
    span = None if start is None or end is None else abs(end - start)
    tolerance = max(1.0e-4, (span or 0.0) * 1.0e-5)
    candidates: list[tuple] = []
    rejected: list[dict] = []

    if direction is None or start is None or end is None or span is None or span <= _CONTINUATION_MIN_LENGTH:
        return set(), {
            "status": "unresolved",
            "reason": "open_sweep_course_context_incomplete",
            "transitionFaceIds": sorted(transition_face_ids, key=face_sort_key),
            "entryRimEdgeIds": sorted(index.representative(sig) for sig in entry_rim),
            "boundaryEdgeIds": sorted(index.representative(sig) for sig in boundary),
            "candidateContinuationRailEdgeIds": [],
            "rejectedBoundaryEdges": [],
        }

    lower = min(start, end)
    upper = max(start, end)
    for sig in sorted(boundary, key=lambda item: index.representative(item)):
        edge_id = index.representative(sig)
        edge = model.edges.get(edge_id)
        reasons: list[str] = []
        extent = None
        if edge is None:
            reasons.append("edge_unavailable")
        elif edge.curve_type != CurveType.LINE:
            reasons.append("curve_not_line")
        else:
            edge_direction = _line_direction(model, edge)
            if edge_direction is None:
                reasons.append("line_direction_unavailable")
            elif abs(_dot(edge_direction, direction)) < _CONTINUATION_PARALLEL_COS:
                reasons.append("not_parallel_to_course")
            points = _edge_points(model, edge)
            if points:
                projections = [_dot(point, direction) for point in points]
                extent = (min(projections), max(projections))
            else:
                reasons.append("course_projection_unavailable")

            if extent is not None:
                if extent[1] - extent[0] <= tolerance and (
                    abs((extent[0] + extent[1]) * 0.5 - lower) <= tolerance
                    or abs((extent[0] + extent[1]) * 0.5 - upper) <= tolerance
                ):
                    reasons.append("longitudinal_start_or_end_transverse_boundary")
                elif abs(extent[0] - lower) > tolerance or abs(extent[1] - upper) > tolerance:
                    reasons.append("does_not_cover_full_course_span")

        record = {
            "edgeId": edge_id,
            "curveType": (
                edge.curve_type.value
                if edge is not None and hasattr(edge.curve_type, "value")
                else str(edge.curve_type) if edge is not None else None
            ),
            "courseProjectionExtent": (
                [round(extent[0], 6), round(extent[1], 6)]
                if extent is not None else None
            ),
            "reasons": reasons,
        }
        if not reasons:
            candidates.append(sig)
        else:
            rejected.append(record)

    if len(candidates) == 1:
        status = "unique"
        reason = "unique_full_course_longitudinal_line_continuation"
    elif not candidates:
        status = "unresolved"
        reason = "no_unique_full_course_continuation_rail"
    else:
        status = "unresolved"
        reason = "multiple_full_course_continuation_rails"
    return set(candidates) if status == "unique" else set(), {
        "status": status,
        "reason": reason,
        "transitionFaceIds": sorted(transition_face_ids, key=face_sort_key),
        "entryRimEdgeIds": sorted(index.representative(sig) for sig in entry_rim),
        "boundaryEdgeIds": sorted(index.representative(sig) for sig in boundary),
        "candidateContinuationRailEdgeIds": sorted(
            index.representative(sig) for sig in candidates
        ),
        "rejectedBoundaryEdges": rejected,
    }


def _line_direction(model: PartModel, edge) -> list[float] | None:
    direction = _unit(edge.parameters.get("direction"))
    if direction is not None:
        return direction
    points = _edge_points(model, edge)
    if len(points) < 2:
        return None
    best = max(
        (
            (math.dist(left, right), left, right)
            for left in points
            for right in points
        ),
        key=lambda item: item[0],
    )
    if best[0] <= _CONTINUATION_MIN_LENGTH:
        return None
    return _unit([best[2][axis] - best[1][axis] for axis in range(3)])


def _edge_points(model: PartModel, edge) -> list[list[float]]:
    points = [
        list(point)
        for point in (getattr(edge, "sample_points", None) or [])
        if _is_vector(point)
    ]
    if points:
        return points
    return [
        list(model.vertices[vertex_id].coordinates)
        for vertex_id in (getattr(edge, "vertex_ids", None) or [])
        if vertex_id in model.vertices
    ]


def _is_vector(value) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    try:
        return all(math.isfinite(float(item)) for item in value)
    except (TypeError, ValueError):
        return False


def _finite_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _unit(value) -> list[float] | None:
    if not _is_vector(value):
        return None
    values = [float(item) for item in value]
    norm = math.sqrt(sum(item * item for item in values))
    if norm <= _CONTINUATION_MIN_LENGTH:
        return None
    return [item / norm for item in values]


def _dot(left, right) -> float:
    return sum(float(left[index]) * float(right[index]) for index in range(3))


def _crossing_concavity(
    model: PartModel,
    index: EdgeIndex,
    from_faces: set[str],
    to_faces: set[str],
    rim_sigs: set,
) -> tuple[str, dict[str, float]]:
    """Length-weighted material-side concavity of one rim crossing."""
    weights: dict[str, float] = {}
    for face_id in from_faces:
        face = model.faces.get(face_id)
        if face is None:
            continue
        for adjacency in face.adjacencies:
            if adjacency.adjacent_face_id not in to_faces:
                continue
            edge_id = adjacency.shared_edge_id
            if index.edge_sig.get(edge_id) not in rim_sigs:
                continue
            edge = model.edges.get(edge_id)
            length = float(getattr(edge, "length", 0.0) or 0.0) if edge is not None else 0.0
            key = _concavity_value(adjacency.connection_type)
            weights[key] = weights.get(key, 0.0) + max(length, 0.0)

    total = sum(weights.values())
    if total <= 0.0:
        return UNKNOWN, {}
    shares = {key: value / total for key, value in weights.items()}
    best = max(shares.items(), key=lambda item: (item[1], item[0]))
    if best[0] == UNKNOWN or best[1] < _CROSSING_DOMINANCE:
        return MIXED, shares
    return best[0], shares


def _effective_concavity(
    model: PartModel,
    index: EdgeIndex,
    classifier: SolidPointClassifier | None,
    steps: list[EndpointStep],
    exit_rim: set,
) -> tuple[str, str, str]:
    """Combine the crossings of one endpoint path into a single verdict."""
    values = [step.concavity for step in steps]
    if any(value in {MIXED, UNKNOWN} for value in values):
        return MIXED, "crossing_inconsistent", "at least one rim crossing has no dominant concavity"

    definite = [value for value in values if value in {CONCAVE, CONVEX}]
    if definite:
        if len(set(definite)) > 1:
            return MIXED, "transition_disagreement", "transition crossings disagree: " + ",".join(values)
        method = "direct_edge" if len(steps) == 1 else "transition_agreement"
        return definite[0], method, ""

    # Every crossing is tangential — a fillet.  The sign has to come from the
    # geometry of the blend itself, spanning entry rim to exit rim.
    return _chord_test(model, index, classifier, set(steps[0].rim_sigs), exit_rim)


def _chord_test(
    model: PartModel,
    index: EdgeIndex,
    classifier: SolidPointClassifier | None,
    entry_rim: set,
    exit_rim: set,
) -> tuple[str, str, str]:
    """Classify the midpoint of chords spanning an all-tangential transition."""
    if classifier is None or not classifier.available:
        return SMOOTH, "chord_solid_classification", "no solid classifier available for the tangential transition"
    entry_points = _rim_points(model, index, entry_rim)
    exit_points = _rim_points(model, index, exit_rim)
    if not entry_points or not exit_points:
        return SMOOTH, "chord_solid_classification", "tangential transition has no sampled rim points"

    tolerance = _chord_tolerance(model)
    verdicts: list[str] = []
    for point_a in _spread(entry_points, _CHORD_SAMPLES):
        point_b = min(exit_points, key=lambda candidate: _distance_sq(point_a, candidate))
        length = math.sqrt(_distance_sq(point_a, point_b))
        if length <= _CHORD_MIN_LENGTH:
            continue
        midpoint = [(point_a[axis] + point_b[axis]) * 0.5 for axis in range(3)]
        state = classifier.state(midpoint, min(tolerance, length * 0.05))
        if state == "in":
            verdicts.append(CONVEX)
        elif state == "out":
            verdicts.append(CONCAVE)

    if not verdicts:
        return SMOOTH, "chord_solid_classification", "chord midpoints lie on the surface: no measurable turn"
    if len(set(verdicts)) > 1:
        return MIXED, "chord_solid_classification", "chord samples disagree: " + ",".join(sorted(set(verdicts)))
    return verdicts[0], "chord_solid_classification", ""


def _rim_points(model: PartModel, index: EdgeIndex, rim_sigs: set) -> list[list[float]]:
    points: list[list[float]] = []
    for sig in sorted(rim_sigs):
        edge_id = index.representative(sig)
        edge = model.edges.get(edge_id)
        if edge is None:
            continue
        samples = [list(point) for point in (getattr(edge, "sample_points", None) or [])]
        if samples:
            points.append([float(value) for value in samples[len(samples) // 2]])
            continue
        coordinates = [
            list(model.vertices[vertex_id].coordinates)
            for vertex_id in (getattr(edge, "vertex_ids", None) or [])
            if vertex_id in model.vertices
        ]
        if len(coordinates) == 2:
            points.append([(float(coordinates[0][axis]) + float(coordinates[1][axis])) * 0.5 for axis in range(3)])
    return points


def _spread(points: list[list[float]], count: int) -> list[list[float]]:
    if len(points) <= count:
        return points
    stride = len(points) / float(count)
    return [points[int(idx * stride)] for idx in range(count)]


def _distance_sq(left: list[float], right: list[float]) -> float:
    return sum((float(left[axis]) - float(right[axis])) ** 2 for axis in range(3))


def _chord_tolerance(model: PartModel) -> float:
    extent = 0.0
    for face in model.faces.values():
        bbox = getattr(face, "bbox", None)
        if not isinstance(bbox, dict):
            continue
        lower, upper = bbox.get("min"), bbox.get("max")
        if not (isinstance(lower, (list, tuple)) and isinstance(upper, (list, tuple))):
            continue
        for axis in range(3):
            try:
                extent = max(extent, abs(float(upper[axis]) - float(lower[axis])))
            except (TypeError, ValueError, IndexError):
                continue
    return max(1.0e-7, extent * 1.0e-6)


def _status_for(effective: str) -> str:
    if effective in {CONCAVE, CONVEX}:
        return "resolved"
    if effective == MIXED:
        return "mixed"
    if effective == SMOOTH:
        return "smooth"
    return "unknown"


def _concavity_value(value: object) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _unresolved(
    status: str,
    steps,
    transition_faces,
    detail: str,
    *,
    continuation_constraints: list[dict] | None = None,
) -> EndpointResolution:
    return EndpointResolution(
        status=status,
        effective_concavity=UNKNOWN,
        method="none",
        terminal_face_ids=[],
        transition_face_ids=sorted(set(transition_faces), key=face_sort_key),
        steps=list(steps),
        detail=detail,
        continuation_constraints=list(continuation_constraints or ()),
    )
