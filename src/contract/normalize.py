"""PartModel normalization: seam-face merging and axial-extent annotation.

Pure ``PartModel`` data operations, which is why they live in ``contract``
rather than in a kernel line: whichever kernel filled the table may apply
them.  The OCC line needs both (its STEP reader splits periodic faces at
seams); the NX line's translator already sews seams, so its merge pass is a
no-op, but the axial-extent annotation is load-bearing on both.

The merge is **topology-preserving**: when two seam-split fragments of one
surface are merged, every non-seam boundary edge of the removed face survives
as part of the merged keeper face (reconstructed closed loops, rebuilt
adjacency).  Only the internal seam boundary shared by the two fragments may
disappear.  If the surviving boundary cannot be reconstructed into closed
loops, the merge is conservatively skipped and the model is left untouched.
"""

import math
import re

from contract.partmodel import AdjacencyInfo, FaceData, LoopData, PartModel, SurfaceType


def _union_bboxes(left: dict | None, right: dict | None) -> dict | None:
    """Return the AABB union while preserving a valid side when one is absent."""
    if left is None:
        return right
    if right is None:
        return left
    left_min, left_max = left.get("min"), left.get("max")
    right_min, right_max = right.get("min"), right.get("max")
    if not all(isinstance(value, list) and len(value) == 3 for value in (left_min, left_max, right_min, right_max)):
        return left
    return {
        "min": [min(float(left_min[index]), float(right_min[index])) for index in range(3)],
        "max": [max(float(left_max[index]), float(right_max[index])) for index in range(3)],
    }


# ---------------------------------------------------------------------------
# Tolerance constants (from CLAUDE.md)
# ---------------------------------------------------------------------------

_RADIUS_TOL = 1e-6          # mm — radius difference
_AXIS_ANGLE_TOL = 1e-4      # rad — axis / normal direction difference
_AXIS_DIST_TOL = 1e-4       # mm — coaxiality lateral offset tolerance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _angle_between(v1: list[float], v2: list[float]) -> float:
    """Angle in radians between two 3-D vectors, clamped to [0, pi]."""
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)


def _axis_angle_between(v1: list[float], v2: list[float]) -> float:
    """Undirected axis angle in radians.

    Axis orientation can flip while still representing the same geometric
    axis, so compare both parallel and anti-parallel directions.
    """
    angle = _angle_between(v1, v2)
    return min(angle, abs(math.pi - angle))


def _point_key(params: dict) -> list[float] | None:
    return params.get("axis_location") or params.get("apex")


def _axis_line_match(p1: dict, p2: dict, *, same_point: bool = False) -> bool:
    """Do the two surfaces sit on the same axis line?

    Lateral offset only, by default: a cylinder's ``axis_location`` is any
    point on its axis, so sliding it along the axis still describes the same
    line and the same cylinder.

    ``same_point`` additionally requires the two reference points to coincide
    *along* the axis, at the same tolerance.  A cone's reference point is its
    apex, which is not free to slide: two cones with the same half angle on
    one axis but with apexes apart are two different cones - a countersink and
    the chamfer above it, say - and merging them would fuse them into one face
    (node 27B §5 #3).  A surface whose reference point the parser could not
    read is not made to fail this check; it never had a point to compare.
    """
    ax1 = p1.get("axis_direction", [0.0, 0.0, 1.0])
    ax2 = p2.get("axis_direction", [0.0, 0.0, 1.0])
    if _axis_angle_between(ax1, ax2) >= _AXIS_ANGLE_TOL:
        return False
    loc1 = _point_key(p1)
    loc2 = _point_key(p2)
    if loc1 is None or loc2 is None:
        return True
    d = ax1
    mag = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    if mag <= 0:
        return True
    dx, dy, dz = d[0] / mag, d[1] / mag, d[2] / mag
    v = [loc2[0] - loc1[0], loc2[1] - loc1[1], loc2[2] - loc1[2]]
    v_dot_d = v[0] * dx + v[1] * dy + v[2] * dz
    perp = math.sqrt(
        (v[0] - v_dot_d * dx) ** 2
        + (v[1] - v_dot_d * dy) ** 2
        + (v[2] - v_dot_d * dz) ** 2
    )
    if same_point and abs(v_dot_d) >= _AXIS_DIST_TOL:
        return False
    return perp < _AXIS_DIST_TOL


def _params_match(st: SurfaceType, p1: dict, p2: dict) -> bool:
    """Return ``True`` if the key geometric parameters of two faces of the
    same *SurfaceType* match within the prescribed tolerances.

    +-------------+-------------------------------------+
    | SurfaceType | checks                              |
    +=============+=====================================+
    | CYLINDER    | ``radius`` < 1e-6, ``axis`` < 1e-4 |
    +-------------+-------------------------------------+
    | PLANE       | ``normal`` angle < 1e-4 rad         |
    +-------------+-------------------------------------+
    | CONE        | ``semi_angle`` < 1e-4 rad, axis     |
    |             | *and* apex, including along the axis|
    +-------------+-------------------------------------+
    | other       | always ``False`` (skip)             |
    +-------------+-------------------------------------+

    A cone has no ``radius`` parameter on either line - it is described by its
    half angle and its apex - so the half angle is what stands in for the
    cylinder's radius check, at the same tolerance the function already uses
    for every other angle.  A cone whose half angle is missing does not match:
    merging two fragments on the strength of a shared axis alone would fuse a
    countersink into the chamfer above it.  For the same reason the apex has
    to match *along* the axis as well, not only across it: the cylinder's
    reference point may slide along its own axis and still be the same
    cylinder, the cone's apex may not.
    """
    if st == SurfaceType.CYLINDER:
        r1 = p1.get("radius", 0.0)
        r2 = p2.get("radius", 0.0)
        if abs(r1 - r2) >= _RADIUS_TOL:
            return False
        return _axis_line_match(p1, p2)

    if st == SurfaceType.CONE:
        a1 = p1.get("semi_angle")
        a2 = p2.get("semi_angle")
        if a1 is None or a2 is None:
            return False
        if abs(float(a1) - float(a2)) >= _AXIS_ANGLE_TOL:
            return False
        return _axis_line_match(p1, p2, same_point=True)

    if st == SurfaceType.PLANE:
        n1 = p1.get("normal", [0.0, 0.0, 1.0])
        n2 = p2.get("normal", [0.0, 0.0, 1.0])
        if _angle_between(n1, n2) >= _AXIS_ANGLE_TOL:
            return False
        return True

    # SPHERE, TORUS, BSPLINE, OTHER — not merged
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def annotate_face_axial_extents(model: PartModel) -> PartModel:
    """Persist the axial extent of every cylindrical face into its parameters.

    For each ``CYLINDER`` face, ``axial_min`` / ``axial_max`` are stored as
    the minimum and maximum of ``p · axis`` over all points sampled from the
    face's own boundary edges.  The axial coordinate of a cylinder patch is a
    linear function of the surface parameters, so its extrema lie on the
    boundary — the boundary samples therefore give the true axial range of
    the wall without the AABB-corner overestimation that would affect tilted
    cylinders.  Must run before ``normalize_seam_faces`` so every face still
    has its complete pre-merge loops.
    """
    for face in model.faces.values():
        if face.surface_type != SurfaceType.CYLINDER:
            continue
        axis = face.parameters.get("axis_direction")
        if not (isinstance(axis, (list, tuple)) and len(axis) == 3):
            continue
        magnitude = math.sqrt(sum(float(component) ** 2 for component in axis))
        if magnitude <= 1e-12:
            continue
        unit = [float(component) / magnitude for component in axis]

        projections: list[float] = []
        for loop_id in face.loop_ids:
            loop = model.loops.get(loop_id)
            if loop is None:
                continue
            for edge_id in loop.edge_ids:
                edge = model.edges.get(edge_id)
                if edge is None:
                    continue
                samples = getattr(edge, "sample_points", None) or []
                points = samples if samples else [
                    list(model.vertices[vertex_id].coordinates)
                    for vertex_id in edge.vertex_ids
                    if vertex_id in model.vertices
                ]
                for point in points:
                    if isinstance(point, (list, tuple)) and len(point) == 3:
                        projections.append(
                            sum(float(point[index]) * unit[index] for index in range(3))
                        )

        if projections:
            face.parameters["axial_min"] = min(projections)
            face.parameters["axial_max"] = max(projections)

    return model


def _union_axial_extents(keeper: FaceData, removed: FaceData) -> None:
    """Widen the keeper's persisted axial extent with the removed face's one."""
    keeper_min = keeper.parameters.get("axial_min")
    keeper_max = keeper.parameters.get("axial_max")
    removed_min = removed.parameters.get("axial_min")
    removed_max = removed.parameters.get("axial_max")
    if not all(isinstance(value, (int, float)) for value in (keeper_min, keeper_max, removed_min, removed_max)):
        return
    keeper.parameters["axial_min"] = min(float(keeper_min), float(removed_min))
    keeper.parameters["axial_max"] = max(float(keeper_max), float(removed_max))


def normalize_seam_faces(model: PartModel) -> PartModel:
    """Detect and merge faces split by seam edges, in place.

    **This function consumes its argument.**  It rewrites *model* and returns
    the very same object; the caller must treat the pre-merge model as gone
    the moment the call is made.  Both production call sites
    (``occline.parse.process_step_file`` and ``nxline.partmodel``) already
    write ``model = normalize_seam_faces(model)`` and drop the pre-merge
    object, and a caller that does want to keep it can pass
    ``copy.deepcopy(model)`` itself.  The deep copy this function used to make
    unconditionally was 79% of the pass (70,052 ``deepcopy`` calls on a
    119-face part) and every byte of it was thrown away.

    **Merge conditions**:

    1. The two faces are mutually adjacent.
    2. They share the same ``SurfaceType``.
    3. Their key geometric parameters match within tolerance (see
       ``_params_match``).
    4. Cylinders use a shared ``is_seam = True`` edge. Cones use a
       ``SMOOTH`` cone-cone adjacency on the same axis line because the
       STEP parser often does not mark those smooth split edges as seams.

    **Merge operation** (dataclass level only, per DC3):

    - Retain the ``FaceData`` with larger area; discard the other.
    - Redirect all adjacency references to the discarded face so they
      point to the retained face.
    - Remove *only* the internal seam edges that connected the two faces
      from ``edges`` and ``ShellData.edge_ids``.  Non-seam boundary edges
      of the discarded face are kept and spliced into the retained face's
      reconstructed loops.
    - Rebuild the retained face's ``LoopData`` from the surviving boundary
      edge connectivity (closed cycles; preserved inner loops).
    - Rebuild the retained face's adjacency list as the union of both
      faces' third-party adjacencies (reciprocal, deduplicated).
    - Append the discarded face ID to the retained face's
      ``merged_from`` list.
    - Preserve the discarded face area in merge provenance.
    - Log every merge (and every conservatively skipped merge) to
      ``normalization_log``.
    """
    # ---- index: face_id -> {adjacent_face_id: shared_edge_id} ---------------
    def _build_adj_index() -> dict[str, dict[str, str]]:
        idx: dict[str, dict[str, str]] = {}
        for fid, fd in model.faces.items():
            idx[fid] = {}
            for ai in fd.adjacencies:
                idx[fid][ai.adjacent_face_id] = ai.shared_edge_id
        return idx

    adj_index = _build_adj_index()
    logged_skips: set[tuple[str, str]] = set()

    # ---- merge loop — restart after each successful merge -------------------
    merged_any = True
    while merged_any:
        merged_any = False

        face_ids = list(model.faces.keys())
        position = {fid: index for index, fid in enumerate(face_ids)}
        for i in range(len(face_ids)):
            fid_a = face_ids[i]
            if fid_a not in model.faces:
                continue
            face_a = model.faces[fid_a]

            # --- condition 1: mutually adjacent ------------------------------
            # Walking all of ``face_ids[i+1:]`` only to reject everything that
            # is not a neighbour is what made this pass quadratic in the face
            # count *on every restart*, and it restarts after every merge.  The
            # neighbours are walked instead, in the order the full scan would
            # have reached them, so the sequence of pairs that get past this
            # condition - and therefore the merge order, which the merge result
            # depends on - is exactly the same sequence as before.
            a_adj = adj_index.get(fid_a, {})
            later_neighbours = sorted(
                (fid for fid in a_adj if position.get(fid, -1) > i),
                key=position.__getitem__,
            )
            for fid_b in later_neighbours:
                if fid_b not in model.faces:
                    continue
                face_b = model.faces[fid_b]

                b_adj = adj_index.get(fid_b, {})
                if fid_a not in b_adj:
                    continue

                # --- condition 2: same surface type -------------------------
                if face_a.surface_type != face_b.surface_type:
                    continue

                # --- condition 3: parameter match ---------------------------
                if not _params_match(face_a.surface_type,
                                     face_a.parameters,
                                     face_b.parameters):
                    continue

                # --- condition 4: at least one genuinely shared seam edge ----
                shared_ids = _shared_edge_ids(face_a, face_b, model)
                seam_set = _seam_edge_ids(face_a, face_b, shared_ids, model)
                if not seam_set:
                    continue

                # === MERGE (topology preserving) =============================
                if face_a.area >= face_b.area:
                    keeper_id, removed_id = fid_a, fid_b
                    keeper, removed = face_a, face_b
                else:
                    keeper_id, removed_id = fid_b, fid_a
                    keeper, removed = face_b, face_a

                skip_reason = _merge_faces(model, keeper, removed, keeper_id, removed_id, seam_set)
                if skip_reason is not None:
                    # Conservative skip: the surviving boundary could not be
                    # reconstructed into closed loops; nothing was modified.
                    # Log each skipped pair only once so the log stays compact.
                    key = (keeper_id, removed_id)
                    if key not in logged_skips:
                        logged_skips.add(key)
                        model.normalization_log.append(f"SKIP merge {removed_id} into {keeper_id}: {skip_reason}")
                    continue

                merged_any = True
                break  # restart outer while with rebuilt index

            if merged_any:
                break

        if merged_any:
            adj_index = _build_adj_index()

    return model


# ---------------------------------------------------------------------------
# Topology-preserving merge helpers
# ---------------------------------------------------------------------------

_VERTEX_MERGE_ROUND = 4  # coordinate digits used to merge coincident endpoints
_DEGENERATE_LENGTH_TOL = 1e-9  # mm — zero-length apex edges are not boundary curves


def _shared_edge_ids(face_a: FaceData, face_b: FaceData, model: PartModel) -> set[str]:
    """Edge ids referenced by both faces as shared boundaries."""
    a_ids = {
        ai.shared_edge_id
        for ai in face_a.adjacencies
        if ai.adjacent_face_id == face_b.id and ai.shared_edge_id in model.edges
    }
    b_ids = {
        ai.shared_edge_id
        for ai in face_b.adjacencies
        if ai.adjacent_face_id == face_a.id and ai.shared_edge_id in model.edges
    }
    return a_ids & b_ids


def _seam_edge_ids(face_a: FaceData, face_b: FaceData, shared_ids: set[str], model: PartModel) -> set[str]:
    """Subset of *shared_ids* that are genuine merge seams.

    Cylinders (and other non-cone surfaces) use the parser's ``is_seam``
    flag; cones use a ``SMOOTH`` cone-cone adjacency on the same axis line
    because the STEP parser often does not mark those smooth split edges.
    """
    if not shared_ids:
        return set()
    if face_a.surface_type == SurfaceType.CONE:
        smooth_ids = {
            ai.shared_edge_id
            for ai in face_a.adjacencies
            if ai.adjacent_face_id == face_b.id
            and ai.shared_edge_id in shared_ids
            and ai.connection_type.value == "SMOOTH"
        }
        return smooth_ids & shared_ids
    return {eid for eid in shared_ids if model.edges[eid].is_seam}


def _merge_faces(
    model: PartModel,
    keeper: FaceData,
    removed: FaceData,
    keeper_id: str,
    removed_id: str,
    seam_set: set[str],
) -> str | None:
    """Merge *removed* into *keeper* while preserving all non-seam topology.

    Returns ``None`` on success, or a skip-reason string when the merge cannot
    be performed safely (nothing is modified in that case).  The caller owns
    skip logging so each skipped pair is recorded only once.
    """
    # ---- 0. defensive: a seam edge must never be referenced by a third face
    # Sorted, because the first offender found becomes the skip message, and
    # Python randomises the hash of a string per process: the same conservative
    # skip could otherwise be logged with a different edge id on two runs of
    # the same input (27B leftover 2).  No instance on this corpus - the corpus
    # skips nothing here - which is exactly why it was worth closing.
    for se_id in sorted(seam_set):
        for fid, face in model.faces.items():
            if fid in (keeper_id, removed_id):
                continue
            for lid in face.loop_ids:
                loop = model.loops.get(lid)
                if loop is not None and se_id in loop.edge_ids:
                    return f"seam edge {se_id} referenced by third-party face {fid} loop {lid}"

    # ---- 1. collect pre-merge loop topology --------------------------------
    def _loop_entries(face: FaceData) -> list[tuple[str, LoopData]]:
        return [(lid, model.loops[lid]) for lid in face.loop_ids if lid in model.loops]

    def _has_seam(loop: LoopData) -> bool:
        return any(eid in seam_set for eid in loop.edge_ids)

    seamed_loop_ids = [lid for lid, loop in _loop_entries(keeper) + _loop_entries(removed) if _has_seam(loop)]
    kept_loop_ids = [lid for lid, loop in _loop_entries(keeper) + _loop_entries(removed) if not _has_seam(loop)]
    seamed_edges: set[str] = set()
    degenerate_edges: set[str] = set()
    for _lid, loop in _loop_entries(keeper) + _loop_entries(removed):
        if not _has_seam(loop):
            continue
        for eid in loop.edge_ids:
            if eid in seam_set or eid not in model.edges:
                continue
            if model.edges[eid].length <= _DEGENERATE_LENGTH_TOL:
                # zero-length apex edges of cone fragments are degenerate
                # boundary points, not real boundary curves; they cannot be
                # part of a reconstructed loop.
                degenerate_edges.add(eid)
            else:
                seamed_edges.add(eid)

    # ---- 2. dry-run: reconstruct closed loops from the surviving boundary --
    rebuilt = _rebuild_closed_loops(model, seamed_edges)
    if rebuilt is None:
        return f"boundary reconstruction failed for {len(seamed_edges)} surviving boundary edges"

    # ---- 3. provenance (unchanged) -----------------------------------------
    removed_source = {
        "face_id": removed_id,
        "surface_type": removed.surface_type.value,
        "loop_ids": list(removed.loop_ids),
        "loops": [
            {
                "loop_id": lid,
                "edge_ids": list(model.loops[lid].edge_ids),
                "is_outer": model.loops[lid].is_outer,
            }
            for lid in removed.loop_ids
            if lid in model.loops
        ],
        "adjacencies": [
            {
                "adjacent_face_id": ai.adjacent_face_id,
                "shared_edge_id": ai.shared_edge_id,
                "connection_type": ai.connection_type.value,
            }
            for ai in removed.adjacencies
        ],
        "bbox": removed.bbox,
        "area": removed.area,
        "merged_from": list(removed.merged_from),
        "merge_sources": list(removed.merge_sources),
    }

    # ---- 4. remove internal seam edges from model and shells ---------------
    # ``list.remove`` takes the first occurrence only, so a shell that listed
    # one edge several times kept the rest as ids of edges the model no longer
    # has - 215 dangling ids on this corpus (finding B-02).  Filtering is
    # immune to the repetition whether or not the parser still produces it.
    for se_id in seam_set:
        model.edges.pop(se_id, None)
    for shell in model.shells:
        if any(eid in seam_set for eid in shell.edge_ids):
            shell.edge_ids = [eid for eid in shell.edge_ids if eid not in seam_set]

    # ---- 5. rebuild keeper loops -------------------------------------------
    new_loop_ids: list[str] = []
    for cycle in rebuilt:
        loop_id = _new_loop_id(model, keeper_id)
        model.loops[loop_id] = LoopData(id=loop_id, edge_ids=list(cycle), is_outer=True)
        new_loop_ids.append(loop_id)
    for lid in seamed_loop_ids:
        model.loops.pop(lid, None)
    keeper.loop_ids = kept_loop_ids + new_loop_ids

    # ---- 6. rebuild keeper adjacency ----------------------------------------
    _rebuild_adjacencies(model, keeper, removed, keeper_id, removed_id, seam_set)

    # ---- 7. merge metadata --------------------------------------------------
    keeper.merged_from.append(removed_id)
    keeper.merged_from.extend(removed.merged_from)
    keeper.merge_sources.append(removed_source)
    keeper.merge_sources.extend(removed.merge_sources)
    keeper.bbox = _union_bboxes(keeper.bbox, removed.bbox)
    _union_axial_extents(keeper, removed)
    keeper_area = keeper.area
    removed_area = removed.area
    if (
        isinstance(keeper_area, (int, float))
        and isinstance(removed_area, (int, float))
        and math.isfinite(float(keeper_area))
        and math.isfinite(float(removed_area))
        and float(keeper_area) >= 0.0
        and float(removed_area) >= 0.0
    ):
        # ``keeper`` may already contain earlier merged fragments.  Adding
        # only the current removed face keeps chained merges exact without
        # double-counting nested provenance.
        keeper.area = float(keeper_area) + float(removed_area)
    else:
        model.normalization_log.append(
            f"Area not accumulated for merge {removed_id} into {keeper_id}: invalid fragment area"
        )

    # ---- 8. remove the discarded face ---------------------------------------
    del model.faces[removed_id]
    for shell in model.shells:
        if removed_id in shell.face_ids:
            shell.face_ids.remove(removed_id)

    # ---- 9. log -------------------------------------------------------------
    st_name = keeper.surface_type.value
    extra = ""
    merge_reason = "seam edge"
    if keeper.surface_type == SurfaceType.CYLINDER:
        r = keeper.parameters.get("radius", "?")
        extra = f", r={r}"
    elif keeper.surface_type == SurfaceType.CONE:
        merge_reason = "smooth cone adjacency"
        semi = keeper.parameters.get("semi_angle", "?")
        extra = f", semi_angle={semi}"
    model.normalization_log.append(
        f"Merged {removed_id} into {keeper_id} ({merge_reason}, {st_name}{extra}); "
        f"rebuilt {len(rebuilt)} loop(s) from {len(seamed_edges)} surviving boundary edges"
        + (f"; {len(degenerate_edges)} degenerate edge(s) kept as orphan" if degenerate_edges else "")
    )
    return None


def _rebuild_closed_loops(model: PartModel, edge_ids: set[str]) -> list[list[str]] | None:
    """Group *edge_ids* into closed loops by endpoint connectivity.

    Vertices are merged by rounded coordinates (the parser already deduplicates
    by OCCT hash, so this only guards against coincident duplicate entities).
    Returns one ordered edge-id list per closed cycle, or ``None`` when the
    boundary cannot be decomposed (dangling edges, vertices with degree != 2,
    ...).  Edge ids are already unique (set input); geometrically identical
    copies with different ids fail the degree check and cause a conservative
    skip rather than losing topology.
    """
    node_index: dict[tuple, int] = {}
    adjacency: dict[int, list[tuple[int, str]]] = {}

    def _node_of(point: list[float]) -> int:
        key = (
            round(point[0], _VERTEX_MERGE_ROUND),
            round(point[1], _VERTEX_MERGE_ROUND),
            round(point[2], _VERTEX_MERGE_ROUND),
        )
        node = node_index.get(key)
        if node is None:
            node = len(node_index)
            node_index[key] = node
        return node

    for eid in sorted(edge_ids):
        edge = model.edges[eid]
        points = [model.vertices[vid].coordinates for vid in edge.vertex_ids if vid in model.vertices]
        if len(points) != 2:
            return None
        left = _node_of(points[0])
        right = _node_of(points[1])
        if left == right:
            adjacency.setdefault(left, []).append((left, eid))
        else:
            adjacency.setdefault(left, []).append((right, eid))
            adjacency.setdefault(right, []).append((left, eid))

    # every vertex must have degree exactly 2 (a self-loop counts as 2)
    for node, nbrs in adjacency.items():
        degree = sum(2 if nbr == node else 1 for nbr, _eid in nbrs)
        if degree != 2:
            return None

    used_edges: set[str] = set()
    loops: list[list[str]] = []
    for start in sorted(adjacency):
        if all(eid in used_edges for _nbr, eid in adjacency[start]):
            continue
        nbrs = adjacency[start]
        if len(nbrs) == 1 and nbrs[0][0] == start:
            loops.append([nbrs[0][1]])
            used_edges.add(nbrs[0][1])
            continue
        cycle: list[str] = []
        cur = start
        while True:
            chosen = next(((nbr, eid) for nbr, eid in adjacency[cur] if eid not in used_edges), None)
            if chosen is None:
                return None
            nbr, eid = chosen
            used_edges.add(eid)
            cycle.append(eid)
            cur = nbr
            if cur == start:
                break
        loops.append(cycle)

    if len(used_edges) != len(edge_ids):
        return None
    return loops


def _rebuild_adjacencies(
    model: PartModel,
    keeper: FaceData,
    removed: FaceData,
    keeper_id: str,
    removed_id: str,
    seam_set: set[str],
) -> None:
    """Union keeper/removed adjacencies onto the keeper, consistently redirected.

    Third-party references to the removed face are redirected everywhere;
    keeper<->removed contacts (seam and any other shared edges) disappear with
    the merge; remaining records are deduplicated.  Surviving reciprocal
    adjacencies (keeper <-> C on both sides) are preserved.
    """
    # snapshot the two faces' adjacency records before any redirection
    combined = [
        AdjacencyInfo(ai.adjacent_face_id, ai.shared_edge_id, ai.connection_type)
        for ai in list(keeper.adjacencies) + list(removed.adjacencies)
    ]
    # redirect every reference that points at the removed face
    for fid, face in model.faces.items():
        for ai in face.adjacencies:
            if ai.adjacent_face_id == removed_id:
                ai.adjacent_face_id = keeper_id
    cleaned: list[AdjacencyInfo] = []
    for ai in combined:
        if ai.adjacent_face_id == keeper_id:
            continue  # keeper<->removed contact: internal to the merged face
        if ai.shared_edge_id in seam_set:
            continue
        cleaned.append(ai)
    seen: set[tuple[str, str]] = set()
    result: list[AdjacencyInfo] = []
    for ai in cleaned:
        key = (ai.adjacent_face_id, ai.shared_edge_id)
        if key in seen:
            continue
        seen.add(key)
        result.append(ai)
    keeper.adjacencies = result
    removed.adjacencies = []


def _new_loop_id(model: PartModel, face_id: str) -> str:
    """Generate a free ``loop_<face_idx>_<n>`` id for a rebuilt loop."""
    match = re.fullmatch(r"face_(\d+)", face_id)
    index = match.group(1) if match else "m"
    base = f"loop_{index}_"
    counter = 0
    while f"{base}{counter}" in model.loops:
        counter += 1
    return f"{base}{counter}"
