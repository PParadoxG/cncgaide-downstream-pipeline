"""Topology-only closed side-wall band (SWB) discovery — shared, non-owning.

This module is **structural analysis, not a feature**.  It owns no faces,
registers no node and names no ontology class; it answers one question for
whatever manufacturing-feature module asks it: where are the closed side-wall
bands, and how does each one connect to its neighbours.  Feature semantics
live in the feature modules (``Pocket.py`` today), endpoint traversal and
material-side turning live in ``band_endpoints.py``.

It implements the generic B-Rep topology concepts:

- candidate wall adjacency,
- external wall-band boundary extraction,
- rim grouping into closed cycles,
- closed-cycle validation,
- per-wall rim participation,
- inner-loop matching,
- root-anchored propagation lineage across partner rims.

Nothing here decides Pocket vs Boss semantics, computes machining direction,
or touches AAGNet evidence.  The wall-band notion implemented here is the
*rim-anchored* one:

    A wall band is a set of faces ``W`` that is connected under wall-to-wall
    adjacency, together with two edge-disjoint closed edge cycles ``RimA``
    and ``RimB`` such that every wall face contributes at least one edge to
    each rim.

The connectivity requirement is what stops a rim spanning several unrelated
features from collecting their far-side faces into one disconnected blob:
such faces each touch both rims, so per-wall participation alone accepts
them.  Inner-loop anchoring guarantees connectivity for free, which is why
the requirement only became load-bearing once rim propagation was added.

Discovery is anchored on every inner loop of every face: the anchor loop is
the candidate ``RimA``; the far-side faces of the anchor rim that are
unclaimed form the initial band; the partner ``RimB`` is recovered from the
external boundary of that band (edges with exactly one incident band face),
grouped into closed cycles, and selected by wall participation.  Extra
boundary topology (wall holes, intersecting features, split rims) is
tolerated: it produces additional closed cycles that fail the per-wall
participation test and are reported as ``nonPrincipalRimCycles`` instead of
rejecting the band.

Rim propagation
---------------
Discovery does not stop at the first band behind an inner loop.  Once a band
is accepted, its verified complete principal ``RimB`` becomes the ``RimA`` of
a further search whose *incoming side* is the band just accepted, so the next
band is looked for on the other side of that rim:

    face.inner_loop -> SWB_0 -> RimB -> SWB_1 -> RimB -> SWB_2 -> ...

This recovers the real Pocket/Boss side wall behind a mouth chamfer or fillet
ring, which the ring itself would otherwise hide.  Propagation is
non-greedy — every accepted band on the chain is kept, the deeper band never
replaces the shallower one — and no semantic distinction between transition
rings and principal walls is made here.

Only inner loops act as *root* seeds, so raw exterior boundaries never become
search entry points; the inner-loop locality evidence of the root is
inherited along the chain, which is why a propagated band is not required to
match an inner loop itself.  Seeds are visited breadth-first and deduplicated
on ``(incoming rim signature set, incoming wall face set)``, so the traversal
is finite, deterministic, and cannot walk back into the band it came from.

Wall surface type is deliberately unrestricted: any unclaimed face may act as
a wall.  The only surface-type sanity rule is the single-face guard (a single
planar face can never wrap around, so a flat ring is not a side-wall band).
In particular a mouth chamfer or fillet ring is a perfectly legal band here —
discovery does not, and must not, try to tell a transition ring from a
principal wall.  That distinction needs semantic evidence and is made one
layer up, in ``transition_evidence`` / ``band_endpoints``.

Propagation direction is the lineage a feature module needs: ``rim_a`` is
always the rim the band was *entered* by, walking away from the inner-loop
root, and ``rim_b`` the rim it leads on to.  Pocket and Boss differ only by
the sign of the material turn along that direction, so the direction has to
come from the root-anchored lineage rather than from asking whether some rim
still happens to be an inner loop.

Edge identity
-------------
STEP exporters sometimes emit the same physical curve as several *duplicate
edge entities* (each face loop references its own copy).  The parser keeps
one id per OCCT entity, so such copies are not the same edge id.  This module
therefore uses a deterministic geometric-signature equivalence (curve type +
length + sampled point cloud, identical to the canonical equivalence used by
``hole_topology`` for seam-split copies) to treat coincident duplicates as
one logical edge.  All rim / loop comparisons happen on signatures; reported
edge ids are the canonical representative (smallest id) of each signature
class.  No fuzzy / approximate geometric matching is used.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from contract.partmodel import PartModel, SurfaceType

_MAX_CYCLES_PER_ANCHOR = 64
_MAX_EXTRA_CYCLES = 8
_VERTEX_MERGE_ROUND = 4
_SIGNATURE_ROUND = 6
# Hard traversal bound.  Visited-state dedup already makes propagation finite;
# this only caps pathological chains and is reported when it bites.
_MAX_DISCOVERY_DEPTH = 8

SEED_INNER_LOOP = "inner_loop"
SEED_PROPAGATED_RIM = "propagated_rim"


@dataclass(frozen=True)
class RimMatch:
    """Exact inner-loop match record for one principal rim."""

    face_id: str
    loop_id: str
    method: str


@dataclass
class WallBandCandidate:
    """One accepted closed side-wall band (walls only; no semantics)."""

    anchor_face_id: str
    anchor_loop_id: str
    wall_face_ids: list[str]
    excluded_wall_candidate_face_ids: list[str]
    wall_surface_types: dict[str, str]
    wall_wall_edge_count: int
    rim_a_edge_ids: list[str]
    rim_b_edge_ids: list[str]
    rim_a_wall_face_ids: list[str]
    rim_b_wall_face_ids: list[str]
    rim_a_adjacent_face_ids: list[str]
    rim_b_adjacent_face_ids: list[str]
    rim_a_is_inner_loop: bool
    rim_b_is_inner_loop: bool
    rim_a_match: RimMatch | None
    rim_b_match: RimMatch | None
    extra_cycle_edge_sets: list[list[str]]
    acceptance_reason: str
    # --- propagation bookkeeping (root band = depth 0, inner-loop seeded) ---
    discovery_depth: int = 0
    seed_type: str = SEED_INNER_LOOP
    candidate_key: str = ""
    parent_candidate_key: str | None = None
    incoming_wall_face_ids: list[str] = field(default_factory=list)
    # Rim signature sets, kept so downstream consumers can traverse outward
    # from either end without re-deriving them.  ``rim_a`` is always the
    # *incoming* rim in root-anchored propagation order and ``rim_b`` the
    # *outgoing* one — the direction a Pocket/Boss decision depends on.
    rim_a_sigs: frozenset = frozenset()
    rim_b_sigs: frozenset = frozenset()


@dataclass
class WallBandRejection:
    """Compact rejection record for one seed (inner loop or propagated rim)."""

    anchor_face_id: str
    anchor_loop_id: str
    reason: str
    detail: str = ""
    seed_type: str = SEED_INNER_LOOP
    discovery_depth: int = 0


@dataclass(frozen=True)
class _Seed:
    """One wall-band search entry: a candidate RimA plus its incoming side."""

    rim_sigs: frozenset
    incoming_face_ids: frozenset
    root_anchor_face_id: str
    root_anchor_loop_id: str
    seed_type: str
    depth: int
    parent_candidate_key: str | None

    @property
    def state(self) -> tuple[frozenset, frozenset]:
        return (self.rim_sigs, self.incoming_face_ids)


@dataclass
class WallBandAnalysis:
    candidates: list[WallBandCandidate] = field(default_factory=list)
    rejections: list[WallBandRejection] = field(default_factory=list)
    wall_candidate_face_count: int = 0
    inner_loop_anchor_count: int = 0
    root_seed_count: int = 0
    propagated_seed_count: int = 0
    duplicate_band_count: int = 0
    depth_limit_hits: int = 0


@dataclass
class EdgeIndex:
    """Per-part edge bookkeeping with geometric-signature equivalence."""

    edge_sig: dict[str, tuple]
    sig_to_edge_ids: dict[tuple, list[str]]
    sig_incidence: dict[tuple, set[str]]
    loop_duplicate_edges: set[str]
    self_seam_sigs: set[tuple]

    def representative(self, sig: tuple) -> str:
        return self.sig_to_edge_ids[sig][0]


def discover_wall_bands(model: PartModel, claimed_face_ids: set[str] | None = None) -> WallBandAnalysis:
    """Discover all closed wall bands reachable from inner loops.

    Root seeds are inner loops; every accepted band additionally seeds a
    further search across its verified partner rim (see module docstring).

    ``claimed_face_ids`` are faces already owned by earlier feature modules
    (ThroughHole / BlindHole).  A band is rejected when one of its far-side
    faces is claimed by an *earlier* module; overlap with this module's own
    earlier candidates is tolerated — adjacent bands may share a wall face,
    and a band already claimed here must not block topology propagation.
    """
    claimed = set(claimed_face_ids or ())
    index = build_edge_index(model)
    wall_ids = set(model.faces)

    anchors: list[tuple[str, str, frozenset]] = []
    for face_id in sorted(model.faces, key=face_sort_key):
        face = model.faces[face_id]
        for loop_id in face.loop_ids:
            loop = model.loops.get(loop_id)
            if loop is None or loop.is_outer or not loop.edge_ids:
                continue
            rim_sigs = frozenset(index.edge_sig[eid] for eid in loop.edge_ids if eid in index.edge_sig)
            if not rim_sigs:
                continue
            anchors.append((face_id, loop_id, rim_sigs))

    analysis = WallBandAnalysis(
        wall_candidate_face_count=len(wall_ids),
        inner_loop_anchor_count=len(anchors),
    )
    self_claimed: set[str] = set()
    seen_bands: dict[frozenset[str], str] = {}
    visited: set[tuple[frozenset, frozenset]] = set()
    queue: deque[_Seed] = deque()

    # Root seeds first, in face/loop order, so root discovery is unchanged by
    # the presence of propagation; the FIFO queue then walks depth by depth.
    for anchor_face_id, anchor_loop_id, rim_a in anchors:
        seed = _Seed(
            rim_sigs=rim_a,
            incoming_face_ids=frozenset({anchor_face_id}),
            root_anchor_face_id=anchor_face_id,
            root_anchor_loop_id=anchor_loop_id,
            seed_type=SEED_INNER_LOOP,
            depth=0,
            parent_candidate_key=None,
        )
        if seed.state in visited:
            continue
        visited.add(seed.state)
        queue.append(seed)
    analysis.root_seed_count = len(queue)

    while queue:
        seed = queue.popleft()
        result = _analyze_seed(model, index, wall_ids, claimed, self_claimed, seed)
        if isinstance(result, WallBandRejection):
            analysis.rejections.append(result)
            continue
        key = frozenset(result.wall_face_ids)
        existing = seen_bands.get(key)
        if existing is None:
            candidate_key = f"WB{len(analysis.candidates):03d}"
            result.candidate_key = candidate_key
            seen_bands[key] = candidate_key
            analysis.candidates.append(result)
            self_claimed.update(result.wall_face_ids)
        else:
            # Same wall set reached again from a different seed: report it
            # once, but keep propagating so a rim only this seed exposes is
            # still followed.
            candidate_key = existing
            analysis.duplicate_band_count += 1
        if seed.depth >= _MAX_DISCOVERY_DEPTH:
            analysis.depth_limit_hits += 1
            continue
        # Cross the verified partner rim: the band just accepted is the
        # incoming side, so the next band is looked for beyond the rim rather
        # than back where we came from.
        nxt = _Seed(
            rim_sigs=result.rim_b_sigs,
            incoming_face_ids=key,
            root_anchor_face_id=seed.root_anchor_face_id,
            root_anchor_loop_id=seed.root_anchor_loop_id,
            seed_type=SEED_PROPAGATED_RIM,
            depth=seed.depth + 1,
            parent_candidate_key=candidate_key,
        )
        if not nxt.rim_sigs or nxt.state in visited:
            continue
        visited.add(nxt.state)
        queue.append(nxt)
        analysis.propagated_seed_count += 1
    return analysis


# ---------------------------------------------------------------------------
# Diagnostics rendering
#
# The SWB layer owns no feature node, so its findings are published as part
# diagnostics instead.  These builders used to live in the retired
# ``PocketBossCandidate`` audit feature; the audit value was worth keeping,
# the face ownership was not.
# ---------------------------------------------------------------------------

_MAX_REJECTED_SEEDS_IN_DIAGNOSTICS = 40


def analysis_diagnostics(analysis: WallBandAnalysis) -> dict[str, Any]:
    """Whole-part SWB diagnostics for ``result.diagnostics.sideWallBands``."""
    return {
        "role": "shared_structural_analysis_not_a_feature",
        "ownsFaces": False,
        "aagnetUsed": False,
        "method": "topology_only_closed_wall_band_v2_rim_propagation",
        "wallSurfaceTypePolicy": "unrestricted",
        "innerLoopAnchorCount": analysis.inner_loop_anchor_count,
        "rootSeedCount": analysis.root_seed_count,
        "propagatedSeedCount": analysis.propagated_seed_count,
        "duplicateBandCount": analysis.duplicate_band_count,
        "depthLimitHits": analysis.depth_limit_hits,
        "wallCandidateFaceCount": analysis.wall_candidate_face_count,
        "bandCount": len(analysis.candidates),
        "bandCountByDiscoveryDepth": _depth_summary(analysis.candidates),
        "innerLoopMatchMethod": "exact_signature_set_match",
        "edgeIdEquivalence": "geometric_signature_based_canonical_representatives",
        "bands": [candidate_diagnostics(candidate) for candidate in analysis.candidates],
        "rejectionSummary": _rejection_summary(analysis.rejections),
        "rejectedSeeds": [_rejection_dict(item) for item in analysis.rejections[:_MAX_REJECTED_SEEDS_IN_DIAGNOSTICS]],
    }


def candidate_diagnostics(candidate: WallBandCandidate) -> dict[str, Any]:
    """One band's structural record, reusable inside a feature node."""
    return {
        "method": "closed_wall_band_v2",
        "acceptanceReason": candidate.acceptance_reason,
        "candidateKey": candidate.candidate_key,
        "discoveryDepth": candidate.discovery_depth,
        "seedType": candidate.seed_type,
        "parentCandidateKey": candidate.parent_candidate_key,
        "incomingRimEdgeIds": list(candidate.rim_a_edge_ids),
        "incomingWallFaceIds": list(candidate.incoming_wall_face_ids),
        "rootAnchor": {
            "faceId": candidate.anchor_face_id,
            "loopId": candidate.anchor_loop_id,
            "loopIsInner": True,
        },
        "wallFaceIds": list(candidate.wall_face_ids),
        "wallSurfaceTypes": dict(candidate.wall_surface_types),
        "wallConnectivitySummary": {
            "wallFaceCount": len(candidate.wall_face_ids),
            "wallWallEdgeCount": candidate.wall_wall_edge_count,
            "circumferentiallyClosed": True,
            "closedVia": "two_principal_rim_cycles",
        },
        "excludedWallCandidateFaceIds": list(candidate.excluded_wall_candidate_face_ids),
        "rimA": _rim_diagnostics(candidate.rim_a_edge_ids, candidate.rim_a_wall_face_ids, candidate.rim_a_adjacent_face_ids, candidate.rim_a_is_inner_loop, candidate.rim_a_match),
        "rimB": _rim_diagnostics(candidate.rim_b_edge_ids, candidate.rim_b_wall_face_ids, candidate.rim_b_adjacent_face_ids, candidate.rim_b_is_inner_loop, candidate.rim_b_match),
        "nonPrincipalRimCycles": [list(edge_ids) for edge_ids in candidate.extra_cycle_edge_sets],
    }


def _rim_diagnostics(edge_ids, wall_face_ids, adjacent_face_ids, is_inner_loop: bool, match) -> dict[str, Any]:
    return {
        "edgeIds": list(edge_ids),
        "contributingWallFaceIds": list(wall_face_ids),
        "adjacentFaceIds": list(adjacent_face_ids),
        "isInnerLoop": bool(is_inner_loop),
        "innerLoopMatch": {"faceId": match.face_id, "loopId": match.loop_id, "method": match.method} if match is not None else None,
    }


def _depth_summary(candidates: list[WallBandCandidate]) -> dict[str, int]:
    summary: dict[int, int] = {}
    for candidate in candidates:
        summary[candidate.discovery_depth] = summary.get(candidate.discovery_depth, 0) + 1
    return {str(depth): count for depth, count in sorted(summary.items())}


def _rejection_summary(rejections: list[WallBandRejection]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for item in rejections:
        summary[item.reason] = summary.get(item.reason, 0) + 1
    return dict(sorted(summary.items()))


def _rejection_dict(item: WallBandRejection) -> dict[str, Any]:
    return {
        "rootAnchorFaceId": item.anchor_face_id,
        "rootAnchorLoopId": item.anchor_loop_id,
        "seedType": item.seed_type,
        "discoveryDepth": item.discovery_depth,
        "reason": item.reason,
    }


# ---------------------------------------------------------------------------
# Per-seed analysis
# ---------------------------------------------------------------------------

def _analyze_seed(
    model: PartModel,
    index: EdgeIndex,
    wall_ids: set[str],
    claimed: set[str],
    self_claimed: set[str],
    seed: _Seed,
) -> WallBandCandidate | WallBandRejection:
    """Recover the wall band lying beyond ``seed.rim_sigs``.

    The seed rim plays the role of ``RimA``; ``seed.incoming_face_ids`` is the
    side the search arrives from (the anchor face for a root seed, the parent
    band for a propagated seed) and is excluded from the new band, so the
    search always crosses the rim instead of turning back.
    """
    rim_a = seed.rim_sigs
    anchor_face_id = seed.root_anchor_face_id
    anchor_loop_id = seed.root_anchor_loop_id
    is_root = seed.seed_type == SEED_INNER_LOOP

    def _reject(reason: str, detail: str = "") -> WallBandRejection:
        return WallBandRejection(anchor_face_id, anchor_loop_id, reason, detail, seed.seed_type, seed.depth)

    # ---- 1. far-side faces of the seed rim --------------------------------
    far_faces: set[str] = set()
    for sig in rim_a:
        far_faces.update(index.sig_incidence.get(sig, ()))
    far_faces -= seed.incoming_face_ids
    if not far_faces:
        return _reject("anchor_rim_has_no_wall_faces", "no new face shares any edge of the seed rim")

    intruders = sorted(face_id for face_id in far_faces if face_id in claimed and face_id not in self_claimed)
    if intruders:
        return _reject("contains_claimed_face", "claimed by earlier feature modules: " + ",".join(intruders))

    band_raw = {face_id for face_id in far_faces if face_id in wall_ids}
    if not band_raw:
        return _reject("unsupported_surface_type", "far side of the anchor rim has no wall face")

    # ---- 2. external boundary of the band (partner-rim candidates) --------
    external: dict[tuple, set[str]] = {}
    for sig, faces in index.sig_incidence.items():
        if sig in rim_a or sig in index.self_seam_sigs:
            continue
        if len(faces & band_raw) == 1:
            external[sig] = faces

    cycles = _closed_cycles(model, index, list(external))
    if not cycles:
        return _reject("rim_not_closed", "no closed boundary cycle at the far end of the wall band")

    # ---- 3. choose the partner rim by wall participation ------------------
    scored: list[tuple[int, set[str], list[tuple]]] = []
    for cycle in cycles:
        participating: set[str] = set()
        for sig in cycle:
            participating.update(external.get(sig, ()) & band_raw)
        scored.append((len(participating), participating, cycle))
    scored.sort(key=lambda item: (-item[0], sorted(item[1]), sorted(item[2])))
    tied = [item for item in scored if item[0] == scored[0][0]]
    if len(tied) > 1:
        return _reject("principal_rim_count_not_two", f"{len(tied)} far-end boundary cycles tie on wall participation")
    _, band_set, rim_b = tied[0]
    if not band_set:
        return _reject("rim_not_closed", "no wall face participates in any far-end boundary cycle")

    # ---- 4. per-wall participation on both principal rims -----------------
    def _shares(face_id: str, sigs) -> bool:
        return any(face_id in index.sig_incidence.get(sig, ()) for sig in sigs)

    missing_a = sorted(face_id for face_id in band_set if not _shares(face_id, rim_a))
    if missing_a:
        return _reject("wall_missing_rim_a_contribution", ",".join(missing_a))
    missing_b = sorted(face_id for face_id in band_set if not _shares(face_id, rim_b))
    if missing_b:
        return _reject("wall_missing_rim_b_contribution", ",".join(missing_b))

    # ---- 4b. the walls must form ONE connected ring -----------------------
    # An inner-loop anchor bounds a single opening, so a root band is
    # connected by construction.  A propagated rim can instead span several
    # unrelated features at the far end of a band: their far-side faces each
    # touch both rims without ever touching each other, which satisfies the
    # per-wall participation test while forming disconnected groups rather
    # than a band.  Connectivity is what "circumferentially closed" already
    # asserts, so it is required of every band, root and propagated alike.
    adjacency, wall_wall_edges = _wall_adjacency(index, band_set)
    components = _connected_component_count(adjacency)
    if components > 1:
        return _reject("wall_band_not_connected", f"{len(band_set)} wall faces form {components} disconnected groups")

    if len(band_set) == 1:
        single = next(iter(band_set))
        if model.faces[single].surface_type == SurfaceType.PLANE:
            return _reject("not_closed_wall_band", "single planar wall face cannot form a circumferentially closed band")
    # ---- 5. inner-loop matching -------------------------------------------
    # Only a root band has to prove inner-loop locality itself.  A propagated
    # band inherits it through the accepted parent band and its verified
    # partner rim, so it is reported but not gated on.
    rim_a_match = match_inner_loop(model, index, rim_a, band_set)
    rim_b_match = match_inner_loop(model, index, rim_b, band_set)
    if is_root and rim_a_match is None and rim_b_match is None:
        return _reject("neither_rim_is_inner_loop", "neither principal rim matches an adjacent face inner loop")

    extra_cycles = [sorted(index.representative(sig) for sig in cycle) for cycle in cycles if set(cycle) != set(rim_b)][:_MAX_EXTRA_CYCLES]
    band = sorted(band_set, key=face_sort_key)
    return WallBandCandidate(
        anchor_face_id=anchor_face_id,
        anchor_loop_id=anchor_loop_id,
        wall_face_ids=band,
        excluded_wall_candidate_face_ids=sorted(band_raw - band_set, key=face_sort_key),
        wall_surface_types={face_id: model.faces[face_id].surface_type.value for face_id in band},
        wall_wall_edge_count=wall_wall_edges,
        rim_a_edge_ids=sorted(index.representative(sig) for sig in rim_a),
        rim_b_edge_ids=sorted(index.representative(sig) for sig in rim_b),
        rim_a_wall_face_ids=sorted(face_id for face_id in band_set if _shares(face_id, rim_a)),
        rim_b_wall_face_ids=sorted(face_id for face_id in band_set if _shares(face_id, rim_b)),
        rim_a_adjacent_face_ids=_rim_adjacent_faces(model, index, rim_a, band_set),
        rim_b_adjacent_face_ids=_rim_adjacent_faces(model, index, rim_b, band_set),
        rim_a_is_inner_loop=rim_a_match is not None,
        rim_b_is_inner_loop=rim_b_match is not None,
        rim_a_match=rim_a_match,
        rim_b_match=rim_b_match,
        extra_cycle_edge_sets=extra_cycles,
        acceptance_reason="closed_wall_band_with_inner_rim" if is_root else "closed_wall_band_from_propagated_rim",
        discovery_depth=seed.depth,
        seed_type=seed.seed_type,
        parent_candidate_key=seed.parent_candidate_key,
        incoming_wall_face_ids=sorted(seed.incoming_face_ids, key=face_sort_key),
        rim_a_sigs=frozenset(rim_a),
        rim_b_sigs=frozenset(rim_b),
    )


def _wall_adjacency(index: EdgeIndex, band: set[str]) -> tuple[dict[str, set[str]], int]:
    """Wall-to-wall adjacency inside the band, by canonical edge signature."""
    adjacency: dict[str, set[str]] = {face_id: set() for face_id in band}
    wall_wall_edges = 0
    for faces in index.sig_incidence.values():
        inside = faces & band
        if len(inside) != 2:
            continue
        wall_wall_edges += 1
        left, right = sorted(inside)
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency, wall_wall_edges


def _connected_component_count(adjacency: dict[str, set[str]]) -> int:
    seen: set[str] = set()
    components = 0
    for start in sorted(adjacency):
        if start in seen:
            continue
        components += 1
        stack = [start]
        seen.add(start)
        while stack:
            for neighbor in adjacency[stack.pop()]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
    return components


def _rim_adjacent_faces(model: PartModel, index: EdgeIndex, rim_sigs, band: set[str]) -> list[str]:
    """Faces sharing at least one rim edge (by signature), excluding the band walls."""
    adjacent: set[str] = set()
    for sig in rim_sigs:
        adjacent.update(index.sig_incidence.get(sig, ()))
    return sorted(adjacent - band, key=face_sort_key)


def match_inner_loop(model: PartModel, index: EdgeIndex, rim_sigs, exclude_face_ids: set[str]) -> RimMatch | None:
    """Strict topology match: the complete rim equals a complete inner loop.

    Matching method: exact edge-set equality on canonical geometric
    signatures (``set(rim_signatures) == set(loop_edge_signatures)``).  This
    is the deterministic canonical equivalence used across the parser's
    duplicate / split edge entities — not a fuzzy geometric approximation.

    Discovery uses it on the near side of a band; it is public because the
    same question — *does this rim bound a hole in the face beyond it?* — is
    what tells a genuine opening from a face the rim merely runs along, and
    ``ThroughCut`` asks it of the terminal region at each end of a band.
    ``exclude_face_ids`` is the side the rim is being looked at *from*.
    """
    rim_set = set(rim_sigs)
    adjacent: set[str] = set()
    for sig in rim_sigs:
        adjacent.update(index.sig_incidence.get(sig, ()))
    for face_id in sorted(adjacent - set(exclude_face_ids), key=face_sort_key):
        face = model.faces.get(face_id)
        if face is None:
            continue
        for loop_id in face.loop_ids:
            loop = model.loops.get(loop_id)
            if loop is None or loop.is_outer:
                continue
            loop_sigs = {index.edge_sig[eid] for eid in loop.edge_ids if eid in index.edge_sig}
            if loop_sigs == rim_set:
                return RimMatch(face_id=face_id, loop_id=loop_id, method="exact_signature_set_match")
    return None


# ---------------------------------------------------------------------------
# Edge index
# ---------------------------------------------------------------------------

def build_edge_index(model: PartModel) -> EdgeIndex:
    incidence: dict[str, set[str]] = {}
    for face_id, face in model.faces.items():
        for loop_id in face.loop_ids:
            loop = model.loops.get(loop_id)
            if loop is None:
                continue
            for edge_id in loop.edge_ids:
                if edge_id in model.edges:
                    incidence.setdefault(edge_id, set()).add(face_id)

    duplicated: set[str] = set()
    for loop in model.loops.values():
        seen: set[str] = set()
        for edge_id in loop.edge_ids:
            if edge_id in seen:
                duplicated.add(edge_id)
            seen.add(edge_id)

    edge_sig: dict[str, tuple] = {}
    sig_to_edge_ids: dict[tuple, list[str]] = {}
    for edge_id in sorted(model.edges):
        sig = _edge_signature(model.edges[edge_id], model)
        edge_sig[edge_id] = sig
        sig_to_edge_ids.setdefault(sig, []).append(edge_id)

    sig_incidence: dict[tuple, set[str]] = {}
    for sig, edge_ids in sig_to_edge_ids.items():
        faces: set[str] = set()
        for edge_id in edge_ids:
            faces.update(incidence.get(edge_id, ()))
        sig_incidence[sig] = faces

    self_seam_sigs: set[tuple] = set()
    for sig, edge_ids in sig_to_edge_ids.items():
        if len(sig_incidence[sig]) != 1:
            continue
        if all(model.edges[edge_id].is_seam or edge_id in duplicated for edge_id in edge_ids):
            self_seam_sigs.add(sig)

    return EdgeIndex(
        edge_sig=edge_sig,
        sig_to_edge_ids=sig_to_edge_ids,
        sig_incidence=sig_incidence,
        loop_duplicate_edges=duplicated,
        self_seam_sigs=self_seam_sigs,
    )


def _edge_signature(edge, model: PartModel | None = None) -> tuple:
    """Deterministic geometric signature: curve type + length + point cloud.

    The point cloud is the parser's persisted ``sample_points`` when present,
    otherwise the edge endpoint coordinates.  Two edges with the same
    signature are geometrically coincident copies of one logical curve.
    """
    samples = [list(point) for point in getattr(edge, "sample_points", None) or []]
    if not samples and model is not None:
        samples = [
            list(model.vertices[vertex_id].coordinates)
            for vertex_id in getattr(edge, "vertex_ids", None) or []
            if vertex_id in model.vertices
        ]
    return _signature_from_points(edge, samples)


def _signature_from_points(edge, samples: list[list[float]]) -> tuple:
    rounded = tuple(sorted(tuple(round(float(value), _SIGNATURE_ROUND) for value in point) for point in samples)) if samples else ()
    curve_type = edge.curve_type.value if hasattr(edge.curve_type, "value") else str(edge.curve_type)
    return (curve_type, round(float(edge.length), _SIGNATURE_ROUND), rounded)


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def _closed_cycles(model: PartModel, index: EdgeIndex, sigs: list[tuple]) -> list[list[tuple]]:
    """Enumerate simple closed cycles of the edge subgraph (over signatures).

    Vertices are merged by rounded coordinates (the parser already deduplicates
    vertices by OCCT hash, so this only guards against coincident duplicates).
    Self-loop edges (closed circle edges) are valid single-edge cycles.
    """
    node_points: list[list[float]] = []
    node_index: dict[tuple, int] = {}

    def _node_of(point: list[float]) -> int:
        key = (round(point[0], _VERTEX_MERGE_ROUND), round(point[1], _VERTEX_MERGE_ROUND), round(point[2], _VERTEX_MERGE_ROUND))
        node = node_index.get(key)
        if node is None:
            node = len(node_points)
            node_points.append(point)
            node_index[key] = node
        return node

    adjacency: dict[int, list[tuple[int, tuple]]] = {}
    for sig in sorted(sigs):
        edge_id = index.representative(sig)
        edge = model.edges[edge_id]
        points = [
            list(model.vertices[vertex_id].coordinates)
            for vertex_id in edge.vertex_ids
            if vertex_id in model.vertices
        ]
        if len(points) != 2:
            continue
        left = _node_of(points[0])
        right = _node_of(points[1])
        if left == right:
            adjacency.setdefault(left, []).extend([(left, sig), (left, sig)])
        else:
            adjacency.setdefault(left, []).append((right, sig))
            adjacency.setdefault(right, []).append((left, sig))

    seen: set[frozenset] = set()
    cycles: list[list[tuple]] = []

    def _dfs(start: int, node: int, path_edges: list[tuple], visited: set[int]) -> None:
        if len(cycles) >= _MAX_CYCLES_PER_ANCHOR:
            return
        for neighbor, sig in adjacency.get(node, []):
            if neighbor == start and path_edges and sig not in path_edges:
                signature = frozenset([*path_edges, sig])
                if signature not in seen:
                    seen.add(signature)
                    cycles.append([*path_edges, sig])
                continue
            if neighbor == node and not path_edges:
                signature = frozenset([sig])
                if signature not in seen:
                    seen.add(signature)
                    cycles.append([sig])
                continue
            if neighbor in visited:
                continue
            visited.add(neighbor)
            path_edges.append(sig)
            _dfs(start, neighbor, path_edges, visited)
            path_edges.pop()
            visited.remove(neighbor)

    for start in sorted(adjacency):
        _dfs(start, start, [], {start})
        if len(cycles) >= _MAX_CYCLES_PER_ANCHOR:
            break
    return cycles


def face_sort_key(face_id: str) -> tuple[int, str]:
    match = re.fullmatch(r"face_(\d+)", face_id)
    return (int(match.group(1)), face_id) if match else (10**9, face_id)
