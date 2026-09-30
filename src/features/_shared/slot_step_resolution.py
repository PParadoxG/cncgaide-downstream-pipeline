"""Semantic resolution shared by the formal :mod:`Slot` and :mod:`Step` nodes.

``open_sweep`` deliberately stops at a topology/geometry proof.  This module
is the next, owning-feature boundary: it resolves the effective material side
of both transverse terminals, applies structural Slot dispatch, and aggregates
the complete AAGNet per-face softmax only for the genuinely ambiguous
two-planar-face/one-longitudinal-rail form.  Keeping this code outside
``open_sweep.py`` is intentional; the shared OS analysis remains
semantic-neutral and non-owning.

The resolver never uses AAGNet instance grouping, top-k truncation, geometric
Slot-vs-Step heuristics, or a score margin.  A candidate is rejected when the
evidence is absent/malformed or when the two family scores are tied.

A candidate dispatched to Slot then has to survive two solid-classifier
proofs, which together say what a slot is: ``prove_two_ended_exterior`` that
it is open along its course, and ``prove_transverse_enclosure`` that it is
closed across it.  Either failing is a named rejection carrying its own
sample points, so the decision can be reopened without re-running anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from contract.partmodel import CurveType, PartModel, SurfaceType

from contract.evidence import FEATURE_NAMES, evidence_by_partmodel_face
from features._shared.band_endpoints import (
    CONVEX,
    EndpointResolution,
    SolidPointClassifier,
    resolve_endpoint,
)
from features._shared.closed_wall_band import EdgeIndex, build_edge_index
from topology.faces import face_sort_key
from topology.vectors import coerced_unit_vector, dot, norm
from features._shared.open_sweep import OpenSweepAnalysis, OpenSweepCandidate
from features._shared.transition_evidence import TransitionEvidence


SLOT_FAMILIES = (
    "triangular_through_slot",
    "rectangular_through_slot",
    "circular_through_slot",
    "rectangular_blind_slot",
    "v_circular_end_blind_slot",
    "h_circular_end_blind_slot",
)
STEP_FAMILIES = (
    "rectangular_through_step",
    "2sides_through_step",
    "slanted_through_step",
    "triangular_blind_step",
    "circular_blind_step",
    "rectangular_blind_step",
)

_FAMILY_LABELS = SLOT_FAMILIES + STEP_FAMILIES
_FAMILY_INDEXES = {label: FEATURE_NAMES.index(label) for label in _FAMILY_LABELS}
_SOFTMAX_TOL = 2.0e-3
_VECTOR_TOL = 1.0e-12
_PLANE_PARALLEL_TOL = 1.0e-6


@dataclass
class CandidateResolution:
    """Auditable semantic decision for one accepted OS candidate."""

    candidate: OpenSweepCandidate
    decision: str | None = None
    reject_reason: str | None = None
    reject_detail: str = ""
    terminal_resolutions: list[EndpointResolution] = field(default_factory=list)
    terminal_gate: dict[str, Any] = field(default_factory=dict)
    face_evidence: list[dict[str, Any]] = field(default_factory=list)
    slot_score: float | None = None
    step_score: float | None = None
    longitudinal_open_proof: dict[str, Any] | None = None
    transverse_enclosure_proof: dict[str, Any] | None = None
    step_structure: dict[str, Any] | None = None
    semantic_basis: str | None = None
    aagnet_applicable: bool = False

    @property
    def accepted(self) -> bool:
        return self.decision in {"Slot", "Step"}

    def diagnostics(self) -> dict[str, Any]:
        candidate = self.candidate
        return {
            "candidateKey": candidate.candidate_key,
            "principalFaceIds": list(candidate.principal_face_ids),
            "discoveryMode": candidate.discovery_mode,
            "terminalResolution": {
                "required": "both effective terminal material-side relations must resolve to CONVEX",
                "gatePassed": bool(self.terminal_gate.get("passed", False)),
                "terminals": [
                    {
                        "principalFaceId": candidate.profile_terminal_face_ids[index],
                        "railEdgeId": candidate.profile_terminal_rail_edge_ids[index],
                        "directOsContext": candidate.profile_terminal_effective_contexts[index],
                        "resolution": self.terminal_resolutions[index].to_dict()
                        if index < len(self.terminal_resolutions)
                        else None,
                    }
                    for index in range(2)
                ],
                "detail": self.terminal_gate.get("detail", ""),
            },
            "faceEvidence": self.face_evidence,
            "slotFamilyScore": self.slot_score,
            "stepFamilyScore": self.step_score,
            "semanticDecision": self.decision,
            "rejected": self.reject_reason is not None,
            "rejectReason": self.reject_reason,
            "rejectDetail": self.reject_detail,
            "longitudinalOpenProof": self.longitudinal_open_proof,
            "transverseEnclosureProof": self.transverse_enclosure_proof,
            "stepStructure": self.step_structure,
            "semanticBasis": self.semantic_basis,
            "aagnetAggregation": {
                "source": "complete_per_face_softmax" if self.aagnet_applicable else "not_applied",
                "applicability": "ambiguous_two_planar_faces_one_complete_longitudinal_rail"
                if self.aagnet_applicable else "not_applicable_nonambiguous_structural_slot",
                "notAppliedReason": None if self.aagnet_applicable else (
                    "candidate is not the genuinely ambiguous exactly-two-planar-face/one-rail form"
                ),
                "slotFamilies": list(SLOT_FAMILIES),
                "stepFamilies": list(STEP_FAMILIES),
                "rawVotesEqualWeightWithinMergedPartModelFace": True,
                "principalFacesEqualWeight": True,
                "nativeInstanceGroupingUsed": False,
                "marginThreshold": None,
            },
        }


def resolve_open_sweep_candidates(
    model: PartModel,
    evidence_document: dict | None,
    sweeps: OpenSweepAnalysis,
    *,
    transition: TransitionEvidence,
    kernel: Any | None = None,
    claimed_face_ids: set[str] | None = None,
    edge_index: EdgeIndex | None = None,
) -> list[CandidateResolution]:
    """Resolve OS candidates once for the Slot and Step owning modules."""

    index = edge_index if edge_index is not None else build_edge_index(model)
    classifier = SolidPointClassifier(kernel)
    resolutions: list[CandidateResolution] = []
    claimed = set(claimed_face_ids or ())

    for candidate in sweeps.candidates:
        resolution = CandidateResolution(candidate=candidate)
        resolutions.append(resolution)
        overlap = sorted(set(candidate.principal_face_ids) & claimed, key=face_sort_key)
        if overlap:
            _reject(resolution, "principal_faces_already_claimed", ",".join(overlap))
            continue

        terminal_resolutions, terminal_gate = _resolve_terminals(
            model, index, transition, classifier, candidate
        )
        resolution.terminal_resolutions = terminal_resolutions
        resolution.terminal_gate = terminal_gate
        if not terminal_gate.get("passed", False):
            _reject(
                resolution,
                "recessed_terminal_context_not_both_convex",
                terminal_gate.get("detail", "both effective terminal contexts did not resolve to CONVEX"),
            )
            continue

        resolution.aagnet_applicable = _is_ambiguous_two_plane_rail(model, candidate)
        if resolution.aagnet_applicable:
            resolution.semantic_basis = "aagnet_full_softmax_ambiguous_two_planar_faces_one_rail"
            grouped = evidence_by_partmodel_face(evidence_document or {})
            aggregate, error = _aggregate_family_evidence(model, grouped, candidate)
            if error is not None:
                resolution.face_evidence = aggregate.get("faceEvidence", [])
                _reject(resolution, error[0], error[1])
                continue
            resolution.face_evidence = aggregate["faceEvidence"]
            resolution.slot_score = aggregate["slotScore"]
            resolution.step_score = aggregate["stepScore"]
            if resolution.slot_score == resolution.step_score:
                _reject(resolution, "aagnet_family_score_tie", "aggregate Slot and Step family scores are equal")
                continue
            wants_slot = resolution.slot_score > resolution.step_score
        else:
            resolution.semantic_basis = "structural_nonambiguous_open_profile_slot"
            resolution.face_evidence = [{
                "status": "not_applicable",
                "reason": "nonambiguous open profile is structurally dispatched to Slot; AAGNet Slot/Step resolver is not applied",
                "principalFaceIds": list(candidate.principal_face_ids),
            }]
            wants_slot = True

        if wants_slot:
            resolution.decision = "Slot"
            proof = prove_two_ended_exterior(model, candidate, classifier)
            resolution.longitudinal_open_proof = proof
            if not proof.get("passed", False):
                _reject(
                    resolution,
                    proof.get("reason", "slot_longitudinal_exterior_proof_failed"),
                    proof.get("detail", ""),
                )
                continue
            # Open along the course is only half of a slot.  Without the other
            # half a re-entrant corner of the part's own outline passes every
            # test above - including, on this corpus, a 0.978 AAGNet vote for
            # the Slot family.  A learned score is evidence; whether there is
            # material on both sides is a fact, and the fact decides.
            enclosure = prove_transverse_enclosure(model, candidate, classifier)
            resolution.transverse_enclosure_proof = enclosure
            if not enclosure.get("passed", False):
                _reject(
                    resolution,
                    enclosure.get("reason", "slot_transverse_enclosure_proof_failed"),
                    enclosure.get("detail", ""),
                )
                continue
        else:
            structure = simple_step_structure(model, candidate)
            resolution.step_structure = structure
            if not structure.get("passed", False):
                _reject(
                    resolution,
                    "step_not_ontology_compatible_simple_step",
                    structure.get("detail", "the candidate is not exactly two planar faces with one straight shared edge"),
                )
                continue
            resolution.decision = "Step"

    return resolutions


def _is_ambiguous_two_plane_rail(model: PartModel, candidate: OpenSweepCandidate) -> bool:
    """Identify the only OS form for which learned Slot/Step semantics apply."""

    if len(candidate.principal_face_ids) != 2:
        return False
    if len(candidate.ordered_profile_face_ids) != 2:
        return False
    if set(candidate.ordered_profile_face_ids) != set(candidate.principal_face_ids):
        return False
    if len(candidate.internal_rail_edge_ids) != 1:
        return False
    if any(
        model.faces.get(face_id) is None
        or model.faces[face_id].surface_type != SurfaceType.PLANE
        for face_id in candidate.principal_face_ids
    ):
        return False
    edge = model.edges.get(candidate.internal_rail_edge_ids[0])
    return edge is not None and edge.curve_type == CurveType.LINE


def _reject(resolution: CandidateResolution, reason: str, detail: str = "") -> None:
    resolution.decision = None
    resolution.reject_reason = reason
    resolution.reject_detail = detail


def _resolve_terminals(
    model: PartModel,
    index: EdgeIndex,
    transition: TransitionEvidence,
    classifier: SolidPointClassifier,
    candidate: OpenSweepCandidate,
) -> tuple[list[EndpointResolution], dict[str, Any]]:
    """Resolve both OS transverse terminals through only known transitions."""

    resolved: list[EndpointResolution] = []
    details: list[str] = []
    for position in range(2):
        face_id = candidate.profile_terminal_face_ids[position]
        edge_id = candidate.profile_terminal_rail_edge_ids[position]
        sig = index.edge_sig.get(edge_id)
        if sig is None:
            details.append(f"terminal {position} rail {edge_id} has no canonical edge signature")
            resolved.append(EndpointResolution(
                status="unresolved",
                effective_concavity="UNKNOWN",
                method="none",
                detail=f"terminal rail {edge_id} has no canonical edge signature",
            ))
            continue
        item = resolve_endpoint(
            model,
            index,
            transition,
            classifier,
            {face_id},
            {sig},
            continuation_context={
                "course_direction": list(candidate.course_direction),
                "course_start": candidate.course_start,
                "course_end": candidate.course_end,
            },
        )
        resolved.append(item)
        if not item.resolved or item.effective_concavity != CONVEX:
            details.append(
                f"terminal {position} resolved as {item.effective_concavity}/{item.status}"
            )
    passed = len(resolved) == 2 and all(
        item.resolved and item.effective_concavity == CONVEX for item in resolved
    )
    return resolved, {
        "passed": passed,
        "detail": "; ".join(details),
        "method": "band_endpoints.resolve_endpoint_with_transition_evidence",
        "transitionBypass": "only transition regions positively identified by TransitionEvidence may be crossed",
    }


def _aggregate_family_evidence(
    model: PartModel,
    grouped: dict[str, list[dict]],
    candidate: OpenSweepCandidate,
) -> tuple[dict[str, Any], tuple[str, str] | None]:
    """Mean raw votes per merged face, then mean principal-face scores."""

    face_records: list[dict[str, Any]] = []
    per_face_slot: list[float] = []
    per_face_step: list[float] = []
    for face_id in candidate.principal_face_ids:
        votes = _votes_for_face(model, grouped, face_id)
        if not votes:
            face_records.append({"faceId": face_id, "voteCount": 0, "status": "missing"})
            return {"faceEvidence": face_records}, (
                "aagnet_family_evidence_missing",
                f"no per-face AAGNet softmax votes map to principal face {face_id}",
            )
        sums = {label: 0.0 for label in _FAMILY_LABELS}
        malformed: str | None = None
        for vote in votes:
            vector = vote.get("softmax")
            if not isinstance(vector, list) or len(vector) != len(FEATURE_NAMES):
                malformed = f"face {face_id} vote has no complete {len(FEATURE_NAMES)}-class softmax"
                break
            numeric_vector: list[float] = []
            for index, item in enumerate(vector):
                if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                    malformed = f"face {face_id} vote has non-finite class-{index} softmax"
                    break
                if float(item) < -_VECTOR_TOL or float(item) > 1.0 + _VECTOR_TOL:
                    malformed = f"face {face_id} vote has out-of-range class-{index} softmax"
                    break
                numeric_vector.append(max(0.0, min(1.0, float(item))))
            if malformed:
                break
            values: dict[str, float] = {}
            for label in _FAMILY_LABELS:
                index = _FAMILY_INDEXES[label]
                values[label] = numeric_vector[index]
            total = sum(numeric_vector)
            if not math.isfinite(total) or abs(total - 1.0) > _SOFTMAX_TOL:
                malformed = f"face {face_id} vote softmax does not sum to one"
                break
            for label, value in values.items():
                sums[label] += value
        if malformed:
            face_records.append({"faceId": face_id, "voteCount": len(votes), "status": "invalid", "detail": malformed})
            return {"faceEvidence": face_records}, ("aagnet_family_evidence_invalid", malformed)

        count = float(len(votes))
        family_scores = {label: sums[label] / count for label in _FAMILY_LABELS}
        slot_scores = {label: family_scores[label] for label in SLOT_FAMILIES}
        step_scores = {label: family_scores[label] for label in STEP_FAMILIES}
        # A family score is probability mass over the six explicitly allowed
        # labels.  Do not divide by six: the natural mass is what the
        # candidate-level Slot-vs-Step comparison is meant to resolve.
        slot_face_score = sum(slot_scores.values())
        step_face_score = sum(step_scores.values())
        per_face_slot.append(slot_face_score)
        per_face_step.append(step_face_score)
        face_records.append({
            "faceId": face_id,
            "voteCount": len(votes),
            "status": "ok",
            "slotFamilyScores": {key: round(value, 9) for key, value in slot_scores.items()},
            "stepFamilyScores": {key: round(value, 9) for key, value in step_scores.items()},
            "slotFaceAggregate": round(slot_face_score, 9),
            "stepFaceAggregate": round(step_face_score, 9),
            "cadFaceIds": [vote.get("cad_face_id") for vote in votes],
        })

    return {
        "faceEvidence": face_records,
        "slotScore": sum(per_face_slot) / len(per_face_slot),
        "stepScore": sum(per_face_step) / len(per_face_step),
    }, None


def _votes_for_face(model: PartModel, grouped: dict[str, list[dict]], face_id: str) -> list[dict]:
    votes = list(grouped.get(str(face_id), []))
    if votes:
        return votes
    face = model.faces.get(face_id)
    raw_ids = [face_id, *(getattr(face, "merged_from", []) if face else [])]
    indexes: set[int] = set()
    for raw in raw_ids:
        text = str(raw)
        if text.startswith("face_") and text[5:].isdigit():
            indexes.add(int(text[5:]))
    # Evidence from older adapters can omit the active PartModel mapping.  A
    # raw-face fallback is still equal-weighted and never uses a native group.
    for candidate_votes in grouped.values():
        for vote in candidate_votes:
            if vote.get("cad_face_id") in indexes:
                votes.append(vote)
    return votes


def simple_step_structure(model: PartModel, candidate: OpenSweepCandidate) -> dict[str, Any]:
    """Check exactly the ontology-compatible simple Step structure."""

    faces = list(candidate.ordered_profile_face_ids)
    if len(faces) != 2 or len(set(faces)) != 2:
        return {"passed": False, "detail": "simple Step requires exactly two distinct principal faces"}
    if any(model.faces.get(face_id) is None or model.faces[face_id].surface_type != SurfaceType.PLANE for face_id in faces):
        return {"passed": False, "detail": "simple Step requires two planar principal faces"}
    if len(candidate.internal_rail_edge_ids) != 1:
        return {"passed": False, "detail": "simple Step requires one shared internal rail"}
    edge = model.edges.get(candidate.internal_rail_edge_ids[0])
    if edge is None or edge.curve_type != CurveType.LINE:
        return {"passed": False, "detail": "the shared Step guide rail is not a straight edge"}

    normals = [_plane_normal(model.faces[face_id]) for face_id in faces]
    if any(item is None for item in normals):
        return {"passed": False, "detail": "one Step planar face has no usable normal"}
    alignment = abs(dot(normals[0], normals[1]))
    if alignment >= 1.0 - _PLANE_PARALLEL_TOL:
        return {"passed": False, "detail": "the two Step planes are coplanar/parallel, so the included angle is not below 180 degrees"}
    included = math.degrees(math.acos(max(-1.0, min(1.0, alignment))))
    return {
        "passed": True,
        "faceIds": faces,
        "sharedStraightEdgeId": candidate.internal_rail_edge_ids[0],
        # Finding C-07: this is the acute angle between the two plane normals,
        # not the included angle of the step - the attribute
        # ``step_included_angle`` is that, and is a different number.  Named
        # for what it measures so the two are not read as one.
        "planeNormalSeparationDeg": round(included, 6),
        "anglePolicy": "non-coplanar structural check only; no 90-degree or threshold classification",
    }


def prove_two_ended_exterior(
    model: PartModel,
    candidate: OpenSweepCandidate,
    classifier: SolidPointClassifier,
) -> dict[str, Any]:
    """Use solid classification to prove both canonical OS ends are exterior."""

    result: dict[str, Any] = {
        "passed": False,
        "method": "BRepClass3d_SolidClassifier_profile_interior_past_course_end",
        "classifierAvailable": classifier.available,
        "ends": [],
    }
    if not classifier.available:
        result.update({
            "reason": "slot_longitudinal_exterior_proof_unavailable",
            "detail": "no B-Rep solid classifier was available; Slot open state is not inferred from AAGNet through/blind evidence",
        })
        return result

    points_start = _end_profile_points(model, candidate, "start")
    points_end = _end_profile_points(model, candidate, "end")
    if not points_start or not points_end:
        result.update({
            "reason": "slot_longitudinal_profile_interior_unavailable",
            "detail": "transverse end edges did not provide a geometric sample point",
        })
        return result

    direction = coerced_unit_vector(candidate.course_direction)
    span = max(abs(candidate.course_end - candidate.course_start), 1.0)
    # Keep the probe local to the canonical end.  A closed cap must not be
    # skipped by a large ray, while an exterior opening is clear immediately.
    epsilons = [max(1.0e-6, span * factor) for factor in (1.0e-6, 1.0e-5, 1.0e-4)]
    for label, raw_points, level, sign in (
        ("start", points_start, candidate.course_start, -1.0),
        ("end", points_end, candidate.course_end, 1.0),
    ):
        center = _profile_interior_sample(raw_points, direction, level)
        states: list[dict[str, Any]] = []
        if center is None:
            result["ends"].append({"end": label, "passed": False, "reason": "profile_interior_sample_unavailable"})
            continue
        for epsilon in epsilons:
            point = [center[index] + sign * epsilon * direction[index] for index in range(3)]
            state = classifier.state(point, max(1.0e-7, epsilon * 0.25))
            states.append({"epsilon": round(epsilon, 9), "point": [round(value, 9) for value in point], "state": state})
        passed = bool(states) and all(item["state"] == "out" for item in states)
        result["ends"].append({
            "end": label,
            "passed": passed,
            "samplePolicy": "ordered_terminal_and_internal_rail_path_centroid_closed_by_mouth_chord",
            "courseLevel": round(level, 9),
            "profileInteriorPoint": [round(value, 9) for value in center],
            "states": states,
            "direction": [round(value, 9) for value in direction],
        })

    if all(item.get("passed", False) for item in result["ends"]) and len(result["ends"]) == 2:
        result["passed"] = True
        result["reason"] = "both_longitudinal_course_ends_classify_exterior"
    else:
        failed = [item.get("end") for item in result["ends"] if not item.get("passed", False)]
        result["reason"] = "slot_longitudinal_end_not_exterior"
        result["detail"] = "solid classification did not prove exterior at: " + ",".join(str(value) for value in failed)
    return result


def prove_transverse_enclosure(
    model: PartModel,
    candidate: OpenSweepCandidate,
    classifier: SolidPointClassifier,
) -> dict[str, Any]:
    """Use solid classification to prove the mouth is flanked by material.

    A slot is open *along* its course and closed *across* it.  The sibling
    proof above establishes the first half; without this one the second half
    is assumed, and a re-entrant corner of the part's own outline satisfies
    every remaining criterion: two faces sweeping a stable profile, both
    transverse terminals reading CONVEX, both course ends exterior.  What such
    a region is missing is a flank - step sideways off the mouth and there is
    no part there, because the "slot" is the outside of the part.

    So walk past each terminal rail, away from the mouth, and dip below the
    mouth by the same amount.  Inside that step a slot has the material its
    wall is cut into.  Sampling several stations along the rail rather than
    one point keeps a locally thin flank, or a rail that runs off the end of
    its neighbour, from deciding the whole feature.

    The step is a geometric probe, not a manufacturing limit: it is scaled by
    the candidate's own mouth and reach, so nothing here is a threshold and
    nothing belongs to the ontology.
    """

    result: dict[str, Any] = {
        "passed": False,
        "method": "BRepClass3d_SolidClassifier_material_beyond_each_terminal_rail",
        "classifierAvailable": classifier.available,
        "flanks": [],
    }
    if not classifier.available:
        result.update({
            "reason": "slot_transverse_enclosure_proof_unavailable",
            "detail": "no B-Rep solid classifier was available; Slot enclosure is not inferred from the terminal contexts alone",
        })
        return result

    rails = list(candidate.profile_terminal_rail_edge_ids)
    stations = [_rail_stations(model, edge_id) for edge_id in rails]
    if len(rails) != 2 or any(not item for item in stations):
        result.update({
            "reason": "slot_transverse_rail_samples_unavailable",
            "detail": "the two transverse terminal rails did not both provide geometric sample points",
        })
        return result

    course = coerced_unit_vector(candidate.course_direction)
    centres = [[sum(p[i] for p in item) / len(item) for i in range(3)] for item in stations]
    chord = _perpendicular(centres[1], centres[0], course)
    span = norm([centres[1][i] - centres[0][i] for i in range(3)])
    middle = [(centres[0][i] + centres[1][i]) / 2.0 for i in range(3)]
    interior = _profile_interior_sample(
        _end_profile_points(model, candidate, "start"), course, candidate.course_start
    )
    inward = _perpendicular(interior, middle, course) if interior is not None else None
    if chord is None or inward is None or span <= 0.0:
        result.update({
            "reason": "slot_transverse_frame_unavailable",
            "detail": "the mouth chord and the direction into the cavity were not both resolvable",
        })
        return result

    reach = abs(sum((interior[i] - middle[i]) * inward[i] for i in range(3)))
    scale = min(span, reach) if reach > 0.0 else span
    steps = [max(1.0e-6, scale * factor) for factor in (0.01, 0.05)]
    for label, points, sign in (("start", stations[0], -1.0), ("end", stations[1], 1.0)):
        states: list[dict[str, Any]] = []
        for point in points:
            for step in steps:
                probe = [point[i] + sign * step * chord[i] + step * inward[i] for i in range(3)]
                states.append({
                    "step": round(step, 9),
                    "point": [round(value, 9) for value in probe],
                    "state": classifier.state(probe, max(1.0e-7, step * 0.25)),
                })
        result["flanks"].append({
            "flank": label,
            "passed": bool(states) and all(item["state"] == "in" for item in states),
            "railEdgeId": rails[0] if label == "start" else rails[1],
            "samplePolicy": "three_stations_along_the_terminal_rail_stepped_outward_and_inward",
            "states": states,
        })

    if len(result["flanks"]) == 2 and all(item["passed"] for item in result["flanks"]):
        result["passed"] = True
        result["reason"] = "material_lies_beyond_both_transverse_terminal_rails"
        return result

    # Either way there is no node, but the two failures are not the same claim
    # and must not be filed under one sentence.  ``out`` is a finding: the
    # flank was sampled and there is nothing there.  ``on`` is a refusal: every
    # sample stayed on a surface, so the probe never got into a region to
    # report on, and saying "no material" would be asserting what was not
    # measured.
    failed = [item for item in result["flanks"] if not item["passed"]]
    outside = [item["flank"] for item in failed
               if any(state["state"] == "out" for state in item["states"])]
    if outside:
        result["reason"] = "slot_transverse_flank_not_enclosed"
        result["detail"] = (
            "solid classification found air beyond: " + ",".join(str(value) for value in outside)
            + "; the region is bounded by the part outline rather than cut into it"
        )
    else:
        result["reason"] = "slot_transverse_enclosure_not_proven"
        result["detail"] = (
            "no sample beyond " + ",".join(str(item["flank"]) for item in failed)
            + " reached material: " + ",".join(sorted({state["state"] for item in failed for state in item["states"]}))
            + "; the flank was not shown to be material and the candidate is not published as a Slot"
        )
    return result


def _rail_stations(model: PartModel, edge_id: str, count: int = 3) -> list[list[float]]:
    """A few points spread along one rail, not just its midpoint."""
    edge = model.edges.get(edge_id)
    if edge is None:
        return []
    points = [list(point) for point in (getattr(edge, "sample_points", None) or [])]
    if not points:
        points = [list(model.vertices[vertex_id].coordinates)
                  for vertex_id in getattr(edge, "vertex_ids", []) if vertex_id in model.vertices]
    if len(points) <= 1:
        return points
    last = len(points) - 1
    picked = [points[int(round(fraction * last))] for fraction in (0.25, 0.5, 0.75)][:count]
    return picked


def _perpendicular(head: list[float] | None, tail: list[float], axis: list[float] | None) -> list[float] | None:
    """Unit ``head - tail`` with any component along ``axis`` removed."""
    if head is None:
        return None
    delta = [head[index] - tail[index] for index in range(3)]
    if axis is not None:
        along = dot(delta, axis)
        delta = [delta[index] - along * axis[index] for index in range(3)]
    return coerced_unit_vector(delta)


def _end_profile_points(model: PartModel, candidate: OpenSweepCandidate, end: str) -> list[list[float]]:
    records = (
        candidate.longitudinal_start_edge_ids_by_face
        if end == "start"
        else candidate.longitudinal_end_edge_ids_by_face
    )
    seen: set[str] = set()
    points: list[list[float]] = []
    for record in records:
        for edge_id in record.get("edgeIds", []):
            if edge_id in seen:
                continue
            seen.add(edge_id)
            edge = model.edges.get(edge_id)
            if edge is None:
                continue
            samples = [list(point) for point in (getattr(edge, "sample_points", None) or [])]
            if samples:
                points.extend(samples)
                continue
            points.extend(
                [list(model.vertices[vertex_id].coordinates) for vertex_id in getattr(edge, "vertex_ids", []) if vertex_id in model.vertices]
            )
    # Prefer the ordered transverse profile path recovered from the terminal
    # rails and internal longitudinal rails.  Its centroid, when closed by
    # the mouth chord, is a cavity sample rather than an arbitrary average of
    # tessellation points.  The raw end-boundary samples remain a fallback for
    # singleton cylindrical profiles, which have no internal rails.
    direction = coerced_unit_vector(candidate.course_direction)
    if direction is not None:
        level = candidate.course_start if end == "start" else candidate.course_end
        rails = [*candidate.profile_terminal_rail_edge_ids, *candidate.internal_rail_edge_ids]
        if len(candidate.profile_terminal_rail_edge_ids) == 2:
            rails = [
                candidate.profile_terminal_rail_edge_ids[0],
                *candidate.internal_rail_edge_ids,
                candidate.profile_terminal_rail_edge_ids[1],
            ]
        path = []
        for edge_id in rails:
            point = _edge_extreme_point_along(model, edge_id, direction, end)
            if point is not None:
                path.append(point)
        if len(path) >= 3:
            return path
    return points


def _edge_extreme_point_along(model, edge_id, direction, end):
    edge = model.edges.get(edge_id)
    if edge is None:
        return None
    points = [list(point) for point in (getattr(edge, "sample_points", None) or [])]
    if not points:
        points = [
            list(model.vertices[vertex_id].coordinates)
            for vertex_id in getattr(edge, "vertex_ids", ())
            if vertex_id in model.vertices
        ]
    if not points:
        return None
    key = lambda point: dot(point, direction)
    return list((min if end == "start" else max)(points, key=key))


def _profile_interior_sample(points: list[list[float]], direction: list[float], level: float) -> list[float] | None:
    if not points:
        return None
    projected: list[list[float]] = []
    for point in points:
        correction = dot(point, direction) - level
        projected.append([point[index] - correction * direction[index] for index in range(3)])
    center = [sum(point[index] for point in projected) / len(projected) for index in range(3)]
    if not all(math.isfinite(value) for value in center):
        return None
    return center


def _plane_normal(face) -> list[float] | None:
    if face is None:
        return None
    value = face.parameters.get("normal")
    unit = coerced_unit_vector(value)
    if unit is None:
        return None
    if face.is_reversed:
        unit = [-value for value in unit]
    return unit
