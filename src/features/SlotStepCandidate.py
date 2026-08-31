"""Temporary viewer/audit feature for shared Open Sweep candidates.

This is deliberately not an ontology feature and contains no Slot-vs-Step
semantic rule.  It exists only until the next semantic resolver node is
implemented.  Ownership is restricted to OS principal sweep faces; terminal
supports, longitudinal end supports, and all other adjacent context remain
unclaimed.  A narrow pre-registration gate admits only candidates whose two
terminal material-side contexts are both explicitly recess-like (all direct
connections at each terminal are ``CONVEX``); OS itself retains every
structurally valid sweep for audit.
"""

from __future__ import annotations

from typing import Any

from contract.partmodel import PartModel

from core.models import FeatureNode
from features._shared.attributes import serialize_cad_face_ids
from features._shared.open_sweep import (
    OpenSweepAnalysis,
    OpenSweepCandidate,
    TERMINAL_RECESS_LIKE,
    candidate_diagnostics,
)


def recognize_and_build(
    model: PartModel,
    evidence_document: dict | None,
    claimed_face_ids: set[str] | None = None,
    *,
    sweeps: OpenSweepAnalysis,
    start_index: int = 1,
    **_: Any,
) -> tuple[list[FeatureNode], set[str], dict[str, Any]]:
    """Register each still-unclaimed OS candidate for visual audit."""
    claimed = claimed_face_ids if claimed_face_ids is not None else set()
    nodes: list[FeatureNode] = []
    rejected: list[dict[str, Any]] = []
    for candidate in sweeps.candidates:
        if not _has_two_recess_like_terminals(candidate):
            rejected.append({
                "candidateKey": candidate.candidate_key,
                "reason": "profile_terminals_are_not_both_convex_to_external_support",
                "terminalEffectiveContexts": list(candidate.profile_terminal_effective_contexts),
                "terminalMaterialConnections": list(candidate.profile_terminal_material_connections),
            })
            continue
        principal = list(candidate.principal_face_ids)
        overlap = sorted(set(principal) & claimed)
        if overlap:
            rejected.append({
                "candidateKey": candidate.candidate_key,
                "reason": "principal_faces_claimed_after_open_sweep_analysis",
                "faceIds": overlap,
            })
            continue
        cad_ids = serialize_cad_face_ids(model, principal)
        node = FeatureNode(
            nodeId=f"SlotStepCandidate_{start_index + len(nodes):03d}",
            featureType="SlotStepCandidate",
            partmodelFaceIds=principal,
            attributes={"cadFaceIds": cad_ids},
            missingAttributes={
                "cadFaceIds": "no source CAD face ids were recoverable from the OS principal faces"
            } if cad_ids is None else {},
            diagnostics={
                "role": "temporary_open_sweep_viewer_audit_feature_not_ontology_semantics",
                "temporary": True,
                "slotStepSemanticResolution": "deferred_to_next_node",
                "ownershipPolicy": "principalSweepGeometry only; support and longitudinal-end context remain unclaimed",
                "openSweep": candidate_diagnostics(candidate),
            },
        )
        nodes.append(node)
        claimed.update(principal)
    diagnostics = {
        "featureModule": "SlotStepCandidate",
        "temporary": True,
        "ontologyFeature": False,
        "method": "register_open_sweep_candidates_for_viewer_audit_v1",
        "aagnetUsed": False,
        "candidateCount": len(sweeps.candidates),
        "acceptedCount": len(nodes),
        "claimedFaceCount": sum(len(node.partmodelFaceIds) for node in nodes),
        "ownershipPolicy": "OS principal faces only",
        "recessLikeGate": (
            "register only when both profile-terminal material-side contexts "
            "are explicitly RECESS_LIKE (all direct support connections CONVEX)"
        ),
        "rejectedCandidates": rejected,
    }
    return nodes, claimed, diagnostics


def _has_two_recess_like_terminals(candidate: OpenSweepCandidate) -> bool:
    """True only for the unambiguous recessed terminal sign pattern.

    A recess-like open sweep reaches its external supports through material-
    side ``CONVEX`` terminal rails.  Each terminal effective context is
    ``RECESS_LIKE`` only when all of that terminal's direct external-support
    connections are ``CONVEX``.  Any concave, smooth, mixed, ambiguous, or
    unresolved terminal remains in OS diagnostics but is not registered as a
    temporary SlotStepCandidate.
    """
    return (
        len(candidate.profile_terminal_effective_contexts) == 2
        and all(
            context == TERMINAL_RECESS_LIKE
            for context in candidate.profile_terminal_effective_contexts
        )
    )
