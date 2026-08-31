"""StepBore recognition and ontology attribute extraction.

A StepBore is a non-terminal cylindrical section of a multi-step coaxial hole: its major end
opens outward or into the wider section before it, and its minor end is closed by an annular
shoulder that caps only the ring, the centre continuing into the next, narrower section.

Recognition is a re-typing, not a new face search.  The low-level hole layer already carves a
ladder correctly - each interior cylinder comes out as a BlindHole owning that cylinder plus the
annular shoulder that caps it, because the shoulder reads as its "bottom".  What was missing was
a class saying that this is a *section of a bore*, not a bore in its own right.  So this module
takes the run the shared ladder analysis proved, keeps the face ownership the low-level nodes
already established, and republishes the non-terminal sections under the right leaf class with
the right attribute set.  The terminal section keeps its own ThroughHole / BlindHole node: it
really is where the bore ends.

The attribute set is BlindHole's minus ``hole_bottom_type`` and ``derived_floor_thickness``.
An annular shoulder is not a bottom - material closes the ring only - so neither a bottom
classification nor a floor thickness has a defined value here, and per the project's rule that
every attribute in a class's set must be defined for every instance of it, they are absent
rather than null.
"""

from __future__ import annotations

from typing import Any

from contract.partmodel import PartModel

from core.models import FeatureNode
from features._shared.attributes import canonical_axis, rounded_scalar, serialize_cad_face_ids
from features._shared.compound_resolution import _axis_point_at_coordinate, _compound_wall_thickness, _ratio
from features._shared.hole_ladder import HoleRun, LadderAnalysis, discover_hole_ladders
from features._shared.hole_topology import valid_vector


_FIELDS = (
    "cadFaceIds", "posX", "posY", "posZ", "feature_axis", "machining_direction",
    "depth", "diameter", "derived_min_wall_thickness", "derived_ratio_depth_to_diameter",
)

_REASONS = {
    "cadFaceIds": "no source CAD face ids were recoverable from the section faces",
    "posX": "the annular shoulder plane centre was unavailable",
    "posY": "the annular shoulder plane centre was unavailable",
    "posZ": "the annular shoulder plane centre was unavailable",
    "feature_axis": "the section axis could not be oriented",
    "machining_direction": "the entry-to-shoulder axis could not be oriented",
    "depth": "the section wall has no positive axial extent",
    "diameter": "the section wall has no positive radius",
    "derived_min_wall_thickness": "no non-self material surface was reached from sampled section wall normals",
    "derived_ratio_depth_to_diameter": "positive section depth and diameter were not both available",
}

# The low-level leaves a step-bore section can be re-typed from.  A section that some other
# feature already owns is left alone; ownership conflicts are the resolver's business, not this
# module's, and they are reported rather than overridden.
_COMPONENT_TYPES = {"ThroughHole", "BlindHole"}


def recognize_and_build(
    model: PartModel,
    evidence_document: dict | None = None,
    claimed_face_ids: set[str] | None = None,
    *,
    component_nodes: list[FeatureNode] | None = None,
    ladder: LadderAnalysis | None = None,
    start_index: int = 1,
    kernel=None,
    **_: Any,
) -> tuple[list[FeatureNode], set[str], dict[str, Any]]:
    """Re-type the non-terminal sections of every proven run of three or more."""

    claimed = set(claimed_face_ids or ())
    ladder = ladder if ladder is not None else discover_hole_ladders(model)
    components = list(component_nodes or [])
    owner_of: dict[str, FeatureNode] = {}
    for node in components:
        for face_id in node.partmodelFaceIds:
            owner_of.setdefault(str(face_id), node)

    nodes: list[FeatureNode] = []
    consumed: list[str] = []
    withheld: list[dict[str, Any]] = []
    for run in ladder.runs_of_size_at_least(3):
        for index, segment in enumerate(run.segments[:-1]):
            owner = owner_of.get(segment.faceId)
            if owner is not None and owner.featureType not in _COMPONENT_TYPES:
                withheld.append({
                    "faceId": segment.faceId,
                    "reason": "section_face_is_already_owned_by_a_non_hole_feature",
                    "conflictingNodeId": owner.nodeId,
                    "conflictingFeatureType": owner.featureType,
                })
                continue
            face_ids = _section_faces(segment, owner)
            node = _build_node(model, run, index, segment, face_ids,
                               owner.nodeId if owner else None,
                               start_index + len(nodes), kernel)
            nodes.append(node)
            claimed.update(face_ids)
            if owner is not None:
                consumed.append(owner.nodeId)

    diagnostics = {
        "featureModule": "StepBore",
        "method": "coaxial_run_non_terminal_section_retyping_v1",
        "candidateRunCount": len(ladder.runs_of_size_at_least(3)),
        "acceptedCount": len(nodes),
        "consumedNodeIds": sorted(set(consumed)),
        "withheldSections": withheld,
        "checks": [
            "proven coaxial run of three or more cylinders",
            "single entry at the largest-diameter end",
            "monotonically non-increasing diameters from the entry",
            "real internal annular shoulder capping the minor end",
            "section is not the terminal member of the run",
        ],
    }
    return nodes, claimed, diagnostics


def _section_faces(segment, owner: FeatureNode | None) -> list[str]:
    """The cylinder, its shoulder, and whatever else the consumed low-level node owned.

    Taking the component's faces over rather than recomputing them is deliberate: a section
    replaces its low-level node entirely, so a face the node owned but this module did not think
    to list would stay claimed while belonging to nobody, and the residual passes would never
    see it.
    """
    face_ids = [segment.faceId]
    if segment.shoulderFaceId:
        face_ids.append(segment.shoulderFaceId)
    if owner is not None:
        face_ids.extend(str(face_id) for face_id in owner.partmodelFaceIds)
    return list(dict.fromkeys(face_ids))


def _build_node(model: PartModel, run: HoleRun, index: int, segment, face_ids: list[str],
                component_node_id: str | None, node_index: int, kernel) -> FeatureNode:
    attributes: dict[str, Any] = {name: None for name in _FIELDS}
    attributes["cadFaceIds"] = serialize_cad_face_ids(model, face_ids)
    attributes["feature_axis"] = canonical_axis(run.axis)
    attributes["machining_direction"] = [round(float(value), 6) + 0.0 for value in run.machiningDirection]
    attributes["depth"] = rounded_scalar(segment.length)
    attributes["diameter"] = rounded_scalar(segment.diameter)
    attributes["derived_ratio_depth_to_diameter"] = _ratio(attributes["depth"], attributes["diameter"])

    # pos is the centre of the section's minor rim - where a planar shoulder or a conical
    # transition closes it - which is the root of this section, the same convention BlindHole
    # uses when it takes the centre of its bottom.
    position = (_axis_point_at_coordinate(run.origin, run.axis, segment.minorCoordinate)
                if segment.minorCoordinate is not None else None)
    for axis_index, coordinate in enumerate(("posX", "posY", "posZ")):
        attributes[coordinate] = rounded_scalar(position[axis_index]) if valid_vector(position) else None

    wall_diagnostics: dict[str, Any] = {}
    attributes["derived_min_wall_thickness"] = _compound_wall_thickness(
        model, kernel, face_ids, [segment.faceId], diagnostics=wall_diagnostics,
    )
    return FeatureNode(
        nodeId=f"StepBore_{node_index:03d}",
        featureType="StepBore",
        partmodelFaceIds=list(face_ids),
        attributes=attributes,
        missingAttributes={name: _REASONS[name] for name in _FIELDS if attributes.get(name) is None},
        diagnostics={
            "method": "coaxial_run_non_terminal_section_retyping_v1",
            "sectionFaceId": segment.faceId,
            "transitionFaceId": segment.transitionFaceId,
            "transitionKind": segment.transitionKind,
            "transitionProof": segment.transitionProof,
            "shoulderFaceId": segment.shoulderFaceId,
            "componentNodeId": component_node_id,
            "sectionIndex": index,
            "runSize": run.size,
            "runFaceIds": [item.faceId for item in run.segments],
            "runDiameters": [round(item.diameter, 6) for item in run.segments],
            "entryCoordinate": round(run.entryCoordinate, 6),
            "interval": [round(segment.interval[0], 6), round(segment.interval[1], 6)],
            "isTerminalSection": False,
            "wallThickness": wall_diagnostics,
        },
    )
