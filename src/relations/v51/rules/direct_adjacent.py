"""DIRECT_ADJACENT — proven B-Rep face adjacency between two feature nodes.

Definition (strict, first edition)
---------------------------------
An undirected ``DIRECT_ADJACENT`` edge exists between final feature nodes A
and B if and only if some face owned by A and some face owned by B are
adjacent in the merged ``PartModel`` face adjacency graph, i.e. the two faces
share a real B-Rep edge.

What this rule deliberately does **not** do:

* no bounding box, centroid distance or any spatial proximity test;
* no neighbourhood threshold, angular tolerance or coaxiality test;
* no transitive derivation across a Chamfer/Fillet or any other transition
  feature — a wall separated from a floor by a fillet is adjacent to the
  fillet, not to the floor.

The only input is ``FaceData.adjacencies``, which the parser derives from
``TopExp.MapShapesAndAncestors`` edge/face incidence and which the normalizer
rebuilds onto the surviving face after every seam merge.  That makes the
relation exactly as objective as the shared-edge topology itself.

Aggregation: several face pairs may connect the same two features.  They
produce one feature-level edge whose evidence lists every contributing face
pair and shared edge.
"""

from __future__ import annotations

from collections import Counter

from core.models import FeatureEdge
from relations.base import RelationContext, RelationRule
from relations.v51.registry import register_rule


@register_rule
class DirectAdjacent(RelationRule):
    rule_id = "direct_adjacent_v1"
    edge_type = "DIRECT_ADJACENT"
    priority = 10
    enabled_by_default = True
    directed = False
    description = (
        "Undirected edge between two final feature nodes whose owned PartModel "
        "faces share at least one real B-Rep edge."
    )

    def build_edges(self, context: RelationContext) -> list[FeatureEdge]:
        if len(context.nodes) < 2:
            return []

        owners = context.face_owners
        # Keyed by the canonical (nodeA, nodeB) pair with nodeA < nodeB so the
        # two reciprocal halves of one shared edge land in the same bucket.
        buckets: dict[tuple[str, str], dict[tuple, dict]] = {}

        for face_id, face in context.model.faces.items():
            source_nodes = owners.get(str(face_id))
            if not source_nodes:
                continue
            for adjacency in getattr(face, "adjacencies", []):
                target_face_id = str(adjacency.adjacent_face_id)
                target_nodes = owners.get(target_face_id)
                if not target_nodes:
                    continue
                for source_node in source_nodes:
                    for target_node in target_nodes:
                        # Adjacency inside one feature is real topology but not
                        # a relation: a feature is not adjacent to itself.
                        if source_node == target_node:
                            continue
                        pair = _sorted_pair(source_node, target_node)
                        record = _face_pair_record(
                            context,
                            pair,
                            source_node,
                            str(face_id),
                            target_face_id,
                            adjacency,
                        )
                        buckets.setdefault(pair, {}).setdefault(_record_key(record), record)

        edges: list[FeatureEdge] = []
        for (source_node, target_node), records in buckets.items():
            pairs = sorted(records.values(), key=_record_key)
            edges.append(
                FeatureEdge(
                    edgeId="",
                    source=source_node,
                    target=target_node,
                    edgeType=self.edge_type,
                    directed=False,
                    sourceRule=self.rule_id,
                    evidence=_evidence(context, source_node, target_node, pairs),
                    diagnostics={
                        "adjacencySource": "PartModel.FaceData.adjacencies (shared B-Rep edge)",
                        "spatialTestsUsed": [],
                    },
                )
            )
        return edges


def _sorted_pair(node_a: str, node_b: str) -> tuple[str, str]:
    return (node_a, node_b) if node_a <= node_b else (node_b, node_a)


def _face_pair_record(
    context: RelationContext,
    pair: tuple[str, str],
    seen_from_node: str,
    seen_from_face: str,
    other_face: str,
    adjacency,
) -> dict:
    """One shared-edge record, oriented to the canonical (source, target) pair."""

    if seen_from_node == pair[0]:
        source_face, target_face = seen_from_face, other_face
    else:
        source_face, target_face = other_face, seen_from_face
    connection = getattr(adjacency, "connection_type", None)
    return {
        "sourcePartmodelFaceId": source_face,
        "targetPartmodelFaceId": target_face,
        "sharedEdgeId": str(getattr(adjacency, "shared_edge_id", "")) or None,
        "connectionType": getattr(connection, "value", connection),
        "sourceSurfaceType": _surface_type(context, source_face),
        "targetSurfaceType": _surface_type(context, target_face),
    }


def _record_key(record: dict) -> tuple:
    return (
        str(record.get("sourcePartmodelFaceId")),
        str(record.get("targetPartmodelFaceId")),
        str(record.get("sharedEdgeId")),
    )


def _surface_type(context: RelationContext, face_id: str) -> str | None:
    face = context.model.faces.get(face_id)
    surface = getattr(face, "surface_type", None)
    return getattr(surface, "value", surface)


def _evidence(context: RelationContext, source_node: str, target_node: str, pairs: list[dict]) -> dict:
    source_faces = sorted({pair["sourcePartmodelFaceId"] for pair in pairs})
    target_faces = sorted({pair["targetPartmodelFaceId"] for pair in pairs})
    shared_edges = sorted({pair["sharedEdgeId"] for pair in pairs if pair["sharedEdgeId"]})
    return {
        "relationBasis": "sharedBRepEdge",
        "adjacentFacePairs": pairs,
        "facePairCount": len(pairs),
        "sharedEdgeIds": shared_edges,
        "sharedEdgeCount": len(shared_edges),
        "connectionTypes": dict(sorted(Counter(pair["connectionType"] or "UNKNOWN" for pair in pairs).items())),
        "sourceFeatureType": _feature_type(context, source_node),
        "targetFeatureType": _feature_type(context, target_node),
        "sourcePartmodelFaceIds": source_faces,
        "targetPartmodelFaceIds": target_faces,
    }


def _feature_type(context: RelationContext, node_id: str) -> str | None:
    node = context.node_by_id.get(node_id)
    return getattr(node, "featureType", None)
