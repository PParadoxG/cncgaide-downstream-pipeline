"""Canonical v8 feature result models.

Feature attributes deliberately live on ``FeatureNode``.  There is no
separate recognition result / geometry-extractor handoff in this pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FeatureNode:
    """One ontology-named feature with direct, ontology-named attributes."""

    nodeId: str
    featureType: str
    partmodelFaceIds: list[str]
    missingAttributes: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    # Quantities the pipeline measured *about* this feature, kept apart from
    # the recogniser's own attributes (node 19' §6: a derived value must never
    # be published where a value the recogniser stated would be).  One entry
    # per quantity: ``{"value": ..., "derivedBy": "nx_kernel", ...}``, with a
    # ``reason`` instead of a value when the measurement was refused.
    derived: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Flatten ontology attributes onto the node's top level."""
        result = {
            "nodeId": self.nodeId,
            "featureType": self.featureType,
            **self.attributes,
            "missingAttributes": self.missingAttributes,
            "diagnostics": self.diagnostics,
        }
        # Written only when something was derived: a line that derives nothing
        # (the open-source line keeps these on the node top level as ontology
        # attributes) publishes byte-identical nodes.
        if self.derived:
            result["derived"] = self.derived
        # Keep the parser-facing IDs alongside ontology-facing cadFaceIds for
        # face-map, graph, and OCC viewer traceability.
        result["partmodelFaceIds"] = list(self.partmodelFaceIds)
        return result

    # Keys ``to_dict`` writes that are *not* ontology attributes.  Everything
    # else on a serialized node is an attribute, which is what lets a stored
    # ``result.json`` be turned back into the exact node set the heuristic
    # layer saw, without a list of attribute names anywhere.
    RESERVED_KEYS = frozenset({"nodeId", "featureType", "partmodelFaceIds", "missingAttributes", "diagnostics", "derived"})

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureNode":
        """Inverse of :meth:`to_dict` (field for field)."""
        return cls(
            nodeId=payload["nodeId"],
            featureType=payload["featureType"],
            partmodelFaceIds=list(payload.get("partmodelFaceIds") or []),
            missingAttributes=dict(payload.get("missingAttributes") or {}),
            diagnostics=dict(payload.get("diagnostics") or {}),
            attributes={key: value for key, value in payload.items() if key not in cls.RESERVED_KEYS},
            derived=dict(payload.get("derived") or {}),
        )


@dataclass
class FeatureEdge:
    """Unified relation edge between two final feature nodes.

    ``source``/``target`` always name node ids that survive into
    ``result.json``.  ``properties`` carries the relation's own ontology-named
    quantities (``crossingAngle``, ``axisSpacing``, ``pocketClearance``,
    ``slotClearance``, the LOCATED_ON terminal/host fields) — the frozen Node
    2 contract calls these relation-valued properties, so like a node's
    attributes they belong on the edge itself rather than buried in the proof.
    ``evidence`` carries the witnesses a reader needs to audit the edge (for
    DIRECT_ADJACENT: the contributing face pairs and their shared B-Rep
    edges); ``diagnostics`` records how the rule reached it.
    """

    edgeId: str
    source: str
    target: str
    edgeType: str
    directed: bool = False
    sourceRule: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureEdge":
        """Inverse of :meth:`to_dict` (field for field)."""
        return cls(
            edgeId=payload["edgeId"],
            source=payload["source"],
            target=payload["target"],
            edgeType=payload["edgeType"],
            directed=bool(payload.get("directed", False)),
            sourceRule=payload.get("sourceRule") or "",
            properties=dict(payload.get("properties") or {}),
            evidence=dict(payload.get("evidence") or {}),
            diagnostics=dict(payload.get("diagnostics") or {}),
        )
