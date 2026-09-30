"""Relation rule interface and the read-only context handed to every rule."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from contract.partmodel import PartModel

from core.models import FeatureEdge, FeatureNode
from topology.signatures import SignatureView


@dataclass(frozen=True)
class RelationContext:
    """Everything a relation rule is allowed to see.

    ``nodes`` is the *final* node list: compound replacements have already
    consumed their component nodes, so a rule can never reference a node id
    that no longer exists in ``result.json``.  ``faceOwners`` is derived from
    exactly that list, which is why an edge can only be anchored on a face
    owned by a surviving node.
    """

    model: PartModel
    nodes: tuple[FeatureNode, ...]
    node_by_id: dict[str, FeatureNode] = field(default_factory=dict)
    face_owners: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # The geometry kernel is already built by the orchestrator for feature
    # extraction.  Relation rules may use it for a relation-specific exact
    # measurement (for example Pocket side-wall clearance); rules that only
    # need PartModel topology remain usable in unit tests without it.
    kernel: object | None = None
    # Mutable by design: rules record conservative no-edge diagnostics here
    # without changing the small ``build_edges`` return contract.
    rule_diagnostics: dict[str, dict] = field(default_factory=dict)
    # Memo for the expensive terminal/boundary resolutions.  Every pairwise
    # rule would otherwise re-walk a node's rims once per candidate partner,
    # which is quadratic in the node count for no new information.
    cache: dict[str, dict] = field(default_factory=dict)
    # Pre-resolved terminals, injected by a caller that already knows them.
    # This is the *only* way to supply them: reading such a hook out of a
    # node's diagnostics would let any upstream recognizer assert a
    # LOCATED_ON host without going through the proof, host resolution and
    # ambiguity rejection this layer exists to perform.
    terminal_overrides: dict[str, list[dict]] = field(default_factory=dict)
    # One geometric-edge-identity view for this model, filled in below.  Every
    # rim walk a rule performs needs one; before node 28 each call built its
    # own, so a part with n hole terminals signed its edges n times over.  It
    # is a memo of a pure function of ``model``, like ``cache`` above.
    signatures: SignatureView | None = None

    def __post_init__(self) -> None:
        if self.signatures is None:
            object.__setattr__(self, "signatures", SignatureView(self.model))

    @classmethod
    def build(
        cls,
        model: PartModel,
        nodes: list[FeatureNode],
        *,
        kernel=None,
        terminal_overrides: dict[str, list[dict]] | None = None,
    ) -> "RelationContext":
        node_by_id = {node.nodeId: node for node in nodes}
        owners: dict[str, list[str]] = {}
        for node in nodes:
            for face_id in node.partmodelFaceIds:
                bucket = owners.setdefault(str(face_id), [])
                if node.nodeId not in bucket:
                    bucket.append(node.nodeId)
        return cls(
            model=model,
            nodes=tuple(nodes),
            node_by_id=node_by_id,
            face_owners={face_id: tuple(value) for face_id, value in owners.items()},
            kernel=kernel,
            terminal_overrides=dict(terminal_overrides or {}),
        )


class RelationRule(ABC):
    """One relation type.  Pure function of ``RelationContext``."""

    rule_id: str = "unnamed_relation"
    edge_type: str = "UNKNOWN"
    priority: int = 1000
    enabled_by_default: bool = True
    directed: bool = False
    description: str = ""

    @abstractmethod
    def build_edges(self, context: RelationContext) -> list[FeatureEdge]:
        """Return the edges this rule proves for the given final node set."""

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<{type(self).__name__} {self.rule_id} -> {self.edge_type}>"
