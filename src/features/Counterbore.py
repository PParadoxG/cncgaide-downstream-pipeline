"""Counterbore compound-hole resolver and ontology attribute extraction."""

from __future__ import annotations

from typing import Any

from contract.partmodel import PartModel

from core.models import FeatureNode
from features._shared.compound_resolution import CompoundResolution, resolve_counterbore


def recognize_and_build(
    model: PartModel,
    evidence_document: dict | None,
    claimed_face_ids: set[str] | None = None,
    *,
    component_nodes: list[FeatureNode] | None = None,
    start_index: int = 1,
    kernel=None,
    ladder=None,
    intervals=None,
    **_: Any,
) -> tuple[list[FeatureNode], set[str], dict[str, Any]]:
    """Resolve only from completed ThroughHole/BlindHole component nodes."""

    resolution = resolve_counterbore(
        model,
        evidence_document,
        list(component_nodes or []),
        claimed_face_ids,
        start_index=start_index,
        kernel=kernel,
        ladder=ladder,
        intervals=intervals,
    )
    return resolution.nodes, resolution.claimed, resolution.diagnostics
