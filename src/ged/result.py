"""Public GED result contract (graph-agnostic)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class GedBreakdown:
    node_cost: float
    fixed_cost: float
    edge_cost: float
    scale_cost: float
    n_sub: float
    n_type_change: float
    n_del: float
    n_ins: float
    edges_preserved: float
    edges_changed: float
    edges_del: float
    edges_ins: float


@dataclass(frozen=True)
class GedResult:
    graph_id_a: str
    graph_id_b: str
    raw_cost: float
    upper_bound: float
    normalized_distance: float
    similarity_percent: float
    breakdown: GedBreakdown
    edit_path: Mapping[str, Any]
    config_name: str

    def to_dict(self, *, include_edit_path: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "graph_id_a": self.graph_id_a,
            "graph_id_b": self.graph_id_b,
            "raw_cost": self.raw_cost,
            "upper_bound": self.upper_bound,
            "normalized_distance": self.normalized_distance,
            "similarity_percent": self.similarity_percent,
            "breakdown": asdict(self.breakdown),
            "config": self.config_name,
        }
        if include_edit_path:
            payload["edit_path"] = dict(self.edit_path)
        return payload
