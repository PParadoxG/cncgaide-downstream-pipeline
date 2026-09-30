"""Reading points off an edge.

One function, because four modules had grown their own and they disagreed only
about which numbers count as a point (Node 26 finding C-12).
"""

from __future__ import annotations

from contract.partmodel import PartModel

from topology.vectors import valid_vector


def edge_points(model: PartModel, edge) -> list[list[float]]:
    """The edge's sampled points, or its endpoint vertices as the fallback.

    An empty list means the edge has neither, which is a real answer: an edge
    whose vertices the model does not carry cannot be walked.
    """
    points = [list(point) for point in (getattr(edge, "sample_points", None) or [])
              if valid_vector(point)]
    if points:
        return points
    return [
        list(model.vertices[vertex_id].coordinates)
        for vertex_id in (getattr(edge, "vertex_ids", None) or [])
        if vertex_id in model.vertices
    ]
