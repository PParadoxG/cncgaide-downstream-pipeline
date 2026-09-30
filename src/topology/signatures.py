"""Geometric edge identity: one rule, and the per-model views built on it.

STEP exporters emit the same physical curve as several *duplicate edge
entities* - each face loop references its own copy - and seam merging splits a
closed curve into halves.  The parser keeps one id per OCCT entity, so those
copies are not the same edge id.  A *signature* is the deterministic answer to
"is this the same curve": curve type, length, and the rounded point cloud.  No
fuzzy matching, no tolerance beyond the rounding.

There used to be two of these functions, in ``hole_topology`` and in
``closed_wall_band``, and before Node 27 they disagreed about edges that carry
no ``sample_points`` (Node 26 finding C-01).  They agree now, so there is one.
"""

from __future__ import annotations

from contract.partmodel import PartModel

#: Decimals kept in a signature.  Coarser than the parser's precision on
#: purpose: two copies of one curve differ in the last bits.
SIGNATURE_ROUND = 6

#: An edge shorter than this is a degenerate apex artefact (the zero-length
#: circle a cone stores at its tip), not a boundary anyone can walk.
DEGENERATE_LENGTH_TOL = 1e-9


def edge_signature(edge, model: PartModel | None = None) -> tuple:
    """Curve type + length + point cloud for one edge.

    The point cloud is the parser's persisted ``sample_points`` when present.
    Without them, and given a *model*, the endpoint vertices stand in - which
    matters, because otherwise every such edge of one curve type and length
    collapses into a single signature class.  The parameter stays optional so a
    bare edge can still be signed.
    """
    samples = [list(point) for point in getattr(edge, "sample_points", None) or []]
    if not samples and model is not None:
        samples = [
            list(model.vertices[vertex_id].coordinates)
            for vertex_id in getattr(edge, "vertex_ids", None) or []
            if vertex_id in model.vertices
        ]
    return signature_from_points(edge, samples)


def signature_from_points(edge, samples: list[list[float]]) -> tuple:
    rounded = tuple(sorted(tuple(round(float(value), SIGNATURE_ROUND) for value in point) for point in samples)) if samples else ()
    curve_type = edge.curve_type.value if hasattr(edge.curve_type, "value") else str(edge.curve_type)
    return (curve_type, round(float(edge.length), SIGNATURE_ROUND), rounded)


def carries_length(model: PartModel, edge_id: str) -> bool:
    """Is this a real edge of the model, or a zero-length apex artefact?

    One rule for every place that turns loop edges into signatures - the cap
    test had it, the rim set and the incidence index did not (finding C-01).
    """
    edge = model.edges.get(edge_id)
    return edge is not None and edge.length > DEGENERATE_LENGTH_TOL


class SignatureView:
    """One model's edge signatures, and the incidence index built from them.

    There used to be two carriers for this: a lazy per-edge lookup for callers
    that held no table, and an eagerly built dictionary plus a separate
    incidence dictionary for callers that did, threaded through as three
    optional parameters (Node 27 CP-1 / CP-3, Node 27C leftover #3).  Optional
    meant every entry point had to decide again whether to pay, and two of them
    decided differently.  One object, asked for once per model, required
    wherever the walk needs it.

    Signing is lazy and memoised: the rim walk asks for the same edge's
    signature once per face loop it appears in, per rim, per wall - hundreds of
    thousands of times on one part - and pays for each edge once.  The view is
    deliberately *not* a module-level cache keyed on edge id: ``edge_0`` names
    a different edge in every model, and several models are alive in one
    process (batch refresh, replay tests), so a global cache would be a
    correctness bug rather than an optimisation.
    """

    __slots__ = ("_model", "_cache", "_incidence")

    def __init__(self, model: PartModel) -> None:
        self._model = model
        self._cache: dict[str, tuple] = {}
        self._incidence: dict[tuple, set[str]] | None = None

    @property
    def model(self) -> PartModel:
        return self._model

    def __contains__(self, edge_id: str) -> bool:
        return edge_id in self._model.edges

    def __getitem__(self, edge_id: str) -> tuple:
        signature = self._cache.get(edge_id)
        if signature is None:
            signature = edge_signature(self._model.edges[edge_id], self._model)
            self._cache[edge_id] = signature
        return signature

    def get(self, edge_id: str, default=None):
        if edge_id not in self._model.edges:
            return default
        return self[edge_id]

    @property
    def incidence(self) -> dict[tuple, set[str]]:
        """``signature -> the faces whose loops carry an edge with that signature``.

        ``classify_rim_end`` asks exactly this question once per rim, and used
        to answer it by walking every face, every loop and every edge of the
        whole model each time.  The answer depends only on the model, so it is
        built once, here.

        This is not ``edge_index.EdgeIndex.sig_incidence``, even though both are
        built from the same signature function: this one drops degenerate
        zero-length edges, that one keeps every edge a loop names.
        """
        if self._incidence is None:
            incidence: dict[tuple, set[str]] = {}
            for face_id, face in self._model.faces.items():
                for loop_id in face.loop_ids:
                    loop = self._model.loops.get(loop_id)
                    if loop is None:
                        continue
                    for edge_id in loop.edge_ids:
                        if not carries_length(self._model, edge_id):
                            continue
                        signature = self.get(edge_id)
                        if signature is not None:
                            incidence.setdefault(signature, set()).add(face_id)
            self._incidence = incidence
        return self._incidence
