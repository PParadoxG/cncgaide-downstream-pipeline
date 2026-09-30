"""Per-model edge bookkeeping under geometric-signature equivalence.

One pass over the model answers the four questions every boundary walk asks:
which signature an edge has, which edge ids share a signature, which faces
carry a signature in one of their loops, and which signatures are a face's own
seam rather than a boundary with a neighbour.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from contract.partmodel import PartModel

from topology.signatures import edge_signature

#: A PartModel edge id, exactly - the twin of ``contract.face_ids._FACE_ID``.
#: Anchored for the same reason: ``edge_12_a`` is not edge 12, it is an id no
#: line issues.
_EDGE_ID = re.compile(r"edge_(\d+)\Z")


def edge_sort_key(edge_id: object) -> tuple[int | float, str]:
    """Sort edges by their number, so ``edge_9`` precedes ``edge_10``.

    The twin of ``topology.faces.face_sort_key``, and it exists for the same
    reason node 26 finding C-15 gave for the faces: string order puts
    ``edge_10`` before ``edge_2``, and this index's ``representative(sig)`` -
    the first id in a signature's bucket - decides which edge every downstream
    walk actually follows.  Lexical order made that choice depend on how many
    digits the part's edge count happened to reach.

    An id in any other shape sorts last (``inf``) under its own text, rather
    than raising: an index that refuses to be built is a worse answer than one
    that puts an unrecognised id somewhere definite.
    """
    text = str(edge_id)
    match = _EDGE_ID.fullmatch(text)
    return (int(match.group(1)) if match else math.inf, text)


@dataclass
class EdgeIndex:
    """Per-part edge bookkeeping with geometric-signature equivalence."""

    edge_sig: dict[str, tuple]
    sig_to_edge_ids: dict[tuple, list[str]]
    sig_incidence: dict[tuple, set[str]]
    loop_duplicate_edges: set[str]
    self_seam_sigs: set[tuple]

    def representative(self, sig: tuple) -> str:
        return self.sig_to_edge_ids[sig][0]


def build_edge_index(model: PartModel) -> EdgeIndex:
    incidence: dict[str, set[str]] = {}
    for face_id, face in model.faces.items():
        for loop_id in face.loop_ids:
            loop = model.loops.get(loop_id)
            if loop is None:
                continue
            for edge_id in loop.edge_ids:
                if edge_id in model.edges:
                    incidence.setdefault(edge_id, set()).add(face_id)

    duplicated: set[str] = set()
    for loop in model.loops.values():
        seen: set[str] = set()
        for edge_id in loop.edge_ids:
            if edge_id in seen:
                duplicated.add(edge_id)
            seen.add(edge_id)

    edge_sig: dict[str, tuple] = {}
    sig_to_edge_ids: dict[tuple, list[str]] = {}
    for edge_id in sorted(model.edges, key=edge_sort_key):
        sig = edge_signature(model.edges[edge_id], model)
        edge_sig[edge_id] = sig
        sig_to_edge_ids.setdefault(sig, []).append(edge_id)

    sig_incidence: dict[tuple, set[str]] = {}
    for sig, edge_ids in sig_to_edge_ids.items():
        faces: set[str] = set()
        for edge_id in edge_ids:
            faces.update(incidence.get(edge_id, ()))
        sig_incidence[sig] = faces

    self_seam_sigs: set[tuple] = set()
    for sig, edge_ids in sig_to_edge_ids.items():
        if len(sig_incidence[sig]) != 1:
            continue
        if all(model.edges[edge_id].is_seam or edge_id in duplicated for edge_id in edge_ids):
            self_seam_sigs.add(sig)

    return EdgeIndex(
        edge_sig=edge_sig,
        sig_to_edge_ids=sig_to_edge_ids,
        sig_incidence=sig_incidence,
        loop_duplicate_edges=duplicated,
        self_seam_sigs=self_seam_sigs,
    )
