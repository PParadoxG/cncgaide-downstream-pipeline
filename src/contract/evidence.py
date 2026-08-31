"""The per-face evidence vocabulary: the class names a vote vector is indexed by.

Evidence itself is a plain dict - it is written to ``aagnet_evidence.json``
and read back without any model in the process.  What was *not* plain was the
vocabulary: two modules in the shared feature layer imported this list from
``occline.aagnet.per_face``, and importing that module loads torch, occwl
and pythonOCC.  A layer that only wanted to know which column of a softmax
means "chamfer" was pulling in a geometry kernel and a neural network to find
out.

The list therefore lives here, where both the producer of the evidence and
its consumers can read it without depending on either.

The order is load-bearing twice over: it is the column order of every vote
vector in a stored ``aagnet_evidence.json``, and it is the class count the
segmentor was trained with.  Appending is a retraining decision, and
reordering invalidates every stored run.
"""

from __future__ import annotations

FEATURE_NAMES = [
    "chamfer",
    "through_hole",
    "triangular_passage",
    "rectangular_passage",
    "6sides_passage",
    "triangular_through_slot",
    "rectangular_through_slot",
    "circular_through_slot",
    "rectangular_through_step",
    "2sides_through_step",
    "slanted_through_step",
    "Oring",
    "blind_hole",
    "triangular_pocket",
    "rectangular_pocket",
    "6sides_pocket",
    "circular_end_pocket",
    "rectangular_blind_slot",
    "v_circular_end_blind_slot",
    "h_circular_end_blind_slot",
    "triangular_blind_step",
    "circular_blind_step",
    "rectangular_blind_step",
    "round",
    "stock",
]


# ---------------------------------------------------------------------------
# reading one evidence document
# ---------------------------------------------------------------------------
# Which face a row is about, which row to believe, and what the whole set of
# rows for a face says.  Three plain dict readers: they define how the stored
# document is understood, so they belong with the vocabulary rather than with
# whichever producer happened to write it.


def evidence_by_partmodel_face(evidence_doc: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for item in evidence_doc.get("per_face_evidence", []):
        face_id = item.get("partmodel_face_id")
        if face_id:
            grouped.setdefault(str(face_id), []).append(item)
    return grouped


def best_evidence_for_face(face_id: str, grouped: dict[str, list[dict]]) -> dict | None:
    votes = grouped.get(face_id, [])
    if not votes:
        return None
    return max(votes, key=lambda item: item.get("top_softmax_score", -1.0))


def member_votes(face_id: str, grouped: dict[str, list[dict]]) -> list[dict]:
    result = []
    for item in grouped.get(face_id, []):
        result.append({
            "cad_face_id": item.get("cad_face_id"),
            "top_label": item.get("top_label"),
            "top_softmax_score": item.get("top_softmax_score"),
            "top_k_labels": [entry.get("label") for entry in item.get("top_k", [])[:5]],
        })
    return result
