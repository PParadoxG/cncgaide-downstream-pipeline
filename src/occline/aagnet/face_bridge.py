"""Tell the evidence producer which source CAD face is which PartModel face.

The other direction - reading an evidence document back - is not here: that is
:mod:`contract.evidence`, because the readers describe the stored format and
every consumer needs them, while this mapping is built for one particular
producer from the model it is about to be run on.
"""

from __future__ import annotations

from contract.partmodel import PartModel
from occline.face_ids import raw_cad_face_id


def build_cad_face_mappings(model: PartModel) -> dict:
    """Map each merge-before OCC CAD face id to its active PartModel face.

    The rebuilt v6 pipeline does not keep the old FaceLedger abstraction. The
    current parser keeps merge provenance on each active face, so each source
    ``face_N`` id can be mapped directly to the active PartModel face id.
    """
    cad_to_partmodel: dict[str, str] = {}
    for face_id, face in model.faces.items():
        for cad_id in [face_id, *getattr(face, "merged_from", [])]:
            idx = raw_cad_face_id(cad_id)
            if idx is not None:
                cad_to_partmodel[str(idx)] = face_id

    return {"cad_face_to_partmodel_face": cad_to_partmodel}
