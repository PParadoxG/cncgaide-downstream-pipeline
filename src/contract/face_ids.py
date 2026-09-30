"""One place that knows how a PartModel face id encodes a raw CAD face number.

``cadFaceIds`` (raw, pre-merge) and ``partmodelFaceIds`` (post-merge, active)
are two different identity layers, and the bridge between them is the *shape*
of the id string: an active face ``face_7`` was raw CAD face ``7``, and a merged
face carries the rest of its members in ``merged_from``.

That shape had become an unwritten contract enforced in several places at once,
each with its own rule - ``rsplit`` on the last underscore here, ``startswith``
plus ``split`` there, ``re.fullmatch`` in two more.  They agree on ``face_<n>``
and disagree on everything else, which means an id in any other shape produced
a *silently empty* answer rather than a complaint.  This module states the rule
once and refuses out loud: a caller gets the numbers it could read and the ids
it could not, and decides what to do about the second list.

It sits at the seam because both kernel lines and the display layer read the
same id space, and the display layer may not import a kernel line - node 27B
left the rule inside ``occline`` for exactly one node, with the two remaining
copies in ``visualization`` noted as a coincidence rather than a constraint.
Standard library only, like everything else in ``contract``.
"""

from __future__ import annotations

import re

#: An active PartModel face id, exactly.  Anchored on purpose: ``face_12_a`` is
#: not raw CAD face 12, it is an id this line does not issue.
_FACE_ID = re.compile(r"face_(\d+)\Z")


def raw_cad_face_id(face_id: object) -> int | None:
    """The zero-based raw CAD face number in *face_id*, or ``None``."""
    match = _FACE_ID.fullmatch(str(face_id))
    return int(match.group(1)) if match else None


def raw_cad_face_ids(face) -> tuple[list[int], list[str]]:
    """``(raw cad face numbers, ids that are not face ids)`` for one face.

    The face's own id comes first, then everything merged into it.  Numbers are
    returned sorted and deduplicated - the order every consumer of a merged
    face's members already used - and anything unreadable is handed back as a
    string so the caller can record a named gap instead of quietly measuring
    fewer faces than the model has.
    """
    numbers: set[int] = set()
    unreadable: list[str] = []
    for candidate in (face.id, *getattr(face, "merged_from", ())):
        number = raw_cad_face_id(candidate)
        if number is None:
            unreadable.append(str(candidate))
        else:
            numbers.add(number)
    return sorted(numbers), unreadable
