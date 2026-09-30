"""Compatibility surface: the rule now lives at the seam.

Node 27B put the one reading of ``face_<n>`` here, where the parser and the
AAGNet bridge could both reach it, and recorded that the two remaining copies
in ``src/visualization/id_map.py`` agreed with it by coincidence rather than by
constraint - the display layer may not import a kernel line, so it could not
have used this module.  Node 28 moved the rule to ``contract.face_ids``, which
both sides may name, and left this behind for the two parse-side callers.
"""

from __future__ import annotations

from contract.face_ids import raw_cad_face_id, raw_cad_face_ids  # noqa: F401

__all__ = ["raw_cad_face_id", "raw_cad_face_ids"]
