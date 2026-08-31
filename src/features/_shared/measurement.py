"""Kernel measurements as the feature modules ask for them.

Two of the kernel's services - remaining stock under a floor, nearest wall
from a wall - are asked from eight places, always the same way, and always
with a kernel that *might* not be there (unit tests build a model and no
solid).  These two lines are where that case is decided once: no kernel means
no measurement and no claim, never a guessed number and never an exception.

Nothing else belongs here.  A measurement that one feature needs stays in
that feature's own file.
"""

from __future__ import annotations


def remaining_floor_thickness(
    kernel,
    feature_cad_face_ids,
    bottom_cad_face_ids,
    direction,
) -> float | None:
    """Nearest material below a feature floor, or ``None`` without a kernel."""
    if kernel is None:
        return None
    return kernel.remaining_floor_thickness(feature_cad_face_ids, bottom_cad_face_ids, direction)


def min_wall_thickness(
    kernel,
    feature_cad_face_ids,
    wall_cad_face_ids,
    *,
    material_side: bool = True,
    diagnostics: dict | None = None,
) -> float | None:
    """Nearest non-self surface from a wall, or ``None`` without a kernel."""
    if kernel is None:
        return None
    return kernel.min_wall_thickness(
        feature_cad_face_ids,
        wall_cad_face_ids,
        material_side=material_side,
        diagnostics=diagnostics,
    )
