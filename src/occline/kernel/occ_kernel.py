"""The OCC line's implementation of :class:`~contract.kernel.KernelServices`.

One object, held by the orchestrator and handed to every layer that needs to
ask the solid something.  It owns the loaded shape, the raw-CAD-face table and
a single solid classifier, and forwards each question to the module that
already knew how to answer it.  Nothing above the seam constructs this, and
nothing above the seam names ``OCC``.
"""

from __future__ import annotations

from pathlib import Path

from contract.kernel import FaceMesh, FaceSetDistance, KernelServices, POINT_UNAVAILABLE
from occline.kernel import occ_distance, occ_sampling, occ_tessellation
from occline.kernel.occ_context import OccGeometryContext
from occline.kernel.remaining_floor_thickness import (
    compute_min_wall_thickness,
    compute_remaining_floor_thickness,
)


class OccKernelServices(KernelServices):
    """Geometry services answered by pythonOCC against one loaded STEP shape."""

    def __init__(self, context: OccGeometryContext) -> None:
        self._context = context
        # One classifier for the whole part.  Every call performs before it
        # reads the state, so sharing it across layers is safe and spares the
        # repeated build of its spatial index.
        self._classifier = occ_sampling.SolidClassifier(context.shape)

    @classmethod
    def from_step(cls, source_step: str | Path) -> "OccKernelServices":
        return cls(OccGeometryContext.from_step(source_step))

    @classmethod
    def from_loaded_shape(cls, shape, source_step: str | Path) -> "OccKernelServices":
        """Services over a shape the caller has already read from *source_step*.

        Identical to :meth:`from_step` except that the file is not read again;
        the resolved path is still recorded, so the context says the same thing
        about where the geometry came from.
        """
        return cls(OccGeometryContext(shape, Path(source_step).resolve()))

    # The exact strings the stored corpus already carries: changing one would
    # change every result.json for no measurement reason.
    _PROVENANCE = {
        "face_set_distance": "BRepExtrema_DistShapeShape_side_wall_compounds_v1",
        "min_wall_thickness": "sampled_wall_normals_material_side_excluding_self_hits",
    }

    def provenance(self, key: str) -> str:
        return self._PROVENANCE.get(key) or super().provenance(key)

    # --------------------------------------------------------- solid queries

    def supports_point_classification(self) -> bool:
        return self._classifier.available

    def classify_point(self, point: list[float], tolerance: float) -> str:
        if not self._classifier.available:
            return POINT_UNAVAILABLE
        return self._classifier.state(point, tolerance)

    def face_set_distance(
        self,
        left_cad_face_ids: list[int],
        right_cad_face_ids: list[int],
    ) -> FaceSetDistance:
        return occ_distance.face_set_distance(self._context, left_cad_face_ids, right_cad_face_ids)

    # ----------------------------------------------------------- ray casts

    def remaining_floor_thickness(
        self,
        feature_cad_face_ids: set[int],
        bottom_cad_face_ids: set[int],
        direction: object,
    ) -> float | None:
        return compute_remaining_floor_thickness(
            self._context, feature_cad_face_ids, bottom_cad_face_ids, direction
        )

    def min_wall_thickness(
        self,
        feature_cad_face_ids: set[int],
        wall_cad_face_ids: set[int],
        *,
        material_side: bool = True,
        diagnostics: dict | None = None,
    ) -> float | None:
        return compute_min_wall_thickness(
            self._context,
            feature_cad_face_ids,
            wall_cad_face_ids,
            material_side=material_side,
            diagnostics=diagnostics,
        )

    # -------------------------------------------------------- tessellation

    def tessellation_bounds(self) -> list[float]:
        return occ_tessellation.bounding_box(self._context.shape)

    def tessellate(
        self,
        linear_deflection: float,
        angular_deflection: float,
        edge_deflection: float,
    ) -> list[FaceMesh]:
        return occ_tessellation.tessellate(
            self._context.shape,
            self._context.cad_faces,
            linear_deflection,
            angular_deflection,
            edge_deflection,
        )
