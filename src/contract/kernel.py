"""The second seam: geometry questions that data cannot answer.

Most of what the pipeline needs about a solid is settled once and written
into a :class:`~contract.partmodel.PartModel`.  A handful of questions are
not like that, because *what to ask depends on what recognition has already
decided*: how far this pocket's walls are from that one's, whether the
midpoint of a chord the endpoint traversal just constructed is inside the
material, how much stock is left under a floor this module has just claimed.
Those cannot be precomputed into a JSON file - the query points do not exist
until the layer that asks them has run.

:class:`KernelServices` is the whole of that surface.  Each kernel line
supplies one implementation (``occline.kernel.occ_kernel`` for the OCC line);
every layer above the seam takes one as an argument and never imports a
kernel package.  The methods here were **derived from the call sites that
existed**, not designed in advance, and each was checked to have a
counterpart on the NX side so the interface does not quietly become
OCC-shaped:

===============================  ==========================================
method                           NX counterpart
===============================  ==========================================
``classify_point``               ``uf.Modeling.TraceARay`` parity  (probe)
``face_set_distance``            ``MeasureManager.NewDistance``
``remaining_floor_thickness``    ``uf.Modeling.TraceARay``
``min_wall_thickness``           ``uf.Modeling.TraceARay``
``tessellation_bounds``          ``uf.Modeling.AskBoundingBox``
``tessellate``                   ``uf.Facet``                      (probe)
===============================  ==========================================

Two former methods, ``sample_face_interior`` and ``face_area_centroid``, are
gone: both were static face facts, and they now travel inside
:class:`~contract.partmodel.FaceData` (``interior_samples`` / ``centroid``),
filled at parse time by whichever kernel line built the model.

Faces are addressed by **raw CAD face id**: the zero-based index of the face
in the kernel's own traversal order, the same integer ``FaceData.merged_from``
and ``cadFaceIds`` already carry.  A second kernel filling the same PartModel
has to reproduce that numbering; where it cannot, the mismatch is a fact
about the two kernels and belongs in the record, not hidden behind an
adapter.

Refusal is a first-class answer.  An implementation that cannot measure
something returns ``None`` / an empty list / ``POINT_UNAVAILABLE`` and the
caller records a named gap - no method here may guess, and none may raise to
signal "not supported".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# ``classify_point`` answers, and the only strings it may return.
POINT_IN = "in"
POINT_OUT = "out"
POINT_ON = "on"
POINT_UNKNOWN = "unknown"
POINT_UNAVAILABLE = "unavailable"

# ``FaceSetDistance.status`` vocabulary.  ``measured`` is the only one that
# carries a distance; the rest are named refusals that reach the relation
# layer's evidence verbatim.
DISTANCE_MEASURED = "measured"
DISTANCE_FACE_MAPPING_FAILED = "cad_face_mapping_failed"
DISTANCE_SOLVER_FAILED = "distance_solver_failed"
DISTANCE_NONFINITE = "nonfinite_distance"
DISTANCE_SOLVER_EXCEPTION = "distance_solver_exception"


@dataclass(frozen=True)
class FaceSetDistance:
    """Minimum distance between two sets of faces, or a named refusal.

    ``witness_cad_face_ids`` is the pair of faces the nearest points actually
    sit on, when the implementation can prove it.  It is deliberately
    optional: claiming the first input face was the witness would be a guess,
    and this measurement feeds a published relation property.
    """

    status: str
    distance: float | None = None
    point_a: list[float] | None = None
    point_b: list[float] | None = None
    witness_cad_face_ids: list[int] | None = None
    witness_reason: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class FaceMesh:
    """One raw CAD face's triangulation in world space, plus its boundary polylines.

    Flat float/int lists rather than a kernel's mesh object: this crosses the
    seam, so it carries numbers only.  ``indices`` are local to this face's
    ``positions`` (the caller applies its own vertex offset when packing
    several faces into one buffer), and ``edge_segments`` is a flat run of
    ``x1,y1,z1,x2,y2,z2`` per line segment.
    """

    cad_face_id: int
    positions: list[float] = field(default_factory=list)
    normals: list[float] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
    edge_segments: list[float] = field(default_factory=list)


class KernelServices(ABC):
    """Live geometry queries, as one kernel line implements them."""

    def provenance(self, key: str) -> str:
        """The kernel line's own name for one of its measurement methods.

        These strings land in stored evidence (``result.json``), so each
        implementation answers with its truthful names — the OCC line keeps
        the literals its corpus already carries, the NX line names its NX
        routes.  The base answer is a named "unspecified" rather than a
        borrowed vocabulary.  Keys in use: ``face_set_distance``,
        ``min_wall_thickness``.
        """
        return f"unspecified_kernel_method:{key}"

    # --------------------------------------------------------- solid queries

    @abstractmethod
    def supports_point_classification(self) -> bool:
        """Whether :meth:`classify_point` can answer at all.

        Asked separately because "no classifier available" and "the point sits
        on the surface" lead to different recorded reasons, and a tangential
        transition must not be called smooth for the wrong one.
        """

    @abstractmethod
    def classify_point(self, point: list[float], tolerance: float) -> str:
        """``POINT_IN`` / ``POINT_OUT`` / ``POINT_ON`` against the solid."""

    @abstractmethod
    def face_set_distance(
        self,
        left_cad_face_ids: list[int],
        right_cad_face_ids: list[int],
    ) -> FaceSetDistance:
        """Minimum distance between two face sets, with the nearest point pair."""

    # --------------------------------------------------------------- ray casts

    @abstractmethod
    def remaining_floor_thickness(
        self,
        feature_cad_face_ids: set[int],
        bottom_cad_face_ids: set[int],
        direction: object,
    ) -> float | None:
        """Nearest material below a feature floor, cast along *direction*."""

    @abstractmethod
    def min_wall_thickness(
        self,
        feature_cad_face_ids: set[int],
        wall_cad_face_ids: set[int],
        *,
        material_side: bool = True,
        diagnostics: dict | None = None,
    ) -> float | None:
        """Nearest non-self surface from a wall, cast into material or into air.

        *diagnostics*, when given, is filled with the sampling and hit counts
        that let a reviewer re-open the measurement without re-running the
        pipeline.
        """

    # ------------------------------------------------------------ tessellation

    @abstractmethod
    def tessellation_bounds(self) -> list[float]:
        """``[xmin, ymin, zmin, xmax, ymax, zmax]`` of the whole solid."""

    @abstractmethod
    def tessellate(
        self,
        linear_deflection: float,
        angular_deflection: float,
        edge_deflection: float,
    ) -> list[FaceMesh]:
        """One :class:`FaceMesh` per raw CAD face, in traversal order.

        The deflections are the caller's display policy, not a kernel fact:
        the viewer derives them from the part's size and passes them in, so
        the same tolerance decision holds whichever kernel draws the part.
        A face that tessellates to nothing still gets an entry with empty
        buffers - a face's index in the returned list is its id.
        """
