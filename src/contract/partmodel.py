"""The normalized B-Rep data every layer downstream of a kernel reads.

These dataclasses *are* the data seam.  A ``PartModel`` is whatever a kernel
line managed to say about one solid in plain numbers and strings — no live
geometry handle survives into it, which is what lets the feature, relation,
heuristic, GED and viewer layers run against a model that was filled by a
different kernel entirely.

``PARTMODEL_SCHEMA_VERSION`` names the shape of that data.  It is not written
into ``partmodel.json`` yet: the on-disk form is still the ``__type__``-tagged
encoding :mod:`contract.partmodel_json` produces, and stamping a version into
it would change every stored artifact.  Bump the constant when a field is
added or its meaning changes, and let the writer catch up separately.
"""

from dataclasses import dataclass, field
from enum import Enum

# The field set below, as the pipeline currently expects it.
# v2: FaceData grew ``centroid`` and ``interior_samples`` - two face facts
# that used to be kernel queries and are now filled at parse time.
# v3: PartMetadata grew ``geometry_provider`` - which kernel line filled the
# table.  Empty means the OCC line (every model stored before the field
# existed was OCC-parsed, so absence and occline coincide).
PARTMODEL_SCHEMA_VERSION = "v8.partmodel.v3"


class SurfaceType(str, Enum):
    PLANE = "PLANE"
    CYLINDER = "CYLINDER"
    CONE = "CONE"
    SPHERE = "SPHERE"
    TORUS = "TORUS"
    BSPLINE = "BSPLINE"
    REVOLUTION = "REVOLUTION"
    EXTRUSION = "EXTRUSION"
    OTHER = "OTHER"


class CurveType(str, Enum):
    LINE = "LINE"
    CIRCLE = "CIRCLE"
    ELLIPSE = "ELLIPSE"
    BSPLINE = "BSPLINE"
    OTHER = "OTHER"


class ConcavityType(str, Enum):
    CONCAVE = "CONCAVE"
    CONVEX = "CONVEX"
    SMOOTH = "SMOOTH"
    UNKNOWN = "UNKNOWN"


@dataclass
class VertexData:
    id: str
    coordinates: tuple[float, float, float]


@dataclass
class EdgeData:
    id: str
    curve_type: CurveType
    parameters: dict
    length: float
    vertex_ids: list[str]
    is_seam: bool = False
    sample_points: list[list[float]] = field(default_factory=list)


@dataclass
class AdjacencyInfo:
    adjacent_face_id: str
    shared_edge_id: str
    connection_type: ConcavityType


@dataclass
class LoopData:
    id: str
    edge_ids: list[str]
    is_outer: bool = True


@dataclass
class FaceData:
    """One active face, as numbers and strings.

    ``centroid`` and ``interior_samples`` are static face facts a kernel line
    fills at parse time (they used to be ``KernelServices`` queries):

    * ``centroid`` is the surface-area centroid of the whole (possibly merged)
      face; ``area`` is the weight to pool it with across faces.
    * ``interior_samples`` are ``{"point": [x,y,z], "normal": [x,y,z]}`` pairs
      strictly inside the face's trim, outward unit normals, spread over the
      face in a deterministic order (merged members pooled by ascending raw
      face id).  An empty list means the kernel line could not sample, never
      "this face has no interior".

    ``parameters`` carries the *underlying surface's own* description, and
    ``is_reversed`` says how the face uses it.  For a plane that pairing is
    pinned here, because two kernel lines answered it differently until node
    21' and nothing in this file had said which answer was the contract:

    * ``parameters["normal"]`` is the **natural** surface normal - the one the
      surface geometry defines by itself, with no regard for which side of it
      the solid material lies on.
    * ``is_reversed`` is ``True`` when the face's outward normal is that
      vector **negated**.  So ``outward = -normal if is_reversed else normal``,
      and that single expression must hold whichever line filled the model.

    A filler that only has the already-oriented normal to hand (NX's
    ``AskFaceData`` returns exactly that) must negate it back before storing,
    not store the oriented one and leave the flag to be read some other way.
    ``interior_samples`` normals are the same outward direction, so the two
    fields are a cross-check: they agree, or the model is wrong.
    """

    id: str
    surface_type: SurfaceType
    parameters: dict
    area: float
    is_outer: bool
    is_reversed: bool
    loop_ids: list[str]
    bbox: dict | None = None
    adjacencies: list[AdjacencyInfo] = field(default_factory=list)
    merged_from: list[str] = field(default_factory=list)
    merge_sources: list[dict] = field(default_factory=list)
    centroid: list[float] | None = None
    interior_samples: list[dict] = field(default_factory=list)


@dataclass
class ShellData:
    id: str
    face_ids: list[str] = field(default_factory=list)
    edge_ids: list[str] = field(default_factory=list)
    vertex_ids: list[str] = field(default_factory=list)


@dataclass
class PartMetadata:
    source_file: str
    ap_version: str
    part_name: str
    extraction_timestamp: str
    # Which kernel line filled this model ("occline" / "nxline").  Empty means
    # occline: every model stored before the field existed was OCC-parsed, and
    # the OCC line deliberately does not write it so its stored bytes stand.
    geometry_provider: str = ""
    # The length unit the *source file* declares, in its own words ("MILLI
    # METRE", "INCH"), not the unit this model's numbers are in.  Every length
    # in a PartModel is millimetres - the OCC line pins the conversion and
    # asserts it on every load (node 27B B-11), and the NX line reports what
    # NX hands over - so this field is provenance, not a scale factor, and
    # nothing computes with it.  Empty means the file did not say, said more
    # than one thing, or the line that filled the model cannot ask.
    length_unit: str = ""


@dataclass
class PartModel:
    metadata: PartMetadata
    shells: list[ShellData] = field(default_factory=list)
    faces: dict[str, FaceData] = field(default_factory=dict)
    edges: dict[str, EdgeData] = field(default_factory=dict)
    vertices: dict[str, VertexData] = field(default_factory=dict)
    loops: dict[str, LoopData] = field(default_factory=dict)
    pmi_annotations: list[dict] = field(default_factory=list)
    normalization_log: list[str] = field(default_factory=list)

