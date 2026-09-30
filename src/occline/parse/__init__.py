"""Stage 1 main entry point — ``process_step_file(filepath) -> PartModel``.

``process_step_shape`` is the same run returning the loaded shape as well, for
the composition root that also needs kernel services over it.

Orchestrates the full pipeline: load → metadata → PMI → traversal →
geometry → adjacency → assembly → normalization → output.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Core.BRepTools import breptools
from OCC.Core.TopoDS import TopoDS_Shape

from occline.parse.adjacency import ShapeTopology, build_adjacency_infos, detect_seam_edges
from occline.parse.face_sampling import fill_face_samples
from occline.parse.geometry import (
    extract_edge_geometry,
    extract_face_area,
    extract_face_bbox,
    extract_face_geometry,
    extract_face_orientation,
    extract_vertex_coords,
)
from contract.partmodel import (
    AdjacencyInfo,
    EdgeData,
    FaceData,
    LoopData,
    PartMetadata,
    PartModel,
    ShellData,
    VertexData,
)
from contract.normalize import annotate_face_axial_extents, normalize_seam_faces
from occline.parse.reader import (detect_ap_version, extract_length_unit, extract_metadata,
                                  extract_pmi_annotations, load_step)
from occline.parse.traversal import (
    iter_edges,
    iter_faces,
    iter_shells,
    iter_vertices,
    iter_wires,
)

logger = logging.getLogger(__name__)


def process_step_file(filepath: str) -> PartModel:
    """Run the full Stage 1 pipeline on a STEP file and return a ``PartModel``.

    This is the single public entry point.  All internal steps are
    orchestrated here in the canonical order.
    """
    return process_step_shape(filepath)[0]


def process_step_shape(filepath: str) -> tuple[PartModel, TopoDS_Shape]:
    """The parsed ``PartModel`` *and* the shape it was parsed from.

    Reading a STEP is the single most expensive operation on this line, and a
    part used to pay for it twice: once here and once when the composition
    root built the kernel services for the same file.  A caller that needs
    both asks for both.

    Sharing the shape also retires an assumption nobody had written down:
    ``cadFaceIds`` is a position in the traversal order, so the two loads had
    to enumerate faces identically for a kernel answer to line up with a
    parsed face at all.  One shape cannot disagree with itself.
    """
    fname = Path(filepath).name
    logger.info("Processing %s", fname)

    # ---- a. Load ------------------------------------------------------------
    shape = load_step(filepath)

    # ---- b. Metadata --------------------------------------------------------
    ap = detect_ap_version(filepath)
    meta_raw = extract_metadata(filepath)
    metadata = PartMetadata(
        source_file=fname,
        ap_version=ap,
        part_name=meta_raw.get("part_name", ""),
        extraction_timestamp=datetime.now(timezone.utc).isoformat(),
        length_unit=extract_length_unit(filepath),
    )

    # ---- c. PMI -------------------------------------------------------------
    pmi = extract_pmi_annotations(filepath, ap)

    # ---- d. Traversal — build structural hierarchy --------------------------
    # d1. Vertices
    vertex_hash_to_id: dict[int, str] = {}
    vertices: dict[str, VertexData] = {}
    _v_counter = 0
    for topods_edge in iter_edges(shape):
        for v in iter_vertices(topods_edge):
            vh = hash(v)
            if vh not in vertex_hash_to_id:
                vid = f"vertex_{_v_counter}"
                _v_counter += 1
                vertex_hash_to_id[vh] = vid
                vertices[vid] = VertexData(
                    id=vid,
                    coordinates=extract_vertex_coords(v),
                )

    # d2. Edges
    edge_hash_to_id: dict[int, str] = {}
    edges: dict[str, EdgeData] = {}
    _e_counter = 0
    for topods_edge in iter_edges(shape):
        eh = hash(topods_edge)
        # Finding B-01: ``iter_edges`` walks the shape face by face, so a
        # shared edge arrives once per face that carries it, and the loop used
        # to mint a fresh id and a fresh EdgeData every time - overwriting the
        # map, so ``loop.edge_ids`` referenced only the last one and every
        # earlier record became an orphan nothing could reach.  1854 of 3489
        # records on this corpus.  The vertex loop above has always guarded
        # this way; the edge loop now does too.
        if eh in edge_hash_to_id:
            continue
        eid = f"edge_{_e_counter}"
        _e_counter += 1
        edge_hash_to_id[eh] = eid
        ctype, params, length = extract_edge_geometry(topods_edge)
        v_ids = [vertex_hash_to_id[hash(v)] for v in iter_vertices(topods_edge)]
        edges[eid] = EdgeData(
            id=eid,
            curve_type=ctype,
            parameters=params,
            length=length,
            vertex_ids=v_ids,
            is_seam=False,
            sample_points=params.pop("sample_points", []),
        )

    # d3. Faces (with loops / wires)
    face_hash_to_id: dict[int, str] = {}
    faces: dict[str, FaceData] = {}
    _f_counter = 0
    loops: dict[str, LoopData] = {}
    for topods_face in iter_faces(shape):
        fh = hash(topods_face)
        fid = f"face_{_f_counter}"
        _f_counter += 1
        face_hash_to_id[fh] = fid

        stype, params = extract_face_geometry(topods_face)
        area = extract_face_area(topods_face)
        is_rev = extract_face_orientation(topods_face)

        _outer_wire = breptools.OuterWire(topods_face)
        _outer_wire_hash = hash(_outer_wire)

        loop_ids: list[str] = []
        for w_idx, wire in enumerate(iter_wires(topods_face)):
            loop_id = f"loop_{_f_counter - 1}_{w_idx}"
            loop_ids.append(loop_id)
            edge_ids_for_loop = [
                edge_hash_to_id[hash(e)]
                for e in iter_edges(wire)
                if hash(e) in edge_hash_to_id
            ]
            loops[loop_id] = LoopData(
                id=loop_id,
                edge_ids=edge_ids_for_loop,
                is_outer=(hash(wire) == _outer_wire_hash),
            )

        faces[fid] = FaceData(
            id=fid,
            surface_type=stype,
            parameters=params,
            area=area,
            is_outer=True,     # set below from shell orientation
            is_reversed=is_rev,
            loop_ids=loop_ids,
            bbox=extract_face_bbox(topods_face),
            adjacencies=[],
        )

    # d4. Shells — determine is_outer for faces
    shells: list[ShellData] = []
    for s_idx, shell in enumerate(iter_shells(shape)):
        sd = ShellData(id=f"shell_{s_idx}")
        is_outer_shell = (shell.Orientation() != TopAbs_REVERSED)

        for f in iter_faces(shell):
            fh = hash(f)
            fid = face_hash_to_id.get(fh)
            if fid and fid in faces:
                sd.face_ids.append(fid)
                faces[fid].is_outer = is_outer_shell

        seen_edge_ids: set[str] = set()
        for e in iter_edges(shell):
            eh = hash(e)
            eid = edge_hash_to_id.get(eh)
            # Same walk, same repetition: a shell's shared edge arrived once
            # per face and was appended each time (finding B-01).
            if eid and eid not in seen_edge_ids:
                seen_edge_ids.add(eid)
                sd.edge_ids.append(eid)

        shells.append(sd)

    # ---- e. Geometry already extracted in step d ----------------------------

    # ---- f. Adjacency with concavity + seam detection -----------------------
    # One incidence map, one hash lookup, one surface-parameter table for both
    # passes: neither can change what the other sees, and the shape is not
    # modified between them.
    topology = ShapeTopology(shape)
    adj_infos = build_adjacency_infos(shape, topology)
    seam_hashes = detect_seam_edges(shape, topology)

    # Mark seam edges
    for sh in seam_hashes:
        eid = edge_hash_to_id.get(sh)
        if eid and eid in edges:
            edges[eid].is_seam = True

    # Populate face adjacencies (translate hash keys → string IDs)
    for fh, infos in adj_infos.items():
        fid = face_hash_to_id.get(fh)
        if fid is None or fid not in faces:
            continue
        for info in infos:
            adj_fid = face_hash_to_id.get(info.adjacent_face_id)
            shared_eid = edge_hash_to_id.get(info.shared_edge_id, "?")
            if adj_fid is None:
                continue
            faces[fid].adjacencies.append(
                AdjacencyInfo(
                    adjacent_face_id=adj_fid,
                    shared_edge_id=shared_eid,
                    connection_type=info.connection_type,
                )
            )

    # ---- g. Assemble --------------------------------------------------------
    model = PartModel(
        metadata=metadata,
        shells=shells,
        faces=faces,
        edges=edges,
        loops=loops,
        vertices=vertices,
        pmi_annotations=pmi,
    )

    # ---- h. Normalize -------------------------------------------------------
    model = annotate_face_axial_extents(model)
    model = normalize_seam_faces(model)

    # ---- h2. Parse-time face facts ------------------------------------------
    # Centroid and interior samples used to be kernel queries; they are static
    # face facts, so the line that parsed the shape fills them.  After the
    # merge, so a merged face pools its members.
    fill_face_samples(model, list(iter_faces(shape)))

    # ---- i. Return ----------------------------------------------------------
    logger.info(
        "Finished %s: %d shells, %d faces, %d edges, %d vertices, %d PMI entries",
        fname,
        len(model.shells),
        len(model.faces),
        len(model.edges),
        len(model.vertices),
        len(model.pmi_annotations),
    )
    return model, shape

