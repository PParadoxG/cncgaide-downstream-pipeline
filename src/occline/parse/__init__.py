"""Stage 1 main entry point — ``process_step_file(filepath) -> PartModel``.

Orchestrates the full pipeline: load → metadata → PMI → traversal →
geometry → adjacency → assembly → normalization → output.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Core.BRepTools import breptools

from occline.parse.adjacency import build_adjacency_infos, detect_seam_edges
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
from occline.parse.reader import detect_ap_version, extract_metadata, extract_pmi_annotations, load_step
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

        for e in iter_edges(shell):
            eh = hash(e)
            eid = edge_hash_to_id.get(eh)
            if eid:
                sd.edge_ids.append(eid)

        for v in iter_vertices(shell) if hasattr(shell, '__iter__') else []:
            pass  # vertices collected via edges above

        shells.append(sd)

    # ---- e. Geometry already extracted in step d ----------------------------

    # ---- f. Adjacency with concavity + seam detection -----------------------
    adj_infos = build_adjacency_infos(shape)
    seam_hashes = detect_seam_edges(shape)

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
    return model

