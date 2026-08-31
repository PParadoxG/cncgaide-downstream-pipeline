"""The full-NX assembly's composition root.  Runs under ``run_journal.exe``.

This is the (nxline, nx_native) pairing end to end, inside one NX session: the
NX kernel fills the PartModel, NX's own feature recognition says what the
features are, and ``nxline.native`` projects that answer onto feature nodes.
The viewer artifact writers run on top, imported from ``src`` off the path
this file wires itself (plus ``src/nxline/vendor`` for pure-python yaml; NX's
Python bundles none).

**The recipe is the Find Features dialog's factory default** (node 17',
verified against the GUI on 2026-08-28): one Parametric round over every
registered type, All Geometry, no mapping, no access direction, no limit
faces.  Attributes are *enumerated* from ``CAMFeature.Attributes`` - the
collection the Machining Feature Navigator's tooltip shows - so the
installation schema no longer has to be parsed to know what to ask for.

Zero OCCT is enforced, not assumed: an import blocker is installed before
anything else, so any accidental ``import OCC`` raises, and the per-part
summary records that no OCC module ever entered ``sys.modules``.

Job file (env ``GATHERSEED_NX_JOB``):
    {"projectRoot": ..., "outputTag": ...,
     "parts": [{"partId", "prt", "outDir", "summary", "sourceStep"}, ...]}

``outDir`` is where this part's artifacts go, in the store's shapes (partmodel
/ result / heuristics / id_map / graph artifacts / mesh + view).  The store
points it at a staging directory and publishes afterwards, so a half-written
part is never visible; the journal itself does not care which directory it was
handed.  ``summary`` is the one file written outside it, for the CPython
driver to read back.
"""

import json
import os
import sys
import traceback


class _NoOcc:
    """Meta-path blocker: the full-NX process may never import OCC."""

    def find_module(self, name, path=None):
        return self if name == "OCC" or name.startswith("OCC.") else None

    def find_spec(self, name, path=None, target=None):
        if name == "OCC" or name.startswith("OCC."):
            raise ImportError("the full-NX assembly forbids importing " + name)
        return None

    def load_module(self, name):
        raise ImportError("the full-NX assembly forbids importing " + name)


sys.meta_path.insert(0, _NoOcc())

# ``run_journal.exe`` ignores PYTHONPATH entirely (probe: sys.path is the
# script directory plus NX's own trees), so the composition root wires its
# own imports from where it lives: this file sits in ``src/nxline``, which
# puts ``src`` one level up and the vendored pure-python yaml next door.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "vendor"))
sys.path.insert(0, os.path.dirname(_HERE))

import NXOpen  # noqa: E402
import NXOpen.CAM  # noqa: E402
import NXOpen.UF  # noqa: E402


def process_part(session, uf, job, root, tag, first_part, environment):
    from collections import Counter

    from contract.partmodel_json import save_json
    from heuristics import outcome_counts, run_heuristics
    from heuristics.nx._shared import NxHeuristicContext
    from heuristics.nx.registry import NxHeuristicRegistry
    from nxline import derive, journal_native, native
    from nxline.kernel import NxKernelServices
    from nxline.partmodel import build_partmodel
    from relations import run_relations
    from relations.nx.registry import NxRelationRegistry
    from contract.registry import LINE_NX
    from core.paths import mirror_part_paths, part_paths
    from visualization.graph_view import build_graph_view, save_graph_artifacts
    from visualization.id_map import build_id_map, save_id_map
    from visualization.palette import load_palette
    from visualization.viewer_data import build_part_viewer_artifacts

    part_id = job["partId"]
    summary = {"partId": part_id, "status": "success"}
    work = None
    try:
        part, _ = session.Parts.OpenActiveDisplay(
            job["prt"], NXOpen.DisplayPartOption.AllowAdditional)
        work = session.Parts.Work
        if first_part:
            session.CreateCamSession()
            try:
                session.CAMSession.SpecifyConfiguration("${UGII_CAM_CONFIG_DIR}cam_general")
            except Exception:
                pass
        bodies = [b for b in work.Bodies]
        body = bodies[0]
        paths = mirror_part_paths(part_paths(root, tag, part_id, LINE_NX), job["outDir"])
        paths["root"].mkdir(parents=True, exist_ok=True)

        # -- geometry slot: the NX kernel fills the table
        model, cad_faces = build_partmodel(
            uf, work, body,
            source_file=os.path.basename(job.get("sourceStep", job["prt"])),
            part_name=part_id)
        save_json(model, str(paths["partmodel"]))
        face_id_by_tag = {face.Tag: f"face_{index}" for index, face in enumerate(cad_faces)}

        # -- recognition slot: NX's own answer, asked the standard way
        raw = journal_native.extract(work, bodies, environment)

        # -- projection: NX's Type is the class, NX's attributes are the
        # attributes.  No re-segmentation, no re-measurement.
        nodes, diagnostics = native.project(part_id, raw, face_id_by_tag, model=model)
        # -- tier E (node 20'b): the kernel is alive exactly here, so the
        # ``nxd:`` quantities are measured now and stored in each node's
        # separate ``derived`` block - never beside NX's own words - and the
        # two clearance relations get the same live kernel.  Everything
        # downstream, the seconds-tier recompute included, reads the results
        # off disk.
        kernel = NxKernelServices(uf, work, body, cad_faces, model)
        diagnostics["derived"] = derive.derive_nodes(model, nodes, kernel)
        # -- the shared chain, in the nx_v1 dialect.  The rules are written in
        # NX's own class and attribute names and read the ontology under
        # ``ontology/nx_v1/source``; nothing is translated into or out of
        # v5.1, and no derived quantity is written back into a node's NX
        # attributes.
        edges, relation_diagnostics = run_relations(
            model, nodes, kernel=kernel, registry=NxRelationRegistry)
        diagnostics["relations"] = relation_diagnostics
        result = {
            "schemaVersion": "v8.result.v1",
            "partId": part_id,
            "provider": "nx_fbm",
            "projection": native.PROJECTION_ID,
            "geometryProvider": "nxline",
            "sourceStep": job.get("sourceStep"),
            "nodes": [node.to_dict() for node in nodes],
            "edges": [edge.to_dict() for edge in edges],
            "diagnostics": diagnostics,
        }
        paths["result"].write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        heuristics_payload, heuristic_summary = run_heuristics(
            model, nodes, edges, relation_diagnostics=relation_diagnostics,
            registry=NxHeuristicRegistry, context_factory=NxHeuristicContext)
        paths["heuristics"].write_text(
            json.dumps({"partId": part_id, **heuristics_payload}, indent=1, ensure_ascii=False),
            encoding="utf-8")

        id_map = build_id_map(part_id, model, nodes)
        save_id_map(id_map, paths["id_map"])
        palette = load_palette(root)
        graph_view = build_graph_view(part_id, result, id_map, palette,
                                      job.get("sourceStep", ""), tag)
        save_graph_artifacts(graph_view, paths)
        counts = Counter(node.featureType for node in nodes)
        claimed = {face_id for node in nodes for face_id in node.partmodelFaceIds}
        viewer_entry = build_part_viewer_artifacts(
            part_id, kernel, graph_view, id_map, palette, paths,
            face_count=len(model.faces), claimed_face_count=len(claimed),
            hidden_feature_types=diagnostics["defaultHiddenTypes"],
            feature_labels=native.feature_labels(nodes),
            used_attributes=native.used_attributes(nodes))

        summary["entry"] = {
            "partId": part_id,
            "sourceStep": job.get("sourceStep"),
            "sourceStepAbs": job.get("sourceStepAbs"),
            "faceCount": len(model.faces),
            "claimedFaceCount": len(claimed),
            "nodeCount": len(nodes),
            "edgeCount": len(edges),
            "featureCounts": dict(sorted(counts.items())),
            "relationCounts": relation_diagnostics["relationCounts"],
            "heuristicCounts": outcome_counts(heuristic_summary),
        }
        summary["viewerEntry"] = viewer_entry
        summary["heuristicSummary"] = heuristic_summary
        summary["heuristicCounts"] = outcome_counts(heuristic_summary)
    except Exception:
        summary["status"] = "error"
        summary["error"] = traceback.format_exc()[-2000:]
    finally:
        try:
            if work is not None:
                work.Close(NXOpen.BasePartCloseWholeTree.TrueValue,
                           NXOpen.BasePartCloseModified.CloseModified, None)
        except Exception:
            pass
    # Zero OCCT is enforced by the import blocker above; recording what did
    # enter sys.modules is how the run proves it rather than promising it.
    summary["occModules"] = sorted(
        name for name in sys.modules if name == "OCC" or name.startswith("OCC."))
    return summary


def describe_environment():
    """What about this machine could change what NX says.

    The interface language is here because ``TypeDisplayName`` *is* an
    interface string: run this on a Chinese NX and every label in the store
    silently turns Chinese while the ``Type`` keys stay put.  Recording it is
    what lets a reader of the artifacts tell which happened.
    """
    return {
        "ugiiLang": os.environ.get("UGII_LANG") or "",
        "nxVersion": os.environ.get("UGII_VERSION") or "",
    }


def main():
    with open(os.environ["GATHERSEED_NX_JOB"], "r", encoding="utf-8") as handle:
        jobfile = json.load(handle)
    root = jobfile["projectRoot"]
    tag = jobfile["outputTag"]

    session = NXOpen.Session.GetSession()
    uf = NXOpen.UF.UFSession.GetUFSession()
    environment = describe_environment()

    for index, job in enumerate(jobfile["parts"]):
        print("[%d/%d] %s" % (index + 1, len(jobfile["parts"]), job["partId"]))
        summary = process_part(session, uf, job, root, tag, index == 0, environment)
        out_path = job["summary"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=1, ensure_ascii=False, default=str)
        print("  ->", summary["status"],
              "occModules=%d" % len(summary.get("occModules", [])))
    print("journal_fullnx done")


if __name__ == "__main__":
    main()
