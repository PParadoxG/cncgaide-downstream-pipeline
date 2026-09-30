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
    """Meta-path blocker: the full-NX process may never import OCC.

    ``find_spec`` is the whole protocol: NX 2506 runs Python 3.12, which
    removed the ``find_module`` / ``load_module`` finder methods this class
    used to carry alongside it (node 26E finding E-13).  They were never
    called there, and code that looks like it is guarding something while
    being unreachable is worse than one method fewer.
    """

    def find_spec(self, name, path=None, target=None):
        if name == "OCC" or name.startswith("OCC."):
            raise ImportError("the full-NX assembly forbids importing " + name)
        return None


sys.meta_path.insert(0, _NoOcc())


def _occ_modules():
    """Which OCC modules made it into ``sys.modules`` - the blocker's receipt."""
    return sorted(name for name in sys.modules
                  if name == "OCC" or name.startswith("OCC."))

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

    from contract import refusals
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
    summary = {"partId": part_id, "status": "success", "runToken": job.get("runToken")}
    work = None
    try:
        part, _ = session.Parts.OpenActiveDisplay(
            job["prt"], NXOpen.DisplayPartOption.AllowAdditional)
        work = session.Parts.Work
        if first_part:
            session.CreateCamSession()
            try:
                session.CAMSession.SpecifyConfiguration("${UGII_CAM_CONFIG_DIR}cam_general")
            except Exception as error:
                # The recipe claims a CAM configuration.  If it did not take,
                # that is a fact about this batch's data and has to be
                # readable, not a difference nobody can see.
                summary["camConfigurationProblem"] = "%s: %s" % (
                    type(error).__name__, error)
        bodies = [b for b in work.Bodies]
        solids = [candidate for candidate in bodies if candidate.IsSolidBody]
        if len(solids) > 1:
            # Node 27 ruling: a multi-body part is refused whole, in the same
            # words on both lines.  Building the first body and saying nothing
            # would publish a part that quietly lost the rest of its geometry.
            summary["status"] = "error"
            summary["error"] = refusals.multi_body_unsupported(len(solids))
            summary["bodyCount"] = len(solids)
            summary["occModules"] = _occ_modules()
            return summary
        body = solids[0] if solids else bodies[0]
        paths = mirror_part_paths(part_paths(root, tag, part_id, LINE_NX), job["outDir"])
        paths["root"].mkdir(parents=True, exist_ok=True)

        # -- geometry slot: the NX kernel fills the table
        model, cad_faces = build_partmodel(
            uf, work, body,
            source_file=os.path.basename(job.get("sourceStep", job["prt"])),
            part_name=part_id,
            part_units_enum=getattr(NXOpen.BasePart, "Units", None))
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
        paths["result"].write_text(json.dumps(result, ensure_ascii=False,
                                              separators=(",", ":")),
                                   encoding="utf-8", newline="")
        heuristics_payload, heuristic_summary = run_heuristics(
            model, nodes, edges, relation_diagnostics=relation_diagnostics,
            registry=NxHeuristicRegistry, context_factory=NxHeuristicContext)
        paths["heuristics"].write_text(
            json.dumps({"partId": part_id, **heuristics_payload},
                       ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8", newline="")

        id_map = build_id_map(part_id, model, nodes)
        save_id_map(id_map, paths["id_map"])
        palette = load_palette(root)
        graph_view = build_graph_view(part_id, result, id_map, palette,
                                      job.get("sourceStep", ""))
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
            # No ``sourceStepAbs``: the job file has never carried one, so the
            # key was null in every record ever written (node 26E E-14, node
            # 26P P-08).  The store writes the absolute path at the record's
            # top level, which is where its only reader looks, and a machine
            # path has no business in ``runs/<tag>/nx/`` either.
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
    summary["occModules"] = _occ_modules()
    return summary


def describe_environment(session=None):
    """What about this machine could change what NX says.

    The interface language is here because ``TypeDisplayName`` *is* an
    interface string: run this on a Chinese NX and every label in the store
    silently turns Chinese while the ``Type`` keys stay put.  Recording it is
    what lets a reader of the artifacts tell which happened.

    ``nxVersion`` used to read ``os.environ["UGII_VERSION"]`` and was
    therefore always empty: ``run_journal.exe`` does not put that variable in
    the OS environment, and NX keeps its own table (finding E-06, measured on
    this machine: ``os.environ`` says nothing, NX says ``v2506``).  Two
    answers now, both from NX itself and both named for what they are - the
    release number, which is documented, versioned and needs no licence, and
    NX's own reading of the variable.  Missing when there is no session,
    because a value that is not there must not read as an empty release.
    """
    environment = {
        "ugiiLang": _nx_environment(session, "UGII_LANG") or os.environ.get("UGII_LANG") or "",
        "nxVersion": _nx_environment(session, "UGII_VERSION") or "",
        "nxReleaseNumber": "",
    }
    if session is not None:
        try:
            environment["nxReleaseNumber"] = str(session.ReleaseNumber or "")
        except Exception as error:  # noqa: BLE001 - a named gap, never a guess
            environment["nxReleaseNumberError"] = "%s: %s" % (type(error).__name__, error)
    return environment


def _nx_environment(session, name):
    """One NX environment variable, as NX itself reads it.

    ``run_journal.exe`` starts with an OS environment that does not carry
    every ``UGII_*`` variable NX has; ``GetEnvironmentVariableValue`` asks NX.
    """
    if session is None:
        return ""
    try:
        return str(session.GetEnvironmentVariableValue(name) or "")
    except Exception:  # noqa: BLE001 - the caller falls back and says nothing false
        return ""


def main():
    with open(os.environ["GATHERSEED_NX_JOB"], "r", encoding="utf-8") as handle:
        jobfile = json.load(handle)
    root = jobfile["projectRoot"]
    tag = jobfile["outputTag"]

    session = NXOpen.Session.GetSession()
    uf = NXOpen.UF.UFSession.GetUFSession()
    environment = describe_environment(session)

    for index, job in enumerate(jobfile["parts"]):
        print("[%d/%d] %s" % (index + 1, len(jobfile["parts"]), job["partId"]))
        summary = process_part(session, uf, job, root, tag, index == 0, environment)
        out_path = job["summary"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # No ``default=str``: a summary that cannot be serialised is a bug in
        # what was put into it, and stringifying it hands the driver a
        # sentence where it expected a structure (node 26E finding E-15).
        # It is named instead, and the batch carries on to the next part.
        try:
            payload = json.dumps(summary, indent=1, ensure_ascii=False)
        except (TypeError, ValueError) as error:
            summary = {
                "partId": job["partId"],
                "status": "error",
                "runToken": job.get("runToken"),
                "error": "the journal could not serialise this part's summary: %s: %s"
                         % (type(error).__name__, error),
            }
            payload = json.dumps(summary, indent=1, ensure_ascii=False)
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(payload)
        print("  ->", summary["status"],
              "occModules=%d" % len(summary.get("occModules", [])))
    print("journal_fullnx done")


if __name__ == "__main__":
    main()
