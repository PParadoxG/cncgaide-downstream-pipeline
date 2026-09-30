"""One part's recognition, from a filled PartModel to fixed nodes, edges and verdicts.

Everything here is a pure reader of three things: the ``PartModel``, the
per-face evidence, and a :class:`~contract.kernel.KernelServices`.  It never
opens a file, never touches a STEP and never names a kernel - which is the
point.  The orchestrator calls it with an OCC kernel after parsing; the
downstream replay guard calls it with a stub kernel and a ``partmodel.json``
read off disk, and both run the same code in the same order.  A sequence that
only existed inline in the orchestrator could not be exercised that way, and
a second copy of the order would have drifted from the first.

The feature modules are called in a fixed, explicit order.  No automatic
feature discovery or separate property-processing pass is involved.

Between the hole modules and the side-wall features this builds the *shared*
structural analysis once - closed side-wall bands and AAGNet transition-face
evidence - and hands the same objects to Pocket and Boss, which read them
with mirrored sign patterns.  Neither owns faces or appears as a node; the
side-wall band analysis is published under ``diagnostics.sideWallBands`` for
audit.  After those consumers, Open Sweep analysis runs on their unclaimed
residue and the formal Slot/Step resolver exposes only semantically accepted
principal sweep faces.

The relation layer runs once the final node set can no longer change, and the
heuristic layer runs after it, once the edges are fixed as well.  Both are
invoked through a single call into their package, both discover their rules
themselves, and neither writes back into the nodes: adding a relation or a
heuristic does not touch this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from contract.partmodel import PartModel
from core.models import FeatureEdge, FeatureNode
from features import (
    BlindHole,
    Boss,
    Chamfer,
    Counterbore,
    Countersink,
    Fillet,
    FreeformFace,
    PlanarFace,
    Pocket,
    Slot,
    Step,
    StepBore,
    ThroughCut,
    ThroughHole,
)
from features._shared.closed_wall_band import analysis_diagnostics, build_edge_index, discover_wall_bands
from features._shared.hole_ladder import discover_hole_ladders
from topology.faces import FaceIntervalTable
from features._shared.open_sweep import analysis_diagnostics as open_sweep_diagnostics, discover_open_sweeps
from features._shared.slot_step_resolution import resolve_open_sweep_candidates
from features._shared.transition_evidence import build_transition_evidence
from heuristics import outcome_counts, run_heuristics
from heuristics.v51.registry import HeuristicRegistry
from relations import run_relations
from relations.v51.registry import RelationRegistry


@dataclass
class Recognition:
    """Everything one part's recognition produced, before anything is written."""

    nodes: list[FeatureNode] = field(default_factory=list)
    edges: list[FeatureEdge] = field(default_factory=list)
    heuristics: dict = field(default_factory=dict)
    heuristic_summary: dict = field(default_factory=dict)
    relation_diagnostics: dict = field(default_factory=dict)
    # The block published as ``result.json``'s ``diagnostics``.
    diagnostics: dict = field(default_factory=dict)

    def heuristic_counts(self) -> dict:
        return outcome_counts(self.heuristic_summary)


def run_recognition(model: PartModel, evidence: dict, kernel=None) -> Recognition:
    """Recognise one part: features, then relations, then heuristics.

    *kernel* is a :class:`~contract.kernel.KernelServices`, or ``None``
    when no kernel is available - in which case every measurement that needs
    one is recorded as a named gap rather than guessed.
    """
    # Explicit fixed order. ThroughHole tests classification but only claims
    # verified open/open walls; BlindHole then sees all remaining candidates.
    #
    # Ownership convention (one convention, every module): a module copies the
    # claimed set it is handed, adds what it owns, and returns the copy.  It
    # never writes into the caller's set, so the running total is whatever the
    # caller rebinds below - which is why every call here rebinds ``claimed``.
    claimed: set[str] = set()
    through_nodes, claimed, through_diagnostics = ThroughHole.recognize_and_build(
        model, evidence, claimed, kernel=kernel
    )
    blind_nodes, claimed, blind_diagnostics = BlindHole.recognize_and_build(model, evidence, claimed, kernel=kernel)
    # Third shared structural analysis, and the first to run: which cylindrical faces form one
    # coaxial stepped bore, in what order, and what sits on its ends.  It owns nothing and names
    # no class; the compound layer dispatches on it (a run of one / two / three-or-more is a
    # hole / Counterbore / StepBore sequence) and Chamfer uses the cones it releases.  It runs
    # here because the low-level hole nodes it re-types are complete, and because Chamfer -
    # further down - needs its answer.
    # It reads one further shared analysis: the face-interval table, which
    # memoises the axial span of a face along an axis (and with it the rim
    # walk a cylinder needs), and carries the part's one signature view.
    intervals = FaceIntervalTable(model)
    ladder = discover_hole_ladders(model, intervals)
    # Fourth shared structural analysis: the per-part edge index (geometric
    # signature equivalence, signature incidence, loop duplicates).  It is a
    # pure function of the model, so it is built here once and handed to every
    # consumer instead of being rebuilt six times over one part.
    edge_index = build_edge_index(model)
    # Shared structural analysis on the faces the hole modules left: closed
    # side-wall bands (owns nothing) and AAGNet chamfer/round transition
    # hints (a hint, not a decision).  Both are built once and consumed by
    # the manufacturing-feature modules below.
    band_analysis = discover_wall_bands(model, claimed, edge_index=edge_index)
    transition = build_transition_evidence(model, evidence)
    pocket_nodes, claimed, pocket_diagnostics = Pocket.recognize_and_build(
        model, evidence, claimed, bands=band_analysis, transition=transition, kernel=kernel,
        edge_index=edge_index,
    )
    # Boss consumes the same band analysis with the mirrored sign pattern, and
    # runs after Pocket so a cavity floor is already owned before the bosses
    # standing on it are registered.
    boss_nodes, claimed, boss_diagnostics = Boss.recognize_and_build(
        model, evidence, claimed, bands=band_analysis, transition=transition, kernel=kernel,
        edge_index=edge_index,
    )
    # ThroughCut reads the same bands with the third sign pattern
    # (CONVEX -> CONVEX) and additionally proves a real opening at both ends.
    # It runs last of the band consumers because a round through hole belongs
    # to ThroughHole, which claimed its walls before the bands were built.
    cut_nodes, claimed, cut_diagnostics = ThroughCut.recognize_and_build(
        model, evidence, claimed, bands=band_analysis, transition=transition, kernel=kernel,
        edge_index=edge_index,
    )
    # Open Sweep is a second shared, non-semantic structural analysis.  It
    # runs only after all currently stable manufacturing features have taken
    # ownership.  Formal Slot/Step resolve the shared candidates before
    # transition/residual registration can consume their principal geometry.
    sweep_analysis = discover_open_sweeps(model, claimed, edge_index=edge_index)
    slot_step_resolutions = resolve_open_sweep_candidates(
        model,
        evidence,
        sweep_analysis,
        transition=transition,
        kernel=kernel,
        claimed_face_ids=claimed,
        edge_index=edge_index,
    )
    slot_nodes, claimed, slot_diagnostics = Slot.recognize_and_build(
        model,
        evidence,
        claimed,
        sweeps=sweep_analysis,
        transition=transition,
        kernel=kernel,
        resolutions=slot_step_resolutions,
    )
    step_nodes, claimed, step_diagnostics = Step.recognize_and_build(
        model,
        evidence,
        claimed,
        sweeps=sweep_analysis,
        transition=transition,
        kernel=kernel,
        resolutions=slot_step_resolutions,
    )
    # The SWB consumers deliberately leave transition faces unclaimed.  Once
    # their endpoint traversal is complete, register the remaining connected
    # transition regions as their own EdgeTreatment nodes.
    # Chamfer additionally receives the cones the hole grammar declined to absorb into a
    # compound.  AAGNet usually proposes them anyway, but a wide screw seat does not look like
    # a chamfer to a classifier, and a released cone that nobody registers would end up owned
    # by no feature at all: the residual passes take planes and freeform surfaces, never cones.
    chamfer_nodes, claimed, chamfer_diagnostics = Chamfer.recognize_and_build(
        model, evidence, claimed, transition=transition, kernel=kernel,
        extra_face_ids=ladder.released_cone_face_ids,
    )
    fillet_nodes, claimed, fillet_diagnostics = Fillet.recognize_and_build(
        model, evidence, claimed, transition=transition, kernel=kernel
    )

    low_level_nodes = [
        *through_nodes,
        *blind_nodes,
        *pocket_nodes,
        *boss_nodes,
        *cut_nodes,
        *slot_nodes,
        *step_nodes,
        *chamfer_nodes,
        *fillet_nodes,
    ]
    (
        active_low_level,
        counterbore_nodes,
        countersink_nodes,
        step_bore_nodes,
        claimed,
        counterbore_diagnostics,
        countersink_diagnostics,
        step_bore_diagnostics,
    ) = _resolve_compounds(model, evidence, claimed, low_level_nodes, kernel, ladder, intervals)

    # Residual registration is intentionally last: it never competes with a
    # manufacturing feature or a compound replacement for ownership.
    planar_nodes, claimed, planar_diagnostics = PlanarFace.recognize_and_build(
        model, evidence, claimed, kernel=kernel
    )
    freeform_nodes, claimed, freeform_diagnostics = FreeformFace.recognize_and_build(
        model, evidence, claimed, kernel=kernel
    )
    nodes = [
        *active_low_level,
        *counterbore_nodes,
        *countersink_nodes,
        *step_bore_nodes,
        *planar_nodes,
        *freeform_nodes,
    ]
    # The final node set is fixed from here on: compound resolution has
    # already retired its component nodes and residual registration has run.
    # The relation layer is a pure reader of that set plus the PartModel; it
    # never edits featureType, face ownership, attributes or
    # missingAttributes.  Rules are discovered from ``relations.v51.rules``,
    # so adding a relation needs no edit here.  This module is the v5.1
    # chain's driver - the feature modules above are v5.1 recognizers - so it
    # names the v5.1 dialect explicitly rather than taking a default.
    edges, relation_diagnostics = run_relations(
        model, nodes, kernel=kernel, registry=RelationRegistry
    )
    # The heuristic layer is the last reader.  It sees the final nodes, the
    # materialized edges and the PartModel, and writes only its own file: a
    # verdict is never folded back into an attribute, an edge or a
    # diagnostic.  Its rules are discovered from ``heuristics.v51.rules``
    # exactly like the relation rules, so adding one needs no edit here.
    heuristics, heuristic_summary = run_heuristics(
        model, nodes, edges, relation_diagnostics=relation_diagnostics,
        registry=HeuristicRegistry,
    )

    return Recognition(
        nodes=nodes,
        edges=edges,
        heuristics=heuristics,
        heuristic_summary=heuristic_summary,
        relation_diagnostics=relation_diagnostics,
        diagnostics={
            "aagnetStatus": evidence.get("status"),
            "featureModules": {
                "ThroughHole": through_diagnostics,
                "BlindHole": blind_diagnostics,
                "Pocket": pocket_diagnostics,
                "Boss": boss_diagnostics,
                "ThroughCut": cut_diagnostics,
                "Slot": slot_diagnostics,
                "Step": step_diagnostics,
                "Chamfer": chamfer_diagnostics,
                "Fillet": fillet_diagnostics,
                "Counterbore": counterbore_diagnostics,
                "Countersink": countersink_diagnostics,
                "StepBore": step_bore_diagnostics,
                "PlanarFace": planar_diagnostics,
                "FreeformFace": freeform_diagnostics,
            },
            "holeLadders": ladder.diagnostics(),
            "sideWallBands": analysis_diagnostics(band_analysis),
            "openSweeps": open_sweep_diagnostics(sweep_analysis),
            "relations": relation_diagnostics,
            "cadFaceIdsEncoding": "transitional zero-based source CAD traversal indexes, comma-separated; RDF/AP242 projection is deferred",
        },
    )


def _resolve_compounds(
    model,
    evidence: dict,
    claimed: set[str],
    low_level_nodes: list,
    kernel=None,
    ladder=None,
    intervals=None,
) -> tuple[list, list, list, list, set[str], dict, dict, dict]:
    """Dispatch each proven coaxial run to the one leaf class its shape names.

    The three outcomes are mutually exclusive by construction, because they are selected by the
    length of the run: a lone cylinder may take a cone (Countersink), a run of two is a
    Counterbore, a run of three or more is a sequence of StepBores plus its terminal hole.  The
    Countersink-then-Counterbore order is kept, but it is no longer load-bearing - under this
    grammar the two can no longer compete for the same cylinder.
    """
    countersink_nodes, claimed, countersink_diagnostics = Countersink.recognize_and_build(
        model, evidence, claimed, component_nodes=low_level_nodes, kernel=kernel, ladder=ladder, intervals=intervals
    )
    countersink_consumed = set(countersink_diagnostics.get("consumedNodeIds", []))
    after_countersink = [node for node in low_level_nodes if node.nodeId not in countersink_consumed]
    counterbore_nodes, claimed, counterbore_diagnostics = Counterbore.recognize_and_build(
        model, evidence, claimed, component_nodes=after_countersink, kernel=kernel, ladder=ladder, intervals=intervals
    )
    counterbore_consumed = set(counterbore_diagnostics.get("consumedNodeIds", []))
    after_counterbore = [node for node in after_countersink if node.nodeId not in counterbore_consumed]
    step_bore_nodes, claimed, step_bore_diagnostics = StepBore.recognize_and_build(
        model, evidence, claimed, component_nodes=after_counterbore, kernel=kernel, ladder=ladder, intervals=intervals
    )
    step_bore_consumed = set(step_bore_diagnostics.get("consumedNodeIds", []))
    active_low_level = [node for node in after_counterbore if node.nodeId not in step_bore_consumed]
    return (
        active_low_level,
        counterbore_nodes,
        countersink_nodes,
        step_bore_nodes,
        claimed,
        counterbore_diagnostics,
        countersink_diagnostics,
        step_bore_diagnostics,
    )
