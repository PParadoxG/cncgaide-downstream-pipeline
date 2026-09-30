"""Variant-family evaluation: the assertion set of node 31 (thesis 5.3).

A variant corpus (``inputs_variants/`` refreshed to ``runs/variants``) is a
set of families ``V<b>.<axis><level>_<name>``.  Member ``.0`` is the family
base - a whole part of its own, not a shared reference - and every other
member is **one** perturbation of that base along one axis, at one magnitude
level.  There is no cumulative chain: ``V1.G2`` is not ``V1.G1`` plus more,
it is the same axis at a larger magnitude, so "expected ranking = member
index" (the F1-F5 corpus of node 21') has no meaning here and is gone.

The axis letter says what was perturbed (node 29 design):

    I  invariance    pose / mirror - the part is the same part
    G  geometry      one dimension changed, at level 1 or 2
    S  semantics     a feature that should change class
    T  topology      a feature added or removed
    X  cross-line    a probe aimed at the two lines disagreeing

A two-letter axis (``GT``, ``GS``) is a **combination** member: two axes at
once, one per family.  Combination membership is read from the axis letters
and from nothing else - the manifest's ``perturbation`` list has length two
for several single-axis members as well, so counting entries there would
mislabel them (node 31 §1).

Three of the six checks are assertions and three are observations:

    (a) self-retrieval  every non-base member's nearest neighbour is in its
                        own family; a base's nearest neighbour is one of its
                        own I-axis members
    (b) pose            an I-axis member is at distance <= ZERO_EPS from its
                        base, unless *recognition* drifted, which is reported
                        as drift rather than as a GED failure
    (c) same-axis order distance grows with the magnitude level
    (d) additivity      d(base, combo) against the sum of its two single-axis
                        parts - a ratio, reported, with no pass line
    (e) axis separability  the median distance per axis, in order
    (f) cross-family    the same axis+level across the four families

The report doubles as a recognition-stability sheet: every member lists its
leaf-type and relation histogram deltas against its base.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics
from typing import Any, Callable, Iterable

from ged.contract.schema import GedGraph
from ged.cost.config import CostConfig
from ged.cost.default import default_config
from ged.result import GedResult
from ged.retrieval.corpus import Corpus, run_label
from ged.retrieval.query import rank
from ged.solver.bp import ged

#: How one pair is priced.  The default is the solver; a caller that already
#: has the pair - node 31's sweep re-runs the whole assertion set at 100+
#: configurations - passes a memo in front of it.  Whatever it is, it must
#: return the *induced edit path* for the orientation it was asked about, so
#: that the report's edits are the report's own pair (``retrieval.cache
#: .orient`` is how a memo flips one exactly).
Solve = Callable[[GedGraph, GedGraph, CostConfig], GedResult]


#: ``V<family>.<code>_<name>``; the code is ``0`` for a base, else one or two
#: axis letters followed by the magnitude level.
FAMILY_RE = re.compile(r"^(V\d+)\.(0|[IGSTX]{1,2}\d+)_")
_CODE_RE = re.compile(r"^([IGSTX]{1,2})(\d+)$")

#: The axis whose members are the same part in a different pose.
POSE_AXIS = "I"
#: The axes on which ``level`` is a **magnitude**, so that (c) can ask for an
#: order.  Only G is one: node 29 built its two levels as the same feature's
#: same parameter moved further (``port_diameter`` 12 -> 14 -> 20,
#: ``pocket_corner_radius`` 5 -> 2 -> 0).  On S, T and X the two levels are
#: two *different* specimens - ``V1.T1`` adds a gauge port while ``V1.T2``
#: re-hosts the gallery - so "level 2 must be further than level 1" would be
#: a claim about two unrelated perturbations, which is not a claim about the
#: distance.  Their distances are reported per member and compared to nothing.
ORDERED_AXES: tuple[str, ...] = ("G",)

ZERO_EPS = 1.0e-9
MICRO_EDIT_COST = 1.0e-3


@dataclass(frozen=True)
class Member:
    """One part id, parsed.  ``axis`` is ``""`` and ``level`` 0 for a base."""

    part_id: str
    family: str
    code: str
    axis: str
    level: int

    @property
    def is_base(self) -> bool:
        return self.code == "0"

    @property
    def is_combo(self) -> bool:
        return len(self.axis) == 2

    @property
    def is_pose(self) -> bool:
        return self.axis == POSE_AXIS

    @property
    def axes(self) -> tuple[str, ...]:
        """The single axes this member perturbs (one, or two for a combo)."""
        return tuple(self.axis) if self.axis else ()

    def sort_key(self) -> tuple[int, str, int]:
        return (0 if self.is_base else 1, self.axis, self.level)


def parse_member(part_id: str) -> Member | None:
    """``V2.GS1_combo`` -> Member(family V2, axis GS, level 1); else None."""
    match = FAMILY_RE.match(part_id)
    if not match:
        return None
    family, code = match.group(1), match.group(2)
    if code == "0":
        return Member(part_id=part_id, family=family, code=code, axis="", level=0)
    code_match = _CODE_RE.match(code)
    if not code_match:
        return None
    return Member(part_id=part_id, family=family, code=code,
                  axis=code_match.group(1), level=int(code_match.group(2)))


def discover_families(ids: Iterable[str]) -> dict[str, list[Member]]:
    """Family name -> its members, base first then by (axis, level)."""
    families: dict[str, list[Member]] = {}
    for part_id in ids:
        member = parse_member(part_id)
        if member is None:
            continue
        families.setdefault(member.family, []).append(member)
    for members in families.values():
        members.sort(key=Member.sort_key)
    return dict(sorted(families.items()))


@dataclass
class MemberRow:
    member: Member
    result: GedResult
    type_delta: dict[str, int]
    relation_delta: dict[str, int]

    @property
    def part_id(self) -> str:
        return self.member.part_id

    @property
    def distance(self) -> float:
        return self.result.normalized_distance

    @property
    def identification_drift(self) -> bool:
        return bool(self.type_delta)


@dataclass
class FamilyEval:
    family: str
    base: str
    rows: list[MemberRow] = field(default_factory=list)
    #: (b) I-axis members that moved without the recogniser moving.
    pose_failures: list[str] = field(default_factory=list)
    #: (b) I-axis members that moved because the recogniser did.
    pose_drift: list[str] = field(default_factory=list)
    #: (c) ``(axis, lower member, higher member)`` where the order inverted.
    inversions: list[tuple[str, str, str]] = field(default_factory=list)

    def row(self, part_id: str) -> MemberRow | None:
        for row in self.rows:
            if row.part_id == part_id:
                return row
        return None

    def rows_on(self, axis: str) -> list[MemberRow]:
        return [r for r in self.rows if r.member.axis == axis]

    @property
    def monotonic(self) -> bool:
        return not self.inversions

    @property
    def pose_passed(self) -> bool:
        return not self.pose_failures

    @property
    def passed(self) -> bool:
        return self.monotonic and self.pose_passed


@dataclass
class VariantEvaluation:
    run_root: Path
    line: str
    config: dict[str, Any]
    config_hash: str
    semantics: str
    families: list[FamilyEval]
    nearest_neighbour_failures: list[dict[str, Any]]
    nearest_neighbour_drift: list[dict[str, Any]]
    nearest_neighbours: dict[str, tuple[str, float]]
    additivity: list[dict[str, Any]]
    axis_separability: dict[str, dict[str, Any]]
    cross_family: list[dict[str, Any]]
    corpus_audit: dict[str, Any]
    generated_at: str

    @property
    def outcome(self) -> str:
        """``PASS`` / ``FAIL`` / ``NOT_EVALUABLE``.

        A corpus with no variant family is not a corpus that passed: there was
        nothing to judge.  ``all([])`` is true, so the old two-valued verdict
        printed PASS for a mistyped ``--output-tag``, an unrefreshed
        ``runs/variants`` or a corpus whose ids lost their family prefix -
        exactly the three cases this gate exists to catch (node 26 F-03 /
        node 26 P-10).  The project's own heuristic discipline states it: an
        empty truth is not applicable, not a pass.
        """
        if not self.families:
            return "NOT_EVALUABLE"
        if all(f.passed for f in self.families) and not self.nearest_neighbour_failures:
            return "PASS"
        return "FAIL"

    @property
    def passed(self) -> bool | None:
        """True, False, or **None** when there was nothing to evaluate."""
        outcome = self.outcome
        return None if outcome == "NOT_EVALUABLE" else outcome == "PASS"

    def assertion_summary(self) -> dict[str, dict[str, Any]]:
        """The three assertions, each as ``{passed, failures}``."""
        return {
            "a_self_retrieval": {
                "passed": not self.nearest_neighbour_failures,
                "failures": list(self.nearest_neighbour_failures),
                "drift": list(self.nearest_neighbour_drift),
            },
            "b_pose": {
                "passed": all(f.pose_passed for f in self.families),
                "failures": [{"family": f.family, "members": list(f.pose_failures)}
                             for f in self.families if f.pose_failures],
                "drift": [{"family": f.family, "members": list(f.pose_drift)}
                          for f in self.families if f.pose_drift],
            },
            "c_same_axis_order": {
                "passed": all(f.monotonic for f in self.families),
                "failures": [{"family": f.family, "axis": axis, "lower": lo, "higher": hi}
                             for f in self.families for axis, lo, hi in f.inversions],
            },
        }


def _delta(base: dict[str, int], other: dict[str, int]) -> dict[str, int]:
    keys = set(base) | set(other)
    return {k: other.get(k, 0) - base.get(k, 0) for k in sorted(keys) if other.get(k, 0) != base.get(k, 0)}


def evaluate_families(corpus: Corpus, config: CostConfig, *,
                      solve: Solve = ged) -> list[FamilyEval]:
    evaluations: list[FamilyEval] = []
    for family, members in discover_families(corpus.ids()).items():
        bases = [m for m in members if m.is_base]
        if len(bases) != 1:
            raise ValueError(
                f"{family}: expected exactly one base member, found {[m.part_id for m in bases]}")
        base_id = bases[0].part_id
        base = corpus[base_id]
        evaluation = FamilyEval(family=family, base=base_id)
        for member in members:
            if member.is_base:
                continue
            other = corpus[member.part_id]
            result = solve(base, other, config)
            evaluation.rows.append(MemberRow(
                member=member,
                result=result,
                type_delta=_delta(dict(base.meta.get("typeHistogram", {})),
                                  dict(other.meta.get("typeHistogram", {}))),
                relation_delta=_delta(dict(base.meta.get("relationHistogram", {})),
                                      dict(other.meta.get("relationHistogram", {}))),
            ))
        evaluation.rows.sort(key=lambda r: r.member.sort_key())
        _check_pose(evaluation)
        _check_axis_order(evaluation)
        evaluations.append(evaluation)
    return evaluations


def _check_pose(evaluation: FamilyEval) -> None:
    """(b): an I-axis member is the same part, so its distance must be zero."""
    for row in evaluation.rows_on(POSE_AXIS):
        if row.distance <= ZERO_EPS:
            continue
        if row.identification_drift:
            evaluation.pose_drift.append(row.part_id)
        else:
            evaluation.pose_failures.append(row.part_id)


def _check_axis_order(evaluation: FamilyEval) -> None:
    """(c): within one axis, a larger level is at least as far as a smaller.

    Only axes that actually carry two levels in this family are judged; a
    single-level axis has no order to invert.  The pose axis is exempt - it
    is judged by (b), where every level is expected to be zero.
    """
    for axis in ORDERED_AXES:
        rows = sorted(evaluation.rows_on(axis), key=lambda r: r.member.level)
        for lower, higher in zip(rows, rows[1:]):
            if higher.distance + ZERO_EPS < lower.distance:
                evaluation.inversions.append((axis, lower.part_id, higher.part_id))


def nearest_neighbour_check(
    corpus: Corpus, config: CostConfig, *, exempt: frozenset[str] = frozenset(),
    drift_families: frozenset[str] = frozenset(), solve: Solve = ged,
) -> tuple[dict[str, tuple[str, float]], list[dict[str, Any]], list[dict[str, Any]]]:
    """(a) self-retrieval.

    A non-base member's nearest neighbour must be a member of its own family.
    A **base**'s nearest neighbour must be one of its own I-axis members: the
    pose variants are the same part, so nothing else can legitimately be
    nearer.  The four bases of this corpus are four different parts, so - and
    unlike the F1-F5 corpus, where every base was one shared reference - a
    foreign base is a failure like any other foreign part.

    Two exemptions, both for recognition drift and both recorded rather than
    forgiven: a member in ``exempt`` (an I-axis member the recogniser moved)
    whose nearest neighbour is foreign, and a base of a family in
    ``drift_families`` whose nearest neighbour is one of its **own** members
    but not a pose one - if the pose variant is no longer the same graph, the
    base has no reason to find it first, and that is the recognition layer's
    doing.  A base nearest to another family is a failure either way.
    """
    families = discover_families(corpus.ids())
    member_of = {m.part_id: m for members in families.values() for m in members}
    nearest: dict[str, tuple[str, float]] = {}
    failures: list[dict[str, Any]] = []
    drift_notes: list[dict[str, Any]] = []
    graphs = list(corpus)
    for part_id, member in sorted(member_of.items()):
        ranked = rank(corpus[part_id], graphs, config=config, top=1, solve=solve)
        if not ranked:
            continue
        best = ranked[0]
        best_member = member_of.get(best.graph_id)
        own_family = bool(best_member) and best_member.family == member.family
        nearest[part_id] = (best.graph_id, best.result.normalized_distance)
        if member.is_base:
            wanted = "an I-axis member of its own family"
            ok = own_family and best_member.is_pose
            tolerated = own_family and member.family in drift_families
        else:
            wanted = "a member of its own family"
            ok = own_family
            tolerated = part_id in exempt
        if ok:
            continue
        record = {
            "member": part_id,
            "family": member.family,
            "expected": wanted,
            "nearest": best.graph_id,
            "nearest_family": best_member.family if best_member else None,
            "distance": best.result.normalized_distance,
        }
        (drift_notes if tolerated else failures).append(record)
    return nearest, failures, drift_notes


# --------------------------------------------------------------------------
# (d) (e) (f): observations


def additivity(families: list[FamilyEval]) -> list[dict[str, Any]]:
    """(d) ``d(base, combo)`` against ``d(base, X1) + d(base, Y1)``.

    Reported, never judged: the combination member is one part cut with both
    perturbations, and whether a graph distance is additive under that is the
    question, not the requirement.  A part is skipped, with the reason, when
    one of its single-axis counterparts at the same level is absent.
    """
    rows: list[dict[str, Any]] = []
    for family in families:
        for row in family.rows:
            if not row.member.is_combo:
                continue
            parts: list[dict[str, Any]] = []
            missing: list[str] = []
            for axis in row.member.axes:
                single = next((r for r in family.rows_on(axis)
                               if r.member.level == row.member.level), None)
                if single is None:
                    missing.append(f"{axis}{row.member.level}")
                else:
                    parts.append({"axis": axis, "member": single.part_id,
                                  "distance": single.distance})
            total = sum(p["distance"] for p in parts)
            rows.append({
                "family": family.family,
                "combo": row.part_id,
                "axes": list(row.member.axes),
                "combo_distance": row.distance,
                "parts": parts,
                "sum_of_parts": total,
                "ratio": (row.distance / total) if total > ZERO_EPS else None,
                "missing_parts": missing,
            })
    return rows


def axis_separability(families: list[FamilyEval]) -> dict[str, dict[str, Any]]:
    """(e) per axis: how far its members land from their base.

    Single axes only - a combination member perturbs two of them and belongs
    to neither column.  The median is over all four families, so the number
    answers "what does an S-axis change cost, in general", which is the
    question the design asked (roadmap 2.4).
    """
    by_axis: dict[str, list[dict[str, Any]]] = {}
    for family in families:
        for row in family.rows:
            if row.member.is_combo or not row.member.axis:
                continue
            by_axis.setdefault(row.member.axis, []).append(
                {"member": row.part_id, "family": family.family,
                 "level": row.member.level, "distance": row.distance})
    summary: dict[str, dict[str, Any]] = {}
    for axis, entries in sorted(by_axis.items()):
        distances = sorted(e["distance"] for e in entries)
        summary[axis] = {
            "count": len(distances),
            "median": statistics.median(distances),
            "min": distances[0],
            "max": distances[-1],
            "members": entries,
        }
    return summary


def cross_family(families: list[FamilyEval]) -> list[dict[str, Any]]:
    """(f) the same ``axis+level`` seen across the families.

    The spread of one perturbation over four different base parts is what
    says whether the normalisation carries across parts; it is reported as
    median, range and relative range, and judged by nobody.
    """
    by_code: dict[str, list[dict[str, Any]]] = {}
    for family in families:
        for row in family.rows:
            by_code.setdefault(row.member.code, []).append(
                {"family": family.family, "member": row.part_id, "distance": row.distance})
    rows: list[dict[str, Any]] = []
    for code, entries in sorted(by_code.items()):
        distances = sorted(e["distance"] for e in entries)
        median = statistics.median(distances)
        spread = distances[-1] - distances[0]
        rows.append({
            "code": code,
            "families": len(entries),
            "median": median,
            "min": distances[0],
            "max": distances[-1],
            "range": spread,
            "relative_range": (spread / median) if median > ZERO_EPS else None,
            "members": entries,
        })
    return rows


def evaluate(corpus: Corpus, *, config: CostConfig | None = None,
             solve: Solve = ged) -> VariantEvaluation:
    from ged.cost.presets import config_hash as _config_hash
    from ged.semantics import semantics_stamp

    config = config or default_config()
    families = evaluate_families(corpus, config, solve=solve)
    drifted = frozenset(member for fam in families for member in fam.pose_drift)
    drift_families = frozenset(fam.family for fam in families if fam.pose_drift)
    nearest, failures, drift_notes = nearest_neighbour_check(
        corpus, config, exempt=drifted, drift_families=drift_families, solve=solve)
    return VariantEvaluation(
        run_root=corpus.run_root,
        line=corpus.line,
        config=config.describe(),
        config_hash=_config_hash(config),
        semantics=semantics_stamp(config.representation),
        families=families,
        nearest_neighbour_failures=failures,
        nearest_neighbour_drift=drift_notes,
        nearest_neighbours=nearest,
        additivity=additivity(families),
        axis_separability=axis_separability(families),
        cross_family=cross_family(families),
        corpus_audit=corpus.audit_summary(),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# --------------------------------------------------------------------------
# report


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def _significant_edits(result: GedResult) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    forward = result.edit_path["a_to_b"]
    nodes = [e for e in forward["nodes"] if float(e.get("cost", 0.0)) >= MICRO_EDIT_COST]
    edges = [e for e in forward["edges"] if float(e.get("cost", 0.0)) >= MICRO_EDIT_COST]
    micro = sum(
        1
        for e in list(forward["nodes"]) + list(forward["edges"])
        if 0.0 < float(e.get("cost", 0.0)) < MICRO_EDIT_COST
    )
    return nodes, edges, micro


def _node_edit_line(edit: dict[str, Any]) -> str:
    op = edit["operation"]
    if op == "delete":
        return f"- delete `{edit['source']}` ({edit['source_type']}) cost {_fmt(edit['cost'])}"
    if op == "insert":
        return f"- insert `{edit['target']}` ({edit['target_type']}) cost {_fmt(edit['cost'])}"
    changed = [
        f"{a['attribute']} {a['left']}→{a['right']}"
        for a in edit.get("attributes", [])
        if a.get("distance", 0.0) > 1.0e-6
    ]
    skipped = [a["attribute"] for a in edit.get("attributes", []) if a.get("status") == "skipped_null"]
    head = (
        f"- {op} `{edit['source']}` → `{edit['target']}` "
        f"({edit['source_type']}→{edit['target_type']}) cost {_fmt(edit['cost'])}"
    )
    tail = []
    if changed:
        tail.append("changed: " + ", ".join(changed))
    if skipped:
        tail.append("skipped null: " + ", ".join(skipped))
    return head + (" — " + "; ".join(tail) if tail else "")


def _edge_edit_line(edit: dict[str, Any]) -> str:
    op = edit["operation"]
    kind = edit["type"]
    if op == "delete":
        return f"- delete {kind} {edit['source_edge']} cost {_fmt(edit['cost'])}"
    if op == "insert":
        return f"- insert {kind} {edit['target_edge']} cost {_fmt(edit['cost'])}"
    return f"- change {kind} {edit['source_edge']} cost {_fmt(edit['cost'])} — {json.dumps(edit.get('details'), ensure_ascii=False)}"


def evaluation_payload(evaluation: VariantEvaluation) -> dict[str, Any]:
    """The machine-readable twin of the markdown report."""
    return {
        "schemaVersion": "v8.ged_variant_eval.v2",
        "generatedAt": evaluation.generated_at,
        "runRoot": run_label(evaluation.run_root),
        "line": evaluation.line,
        "config": evaluation.config,
        "configHash": evaluation.config_hash,
        "semantics": evaluation.semantics,
        "outcome": evaluation.outcome,
        "assertions": evaluation.assertion_summary(),
        "families": [
            {
                "family": fam.family,
                "base": fam.base,
                "passed": fam.passed,
                "members": [
                    {
                        "member": row.part_id,
                        "axis": row.member.axis,
                        "level": row.member.level,
                        "combo": row.member.is_combo,
                        "distance": row.distance,
                        "similarityPercent": row.result.similarity_percent,
                        "breakdown": {
                            "node": row.result.breakdown.node_cost,
                            "edge": row.result.breakdown.edge_cost,
                            "fixed": row.result.breakdown.fixed_cost,
                            "scale": row.result.breakdown.scale_cost,
                        },
                        "typeDelta": row.type_delta,
                        "relationDelta": row.relation_delta,
                        "nearest": list(evaluation.nearest_neighbours.get(row.part_id, ())) or None,
                    }
                    for row in fam.rows
                ],
                "poseFailures": list(fam.pose_failures),
                "poseDrift": list(fam.pose_drift),
                "inversions": [{"axis": a, "lower": lo, "higher": hi} for a, lo, hi in fam.inversions],
                "baseNearest": list(evaluation.nearest_neighbours.get(fam.base, ())) or None,
            }
            for fam in evaluation.families
        ],
        "observations": {
            "d_additivity": evaluation.additivity,
            "e_axis_separability": evaluation.axis_separability,
            "f_cross_family": evaluation.cross_family,
        },
        "corpusAudit": evaluation.corpus_audit,
    }


def render_report(evaluation: VariantEvaluation) -> str:
    lines: list[str] = []
    lines.append("# GED variant-family report")
    lines.append("")
    lines.append(f"Generated: `{evaluation.generated_at}`  ")
    lines.append(f"Corpus: `{run_label(evaluation.run_root)}` line `{evaluation.line}`  ")
    lines.append(f"Cost config: `{json.dumps(evaluation.config)}` hash `{evaluation.config_hash}`  ")
    lines.append(f"Semantics stamp: `{evaluation.semantics}`  ")
    lines.append(f"Overall: **{evaluation.outcome}**")
    if evaluation.outcome == "NOT_EVALUABLE":
        lines.append("")
        lines.append("No variant family was found in this corpus: a family is a set of part "
                     "ids sharing a `V<b>.<axis><level>_` prefix, and nothing here has one.  "
                     "There is no verdict to report.")
    lines.append("")
    lines.append("A distance is only comparable to another distance from the same line and the "
                 "same configuration hash; nothing here may be read against the other line.")
    lines.append("")
    lines.append("Generated artifact — do not hand-edit; change the projection, cost config or solver and rerun.")
    lines.append("")

    assertions = evaluation.assertion_summary()
    lines.append("## Assertions")
    lines.append("")
    lines.append("| Assertion | Verdict | Detail |")
    lines.append("|---|---|---|")
    a = assertions["a_self_retrieval"]
    detail_a = "; ".join(
        f"`{f['member']}` → `{f['nearest']}`" for f in a["failures"]) or "every member retrieves its own family"
    lines.append(f"| (a) self-retrieval | {'PASS' if a['passed'] else 'FAIL'} | {detail_a} |")
    b = assertions["b_pose"]
    detail_b = "; ".join(
        f"{f['family']}: {', '.join(f['members'])}" for f in b["failures"]) or "every I-axis member is at zero"
    lines.append(f"| (b) pose invariance | {'PASS' if b['passed'] else 'FAIL'} | {detail_b} |")
    c = assertions["c_same_axis_order"]
    detail_c = "; ".join(
        f"{f['family']} {f['axis']}: `{f['lower']}` > `{f['higher']}`" for f in c["failures"]) or "levels are ordered"
    lines.append(f"| (c) same-axis order | {'PASS' if c['passed'] else 'FAIL'} | {detail_c} |")
    lines.append("")
    if a["drift"]:
        lines.append("Recognition-drift notes for (a) (an I-axis member whose leaf set changed; its "
                     "nearest neighbour is foreign, but the recognition layer moved it, not the GED):")
        for note in a["drift"]:
            lines.append(f"- `{note['member']}` → `{note['nearest']}` at {_fmt(note['distance'])}")
        lines.append("")
    if b["drift"]:
        lines.append("Recognition-drift notes for (b) (a non-zero pose distance whose leaf histogram moved):")
        for note in b["drift"]:
            lines.append(f"- {note['family']}: {', '.join(note['members'])}")
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Family | Base | Members | Same-axis order | Pose | Base nearest |")
    lines.append("|---|---|---:|---|---|---|")
    for fam in evaluation.families:
        order = "ordered" if fam.monotonic else "INVERSION: " + "; ".join(
            f"{axis} {lo} > {hi}" for axis, lo, hi in fam.inversions)
        pose = "zero" if not fam.pose_failures else "FAIL: " + ", ".join(fam.pose_failures)
        if fam.pose_drift:
            pose += f" (drift: {', '.join(fam.pose_drift)})"
        nn = evaluation.nearest_neighbours.get(fam.base)
        nn_text = f"`{nn[0]}` ({_fmt(nn[1])})" if nn else "—"
        lines.append(f"| {fam.family} | `{fam.base}` | {len(fam.rows)} | {order} | {pose} | {nn_text} |")
    lines.append("")

    lines.append("## (d) Additivity of the combination members")
    lines.append("")
    lines.append("Observation, not an assertion.")
    lines.append("")
    lines.append("| Family | Combo | Axes | d(combo) | sum of parts | ratio |")
    lines.append("|---|---|---|---:|---:|---:|")
    for row in evaluation.additivity:
        parts = " + ".join(f"{p['axis']} {_fmt(p['distance'])}" for p in row["parts"]) or "—"
        lines.append(
            f"| {row['family']} | `{row['combo']}` | {'/'.join(row['axes'])} | "
            f"{_fmt(row['combo_distance'])} | {_fmt(row['sum_of_parts'])} ({parts}) | {_fmt(row['ratio'])} |")
    lines.append("")

    lines.append("## (e) Axis separability")
    lines.append("")
    lines.append("Observation, not an assertion.  Single axes only.")
    lines.append("")
    lines.append("| Axis | members | median | min | max |")
    lines.append("|---|---:|---:|---:|---:|")
    for axis, entry in evaluation.axis_separability.items():
        lines.append(f"| {axis} | {entry['count']} | {_fmt(entry['median'])} | "
                     f"{_fmt(entry['min'])} | {_fmt(entry['max'])} |")
    order = " < ".join(axis for axis, _ in sorted(
        evaluation.axis_separability.items(), key=lambda kv: kv[1]["median"]))
    lines.append("")
    lines.append(f"Observed order by median: **{order}**")
    lines.append("")

    lines.append("## (f) Cross-family consistency")
    lines.append("")
    lines.append("Observation, not an assertion.")
    lines.append("")
    lines.append("| Code | families | median | min | max | range | range/median |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in evaluation.cross_family:
        lines.append(
            f"| {row['code']} | {row['families']} | {_fmt(row['median'])} | {_fmt(row['min'])} | "
            f"{_fmt(row['max'])} | {_fmt(row['range'])} | {_fmt(row['relative_range'])} |")
    lines.append("")

    lines.append("## Corpus audit")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(evaluation.corpus_audit, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    for fam in evaluation.families:
        lines.append(f"## {fam.family}: base `{fam.base}`")
        lines.append("")
        lines.append("| Member | axis | level | distance | similarity % | node | fixed | edge | scale | sub/type/del/ins | e preserved/changed/del/ins | nearest |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
        for row in fam.rows:
            b_ = row.result.breakdown
            nn = evaluation.nearest_neighbours.get(row.part_id)
            nn_text = f"`{nn[0]}` ({_fmt(nn[1])})" if nn else "—"
            axis = row.member.axis + (" (combo)" if row.member.is_combo else "")
            lines.append(
                f"| `{row.part_id}` | {axis} | {row.member.level} | {_fmt(row.distance)} | "
                f"{row.result.similarity_percent:.3f} | "
                f"{_fmt(b_.node_cost)} | {_fmt(b_.fixed_cost)} | {_fmt(b_.edge_cost)} | {_fmt(b_.scale_cost)} | "
                f"{b_.n_sub:g}/{b_.n_type_change:g}/{b_.n_del:g}/{b_.n_ins:g} | "
                f"{b_.edges_preserved:g}/{b_.edges_changed:g}/{b_.edges_del:g}/{b_.edges_ins:g} | {nn_text} |"
            )
        lines.append("")
        lines.append("### Identification deltas vs base (recognition stability)")
        lines.append("")
        lines.append("| Member | leaf-type delta | relation delta |")
        lines.append("|---|---|---|")
        for row in fam.rows:
            lines.append(
                f"| `{row.part_id}` | {json.dumps(row.type_delta) if row.type_delta else '—'} | "
                f"{json.dumps(row.relation_delta) if row.relation_delta else '—'} |"
            )
        lines.append("")
        lines.append("### Cost-bearing edits (base → member, A→B direction)")
        lines.append("")
        for row in fam.rows:
            nodes, edges, micro = _significant_edits(row.result)
            lines.append(f"#### `{row.part_id}`")
            lines.append("")
            if not nodes and not edges:
                lines.append("- no edit with cost ≥ 1e-3")
            for edit in nodes:
                lines.append(_node_edit_line(edit))
            for edit in edges:
                lines.append(_edge_edit_line(edit))
            if micro:
                lines.append(f"- {micro} micro edit(s) below 1e-3 omitted (typically normalised-metric drift with L_ref)")
            fixed = row.result.edit_path["a_to_b"]["fixed_nodes"]
            fixed_text = ", ".join(
                f"{f['type']}: {', '.join(f'{k}={v:.4f}' for k, v in f['differences'].items() if v > 0) or 'identical'}"
                for f in fixed
            )
            lines.append(f"- fixed aggregates: {fixed_text}")
            lines.append("")
    return "\n".join(lines) + "\n"


def write_report(evaluation: VariantEvaluation, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_report(evaluation), encoding="utf-8")
    return target


def write_payload(evaluation: VariantEvaluation, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(evaluation_payload(evaluation), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return target
