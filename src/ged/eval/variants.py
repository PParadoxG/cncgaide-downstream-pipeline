"""Variant-family evaluation: within-family ranking, pose invariance, self-retrieval.

A variant corpus (``inputs_variants/`` refreshed to ``runs/variants``) is a
set of families ``F<k>.<i>_<name>``.  Member ``.0`` is the family base and
members are *cumulative* perturbations, so the expected ranking against the
base is the member index.  Family ``F5`` is the pose family (re-export,
translate, rotate): every member should be at distance 0 unless the
*recognition* layer itself drifted, which the evaluation reports separately
as an identification delta instead of a GED failure.

The report doubles as an AAGNet / recognition stability sheet: for every
member it lists the leaf-type and relation histogram deltas against the base.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from ged.cost.config import CostConfig
from ged.cost.default import default_config
from ged.result import GedResult
from ged.retrieval.corpus import Corpus
from ged.retrieval.query import rank
from ged.solver.bp import ged


FAMILY_RE = re.compile(r"^(F\d+)\.(\d+)_")
POSE_FAMILIES: frozenset[str] = frozenset(("F5",))
ZERO_EPS = 1.0e-9
MICRO_EDIT_COST = 1.0e-3


@dataclass
class MemberRow:
    member: str
    index: int
    result: GedResult
    type_delta: dict[str, int]
    relation_delta: dict[str, int]

    @property
    def identification_drift(self) -> bool:
        return bool(self.type_delta)


@dataclass
class FamilyEval:
    family: str
    base: str
    rows: list[MemberRow] = field(default_factory=list)
    is_pose_family: bool = False
    inversions: list[tuple[str, str]] = field(default_factory=list)
    pose_failures: list[str] = field(default_factory=list)
    pose_drift: list[str] = field(default_factory=list)

    @property
    def monotonic(self) -> bool:
        return not self.inversions

    @property
    def passed(self) -> bool:
        return self.monotonic and not self.pose_failures


@dataclass
class VariantEvaluation:
    run_root: Path
    config: dict[str, Any]
    families: list[FamilyEval]
    nearest_neighbour_failures: list[dict[str, Any]]
    nearest_neighbour_drift: list[dict[str, Any]]
    nearest_neighbours: dict[str, tuple[str, float]]
    corpus_audit: dict[str, Any]
    generated_at: str

    @property
    def passed(self) -> bool:
        return all(f.passed for f in self.families) and not self.nearest_neighbour_failures


def discover_families(ids: list[str]) -> dict[str, list[tuple[int, str]]]:
    families: dict[str, list[tuple[int, str]]] = {}
    for graph_id in ids:
        match = FAMILY_RE.match(graph_id)
        if not match:
            continue
        families.setdefault(match.group(1), []).append((int(match.group(2)), graph_id))
    for members in families.values():
        members.sort()
    return dict(sorted(families.items()))


def _delta(base: dict[str, int], other: dict[str, int]) -> dict[str, int]:
    keys = set(base) | set(other)
    return {k: other.get(k, 0) - base.get(k, 0) for k in sorted(keys) if other.get(k, 0) != base.get(k, 0)}


def evaluate_families(corpus: Corpus, config: CostConfig) -> list[FamilyEval]:
    evaluations: list[FamilyEval] = []
    for family, members in discover_families(corpus.ids()).items():
        bases = [gid for index, gid in members if index == 0]
        if len(bases) != 1:
            raise ValueError(f"{family}: expected exactly one base member, found {bases}")
        base_id = bases[0]
        base = corpus[base_id]
        evaluation = FamilyEval(family=family, base=base_id, is_pose_family=family in POSE_FAMILIES)
        for index, gid in members:
            if gid == base_id:
                continue
            other = corpus[gid]
            result = ged(base, other, config)
            row = MemberRow(
                member=gid,
                index=index,
                result=result,
                type_delta=_delta(dict(base.meta.get("typeHistogram", {})), dict(other.meta.get("typeHistogram", {}))),
                relation_delta=_delta(
                    dict(base.meta.get("relationHistogram", {})), dict(other.meta.get("relationHistogram", {}))
                ),
            )
            evaluation.rows.append(row)
        evaluation.rows.sort(key=lambda r: r.index)
        if evaluation.is_pose_family:
            for row in evaluation.rows:
                if row.result.normalized_distance <= ZERO_EPS:
                    continue
                if row.identification_drift:
                    evaluation.pose_drift.append(row.member)
                else:
                    evaluation.pose_failures.append(row.member)
        else:
            for prev, curr in zip(evaluation.rows, evaluation.rows[1:]):
                if curr.result.normalized_distance + ZERO_EPS < prev.result.normalized_distance:
                    evaluation.inversions.append((prev.member, curr.member))
        evaluations.append(evaluation)
    return evaluations


def nearest_neighbour_check(
    corpus: Corpus, config: CostConfig, *, exempt: frozenset[str] = frozenset()
) -> tuple[dict[str, tuple[str, float]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Every non-base member's nearest neighbour must be a member of its own
    family or any family base (all bases are the same part).

    Members in ``exempt`` (pose-family members whose recognition drifted) are
    still ranked and reported, but a foreign nearest neighbour is recorded as
    a drift note instead of a failure: the recognition layer, not the GED,
    moved them.
    """
    families = discover_families(corpus.ids())
    family_of = {gid: fam for fam, members in families.items() for _, gid in members}
    bases = {gid for members in families.values() for index, gid in members if index == 0}
    nearest: dict[str, tuple[str, float]] = {}
    failures: list[dict[str, Any]] = []
    drift_notes: list[dict[str, Any]] = []
    graphs = list(corpus)
    for gid, family in family_of.items():
        if gid in bases:
            continue
        ranked = rank(corpus[gid], graphs, config=config, top=1)
        if not ranked:
            continue
        best = ranked[0]
        nearest[gid] = (best.graph_id, best.result.normalized_distance)
        if family_of.get(best.graph_id) != family and best.graph_id not in bases:
            record = {
                "member": gid,
                "family": family,
                "nearest": best.graph_id,
                "nearest_family": family_of.get(best.graph_id),
                "distance": best.result.normalized_distance,
            }
            (drift_notes if gid in exempt else failures).append(record)
    return nearest, failures, drift_notes


def evaluate(corpus: Corpus, *, config: CostConfig | None = None) -> VariantEvaluation:
    config = config or default_config()
    families = evaluate_families(corpus, config)
    drifted = frozenset(member for fam in families if fam.is_pose_family for member in fam.pose_drift)
    nearest, failures, drift_notes = nearest_neighbour_check(corpus, config, exempt=drifted)
    return VariantEvaluation(
        run_root=corpus.run_root,
        config=config.describe(),
        families=families,
        nearest_neighbour_failures=failures,
        nearest_neighbour_drift=drift_notes,
        nearest_neighbours=nearest,
        corpus_audit=corpus.audit_summary(),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# --------------------------------------------------------------------------
# report


def _fmt(value: float) -> str:
    return f"{value:.6f}"


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


def render_report(evaluation: VariantEvaluation) -> str:
    lines: list[str] = []
    lines.append("# GED variant-family report")
    lines.append("")
    lines.append(f"Generated: `{evaluation.generated_at}`  ")
    lines.append(f"Corpus: `{evaluation.run_root}`  ")
    lines.append(f"Cost config: `{json.dumps(evaluation.config)}`  ")
    lines.append(f"Overall: **{'PASS' if evaluation.passed else 'FAIL'}**")
    lines.append("")
    lines.append("Generated artifact — do not hand-edit; change the projection, cost config or solver and rerun.")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Family | Base | Members | Ranking | Pose check |")
    lines.append("|---|---|---:|---|---|")
    for fam in evaluation.families:
        if fam.is_pose_family:
            ranking = "n/a"
            pose = "PASS" if not fam.pose_failures else "FAIL: " + ", ".join(fam.pose_failures)
            if fam.pose_drift:
                pose += f" (identification drift: {', '.join(fam.pose_drift)})"
        else:
            ranking = "monotonic" if fam.monotonic else "INVERSION: " + "; ".join(f"{a} > {b}" for a, b in fam.inversions)
            pose = "—"
        lines.append(f"| {fam.family} | `{fam.base}` | {len(fam.rows)} | {ranking} | {pose} |")
    lines.append("")
    if evaluation.nearest_neighbour_failures:
        lines.append("Nearest-neighbour failures (member whose closest graph is from another family):")
        for failure in evaluation.nearest_neighbour_failures:
            lines.append(
                f"- `{failure['member']}` → `{failure['nearest']}` ({failure['nearest_family']}) at {_fmt(failure['distance'])}"
            )
    else:
        lines.append("Nearest-neighbour check: every non-base member is closest to its own family or a base.")
    if evaluation.nearest_neighbour_drift:
        lines.append("")
        lines.append("Recognition-drift notes (pose-family members whose leaf set changed; nearest neighbour is foreign but this is not a GED failure):")
        for note in evaluation.nearest_neighbour_drift:
            lines.append(
                f"- `{note['member']}` → `{note['nearest']}` ({note['nearest_family']}) at {_fmt(note['distance'])}"
            )
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
        lines.append("| Member | distance | similarity % | node | fixed | edge | scale | sub/type/del/ins | e preserved/changed/del/ins | nearest |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|---|")
        for row in fam.rows:
            b = row.result.breakdown
            nn = evaluation.nearest_neighbours.get(row.member)
            nn_text = f"`{nn[0]}` ({_fmt(nn[1])})" if nn else "—"
            lines.append(
                f"| `{row.member}` | {_fmt(row.result.normalized_distance)} | {row.result.similarity_percent:.3f} | "
                f"{_fmt(b.node_cost)} | {_fmt(b.fixed_cost)} | {_fmt(b.edge_cost)} | {_fmt(b.scale_cost)} | "
                f"{b.n_sub:g}/{b.n_type_change:g}/{b.n_del:g}/{b.n_ins:g} | "
                f"{b.edges_preserved:g}/{b.edges_changed:g}/{b.edges_del:g}/{b.edges_ins:g} | {nn_text} |"
            )
        lines.append("")
        lines.append("### Identification deltas vs base (recognition stability)")
        lines.append("")
        lines.append("| Member | leaf-type delta | relation delta |")
        lines.append("|---|---|---|")
        for row in fam.rows:
            lines.append(
                f"| `{row.member}` | {json.dumps(row.type_delta) if row.type_delta else '—'} | "
                f"{json.dumps(row.relation_delta) if row.relation_delta else '—'} |"
            )
        lines.append("")
        lines.append("### Cost-bearing edits (base → member, A→B direction)")
        lines.append("")
        for row in fam.rows:
            nodes, edges, micro = _significant_edits(row.result)
            lines.append(f"#### `{row.member}`")
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
