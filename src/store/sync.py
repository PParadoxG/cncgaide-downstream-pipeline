"""Synchronise ``runs/<tag>`` with the input directory, incrementally.

``plan`` decides per part; ``execute`` carries the plan out and rebuilds the
two indexes.  Nothing here is ever a whole-run wipe:

* a part that needs rebuilding is processed into a *staging* directory
  (``runs/<tag>/.staging/<id>/``) and published file by file with
  ``os.replace`` - a reader sees the old file until the instant the new one is
  complete, and the rest of the corpus is never touched;
* a part whose only stale layer is ``heuristics`` gets its verdicts recomputed
  from the stored ``partmodel.json`` + ``result.json`` - no STEP parsing, no
  AAGNet - and only ``heuristics.json`` and its record change;
* a part whose source STEP disappeared is dropped, because artifacts of a
  CAD file nobody has any more are not worth keeping; ``keep_missing`` holds
  them for the case where the input directory is temporarily unreadable.

The AAGNet bundle is loaded only when at least one part is actually rebuilt,
which is what makes "nothing but thresholds.yaml changed" a matter of seconds.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from contract.registry import LINE_NX, LINE_OCC, LINE_ORDER, LINES
from core.paths import (
    collect_step_files,
    ensure_run_paths,
    make_part_id,
    mirror_part_paths,
    part_dir,
    part_paths,
    project_relative,
    resolve_input_path,
    run_paths,
)
from store import stamps as stamping
from store.index import rebuild_index
from store.records import (
    NOT_BUILT,
    atomic_write_text,
    get_line,
    line_block,
    load_record,
    new_record,
    not_built_block,
    record_path,
    records_by_id,
    save_record,
    set_line,
    untracked_part_dirs,
    utc_now,
)

STAGING_DIR = ".staging"
JOB_ID_ENV = "GATHERSEED_JOB_ID"

MODES = ("stale", "all")


# -------------------------------------------------------------------- plan

@dataclass
class PlanItem:
    """One part: the light it shows, and what each line is owed."""

    partId: str
    stepPath: Path | None
    light: str
    lightReasons: list[str] = field(default_factory=list)
    # layers some already-built line of this part is behind on; empty when the
    # part was built under the current rules.  Never part of the light.
    rulesChanged: list[str] = field(default_factory=list)
    actions: dict[str, stamping.LineAction] = field(default_factory=dict)
    record: dict | None = None

    def action(self, line: str) -> str:
        entry = self.actions.get(line)
        return entry.action if entry is not None else stamping.SKIP


@dataclass
class Plan:
    projectRoot: Path
    outputTag: str
    inputPath: Path
    inputRoot: Path
    mode: str
    only: list[str] | None
    stamps: dict[str, str]
    items: list[PlanItem] = field(default_factory=list)
    # part directories with neither a record nor a source STEP
    untracked: list[str] = field(default_factory=list)
    buildViewer: bool = True
    lines: tuple[str, ...] = LINE_ORDER

    def counts(self) -> dict:
        counts: dict = {light: 0 for light in stamping.LIGHTS}
        counts["untracked"] = len(self.untracked)
        counts["rulesChanged"] = sum(1 for item in self.items if item.rulesChanged)
        per_line = {line: {action: 0 for action in stamping.ACTIONS} for line in self.lines}
        for item in self.items:
            counts[item.light] = counts.get(item.light, 0) + 1
            for line in self.lines:
                per_line[line][item.action(line)] += 1
        counts["lines"] = per_line
        return counts

    def needs_bundle(self) -> bool:
        """AAGNet is loaded only if the open-source line actually rebuilds."""
        return any(item.action(LINE_OCC) == stamping.BUILD for item in self.items)

    def needs_line(self, line: str) -> list[PlanItem]:
        return [item for item in self.items
                if item.action(line) in (stamping.BUILD, stamping.HEURISTICS_ONLY)]

    def describe(self) -> str:
        out = [f"plan for runs/{self.outputTag} from {self.inputPath} (mode={self.mode}):"]
        for item in self.items:
            why = "; ".join(item.lightReasons) if item.lightReasons else "source unchanged"
            out.append(f"  {item.light:15s} {item.partId}  - {why}")
            for line in self.lines:
                entry = item.actions.get(line)
                if entry is None:
                    continue
                detail = "; ".join(entry.reasons) if entry.reasons else "up to date"
                out.append(f"      {line:3s} {entry.action:16s} {detail}")
        for name in self.untracked:
            out.append(f"  {'untracked':15s} {name}  - part directory without record or source")
        out.append("  " + plan_summary(self.counts()))
        return "\n".join(out)


def _line_counts(counts: dict, line: str) -> dict:
    return (counts.get("lines") or {}).get(line) or {}


def plan_summary(counts: dict) -> str:
    """What a plan is about to do, one line."""
    out = [
        f"{counts.get(stamping.FRESH, 0)} unchanged",
        f"{counts.get(stamping.CHANGED, 0)} changed",
        f"{counts.get(stamping.NEW, 0)} new",
        f"{counts.get(stamping.SOURCE_MISSING, 0)} source missing",
    ]
    for line in LINE_ORDER:
        per = _line_counts(counts, line)
        todo = [f"{per.get(stamping.BUILD, 0)} build"]
        if per.get(stamping.HEURISTICS_ONLY):
            todo.append(f"{per[stamping.HEURISTICS_ONLY]} heuristics-only")
        if per.get(stamping.FAILED):
            todo.append(f"{per[stamping.FAILED]} failed")
        out.append(f"{line}: " + ", ".join(todo))
    if counts.get("rulesChanged"):
        out.append(f"{counts['rulesChanged']} behind current rules")
    if counts.get("untracked"):
        out.append(f"{counts['untracked']} untracked")
    return "; ".join(out)


def run_summary(counts: dict) -> str:
    """What a sync did, one line."""
    out = [f"{counts.get('unchanged', 0)} unchanged"]
    for line in LINE_ORDER:
        per = _line_counts(counts, line)
        done = [f"{per.get('built', 0)} built"]
        for key, label in (("heuristics_only", "heuristics-only"),
                           ("error", "failed"), ("not_built", "not built")):
            if per.get(key):
                done.append(f"{per[key]} {label}")
        out.append(f"{line}: " + ", ".join(done))
    if counts.get("dropped"):
        out.append(f"{counts['dropped']} dropped")
    if counts.get("untracked"):
        out.append(f"{counts['untracked']} untracked")
    return "; ".join(out)


def plan(
    project_root: str | Path,
    input_path: str | Path,
    output_tag: str = "latest",
    *,
    only: list[str] | None = None,
    mode: str = "stale",
    limit: int = 0,
    build_viewer: bool = True,
    lines: tuple[str, ...] = LINE_ORDER,
) -> Plan:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    unknown_lines = [line for line in lines if line not in LINES]
    if unknown_lines:
        raise ValueError("unknown line(s): " + ", ".join(unknown_lines))
    root = Path(project_root).resolve()
    input_abs = resolve_input_path(input_path, root)
    input_root = input_abs if input_abs.is_dir() else input_abs.parent
    step_files = collect_step_files(input_abs)
    if limit > 0:
        step_files = step_files[:limit]
    steps = {make_part_id(path, input_root): path for path in step_files}
    records = records_by_id(root, output_tag)
    stamps = stamping.layer_stamps(root)
    run = run_paths(root, output_tag)

    ids = sorted(set(steps) | set(records) if limit <= 0 else steps, key=str.lower)
    if only:
        unknown = [part_id for part_id in only if part_id not in ids]
        if unknown:
            raise ValueError("unknown part id(s): " + ", ".join(unknown))
        wanted = set(only)
        ids = [part_id for part_id in ids if part_id in wanted]

    items: list[PlanItem] = []
    for part_id in ids:
        record = records.get(part_id)
        step_path = steps.get(part_id)
        verdict = stamping.light(record, step_path)
        actions: dict[str, stamping.LineAction] = {}
        for line in lines:
            entry = stamping.plan_line(
                record, line, step_path, stamps,
                want_viewer=build_viewer, force=(mode == "all"),
            )
            # The cheap path needs the artifacts it recomputes from; without
            # them it is a full build, and the plan says so rather than
            # discovering it half way through the sync.
            if entry.action == stamping.HEURISTICS_ONLY:
                paths = part_paths(root, output_tag, part_id, line)
                if not (paths["partmodel"].is_file() and paths["result"].is_file()):
                    entry = stamping.LineAction(
                        line, stamping.BUILD,
                        entry.reasons + ["stored partmodel.json/result.json missing"],
                        entry.staleLayers)
            actions[line] = entry
        items.append(PlanItem(
            partId=part_id,
            stepPath=step_path,
            light=verdict.state,
            lightReasons=list(verdict.reasons),
            rulesChanged=stamping.rules_changed(record, stamps),
            actions=actions,
            record=record,
        ))

    untracked = [name for name in untracked_part_dirs(run["run_root"]) if name not in steps]
    return Plan(
        projectRoot=root,
        outputTag=output_tag,
        inputPath=input_abs,
        inputRoot=input_root,
        mode=mode,
        only=list(only) if only else None,
        stamps=stamps,
        items=items,
        untracked=untracked,
        buildViewer=build_viewer,
        lines=tuple(lines),
    )


# ----------------------------------------------------------------- execute

def new_job_id() -> str:
    return os.environ.get(JOB_ID_ENV) or f"job_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{os.getpid()}"


def _empty_line_counts() -> dict:
    return {"built": 0, "heuristics_only": 0, "error": 0, "not_built": 0, "skipped": 0}


def execute(
    plan_: Plan,
    *,
    device: str = "auto",
    build_viewer: bool | None = None,
    keep_missing: bool = False,
    job_id: str | None = None,
) -> dict:
    """Carry out *plan_*; rebuild the indexes; return a summary.

    Lines are built in registry order, the open-source one first: it needs
    neither a licence nor a vendor install, so it is the one that always
    produces something, and the NX line's absence never costs the part its
    open-source artifacts.
    """
    root = plan_.projectRoot
    tag = plan_.outputTag
    if build_viewer is None:
        build_viewer = plan_.buildViewer
    job_id = job_id or new_job_id()
    ensure_run_paths(root, tag)

    counts: dict = {
        "unchanged": 0, "dropped": 0, "untracked": 0,
        "lines": {line: _empty_line_counts() for line in plan_.lines},
    }
    processed: list[str] = []
    failed_now: list[str] = []

    live = [item for item in plan_.items if item.light != stamping.SOURCE_MISSING]
    for item in plan_.items:
        if item.light != stamping.SOURCE_MISSING:
            continue
        if keep_missing:
            print(f"KEEP {item.partId}: source missing (kept by --keep-missing)")
        else:
            _remove_part_dir(root, tag, item.partId)
            counts["dropped"] += 1
            print(f"DROP {item.partId}: source STEP gone, artifacts removed")

    for line in plan_.lines:
        builder = LINE_BUILDERS.get(line)
        todo = [item for item in live if item.action(line) != stamping.SKIP]
        if builder is None:
            for item in todo:
                counts["lines"][line]["not_built"] += 1
            if todo:
                print(f"{line}: no builder registered; {len(todo)} part(s) left alone")
            continue
        outcome = builder(root, tag, plan_, todo, device, build_viewer, job_id)
        for key, value in outcome.items():
            counts["lines"][line][key] = counts["lines"][line].get(key, 0) + value
        for item in todo:
            if item.partId not in processed:
                processed.append(item.partId)
    counts["unchanged"] = sum(
        1 for item in live
        if all(item.action(line) == stamping.SKIP for line in plan_.lines))

    if keep_missing:
        counts["untracked"] = len(plan_.untracked)
    else:
        for name in plan_.untracked:
            _remove_part_dir(root, tag, name)
            counts["dropped"] += 1
            print(f"DROP {name}: part directory with neither record nor source")

    for item in live:
        record = load_record(record_path(root, tag, item.partId))
        for line in plan_.lines:
            block = (record or {}).get("lines", {}).get(line) or {}
            if block.get("status") == "FAIL" and item.partId not in failed_now:
                failed_now.append(item.partId)

    manifest, index = rebuild_index(
        root, tag,
        input_path=str(plan_.inputPath),
        build_viewer=True if build_viewer else None,
        stamps=plan_.stamps,
    )
    if index is not None:
        failures = index.get("checkFailures", [])
        print(f"Viewer data: {len(index.get('parts', []))} parts, {len(failures)} check failure(s)")
        for failure in failures:
            print(f"  {failure['status']} {failure['partId']}: {failure['check']}: {failure['detail']}")
    print("Store: " + run_summary(counts))
    return {
        "jobId": job_id,
        "counts": counts,
        "processed": processed,
        "failedParts": failed_now,
        "manifest": manifest,
        "viewerIndex": index,
    }


# ------------------------------------------------------------ line builders
#
# One entry per line that this store can build.  A line with no entry is
# simply not built, and the plan still reports what it would owe - which is
# what "the store knows about a line it cannot produce yet" looks like.

def _build_occ_line(root: Path, tag: str, plan_: Plan, todo: list[PlanItem],
                    device: str, build_viewer: bool, job_id: str) -> dict:
    """The open-source line: OCC parses, AAGNet votes, the shared chain runs."""
    counts = _empty_line_counts()
    bundle = None
    if any(item.action(LINE_OCC) == stamping.BUILD for item in todo):
        from occline.aagnet.per_face import load_aagnet_bundle

        print("Loading AAGNet model...")
        bundle = load_aagnet_bundle(device)

    for item in todo:
        action = item.action(LINE_OCC)
        if action == stamping.BUILD:
            ok = _rebuild_part(root, tag, item, LINE_OCC, bundle, build_viewer, plan_.stamps, job_id)
            counts["built" if ok else "error"] += 1
        elif action == stamping.HEURISTICS_ONLY:
            ok = _recompute_part_heuristics(root, tag, item, LINE_OCC, plan_.stamps, job_id)
            counts["heuristics_only" if ok else "error"] += 1
        elif action == stamping.FAILED:
            counts["error"] += 1
    return counts


def _mark_not_built(root: Path, tag: str, items: list[PlanItem], reason: str,
                    job_id: str, counts: dict) -> dict:
    """Record, for each part, that the NX line could not be built and why.

    Deliberately not a failure: the reason is about this machine, not about
    these parts, so the block carries no stamps, the open-source artifacts are
    untouched, and the next sync tries again without anything having to change.
    """
    for item in items:
        record = part_record(root, tag, item)
        set_line(record, LINE_NX, not_built_block(reason, job_id=job_id))
        save_record(record_path(root, tag, item.partId), record)
        counts["not_built"] += 1
    print(f"{LINE_NX}: not built for {len(items)} part(s): {reason[:160]}")
    return counts


def _build_nx_line(root: Path, tag: str, plan_: Plan, todo: list[PlanItem],
                   device: str, build_viewer: bool, job_id: str) -> dict:
    """The NX line: NX fills the PartModel, NX FBM asserts, in one session.

    The cheap path runs first and on its own: recomputing verdicts reads the
    stored PartModel and result, so it works with no NX and no licence, and a
    threshold edit must not be held hostage to either.
    """
    counts = _empty_line_counts()
    build: list[PlanItem] = []
    for item in todo:
        action = item.action(LINE_NX)
        if action == stamping.HEURISTICS_ONLY:
            ok = _recompute_part_heuristics(root, tag, item, LINE_NX, plan_.stamps, job_id)
            counts["heuristics_only" if ok else "error"] += 1
        elif action == stamping.BUILD:
            build.append(item)
        elif action == stamping.FAILED:
            counts["error"] += 1
    if not build:
        return counts

    from nxline.config import NX_ROOT, nx_available, probe_license
    from nxline.fullnx import run_nx_line

    if not nx_available():
        return _mark_not_built(root, tag, build, f"NX install not found under {NX_ROOT}",
                               job_id, counts)
    reachable, detail = probe_license()
    if not reachable:
        return _mark_not_built(root, tag, build, f"NX licence unavailable: {detail}",
                               job_id, counts)

    staged: dict[str, tuple[Path, PlanItem]] = {}
    jobs = []
    for item in build:
        staging, _paths = _staging_paths(root, tag, item.partId, LINE_NX)
        staged[item.partId] = (staging, item)
        jobs.append({"partId": item.partId, "stepPath": str(item.stepPath),
                     "outDir": str(staging),
                     "sourceStep": project_relative(root, item.stepPath)})

    print(f"{LINE_NX}: building {len(jobs)} part(s) in one NX session...")
    outcome = run_nx_line(root, tag, jobs)
    if outcome["status"] == "blocked":
        for staging, _item in staged.values():
            shutil.rmtree(staging, ignore_errors=True)
        return _mark_not_built(root, tag, build,
                               f"NX licence unavailable: {outcome.get('detail', '')}",
                               job_id, counts)

    stamps = {layer: plan_.stamps[layer] for layer in stamping.LINE_LAYERS[LINE_NX]}
    for item in build:
        staging, _ = staged[item.partId]
        summary = outcome["parts"].get(item.partId) or {
            "status": "error", "error": "the NX session reported nothing for this part"}
        record = part_record(root, tag, item)
        live_root = part_paths(root, tag, item.partId, LINE_NX)["root"]
        if summary.get("status") == "blocked":
            # One part's translation could not reach the licence server while
            # others got through.  Still the environment, still not this
            # part's failure.
            shutil.rmtree(staging, ignore_errors=True)
            set_line(record, LINE_NX, not_built_block(
                f"NX licence unavailable: {(summary.get('error') or '')[-300:]}", job_id=job_id))
            save_record(record_path(root, tag, item.partId), record)
            print(f"{LINE_NX} {item.partId}: not built (licence unavailable)")
            counts["not_built"] += 1
            continue
        if summary.get("status") != "success":
            shutil.rmtree(staging, ignore_errors=True)
            set_line(record, LINE_NX, line_block(
                status="FAIL", stamps=stamps,
                fingerprint_sha=record["fingerprint"]["sha256"],
                device="nx", job_id=job_id,
                error=(summary.get("error") or summary.get("detail") or "")[-400:],
            ))
            save_record(record_path(root, tag, item.partId), record)
            print(f"FAIL {LINE_NX} {item.partId}: {(summary.get('error') or '')[-160:]}")
            counts["error"] += 1
            continue
        entry = summary.get("entry", {})
        _publish(staging, live_root)
        set_line(record, LINE_NX, line_block(
            status="PASS", stamps=stamps,
            fingerprint_sha=record["fingerprint"]["sha256"],
            device="nx", job_id=job_id,
            entry=entry,
            viewer_entry=summary.get("viewerEntry"),
            heuristic_summary=summary.get("heuristicSummary", {}),
            heuristic_counts=summary.get("heuristicCounts", {}),
            artifact_digest=stamping.artifact_digest(live_root),
        ))
        record["lines"][LINE_NX]["occModules"] = summary.get("occModules", [])
        save_record(record_path(root, tag, item.partId), record)
        counts_line = entry.get("featureCounts") or {}
        print(f"PASS {LINE_NX} {item.partId}: {entry.get('nodeCount')} nodes, "
              f"{entry.get('edgeCount')} edges "
              f"({', '.join(f'{k}x{v}' for k, v in counts_line.items())})")
        counts["built"] += 1
    return counts


LINE_BUILDERS = {LINE_OCC: _build_occ_line, LINE_NX: _build_nx_line}


# ------------------------------------------------------------- rebuild one

def _staging_paths(root: Path, tag: str, part_id: str, line: str) -> tuple[Path, dict[str, Path]]:
    """A mirror of one line's live folder, under ``.staging/<id>/<line>/``.

    Staging mirrors the live layout one level deeper than it used to: a part
    now has one folder per line, and a rebuild replaces one of them without
    touching the other or the record between them.
    """
    live = part_paths(root, tag, part_id, line)
    staging = run_paths(root, tag)["run_root"] / STAGING_DIR / part_id / line
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    return staging, mirror_part_paths(live, staging)


def _replace_with_retry(source: Path, target: Path, attempts: int = 20) -> None:
    """``os.replace`` that tolerates a reader holding the old file briefly."""
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def _publish(staging: Path, live: Path) -> None:
    """Move one line's staged files into place, one atomic replace each, then
    drop whatever an older build of *that line* left behind.

    Only the named line folder is touched: the other line's folder and the
    record above them both are not in *live* at all, so a rebuild of one line
    can never disturb them."""
    live.mkdir(parents=True, exist_ok=True)
    published: set[str] = set()
    for child in sorted(staging.iterdir()):
        if child.is_file():
            _replace_with_retry(child, live / child.name)
            published.add(child.name)
    for child in live.iterdir():
        if child.is_file() and child.name not in published:
            try:
                child.unlink()
            except OSError:
                pass
    shutil.rmtree(staging, ignore_errors=True)
    try:  # the per-part staging dir is empty once its last line published
        staging.parent.rmdir()
    except OSError:
        pass


def part_record(root: Path, tag: str, item: PlanItem) -> dict:
    """The part's record, refreshed with the source identity seen right now.

    Loaded rather than recreated so a build of one line never drops what the
    other line recorded.  The *part-level* fingerprint tracks the STEP as it is
    today - which is what the light reads; each line block keeps the sha it was
    actually built from, which is what decides whether that line is behind.
    """
    step_path = item.stepPath
    assert step_path is not None
    record = load_record(record_path(root, tag, item.partId))
    if record is None:
        record = new_record(
            part_id=item.partId,
            source_step=project_relative(root, step_path),
            source_step_abs=str(step_path.resolve()),
            fingerprint=stamping.fingerprint(step_path),
        )
    else:
        record["sourceStep"] = project_relative(root, step_path)
        record["sourceStepAbs"] = str(step_path.resolve())
        record["fingerprint"] = stamping.fingerprint(step_path)
    return record


def _rebuild_part(root: Path, tag: str, item: PlanItem, line: str, bundle: dict | None,
                  build_viewer: bool, stamps: dict, job_id: str) -> bool:
    from core.orchestrator import _process_one

    step_path = item.stepPath
    assert step_path is not None
    staging, paths = _staging_paths(root, tag, item.partId, line)
    live_root = part_paths(root, tag, item.partId, line)["root"]
    record = part_record(root, tag, item)
    fingerprint = record["fingerprint"]
    try:
        entry = _process_one(root, tag, step_path, item.partId, bundle, build_viewer, paths=paths)
        viewer_entry = entry.pop("_viewer", None)
        heuristic_summary = entry.pop("heuristicSummary", {})
        if viewer_entry is not None:
            viewer_entry["heuristicCounts"] = entry.get("heuristicCounts", {})
            viewer_entry["heuristicSummary"] = heuristic_summary
        _publish(staging, live_root)
        set_line(record, line, line_block(
            status="PASS",
            stamps={layer: stamps[layer] for layer in stamping.LINE_LAYERS[line]},
            fingerprint_sha=fingerprint["sha256"],
            device=bundle.get("device") if bundle else None,
            job_id=job_id,
            entry=entry,
            viewer_entry=viewer_entry,
            heuristic_summary=heuristic_summary,
            heuristic_counts=entry.get("heuristicCounts", {}),
            artifact_digest=stamping.artifact_digest(live_root),
        ))
        save_record(record_path(root, tag, item.partId), record)
        verdicts = entry.get("heuristicCounts", {})
        print(
            f"PASS {line} {item.partId}: {entry['nodeCount']} nodes, {entry['edgeCount']} edges, "
            f"H {verdicts.get('PASS', 0)} met / {verdicts.get('FAIL', 0)} not met / "
            f"{verdicts.get('NOT_EVALUABLE', 0)} not evaluable"
        )
        return True
    except Exception as exc:  # noqa: BLE001 - one part must not stop the corpus
        shutil.rmtree(staging, ignore_errors=True)
        set_line(record, line, line_block(
            status="FAIL",
            stamps={layer: stamps[layer] for layer in stamping.LINE_LAYERS[line]},
            fingerprint_sha=fingerprint["sha256"],
            device=bundle.get("device") if bundle else None,
            job_id=job_id,
            error=str(exc),
            traceback=traceback.format_exc(),
        ))
        live_root.mkdir(parents=True, exist_ok=True)
        save_record(record_path(root, tag, item.partId), record)
        print(f"FAIL {line} {item.partId}: {type(exc).__name__}: {exc}")
        return False


# ----------------------------------------------------- heuristics-only path

def heuristic_dialect(line: str):
    """``(registry, context factory)`` for the rules a line's nodes can answer.

    A dialect is one ontology's rule set, and a line's nodes are named in one
    ontology's vocabulary, so this is where the two meet.  It is a composition
    root's job: neither the heuristic machinery nor a rule module knows which
    kernel line produced the nodes it is looking at.
    """
    if line == LINE_NX:
        from heuristics.nx._shared import NxHeuristicContext
        from heuristics.nx.registry import NxHeuristicRegistry

        return NxHeuristicRegistry, NxHeuristicContext
    from heuristics._shared import HeuristicContext
    from heuristics.v51.registry import HeuristicRegistry

    return HeuristicRegistry, HeuristicContext


def recompute_heuristics(project_root: str | Path, output_tag: str, part_id: str,
                         line: str = LINE_OCC) -> tuple[dict, dict]:
    """Re-run the heuristic layer on the stored inputs of one part.

    Reads ``partmodel.json`` (back into a ``PartModel``), ``result.json``
    (nodes, edges and the relation layer's diagnostics) and writes a fresh
    ``heuristics.json``.  Returns ``(payload, summary)`` exactly like
    :func:`heuristics.run_heuristics`.

    A line that evaluates no rule says so in its own ``result.json``
    (``diagnostics.notEvaluated.heuristics``), and this re-emits that same
    named blank instead of running the rules.  The check is on what the stored
    result states, not on which line it came from: the rules are written
    against v5.1 ontology attributes, and running them over a result whose
    attributes are a vendor's own names would answer a question nobody asked.
    """
    from contract.partmodel_json import from_json
    from core.models import FeatureEdge, FeatureNode
    from heuristics import run_heuristics

    paths = part_paths(project_root, output_tag, part_id, line)
    model = from_json(paths["partmodel"].read_text(encoding="utf-8"))
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    reason = ((result.get("diagnostics") or {}).get("notEvaluated") or {}).get("heuristics")
    if reason:
        from heuristics import not_evaluated

        payload, summary = not_evaluated(reason, node_count=len(result.get("nodes") or []))
        atomic_write_text(
            paths["heuristics"],
            json.dumps({"partId": part_id, **payload}, indent=1, ensure_ascii=False),
        )
        return payload, summary
    nodes = [FeatureNode.from_dict(node) for node in result.get("nodes", [])]
    edges = [FeatureEdge.from_dict(edge) for edge in result.get("edges", [])]
    relation_diagnostics = (result.get("diagnostics") or {}).get("relations") or {}
    registry, context_factory = heuristic_dialect(line)
    payload, summary = run_heuristics(
        model, nodes, edges, relation_diagnostics=relation_diagnostics,
        registry=registry, context_factory=context_factory)
    atomic_write_text(
        paths["heuristics"],
        json.dumps({"partId": part_id, **payload}, indent=1, ensure_ascii=False),
    )
    return payload, summary


def _recompute_part_heuristics(root: Path, tag: str, item: PlanItem, line: str,
                               stamps: dict, job_id: str) -> bool:
    from heuristics import outcome_counts

    path = record_path(root, tag, item.partId)
    record = load_record(path) or item.record
    block = get_line(record, line) or {}
    try:
        _, summary = recompute_heuristics(root, tag, item.partId, line)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {line} {item.partId} (heuristics only): {type(exc).__name__}: {exc}")
        if record is not None:
            block["status"] = "FAIL"
            block["error"] = f"heuristics recompute: {exc}"
            block["traceback"] = traceback.format_exc()
            block["jobId"] = job_id
            set_line(record, line, block)
            save_record(path, record)
        return False
    counts = outcome_counts(summary)
    block["heuristicSummary"] = summary
    block["heuristicCounts"] = counts
    if isinstance(block.get("entry"), dict):
        block["entry"]["heuristicCounts"] = counts
    if isinstance(block.get("viewerEntry"), dict):
        block["viewerEntry"]["heuristicCounts"] = counts
        block["viewerEntry"]["heuristicSummary"] = summary
    # Refresh every partial layer this line reads, not one named layer:
    # after the rules were split by ontology there is more than one, and
    # leaving a sibling behind would make the part look stale forever.
    for layer in stamping.LINE_LAYERS[line] & stamping.PARTIAL_LAYERS:
        block.setdefault("stamps", {})[layer] = stamps[layer]
    block["producedAt"] = utc_now()
    block["jobId"] = job_id
    block["artifactDigest"] = stamping.artifact_digest(
        part_paths(root, tag, item.partId, line)["root"])
    set_line(record, line, block)
    save_record(path, record)
    print(
        f"HEUR {line} {item.partId}: H {counts.get('PASS', 0)} met / {counts.get('FAIL', 0)} not met / "
        f"{counts.get('NOT_EVALUABLE', 0)} not evaluable (heuristics only)"
    )
    return True


# ---------------------------------------------------------- store actions

def _remove_part_dir(root: Path, tag: str, part_id: str) -> None:
    target = part_dir(root, tag, part_id)
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    staging = run_paths(root, tag)["run_root"] / STAGING_DIR / part_id
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)


def remove_part(project_root: str | Path, output_tag: str, part_id: str, *, delete_source: bool = False) -> dict:
    """Drop one part from the store (and optionally its STEP), rebuild indexes."""
    root = Path(project_root).resolve()
    record = load_record(record_path(root, output_tag, part_id))
    existed = part_dir(root, output_tag, part_id).exists()
    _remove_part_dir(root, output_tag, part_id)
    source_removed = None
    if delete_source and record and record.get("sourceStepAbs"):
        source = Path(record["sourceStepAbs"])
        if source.is_file():
            source.unlink()
            source_removed = str(source)
    rebuild_index(root, output_tag)
    return {"partId": part_id, "removed": existed, "sourceRemoved": source_removed}


def add_parts(
    project_root: str | Path,
    step_files: list[str | Path],
    *,
    input_path: str | Path = "inputs",
    name: str | None = None,
    overwrite: bool = False,
) -> list[tuple[str, Path]]:
    """Copy STEP files into the input directory; return ``(partId, path)`` pairs.

    The sync itself is the caller's job (``plan(..., only=ids)``), so the
    server can do the copy in-process and the rebuild in a subprocess.
    """
    root = Path(project_root).resolve()
    input_abs = resolve_input_path(input_path, root)
    input_abs.mkdir(parents=True, exist_ok=True)
    if name and len(step_files) != 1:
        raise ValueError("--name applies to exactly one file")
    added: list[tuple[str, Path]] = []
    for source in step_files:
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(str(source))
        if source.suffix.lower() not in {".step", ".stp"}:
            raise ValueError(f"not a STEP file: {source}")
        target_name = name or source.name
        if not Path(target_name).suffix:
            target_name += source.suffix
        target = input_abs / target_name
        if target.exists() and not overwrite:
            raise FileExistsError(str(target))
        tmp = target.with_name(target.name + ".tmp")
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)
        added.append((make_part_id(target, input_abs), target))
    return added


def line_row(record: dict | None, line: str, stamps: dict[str, str]) -> dict:
    """One line's state for one part, as the page reads it.

    ``status`` is what the line last did, ``behind`` is whether its code moved
    since - kept apart because a line can be perfectly built and still owe a
    rebuild, and the page says those two things in different places.
    """
    block = ((record or {}).get("lines") or {}).get(line)
    if not isinstance(block, dict):
        return {"status": None, "reason": "never built"}
    stale = stamping.stale_layers(block, line, stamps)
    row = {
        "status": block.get("status"),
        "producedAt": block.get("producedAt"),
        "behind": stale,
        "nodeCount": (block.get("entry") or {}).get("nodeCount"),
        "edgeCount": (block.get("entry") or {}).get("edgeCount"),
        "featureCounts": (block.get("entry") or {}).get("featureCounts"),
        "heuristicCounts": block.get("heuristicCounts") or {},
    }
    for key in ("reason", "error"):
        if block.get(key):
            row[key] = block[key]
    return row


def store_status(project_root: str | Path, output_tag: str = "latest", input_path: str | None = None) -> list[dict]:
    """One row per part: the light, both lines, plus new inputs and untracked
    directories.  Shared by ``store status`` and ``GET /api/parts``."""
    root = Path(project_root).resolve()
    run = run_paths(root, output_tag)
    records = records_by_id(root, output_tag)
    manifest_input = None
    try:
        manifest_input = json.loads(run["manifest"].read_text(encoding="utf-8")).get("inputPath")
    except (OSError, ValueError):
        pass
    source = input_path or manifest_input
    steps: dict[str, Path] = {}
    if source:
        input_abs = resolve_input_path(source, root)
        input_root = input_abs if input_abs.is_dir() else input_abs.parent
        steps = {make_part_id(path, input_root): path for path in collect_step_files(input_abs)}
    stamps = stamping.layer_stamps(root)
    rows: list[dict] = []
    for part_id in sorted(set(records) | set(steps)):
        record = records.get(part_id)
        verdict = stamping.light(record, steps.get(part_id))
        fingerprint = (record or {}).get("fingerprint") or {}
        lines = {line: line_row(record, line, stamps) for line in LINE_ORDER}
        occ = lines[LINE_OCC]
        rows.append({
            "partId": part_id,
            "status": occ.get("status"),
            "fingerprint": (fingerprint.get("sha256") or "")[:8],
            "producedAt": occ.get("producedAt"),
            "sourceStep": (record or {}).get("sourceStep") or (project_relative(root, steps[part_id]) if part_id in steps else None),
            "error": occ.get("error"),
            "light": {
                "state": verdict.state,
                "reasons": verdict.reasons,
            },
            # The layers this part is behind on: what turns the lights into
            # the "rules moved" colour and feeds the run-level banner.
            "rulesChanged": stamping.rules_changed(record, stamps),
            "lines": lines,
        })
    for name in untracked_part_dirs(run["run_root"]):
        if name in steps:
            continue
        rows.append({
            "partId": name,
            "status": None,
            "fingerprint": "",
            "producedAt": None,
            "sourceStep": None,
            "error": None,
            "light": {
                "state": "untracked",
                "reasons": ["part directory without record or source"],
            },
            "rulesChanged": [],
            "lines": {line: {"status": None, "reason": "never built"} for line in LINE_ORDER},
        })
    return sorted(rows, key=lambda row: row["partId"].lower())
