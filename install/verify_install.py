#!/usr/bin/env python
"""Did this installation compute the right answer, or merely run?

Those are different questions, and only the second one is easy.  A subtly wrong
environment - a mismatched dgl build, weights that silently failed to load, a
different OCCT - still recognises planes and through-holes correctly.  What it
degrades is the delicate end: chamfers, fillets, the compound-hole grammar.  So
"no exception was raised" is not evidence of anything, and this script does not
check for it.

Instead it compares a completed run against a recorded expectation: how many
nodes and edges each line produced, the histogram of feature types, and the
histogram of heuristic outcomes.  The feature histogram is the part that earns
its keep, because that is where a bad install shows up first.

    python verify_install.py                  check runs/latest against the expectation
    python verify_install.py --tag mytag      check a different store
    python verify_install.py --record         write the expectation from the current run

``--record`` is for whoever prepares a release, not for whoever deploys one.
If a check fails, recording over it is exactly the wrong move: the mismatch is
the finding.

The store this reads is the repository the script sits in - ``install/`` in a
delivery, so one level up.  In the *development* repository the same file lives
at ``release/install/``, where one level up is ``release/`` and there is no
store; ``--root`` is how recording a fresh expectation is done from there:

    python release/install/verify_install.py --record --root . --part <id>

The NX line is optional throughout.  A part whose NX block says NOT_BUILT is
reported as skipped with the reason, and the verdict still passes - that is the
designed degradation, not a failure.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

# This file sits in install/, so the repository root - and the store under it -
# is one level up.  The expectation stays next to the script that writes it,
# which is right in both repositories; only the store's root differs, and
# ``--root`` says so explicitly rather than the script guessing from what it
# finds on disk.
DEFAULT_ROOT = Path(__file__).resolve().parents[1]
EXPECTATION = Path(__file__).resolve().parent / "expected_install.json"


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except ValueError as error:
        raise SystemExit(f"{path} is not valid JSON: {error}")


def observe(root: Path, tag: str, part_id: str, line: str) -> dict | None:
    """The four numbers for one line of one part, or None if it was not built."""

    line_dir = root / "runs" / tag / "parts" / part_id / line
    result = _load(line_dir / "result.json")
    if result is None:
        return None
    heuristics = _load(line_dir / "heuristics.json") or {}
    records = heuristics.get("records") or []
    return {
        "nodes": len(result.get("nodes") or []),
        "edges": len(result.get("edges") or []),
        "featureTypes": dict(collections.Counter(
            node.get("featureType") for node in result.get("nodes") or [])),
        "heuristicRecords": len(records),
        "outcomes": dict(collections.Counter(
            record.get("outcome") for record in records)),
    }


def not_built_reason(root: Path, tag: str, part_id: str, line: str) -> str | None:
    record = _load(root / "runs" / tag / "parts" / part_id / "record.json") or {}
    block = (record.get("lines") or {}).get(line) or {}
    if block.get("status") == "NOT_BUILT":
        return block.get("reason") or "no reason recorded"
    return None


def compare(expected: dict, actual: dict) -> list[str]:
    """Every difference, named.  One line each, most important first."""

    problems = []
    for key in ("nodes", "edges", "heuristicRecords"):
        if expected.get(key) != actual.get(key):
            problems.append(f"{key}: expected {expected.get(key)}, got {actual.get(key)}")
    for key in ("featureTypes", "outcomes"):
        want, have = expected.get(key) or {}, actual.get(key) or {}
        for name in sorted(set(want) | set(have)):
            if want.get(name, 0) != have.get(name, 0):
                problems.append(
                    f"{key[:-1]} {name}: expected {want.get(name, 0)}, got {have.get(name, 0)}")
    return problems


def do_record(root: Path, tag: str, only: list[str] | None = None) -> int:
    parts_dir = root / "runs" / tag / "parts"
    if not parts_dir.is_dir():
        raise SystemExit(f"no store at {parts_dir} - run refresh first")
    if only:
        missing = [p for p in only if not (parts_dir / p).is_dir()]
        if missing:
            raise SystemExit(f"not in the store: {', '.join(missing)}")
    parts: dict = {}
    for part_dir in sorted(p for p in parts_dir.iterdir() if p.is_dir()):
        if only and part_dir.name not in only:
            continue
        lines = {}
        for line in ("occ", "nx"):
            seen = observe(root, tag, part_dir.name, line)
            if seen is not None:
                lines[line] = seen
        if lines:
            parts[part_dir.name] = lines
    if not parts:
        raise SystemExit(f"{parts_dir} holds no built parts")
    payload = {
        "_comment": "Recorded by verify_install.py --record. Regenerate rather "
                    "than edit: these numbers are an observation, not a setting.",
        "tag": tag,
        "parts": parts,
    }
    # newline="\n" so regenerating this on another platform rewrites its content
    # rather than every line of it.  It is regenerated often; its diff should be
    # the numbers that moved.
    with open(EXPECTATION, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    total = sum(len(v) for v in parts.values())
    print(f"recorded {len(parts)} part(s), {total} line(s) -> {EXPECTATION.name}")
    for part_id, lines in parts.items():
        for line, seen in lines.items():
            print(f"  {part_id} [{line}]  {seen['nodes']} nodes  {seen['edges']} edges "
                  f"  {seen['heuristicRecords']} heuristic records")
    return 0


def do_check(root: Path, tag: str) -> int:
    expectation = _load(EXPECTATION)
    if expectation is None:
        print(f"FAILED: {EXPECTATION.name} is missing.  This release was packaged "
              "without its expected result; report this rather than working around it.")
        return 2
    parts = expectation.get("parts") or {}
    if not parts:
        print(f"FAILED: {EXPECTATION.name} records no parts.")
        return 2

    # Nothing has been built yet.  That is not a mismatch, and saying so in the
    # words of one would send someone to report a defect that is not there.
    store = root / "runs" / tag
    if not (store / "parts").is_dir():
        print(f"NOT CHECKED: there is no store at {store.as_posix()}.")
        print("Run refresh first - see step 3 of install/SETUP.md - then run this again.")
        return 3

    failures, skipped, checked = [], [], 0
    for part_id in sorted(parts):
        for line in sorted(parts[part_id]):
            expected = parts[part_id][line]
            actual = observe(root, tag, part_id, line)
            label = f"{part_id} [{line}]"
            if actual is None:
                reason = not_built_reason(root, tag, part_id, line)
                if line == "nx" and reason:
                    skipped.append(f"{label}: NX line not built - {reason}")
                    continue
                failures.append(f"{label}: nothing was produced for this line")
                continue
            problems = compare(expected, actual)
            checked += 1
            if problems:
                failures.append(label + "\n" + "\n".join("      " + p for p in problems))
            else:
                print(f"  ok   {label}  {actual['nodes']} nodes  {actual['edges']} edges "
                      f"  {actual['heuristicRecords']} heuristic records")

    for note in skipped:
        print(f"  skip {note}")

    print()
    if failures:
        print(f"MISMATCH in {len(failures)} of {checked} checked line(s):")
        for failure in failures:
            print("  - " + failure)
        print()
        print("Do not adjust anything to make this pass.  The mismatch is the finding:")
        print("send this whole output to whoever gave you the repository.")
        return 1
    if not checked:
        print("FAILED: nothing could be checked.  Did refresh run?")
        return 2
    print(f"INSTALL VERIFIED - {checked} line(s) match the expected result"
          + (f", {len(skipped)} skipped" if skipped else "") + ".")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tag", default="latest", help="store tag under runs/ (default: latest)")
    parser.add_argument("--record", action="store_true",
                        help="write the expectation from the current run (release preparation)")
    parser.add_argument("--part", action="append", metavar="ID",
                        help="with --record, record only this part (repeatable); a "
                             "release's expectation covers its shipped fixture, not the "
                             "whole development corpus")
    parser.add_argument("--root", default=None, metavar="DIR",
                        help="repository holding runs/<tag> (default: the directory "
                             "above this script; pass '.' when running this from the "
                             "development repository, where it lives one level deeper)")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else DEFAULT_ROOT
    if args.record:
        return do_record(root, args.tag, args.part)
    if args.part:
        parser.error("--part only applies with --record")
    return do_check(root, args.tag)


if __name__ == "__main__":
    sys.exit(main())
