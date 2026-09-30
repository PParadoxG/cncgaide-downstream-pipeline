"""CLI for the v8 explicit two-hole migration pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from contract.registry import LINE_OCC, LINES  # noqa: E402  (needs the path above)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V8 explicit CAD feature pipeline")
    commands = parser.add_subparsers(dest="command")

    refresh = commands.add_parser(
        "refresh",
        help="synchronise runs/<tag> with the input directory: new/changed parts are processed, the rest kept",
    )
    refresh.add_argument("--input", default="inputs")
    refresh.add_argument("--output-tag", default="latest")
    refresh.add_argument("--limit", type=int, default=0)
    refresh.add_argument("--device", default="auto")
    refresh.add_argument("--no-viewer", action="store_true", help="skip the viewer mesh/view artifacts")
    refresh.add_argument("--all", action="store_true", help="reprocess every part, not only the stale ones")
    refresh.add_argument("--only", nargs="+", metavar="ID", help="restrict the sync to these part ids")
    refresh.add_argument("--keep-missing", action="store_true",
                         help="keep parts whose source STEP is gone (they are dropped by default)")
    refresh.add_argument("--no-nx", action="store_true",
                         help="build the open-source line only, leaving the NX line as it is")
    refresh.add_argument("--no-aagnet", action="store_true",
                         help="build the open-source line without AAGNet: the model is not "
                              "loaded and the evidence document records that it was disabled")
    refresh.add_argument("--dry-run", action="store_true", help="print the plan and exit")

    store = commands.add_parser("store", help="inspect and maintain the parts store (runs/<tag>)")
    store_commands = store.add_subparsers(dest="store_command")

    status = store_commands.add_parser("status", help="one line per part: the light, then each line's state")
    status.add_argument("--output-tag", default="latest")
    status.add_argument("--input", default=None, help="input directory (default: the one the run was made from)")
    status.add_argument("--json", action="store_true")

    add = store_commands.add_parser("add", help="copy STEP files into the input directory and build them")
    add.add_argument("step", nargs="+", help="STEP file(s) to add")
    add.add_argument("--name", default=None, help="file name inside the input directory (one file only)")
    add.add_argument("--overwrite", action="store_true")
    add.add_argument("--input", default="inputs")
    add.add_argument("--output-tag", default="latest")
    add.add_argument("--device", default="auto")
    add.add_argument("--no-viewer", action="store_true")

    remove = store_commands.add_parser("remove", help="drop a part from the store")
    remove.add_argument("part_id")
    remove.add_argument("--delete-source", action="store_true", help="also delete its STEP file")
    remove.add_argument("--output-tag", default="latest")

    reindex = store_commands.add_parser("rebuild-index", help="rebuild manifest.json and viewer_index.json from the records")
    reindex.add_argument("--output-tag", default="latest")

    snap = store_commands.add_parser("snapshot", help="copy a run to a new tag as a frozen snapshot")
    snap.add_argument("--to", required=True, dest="to_tag")
    snap.add_argument("--from", default="latest", dest="from_tag")

    view = commands.add_parser("view", help="serve the web viewer for a finished run")
    view.add_argument("--output-tag", default="latest")
    view.add_argument("--input", default="inputs", help="input directory the page's refresh/upload act on")
    view.add_argument("--port", type=_port, default=8765, help="port, or 'auto' for any free one")
    view.add_argument("--no-browser", action="store_true")
    view.add_argument("--no-refresh-api", action="store_true", help="serve read-only, no refresh button")
    view.add_argument("--list", action="store_true", help="list the viewers that are running, then exit")
    view.add_argument("--stop", action="store_true", help="stop the viewer on --port (or --all), then exit")
    view.add_argument("--all", action="store_true", help="with --stop: stop every viewer found")
    view.add_argument("--replace", action="store_true", help="stop the viewer already on --port and take over")
    view.add_argument(
        "--idle-timeout",
        type=float,
        default=None,
        help="stop after this many idle seconds; defaults to 1800 with --no-browser, 0 otherwise",
    )

    ged = commands.add_parser("ged", help="feature-graph edit distance over a finished run")
    ged_commands = ged.add_subparsers(dest="ged_command")

    def cost_options(command) -> None:
        command.add_argument("--config", default=None, metavar="NAME",
                             help="cost preset (see `ged configs`; default v0_default)")
        command.add_argument("--weights", default=None, metavar="K=V,...",
                             help="override single weights on top of the preset, e.g. W_REL=0.8,C_DEL=1 (or a JSON object)")
        command.add_argument("--no-local", action="store_true", help="alias for --config v0_no_local")
        # Which kernel line's corpus to search.  Not a filter on one corpus:
        # each line's parts are written in their own ontology and priced with
        # their own tables, so this picks a whole scale.  Distances from two
        # lines are not comparable and are never mixed in one answer.
        command.add_argument("--line", default=LINE_OCC, choices=sorted(LINES),
                             help=f"kernel line whose corpus to use (default {LINE_OCC})")

    configs_cmd = ged_commands.add_parser("configs", help="list the cost presets with their hashes")
    configs_cmd.add_argument("--json", action="store_true")

    compare = ged_commands.add_parser("compare", help="distance between two parts of one run")
    compare.add_argument("part_a")
    compare.add_argument("part_b")
    compare.add_argument("--output-tag", default="latest")
    compare.add_argument("--edit-path", action="store_true", help="print the full edit path as JSON")
    cost_options(compare)

    rank_cmd = ged_commands.add_parser("rank", help="rank every part of a run against a query part")
    rank_cmd.add_argument("--query", required=True)
    rank_cmd.add_argument("--output-tag", default="latest")
    rank_cmd.add_argument("--top", type=int, default=0, help="show only the best N (0 = all)")
    rank_cmd.add_argument("--prefilter-keep", type=int, default=0, help="run GED only on the N best pre-filter candidates (0 = all)")
    rank_cmd.add_argument("--candidates", nargs="+", metavar="ID", default=None, help="restrict the candidates to these parts")
    rank_cmd.add_argument("--json", action="store_true", help="print exactly what GET /api/similarity/rank returns")
    cost_options(rank_cmd)

    eval_cmd = ged_commands.add_parser("eval-variants", help="within-family ranking / pose / self-retrieval report")
    eval_cmd.add_argument("--output-tag", default="variants")
    # One report per line: the two are written in different ontologies and
    # priced on different scales, so they must not overwrite each other.
    # A JSON twin is written beside whatever path this resolves to.
    eval_cmd.add_argument("--report", default=None,
                          help="report path; default runs/<tag>/ged/GED_VARIANT_REPORT_<line>.md")
    cost_options(eval_cmd)
    return parser


def _port(value: str) -> int:
    """``--port auto`` picks a free one; the default still fails loudly."""
    return 0 if str(value).lower() in {"auto", "0"} else int(value)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "refresh":
        return _run_refresh(args)

    if args.command == "store":
        return _run_store(args)

    if args.command == "view":
        from visualization import instances
        from visualization.server import serve

        if args.list or args.stop:
            return _manage_viewers(args, instances)

        return serve(
            PROJECT_ROOT,
            output_tag=args.output_tag,
            port=args.port,
            open_browser=not args.no_browser,
            allow_refresh=not args.no_refresh_api,
            replace=args.replace,
            idle_timeout=args.idle_timeout,
            input_path=args.input,
        )

    if args.command == "ged":
        return _run_ged(args)

    build_parser().print_help()
    return 0


def _run_refresh(args) -> int:
    from contract.registry import LINE_OCC, LINE_ORDER
    from store import stamps, sync

    # With the model off the plan must ask for the device the build will
    # record, not for the one it would have inferred on: that is what makes a
    # part built without AAGNet out of date the moment the model comes back,
    # and a part built with it out of date the moment it goes away.
    device = stamps.AAGNET_DISABLED if args.no_aagnet else args.device
    try:
        plan = sync.plan(
            PROJECT_ROOT,
            args.input,
            args.output_tag,
            only=args.only,
            mode="all" if args.all else "stale",
            limit=args.limit,
            build_viewer=not args.no_viewer,
            lines=(LINE_OCC,) if args.no_nx else LINE_ORDER,
            device=device,
        )
    except ValueError as error:
        print(f"[refresh] {error}")
        return 2
    if args.dry_run:
        print(plan.describe())
        return 0
    summary = sync.execute(
        plan, device=device, build_viewer=not args.no_viewer,
        keep_missing=args.keep_missing, aagnet=not args.no_aagnet,
    )
    manifest = summary["manifest"]
    # One line on purpose: it is the last thing the page's job bar shows.
    print(json.dumps({
        "processed": len(summary["processed"]),
        "failedNow": summary["failedParts"],
        "passed": sum(item["status"] == "PASS" for item in manifest["statuses"]),
        "failed": sum(item["status"] == "FAIL" for item in manifest["statuses"]),
    }))
    # The exit code speaks about this sync: a part that failed long ago and
    # was deliberately not retried does not make today's run a failure.
    return 1 if summary["failedParts"] else 0


def _run_store(args) -> int:
    from store import sync

    if args.store_command == "status":
        rows = sync.store_status(PROJECT_ROOT, args.output_tag, args.input)
        if args.json:
            print(json.dumps(rows, indent=2, ensure_ascii=False))
            return 0
        if not rows:
            print(f"[store] runs/{args.output_tag}: no parts")
            return 0
        width = max(len(row["partId"]) for row in rows)
        for row in rows:
            light = row["light"]["state"]
            if row.get("rulesChanged"):
                light += " [rules: " + ",".join(row["rulesChanged"]) + "]"
            lines = "  ".join(
                f"{name}={(state.get('status') or 'none').lower()}"
                + (f"({state['reason']})" if state.get("reason") else "")
                # A line whose record says PASS while its files are gone reads
                # as PASS everywhere else; this is where it stops doing that.
                + (f"(artifacts missing: {','.join(state['artifactsMissing'])})"
                   if state.get("artifactsMissing") else "")
                for name, state in (row.get("lines") or {}).items()
            )
            print(
                f"{row['partId']:{width}s}  {(row['fingerprint'] or '-'):8s}  {light:34s}  "
                f"{lines:44s}  {row['producedAt'] or '-'}"
            )
        return 0

    if args.store_command == "add":
        try:
            added = sync.add_parts(
                PROJECT_ROOT, args.step, input_path=args.input, name=args.name, overwrite=args.overwrite
            )
        except (OSError, ValueError) as error:
            print(f"[store] {type(error).__name__}: {error}")
            return 2
        ids = [part_id for part_id, _ in added]
        for part_id, path in added:
            print(f"[store] added {path} -> {part_id}")
        plan = sync.plan(PROJECT_ROOT, args.input, args.output_tag, only=ids, build_viewer=not args.no_viewer)
        summary = sync.execute(plan, device=args.device, build_viewer=not args.no_viewer)
        return 1 if summary["failedParts"] else 0

    if args.store_command == "remove":
        outcome = sync.remove_part(PROJECT_ROOT, args.output_tag, args.part_id, delete_source=args.delete_source)
        print(json.dumps(outcome, indent=2, ensure_ascii=False))
        return 0 if outcome["removed"] else 1

    if args.store_command == "rebuild-index":
        from store.index import rebuild_index

        manifest, index = rebuild_index(PROJECT_ROOT, args.output_tag)
        print(
            f"[store] runs/{args.output_tag}: {manifest['runId']}, {len(manifest['parts'])} part(s), "
            f"viewer index {'written' if index is not None else 'not written'}"
        )
        return 0

    if args.store_command == "snapshot":
        from store.snapshot import snapshot

        try:
            target = snapshot(PROJECT_ROOT, args.from_tag, args.to_tag)
        except (OSError, ValueError) as error:
            print(f"[store] {type(error).__name__}: {error}")
            return 2
        print(f"[store] snapshot written to {target}")
        return 0

    # argparse exits from --help, so nothing after this line runs.
    build_parser().parse_args(["store", "--help"])


def _run_ged(args) -> int:
    from ged.contract.representation import for_line
    from ged.cost.presets import WEIGHT_SPECS, cli_weights, list_presets, parse_weights, resolve_config
    from ged.retrieval.service import PartNotFound, ProjectionFailed, SimilarityService

    if args.ged_command == "configs":
        presets = list_presets()
        if args.json:
            print(json.dumps({"configs": presets, "weights": WEIGHT_SPECS}, ensure_ascii=False, indent=2))
            return 0
        for preset in presets:
            print(f"{preset['name']:16s} {preset['hash']}  {cli_weights(preset['weights'])}")
        print("weights: " + ", ".join(f"{key} [{spec['min']:g}..{spec['max']:g}]" for key, spec in WEIGHT_SPECS.items()))
        return 0

    if args.ged_command not in ("compare", "rank", "eval-variants"):
        build_parser().parse_args(["ged", "--help"])
        return 0

    run_root = PROJECT_ROOT / "runs" / args.output_tag
    if not run_root.is_dir():
        print(f"[ged] no run at {run_root}; refresh with --output-tag {args.output_tag} first")
        return 2
    config_name = args.config or ("v0_no_local" if args.no_local else None)
    try:
        weights = parse_weights(args.weights)
        config, info = resolve_config(config_name, weights,
                                      representation=for_line(args.line))
    except (KeyError, ValueError) as error:
        print(f"[ged] {error}")
        return 2
    service = SimilarityService(run_root, line=args.line)

    if args.ged_command == "compare":
        try:
            payload = service.compare(args.part_a, args.part_b, config_name=config_name, weights=weights)
        except (PartNotFound, ProjectionFailed) as error:
            print(f"[ged] not loaded: {error}")
            return 2
        if not args.edit_path:
            payload.pop("edit_path", None)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.ged_command == "rank":
        try:
            payload = service.rank(
                args.query,
                config_name=config_name,
                weights=weights,
                candidate_ids=args.candidates,
                top=args.top or None,
                prefilter_keep=args.prefilter_keep or None,
            )
        except (PartNotFound, ProjectionFailed) as error:
            print(f"[ged] query {args.query} not in run ({error})")
            return 2
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        corpus = payload["corpus"]
        cfg = payload["config"]
        print(
            f"[ged] query {args.query} on line {payload['line']} ({payload['representation']}): "
            f"{payload['candidates']} candidates ({corpus['loaded']} loaded, "
            f"{len(corpus['failed'])} failed)  config {cfg['name']} #{cfg['hash']}  {payload['timingMs']:.0f} ms"
        )
        for row in payload["ranked"]:
            b = row["breakdown"]
            cached = "  (cached)" if row["cached"] else ""
            print(
                f"{row['rank']:3d}. {row['partId']:50s} d={row['distance']:.6f} "
                f"sim={row['similarity']:7.3f}%  node={b['node_cost']:.3f} edge={b['edge_cost']:.3f} "
                f"fixed={b['fixed_cost']:.4f} scale={b['scale_cost']:.4f}  prefilter={row['prefilterScore']:.3f}{cached}"
            )
        for part_id, reason in corpus["failed"].items():
            print(f"     failed: {part_id}: {reason}")
        return 0

    if args.ged_command == "eval-variants":
        from ged.eval.variants import evaluate, write_payload, write_report
        from ged.retrieval.corpus import load_corpus

        corpus = load_corpus(run_root, line=args.line)
        if corpus.failed:
            print(f"[ged] {len(corpus.failed)} part(s) failed to project: {corpus.failed}")
        evaluation = evaluate(corpus, config=config)
        report = Path(args.report) if args.report else (
            run_root / "ged" / f"GED_VARIANT_REPORT_{args.line}.md")
        write_report(evaluation, report)
        # The machine-readable twin, beside the markdown: node 31 reads the
        # numbers back rather than parsing tables out of prose.
        payload_path = write_payload(evaluation, report.with_suffix(".json"))
        print(f"[ged] config {info['name']} #{info['hash']} semantics {evaluation.semantics}")
        for fam in evaluation.families:
            status = "PASS" if fam.passed else "FAIL"
            detail = ""
            if fam.inversions:
                detail = " inversions: " + "; ".join(
                    f"{axis} {lo} > {hi}" for axis, lo, hi in fam.inversions)
            if fam.pose_failures:
                detail += " pose failures: " + ", ".join(fam.pose_failures)
            if fam.pose_drift:
                detail += " identification drift: " + ", ".join(fam.pose_drift)
            print(f"[ged] {fam.family} {status} ({len(fam.rows)} members){detail}")
        if evaluation.nearest_neighbour_failures:
            print(f"[ged] self-retrieval failures: {len(evaluation.nearest_neighbour_failures)}")
            for failure in evaluation.nearest_neighbour_failures:
                # ASCII arrow: this line goes to a console that may be cp936.
                print(f"[ged]   {failure['member']} -> {failure['nearest']} "
                      f"({failure['distance']:.6f}); expected {failure['expected']}")
        if evaluation.outcome == "NOT_EVALUABLE":
            # A gate with nothing to judge must not answer "pass" (node 26
            # F-03): a mistyped tag and an unrefreshed variant corpus both
            # land here, and both used to exit 0.
            print(f"[ged] NOT EVALUABLE: no variant family in {run_root} "
                  f"(a family is part ids sharing a V<b>.<axis><level>_ prefix)")
            print(f"[ged] report written to {report} and {payload_path}")
            return 2
        print(f"[ged] {evaluation.outcome}: report written to {report} and {payload_path}")
        return 0 if evaluation.passed else 1

    return 0


def _manage_viewers(args, instances) -> int:
    """``--list`` / ``--stop``: find and stop viewers without hunting for PIDs."""
    running = instances.discover(PROJECT_ROOT, extra_ports=[args.port] if args.port else [])
    if args.list:
        if not running:
            print("[viewer] no viewer is running")
            return 0
        for status in running:
            label = f" label={status['label']}" if status.get("label") else ""
            print(
                f"[viewer] port {status['port']}  pid {status['pid']}  tag={status['outputTag']}"
                f"  started {status['startedAt']}{label}"
            )
            print(f"                 data {status['runRoot']}")
        return 0

    targets = running if args.all else [item for item in running if item["port"] == args.port]
    if not targets:
        scope = "any port" if args.all else f"port {args.port}"
        print(f"[viewer] no viewer to stop on {scope}")
        return 0
    failed = 0
    for status in targets:
        stopped, detail = instances.stop(status["port"])
        print(f"[viewer] port {status['port']} (pid {status['pid']}): {detail}")
        failed += 0 if stopped else 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
