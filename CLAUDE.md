# CNCgAIde — notes for an agent working here

Deploying this for the first time? [install/SETUP.md](install/SETUP.md) is the
procedure; it
ends with the viewer running and the browser open on it. Follow it rather than
inferring how to install a Python project, and if a step fails, send the output
rather than working around it — the package versions are verified together and
a substituted one can install cleanly and answer wrongly.

[README.md](README.md) says what the system does and how the web interface
works. Almost everything is done there rather than on the command line.


## Running it

Use the environment's interpreter by full path and set `PYTHONPATH=src`; that
avoids the commonest mistake, which is running from a different environment
without noticing.

```
set PYTHONPATH=src && "PY" run_pipeline.py refresh --input inputs --output-tag latest --device cpu
```

`refresh` is incremental: it processes what is new or changed and leaves the
rest alone. To see its intentions without acting on them, add `--dry-run`; it
prints one line per part and one action per recognition line.

The viewer has its own lifecycle — start it through the CLI rather than any
generic server command, and stop it the same way:

```
set PYTHONPATH=src && "PY" run_pipeline.py view --output-tag latest
```

```
set PYTHONPATH=src && "PY" run_pipeline.py view --stop --port 8765
```

If the port is taken it says so instead of silently moving; `view --list` shows
what is running.

Files under `runs/` are written by the pipeline and read back by it to decide
what needs recomputing, so editing one by hand makes the system disagree with
itself. Delete and re-run instead.


## Outputs that look like errors and are not

**`nx: NOT_BUILT`.** The NX recognition line needs a local Siemens NX install
and a reachable licence server. Without either, the part records `NOT_BUILT`
with the reason and is retried on the next refresh without anything having to
change. The open-source results are complete and untouched. This is the designed
behaviour, not a failure.

**`missingAttributes` on a feature.** An attribute that could not be measured is
recorded as null with a named reason. "We could not measure this, and here is
why" is deliberately kept distinct from "this does not apply here".

**A verdict of `NOT_EVALUABLE`.** Same idea: a rule missing its inputs says so
rather than guessing or quietly passing.

**A part disappearing from the list.** Removing a STEP file from `inputs/`
removes the part on the next refresh. That is the sync working.

**The two lines reporting different numbers.** Expected, and the point of the
system — two recognisers with two vocabularies. Their verdict rates are not
comparable and must not be added, averaged or shown side by side without saying
so; similarity distances are likewise never compared across lines.


## How it is put together

- **Two recognition lines**, `occ` (OpenCASCADE + a per-face network + this
  project's rules) and `nx` (Siemens NX FBM reported verbatim). Both fill the
  same neutral data structure, and everything above that boundary is shared.
- **`runs/<tag>` is the store.** One directory per part; inside it one folder
  per line holding files with the same names and the same shape, so reading code
  changes lines by changing a path segment.
- **Rebuilding is decided per line.** Whether a part is redone depends on
  whether its STEP changed and whether the code layers *that line reads*
  changed. A change confined to the rules is re-evaluated in seconds from stored
  results, with no geometry re-parsed.
- **The vocabularies are the definition.** Classes, attributes, relations and
  thresholds all come from `ontology/*/source/*.yaml`; no threshold is written
  as a number in the code. Those files and the code are one version and ship
  together.

To re-check an installation at any time:

```
set PYTHONPATH=src && "PY" install/verify_install.py
```
