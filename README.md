# CNCgAIde

A CAD feature-recognition pipeline with a web viewer. Drop STEP files in, get
back the machining features the model contains, how they relate to one another,
where they violate manufacturability rules, and which of your other parts each
one resembles.

Not installed yet? → **[install/SETUP.md](install/SETUP.md)**.


## What it finds

For each STEP file:

1. **Features** — holes (through, blind, counterbore, countersunk, stepped),
   slots, pockets, bosses, steps, through cuts, chamfers, fillets, planar and
   freeform faces.
2. **Attributes** — diameter, depth, wall thickness, corner radius, position,
   axis. What cannot be measured is reported as missing *with a reason*, which
   is not the same as not applying.
3. **Relations** — which features are coaxial, which sit on which, which are
   directly adjacent, how holes are spaced.
4. **Manufacturability verdicts** — rules such as "a hole's depth-to-diameter
   ratio should stay below *n*", each landing on met, not met, not applicable,
   or not evaluable.
5. **Similarity** — a graph edit distance over the feature graphs, so "which of
   my parts is this one like" has an answer.

Everything runs locally. Nothing is uploaded.


## Two recognition lines

The same STEP file is read by two independent recognisers, kept apart on
purpose.

**Open source** parses the geometry with OpenCASCADE, gets per-face evidence
from a neural network, and applies this project's own rules. It speaks a
vocabulary built on the ISO 10303-224 feature definitions.

**NX FBM** uses Siemens NX's own Feature-Based Machining recognition and reports
it verbatim — NX's class names, NX's labels, NX's attribute names and values,
nothing re-cut or renamed.

The disagreements are the interesting part. Where one sees a counterbore the
other may see something with a different name and a different parameter set,
and the viewer lets you put them side by side. Switch lines from the control
next to the CNCgAIde logo; the whole view switches with it, including the
accent colour, so you always know which one you are reading.

**The two lines' numbers are not comparable.** The same rule is one statement
measured two ways, so their pass rates cannot be added or averaged, and
similarity distances are never compared across lines.

The NX line is optional — see `install/SETUP.md`. Without it, parts show it as
"not computed" with the reason, and the open-source line is unaffected.


## Two commands

Everything else happens in the browser. `PY` is the environment's interpreter,
as set up in `install/SETUP.md`.

Process what is in `inputs/` — new and changed parts only, the rest untouched:

```
set PYTHONPATH=src && "PY" run_pipeline.py refresh --input inputs --output-tag latest --device cpu
```

Start the viewer at http://localhost:8765 :

```
set PYTHONPATH=src && "PY" run_pipeline.py view --output-tag latest
```

You can add parts and re-run them from the web interface too, so in normal use
the second command is the only one you need. To stop the server:
`"PY" run_pipeline.py view --stop --port 8765`.


## Using the viewer

Three tabs across the top: **Parts**, **Heuristics**, **Similarity**. The small
house at the top left returns to the parts list.

### Parts — the list

One row per part, with a coloured bar showing the proportion of each feature
type (hover it for the breakdown), when it was produced, and a status light.

The light is only ever about **the STEP file**: it stays quiet when nothing has
changed and speaks up for *source changed*, *not built*, *source missing*,
*queued*, *building*. A separate dashed chip, `rules changed`, means the
analysis code moved rather than your file — the part is still valid, just
computed with older rules.

- **Add parts** (top) uploads STEP files. This is the normal way to get work in;
  you do not have to touch `inputs/` by hand.
- **Refresh stale** processes everything the list has flagged; **Refresh all**
  redoes everything.
- Hovering a row reveals **Similar** (ranks the others against it), **Re-run**
  (recomputes just this one) and **Remove** (asks twice — first about the store,
  then about the STEP file itself).
- **Filter by type ▾** narrows the list to parts containing chosen feature types.

While something is running, a task bar shows what it is doing, the tail of the
log, and how many parts are queued. The list updates in place as parts finish.

### A part — three columns

Click a part to open it. **Left** are the filters with the feature graph below
them, **middle** is the 3D model, **right** is the inspector with the rule
verdicts under it. The dividers between the columns can be dragged.

The part dropdown at the top switches parts without going back to the list.

**The 3D model** uses the usual orbit controls:

| | |
| --- | --- |
| Left-drag | rotate |
| Right-drag | pan |
| Scroll wheel | zoom |
| Hover a face | reads out its face id and which feature owns it |
| Click a face | selects it — the graph and the inspector follow |
| Double-click a face | zooms in on it |

Three controls sit in the top bar, each with a keyboard shortcut: **Fit** (`F`)
frames the selection, or the whole part when nothing is selected; **Isolate**
(`I`) hides everything but the selection; **Edges** (`E`) overlays the B-Rep
edges. Selected faces are drawn again as an x-ray highlight, so they stay
visible from behind. **Find similar** ranks the other parts against this one.
There is also a light/dark toggle.

**The feature graph** draws one node per feature and one edge per relation.
The layout dropdown offers **Grouped**, **Force** and **Concentric** — worth
trying all three, they suit different parts. The search box finds a feature by
name.

Above it, the feature-type chips and the relation chips control what is drawn.
An edge appears only when its type is switched on *and* both of its ends are
visible, and the chips say so: hovering one reads, for example, *25
DIRECT_ADJACENT edges in this part, 3 drawable, 22 have an endpoint whose
feature type is hidden*. Some types start switched off deliberately — flat
background faces on one line, NX's notch features on the other — so the graph
opens on the features rather than on the noise. Clicking a node selects that
feature everywhere.

**The inspector** shows the part's totals (CAD faces, features, relations, how
many faces no feature claimed) and, once you select something, that feature's
attributes — including the ones that could not be measured, each with its
reason.

**The verdicts panel** below it counts the part's rules as *not met* / *not
evaluable* / *met*, with the not-applicable ones noted separately. Filter to
**all**, **not met**, or **selection** — the last showing only the rules that
touch what you have selected. Each rule lists one coloured square per unit it
judged, and each square carries the actual comparison: hovering one reads, for
instance, *H19 · not met · derived_ratio_min_corner_radius_to_depth 0.176471 <
t_corner_depth_ratio 0.33*. Clicking a square selects the feature it judged.
The ⋯ menu adds the rules that are waiting on an input this pipeline does not
own, which are hidden by default.

If a part is behind the rest of the store, a banner across the top says so and
offers to re-run or reload it.

### Heuristics — all parts at once

Opens on the headline — what share of the decided verdicts were met, and the
four counts behind it. Below that, a **part × heuristic** matrix in the same
three colours the part page uses, so a rule that fails across the board and a
part that fails everything are both visible at a glance; a fourth shade marks
the cells where a rule produced no unit to judge.

Then the same data as a table: rule, name, outcomes, how many parts it applies
to, and its not-met rate. Clicking through opens the offending part with that
rule already selected. The `external` toggle brings in the rules waiting on an
input the pipeline does not own.

This view follows the line you are on and never mixes the two, because their
rates are not comparable.

### Similarity — which parts resemble which

Choose the query part; candidates are all the others, or a set you pick and can
save under a name. The **Cost weights** panel controls what "similar" means —
four sliders in simple mode (type, dimensions, relations, size), eight if you
open Advanced. Presets are in the dropdown, the current configuration's name and
hash are shown next to it, and the track of each slider marks the preset value
so you can see what you changed. Press **Query** after any change.

Results come back ranked, each with a similarity score, the raw distance, and a
stacked bar splitting the distance into its channels. Click a row to see the
full breakdown; **Compare** opens the two parts side by side.

Distances from different weights are not comparable — that is what the hash next
to the configuration name is for.

### Compare — two parts side by side

Two linked 3D views and the edit path between them: every insertion, deletion
and substitution the distance is made of, grouped and counted, each showing what
it cost and which attribute contributed most. Click an entry to select the
features it concerns in both views; click a face to find the entry that concerns
it. **Link cameras** makes the two views move together, scaled to each part's
size. **Hide micro edits** clears out the negligible ones.


## Where things are

| Path | What |
| --- | --- |
| `inputs/` | Your STEP files |
| `runs/latest/` | The results — one folder per part, one subfolder per line |
| `config/nx_local.json` | Where NX is on this machine (see `install/SETUP.md`) |
| `ontology/*/source/` | The vocabularies: the classes, attributes, relations and rules the pipeline is defined by |
| `ontology/cncgaide_*_structure_*.html` | The definition pages — where the two vocabularies are stated |
| `install/` | How to build the environment and how to check one: `environment.yml`, its lock file, `SETUP.md`, and `verify_install.py` |

Everything under `runs/` is written by the pipeline and rewritten by the next
refresh.


## Updating

```
git pull
```

Then open the viewer. Parts affected by the update are marked, and the refresh
control on that page brings them up to date — only what actually needs
recomputing is recomputed, and a change confined to the rules is re-evaluated in
seconds without re-reading any geometry.

`git pull` does not update the conda environment. If a release changes
`install/environment.yml` the notes will say so;
`conda env update -f install/environment.yml`
applies it.


## Problems and questions

Open an issue on this repository. Worth including: what you ran, the complete
output, and — if it is about one part — its name and the STEP file if you can
share it. For a part recognised wrongly, the geometry says more than a
description does.


## Third-party code

See [NOTICE](NOTICE).
