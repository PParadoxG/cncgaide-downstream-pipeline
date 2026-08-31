"""Parts store: the ``runs/<tag>`` layout as an incrementally maintained store.

Nothing new is invented on disk beyond one ``record.json`` per part: the
per-part artifacts stay where the pipeline always wrote them, the manifest and
the viewer index are *rebuilt from the records* on every synchronisation, and
the GED and the viewer keep reading the same files they always read.

A part is built once per *line* - one folder each for the open-source line and
the NX line - and the record above them answers two different questions:

* :func:`~store.stamps.light` - has the source STEP changed?  That,
  and only that, is what the page shows next to a part.
* :func:`~store.stamps.plan_line` - what does this sync owe this
  line?  That additionally depends on whether the code the line reads has
  moved, which the page reports once at run level rather than per part.

Any changed layer except ``heuristics``, or a changed source STEP, rebuilds
that line's folder; a change confined to the ``heuristics`` layer recomputes
only the verdicts from the stored ``partmodel.json`` and ``result.json`` - no
STEP parsing, no AAGNet, no NX session, so it costs seconds on either line.
"""

from store.records import (  # noqa: F401
    RECORD_SCHEMA,
    atomic_write_bytes,
    atomic_write_text,
    iter_records,
    load_record,
    record_path,
    save_record,
    untracked_part_dirs,
)
from store.stamps import (  # noqa: F401
    LAYERS,
    LINE_LAYERS,
    Light,
    LineAction,
    artifact_digest,
    fingerprint,
    layer_stamps,
    light,
    plan_line,
    rules_changed,
)
