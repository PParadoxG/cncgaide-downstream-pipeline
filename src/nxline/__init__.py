"""The NX kernel line: everything that only exists because NX is available.

This package fills **both** slots of one line: the geometry provider
(``partmodel.py`` fills a ``PartModel``, ``kernel.py`` implements
``KernelServices``) and the recognition provider ``nx_fbm`` — NX's own
Feature Based Machining recognition, read out by ``journal_native.py`` and
published verbatim by ``native.py``.  Because one line owns both slots, face
identity needs no reconciliation: the PartModel faces *are* the NX faces.

Ground rules, guarded by ``tests/test_node10_seams.py``:

* nothing here imports ``OCC`` — the zero-OCCT terminal state is enforced,
  not promised;
* nothing above the seam imports this package; only the composition roots do;
* ``journal_fullnx.py`` runs inside NX's own Python (``run_journal.exe``) and
  is never imported by CPython code.

What NX states, this line publishes: a node's class is NX's ``Type``, its
label is NX's ``TypeDisplayName``, and its attributes are NX's own names and
values — nothing is re-segmented, renamed or re-measured (node 18').  The
quantities the pipeline measures for itself live in each node's separate
``derived`` block, never beside NX's own words (node 20'b).  An unreachable
licence is a named degradation, never a failure: the part keeps whatever it
already had and the next sync retries.
"""
