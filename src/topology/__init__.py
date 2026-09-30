"""Pure B-Rep topology and vector primitives, above the seam, below the rules.

This package answers questions that have the same answer in every ontology and
on every kernel line: how far a cylindrical wall reaches along its axis, which
edges of a model are geometrically the same curve, where a wall patch stops and
what lies across the boundary, how two direction vectors compare.  Nothing here
names an ontology class, reads AAGNet evidence, or decides whether a candidate
is accepted - those are recognition decisions and they stay in ``features`` (for
v5.1) and in the two relation / heuristic dialects.

Why it is its own package
-------------------------
The functions collected here were written inside ``features/_shared``, which is
the v5.1 rule set.  Two lines read them: the NX line's ``LOCATED_ON`` proof
walks the same rims with the same code.  Importing one of them therefore ran
``features/__init__.py`` and with it every v5.1 leaf module - about thirty
modules - inside NX's bundled Python (Node 26 finding C-10).  Splitting them out
makes that import cost a package boundary rather than an accident, and lets the
layer stamp say the truth: a change here can move either line's results, a
change to a v5.1 leaf class cannot move the NX line's.

Import rule
-----------
Standard library and ``contract`` only.  No kernel line (``occline`` /
``nxline``), no ``OCC``, no ``features``, no ``relations``.  This is guarded by
``tests/test_node10_seams.py``.

Where things live
-----------------
``vectors``      three-component arithmetic and the canonical-form helpers.
``faces``        face identity, ordering, bounding boxes and axial extents.
``signatures``   the geometric edge-identity rule and its per-model views.
``edge_index``   the per-model edge bookkeeping the wall-band walk reads.
``hole_topology``  the cylindrical-wall rim walk and rim-end classification.
"""
