"""The seam layer: what both kernel lines agree on, and nothing else.

``contract`` is the one package that may be imported from either side of the
pipeline without dragging a geometry kernel along.  It holds two things:

* :mod:`contract.partmodel` (+ :mod:`contract.partmodel_json`) — the
  normalized B-Rep *data* every downstream layer reads;
* :mod:`contract.kernel` — the *live geometry queries* that cannot be
  answered from that data and therefore have to stay a service call.

Nothing here imports ``occline.parse``, any shared layer above the seam or ``OCC``: the dependency
runs one way only, from an implementation towards the contract.  A test
enforces it (``tests/test_node10_seams.py``).
"""
