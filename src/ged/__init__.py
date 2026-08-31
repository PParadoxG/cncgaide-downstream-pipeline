"""GatherSeed feature-graph GED.

Query-time similarity between two parts, computed from the finished pipeline
artifacts (``runs/<tag>/parts/<id>/<line>/result.json`` + ``partmodel.json``).
A line is a whole ruler: which representation reads those artifacts is
``ged.contract.representation.BY_LINE``, and a distance is never compared
across two of them.

Two layers, kept in separate packages on purpose:

- ``ged.contract``  – the *representation* layer.  Which nodes, attributes
  and relations enter the graph and what they mean.  This mirrors the
  ontology contract (``NODE_REPORT/GED_Design_Outline_v0.md`` §2) and must
  only change together with it.
- ``ged.cost`` / ``ged.solver`` – the *cost* layer.  Weights, type matrix,
  normalisation, solver.  Internal to GED; free to iterate after the
  ontology freeze.
"""
