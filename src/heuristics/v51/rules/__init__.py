"""One module per heuristic, named after its frozen id.

A module implements the condition of exactly one H and declares which
attributes, relations and named thresholds it reads.  The domain
(``appliesToFeatureType``), the statement and every threshold value come from
``ontology/v5.1/source``; nothing here hard-codes a number, and the condition
expressions in the YAML are documentation rather than an interpreted DSL.

The tests hold the two contracts this arrangement depends on: the implemented
set and the active set in the YAML are the same set, and every name a module
declares appears in that heuristic's frozen condition expression.
"""
