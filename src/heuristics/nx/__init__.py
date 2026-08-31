"""The nx_v1 heuristic dialect: rules written in NX's own vocabulary.

The classes are NX's ``Type`` values, the attributes are NX's own names and
values, and the domains and thresholds come from ``ontology/nx_v1/source``.
Nothing here is translated into or out of the v5.1 ontology: node 19a is
explicit that the same H id under the two dialects yields two numbers that
are not one metric, and that what the two lines are compared on is the shape
of each rule set - which rules survive, which are withdrawn, and why.

The machinery these rules run on (the evaluation context, the four-value
gate, the executor) is two levels up in ``heuristics``.
"""
