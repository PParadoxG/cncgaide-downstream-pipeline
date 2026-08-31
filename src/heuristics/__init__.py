"""Heuristic (DFM rule) evaluation layer.

The layer runs last: after feature recognition, after attribute extraction and
after the relation layer, when neither the node set nor the edge set can
change any more.  It reads the final nodes, the materialized edges and the
merged ``PartModel``, and it writes nothing back - a heuristic result is never
an attribute, a diagnostic or an edge.

What it produces is a verdict per (heuristic, evaluation unit), in four
values: ``NOT_APPLICABLE`` / ``PASS`` / ``FAIL`` / ``NOT_EVALUABLE``.  The
definitions, domains and thresholds come from the dialect's own ontology
source directory; the verdicts stay out of the ontology (decision D-7) and
live in ``runs/<tag>/parts/<id>/heuristics.json``.

Adding a heuristic means dropping one module into that dialect's rule
package (``heuristics.v51.rules``, ``heuristics.nx.rules``), named after its
frozen id.  The registry discovers it and the executor runs it; neither the
orchestrator nor the viewer changes.

This package is the machinery.  The rules themselves are split by *ontology* -
which vocabulary they are written in - not by kernel line, because the
provider registry already lets those two axes come apart.
"""

from heuristics.base import (  # noqa: F401
    FAIL,
    NOT_APPLICABLE,
    NOT_EVALUABLE,
    OUTCOMES,
    PASS,
    FeatureRule,
    HeuristicRule,
    PairRule,
    PartRule,
    Verdict,
)
from heuristics.executor import (  # noqa: F401
    corpus_summary,
    not_evaluated,
    outcome_counts,
    run_heuristics,
)
from heuristics.registry import BaseHeuristicRegistry  # noqa: F401
