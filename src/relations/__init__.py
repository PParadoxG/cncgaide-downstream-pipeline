"""Feature relation layer.

The relation layer is a strictly downstream reader.  It runs once, after
feature recognition, compound resolution and residual registration have all
finished and the final ``FeatureNode`` set is fixed.  It may read only the
final nodes and the merged ``PartModel``; it never touches ``featureType``,
face ownership, attributes or ``missingAttributes``.

Adding a further relation means dropping one module into that dialect's
rule package (``relations.v51.rules``, ``relations.nx.rules``).  The registry
discovers it and the executor runs it; the orchestrator body does not change.

This package is the machinery.  The rules themselves are split by *ontology* -
which vocabulary they are written in - not by kernel line, because the
provider registry already lets those two axes come apart.
"""

from relations.base import RelationContext, RelationRule  # noqa: F401
from relations.executor import relation_counts, run_relations  # noqa: F401
from relations.registry import BaseRelationRegistry  # noqa: F401
