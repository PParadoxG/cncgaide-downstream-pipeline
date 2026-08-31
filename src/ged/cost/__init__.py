"""Cost layer: weights, type matrix, attribute/edge/fixed-node distances.

Everything in this package is a GED-internal tuning decision.  It consumes
the frozen representation (``ged.contract``) and never feeds back into it.
"""

from .config import CostConfig, FixedCostParts, GeometryTerm
from .default import default_config

__all__ = ["CostConfig", "FixedCostParts", "GeometryTerm", "default_config"]
