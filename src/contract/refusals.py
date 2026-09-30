"""Refusals both kernel lines have to word identically.

A refusal is an answer (node 10: "a refusal is a first-class answer"), and an
answer the two lines give in different words is two answers.  When the same
input is out of scope for the pipeline as a whole - not for one kernel, not for
one ontology - the reason belongs here, once, so ``occline`` and ``nxline``
print the same sentence and a reader of two records can see they were refused
for the same thing.

Standard library only, like the rest of ``contract``.
"""

from __future__ import annotations

#: A STEP with more than one solid body.  Building the first body and saying
#: nothing would publish a part that silently lost geometry, so the whole part
#: is refused instead; the count is in the text because "how many" is the first
#: thing a reader asks.
MULTI_BODY_UNSUPPORTED = (
    "multi-body part ({count} solid bodies): refusing rather than picking the first body"
)


def multi_body_unsupported(count: int) -> str:
    """The refusal text for a part carrying *count* solid bodies."""
    return MULTI_BODY_UNSUPPORTED.format(count=int(count))
