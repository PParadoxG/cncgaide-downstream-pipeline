"""The v5.1 heuristic dialect: rules written in ``ontology/v5.1``'s vocabulary.

A dialect is one ontology's heuristic set — its frozen definitions, its
thresholds, its class names — implemented as one hand-written module per H.
The machinery those modules run on (the evaluation context, the four-value
gate, the executor) is one level up in ``heuristics``.

Split by *ontology*, not by kernel line: which vocabulary a rule speaks is a
different question from who filled the ``PartModel``.  The two happen to pair
one-to-one today, and the provider registry already allows them not to.
"""
