"""Ontology source readers: one module per ontology, named after it.

An ontology's YAML source is the single semantic standard of the line that
speaks it, and more than one package needs to read it — the relation dialect
wants the class tree, the heuristic dialect wants the frozen H set and the
thresholds, GED wants the attribute units.  Loading it once, here, is what
keeps those from being several places that have to be right about one file.

Why this package rather than a home inside one of the readers' consumers:

* it belongs to neither dialect, and putting it in either would make the
  other depend backwards on a package it has nothing to do with;
* ``contract`` cannot hold it — that package may import only the standard
  library, and reading YAML needs more than that;
* the NX journal process must be able to import it, which rules out anything
  that pulls in a kernel: it wires ``src`` onto ``sys.path`` itself and has
  only a vendored pure-python yaml.

Modules here are named after the **ontology**, never after a kernel line
(node 19' keeps those two questions apart), and import nothing beyond the
standard library and ``yaml``.
"""
