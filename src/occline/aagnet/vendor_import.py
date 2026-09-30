"""Import AAGNet's vendored trees without publishing their names process-wide.

AAGNet is a research repository, not a package: its modules import each other
by the bare top-level names ``dataset``, ``models`` and ``utils`` — line 4 of
``models/inst_segmentors.py`` is literally ``import models.encoders``.  The
vendor snapshot is upstream's, byte for byte (the ``aagnet`` layer stamp hashes
it), so those import statements cannot be rewritten here.

Until node 28 the adapter bought them by inserting the vendor root at
``sys.path[0]`` and leaving it there.  From then on ``import utils`` resolved
to AAGNet's ``utils`` for **every** module in the process, not just this one —
node 26 finding B-20.  ``utils`` and ``models`` are names a great deal of
third-party code uses, and which one won depended on import order.

What happens instead: for the duration of one ``with`` block the three names
are bound to hand-built package objects whose ``__path__`` points straight at
the vendor directories, and afterwards they are re-homed under the
``occline.aagnet.vendor.AAGNet.`` prefix.  Anything that already occupied those
names is set aside on entry and put back on exit.  ``sys.path`` is never
touched, so nothing else in the process can reach the vendor tree by accident,
and the tree cannot be shadowed by whatever happens to be installed.

The ``__path__`` lists are plain lists on purpose.  A PEP 420 namespace package
found through ``sys.path`` carries a ``_NamespacePath`` that recomputes itself
against ``sys.path`` whenever ``sys.path`` changes, which would empty it here;
a plain list is what makes these packages independent of ``sys.path``.
"""

from __future__ import annotations

import contextlib
import sys
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec
from pathlib import Path
from types import ModuleType
from typing import Iterator

#: The top-level names AAGNet imports itself by.  Nothing outside the vendor
#: tree may rely on these resolving to AAGNet.
VENDOR_TOP_LEVEL_NAMES: tuple[str, ...] = ("dataset", "models", "utils")

#: Where the loaded modules are parked once the ``with`` block is over, so they
#: stay reachable for inspection (and stay out of everyone else's way).
VENDOR_MODULE_PREFIX = "occline.aagnet.vendor.AAGNet."


def _owned(module_name: str) -> bool:
    """True for a top-level vendor name or any submodule of one."""
    return module_name.split(".")[0] in VENDOR_TOP_LEVEL_NAMES


def _package(name: str, directory: Path) -> ModuleType:
    """A package object for *directory* that does not consult ``sys.path``."""
    spec = ModuleSpec(name, None, is_package=True)
    spec.submodule_search_locations = [str(directory)]
    return module_from_spec(spec)


@contextlib.contextmanager
def vendor_top_level(root: Path) -> Iterator[None]:
    """Make AAGNet's ``dataset`` / ``models`` / ``utils`` importable, briefly.

    *root* is the vendored AAGNet directory.  Inside the block, and only
    inside it, ``import models.encoders`` and friends resolve to the vendor
    tree; on the way out every module the block imported is moved to
    ``VENDOR_MODULE_PREFIX + name`` and the bare names are handed back to
    whoever held them.  Modules already imported keep working afterwards:
    Python resolved their references at import time, and this only changes
    what a *future* ``import`` of those names would find.
    """
    shadowed = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if _owned(name)
    }
    for name in VENDOR_TOP_LEVEL_NAMES:
        sys.modules[name] = _package(name, root / name)
    try:
        yield
    finally:
        for name in [key for key in list(sys.modules) if _owned(key)]:
            sys.modules[VENDOR_MODULE_PREFIX + name] = sys.modules.pop(name)
        sys.modules.update(shadowed)
