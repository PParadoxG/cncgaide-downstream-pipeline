"""The OCC kernel line: everything that only exists because OCCT is in the process.

``parse`` fills the PartModel from a STEP file, ``kernel`` implements
:class:`contract.kernel.KernelServices` on the live shape, and ``aagnet``
produces the per-face evidence.  Only the composition roots
(``core`` and ``store.sync``) may import this package;
every shared layer speaks to a kernel line through ``contract`` only.

The package is named ``occline`` rather than ``occ`` because Windows
filesystems are case-insensitive: a package directory named ``occ`` would
collide with ``import OCC``.
"""
