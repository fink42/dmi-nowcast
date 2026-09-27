"""Quiet ``print`` for the vendored STEPS modules.

VENDORING MODIFICATION 9 (log hygiene, no numeric effect). Upstream STEPS
prints a banner, its method table, the per-level AR correlation tables and
one "Computing nowcast for time step t... done." line per timestep to
stdout — on every cycle, straight into the service's container log. The
vendored modules that print import :func:`vprint` under the name ``print``,
so each call site is untouched and the text is only emitted when
``VERBOSE`` is set (debugging a STEPS run by hand).
"""
from __future__ import annotations

import builtins

#: Set True to restore upstream's stdout chatter.
VERBOSE = False


def vprint(*args, **kwargs) -> None:
    if VERBOSE:
        builtins.print(*args, **kwargs)
