"""Vendored subset of pysteps — just enough to run ``dense_lucaskanade``.

Original project: https://github.com/pySTEPS/pysteps
Vendored from pysteps 1.21.1 (BSD-3-Clause). See LICENSE-pysteps and NOTICE
in this directory.

Why vendored: the same reason ``_vendor/pysteps_steps`` is (no musllinux
wheels, a source build that needs an OpenMP toolchain), plus one specific
to this repo — the sidecar deliberately has no ``pysteps`` dependency, and
Phase H needs pysteps' Lucas–Kanade as the literature's reference motion
estimator to score against Farnebäck on identical cases. Adding a
1,500-file dependency to the runtime image to call one function it will
probably never serve is the wrong trade; six files are not.

This tree is separate from ``pysteps_steps`` on purpose: that one was
audited (``sys.settrace``) down to the strict subset STEPS executes and
carries seven memory-motivated modifications to code STEPS runs sixteen
times a timestep. This one is the LK subset, essentially unmodified. A
shared tree would make it impossible to say which of those statements
applies to a given file.

The public entry point::

    from dmi_nowcast_core._vendor.pysteps_motion import dense_lucaskanade

Internal cross-file imports are all relative, so the package works
wherever it is rooted.
"""
from __future__ import annotations

_VENDORED_FROM = "pysteps 1.21.1"

from .motion.lucaskanade import dense_lucaskanade  # noqa: E402, F401

__all__ = ["dense_lucaskanade", "_VENDORED_FROM"]
