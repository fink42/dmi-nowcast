"""Vendored subset of pysteps — just enough to run STEPS forecasts.

Original project: https://github.com/pySTEPS/pysteps
Vendored from pysteps 1.21.1 (BSD-3-Clause). See LICENSE-pysteps and NOTICE
in this directory.

Why vendored: pysteps has no PyPI wheels for ``musllinux_*_cp314`` (HA OS)
and its source build requires a C toolchain with OpenMP that HA OS doesn't
have at runtime. The ~20 .py files in this tree are the strict subset that
``pysteps.nowcasts.steps.forecast`` actually executes — verified by an
``sys.settrace`` audit against the upstream test suite. All Cython modules
(``pysteps.motion._proesmans``, ``_vet``) are excluded; we use scikit-image
for optical flow instead.

Re-export the public entry point so callers can write either::

    from dmi_nowcast_core._vendor.pysteps_steps import forecast  # dev tree
    from ._vendor.pysteps_steps import forecast                  # HA tree

depending on where the integration is loaded from. The internal cross-file
imports inside this package are all relative (``from ..cascade import …``)
so the same source works under either rooting.
"""
from __future__ import annotations

_VENDORED_FROM = "pysteps 1.21.1"

from .nowcasts.steps import forecast  # noqa: E402, F401

__all__ = ["forecast", "_VENDORED_FROM"]
