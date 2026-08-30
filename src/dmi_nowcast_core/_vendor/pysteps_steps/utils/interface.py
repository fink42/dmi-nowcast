"""Minimal stand-in for ``pysteps.utils.interface`` — FFT dispatch only.

The upstream ``utils.interface`` exposes a dispatcher for *every* utility
method in pysteps (~40 of them across cascade, cleansing, conversion,
dimension, images, interpolate, pca, reprojection, spectral, tapering,
transformation). The STEPS forecast code path only ever calls
``utils.get_method`` with an FFT name (``"numpy"`` / ``"scipy"`` /
``"pyfftw"``) — all other calls in upstream are from helper utilities we
don't ship.

By restricting the dispatcher to FFT we avoid having to vendor the dozen
unrelated utility modules, while keeping the call site in ``steps.py``
and ``noise/fftgenerators.py`` working unchanged.

Modified from pysteps 1.21.1 (BSD-3) — original at
``pysteps/utils/interface.py``.
"""
from __future__ import annotations

from . import fft


def get_method(name, **kwargs):
    """FFT-only dispatcher. Returns an object with ``fft2``/``ifft2``/etc.

    Mirrors the FFT branch of upstream ``pysteps.utils.interface.get_method``;
    rejects any non-FFT name explicitly so a future caller asking for
    e.g. ``"upscale"`` fails loudly instead of silently returning ``None``.
    """
    if name is None:
        name = "numpy"
    name = name.lower()
    if name not in ("numpy", "scipy", "pyfftw"):
        raise ValueError(
            f"vendored pysteps_steps.utils.get_method only supports FFT "
            f"backends (numpy/scipy/pyfftw); got {name!r}. Upstream pysteps "
            "exposes many more — vendor more modules if you need them."
        )
    if "shape" not in kwargs:
        raise KeyError("mandatory keyword argument 'shape' not given")
    shape = kwargs.pop("shape")
    if name == "numpy":
        return fft.get_numpy(shape, **kwargs)
    if name == "scipy":
        return fft.get_scipy(shape, **kwargs)
    return fft.get_pyfftw(shape, **kwargs)
