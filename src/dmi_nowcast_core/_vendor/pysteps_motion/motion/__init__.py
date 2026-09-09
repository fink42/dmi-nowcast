"""``pysteps.motion``, reduced to the Lucas-Kanade estimator.

Upstream's ``__init__`` imports every estimator including ``_proesmans``
and ``_vet``, which are Cython extensions this repo does not build.
"""
from .lucaskanade import dense_lucaskanade  # noqa: F401

__all__ = ["dense_lucaskanade"]
