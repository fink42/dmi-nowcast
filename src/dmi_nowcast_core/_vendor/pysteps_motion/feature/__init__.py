"""``pysteps.feature``, reduced to Shi-Tomasi corner detection.

Upstream's ``get_method`` dispatcher also reaches ``blob`` (scikit-image)
and ``tstorm`` (pandas + skimage); ``dense_lucaskanade``'s default is
``shitomasi`` and that is the only one vendored — see
``motion/lucaskanade.py``'s ``_FEATURE_METHODS``.
"""
from .shitomasi import detection  # noqa: F401

__all__ = ["detection"]
