"""``pysteps.utils``, reduced to what ``dense_lucaskanade`` calls.

``morph_opening`` (clutter filter), ``decluster`` / ``detect_outliers``
(sparse-vector cleansing) and ``idwinterp2d`` (the default sparse → dense
interpolation). Upstream's ``get_method`` dispatcher is replaced by
``motion/lucaskanade.py``'s ``_INTERP_METHODS``.
"""
from .cleansing import decluster, detect_outliers  # noqa: F401
from .images import morph_opening  # noqa: F401
from .interpolate import idwinterp2d  # noqa: F401

__all__ = ["decluster", "detect_outliers", "morph_opening", "idwinterp2d"]
