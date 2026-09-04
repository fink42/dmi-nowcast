"""Point sampling of the in-memory national product grids.

One function, two consumers: the ``/forecast`` endpoint (website Phase A
§A3) and the Web Push decision engine (website Phase D). Both must read a
point out of the held grids with *exactly* the same arithmetic — a
notification that fires on a different pixel than the panel shows is a
bug the user cannot diagnose — so the arithmetic lives here, once, and
the browser-side sampler (``frontend/src/lib/nowcast/sampler.ts``) mirrors
it by contract:

    native index = geo.lonlat_to_grid(lon, lat)          (fractional)
    product pixel = round(native.row / f), round(native.col / f)
    outside [0, h) × [0, w)  →  None (off coverage — never "0 %")
    NaN grid value           →  None (nodata pixel)

``f`` is the products' ``downsample_factor``; ``round`` is Python's
banker's rounding, as it always was in ``/forecast``.

The optional observed-rain grid rides the same pixel. It is not part
of :class:`~dmi_nowcast_core.national.NationalProducts` (it is an
observation, not an ensemble reduction) but it is on the same grid by
construction, so it is sampled here rather than through a second path
that could drift.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from dmi_nowcast_core.geo import CompositeGeo
from dmi_nowcast_core.national import NationalProducts


@dataclass(frozen=True)
class PointSample:
    """One point's read-out of the national grids.

    ``p_rain`` is keyed by lead minutes (the products' ``leads_min``) and
    holds ``None`` for a nodata pixel at that lead. ``eta_min`` /
    ``intensity_mm_h`` are ``None`` where the grid is NaN (no ETA within
    the horizon). ``row`` / ``col`` are the product-grid pixel actually
    read, for logging and tests.

    ``observed_mm_h`` is the rain rate measured at the pixel by the newest
    composite — ``None`` when no observed grid was supplied or the pixel is
    nodata. It is the only field here that is not a forecast, and the only
    one that can say "it is raining at this point *now*".
    """

    row: int
    col: int
    p_rain: dict[int, float | None]
    eta_min: float | None
    intensity_mm_h: float | None
    observed_mm_h: float | None = None


def product_pixel(
    products: NationalProducts, geo: CompositeGeo, lat: float, lon: float,
) -> tuple[int, int] | None:
    """Nearest product-grid pixel for ``(lat, lon)``; ``None`` off coverage."""
    idx = geo.lonlat_to_grid(lon, lat)
    f = products.downsample_factor
    row = int(round(idx.row / f))
    col = int(round(idx.col / f))
    h, w = products.eta_min.shape
    if not (0 <= row < h and 0 <= col < w):
        return None
    return row, col


def sample_point(
    products: NationalProducts,
    geo: CompositeGeo,
    lat: float,
    lon: float,
    *,
    observed_mm_h: np.ndarray | None = None,
) -> PointSample | None:
    """Read every product at the nearest pixel; ``None`` when off coverage.

    ``observed_mm_h`` is the cycle's observed-rain grid, on the same
    product grid. Absent — or a differently-shaped grid, which would mean
    a caller paired two different reductions — the sample's
    ``observed_mm_h`` is ``None``: an unknown observation, never a dry one.
    """
    pixel = product_pixel(products, geo, lat, lon)
    if pixel is None:
        return None
    row, col = pixel
    observed: float | None = None
    if (
        observed_mm_h is not None
        and observed_mm_h.shape == products.eta_min.shape
    ):
        observed = finite_or_none(observed_mm_h[row, col])
    return PointSample(
        row=row,
        col=col,
        p_rain={
            int(lead): finite_or_none(products.p_rain[lead][row, col])
            for lead in products.leads_min
        },
        eta_min=finite_or_none(products.eta_min[row, col]),
        intensity_mm_h=finite_or_none(products.intensity_mm_h[row, col]),
        observed_mm_h=observed,
    )


def finite_or_none(value: Any) -> float | None:
    """Grid sample → JSON-safe float; NaN/±inf (nodata, off-composite) → None."""
    v = float(value)
    return v if math.isfinite(v) else None


__all__ = ["PointSample", "product_pixel", "sample_point", "finite_or_none"]
