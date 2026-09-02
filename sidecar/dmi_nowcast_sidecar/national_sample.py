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
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

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
    """

    row: int
    col: int
    p_rain: dict[int, float | None]
    eta_min: float | None
    intensity_mm_h: float | None


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
    products: NationalProducts, geo: CompositeGeo, lat: float, lon: float,
) -> PointSample | None:
    """Read every product at the nearest pixel; ``None`` when off coverage."""
    pixel = product_pixel(products, geo, lat, lon)
    if pixel is None:
        return None
    row, col = pixel
    return PointSample(
        row=row,
        col=col,
        p_rain={
            int(lead): finite_or_none(products.p_rain[lead][row, col])
            for lead in products.leads_min
        },
        eta_min=finite_or_none(products.eta_min[row, col]),
        intensity_mm_h=finite_or_none(products.intensity_mm_h[row, col]),
    )


def finite_or_none(value: Any) -> float | None:
    """Grid sample → JSON-safe float; NaN/±inf (nodata, off-composite) → None."""
    v = float(value)
    return v if math.isfinite(v) else None


__all__ = ["PointSample", "product_pixel", "sample_point", "finite_or_none"]
