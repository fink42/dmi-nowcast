"""Disc sampling around a lat/lon on a radar grid.

Plan §6.4: a short-radius statistic (default 1 km, ~12 pixels at 500 m) is more
robust than nearest-pixel given the grid spacing and fuzzy rain edges.

We compute ``max`` / ``mean`` / ``p90`` within a true Euclidean disc, measured in
pixel space. Stereographic distortion is negligible at this scale (sub-metre
over 1 km near 56°N), so treating projection coordinates as locally Euclidean
is fine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geo import CompositeGeo


@dataclass(frozen=True)
class DiscStats:
    max_mm_h: float
    mean_mm_h: float
    p90_mm_h: float
    n_pixels_in_disc: int
    n_valid: int  # finite, non-NaN pixels actually contributing


def disc_pixel_indices(
    shape: tuple[int, int],
    row_center: float,
    col_center: float,
    radius_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (rows, cols) arrays of pixels within ``radius_px`` of the center.

    Indices are clipped to the array bounds; pixels outside the grid are dropped.
    """
    height, width = shape
    r_lo = max(0, int(np.floor(row_center - radius_px)))
    r_hi = min(height, int(np.ceil(row_center + radius_px)) + 1)
    c_lo = max(0, int(np.floor(col_center - radius_px)))
    c_hi = min(width, int(np.ceil(col_center + radius_px)) + 1)
    if r_hi <= r_lo or c_hi <= c_lo:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    rr, cc = np.meshgrid(
        np.arange(r_lo, r_hi), np.arange(c_lo, c_hi), indexing="ij"
    )
    in_disc = (rr - row_center) ** 2 + (cc - col_center) ** 2 <= radius_px ** 2
    return rr[in_disc], cc[in_disc]


def sample_disc(
    rain_rate: np.ndarray,
    geo: CompositeGeo,
    lon: float,
    lat: float,
    *,
    radius_m: float = 1000.0,
) -> DiscStats:
    """Compute max / mean / p90 rain rate within ``radius_m`` of (lon, lat).

    NaN pixels (``nodata`` in the source HDF5) are ignored. If no valid pixels
    fall within the disc, all stats are NaN and ``n_valid`` is 0.
    """
    pixel_scale_m = (geo.composite.xscale_m + geo.composite.yscale_m) / 2.0
    radius_px = radius_m / pixel_scale_m
    idx = geo.lonlat_to_grid(lon, lat)
    rows, cols = disc_pixel_indices(rain_rate.shape, idx.row, idx.col, radius_px)
    n_pixels = int(rows.size)
    if n_pixels == 0:
        return DiscStats(np.nan, np.nan, np.nan, 0, 0)
    values = rain_rate[rows, cols]
    finite = values[np.isfinite(values)]
    n_valid = int(finite.size)
    if n_valid == 0:
        return DiscStats(np.nan, np.nan, np.nan, n_pixels, 0)
    return DiscStats(
        max_mm_h=float(finite.max()),
        mean_mm_h=float(finite.mean()),
        p90_mm_h=float(np.percentile(finite, 90)),
        n_pixels_in_disc=n_pixels,
        n_valid=n_valid,
    )
