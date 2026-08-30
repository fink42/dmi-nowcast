"""Nowcast baselines (plan §9.2).

For a fixed home location, **persistence and Eulerian-extrapolation are
mathematically identical**: both answer "what is the rain rate at the home
pixel now?" and use that as the prediction for every future horizon. We
implement only ``persistence`` and document this equivalence in the backtest
output rather than emit two identical rows per timestamp.

``lagrangian_mean`` is genuinely different: it advects the home location
upstream by ``mean_velocity * horizon`` and samples the current field there.

Outputs use the same ``DiscStats`` shape as live sampling (max / mean / p90),
so evaluation operates on max-within-disc just like the production code path.
"""
from __future__ import annotations

import numpy as np

from .advect import advect_point
from .geo import CompositeGeo
from .motion import phase_correlation_shift
from .sample import DiscStats, disc_pixel_indices, sample_disc


def persistence(
    rain_now: np.ndarray,
    geo: CompositeGeo,
    lon: float,
    lat: float,
    *,
    radius_m: float = 1000.0,
) -> DiscStats:
    """The rain rate at home stays whatever it is right now."""
    return sample_disc(rain_now, geo, lon, lat, radius_m=radius_m)


def lagrangian_dense(
    rain_now: np.ndarray,
    vy: np.ndarray,
    vx: np.ndarray,
    geo: CompositeGeo,
    lon: float,
    lat: float,
    horizon_minutes: float,
    *,
    dt_minutes: float = 10.0,
    radius_m: float = 1000.0,
) -> DiscStats:
    """Lagrangian extrapolation with a pre-computed dense flow.

    The flow ``(vy, vx)`` should be in pixels per ``dt_minutes``. Compute it
    with ``dense_flow(dbz_prev, dbz_now)`` — on **dBZ** rather than rain rate
    because Farnebäck's polynomial expansion handles the log-scaled field
    better than the long-tailed rain-rate distribution.

    The flow is passed in (not computed internally) so the caller can reuse a
    single flow field across many horizons at the same timestamp.
    """
    pixel_scale_m = (geo.composite.xscale_m + geo.composite.yscale_m) / 2.0
    radius_px = radius_m / pixel_scale_m

    home = geo.lonlat_to_grid(lon, lat)
    src_row, src_col = advect_point(
        home.row, home.col, vy, vx,
        horizon_minutes=horizon_minutes, dt_minutes=dt_minutes,
    )

    rows, cols = disc_pixel_indices(rain_now.shape, src_row, src_col, radius_px)
    n_pixels = int(rows.size)
    if n_pixels == 0:
        return DiscStats(np.nan, np.nan, np.nan, 0, 0)
    values = rain_now[rows, cols]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return DiscStats(np.nan, np.nan, np.nan, n_pixels, 0)
    return DiscStats(
        max_mm_h=float(finite.max()),
        mean_mm_h=float(finite.mean()),
        p90_mm_h=float(np.percentile(finite, 90)),
        n_pixels_in_disc=n_pixels,
        n_valid=int(finite.size),
    )


def lagrangian_mean(
    rain_now: np.ndarray,
    rain_prev: np.ndarray,
    geo: CompositeGeo,
    lon: float,
    lat: float,
    horizon_minutes: float,
    *,
    dt_minutes: float = 5.0,
    radius_m: float = 1000.0,
) -> DiscStats:
    """Lagrangian extrapolation with a single mean motion vector.

    Compute (dy, dx) from ``rain_prev`` → ``rain_now`` via phase correlation,
    then sample the current field at the upstream location for the requested
    horizon.
    """
    dy, dx = phase_correlation_shift(rain_prev, rain_now)
    pixel_scale_m = (geo.composite.xscale_m + geo.composite.yscale_m) / 2.0
    radius_px = radius_m / pixel_scale_m

    home = geo.lonlat_to_grid(lon, lat)
    scale = horizon_minutes / dt_minutes
    sample_row = home.row - dy * scale
    sample_col = home.col - dx * scale

    rows, cols = disc_pixel_indices(rain_now.shape, sample_row, sample_col, radius_px)
    n_pixels = int(rows.size)
    if n_pixels == 0:
        return DiscStats(np.nan, np.nan, np.nan, 0, 0)
    values = rain_now[rows, cols]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return DiscStats(np.nan, np.nan, np.nan, n_pixels, 0)
    return DiscStats(
        max_mm_h=float(finite.max()),
        mean_mm_h=float(finite.mean()),
        p90_mm_h=float(np.percentile(finite, 90)),
        n_pixels_in_disc=n_pixels,
        n_valid=int(finite.size),
    )
