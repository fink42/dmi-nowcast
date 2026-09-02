"""National forecast products from a STEPS ensemble (website Phase A plan §A1).

Reduces a STEPS ensemble array — the ``(n_members, n_timesteps, h, w)`` mm/h
output of :func:`dmi_nowcast_core.probabilistic.run_ensemble` — into national
forecast grids on the same (downsampled) grid:

- ``p_rain[lead]``: per-pixel fraction of ensemble members whose cumulative
  max rain rate exceeds the detection threshold by that lead time,
- ``eta_min``: per-pixel ensemble-median time until rain arrives,
- ``intensity_mm_h``: per-pixel ensemble-median raw rain rate at arrival.

:func:`motion_grids_kmh` adds a fourth, non-ensemble product on the same
grid: the cell-motion field in km/h (east / north components), nodata only
outside radar coverage (and everywhere when the composite holds no echo at
all) — see its docstring.

This module is pure core: numpy only, no sidecar / FastAPI / homeassistant
imports (and none of the heavier core deps like pyproj — geolocation stays
with the caller).

**Lead-time bookkeeping** (Phase A plan §A0, the named bug magnet): leads are
"minutes from now" while ensemble timesteps count from radar-frame time, so
every lead is corrected by ``frame_age_min`` before choosing a timestep
index. Timestep index ``t`` is valid at ``(t + 1) * timestep_min`` minutes
after radar-frame time, matching ``aggregate_at_home``'s
``first_min = (first_idx + 1) * timestep_min`` convention.

**Agreement with the home forecast** (Phase A plan §A4): sampling these grids
at the home pixel must reproduce ``aggregate_at_home``'s probabilities and
ETA window when both are fed the same frame-age-corrected leads; the
agreement test in ``tests/test_national.py`` proves it.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .dense_flow import (
    DEFAULT_EFOLD_KM,
    DEFAULT_SUPPORT_DILATION_PX,
    distance_to_support,
)

# Detection threshold, mm/h. Same value as
# ``probabilistic.DB_THRESHOLD_MM_H``; restated here so this module's import
# graph stays numpy-only (``probabilistic`` pulls in pyproj via ``geo``).
DEFAULT_THRESHOLD_MM_H = 0.1

# Lead times served on the national products (Phase A plan §A1).
DEFAULT_LEADS_MIN = (10, 20, 30, 45, 60)

# Motion product (R2, reworked for issue #6): the search scales, in km, of
# the off-echo fill. A display pixel's motion is estimated from the echo
# inside a Gaussian whose width is the smallest of these that actually
# reaches an echo (the pixel's nearest echo must lie within 2σ) — "look
# nearby first, widen only when nothing is nearby". 25 km is roughly one
# convective cell plus its surroundings on a 2 km product grid; 50 km is a
# cluster; 100 km is a frontal band, and 2×100 km ≈ 200 km is already the
# width of Denmark, so the ladder covers everything the composite can hold
# before the country-wide mean takes over.
DEFAULT_MOTION_FILL_SCALES_KM = (25.0, 50.0, 100.0)


@dataclass(frozen=True)
class NationalProducts:
    """National forecast grids for one radar cycle.

    All grids share the ensemble's (h, w) shape — the ×4-downsampled national
    grid in production — and are float32. Pixels outside the composite (no
    finite value in any member at any timestep) are NaN in every product.

    - ``p_rain[lead]``: fraction of members exceeding ``threshold_mm_h`` by
      ``lead`` minutes from now, in [0, 1]. Keys are integer lead minutes.
    - ``eta_min``: minutes from now until rain (see :func:`national_products`
      for the exact semantics); NaN where rain never reaches the pixel within
      the forecast horizon.
    - ``intensity_mm_h``: ensemble-median raw rain rate at the pixel's ETA
      timestep; NaN wherever ``eta_min`` is NaN.

    The remaining fields are the reduction parameters, kept for the A2
    artifact manifest.
    """

    p_rain: dict[int, np.ndarray]
    eta_min: np.ndarray
    intensity_mm_h: np.ndarray
    leads_min: tuple[int, ...]
    threshold_mm_h: float
    timestep_min: float
    frame_age_min: float
    downsample_factor: int
    n_members: int


def _steps_in_lead(lead_min: float, frame_age_min: float, timestep_min: float,
                   n_timesteps: int) -> int:
    """Number of ensemble timesteps covered by ``lead_min`` minutes from now.

    Effective lead = nominal lead + frame age (timesteps count from
    radar-frame time). ``ceil`` mirrors ``aggregate_at_home``'s
    ``steps_in_lead`` convention — its ``round`` and this ``ceil`` coincide at
    the exact timestep multiples produced by whole-timestep leads — while
    staying inclusive for in-between effective leads. Clamped to
    ``[1, n_timesteps]`` exactly like ``aggregate_at_home``.
    """
    effective = lead_min + frame_age_min
    # Epsilon mirrors ``probabilistic.frame_age_corrected_leads`` so float
    # fuzz at exact timestep multiples can't split the two implementations
    # into different buckets.
    k = math.ceil(effective / timestep_min - 1e-9)
    return max(1, min(k, n_timesteps))


def national_products(
    ensemble: np.ndarray,
    *,
    leads_min: Iterable[int] = DEFAULT_LEADS_MIN,
    threshold_mm_h: float = DEFAULT_THRESHOLD_MM_H,
    timestep_min: float = 5.0,
    frame_age_min: float = 0.0,
    downsample_factor: int = 4,
) -> NationalProducts:
    """Reduce a STEPS ensemble into national probability / ETA / intensity grids.

    ``ensemble`` is the ``(n_members, n_timesteps, h, w)`` mm/h array from
    ``run_ensemble``. ``frame_age_min`` is the age of the newest radar frame
    at compute time; ``downsample_factor`` is recorded as metadata (the grids
    inherit the ensemble's shape — this function never resamples).

    Products, all fully vectorised (the only Python loop is over the handful
    of leads):

    - ``p_rain[lead]``: fraction of members whose cumulative max rate
      (``np.logical_or.accumulate`` of per-step threshold exceedance — the
      boolean equivalent of ``np.maximum.accumulate`` on the rates) exceeds
      ``threshold_mm_h`` within the frame-age-corrected lead. Identical
      arithmetic to ``aggregate_at_home``'s
      ``mean(exceed[:, :steps_in_lead].any(axis=1))``.
    - ``eta_min``: per pixel, the first timestep at which the member
      exceedance fraction reaches ≥ 0.5, converted to minutes from now:
      ``(step_idx + 1) * timestep_min - frame_age_min``, clamped ≥ 0; NaN
      where the fraction never reaches 0.5. Because the per-member
      exceedance is cumulative (hence monotone), the first timestep where
      ≥ half the members have crossed *is* the median of the per-member
      first-crossing times — exactly ``aggregate_at_home``'s ``eta_p50_min``
      semantics, up to (a) ``np.percentile``'s linear interpolation between
      timestep boundaries (agreement within one timestep) and (b) pixels
      where fewer than half the members ever cross: there this map is
      conservatively NaN while ``aggregate_at_home`` still reports a median
      over the crossing minority.
    - ``intensity_mm_h``: ensemble-median of the RAW (not cumulative) rain
      rate at each pixel's ETA timestep, NaN-ignoring across members; NaN
      where there is no ETA. Members still dry at the ETA step contribute
      their raw ~0 mm/h, pulling the median low — a deliberately
      conservative arrival intensity.

    NaN handling: a NaN pixel-value compares False against the threshold, so
    out-of-composite pixels can never produce false positives; pixels with no
    finite value in any member at any timestep are NaN in all products.
    """
    forecast = np.asarray(ensemble)
    if forecast.ndim != 4:
        raise ValueError(
            f"ensemble must be (n_members, n_timesteps, h, w); got shape {forecast.shape}"
        )
    if timestep_min <= 0:
        raise ValueError(f"timestep_min must be > 0, got {timestep_min}")
    if frame_age_min < 0:
        raise ValueError(f"frame_age_min must be >= 0, got {frame_age_min}")
    if downsample_factor < 1:
        raise ValueError(f"downsample_factor must be >= 1, got {downsample_factor}")
    leads: list[int] = []
    for lead in leads_min:
        if float(lead) != int(lead):
            raise ValueError(f"leads_min must be whole minutes, got {lead!r}")
        leads.append(int(lead))
    if not leads:
        raise ValueError("leads_min must not be empty")

    n_members, n_timesteps, h, w = forecast.shape

    # Per-member cumulative threshold exceedance. NaN >= threshold is False,
    # so nodata pixels never count as exceeding.
    with np.errstate(invalid="ignore"):
        exceed = forecast >= threshold_mm_h
    exceed_cum = np.logical_or.accumulate(exceed, axis=1)
    # Fraction of members exceeding by each timestep, (n_timesteps, h, w).
    # Bool sums are exact integers, so the single float32 division is the
    # correctly-rounded member fraction.
    frac_cum = exceed_cum.mean(axis=0, dtype=np.float32)

    # Pixels outside the composite: no finite value anywhere in the ensemble.
    valid = np.isfinite(forecast).any(axis=(0, 1))

    p_rain: dict[int, np.ndarray] = {}
    for lead in leads:
        k = _steps_in_lead(lead, frame_age_min, timestep_min, n_timesteps)
        grid = frac_cum[k - 1].astype(np.float32, copy=True)
        grid[~valid] = np.nan
        p_rain[lead] = grid

    # ETA: first timestep with exceedance fraction >= 0.5 (frac_cum is
    # monotone along the time axis, so argmax finds the first crossing).
    reached = frac_cum >= 0.5
    ever = reached.any(axis=0) & valid
    first_step = np.argmax(reached, axis=0)  # 0 where never reached; masked below
    eta = (first_step + 1).astype(np.float32) * np.float32(timestep_min)
    eta = eta - np.float32(frame_age_min)
    eta = np.maximum(eta, np.float32(0.0))
    eta_min = np.where(ever, eta, np.float32(np.nan)).astype(np.float32)

    # Intensity: ensemble-median RAW rate at each pixel's ETA step. Gather
    # each pixel's ETA step across members, then a NaN-ignoring median.
    gather_idx = first_step[np.newaxis, np.newaxis, :, :]
    at_eta = np.take_along_axis(forecast, gather_idx, axis=1)[:, 0]
    with warnings.catch_warnings():
        # All-NaN member slices legitimately yield NaN.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        intensity = np.nanmedian(at_eta, axis=0).astype(np.float32)
    intensity_mm_h = np.where(ever, intensity, np.float32(np.nan)).astype(np.float32)

    return NationalProducts(
        p_rain=p_rain,
        eta_min=eta_min,
        intensity_mm_h=intensity_mm_h,
        leads_min=tuple(leads),
        threshold_mm_h=float(threshold_mm_h),
        timestep_min=float(timestep_min),
        frame_age_min=float(frame_age_min),
        downsample_factor=int(downsample_factor),
        n_members=int(n_members),
    )


def motion_grids_kmh(
    vy: np.ndarray,
    vx: np.ndarray,
    rain_mm_h: np.ndarray,
    *,
    pixel_km: float,
    timestep_min: float,
    downsample_factor: int = 4,
    support_threshold_mm_h: float = DEFAULT_THRESHOLD_MM_H,
    efold_km: float = DEFAULT_EFOLD_KM,
    dilation_px: float = DEFAULT_SUPPORT_DILATION_PX,
    fill_scales_km: Iterable[float] = DEFAULT_MOTION_FILL_SCALES_KM,
) -> tuple[np.ndarray, np.ndarray]:
    """Cell motion as two physical-unit grids on the ×4 product grid (R2).

    The website's click-anywhere motion arrow samples these two grids the
    same way it samples ``p_rain`` / ``eta`` / ``intensity`` — so they are
    published **in km/h**, east- and north-positive, and the browser does no
    unit algebra and no axis flipping.

    Parameters
    ----------
    vy, vx:
        Native-grid flow in pixels per frame, **as estimated** — the
        ``dense_flow`` (or mean-motion fallback) output after the caller's
        nan→0 / clip sanitise but *before* ``dense_flow.complete_flow``. On
        the echo that is identical to what the advection consumes; off it,
        Farnebäck's exact zeros are replaced here by this function's own
        completion (see below) rather than by ``complete_flow``'s.
    rain_mm_h:
        Native-grid rain rate for the same frame (NaN outside the radar
        composite). Defines the echo support and the coverage mask.
    pixel_km:
        NATIVE pixel size in km (0.5 on the DMI 500 m composite).
    timestep_min:
        Minutes between the two frames the flow was estimated from — the
        "per frame" in ``vy``/``vx``'s units (``dt_min``, ~10 min on the
        fullRange-only feed).
    downsample_factor:
        Product-grid stride ``f``. Sampling matches
        ``probabilistic.run_ensemble``'s velocity handling exactly:
        ``v[::f, ::f] / f`` — stride-slice, **then divide by f**, because a
        given physical motion spans ``f`` times fewer pixels once the pixels
        are ``f`` times bigger. The ``/f`` and the ``f×`` in the pixel size
        cancel, so the published km/h are the native ones; the pairing is
        kept explicit rather than cancelled by hand so this function stays
        readable against ``run_ensemble``.
    support_threshold_mm_h:
        Rain rate a pixel needs to count as echo.
    efold_km, dilation_px:
        The same relaxation geometry ``complete_flow`` uses: a full-weight
        halo of ``dilation_px`` NATIVE pixels around the echo, then an
        ``exp(-d / efold_km)`` handover from the measured flow to the fill.
        Sharing the constants keeps the served arrow and the advection
        agreeing wherever the flow was actually measured.
    fill_scales_km:
        Search scales of the off-echo fill, ascending.

    Returns
    -------
    ``(motion_east_kmh, motion_north_kmh)`` float32 on the downsampled grid.
    Grid rows grow southward, so north = ``-vy``.

    Nodata
    ------
    NaN (→ level 255 once quantised) outside the radar composite, and
    everywhere when the composite holds no echo at all — with nothing
    moving anywhere there is nothing to report. Inside coverage, whenever
    *some* echo exists in the composite, every pixel carries a vector:
    issue #6 was a point in Odense with cells plainly visible and moving,
    reported as "no measured motion" because the nearest echo was 20-plus
    km away.

    Off-echo completion, and why it is not ``complete_flow``'s
    --------------------------------------------------------
    Away from the echo the served field is

        v = w·v_raw + (1 - w)·v_nearest,     w = exp(-d / efold_km)

    where ``d`` is the distance to the nearest echo pixel (minus the
    ``dilation_px`` halo) and ``v_nearest`` is a rain-weighted average of
    the estimated flow over the echo, taken through a Gaussian whose width
    grows with ``d``: σ = max(min(fill_scales_km), d / 2), so a pixel's
    nearest echo always sits inside ~2σ of the search area and the closest
    cells dominate the average by construction. Nothing nearby means the
    search widens — 25 → 50 → 100 km — and past the widest scale the field
    relaxes to the rain-weighted mean over the whole composite. The
    estimate is computed at the ladder's fixed scales (three separable
    Gaussian pairs) and interpolated per pixel in 1/σ, with the global mean
    as the σ→∞ end of the ladder; interpolating rather than hard-switching
    at each 2σ boundary is what keeps the field continuous, since two cells
    with different motion give visibly different answers at two adjacent
    scales (~30 km/h apart in the two-cell test) and a hard switch would
    draw that as a seam in the arrows.

    This deliberately differs from ``dense_flow.complete_flow``, which
    relaxes toward the single rain-weighted **national bulk** vector.
    Both are honest; they answer different questions. The advection and the
    STEPS velocity need one stable, low-variance prior for the whole domain
    — it is what the backtests and the isotonic calibration were fitted
    against, so it must not move. The arrow answers "what is the rain
    nearest me doing", and a country-wide average is the wrong answer to
    that whenever two systems are on the map at once. Hence: same weight
    function, different target, and ``complete_flow`` is left untouched.

    Normalised convolution detail: numerator ``G_σ * (rain·v)`` and
    denominator ``G_σ * rain`` are filtered with ``mode='constant',
    cval=0`` so the ratio stays an unbiased weighted mean at the domain
    edge, and the denominator is guarded against zero.
    """
    from scipy.ndimage import gaussian_filter

    vy_arr = np.asarray(vy, dtype=np.float32)
    vx_arr = np.asarray(vx, dtype=np.float32)
    rain = np.asarray(rain_mm_h, dtype=np.float32)
    if vy_arr.shape != vx_arr.shape or vy_arr.shape != rain.shape:
        raise ValueError(
            "vy, vx and rain_mm_h must share one shape; got "
            f"{vy_arr.shape}, {vx_arr.shape}, {rain.shape}"
        )
    if vy_arr.ndim != 2:
        raise ValueError(f"motion grids need 2-D inputs, got {vy_arr.ndim}-D")
    if pixel_km <= 0:
        raise ValueError(f"pixel_km must be > 0, got {pixel_km}")
    if timestep_min <= 0:
        raise ValueError(f"timestep_min must be > 0, got {timestep_min}")
    if downsample_factor < 1:
        raise ValueError(f"downsample_factor must be >= 1, got {downsample_factor}")

    f = int(downsample_factor)
    # Velocity on the product grid, in PRODUCT pixels per frame.
    vy_ds = vy_arr[::f, ::f] / np.float32(f)
    vx_ds = vx_arr[::f, ::f] / np.float32(f)
    rain_ds = rain[::f, ::f]
    product_pixel_km = float(pixel_km) * f

    # product px/frame → km/h. (product_pixel_km carries the same f the
    # velocity was divided by, so this equals the native-grid conversion.)
    to_kmh = np.float32(product_pixel_km * 60.0 / float(timestep_min))
    east = (vx_ds * to_kmh).astype(np.float32)
    north = (-vy_ds * to_kmh).astype(np.float32)

    nan = np.float32(np.nan)
    blank = np.full(rain_ds.shape, nan, dtype=np.float32)

    finite_v = np.isfinite(east) & np.isfinite(north)
    with np.errstate(invalid="ignore"):
        coverage = np.isfinite(rain_ds)
        support = coverage & (rain_ds >= np.float32(support_threshold_mm_h))
    if not support.any():
        # No echo in the composite: nothing to attach a motion estimate to.
        return blank, blank.copy()

    # Rain weights for both the fill and the global fallback. A support
    # pixel whose velocity is unusable carries no information, so it gets
    # no weight — but it still counts as echo for the distance field.
    weights = np.where(support & finite_v, rain_ds, np.float32(0.0)).astype(np.float32)
    w_sum = float(weights.sum(dtype=np.float64))
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        # Echo, but no usable velocity anywhere on it.
        return blank, blank.copy()

    east_clean = np.where(finite_v, east, np.float32(0.0)).astype(np.float32)
    north_clean = np.where(finite_v, north, np.float32(0.0)).astype(np.float32)

    # Distance to the nearest echo, in km, with the full-weight halo (in
    # NATIVE pixels, as in ``complete_flow``) subtracted off.
    halo_km = max(0.0, float(dilation_px)) * float(pixel_km)
    d_km = distance_to_support(support) * np.float32(product_pixel_km)
    np.subtract(d_km, np.float32(halo_km), out=d_km)
    np.maximum(d_km, np.float32(0.0), out=d_km)

    # Rain-weighted mean over the whole composite: the σ→∞ end of the
    # ladder, always defined now that support exists.
    global_east = np.float32(
        float((weights * east_clean).sum(dtype=np.float64)) / w_sum
    )
    global_north = np.float32(
        float((weights * north_clean).sum(dtype=np.float64)) / w_sum
    )

    scales = tuple(sorted(float(s) for s in fill_scales_km if float(s) > 0.0))

    if scales:
        # Normalised convolution at each fixed scale.
        est_east: list[np.ndarray] = []
        est_north: list[np.ndarray] = []
        w_east = weights * east_clean
        w_north = weights * north_clean
        for sigma_km in scales:
            sigma_px = sigma_km / product_pixel_km
            den = gaussian_filter(weights, sigma_px, mode="constant", cval=0.0)
            num_e = gaussian_filter(w_east, sigma_px, mode="constant", cval=0.0)
            num_n = gaussian_filter(w_north, sigma_px, mode="constant", cval=0.0)
            positive = den > 0
            with np.errstate(invalid="ignore", divide="ignore"):
                est_east.append(
                    np.where(positive, num_e / den, global_east).astype(np.float32)
                )
                est_north.append(
                    np.where(positive, num_n / den, global_north).astype(np.float32)
                )

        # Per-pixel scale: the smallest of the ladder whose 2σ reaches the
        # nearest echo. Interpolated in 1/σ between the two bracketing
        # ladder entries (the global mean is the 1/σ = 0 entry), so the
        # field has no seam where the scale changes.
        sigma_target = np.maximum(d_km * np.float32(0.5), np.float32(scales[0]))
        q_target = (np.float32(1.0) / sigma_target).astype(np.float32)

        q_nodes = [0.0] + [1.0 / s for s in reversed(scales)]
        # Rounding must never push a pixel past the finest node and out of
        # every segment below — that would silently leave it on the global
        # mean.
        np.minimum(q_target, np.float32(q_nodes[-1]), out=q_target)
        nodes_east: list = [global_east] + list(reversed(est_east))
        nodes_north: list = [global_north] + list(reversed(est_north))

        prior_east = np.full(rain_ds.shape, global_east, dtype=np.float32)
        prior_north = np.full(rain_ds.shape, global_north, dtype=np.float32)
        for k in range(len(q_nodes) - 1):
            lo, hi = q_nodes[k], q_nodes[k + 1]
            sel = (q_target > np.float32(lo)) & (q_target <= np.float32(hi))
            if not sel.any():
                continue
            t = ((q_target[sel] - np.float32(lo)) / np.float32(hi - lo)).astype(np.float32)
            for nodes, out in (
                (nodes_east, prior_east),
                (nodes_north, prior_north),
            ):
                a = nodes[k][sel] if isinstance(nodes[k], np.ndarray) else nodes[k]
                b = nodes[k + 1][sel] if isinstance(nodes[k + 1], np.ndarray) else nodes[k + 1]
                out[sel] = a + t * (b - a)
    else:
        prior_east = np.full(rain_ds.shape, global_east, dtype=np.float32)
        prior_north = np.full(rain_ds.shape, global_north, dtype=np.float32)

    # Handover from the measured flow to the fill. w = 1 on the echo and
    # its halo, so there the served value IS the raw estimate.
    if float(efold_km) > 0.0:
        weight = np.exp(-d_km / np.float32(float(efold_km))).astype(np.float32)
    else:
        weight = (d_km <= np.float32(0.0)).astype(np.float32)
    weight = np.where(finite_v, weight, np.float32(0.0)).astype(np.float32)

    one = np.float32(1.0)
    display_east = weight * east_clean + (one - weight) * prior_east
    display_north = weight * north_clean + (one - weight) * prior_north

    valid = coverage & np.isfinite(display_east) & np.isfinite(display_north)
    return (
        np.where(valid, display_east, nan).astype(np.float32),
        np.where(valid, display_north, nan).astype(np.float32),
    )
