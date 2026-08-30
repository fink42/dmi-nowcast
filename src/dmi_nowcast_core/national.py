"""National forecast products from a STEPS ensemble (website Phase A plan §A1).

Reduces a STEPS ensemble array — the ``(n_members, n_timesteps, h, w)`` mm/h
output of :func:`dmi_nowcast_core.probabilistic.run_ensemble` — into national
forecast grids on the same (downsampled) grid:

- ``p_rain[lead]``: per-pixel fraction of ensemble members whose cumulative
  max rain rate exceeds the detection threshold by that lead time,
- ``eta_min``: per-pixel ensemble-median time until rain arrives,
- ``intensity_mm_h``: per-pixel ensemble-median raw rain rate at arrival.

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

# Detection threshold, mm/h. Same value as
# ``probabilistic.DB_THRESHOLD_MM_H``; restated here so this module's import
# graph stays numpy-only (``probabilistic`` pulls in pyproj via ``geo``).
DEFAULT_THRESHOLD_MM_H = 0.1

# Lead times served on the national products (Phase A plan §A1).
DEFAULT_LEADS_MIN = (10, 20, 30, 45, 60)


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
