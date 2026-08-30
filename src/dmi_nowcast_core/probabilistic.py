"""Probabilistic forecast via pysteps STEPS ensemble (plan §6.6 method B, §7.1).

Wraps ``pysteps.nowcasts.steps.forecast`` for our specific needs. STEPS expects
``(ar_order+1, h, w)`` precipitation fields in dB plus a single (2, h, w)
velocity field. We feed it our Farnebäck flow and three consecutive dBZ
frames (we convert dBZ → rain rate → dB before passing in, since STEPS
operates on log-rain-rate, not log-reflectivity).

The output is a probability per lead time and an ETA window, as in plan §5.2.

**Backend**: imports from the vendored ``_vendor.pysteps_steps`` subset
(20 .py files from pysteps 1.21.1 — see ``_vendor/pysteps_steps/NOTICE``).
Upstream pysteps has no installable wheels on HA OS, and source builds need
a C toolchain we can't get on Alpine + cp314, so we ship the small subset
that STEPS actually executes. ``EnsembleUnavailable`` is still raised on
ImportError so unit tests can mock the vendored package out.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .geo import CompositeGeo
from .sample import disc_pixel_indices
from .transform import dbz_to_rain_rate


class EnsembleUnavailable(RuntimeError):
    """Raised when the STEPS forecast can't run.

    Originally raised when upstream pysteps wasn't installed; with the
    vendored ``_vendor.pysteps_steps`` subset that never happens in normal
    operation. Still raised when a test monkey-patches the import path or
    when the vendored copy is somehow missing (corrupt deploy), so the
    integration's existing graceful-degradation path keeps working.
    """

# Convert mm/h ↔ dB for pysteps' STEPS, which operates on log-rain-rate.
ZERO_DB = -15.0  # value substituted where rain rate < threshold (pysteps convention)
DB_THRESHOLD_MM_H = 0.1  # mm/h corresponding to dB threshold


@dataclass(frozen=True)
class HomeProbabilisticForecast:
    """Aggregated ensemble forecast at the home location.

    ``probability_by_lead_min[t]`` = fraction of ensemble members whose
    max-in-disc rain rate at lead time t exceeds the detection threshold.
    """

    leads_min: tuple[float, ...]
    probability_by_lead: tuple[float, ...]
    eta_p25_min: float
    eta_p50_min: float
    eta_p75_min: float
    n_members: int


def rain_to_db(rain: np.ndarray, threshold_mm_h: float = DB_THRESHOLD_MM_H) -> np.ndarray:
    """Rain rate (mm/h) → dB = 10·log10(R). Sub-threshold → ZERO_DB (a constant
    background pysteps uses for the AR(2) noise model)."""
    arr = np.asarray(rain, dtype=np.float32)
    above = arr >= threshold_mm_h
    out = np.full(arr.shape, np.float32(ZERO_DB))
    with np.errstate(divide="ignore", invalid="ignore"):
        out[above] = (10.0 * np.log10(arr[above])).astype(np.float32)
    return out


def db_to_rain(db: np.ndarray) -> np.ndarray:
    """Inverse of ``rain_to_db``. Values at ZERO_DB collapse to 0 mm/h."""
    arr = np.asarray(db, dtype=np.float32)
    out = np.power(np.float32(10.0), arr / np.float32(10.0))
    out = np.where(arr > ZERO_DB + 1e-3, out, np.float32(0.0))
    return out


def run_ensemble(
    dbz_frames: list[np.ndarray],
    vy: np.ndarray,
    vx: np.ndarray,
    *,
    zr_a: float = 200.0,
    zr_b: float = 1.6,
    n_timesteps: int = 12,
    timestep_min: float = 5.0,
    n_ens_members: int = 20,
    n_cascade_levels: int = 6,
    threshold_mm_h: float = DB_THRESHOLD_MM_H,
    seed: int = 42,
    downsample_factor: int = 4,
    pixel_scale_m: float = 500.0,
) -> np.ndarray:
    """Run pysteps STEPS. Returns ``(n_ens_members, n_timesteps, h, w)`` in mm/h.

    ``dbz_frames`` must be at least 3 frames in chronological order (oldest → newest).
    ``vy, vx`` is the Farnebäck flow between the last two frames (pixels/timestep).

    ``downsample_factor`` slices the input grid before STEPS runs. At the
    DMI native resolution (1728×1984, 500 m/pixel) STEPS at 10 members ×
    12 timesteps × 6 cascades takes ~2 minutes — far beyond a single
    coordinator poll. Slicing by 4 gives 432×496 (still ~2 km/pixel
    effective, much finer than the radar's ~1 km true resolution after
    composite-max smoothing) and the same forecast runs in ~6 s on dev
    hardware. The output shape is the downsampled grid; callers that
    want home-pixel sampling pass the same factor to ``aggregate_at_home``
    so the disc lands in the right place.

    ``pixel_scale_m`` is the native radar grid scale (500 m on DMI);
    needed so velocity perturbation gets a sensible km/pixel after the
    downsample.
    """
    if len(dbz_frames) < 3:
        raise ValueError(f"STEPS needs at least 3 input frames; got {len(dbz_frames)}")
    if downsample_factor < 1:
        raise ValueError(f"downsample_factor must be >= 1, got {downsample_factor}")
    # pysteps wants ``(ar_order+1, h, w)`` — take the last 3.
    rain_frames = np.stack([
        dbz_to_rain_rate(d, zr_a=zr_a, zr_b=zr_b) for d in dbz_frames[-3:]
    ])
    db_frames = rain_to_db(rain_frames, threshold_mm_h=threshold_mm_h)

    # pysteps velocity convention: shape (2, h, w) with [0]=vx (east), [1]=vy (south).
    velocity = np.stack([vx, vy]).astype(np.float32)

    if downsample_factor > 1:
        f = downsample_factor
        # Plain stride slicing — Gaussian + decimation would be more correct
        # but the radar field is already smoothed by the composite-max and
        # 5-min temporal aggregation; the difference doesn't show up in
        # ensemble probabilities.
        db_frames = db_frames[:, ::f, ::f]
        # Velocity is in pixels-per-frame on the NATIVE grid. After
        # downsampling, the same physical motion corresponds to fewer
        # downsampled pixels per frame — divide by ``f``.
        velocity = velocity[:, ::f, ::f] / f
        # Effective pixel size grows by ``f``.
        km_per_pixel = pixel_scale_m * f / 1000.0
    else:
        km_per_pixel = pixel_scale_m / 1000.0

    try:
        from ._vendor.pysteps_steps.nowcasts import steps as ps_steps
    except ImportError as exc:  # noqa: BLE001
        raise EnsembleUnavailable(
            "Vendored pysteps subset failed to import; STEPS can't run. "
            "Check src/dmi_nowcast_core/_vendor/pysteps_steps/ is intact "
            "(20 .py files + 8 __init__.py + LICENSE-pysteps + NOTICE)."
        ) from exc

    forecast_db = ps_steps.forecast(
        precip=db_frames,
        velocity=velocity,
        timesteps=n_timesteps,
        n_ens_members=n_ens_members,
        n_cascade_levels=n_cascade_levels,
        precip_thr=10.0 * np.log10(threshold_mm_h),
        kmperpixel=km_per_pixel,
        timestep=timestep_min,
        seed=seed,
    )
    return db_to_rain(forecast_db)


def frame_age_corrected_leads(
    leads_min: Iterable[float],
    frame_age_min: float,
    *,
    n_timesteps: int,
    timestep_min: float,
) -> tuple[float, ...]:
    """Map nominal "minutes from now" leads onto radar-frame-relative leads.

    Ensemble timesteps count from the radar frame's timestamp, but
    ``state.json`` reports leads as minutes from ``generated_at`` ("now").
    A frame that is ``frame_age_min`` old therefore needs
    ``lead + frame_age_min`` minutes of ensemble horizon to cover
    "``lead`` minutes from now" — the same correction the deterministic
    per-lead loop applies via ``advect_field(horizon=lead + frame_age)``.
    Website Phase A plan §A0 (lead-time bookkeeping); Phase A1's national
    products use the identical convention.

    Each corrected lead is **snapped up to a whole timestep multiple**
    (``ceil``): "rain within L minutes" must cover every timestep whose
    valid time falls inside L, and — critically — it puts
    ``aggregate_at_home``'s ``round(lead / timestep)`` bucketing in exactly
    the same bucket as the national products' ``ceil`` bucketing
    (``national._steps_in_lead``) for ANY fractional frame age. Without the
    snap the two disagree whenever the fractional part of
    ``(lead + age) / timestep`` is below one half (e.g. age 2.0, lead 10:
    round(12/5) = 2 steps vs ceil = 3), which would break the Phase A §A4
    home-pixel agreement guarantee in live operation.

    Corrected leads are clamped to the ensemble horizon
    ``n_timesteps * timestep_min`` and floored at ``timestep_min`` (the
    first timestep — mirrors ``aggregate_at_home``'s
    ``max(1, steps_in_lead)``). A negative ``frame_age_min`` (clock skew)
    is treated as zero.

    Callers select ensemble timesteps with the *corrected* leads but must
    report results under the original nominal labels.
    """
    if n_timesteps < 1:
        raise ValueError(f"n_timesteps must be >= 1, got {n_timesteps}")
    if timestep_min <= 0:
        raise ValueError(f"timestep_min must be > 0, got {timestep_min}")
    age = max(0.0, float(frame_age_min))
    out: list[float] = []
    for lead in leads_min:
        effective = float(lead) + age
        # The tiny epsilon keeps an exact multiple (e.g. 15.000000000000002
        # from float addition) from ceiling into the next timestep.
        k = math.ceil(effective / timestep_min - 1e-9)
        k = max(1, min(k, n_timesteps))
        out.append(k * timestep_min)
    return tuple(out)


def aggregate_at_home(
    forecast: np.ndarray,
    geo: CompositeGeo,
    lon: float,
    lat: float,
    *,
    radius_m: float = 1000.0,
    threshold_mm_h: float = DB_THRESHOLD_MM_H,
    timestep_min: float = 5.0,
    leads_min: Iterable[float] = (10.0, 20.0, 30.0, 60.0),
    downsample_factor: int = 1,
) -> HomeProbabilisticForecast:
    """Reduce an ensemble forecast to per-lead probabilities and an ETA window.

    ``downsample_factor`` must match the value passed to ``run_ensemble``.
    The ``geo`` object describes the native radar grid; the forecast may be
    at a coarser resolution. We divide the home pixel index and the disc
    radius by ``downsample_factor`` to land in the forecast's coordinates.
    """
    n_members, n_timesteps, h, w = forecast.shape
    pixel_scale_m = (geo.composite.xscale_m + geo.composite.yscale_m) / 2.0
    # Effective per-pixel size in the (possibly downsampled) forecast grid.
    eff_pixel_scale_m = pixel_scale_m * max(1, downsample_factor)
    radius_px = radius_m / eff_pixel_scale_m
    home = geo.lonlat_to_grid(lon, lat)
    home_row = home.row / max(1, downsample_factor)
    home_col = home.col / max(1, downsample_factor)
    rows, cols = disc_pixel_indices((h, w), home_row, home_col, radius_px)
    if rows.size == 0:
        raise ValueError("home location is outside the grid")

    # max-in-disc per member per timestep
    max_per_member_step = np.zeros((n_members, n_timesteps), dtype=np.float32)
    for m in range(n_members):
        for t in range(n_timesteps):
            values = forecast[m, t, rows, cols]
            finite = values[np.isfinite(values)]
            max_per_member_step[m, t] = float(finite.max()) if finite.size else 0.0

    # First-exceedance time per member (NaN if member never exceeds).
    exceed = max_per_member_step >= threshold_mm_h
    first_idx = np.where(exceed.any(axis=1), np.argmax(exceed, axis=1), -1)
    first_min = np.where(first_idx >= 0, (first_idx + 1) * timestep_min, np.nan)

    leads = tuple(float(L) for L in leads_min)
    probs = []
    for lead in leads:
        # P(rain in next ``lead`` minutes) = fraction of members exceeding by then
        steps_in_lead = int(round(lead / timestep_min))
        steps_in_lead = max(1, min(steps_in_lead, n_timesteps))
        prob = float(np.mean(exceed[:, :steps_in_lead].any(axis=1)))
        probs.append(prob)

    valid = ~np.isnan(first_min)
    if valid.any():
        eta_q = np.percentile(first_min[valid], [25, 50, 75])
        eta_p25, eta_p50, eta_p75 = (float(x) for x in eta_q)
    else:
        eta_p25 = eta_p50 = eta_p75 = float("nan")

    return HomeProbabilisticForecast(
        leads_min=leads,
        probability_by_lead=tuple(probs),
        eta_p25_min=eta_p25,
        eta_p50_min=eta_p50,
        eta_p75_min=eta_p75,
        n_members=int(n_members),
    )
