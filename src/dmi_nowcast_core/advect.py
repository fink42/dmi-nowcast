"""Semi-Lagrangian backward advection.

Plan §6.6 Method A. For each pixel in the output at time ``t+Δ``, trace its
trajectory backward to time ``t`` and sample the current field there. This is
the same approach pysteps' ``nowcasts.extrapolation`` uses, and is more stable
than forward semi-Lagrangian (every output pixel gets exactly one source).

For the backtest baseline we only need a single-point trace (the home pixel),
so we expose both ``advect_point`` and ``advect_field``. The full-field version
is needed for visualisation in Phase 6.

**Why this delegates to the vendored pysteps extrapolator** (R5 fix). The
original implementation was a *one-shot Euler back-step*::

    src_y = ys - vy * (horizon / dt)      # v sampled at the DESTINATION

which samples the velocity at the destination pixel and assumes it holds
over the whole horizon. On a Farnebäck field that is (near) zero away from
the echo, every destination pixel ahead of the rain samples its own zero
velocity, sources itself, and stays dry forever — the "barrier" artifact:
advected rain stalls and dissolves along a stationary line ~20-30 km ahead
of the current echo. Measured on a real case at lead 60: bulk displacement
19 px where the flow implied 61 px, 38 % of wet pixels dried out, and
fold/shock discontinuities (``det(I - s∇v) <= 0``) covered 18 % of the wet
area.

The fix has two halves. This module is the second one: a real trajectory
integration (midpoint rule, sub-stepped at one inter-frame interval per
step) through
``dmi_nowcast_core._vendor.pysteps_steps.extrapolation.semilagrangian``,
which is the well-tested upstream scheme. The first half is
:func:`dmi_nowcast_core.dense_flow.complete_flow`, which gives the far
field a velocity to be advected *by* — without it a correct integrator
still walks nowhere off-echo.

**Velocity convention.** Ours is ``(vy, vx)`` in pixels per ``dt_minutes``,
positive vy = downward/south, positive vx = rightward/east. pysteps stacks
``(2, m, n)`` as ``[0] = x-component, [1] = y-component`` (see
``semilagrangian.interpolate_motion``, which reads ``velocity[0]`` into
``velocity_inc_x``, and ``probabilistic.run_ensemble``'s identical
``np.stack([vx, vy])``), so we pass ``np.stack([vx, vy])`` with
``vel_timestep=1`` and horizons expressed in frames.

**NaN semantics (unchanged, and load-bearing).** Sources outside the grid
come back NaN — genuinely unknown data, not zero rain, because the
composite's coverage edge is a real edge. NaN (nodata) inside the field
stays NaN and bleeds by at most the one-pixel bilinear stencil, exactly as
the previous bilinear sampler did.
"""
from __future__ import annotations

import math
from typing import Iterable, Iterator, Sequence

import numpy as np

#: Longest trajectory sub-step, in inter-frame intervals. The vendored
#: scheme integrates one ``timesteps`` entry at a time with a midpoint
#: correction; pysteps' own convention is one unit timestep per step, and
#: on a sheared field a single 6-frame jump visibly under-shoots a
#: 6 × 1-frame integration.
_MAX_SUBSTEP_FRAMES = 1.0

#: Hard cap on sub-steps per horizon, so an absurd ``horizon_minutes``
#: cannot turn one call into thousands of ``map_coordinates`` passes.
_MAX_SUBSTEPS = 64

_extrapolate_fn = None


def _semilagrangian():
    """Lazily import the vendored extrapolator.

    Deferred because importing anything under ``_vendor.pysteps_steps``
    pulls in the whole STEPS import chain (cascade, noise, timeseries);
    ``import dmi_nowcast_core.advect`` should stay cheap.
    """
    global _extrapolate_fn
    if _extrapolate_fn is None:
        from ._vendor.pysteps_steps.extrapolation.semilagrangian import extrapolate

        _extrapolate_fn = extrapolate
    return _extrapolate_fn


def advect_point(
    row: float,
    col: float,
    vy: np.ndarray,
    vx: np.ndarray,
    *,
    horizon_minutes: float,
    dt_minutes: float = 10.0,
) -> tuple[float, float]:
    """Backward-trace a single point.

    Returns the (row, col) in the current frame that will arrive at the input
    (row, col) after ``horizon_minutes`` of advection by the given flow field
    (which is expressed in pixels per ``dt_minutes``).
    """
    scale = horizon_minutes / dt_minutes
    h, w = vy.shape
    r = max(0, min(h - 1, int(round(row))))
    c = max(0, min(w - 1, int(round(col))))
    return row - float(vy[r, c]) * scale, col - float(vx[r, c]) * scale


def advect_field(
    field: np.ndarray,
    vy: np.ndarray,
    vx: np.ndarray,
    *,
    horizon_minutes: float,
    dt_minutes: float = 10.0,
) -> np.ndarray:
    """Semi-Lagrangian backward advection of the full field.

    ``vy, vx`` are pixels per ``dt_minutes``. Output has the same shape and
    floating dtype; pixels whose backward-traced trajectory leaves the grid
    are returned as NaN (unknown, *not* dry), and NaN in the input stays NaN.

    The trajectory is integrated with the vendored pysteps semi-Lagrangian
    scheme in sub-steps of at most one inter-frame interval, so the velocity
    is sampled *along* the path rather than once at the destination.

    Non-finite velocity components are treated as zero motion (the sidecar
    already sanitises the flow field; this only keeps a stray NaN from
    poisoning a whole trajectory).
    """
    _check_shapes(field, vy, vx)
    if dt_minutes <= 0:
        raise ValueError(f"dt_minutes must be > 0, got {dt_minutes}")
    scale = float(horizon_minutes) / float(dt_minutes)
    if scale < 0:
        # Advecting backwards in time is the same integration with the flow
        # reversed, and keeps the "non-decreasing horizons" contract below.
        vy, vx = -np.asarray(vy), -np.asarray(vx)
        scale = -scale
    fld = _as_float(field)
    velocity = _velocity_stack(vy, vx)
    return next(iter(_integrate(fld, velocity, [scale])))


def advect_field_series(
    field: np.ndarray,
    vy: np.ndarray,
    vx: np.ndarray,
    *,
    horizons_minutes: Sequence[float] | Iterable[float],
    dt_minutes: float = 10.0,
) -> Iterator[np.ndarray]:
    """Advect ``field`` to several horizons in one integration pass.

    Yields one advected field per entry of ``horizons_minutes``, which must
    be non-negative and non-decreasing. Semantics per field are identical to
    :func:`advect_field`; the difference is cost — the trajectory is carried
    forward between horizons (pysteps' ``displacement_prev``) instead of
    being re-integrated from scratch for every lead, which is what an
    animation loop or a per-lead forecast table actually wants.

    Only one advected field is alive at a time (the return value is a
    generator), so the memory profile matches calling :func:`advect_field`
    in a loop. Arguments are validated eagerly, before the first field.
    """
    horizons = [float(h) for h in horizons_minutes]
    _check_shapes(field, vy, vx)
    if dt_minutes <= 0:
        raise ValueError(f"dt_minutes must be > 0, got {dt_minutes}")
    if any(h < 0 for h in horizons):
        raise ValueError("horizons_minutes must be non-negative")
    if any(b < a for a, b in zip(horizons, horizons[1:])):
        raise ValueError("horizons_minutes must be non-decreasing")
    scales = [h / float(dt_minutes) for h in horizons]
    return _integrate(_as_float(field), _velocity_stack(vy, vx), scales)


def _integrate(
    field: np.ndarray,
    velocity: np.ndarray,
    scales: Sequence[float],
) -> Iterator[np.ndarray]:
    """Yield ``field`` advected to each scale (in frames), non-decreasing."""
    if not np.any(np.isfinite(field)):
        # The vendored extrapolator refuses an all-nodata field; warping one
        # is a no-op anyway.
        for _ in scales:
            yield np.full_like(field, np.nan)
        return

    extrapolate = _semilagrangian()
    displacement = None
    reached = 0.0
    for scale in scales:
        if scale == 0.0 and displacement is None:
            # Exact identity — cheaper than a warp, and it leaves the NaN
            # mask untouched instead of bleeding it by the interpolation
            # stencil.
            yield field.copy()
            continue
        prep, last = _substeps(reached, scale)
        if prep:
            # Integrate the trajectory without touching the field: no
            # per-sub-step output array is allocated this way.
            _, displacement = extrapolate(
                None,
                velocity,
                timesteps=prep,
                displacement_prev=displacement,
                return_displacement=True,
                allow_nonfinite_values=True,
                vel_timestep=1.0,
            )
        warped, displacement = extrapolate(
            field,
            velocity,
            timesteps=[last],
            displacement_prev=displacement,
            return_displacement=True,
            allow_nonfinite_values=True,
            outval=np.nan,
            vel_timestep=1.0,
        )
        reached = scale
        yield warped[-1]


def _substeps(reached: float, target: float) -> tuple[list[float], float]:
    """Sub-steps that carry the trajectory from ``reached`` to ``target``.

    Returns ``(prep, last)``: ``prep`` is the *cumulative* times of every
    sub-step but the final one — the form ``semilagrangian.extrapolate``
    wants for its ``timesteps`` list, measured from the displacement
    already reached — and ``last`` is the remaining single increment, which
    is what the closing call (the one that actually warps the field) is
    given. Getting these two conventions mixed up silently doubles the
    displacement, so they are separated by type.

    Breakpoints are anchored on whole inter-frame intervals (pysteps' own
    convention: one unit timestep per step), which also keeps a series of
    horizons stepping the same way a single horizon does. An advance of
    zero or less returns no prep and a zero-length final step, which
    re-warps the field at the displacement reached so far.
    """
    delta = target - reached
    if delta <= 0:
        return [], 0.0
    unit = _MAX_SUBSTEP_FRAMES
    # Whole-frame breakpoints strictly inside (reached, target), then the
    # target itself. Every gap is then <= one frame.
    first = math.floor(reached / unit + 1e-9) + 1
    points = [
        b * unit for b in range(first, math.ceil(target / unit - 1e-9))
    ]
    points.append(target)
    if len(points) > _MAX_SUBSTEPS:
        # Absurd horizon: a fixed number of equal steps, rather than
        # thousands of map_coordinates passes.
        step = delta / _MAX_SUBSTEPS
        return [step * i for i in range(1, _MAX_SUBSTEPS)], step
    prep = [p - reached for p in points[:-1]]
    last = points[-1] - (points[-2] if len(points) > 1 else reached)
    return prep, last


def _check_shapes(field: np.ndarray, vy: np.ndarray, vx: np.ndarray) -> None:
    if field.shape != vy.shape or field.shape != vx.shape:
        raise ValueError("field, vy, vx must all have the same shape")
    if field.ndim != 2:
        raise ValueError(f"field must be 2-D, got shape {field.shape}")


def _as_float(field: np.ndarray) -> np.ndarray:
    """Float view of ``field`` (NaN needs a floating dtype to survive)."""
    arr = np.asarray(field)
    if np.issubdtype(arr.dtype, np.floating):
        return arr
    return arr.astype(np.float32)


def _velocity_stack(vy: np.ndarray, vx: np.ndarray) -> np.ndarray:
    """``(2, h, w)`` pysteps velocity: ``[0] = x (east), [1] = y (south)``."""
    return np.stack([
        np.nan_to_num(np.asarray(vx, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(np.asarray(vy, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
    ])
