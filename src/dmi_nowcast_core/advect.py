"""Semi-Lagrangian backward advection.

Plan §6.6 Method A. For each pixel in the output at time ``t+Δ``, trace its
trajectory backward to time ``t`` and sample the current field there. This is
the same approach pysteps' ``nowcasts.extrapolation`` uses, and is more stable
than forward semi-Lagrangian (every output pixel gets exactly one source).

For the backtest baseline we only need a single-point trace (the home pixel),
so we expose both ``advect_point`` and ``advect_field``. The full-field version
is needed for visualisation in Phase 6.
"""
from __future__ import annotations

import numpy as np


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

    Output has the same shape; pixels whose backward-traced source is outside
    the grid are returned as NaN.
    """
    if field.shape != vy.shape or field.shape != vx.shape:
        raise ValueError("field, vy, vx must all have the same shape")
    h, w = field.shape
    scale = horizon_minutes / dt_minutes
    ys, xs = np.indices((h, w), dtype=np.float32)
    src_y = ys - vy * np.float32(scale)
    src_x = xs - vx * np.float32(scale)
    return _bilinear_sample(field, src_y, src_x)


def _bilinear_sample(field: np.ndarray, sy: np.ndarray, sx: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of ``field`` at fractional (sy, sx). NaN on OOB."""
    h, w = field.shape
    y0 = np.floor(sy).astype(np.int64)
    x0 = np.floor(sx).astype(np.int64)
    y1 = y0 + 1
    x1 = x0 + 1
    # Validity is "source point lies within the grid"; we then clamp y1/x1 so
    # the bilinear at the boundary degenerates to the boundary row/column
    # (fy or fx will be 0, the off-grid neighbour gets zero weight).
    valid = (sy >= 0) & (sy <= h - 1) & (sx >= 0) & (sx <= w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    fy = (sy - y0).astype(np.float32)
    fx = (sx - x0).astype(np.float32)
    f00 = field[y0c, x0c]
    f01 = field[y0c, x1c]
    f10 = field[y1c, x0c]
    f11 = field[y1c, x1c]
    out = (
        (1 - fy) * (1 - fx) * f00
        + (1 - fy) * fx * f01
        + fy * (1 - fx) * f10
        + fy * fx * f11
    )
    return np.where(valid, out, np.float32(np.nan))
