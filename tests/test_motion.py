"""Tests for phase-correlation mean motion estimation."""
from __future__ import annotations

import numpy as np
import pytest

from dmi_nowcast_core.motion import phase_correlation_shift


def _make_blob(shape: tuple[int, int], center: tuple[int, int], radius: int = 5) -> np.ndarray:
    rows, cols = np.indices(shape)
    dist = np.sqrt((rows - center[0]) ** 2 + (cols - center[1]) ** 2)
    return np.where(dist <= radius, 10.0, 0.0).astype(np.float32)


@pytest.mark.parametrize(
    "dy, dx",
    [(0, 0), (3, 4), (-2, 5), (10, -7), (-6, -8)],
)
def test_recovers_known_shift(dy, dx):
    """A blob shifted by (dy, dx) must yield (dy, dx) from phase correlation."""
    prev = _make_blob((64, 64), (32, 32))
    curr = np.roll(prev, shift=(dy, dx), axis=(0, 1))
    got_dy, got_dx = phase_correlation_shift(prev, curr)
    assert (got_dy, got_dx) == (dy, dx)


def test_zero_field_returns_zero_shift():
    z = np.zeros((32, 32), dtype=np.float32)
    assert phase_correlation_shift(z, z) == (0.0, 0.0)


def test_nan_pixels_are_ignored():
    """NaN inputs must not crash; they're treated as zeros."""
    prev = _make_blob((64, 64), (32, 32))
    curr = np.roll(prev, shift=(3, 4), axis=(0, 1))
    prev_with_nan = prev.copy()
    prev_with_nan[0, 0] = np.nan
    curr_with_nan = curr.copy()
    curr_with_nan[-1, -1] = np.nan
    dy, dx = phase_correlation_shift(prev_with_nan, curr_with_nan)
    assert (dy, dx) == (3, 4)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        phase_correlation_shift(np.zeros((10, 10)), np.zeros((10, 11)))
