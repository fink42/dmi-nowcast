"""Tests for semi-Lagrangian backward advection."""
from __future__ import annotations

import numpy as np
import pytest

from dmi_nowcast_core.advect import advect_field, advect_point


def test_advect_point_shifts_by_negative_flow():
    """A point that arrives at (10, 10) after motion (vy=3, vx=2) over 1 frame
    came from (10 - 3, 10 - 2) = (7, 8). dt=10 min, horizon=10 min → scale=1."""
    vy = np.full((20, 20), 3.0, dtype=np.float32)
    vx = np.full((20, 20), 2.0, dtype=np.float32)
    src_y, src_x = advect_point(10.0, 10.0, vy, vx, horizon_minutes=10.0, dt_minutes=10.0)
    assert src_y == pytest.approx(7.0)
    assert src_x == pytest.approx(8.0)


def test_advect_point_scales_with_horizon():
    """horizon=30 min, dt=10 → scale=3."""
    vy = np.full((20, 20), 2.0, dtype=np.float32)
    vx = np.zeros((20, 20), dtype=np.float32)
    src_y, src_x = advect_point(10.0, 10.0, vy, vx, horizon_minutes=30.0, dt_minutes=10.0)
    assert src_y == pytest.approx(4.0)  # 10 - 2*3
    assert src_x == pytest.approx(10.0)


def test_advect_field_translates_a_blob():
    """A blob in the field, advected by uniform downward motion, should appear shifted downward."""
    field = np.zeros((32, 32), dtype=np.float32)
    field[8, 16] = 10.0  # hotspot near top
    vy = np.full((32, 32), 5.0, dtype=np.float32)  # 5 px/frame downward
    vx = np.zeros((32, 32), dtype=np.float32)
    advected = advect_field(field, vy, vx, horizon_minutes=10.0, dt_minutes=10.0)
    # Hotspot should now appear at row 13 (= 8 + 5).
    assert advected[13, 16] == pytest.approx(10.0, abs=0.1)
    assert advected[8, 16] == pytest.approx(0.0, abs=0.1)


def test_advect_field_bilinear_for_fractional_motion():
    """Half-pixel motion → bilinear weighting splits the hotspot across two rows."""
    field = np.zeros((32, 32), dtype=np.float32)
    field[8, 16] = 10.0
    vy = np.full((32, 32), 0.5, dtype=np.float32)
    vx = np.zeros((32, 32), dtype=np.float32)
    advected = advect_field(field, vy, vx, horizon_minutes=10.0, dt_minutes=10.0)
    # Backward trace from (8, 16) → (7.5, 16) → bilinear gives 0.5*field[7]+0.5*field[8]
    # But field[7,16] is 0 and field[8,16] is 10 → backward sample = 5.
    # From (9, 16) → backward trace to (8.5, 16) → 0.5*10+0.5*0 = 5.
    assert advected[8, 16] == pytest.approx(5.0, abs=0.1)
    assert advected[9, 16] == pytest.approx(5.0, abs=0.1)


def test_advect_field_out_of_bounds_becomes_nan():
    """Large motion pulls some destination pixels from off-grid → NaN."""
    field = np.ones((16, 16), dtype=np.float32)
    vy = np.full((16, 16), 100.0, dtype=np.float32)  # ridiculous
    vx = np.zeros((16, 16), dtype=np.float32)
    advected = advect_field(field, vy, vx, horizon_minutes=10.0, dt_minutes=10.0)
    assert np.isnan(advected).all()


def test_advect_field_zero_motion_is_identity():
    rng = np.random.default_rng(7)
    field = rng.standard_normal((24, 24)).astype(np.float32)
    vy = np.zeros_like(field)
    vx = np.zeros_like(field)
    advected = advect_field(field, vy, vx, horizon_minutes=20.0, dt_minutes=10.0)
    np.testing.assert_allclose(advected, field, atol=1e-6)


def test_advect_field_shape_mismatch_raises():
    field = np.zeros((10, 10), dtype=np.float32)
    vy = np.zeros((10, 11), dtype=np.float32)
    vx = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        advect_field(field, vy, vx, horizon_minutes=10.0)
