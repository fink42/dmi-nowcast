"""Tests for nowcast baselines."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dmi_nowcast_core.baselines import lagrangian_dense, lagrangian_mean, persistence
from dmi_nowcast_core.geo import CompositeGeo
from dmi_nowcast_core.parse import parse_composite

FIXTURE = Path(__file__).parent / "fixtures" / "composite_fullrange.h5"
COPENHAGEN = (12.5645, 55.6726)


@pytest.fixture(scope="module")
def geo() -> CompositeGeo:
    return CompositeGeo(parse_composite(FIXTURE))


def test_persistence_returns_current_disc_stats(geo):
    field = np.zeros(geo.composite.reflectivity_dbz.shape, dtype=np.float32)
    idx = geo.lonlat_to_grid(*COPENHAGEN)
    rc, cc = int(round(idx.row)), int(round(idx.col))
    field[rc - 1 : rc + 2, cc - 1 : cc + 2] = 3.0
    stats = persistence(field, geo, *COPENHAGEN)
    assert stats.max_mm_h == pytest.approx(3.0)


def test_lagrangian_recovers_upstream_rain_with_known_motion(geo):
    """If the rain field has moved (dy, dx) from prev to now, then in `horizon`
    minutes it will have moved another (dy, dx)*scale. So the rain at home in
    future is what is now (dy, dx)*scale upstream of home."""
    shape = geo.composite.reflectivity_dbz.shape
    idx = geo.lonlat_to_grid(*COPENHAGEN)
    rc, cc = int(round(idx.row)), int(round(idx.col))

    # Plant a hotspot 10 pixels north (upstream) of home in the *current* field.
    rain_now = np.zeros(shape, dtype=np.float32)
    rain_now[rc - 10, cc] = 5.0
    rain_now[rc - 10, cc + 1] = 5.0
    rain_now[rc - 10, cc - 1] = 5.0
    rain_now[rc - 9, cc] = 5.0
    rain_now[rc - 11, cc] = 5.0

    # Build `rain_prev` so the field moved south-by-10 between prev and now,
    # i.e. dy = +10, dx = 0. Hotspot in prev was at row rc-20.
    rain_prev = np.roll(rain_now, shift=(-10, 0), axis=(0, 1))

    # With 5-minute frame interval and 5-minute horizon, scale = 1.
    # Sampling at (rc - 1*10, cc) should pick up the planted hotspot in rain_now.
    stats = lagrangian_mean(
        rain_now, rain_prev, geo, *COPENHAGEN,
        horizon_minutes=5.0, dt_minutes=5.0, radius_m=1500.0,
    )
    assert stats.max_mm_h == pytest.approx(5.0), (
        f"Expected to recover upstream hotspot; got {stats}"
    )


def test_lagrangian_with_zero_motion_equals_persistence(geo):
    """No motion → Lagrangian reduces to persistence."""
    shape = geo.composite.reflectivity_dbz.shape
    rain = np.zeros(shape, dtype=np.float32)
    idx = geo.lonlat_to_grid(*COPENHAGEN)
    rc, cc = int(round(idx.row)), int(round(idx.col))
    rain[rc - 1 : rc + 2, cc - 1 : cc + 2] = 2.0

    pers = persistence(rain, geo, *COPENHAGEN)
    lag = lagrangian_mean(rain, rain, geo, *COPENHAGEN, horizon_minutes=10.0)
    assert lag.max_mm_h == pytest.approx(pers.max_mm_h)


def test_lagrangian_dense_with_zero_flow_equals_persistence(geo):
    """Zero flow field → dense Lagrangian reduces to persistence."""
    shape = geo.composite.reflectivity_dbz.shape
    rain = np.zeros(shape, dtype=np.float32)
    idx = geo.lonlat_to_grid(*COPENHAGEN)
    rc, cc = int(round(idx.row)), int(round(idx.col))
    rain[rc - 1 : rc + 2, cc - 1 : cc + 2] = 2.0
    vy = np.zeros(shape, dtype=np.float32)
    vx = np.zeros(shape, dtype=np.float32)
    pers = persistence(rain, geo, *COPENHAGEN)
    lag = lagrangian_dense(rain, vy, vx, geo, *COPENHAGEN, horizon_minutes=30.0)
    assert lag.max_mm_h == pytest.approx(pers.max_mm_h)


def test_lagrangian_dense_traces_to_upstream_pixel(geo):
    """With uniform downward flow (vy=5, vx=0), home pixel's rain in 1 frame
    comes from 5 pixels north (upstream) of home."""
    shape = geo.composite.reflectivity_dbz.shape
    idx = geo.lonlat_to_grid(*COPENHAGEN)
    rc, cc = int(round(idx.row)), int(round(idx.col))

    rain_now = np.zeros(shape, dtype=np.float32)
    rain_now[rc - 5, cc] = 7.0  # hotspot 5 px upstream of home

    vy = np.full(shape, 5.0, dtype=np.float32)
    vx = np.zeros(shape, dtype=np.float32)
    lag = lagrangian_dense(
        rain_now, vy, vx, geo, *COPENHAGEN,
        horizon_minutes=10.0, dt_minutes=10.0, radius_m=750.0,
    )
    assert lag.max_mm_h == pytest.approx(7.0)
