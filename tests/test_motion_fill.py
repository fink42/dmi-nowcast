"""The served motion grids' off-echo fill (issue #6).

``national.motion_grids_kmh`` used to publish nodata more than 20 km from
any echo, so a point in Odense with cells plainly visible and moving 50 km
away read "no measured motion". It now fills the whole radar coverage with
the motion of the *nearest* cells: a rain-weighted average over a search
area that starts at 25 km and widens (50, 100 km, then the whole composite)
only when nothing closer is found.

These tests work on a synthetic **product** grid directly — ``pixel_km=2``
with ``downsample_factor=1`` — so every distance in the test is the
distance the algorithm sees, in km, with no stride arithmetic in between.
``dilation_px=0`` for the same reason: the halo is a native-pixel quantity
and would otherwise shift every distance below by a few km.

The one exception is the Odense regression at the bottom, which runs the
real pipeline (parse → Z-R → Farnebäck → grids) on two committed DMI
composites and pins the fix against real radar.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from dmi_nowcast_core.dense_flow import (
    DenseFlowUnavailable,
    dense_flow,
    distance_to_support,
)
from dmi_nowcast_core.geo import CompositeGeo
from dmi_nowcast_core.national import DEFAULT_MOTION_FILL_SCALES_KM, motion_grids_kmh
from dmi_nowcast_core.parse import parse_composite
from dmi_nowcast_core.transform import dbz_to_rain_rate

# Synthetic grid geometry: 2 km pixels, 10 min between frames.
PIXEL_KM = 2.0
TIMESTEP_MIN = 10.0


def _px_per_frame(kmh: float) -> float:
    """km/h → pixels per frame on the synthetic grid."""
    return kmh * TIMESTEP_MIN / 60.0 / PIXEL_KM


def _bearing_toward(east: float, north: float) -> float:
    """Compass bearing the motion points *toward*, degrees clockwise from N."""
    return math.degrees(math.atan2(east, north)) % 360.0


def _angle_between(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _scene(n: int, cells):
    """Build ``(rain, vy, vx, support)`` from ``(row, col, radius_px, east_kmh,
    north_kmh)`` discs on an otherwise rain-free, motion-free grid.

    Zero flow off the echo is what Farnebäck actually returns on a
    featureless plateau, so the synthetic matches the real input.
    """
    rows, cols = np.ogrid[:n, :n]
    rain = np.zeros((n, n), np.float32)
    vy = np.zeros((n, n), np.float32)
    vx = np.zeros((n, n), np.float32)
    support = np.zeros((n, n), bool)
    for row, col, radius, east_kmh, north_kmh in cells:
        disc = (rows - row) ** 2 + (cols - col) ** 2 <= radius * radius
        support |= disc
        rain[disc] = 5.0
        vx[disc] = _px_per_frame(east_kmh)
        vy[disc] = -_px_per_frame(north_kmh)   # rows grow southward
    return rain, vy, vx, support


def _grids(rain, vy, vx, **kwargs):
    kwargs.setdefault("dilation_px", 0.0)
    return motion_grids_kmh(
        vy, vx, rain,
        pixel_km=PIXEL_KM, timestep_min=TIMESTEP_MIN, downsample_factor=1,
        support_threshold_mm_h=0.5, **kwargs,
    )


def test_on_the_echo_the_served_vector_is_the_raw_estimate():
    """Contract with the advection: wherever the flow was actually measured,
    the arrow is that measurement, unmodified. The fill only ever touches
    pixels the optical flow had nothing to track."""
    n = 200
    rain, vy, vx, support = _scene(n, [
        (60, 60, 12, 90.0, -30.0),
        (140, 150, 18, -20.0, 70.0),
    ])
    # Sheared, not uniform: a fill that "happened to agree" would not.
    ramp = np.linspace(-0.5, 0.5, n, dtype=np.float32)[None, :]
    vx = (vx * (1.0 + ramp)).astype(np.float32)

    east, north = _grids(rain, vy, vx)

    to_kmh = PIXEL_KM * 60.0 / TIMESTEP_MIN
    assert np.allclose(east[support], vx[support] * to_kmh, atol=1e-4)
    assert np.allclose(north[support], -vy[support] * to_kmh, atol=1e-4)


def test_nearest_cells_dominate_the_direction():
    """Two systems on the map moving in opposite directions. A point near
    the western cell must be told about the western cell — not about the
    national average of the two, which is what ``complete_flow``'s bulk
    target would have given (and which is zero here by symmetry)."""
    n = 300
    # Centres 150 km apart (75 px), radius 10 km.
    rain, vy, vx, _ = _scene(n, [
        (150, 75, 5, 100.0, 0.0),     # A, western, moving due east
        (150, 150, 5, -100.0, 0.0),   # B, eastern, moving due west
    ])
    east, north = _grids(rain, vy, vx)

    # 30 km from A's edge, 100 km from B's, on the line between them.
    for row, col, want_bearing in ((150, 95, 90.0), (150, 130, 270.0)):
        e, nth = float(east[row, col]), float(north[row, col])
        speed = math.hypot(e, nth)
        bearing = _bearing_toward(e, nth)
        assert _angle_between(bearing, want_bearing) < 10.0, (
            f"pixel {(row, col)} points {bearing:.1f}°, want {want_bearing}°"
        )
        assert abs(speed - 100.0) / 100.0 < 0.15, f"speed {speed:.1f} km/h"

    # And the two halves genuinely disagree — the national mean is 0 km/h.
    assert float(east[150, 95]) > 0.0 > float(east[150, 130])


def test_the_search_widens_until_it_reaches_echo():
    """One cell, and pixels at each rung of the ladder: inside 2×25 km,
    past it, past 2×50 km, and past 2×100 km where the rain-weighted mean
    over the whole composite takes over. All finite, all the cell's
    motion — there is only one thing moving on this map."""
    n = 400
    rain, vy, vx, support = _scene(n, [(200, 120, 5, 0.0, 100.0)])
    east, north = _grids(rain, vy, vx)

    to_kmh = PIXEL_KM * 60.0 / TIMESTEP_MIN
    weights = np.where(support, rain, 0.0)
    global_north = float(
        (weights * (-vy * to_kmh)).sum() / weights.sum()
    )
    assert global_north == pytest.approx(100.0, abs=1e-3)

    distance = distance_to_support(support) * PIXEL_KM
    for offset_km, rung in ((40.0, "2×25"), (80.0, "2×50"), (150.0, "2×100"),
                            (250.0, "past the ladder")):
        col = 120 + 5 + int(offset_km / PIXEL_KM)
        assert distance[200, col] == pytest.approx(offset_km, abs=PIXEL_KM)
        e, nth = float(east[200, col]), float(north[200, col])
        assert np.isfinite(e) and np.isfinite(nth), f"nodata at {rung}"
        assert nth == pytest.approx(100.0, rel=0.02), f"{offset_km} km ({rung})"
        assert abs(e) < 1.0

    # Past 2×max(scales) the vector IS the rain-weighted global mean.
    col = 120 + 5 + int(250.0 / PIXEL_KM)
    assert float(north[200, col]) == pytest.approx(global_north, rel=1e-3)
    assert 2 * max(DEFAULT_MOTION_FILL_SCALES_KM) < 250.0


def test_no_echo_anywhere_is_all_nodata():
    """Nothing is moving, so nothing is reported — the one case where the
    panel still says "no measured cell motion"."""
    n = 64
    rain = np.zeros((n, n), np.float32)
    vy = np.full((n, n), -1.0, np.float32)
    vx = np.full((n, n), 2.0, np.float32)
    east, north = _grids(rain, vy, vx)
    assert np.isnan(east).all() and np.isnan(north).all()


def test_outside_radar_coverage_is_nodata_even_beside_a_cell():
    n = 120
    rain, vy, vx, _ = _scene(n, [(60, 60, 6, 50.0, 0.0)])
    rain[:, :40] = np.nan          # off-composite strip, 8 km from the cell
    east, north = _grids(rain, vy, vx)
    assert np.isnan(east[:, :40]).all()
    assert np.isnan(north[:, :40]).all()
    assert np.isfinite(east[60, 41])


def test_every_covered_pixel_carries_a_vector():
    """Coverage completeness: with any echo at all, the finite set of the
    two grids is exactly the radar coverage mask. This is the property the
    website's sampler relies on to stop drawing "no motion" holes."""
    n = 240
    rain, vy, vx, _ = _scene(n, [(40, 200, 4, 60.0, 10.0)])   # one far corner cell
    rain[200:, :30] = np.nan
    coverage = np.isfinite(rain)
    east, north = _grids(rain, vy, vx)
    assert np.array_equal(np.isfinite(east), coverage)
    assert np.array_equal(np.isfinite(north), coverage)


def test_the_off_echo_field_has_no_seam_where_the_search_widens():
    """Two cells 200 km apart with different speeds, so the 25 / 50 / 100 km
    estimates genuinely disagree (~30 km/h apart at the boundaries). The
    served field is interpolated across the ladder rather than switched, so
    neighbouring pixels stay within a few km/h of each other on a 100 km/h
    field; a hard switch at each 2σ boundary measures ~30 km/h here.

    Measured beyond three e-folds (30 km) from the echo, which is where the
    fill governs — inside that the handover to the raw estimate dominates,
    and the raw field has its own (real) structure.
    """
    n = 350
    rain, vy, vx, support = _scene(n, [
        (175, 70, 5, 100.0, 0.0),
        (175, 170, 5, 40.0, 0.0),
    ])
    east, north = _grids(rain, vy, vx)
    far = distance_to_support(support) * PIXEL_KM >= 30.0
    horizontal = far[:, 1:] & far[:, :-1]
    vertical = far[1:, :] & far[:-1, :]
    assert horizontal.sum() > 1000 and vertical.sum() > 1000

    worst = max(
        float(np.abs(np.diff(grid, axis=axis))[mask].max())
        for grid in (east, north)
        for axis, mask in ((1, horizontal), (0, vertical))
    )
    assert worst < 5.0, f"neighbouring pixels differ by {worst:.1f} km/h"


# --- Odense regression, real radar -------------------------------------------

ODENSE_LON, ODENSE_LAT = 10.39, 55.40
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "odense_20260902"
DOWNSAMPLE = 4
OLD_SUPPORT_RADIUS_KM = 20.0    # the rule issue #6 removed


def test_odense_0400z_gets_the_motion_of_the_cells_it_can_see():
    """Issue #6 itself: 2026-09-02 04:00Z, cells on the map but none within
    20 km of Odense, so the old rule published nodata and the panel said
    "no measured cell motion". The new grids give Odense the motion of the
    cells nearest to it.
    """
    try:
        prev = parse_composite(FIXTURE_DIR / "dk.com.202609020350.500_max.h5")
        now = parse_composite(FIXTURE_DIR / "dk.com.202609020400.500_max.h5")
    except FileNotFoundError:  # pragma: no cover - fixtures are committed
        pytest.skip("Odense composites missing")

    dt_min = (now.timestamp_utc - prev.timestamp_utc).total_seconds() / 60.0
    assert dt_min == pytest.approx(10.0, abs=0.2)

    rain = dbz_to_rain_rate(now.reflectivity_dbz, zr_a=now.zr_a, zr_b=now.zr_b)
    try:
        vy, vx = dense_flow(prev.reflectivity_dbz, now.reflectivity_dbz)
    except DenseFlowUnavailable:  # pragma: no cover - opencv is a hard dep
        pytest.skip("no dense-flow backend installed")
    # The sidecar's sanitise, verbatim (compute.py: nan→0 then clip).
    vy = np.nan_to_num(vy, nan=0.0).astype(np.float32)
    vx = np.nan_to_num(vx, nan=0.0).astype(np.float32)
    np.clip(vy, -30.0, 30.0, out=vy)
    np.clip(vx, -30.0, 30.0, out=vx)

    east, north = motion_grids_kmh(
        vy, vx, rain,
        pixel_km=float(now.xscale_m) / 1000.0, timestep_min=dt_min,
        downsample_factor=DOWNSAMPLE, support_threshold_mm_h=0.5,
    )

    geo = CompositeGeo(now)
    index = geo.lonlat_to_grid(ODENSE_LON, ODENSE_LAT)
    row = round(round(index.row) / DOWNSAMPLE)
    col = round(round(index.col) / DOWNSAMPLE)

    product_pixel_km = float(now.xscale_m) / 1000.0 * DOWNSAMPLE
    rain_ds = rain[::DOWNSAMPLE, ::DOWNSAMPLE]
    support = np.isfinite(rain_ds) & (rain_ds >= 0.5)
    assert support.any(), "the fixture must contain echo somewhere"
    distance_km = distance_to_support(support) * product_pixel_km

    # 1. The old rule would have published nodata here.
    assert distance_km[row, col] > OLD_SUPPORT_RADIUS_KM, (
        f"nearest echo is {distance_km[row, col]:.1f} km away — the old "
        f"{OLD_SUPPORT_RADIUS_KM:.0f} km rule made this pixel nodata"
    )
    assert np.isfinite(rain_ds[row, col]), "Odense is inside radar coverage"

    # 2. The new grids give it a plausible vector.
    e, nth = float(east[row, col]), float(north[row, col])
    assert np.isfinite(e) and np.isfinite(nth), "Odense must get an arrow"
    speed = math.hypot(e, nth)
    bearing = _bearing_toward(e, nth)
    assert 5.0 <= speed <= 120.0, f"speed {speed:.1f} km/h out of range"

    # 3. And it is the motion of the cells nearest to Odense: the
    #    rain-weighted mean of the raw flow over echo within 60 km.
    to_kmh = product_pixel_km * 60.0 / dt_min
    east_raw = (vx[::DOWNSAMPLE, ::DOWNSAMPLE] / DOWNSAMPLE) * to_kmh
    north_raw = -(vy[::DOWNSAMPLE, ::DOWNSAMPLE] / DOWNSAMPLE) * to_kmh
    rows, cols = np.mgrid[0:rain_ds.shape[0], 0:rain_ds.shape[1]]
    near = support & (np.hypot(rows - row, cols - col) * product_pixel_km <= 60.0)
    assert near.sum() >= 5, f"only {near.sum()} echo pixels within 60 km"
    weights = rain_ds[near]
    near_east = float((weights * east_raw[near]).sum() / weights.sum())
    near_north = float((weights * north_raw[near]).sum() / weights.sum())
    near_bearing = _bearing_toward(near_east, near_north)
    near_speed = math.hypot(near_east, near_north)

    assert _angle_between(bearing, near_bearing) < 45.0, (
        f"Odense 04:00Z: served {speed:.1f} km/h toward {bearing:.1f}°, "
        f"nearest cells (<=60 km, n={int(near.sum())}) "
        f"{near_speed:.1f} km/h toward {near_bearing:.1f}°; "
        f"nearest echo {distance_km[row, col]:.1f} km away"
    )
