"""R5 acceptance tests: motion-field completion + real semi-Lagrangian advection.

The bug these pin down is the "barrier" artifact in the animated national
overlay: advected rain stalls and dissolves along a stationary line 20-30 km
ahead of the current echo. Two compounding causes, one test file:

1. ``dense_flow`` fills nodata with a flat -32 dBZ before Farnebäck, so
   featureless areas return *exactly zero* motion. Rain advected into them
   has nowhere to come from. → :func:`dmi_nowcast_core.dense_flow.complete_flow`.
2. ``advect_field`` was a one-shot Euler back-step that sampled the velocity
   at the *destination* pixel, so a dry destination sampled its own zero
   velocity and stayed dry forever. → ``advect_field`` now integrates the
   trajectory (vendored pysteps semi-Lagrangian, sub-stepped).

Everything here is synthetic and deterministic — no radar files, no RNG.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from dmi_nowcast_core.advect import advect_field, advect_field_series
from dmi_nowcast_core.dense_flow import complete_flow

PIXEL_KM = 0.5  # DMI 500 m composite
DT_MIN = 10.0  # fullRange frame spacing
THRESHOLD = 0.5  # config default rain threshold, mm/h


def gaussian_blob(
    shape: tuple[int, int],
    row: float,
    col: float,
    *,
    sigma: float = 14.0,
    peak: float = 8.0,
) -> np.ndarray:
    """A single smooth rain cell, mm/h."""
    y, x = np.indices(shape).astype(np.float32)
    return (peak * np.exp(-(((y - row) ** 2 + (x - col) ** 2) / (2 * sigma**2)))).astype(
        np.float32
    )


def farneback_like_flow(
    rain: np.ndarray,
    vy_val: float,
    vx_val: float = 0.0,
    *,
    halo_px: float = 10.0,
    decay_px: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Motion that exists only where there is echo to track.

    Full storm motion on the echo and for ``halo_px`` beyond it (Farnebäck's
    31-px window straddles the echo edge), then an exponential decay to
    zero — the measured shape of a real Farnebäck field, where the median
    |v| falls from ~25 px/frame near the echo to ~0 beyond 40-60 px.
    """
    distance = distance_transform_edt(rain < THRESHOLD).astype(np.float32)
    weight = np.where(
        distance <= halo_px, 1.0, np.exp(-(distance - halo_px) / decay_px)
    ).astype(np.float32)
    return (vy_val * weight).astype(np.float32), (vx_val * weight).astype(np.float32)


def centroid_row(field: np.ndarray) -> float:
    """Mass-weighted mean row (NaN counted as no rain)."""
    mass = np.nan_to_num(field, nan=0.0)
    rows = np.indices(mass.shape)[0]
    return float((mass * rows).sum() / mass.sum())


def interior_mass(field: np.ndarray, pad: int = 10) -> float:
    """Total rain away from the grid edge, where inflow is unknown anyway."""
    return float(np.nansum(field[pad:-pad, pad:-pad]))


# ---------------------------------------------------------------------------
# 1. Completion: bulk far field, untouched near field
# ---------------------------------------------------------------------------


def test_completion_keeps_the_estimated_flow_on_the_echo():
    """On the echo the Farnebäck estimate is the best information we have."""
    rain = gaussian_blob((220, 220), 110, 110)
    on_echo = rain >= THRESHOLD
    vy = np.where(on_echo, np.float32(6.0), np.float32(0.0)).astype(np.float32)
    vx = np.where(on_echo, np.float32(-4.0), np.float32(0.0)).astype(np.float32)

    out_vy, out_vx = complete_flow(
        vy, vx, rain, pixel_km=PIXEL_KM, support_threshold_mm_h=THRESHOLD
    )

    np.testing.assert_allclose(out_vy[on_echo], 6.0, atol=1e-4)
    np.testing.assert_allclose(out_vx[on_echo], -4.0, atol=1e-4)


def test_completion_far_field_becomes_the_bulk_vector():
    """Far from any echo the completed flow is the rain-weighted storm motion."""
    rain = gaussian_blob((220, 220), 40, 40)
    on_echo = rain >= THRESHOLD
    vy = np.where(on_echo, np.float32(6.0), np.float32(0.0)).astype(np.float32)
    vx = np.where(on_echo, np.float32(-4.0), np.float32(0.0)).astype(np.float32)

    out_vy, out_vx = complete_flow(
        vy, vx, rain, pixel_km=PIXEL_KM, support_threshold_mm_h=THRESHOLD, efold_km=5.0
    )

    # Opposite corner: ~200 px away, i.e. 10 e-folds.
    far = (slice(-30, None), slice(-30, None))
    np.testing.assert_allclose(out_vy[far], 6.0, atol=1e-3)
    np.testing.assert_allclose(out_vx[far], -4.0, atol=1e-3)
    # Most of the grid was zero before and is moving now.
    assert float(np.mean(np.hypot(vy, vx) < 0.5)) > 0.8
    assert float(np.mean(np.hypot(out_vy, out_vx) < 0.5)) < 0.02


def test_completion_relaxes_monotonically_with_distance():
    """The blend weight is exp(-d/tau): closer to the echo means more Farnebäck."""
    rain = gaussian_blob((300, 60), 30, 30)
    on_echo = rain >= THRESHOLD
    vy = np.where(on_echo, np.float32(8.0), np.float32(0.0)).astype(np.float32)
    vx = np.zeros_like(vy)

    out_vy, _ = complete_flow(
        vy, vx, rain, pixel_km=PIXEL_KM, support_threshold_mm_h=THRESHOLD
    )

    # Straight down the column through the blob centre, past the echo edge.
    profile = out_vy[80:280, 30]
    assert np.all(np.diff(profile) > 0), "completed flow must approach bulk monotonically"
    assert profile[0] < 0.7 * 8.0  # near the echo the estimate still dominates
    assert profile[-1] == pytest.approx(8.0, abs=0.1)  # bulk far away


def test_completion_without_echo_returns_the_input_unchanged():
    """No echo, no bulk vector — completion has nothing to say, so it says nothing.

    (The degenerate alternative, relaxing toward (0, 0), would silently
    delete a mean-motion fallback field.)
    """
    dry = np.zeros((64, 64), dtype=np.float32)
    vy = np.full((64, 64), 3.0, dtype=np.float32)
    vx = np.full((64, 64), -1.5, dtype=np.float32)

    out_vy, out_vx = complete_flow(
        vy, vx, dry, pixel_km=PIXEL_KM, support_threshold_mm_h=THRESHOLD
    )

    np.testing.assert_array_equal(out_vy, vy)
    np.testing.assert_array_equal(out_vx, vx)


def test_completion_below_threshold_echo_counts_as_no_echo():
    """Drizzle under the detection threshold is not support."""
    rain = gaussian_blob((64, 64), 32, 32, peak=THRESHOLD / 2)
    vy = np.full((64, 64), 2.0, dtype=np.float32)
    vx = np.zeros((64, 64), dtype=np.float32)

    out_vy, out_vx = complete_flow(
        vy, vx, rain, pixel_km=PIXEL_KM, support_threshold_mm_h=THRESHOLD
    )

    np.testing.assert_array_equal(out_vy, vy)
    np.testing.assert_array_equal(out_vx, vx)


def test_completion_replaces_non_finite_velocity_with_bulk():
    rain = gaussian_blob((128, 128), 64, 64)
    on_echo = rain >= THRESHOLD
    vy = np.where(on_echo, np.float32(5.0), np.float32(0.0)).astype(np.float32)
    vx = np.zeros_like(vy)
    vy[100:110, 100:110] = np.nan  # a patch the estimator gave up on

    out_vy, out_vx = complete_flow(
        vy, vx, rain, pixel_km=PIXEL_KM, support_threshold_mm_h=THRESHOLD
    )

    assert np.isfinite(out_vy).all() and np.isfinite(out_vx).all()
    np.testing.assert_allclose(out_vy[100:110, 100:110], 5.0, atol=1e-4)


def test_completion_rejects_mismatched_shapes_and_bad_pixel_size():
    vy = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        complete_flow(vy, np.zeros((8, 9), dtype=np.float32), np.zeros((8, 8)), pixel_km=0.5)
    with pytest.raises(ValueError):
        complete_flow(vy, vy, np.zeros((8, 8)), pixel_km=0.0)


# ---------------------------------------------------------------------------
# 2. Propagation regression — the bug's signature
# ---------------------------------------------------------------------------


def test_propagation_stalls_without_completion_and_carries_with_it():
    """The barrier, in one test, asserted in both directions.

    A blob moving 10 px/frame for 60 min should travel 60 px. Driven by the
    raw echo-only flow it falls well short and loses most of its mass on the
    way (that dissolving leading edge *is* the artifact users see). Driven
    by the completed flow it lands within 20 % of the flow-implied distance
    with the mass still there.
    """
    shape = (420, 220)
    speed_px = 10.0
    rain = gaussian_blob(shape, 50, 110)
    raw_vy, raw_vx = farneback_like_flow(rain, speed_px)

    implied_px = speed_px * (60.0 / DT_MIN)
    start = centroid_row(rain)
    observed_mass = interior_mass(rain)

    raw = advect_field(rain, raw_vy, raw_vx, horizon_minutes=60.0, dt_minutes=DT_MIN)
    raw_travel = (centroid_row(raw) - start) / implied_px
    raw_mass = interior_mass(raw) / observed_mass

    completed_vy, completed_vx = complete_flow(
        raw_vy, raw_vx, rain, pixel_km=PIXEL_KM, support_threshold_mm_h=THRESHOLD
    )
    completed = advect_field(
        rain, completed_vy, completed_vx, horizon_minutes=60.0, dt_minutes=DT_MIN
    )
    completed_travel = (centroid_row(completed) - start) / implied_px
    completed_mass = interior_mass(completed) / observed_mass

    # Failure mode, documented: without completion the rain stalls short of
    # where the flow says it should be, and most of it never arrives.
    assert raw_travel < 0.80, f"expected a stall, got {raw_travel:.2f} of implied"
    assert raw_mass < 0.50, f"expected the blob to dissolve, kept {raw_mass:.2f}"

    # Fixed behaviour.
    assert 0.80 <= completed_travel <= 1.05, f"travelled {completed_travel:.2f} of implied"
    assert completed_mass > 0.70, f"kept only {completed_mass:.2f} of the mass"
    assert completed_mass > 1.5 * raw_mass


def test_completion_removes_the_zero_motion_majority():
    """71 % of a real composite came back with |v| < 0.5 px/frame. Not any more."""
    rain = gaussian_blob((420, 220), 50, 110)
    raw_vy, raw_vx = farneback_like_flow(rain, 10.0)
    assert float(np.mean(np.hypot(raw_vy, raw_vx) < 0.5)) > 0.5

    vy, vx = complete_flow(
        raw_vy, raw_vx, rain, pixel_km=PIXEL_KM, support_threshold_mm_h=THRESHOLD
    )
    assert float(np.mean(np.hypot(vy, vx) < 0.5)) == 0.0


def test_advection_follows_the_trajectory_not_a_single_euler_step():
    """Curved trajectory, analytically known.

    With ``vy = a`` and ``vx = b·row`` (steady shear), a parcel starting at
    ``(y0, x0)`` is at ``(y0 + aT, x0 + b·y0·T + b·a·T²/2)`` after T frames.
    The old one-shot back-step evaluates the velocity once, at the
    destination, and misses the ``b·a·T²/2`` term entirely.
    """
    shape = (200, 200)
    y0, x0 = 40.0, 60.0
    a, b, frames = 3.0, 0.06, 6.0
    rain = gaussian_blob(shape, y0, x0, sigma=6.0)
    rows = np.indices(shape)[0].astype(np.float32)
    vy = np.full(shape, a, dtype=np.float32)
    vx = (b * rows).astype(np.float32)

    out = advect_field(
        rain, vy, vx, horizon_minutes=frames * DT_MIN, dt_minutes=DT_MIN
    )

    peak = np.unravel_index(int(np.nanargmax(out)), shape)
    exact_col = x0 + b * y0 * frames + b * a * frames**2 / 2.0
    euler_col = x0 + b * (y0 + a * frames) * frames  # v sampled at the destination
    assert peak[0] == pytest.approx(y0 + a * frames, abs=1.0)
    assert peak[1] == pytest.approx(exact_col, abs=1.0)
    # …and the two predictions really are far enough apart to tell apart.
    assert abs(exact_col - euler_col) > 3.0
    assert abs(peak[1] - euler_col) > 2.0


# ---------------------------------------------------------------------------
# 3. Continuity across leads
# ---------------------------------------------------------------------------


def test_advected_field_keeps_its_mass_and_reaches_further_every_lead():
    """No fold pile-up, no dissolving front: mass stays, the front advances."""
    shape = (420, 220)
    rain = gaussian_blob(shape, 50, 110)
    raw_vy, raw_vx = farneback_like_flow(rain, 10.0)
    vy, vx = complete_flow(
        raw_vy, raw_vx, rain, pixel_km=PIXEL_KM, support_threshold_mm_h=THRESHOLD
    )

    leads = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    fields = list(
        advect_field_series(rain, vy, vx, horizons_minutes=leads, dt_minutes=DT_MIN)
    )
    observed_mass = interior_mass(rain)
    observed_wet = rain >= THRESHOLD
    observed_peak = float(rain.max())

    reaches = []
    for lead, field in zip(leads, fields):
        ratio = interior_mass(field) / observed_mass
        assert 0.70 <= ratio <= 1.05, f"lead {lead:.0f}: mass ratio {ratio:.2f}"
        # A fold in the back-map piles several destinations onto one source
        # and invents intensity that was never observed.
        assert np.nanmax(field) <= observed_peak * 1.02, f"lead {lead:.0f}: pile-up"

        newly_wet = (np.nan_to_num(field, nan=0.0) >= THRESHOLD) & ~observed_wet
        assert newly_wet.any(), f"lead {lead:.0f}: front never advanced"
        reaches.append(float(np.median(np.indices(shape)[0][newly_wet])))

    assert all(b > a for a, b in zip(reaches, reaches[1:])), (
        f"newly-wet reach must grow with lead time, got {reaches}"
    )


def test_series_matches_per_lead_calls():
    """The chained-trajectory series is an optimisation, not a different answer.

    Not bit-identical: a series stops at each requested horizon, so it takes
    breakpoints a single call does not, and the extra midpoint corrections
    move a boundary pixel here and there. It has to agree to well inside
    the noise of a 0.1 mm/h radar quantum, though.
    """
    shape = (160, 120)
    rain = gaussian_blob(shape, 40, 60, sigma=8.0)
    vy, vx = farneback_like_flow(rain, 6.0, 2.0)
    vy, vx = complete_flow(vy, vx, rain, pixel_km=PIXEL_KM, support_threshold_mm_h=THRESHOLD)

    leads = [12.0, 25.0, 47.0]
    series = list(
        advect_field_series(rain, vy, vx, horizons_minutes=leads, dt_minutes=DT_MIN)
    )
    for lead, got in zip(leads, series):
        want = advect_field(rain, vy, vx, horizon_minutes=lead, dt_minutes=DT_MIN)
        both = np.isfinite(got) & np.isfinite(want)
        assert float(np.abs(got[both] - want[both]).max()) < 0.02 * float(rain.max())
        disagree = np.isnan(got) ^ np.isnan(want)
        assert disagree.mean() < 1e-3, f"lead {lead}: NaN masks differ on {disagree.sum()} px"


def test_series_handles_a_zero_and_repeated_horizon():
    """The loop renderer asks for lead 0 (frame age only) and can repeat a
    horizon when two leads land in the same frame."""
    field = np.zeros((40, 40), dtype=np.float32)
    field[10, 20] = 10.0
    vy = np.full((40, 40), 3.0, dtype=np.float32)
    vx = np.zeros((40, 40), dtype=np.float32)

    peaks = [
        np.unravel_index(int(np.nanargmax(out)), field.shape)
        for out in advect_field_series(
            field, vy, vx, horizons_minutes=[0.0, 10.0, 10.0, 20.0], dt_minutes=DT_MIN
        )
    ]
    assert peaks == [(10, 20), (13, 20), (13, 20), (16, 20)]


def test_series_rejects_unsorted_or_negative_horizons():
    field = np.zeros((8, 8), dtype=np.float32)
    zero = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        list(advect_field_series(field, zero, zero, horizons_minutes=[20.0, 10.0]))
    with pytest.raises(ValueError):
        list(advect_field_series(field, zero, zero, horizons_minutes=[-5.0]))


# ---------------------------------------------------------------------------
# 4. NaN semantics — unknown must stay unknown
# ---------------------------------------------------------------------------


def _nodata_case():
    shape = (200, 120)
    rain = gaussian_blob(shape, 30, 60, sigma=8.0)
    field = rain.copy()
    field[70:85, :] = np.nan  # a nodata band across the composite
    vy = np.full(shape, 6.0, dtype=np.float32)
    vx = np.zeros(shape, dtype=np.float32)
    return shape, rain, field, vy, vx


def test_nodata_stays_nodata_and_travels_with_the_flow():
    shape, rain, field, vy, vx = _nodata_case()
    out = advect_field(field, vy, vx, horizon_minutes=60.0, dt_minutes=DT_MIN)

    # 6 px/frame × 6 frames = 36 px: the band arrives at rows 106-120.
    assert np.isnan(out[106:121]).all()
    # …and the rows it left behind are known again (rain, or dry, but real).
    assert np.isfinite(out[95:104]).all()


def test_rain_at_the_grid_edge_produces_nan_inflow_not_zeros():
    """The composite's coverage edge is a real edge: what blows in is unknown."""
    shape, rain, field, vy, vx = _nodata_case()
    out = advect_field(field, vy, vx, horizon_minutes=60.0, dt_minutes=DT_MIN)

    # Every destination in the top 36 rows sources from above the grid.
    assert np.isnan(out[:36]).all()
    assert not np.any(out[:36] == 0.0)
    assert np.isfinite(out[36:60]).all()


def test_nothing_teleports_across_a_nodata_barrier():
    """A destination whose trajectory lands *inside* nodata gets NaN.

    Not the rain that happens to sit on the far side of the gap — that
    would invent an observation the radar never made.
    """
    shape, rain, field, vy, vx = _nodata_case()
    out = advect_field(field, vy, vx, horizon_minutes=60.0, dt_minutes=DT_MIN)

    source_rows = np.arange(shape[0]) - 36
    from_band = (source_rows >= 70) & (source_rows <= 84)
    assert np.isnan(out[from_band]).all()

    # Rain sourced from real echo still arrives, undiminished: advecting
    # *over* a gap is fine, it is advecting *out of* one that is not.
    assert float(np.nanmax(out[60:72])) == pytest.approx(float(rain.max()), rel=1e-3)


def test_all_nodata_field_stays_all_nodata():
    field = np.full((32, 32), np.nan, dtype=np.float32)
    vy = np.full((32, 32), 2.0, dtype=np.float32)
    vx = np.zeros((32, 32), dtype=np.float32)
    out = advect_field(field, vy, vx, horizon_minutes=30.0, dt_minutes=DT_MIN)
    assert np.isnan(out).all()
