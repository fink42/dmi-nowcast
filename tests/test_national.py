"""Tests for the national forecast products reduction (website Phase A §A1).

Synthetic ensembles with hand-computable exceedance fractions, frame-age
correction, NaN propagation, and — the A4 acceptance criterion — agreement
with ``aggregate_at_home`` at an arbitrary pixel.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dmi_nowcast_core.geo import CompositeGeo
from dmi_nowcast_core.national import (
    DEFAULT_LEADS_MIN,
    NationalProducts,
    _steps_in_lead,
    motion_grids_kmh,
    national_products,
)
from dmi_nowcast_core.parse import parse_composite
from dmi_nowcast_core.probabilistic import aggregate_at_home
from dmi_nowcast_core.sample import disc_pixel_indices

FIXTURE = Path(__file__).parent / "fixtures" / "composite_fullrange.h5"
COPENHAGEN = (12.5645, 55.6726)


@pytest.fixture(scope="module")
def geo() -> CompositeGeo:
    return CompositeGeo(parse_composite(FIXTURE))


def _staircase_ensemble(
    n_members: int = 8, n_timesteps: int = 12, h: int = 16, w: int = 16
) -> np.ndarray:
    """Member ``m`` rains uniformly at ``m + 1`` mm/h from timestep ``m`` onward.

    Hand-computable: the member-exceedance fraction by timestep ``t`` is
    ``min(t + 1, n_members) / n_members`` at every pixel.
    """
    ens = np.zeros((n_members, n_timesteps, h, w), dtype=np.float32)
    for m in range(n_members):
        ens[m, m:] = np.float32(m + 1.0)
    return ens


# ---------------------------------------------------------------------------
# Exact hand-computed values
# ---------------------------------------------------------------------------


def test_p_rain_exact_fractions_staircase():
    ens = _staircase_ensemble()
    result = national_products(ens, leads_min=(10, 20, 30, 45, 60),
                               timestep_min=5.0, frame_age_min=0.0)
    # lead -> steps k=ceil(lead/5) -> members 0..k-1 have crossed -> k/8
    expected = {10: 2 / 8, 20: 4 / 8, 30: 6 / 8, 45: 1.0, 60: 1.0}
    assert result.leads_min == (10, 20, 30, 45, 60)
    for lead, value in expected.items():
        grid = result.p_rain[lead]
        assert grid.shape == (16, 16)
        assert grid.dtype == np.float32
        np.testing.assert_array_equal(grid, np.float32(value))


def test_eta_exact_staircase():
    # Fraction reaches 4/8 = 0.5 first at timestep index 3 -> (3+1)*5 = 20 min.
    result = national_products(_staircase_ensemble(), timestep_min=5.0,
                               frame_age_min=0.0)
    assert result.eta_min.dtype == np.float32
    np.testing.assert_array_equal(result.eta_min, np.float32(20.0))


def test_intensity_exact_staircase():
    # Raw rates at the ETA step (index 3): members 0..3 -> 1,2,3,4 mm/h,
    # members 4..7 still dry -> 0. Median of [0,0,0,0,1,2,3,4] = 0.5.
    result = national_products(_staircase_ensemble(), timestep_min=5.0,
                               frame_age_min=0.0)
    assert result.intensity_mm_h.dtype == np.float32
    np.testing.assert_array_equal(result.intensity_mm_h, np.float32(0.5))


def test_spatial_variation_no_axis_mixups():
    """Left half rains in every member from t=0, right half stays dry."""
    ens = np.zeros((4, 6, 4, 4), dtype=np.float32)
    ens[:, :, :, :2] = 2.0
    result = national_products(ens, leads_min=(10,), timestep_min=5.0)
    np.testing.assert_array_equal(result.p_rain[10][:, :2], np.float32(1.0))
    np.testing.assert_array_equal(result.p_rain[10][:, 2:], np.float32(0.0))
    np.testing.assert_array_equal(result.eta_min[:, :2], np.float32(5.0))
    assert np.isnan(result.eta_min[:, 2:]).all()
    np.testing.assert_array_equal(result.intensity_mm_h[:, :2], np.float32(2.0))
    assert np.isnan(result.intensity_mm_h[:, 2:]).all()


def test_lead_beyond_horizon_clamps_to_last_timestep():
    ens = _staircase_ensemble()
    result = national_products(ens, leads_min=(600,), timestep_min=5.0)
    np.testing.assert_array_equal(result.p_rain[600], np.float32(1.0))


def test_metadata_fields_roundtrip():
    result = national_products(_staircase_ensemble(), leads_min=(10, 20),
                               threshold_mm_h=0.2, timestep_min=5.0,
                               frame_age_min=3.0, downsample_factor=4)
    assert isinstance(result, NationalProducts)
    assert result.leads_min == (10, 20)
    assert result.threshold_mm_h == 0.2
    assert result.timestep_min == 5.0
    assert result.frame_age_min == 3.0
    assert result.downsample_factor == 4
    assert result.n_members == 8
    assert DEFAULT_LEADS_MIN == (10, 20, 30, 45, 60)


# ---------------------------------------------------------------------------
# Frame-age correction
# ---------------------------------------------------------------------------


def test_frame_age_shifts_lead_indexing():
    """Same ensemble, frame_age 0 vs 4 min: effective lead = lead + 4, so the
    timestep index moves by ceil and every probability climbs one member."""
    ens = _staircase_ensemble()
    fresh = national_products(ens, leads_min=(10, 20, 30, 45, 60),
                              timestep_min=5.0, frame_age_min=0.0)
    aged = national_products(ens, leads_min=(10, 20, 30, 45, 60),
                             timestep_min=5.0, frame_age_min=4.0)
    # ceil((lead+4)/5): 10->3, 20->5, 30->7, 45->10, 60->13 (clamped to 12)
    expected_aged = {10: 3 / 8, 20: 5 / 8, 30: 7 / 8, 45: 1.0, 60: 1.0}
    expected_fresh = {10: 2 / 8, 20: 4 / 8, 30: 6 / 8, 45: 1.0, 60: 1.0}
    for lead in (10, 20, 30, 45, 60):
        np.testing.assert_array_equal(fresh.p_rain[lead], np.float32(expected_fresh[lead]))
        np.testing.assert_array_equal(aged.p_rain[lead], np.float32(expected_aged[lead]))


def test_frame_age_shifts_eta_minutes_from_now():
    aged = national_products(_staircase_ensemble(), timestep_min=5.0,
                             frame_age_min=4.0)
    # Crossing timestep unchanged (index 3, radar-relative 20 min); minutes
    # from now = 20 - 4.
    np.testing.assert_array_equal(aged.eta_min, np.float32(16.0))
    # Intensity is taken at the (radar-relative) ETA step -> unchanged.
    np.testing.assert_array_equal(aged.intensity_mm_h, np.float32(0.5))


def test_frame_age_eta_clamped_at_zero():
    stale = national_products(_staircase_ensemble(), timestep_min=5.0,
                              frame_age_min=25.0)
    np.testing.assert_array_equal(stale.eta_min, np.float32(0.0))


# ---------------------------------------------------------------------------
# Horizon length: what a longer ensemble buys the late leads and the ETA
# ---------------------------------------------------------------------------


def test_steps_in_lead_clamps_only_when_the_ensemble_is_too_short():
    """At the live geometry — 10-min frames, 17-min frame age — the 45- and
    60-min leads need 7 and 8 timesteps. A 6-step ensemble has neither and
    clamps both onto its last step; a 9-step one resolves them apart."""
    assert _steps_in_lead(45, 17.0, 10.0, 6) == 6
    assert _steps_in_lead(60, 17.0, 10.0, 6) == 6      # same step: the bug
    assert _steps_in_lead(45, 17.0, 10.0, 9) == 7
    assert _steps_in_lead(60, 17.0, 10.0, 9) == 8      # distinct: the fix
    # The clamp itself is unchanged — a lead past the horizon still saturates.
    assert _steps_in_lead(180, 17.0, 10.0, 9) == 9


def test_eta_can_exceed_sixty_minus_frame_age_with_a_longer_ensemble():
    """ETA is capped by the ensemble's own horizon: ``n_timesteps *
    timestep - frame_age``. A pixel whose members only cross at the 9th
    timestep is NaN in a 6-step ensemble (rain never arrives inside it) and
    73 min — 90 from the frame, 17 of which have already elapsed — in a
    9-step one."""
    def late_crossers(n_timesteps: int) -> np.ndarray:
        # Every member dry until timestep index 8, then wet.
        ens = np.zeros((8, n_timesteps, 4, 4), dtype=np.float32)
        if n_timesteps > 8:
            ens[:, 8:] = 3.0
        return ens

    short = national_products(late_crossers(6), leads_min=(60,),
                              timestep_min=10.0, frame_age_min=17.0)
    assert np.isnan(short.eta_min).all()
    assert np.isnan(short.intensity_mm_h).all()
    np.testing.assert_array_equal(short.p_rain[60], np.float32(0.0))

    long = national_products(late_crossers(9), leads_min=(60,),
                             timestep_min=10.0, frame_age_min=17.0)
    # (8 + 1) * 10 - 17 = 73 min from now.
    np.testing.assert_array_equal(long.eta_min, np.float32(73.0))
    np.testing.assert_array_equal(long.intensity_mm_h, np.float32(3.0))
    # Lead 60 needs 8 steps (ceil(77/10)); the crossing is at step 9, so the
    # ETA exists while P(<=60) is still 0 — an arrival past the served leads.
    np.testing.assert_array_equal(long.p_rain[60], np.float32(0.0))


def test_late_leads_separate_once_the_horizon_covers_the_frame_age():
    """``P(<=45)`` and ``P(<=60)`` at a 17-min frame age: identical in a
    6-step ensemble (both clamp to its last step), distinct in a 9-step one."""
    short = national_products(_staircase_ensemble(n_timesteps=6),
                              leads_min=(45, 60), timestep_min=10.0,
                              frame_age_min=17.0)
    np.testing.assert_array_equal(short.p_rain[45], np.float32(6 / 8))
    np.testing.assert_array_equal(short.p_rain[60], np.float32(6 / 8))

    long = national_products(_staircase_ensemble(n_timesteps=9),
                             leads_min=(45, 60), timestep_min=10.0,
                             frame_age_min=17.0)
    np.testing.assert_array_equal(long.p_rain[45], np.float32(7 / 8))
    np.testing.assert_array_equal(long.p_rain[60], np.float32(8 / 8))


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_all_nan_pixel_is_nan_in_every_product():
    ens = _staircase_ensemble()
    ens[:, :, 0, 0] = np.nan
    result = national_products(ens, leads_min=(10, 60), timestep_min=5.0)
    for lead in (10, 60):
        assert np.isnan(result.p_rain[lead][0, 0])
        assert np.isfinite(result.p_rain[lead][1:, :]).all()
        assert np.isfinite(result.p_rain[lead][0, 1:]).all()
    assert np.isnan(result.eta_min[0, 0])
    assert np.isnan(result.intensity_mm_h[0, 0])
    assert np.isfinite(result.eta_min[1:, :]).all()


def test_partial_member_nan_counts_as_not_exceeding():
    """A member with NaN at a pixel is a non-exceeder there, never a false
    positive; the intensity median ignores its NaN."""
    ens = np.full((4, 6, 2, 2), 3.0, dtype=np.float32)
    ens[0, :, 0, 0] = np.nan
    result = national_products(ens, leads_min=(10,), timestep_min=5.0)
    assert result.p_rain[10][0, 0] == np.float32(3 / 4)
    np.testing.assert_array_equal(result.p_rain[10][0, 1], np.float32(1.0))
    # 3/4 >= 0.5 at timestep 0 -> ETA 5 min everywhere.
    np.testing.assert_array_equal(result.eta_min, np.float32(5.0))
    # nanmedian([NaN, 3, 3, 3]) = 3.
    assert result.intensity_mm_h[0, 0] == np.float32(3.0)


def test_nan_plus_dry_never_creates_false_positive():
    ens = np.zeros((4, 6, 3, 3), dtype=np.float32)
    ens[0] = np.nan  # one member entirely nodata
    result = national_products(ens, leads_min=(10,), timestep_min=5.0)
    np.testing.assert_array_equal(result.p_rain[10], np.float32(0.0))
    assert np.isnan(result.eta_min).all()
    assert np.isnan(result.intensity_mm_h).all()


def test_fully_nan_ensemble_is_all_nan():
    ens = np.full((3, 4, 2, 2), np.nan, dtype=np.float32)
    result = national_products(ens, leads_min=(10,), timestep_min=5.0)
    assert np.isnan(result.p_rain[10]).all()
    assert np.isnan(result.eta_min).all()
    assert np.isnan(result.intensity_mm_h).all()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rejects_bad_inputs():
    ens = _staircase_ensemble(2, 3, 4, 4)
    with pytest.raises(ValueError, match="n_members"):
        national_products(np.zeros((3, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="whole minutes"):
        national_products(ens, leads_min=(12.5,))
    with pytest.raises(ValueError, match="frame_age_min"):
        national_products(ens, frame_age_min=-1.0)
    with pytest.raises(ValueError, match="timestep_min"):
        national_products(ens, timestep_min=0.0)
    with pytest.raises(ValueError, match="downsample_factor"):
        national_products(ens, downsample_factor=0)
    with pytest.raises(ValueError, match="must not be empty"):
        national_products(ens, leads_min=())


# ---------------------------------------------------------------------------
# THE AGREEMENT TEST (A4 criterion): national grids sampled at the home pixel
# must reproduce aggregate_at_home.
# ---------------------------------------------------------------------------


def _home_block_ensemble(geo: CompositeGeo, downsample_factor: int = 4,
                         n_members: int = 8, n_timesteps: int = 12) -> np.ndarray:
    """Staircase ensemble on the downsampled national grid, uniform over a
    generous block around the home pixel so ``aggregate_at_home``'s
    max-in-disc equals the single-pixel value the national grids hold."""
    h_native, w_native = geo.composite.reflectivity_dbz.shape
    h, w = h_native // downsample_factor, w_native // downsample_factor
    idx = geo.lonlat_to_grid(*COPENHAGEN)
    r0 = int(round(idx.row / downsample_factor))
    c0 = int(round(idx.col / downsample_factor))
    ens = np.zeros((n_members, n_timesteps, h, w), dtype=np.float32)
    for m in range(n_members):
        ens[m, m:, r0 - 8 : r0 + 9, c0 - 8 : c0 + 9] = np.float32(m + 1.0)
    return ens


def _home_pixel(geo: CompositeGeo, shape: tuple[int, int],
                downsample_factor: int, radius_m: float = 1000.0) -> tuple[int, int]:
    """The downsampled-grid pixel(s) aggregate_at_home's disc actually samples."""
    pixel_scale_m = (geo.composite.xscale_m + geo.composite.yscale_m) / 2.0
    radius_px = radius_m / (pixel_scale_m * downsample_factor)
    idx = geo.lonlat_to_grid(*COPENHAGEN)
    rows, cols = disc_pixel_indices(
        shape, idx.row / downsample_factor, idx.col / downsample_factor, radius_px
    )
    assert rows.size >= 1
    # At x4 the 1 km disc is effectively a single pixel; all disc pixels must
    # sit inside the uniform block for the equivalence to be exact.
    r0 = int(round(idx.row / downsample_factor))
    c0 = int(round(idx.col / downsample_factor))
    assert (np.abs(rows - r0) <= 8).all() and (np.abs(cols - c0) <= 8).all()
    return int(rows[0]), int(cols[0])


def test_agreement_with_aggregate_at_home(geo):
    ds = 4
    leads = (10, 20, 30, 45, 60)
    ens = _home_block_ensemble(geo, downsample_factor=ds)
    r, c = _home_pixel(geo, ens.shape[2:], ds)

    national = national_products(ens, leads_min=leads, timestep_min=5.0,
                                 frame_age_min=0.0, downsample_factor=ds)
    home = aggregate_at_home(ens, geo, *COPENHAGEN,
                             leads_min=tuple(float(L) for L in leads),
                             timestep_min=5.0, downsample_factor=ds)

    # (a) probabilities: exact equality, same reduction on both sides.
    for i, lead in enumerate(leads):
        assert float(national.p_rain[lead][r, c]) == home.probability_by_lead[i], (
            f"lead {lead}: national {national.p_rain[lead][r, c]} != "
            f"home {home.probability_by_lead[i]}"
        )
    # Hand check the shared values: staircase -> k/8 capped at 1.
    assert home.probability_by_lead == (0.25, 0.5, 0.75, 1.0, 1.0)

    # (b) ETA: within one timestep. National = first step with fraction >= 0.5
    # (20 min); aggregate's P50 interpolates between member crossings (22.5).
    assert abs(float(national.eta_min[r, c]) - home.eta_p50_min) <= 5.0

    # Away from the block the national grid is dry -- the agreement is not an
    # everything-is-constant artefact.
    assert float(national.p_rain[60][r - 100, c - 100]) == 0.0
    assert np.isnan(national.eta_min[r - 100, c - 100])


def test_agreement_with_aggregate_at_home_frame_aged(geo):
    """Both sides must speak the frame-age-corrected convention: national
    applies frame_age_min internally; aggregate_at_home (which has no frame-age
    parameter) is fed pre-corrected leads. Plan §A0's named bug magnet."""
    ds = 4
    frame_age = 5.0
    leads = (10, 20, 30, 45, 60)
    corrected = tuple(float(L) + frame_age for L in leads)
    ens = _home_block_ensemble(geo, downsample_factor=ds)
    r, c = _home_pixel(geo, ens.shape[2:], ds)

    national = national_products(ens, leads_min=leads, timestep_min=5.0,
                                 frame_age_min=frame_age, downsample_factor=ds)
    home = aggregate_at_home(ens, geo, *COPENHAGEN, leads_min=corrected,
                             timestep_min=5.0, downsample_factor=ds)

    for i, lead in enumerate(leads):
        assert float(national.p_rain[lead][r, c]) == home.probability_by_lead[i]
    assert home.probability_by_lead == (0.375, 0.625, 0.875, 1.0, 1.0)

    # National ETA is minutes-from-now (frame age subtracted); aggregate's is
    # radar-relative. Same frame, then within one timestep.
    assert abs(
        (float(national.eta_min[r, c]) + frame_age) - home.eta_p50_min
    ) <= 5.0


# ---------------------------------------------------------------------------
# R2 — cell-motion grids (km/h, east/north positive, nodata off the echo)
# ---------------------------------------------------------------------------

def _uniform_flow(shape, *, vy_px, vx_px):
    return (np.full(shape, vy_px, np.float32),
            np.full(shape, vx_px, np.float32))


def test_motion_grids_units_pinned_end_to_end():
    """Hand-computed: +8 px/frame east and 4 px/frame NORTH (vy = -4, rows
    grow southward) on a 500 m grid at a 10-min frame interval.

    east  = 8 px × 0.5 km = 4 km per 10 min = 24 km/h
    north = 4 px × 0.5 km = 2 km per 10 min = 12 km/h

    The ×4 downsample must not touch the answer: ``v[::4, ::4] / 4`` on
    2 km pixels is the same physical speed as ``v`` on 500 m pixels — the
    ``/f`` and the ``f×`` in the pixel size cancel. Getting only one of the
    two right is a 16× error, which this test exists to catch.
    """
    shape = (32, 32)
    vy, vx = _uniform_flow(shape, vy_px=-4.0, vx_px=8.0)
    rain = np.full(shape, 2.0, np.float32)  # echo everywhere → all valid

    east, north = motion_grids_kmh(
        vy, vx, rain,
        pixel_km=0.5, timestep_min=10.0, downsample_factor=4,
        support_threshold_mm_h=0.5,
    )
    assert east.shape == north.shape == (8, 8)
    assert np.allclose(east, 24.0, atol=1e-4)
    assert np.allclose(north, 12.0, atol=1e-4)

    # Same physical field, no downsample → identical km/h.
    east1, north1 = motion_grids_kmh(
        vy, vx, rain,
        pixel_km=0.5, timestep_min=10.0, downsample_factor=1,
        support_threshold_mm_h=0.5,
    )
    assert np.allclose(east1, 24.0, atol=1e-4)
    assert np.allclose(north1, 12.0, atol=1e-4)


def test_motion_grid_signs_follow_dense_flow_convention():
    """dense_flow: +vy = southward, +vx = eastward. So published north is
    -vy and published east is +vx — a sign slip here points every arrow on
    the website the wrong way."""
    shape = (16, 16)
    rain = np.full(shape, 2.0, np.float32)
    # Moving south-west: vy positive (down/south), vx negative (west).
    vy, vx = _uniform_flow(shape, vy_px=6.0, vx_px=-6.0)
    east, north = motion_grids_kmh(
        vy, vx, rain, pixel_km=0.5, timestep_min=10.0, downsample_factor=1,
    )
    assert (east < 0).all(), "westward motion must publish negative east"
    assert (north < 0).all(), "southward motion must publish negative north"


def test_motion_grids_nodata_only_outside_radar_coverage():
    """Issue #6: distance from the echo no longer gates the arrow. Every
    pixel inside the composite carries a vector once *any* echo exists;
    only off-composite pixels are nodata."""
    shape = (200, 200)
    vy, vx = _uniform_flow(shape, vy_px=-4.0, vx_px=8.0)
    rain = np.zeros(shape, np.float32)
    rain[100, 100] = 5.0            # single echo pixel
    rain[:, :10] = np.nan           # off-composite strip

    east, north = motion_grids_kmh(
        vy, vx, rain,
        pixel_km=0.5, timestep_min=10.0, downsample_factor=1,
        support_threshold_mm_h=0.5,
    )
    assert np.isnan(east).any() and np.isfinite(east).any()
    assert np.array_equal(np.isnan(east), np.isnan(north))

    assert np.isfinite(east[100, 100])
    assert np.isfinite(east[100, 139])     # 19.5 km away — was inside the
    assert np.isfinite(east[100, 141])     # old 20 km radius, this one was not
    assert np.isfinite(east[199, 199]), "the far corner still gets an arrow"
    assert np.isnan(east[100, 5]), "off-composite pixels never carry motion"
    # The off-composite strip stays nodata even where it is near the echo.
    assert np.isnan(east[:, :10]).all()
    # The flow is uniform, so the fill reproduces it everywhere.
    finite = np.isfinite(east)
    assert np.allclose(east[finite], 24.0, atol=1e-3)
    assert np.allclose(north[finite], 12.0, atol=1e-3)


def test_motion_grids_all_nodata_without_echo():
    shape = (32, 32)
    vy, vx = _uniform_flow(shape, vy_px=1.0, vx_px=1.0)
    east, north = motion_grids_kmh(
        np.asarray(vy), np.asarray(vx), np.zeros(shape, np.float32),
        pixel_km=0.5, timestep_min=10.0, downsample_factor=4,
    )
    assert np.isnan(east).all() and np.isnan(north).all()


def test_motion_grids_reject_bad_inputs():
    shape = (16, 16)
    vy, vx = _uniform_flow(shape, vy_px=1.0, vx_px=1.0)
    rain = np.ones(shape, np.float32)
    with pytest.raises(ValueError, match="share one shape"):
        motion_grids_kmh(vy, vx[:8], rain, pixel_km=0.5, timestep_min=10.0)
    with pytest.raises(ValueError, match="pixel_km"):
        motion_grids_kmh(vy, vx, rain, pixel_km=0.0, timestep_min=10.0)
    with pytest.raises(ValueError, match="timestep_min"):
        motion_grids_kmh(vy, vx, rain, pixel_km=0.5, timestep_min=0.0)
    with pytest.raises(ValueError, match="downsample_factor"):
        motion_grids_kmh(vy, vx, rain, pixel_km=0.5, timestep_min=10.0,
                         downsample_factor=0)
