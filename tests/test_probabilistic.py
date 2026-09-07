"""Tests for the probabilistic ensemble forecast wrapper.

The pysteps STEPS call is expensive (~30 s on a real-size grid), so we test
the dB transforms and the aggregation logic with mocked ensembles. The
end-to-end STEPS run is exercised in the Phase 4 backtest script.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dmi_nowcast_core.geo import CompositeGeo
from dmi_nowcast_core.parse import parse_composite
from dmi_nowcast_core.probabilistic import (
    DB_THRESHOLD_MM_H,
    ZERO_DB,
    aggregate_at_home,
    db_to_rain,
    rain_to_db,
)

FIXTURE = Path(__file__).parent / "fixtures" / "composite_fullrange.h5"
COPENHAGEN = (12.5645, 55.6726)


@pytest.fixture(scope="module")
def geo() -> CompositeGeo:
    return CompositeGeo(parse_composite(FIXTURE))


def test_rain_to_db_basic_values():
    # R=1 → 0 dB; R=10 → 10 dB; R=0.1 → -10 dB (exactly threshold → kept)
    out = rain_to_db(np.array([1.0, 10.0, 0.1, 0.05, 0.0], dtype=np.float32))
    assert out[0] == pytest.approx(0.0, abs=0.01)
    assert out[1] == pytest.approx(10.0, abs=0.01)
    assert out[2] == pytest.approx(-10.0, abs=0.01)
    # below threshold → ZERO_DB
    assert out[3] == ZERO_DB
    assert out[4] == ZERO_DB


def test_db_to_rain_roundtrip_above_threshold():
    rain_in = np.array([0.5, 2.0, 30.0], dtype=np.float32)
    out = db_to_rain(rain_to_db(rain_in))
    np.testing.assert_allclose(out, rain_in, rtol=1e-3)


def test_db_to_rain_collapses_zero_db_to_zero():
    out = db_to_rain(np.array([ZERO_DB, ZERO_DB + 0.0001], dtype=np.float32))
    assert out[0] == 0.0
    assert out[1] == 0.0


def test_aggregate_perfect_dry_ensemble_gives_zero_probability_and_nan_eta(geo):
    h, w = geo.composite.reflectivity_dbz.shape
    dry = np.zeros((5, 12, h, w), dtype=np.float32)
    result = aggregate_at_home(dry, geo, *COPENHAGEN, leads_min=(10, 30, 60))
    assert result.probability_by_lead == (0.0, 0.0, 0.0)
    assert np.isnan(result.eta_p50_min)
    assert result.n_members == 5


def test_aggregate_all_members_wet_gives_probability_one(geo):
    """If every member predicts rain at home from t=0 onwards, P=1 at every lead."""
    h, w = geo.composite.reflectivity_dbz.shape
    wet = np.full((10, 12, h, w), 5.0, dtype=np.float32)
    result = aggregate_at_home(wet, geo, *COPENHAGEN, leads_min=(10, 30, 60), timestep_min=5.0)
    assert result.probability_by_lead == (1.0, 1.0, 1.0)
    assert result.eta_p50_min == pytest.approx(5.0)  # first timestep = 5 min


def test_aggregate_half_wet_half_dry_gives_half_probability(geo):
    """Half the ensemble has rain from t=0, half is dry → P = 0.5."""
    h, w = geo.composite.reflectivity_dbz.shape
    forecast = np.zeros((10, 12, h, w), dtype=np.float32)
    forecast[5:] = 5.0  # last 5 members are wet
    result = aggregate_at_home(forecast, geo, *COPENHAGEN, leads_min=(30,))
    assert result.probability_by_lead == (0.5,)


def test_aggregate_eta_window_widens_with_member_disagreement(geo):
    """If members hit threshold at different timesteps, the ETA P25-P75 should span them."""
    h, w = geo.composite.reflectivity_dbz.shape
    forecast = np.zeros((4, 12, h, w), dtype=np.float32)
    # Members hit threshold at timesteps 0, 1, 2, 3 → 5, 10, 15, 20 min.
    for m in range(4):
        forecast[m, m:, :, :] = 5.0
    result = aggregate_at_home(forecast, geo, *COPENHAGEN, leads_min=(60,), timestep_min=5.0)
    assert result.eta_p25_min < result.eta_p50_min < result.eta_p75_min


def test_aggregate_lead_truncates_correctly(geo):
    """A member wet only at timestep 11 (= 60 min) should be counted in P(60min) but not P(10min)."""
    h, w = geo.composite.reflectivity_dbz.shape
    forecast = np.zeros((10, 12, h, w), dtype=np.float32)
    forecast[0, 11, :, :] = 5.0
    result = aggregate_at_home(forecast, geo, *COPENHAGEN, leads_min=(10, 60), timestep_min=5.0)
    assert result.probability_by_lead[0] == 0.0
    assert result.probability_by_lead[1] == 0.1  # 1 out of 10


def test_run_ensemble_with_vendored_pysteps_end_to_end():
    """Smoke: the vendored ``_vendor.pysteps_steps`` imports and STEPS
    actually runs end-to-end on textured synthetic data. Catches import
    rewrites and __init__.py wiring bugs that the dB-transform unit
    tests above can't reach.

    Uses a 256×256 textured field and a small ensemble (5 members /
    6 timesteps / 6 cascade levels) so the test runs in ~2 s — fast
    enough for CI.
    """
    from dmi_nowcast_core.probabilistic import run_ensemble

    # Build three frames in dBZ with a moving textured blob.
    rng = np.random.default_rng(42)
    H, W = 256, 256
    base = np.zeros((H, W), dtype=np.float32)
    yy, xx = np.indices((H, W))
    for _ in range(15):
        cy = rng.uniform(20, H - 20)
        cx = rng.uniform(20, W - 20)
        sy = rng.uniform(8, 25)
        sx = rng.uniform(8, 25)
        amp = rng.uniform(0.5, 8.0)
        base += (amp * np.exp(-((yy - cy) ** 2 / (2 * sy ** 2)
                                + (xx - cx) ** 2 / (2 * sx ** 2)))).astype(np.float32)
    # Convert rain mm/h → dBZ (Marshall-Palmer): Z = 200·R^1.6 →
    # dBZ ≈ 23 + 16·log10(R). Sub-threshold pixels saturate to a very
    # negative dBZ so dbz_to_rain_rate sees them as effectively zero.
    def _to_dbz(r):
        with np.errstate(divide='ignore', invalid='ignore'):
            d = 10.0 * np.log10(200.0) + 16.0 * np.log10(np.maximum(r, 1e-6))
        return d.astype(np.float32)
    dbz_frames = [_to_dbz(np.roll(base, (s, 0), (0, 1))) for s in (4, 2, 0)]
    vy = np.full((H, W), -2.0, dtype=np.float32)
    vx = np.zeros((H, W), dtype=np.float32)

    out = run_ensemble(
        dbz_frames, vy, vx,
        n_ens_members=5, n_timesteps=6, n_cascade_levels=6,
        downsample_factor=1, pixel_scale_m=1000.0,
        seed=42,
    )
    assert out.shape == (5, 6, H, W)
    # Most outputs should be finite; some NaN edge effects are expected
    finite_frac = float(np.isfinite(out).mean())
    assert finite_frac > 0.85, f"expected mostly finite, got {finite_frac*100:.1f}%"
    # At least some predicted rain — synthetic blob is well above threshold
    finite_out = out[np.isfinite(out)]
    assert (finite_out > 0.1).any(), "expected some predicted rain"


def test_run_ensemble_downsample_halves_runtime():
    """Verify the downsample_factor argument actually slices the input.
    Output shape should be input_shape // downsample_factor."""
    from dmi_nowcast_core.probabilistic import run_ensemble

    def _to_dbz(r):
        with np.errstate(divide='ignore', invalid='ignore'):
            d = 10.0 * np.log10(200.0) + 16.0 * np.log10(np.maximum(r, 1e-6))
        return d.astype(np.float32)

    rng = np.random.default_rng(7)
    H, W = 256, 256
    base = np.zeros((H, W), dtype=np.float32)
    yy, xx = np.indices((H, W))
    for _ in range(15):
        cy, cx = rng.uniform(20, H - 20), rng.uniform(20, W - 20)
        sy, sx = rng.uniform(8, 25), rng.uniform(8, 25)
        base += (rng.uniform(0.5, 8.0) * np.exp(
            -((yy - cy) ** 2 / (2 * sy ** 2) + (xx - cx) ** 2 / (2 * sx ** 2))
        )).astype(np.float32)
    dbz_frames = [_to_dbz(np.roll(base, (i, 0), (0, 1))) for i in (4, 2, 0)]
    vy = np.full((H, W), -2.0, dtype=np.float32)
    vx = np.zeros((H, W), dtype=np.float32)

    out2 = run_ensemble(
        dbz_frames, vy, vx,
        n_ens_members=3, n_timesteps=4, n_cascade_levels=6,
        downsample_factor=2, pixel_scale_m=1000.0,
        seed=42,
    )
    assert out2.shape == (3, 4, 128, 128), f"got {out2.shape}"


def test_aggregate_at_home_subpixel_disc_falls_back_to_nearest_pixel(geo):
    """A sub-pixel disc whose fractional centre sits near a pixel corner
    must sample the nearest pixel (the /forecast convention), not raise
    'outside the grid'. Regression: the public instance's central-DK
    reference point (56.0, 10.0) hit exactly this on the x4 grid."""
    import numpy as np
    from dmi_nowcast_core.probabilistic import aggregate_at_home

    h = geo.composite.reflectivity_dbz.shape[0] // 4
    w = geo.composite.reflectivity_dbz.shape[1] // 4
    forecast = np.zeros((4, 6, h, w), dtype=np.float32)
    forecast[:2, :, :, :] = 5.0  # half the members wet everywhere

    # Pick a lon/lat whose downsampled fractional position lands near a
    # pixel corner: search a few offsets until the naive disc is empty.
    from dmi_nowcast_core.sample import disc_pixel_indices
    target = None
    for drow in np.linspace(0.45, 0.55, 11):
        for dcol in np.linspace(0.45, 0.55, 11):
            rr, cc = disc_pixel_indices((h, w), h // 2 + drow, w // 2 + dcol, 0.5)
            if rr.size == 0:
                target = (h // 2 + drow, w // 2 + dcol)
                break
        if target:
            break
    if target is None:  # implementation includes corners — nothing to test
        return
    lon, lat = geo.grid_to_lonlat(target[0] * 4.0, target[1] * 4.0)
    agg = aggregate_at_home(
        forecast, geo, lon, lat, radius_m=1000.0,
        threshold_mm_h=0.5, timestep_min=10.0,
        leads_min=(10.0, 60.0), downsample_factor=4,
    )
    assert agg.probability_by_lead == (0.5, 0.5)


# ---------------------------------------------------------------------------
# Memory-shaped rewrites in run_ensemble's input/output path (2026-09-07).
# Both are supposed to be EXACTLY equal to the expression form they replaced;
# these tests are the proof, since a corpus is only comparable across
# rebuilds if the ensemble is unchanged.
# ---------------------------------------------------------------------------


def _db_to_rain_reference(db: np.ndarray) -> np.ndarray:
    """The expression form ``db_to_rain`` replaced (four whole-array
    temporaries; ~440 MB of them on the calibration-corpus ensemble)."""
    arr = np.asarray(db, dtype=np.float32)
    out = np.power(np.float32(10.0), arr / np.float32(10.0))
    return np.where(arr > ZERO_DB + 1e-3, out, np.float32(0.0))


@pytest.mark.parametrize("shape", [(), (7,), (3, 4), (2, 3, 4), (4, 2, 5, 6)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_db_to_rain_matches_the_expression_form_elementwise(shape, dtype):
    rng = np.random.default_rng(20260907)
    arr = (rng.random(shape) * 70.0 - 25.0).astype(dtype)
    flat = arr.reshape(-1)
    if flat.size >= 4:  # the values that decide the branch
        flat[0], flat[1], flat[2], flat[3] = np.nan, ZERO_DB, np.inf, -np.inf
    got, expected = db_to_rain(arr), _db_to_rain_reference(arr)
    assert got.dtype == expected.dtype == np.float32
    assert got.shape == expected.shape
    # NaN input goes to 0.0 in BOTH (``NaN > threshold`` is False) — the
    # trap the chunked form has to avoid, since ``NaN <= threshold`` is
    # False as well.
    assert np.array_equal(got, expected, equal_nan=True)


def test_db_to_rain_does_not_mutate_its_input():
    arr = np.array([[-20.0, 5.0], [ZERO_DB, 12.0]], dtype=np.float32)
    before = arr.copy()
    db_to_rain(arr)
    assert np.array_equal(arr, before)


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_zr_and_db_commute_with_stride_slicing(factor):
    """``run_ensemble`` now strides the dBZ inputs BEFORE the Z-R and dB
    conversions instead of after, to keep the two intermediate stacks at
    1/f² of the native grid. Both conversions are elementwise, so the two
    orders agree exactly — this is that identity, on a field that includes
    the sub-threshold values where ``rain_to_db`` branches."""
    from dmi_nowcast_core.transform import dbz_to_rain_rate

    rng = np.random.default_rng(11)
    dbz = (rng.random((32, 40)) * 90.0 - 32.0).astype(np.float32)
    convert_then_slice = rain_to_db(dbz_to_rain_rate(dbz))[::factor, ::factor]
    slice_then_convert = rain_to_db(dbz_to_rain_rate(dbz[::factor, ::factor]))
    assert np.array_equal(convert_then_slice, slice_then_convert, equal_nan=True)


def test_run_ensemble_velocity_downsampling_is_unchanged():
    """The velocity is stride-sliced then divided by f, exactly as before
    the input-path rewrite — a factor-of-f error here would rescale every
    advected probability without changing any shape."""
    from dmi_nowcast_core.probabilistic import run_ensemble

    captured = {}

    def _to_dbz(r):
        with np.errstate(divide="ignore", invalid="ignore"):
            return (10.0 * np.log10(200.0)
                    + 16.0 * np.log10(np.maximum(r, 1e-6))).astype(np.float32)

    rng = np.random.default_rng(3)
    H, W = 64, 64
    base = rng.random((H, W)).astype(np.float32) * 5.0
    frames = [_to_dbz(np.roll(base, s, 0)) for s in (4, 2, 0)]
    vy = np.full((H, W), -3.0, dtype=np.float32)
    vx = np.full((H, W), 6.0, dtype=np.float32)

    from dmi_nowcast_core._vendor.pysteps_steps.nowcasts import steps as ps_steps

    real = ps_steps.forecast

    def spy(**kwargs):
        captured["velocity"] = np.array(kwargs["velocity"])
        captured["kmperpixel"] = kwargs["kmperpixel"]
        captured["precip_shape"] = kwargs["precip"].shape
        return real(**kwargs)

    ps_steps.forecast = spy
    try:
        run_ensemble(
            frames, vy, vx, n_ens_members=2, n_timesteps=2, n_cascade_levels=4,
            downsample_factor=4, pixel_scale_m=500.0, seed=1,
        )
    finally:
        ps_steps.forecast = real

    assert captured["precip_shape"] == (3, 16, 16)
    assert captured["kmperpixel"] == pytest.approx(2.0)  # 500 m x 4
    # pysteps order is [vx, vy], both divided by the downsample factor.
    assert np.allclose(captured["velocity"][0], 6.0 / 4)
    assert np.allclose(captured["velocity"][1], -3.0 / 4)
