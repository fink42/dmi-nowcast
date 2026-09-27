"""Review R4b: the ``cv2`` advection backend against the ``scipy`` one.

``cv2`` runs the vendored midpoint scheme through ``cv2.remap`` in float32.
It is not bit-identical (1/32 px position quantisation, float32
trajectories), so the contract is: the same geometry on exact cases, the
same NaN semantics, and agreement well inside a radar quantum on real and
sheared fields. ``scipy`` stays the default and is untouched.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dmi_nowcast_core import advect
from dmi_nowcast_core.advect import advect_field, advect_field_series

BACKENDS = ("scipy", "cv2")
CORPUS = Path.home() / "dmi-nowcast-corpus-local" / "composites"


def _shear(shape=(128, 144)):
    ys, xs = np.indices(shape, dtype=np.float32)
    field = np.zeros(shape, dtype=np.float32)
    rng = np.random.default_rng(4)
    for _ in range(12):
        cy, cx = rng.uniform(0, shape[0]), rng.uniform(0, shape[1])
        s = rng.uniform(3, 10)
        field += rng.uniform(0.5, 12) * np.exp(
            -((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * s * s))
    field[:6, :20] = np.nan  # a nodata patch inside the grid
    vy = (1.5 + 0.02 * xs + 0.5 * np.sin(xs / 17.0)).astype(np.float32)
    vx = (2.0 + 0.03 * ys - 0.4 * np.cos(ys / 11.0)).astype(np.float32)
    return field.astype(np.float32), vy, vx


@pytest.mark.parametrize("backend", BACKENDS)
def test_uniform_integer_motion_is_an_exact_shift(backend):
    field = np.zeros((32, 32), dtype=np.float32)
    field[8, 16] = 10.0
    vy = np.full((32, 32), 5.0, dtype=np.float32)
    vx = np.full((32, 32), -2.0, dtype=np.float32)
    out = advect_field(field, vy, vx, horizon_minutes=20.0, dt_minutes=10.0,
                       backend=backend)
    assert out[18, 12] == 10.0
    assert np.nansum(out) == 10.0
    # Rows 0-9 and the last 4 columns traced off the grid: unknown.
    assert np.isnan(out[:10]).all()
    assert np.isnan(out[:, -4:]).all()
    assert np.isfinite(out[10:, :-4]).all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_half_pixel_motion_is_bilinear(backend):
    field = np.zeros((32, 32), dtype=np.float32)
    field[8, 16] = 10.0
    vy = np.full((32, 32), 0.5, dtype=np.float32)
    vx = np.zeros((32, 32), dtype=np.float32)
    out = advect_field(field, vy, vx, horizon_minutes=10.0, dt_minutes=10.0,
                       backend=backend)
    assert out[8, 16] == pytest.approx(5.0, abs=1e-5)
    assert out[9, 16] == pytest.approx(5.0, abs=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_everything_off_grid_is_nan(backend):
    field = np.ones((16, 16), dtype=np.float32)
    vy = np.full((16, 16), 100.0, dtype=np.float32)
    vx = np.zeros((16, 16), dtype=np.float32)
    out = advect_field(field, vy, vx, horizon_minutes=10.0, backend=backend)
    assert np.isnan(out).all()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_motion_is_identity_and_keeps_dtype(backend, dtype):
    rng = np.random.default_rng(7)
    field = rng.standard_normal((24, 24)).astype(dtype)
    zero = np.zeros((24, 24), dtype=np.float32)
    out = advect_field(field, zero, zero, horizon_minutes=20.0, backend=backend)
    assert out.dtype == dtype
    np.testing.assert_allclose(out, field, atol=1e-6)


def test_sheared_series_agrees_with_scipy():
    field, vy, vx = _shear()
    horizons = [14.3, 19.3, 24.3, 44.3, 74.3]
    ref = list(advect_field_series(field, vy, vx, horizons_minutes=horizons,
                                   backend="scipy"))
    new = list(advect_field_series(field, vy, vx, horizons_minutes=horizons,
                                   backend="cv2"))
    assert len(ref) == len(new) == len(horizons)
    for r, n in zip(ref, new):
        assert r.dtype == n.dtype == np.float32
        nan_agree = (np.isnan(r) == np.isnan(n)).mean()
        assert nan_agree > 0.995
        both = np.isfinite(r) & np.isfinite(n)
        diff = np.abs(r - n)[both]
        assert np.percentile(diff, 99) < 0.1
        assert ((r >= 0.5) == (n >= 0.5))[both].mean() > 0.995


def test_series_first_field_equals_a_direct_call_under_cv2():
    field, vy, vx = _shear()
    first = next(iter(advect_field_series(
        field, vy, vx, horizons_minutes=[14.3, 24.3], backend="cv2")))
    direct = advect_field(field, vy, vx, horizon_minutes=14.3, backend="cv2")
    np.testing.assert_array_equal(first, direct)


def test_default_backend_is_scipy_and_unchanged():
    field, vy, vx = _shear()
    assert advect.DEFAULT_BACKEND == "scipy"
    a = advect_field(field, vy, vx, horizon_minutes=30.0)
    b = advect_field(field, vy, vx, horizon_minutes=30.0, backend="scipy")
    np.testing.assert_array_equal(a, b)


def test_unknown_backend_is_refused_eagerly():
    field, vy, vx = _shear()
    with pytest.raises(ValueError, match="backend"):
        advect_field(field, vy, vx, horizon_minutes=10.0, backend="numba")
    with pytest.raises(ValueError, match="backend"):
        advect_field_series(field, vy, vx, horizons_minutes=[10.0],
                            backend="numba")


@pytest.mark.skipif(not CORPUS.exists(), reason="local radar corpus not present")
def test_real_cycle_agrees_within_a_radar_quantum():
    from dmi_nowcast_core.dense_flow import estimate_motion
    from dmi_nowcast_core.parse import parse_composite
    from dmi_nowcast_core.transform import dbz_to_rain_rate

    folder = CORPUS / "2026" / "02"
    a = folder / "dk.com.202602280740.500_max.h5"
    b = folder / "dk.com.202602280750.500_max.h5"
    if not (a.exists() and b.exists()):
        pytest.skip("reference frames not in the local corpus")
    ca, cb = parse_composite(a), parse_composite(b)
    rain = dbz_to_rain_rate(cb.reflectivity_dbz, zr_a=cb.zr_a, zr_b=cb.zr_b)
    m = estimate_motion(ca.reflectivity_dbz, cb.reflectivity_dbz, rain,
                        pixel_km=0.5, dt_min=10.0)
    horizons = [15.0, 25.0, 45.0, 75.0]
    ref = list(advect_field_series(rain, m.vy, m.vx, horizons_minutes=horizons,
                                   backend="scipy"))
    new = list(advect_field_series(rain, m.vy, m.vx, horizons_minutes=horizons,
                                   backend="cv2"))
    for r, n in zip(ref, new):
        assert (np.isnan(r) == np.isnan(n)).mean() > 0.9999
        both = np.isfinite(r) & np.isfinite(n)
        diff = np.abs(r - n)[both]
        assert np.percentile(diff, 99.9) < 0.05
        assert ((r >= 0.5) == (n >= 0.5))[both].mean() > 0.9995
