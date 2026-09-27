"""Review R4b: the STEPS ensemble's coverage mask and per-cycle seed.

1. ``rain_to_db`` turns NaN (outside radar coverage) into ZERO_DB, so the
   ensemble used to carry a finite 0 mm/h there and ``national_products``
   read every off-coverage pixel as data. ``run_ensemble`` now returns NaN
   outside the newest frame's coverage — and leaves every value inside it
   bit-identical.
2. The seed: consecutive cycles used the same default seed (identical
   member noise); :func:`ensemble_seed` gives one per radar frame, and the
   vendored STEPS no longer reseeds the process-global RNG.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from dmi_nowcast_core.national import national_products
from dmi_nowcast_core.probabilistic import db_to_rain, ensemble_seed, run_ensemble


def _to_dbz(r: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 10.0 * np.log10(200.0) + 16.0 * np.log10(np.maximum(r, 1e-6))
    return d.astype(np.float32)


def _frames(h: int = 96, w: int = 96, seed: int = 5):
    """Three dBZ frames of a drifting textured field.

    Rows 0-15 and the right-hand 12 columns are nodata (NaN) in every
    frame — outside coverage. Where the rain field is below 0.05 mm/h the
    pixel is ``-inf`` (DMI's undetect) — dry but INSIDE coverage.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((h, w))
    base = np.zeros((h, w), dtype=np.float32)
    for _ in range(10):
        cy, cx = rng.uniform(10, h - 10), rng.uniform(10, w - 10)
        s = rng.uniform(4, 12)
        base += (rng.uniform(1, 8) * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))).astype(np.float32)
    frames = []
    for shift in (4, 2, 0):
        r = np.roll(base, shift, axis=1)
        d = _to_dbz(r)
        d[r < 0.05] = -np.inf
        d[:16, :] = np.nan
        d[:, -12:] = np.nan
        frames.append(d)
    vy = np.zeros((h, w), dtype=np.float32)
    vx = np.full((h, w), 2.0, dtype=np.float32)
    return frames, vy, vx


def _run(frames, vy, vx, *, seed=7, factor=2, spy=None):
    from dmi_nowcast_core._vendor.pysteps_steps.nowcasts import steps as ps_steps

    real = ps_steps.forecast

    def capture(**kwargs):
        out = real(**kwargs)
        if spy is not None:
            spy.append(np.array(out))
        return out

    ps_steps.forecast = capture
    try:
        return run_ensemble(
            frames, vy, vx, n_ens_members=3, n_timesteps=3, n_cascade_levels=4,
            downsample_factor=factor, pixel_scale_m=500.0, seed=seed,
        )
    finally:
        ps_steps.forecast = real


def test_off_coverage_is_nan_and_inside_is_bit_identical():
    frames, vy, vx = _frames()
    captured: list[np.ndarray] = []
    out = _run(frames, vy, vx, spy=captured)
    f = 2
    coverage = ~np.isnan(frames[-1][::f, ::f])
    # Undetect (-inf dBZ) pixels are dry, not off coverage.
    assert np.isneginf(frames[-1][::f, ::f][coverage]).any()
    assert out.shape[2:] == coverage.shape

    assert np.isnan(out[:, :, ~coverage]).all()
    assert np.isfinite(out[:, :, coverage]).all()
    # Inside coverage: exactly what the unmasked conversion produced.
    reference = db_to_rain(captured[0])
    assert np.array_equal(out[:, :, coverage], reference[:, :, coverage])


def test_full_coverage_output_is_the_plain_conversion():
    frames, vy, vx = _frames()
    frames = [np.where(np.isnan(d), -np.inf, d).astype(np.float32) for d in frames]
    captured: list[np.ndarray] = []
    out = _run(frames, vy, vx, spy=captured)
    assert np.array_equal(out, db_to_rain(captured[0]))


def test_national_products_read_off_coverage_as_no_data():
    frames, vy, vx = _frames()
    out = _run(frames, vy, vx)
    coverage = ~np.isnan(frames[-1][::2, ::2])
    products = national_products(
        out, leads_min=(10, 20), timestep_min=10.0, frame_age_min=0.0,
        threshold_mm_h=0.5, downsample_factor=2,
    )
    for lead in (10, 20):
        grid = products.p_rain[lead]
        assert np.isnan(grid[~coverage]).all()
        assert np.isfinite(grid[coverage]).all()
    assert np.isnan(products.eta_min[~coverage]).all()
    assert np.isnan(products.intensity_mm_h[~coverage]).all()


def test_same_seed_reproduces_and_a_new_seed_changes_the_noise():
    frames, vy, vx = _frames()
    a = _run(frames, vy, vx, seed=11)
    b = _run(frames, vy, vx, seed=11)
    c = _run(frames, vy, vx, seed=12)
    assert np.array_equal(a, b, equal_nan=True)
    assert not np.array_equal(a, c, equal_nan=True)
    # The coverage mask does not depend on the seed.
    assert np.array_equal(np.isnan(a), np.isnan(c))


def test_run_ensemble_leaves_the_global_rng_alone():
    frames, vy, vx = _frames()
    np.random.seed(123)
    before = np.random.get_state()
    _run(frames, vy, vx, seed=99)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_ensemble_seed_is_minutes_since_epoch():
    ts = datetime(2026, 9, 8, 17, 40, tzinfo=timezone.utc)
    assert ensemble_seed(ts) == int(ts.timestamp()) // 60
    # DMI's occasional :01 s stamp keeps the frame's seed.
    assert ensemble_seed(ts + timedelta(seconds=1)) == ensemble_seed(ts)
    # Naive = UTC.
    assert ensemble_seed(ts.replace(tzinfo=None)) == ensemble_seed(ts)
    # Consecutive frames differ.
    assert ensemble_seed(ts + timedelta(minutes=10)) == ensemble_seed(ts) + 10
    # Fits the RandomState seed range for centuries.
    assert 0 < ensemble_seed(ts) < 2**32


@pytest.mark.parametrize("factor", [1, 4])
def test_mask_follows_the_downsample(factor):
    frames, vy, vx = _frames(h=64, w=64)
    out = run_ensemble(
        frames, vy, vx, n_ens_members=2, n_timesteps=2, n_cascade_levels=4,
        downsample_factor=factor, pixel_scale_m=500.0, seed=3,
    )
    coverage = ~np.isnan(frames[-1][::factor, ::factor])
    assert np.array_equal(np.isnan(out).any(axis=(0, 1)), ~coverage)
