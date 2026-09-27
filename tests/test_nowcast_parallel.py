"""The persistent nowcast thread pool changes speed, never numbers.

``_vendor.pysteps_steps.parallel`` runs the STEPS members, the ar_order
alignment advections and the native-grid ``map_coordinates`` row blocks
concurrently (VENDORING MODIFICATION 8). Every test here pins a parallel
result against the serial one with ``np.array_equal`` — not allclose.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import map_coordinates as scipy_map_coordinates

from dmi_nowcast_core._vendor.pysteps_steps import parallel
from dmi_nowcast_core._vendor.pysteps_steps.extrapolation import semilagrangian
from dmi_nowcast_core.advect import advect_field_series
from dmi_nowcast_core.probabilistic import run_ensemble

CORPUS = Path.home() / "dmi-nowcast-corpus-local" / "composites"


@pytest.fixture
def serial(monkeypatch):
    """Force the serial path everywhere (parallelism 1)."""
    def _set():
        monkeypatch.setattr(parallel, "_workers", 1)
    return _set


# --- run_each ---------------------------------------------------------------


def test_run_each_returns_results_in_index_order():
    out = parallel.run_each(lambda i: i * i, 37)
    assert out == [i * i for i in range(37)]


def test_run_each_uses_the_same_pool_threads_every_call():
    names: set[str] = set()
    barrier_hits = []

    def task(i):
        names.add(threading.current_thread().name)
        barrier_hits.append(i)
        # enough work that several runners pick items up
        return float(np.linalg.norm(np.ones(20000) * i))

    for _ in range(5):
        parallel.run_each(task, 16)
    pool_threads = {n for n in names if n.startswith("nowcast-pool")}
    # Never more pool threads than the configured parallelism minus the
    # caller, however many calls were made: the pool is persistent.
    assert len(pool_threads) <= parallel.workers() - 1
    assert sorted(barrier_hits) == sorted(list(range(16)) * 5)


def test_run_each_reraises_the_first_error_after_all_runners_stop():
    def boom(i):
        if i == 3:
            raise ValueError("item 3")
        return i

    with pytest.raises(ValueError, match="item 3"):
        parallel.run_each(boom, 10)
    # the pool is still usable afterwards
    assert parallel.run_each(lambda i: i, 4) == [0, 1, 2, 3]


def test_nested_run_each_runs_serially_inside_a_task():
    inner_threads: list[set[str]] = []

    def outer(i):
        me = threading.current_thread().name
        seen = set(parallel.run_each(lambda j: threading.current_thread().name, 8))
        inner_threads.append(seen)
        assert seen == {me}
        return i

    assert parallel.run_each(outer, 4) == [0, 1, 2, 3]
    assert len(inner_threads) == 4


def test_run_each_serial_when_parallelism_is_one(serial):
    serial()
    caller = threading.current_thread().name
    assert parallel.run_each(lambda i: threading.current_thread().name, 6) == [caller] * 6


# --- row-chunked map_coordinates -------------------------------------------


@pytest.mark.parametrize("kw", [
    dict(mode="constant", cval=np.nan, order=1, prefilter=False),
    dict(mode="nearest", order=1, prefilter=False),
])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_chunked_map_coordinates_is_bit_identical(monkeypatch, kw, dtype):
    monkeypatch.setattr(semilagrangian, "_ROW_CHUNK_MIN_PIXELS", 1)
    rng = np.random.default_rng(7)
    m, n = 301, 257
    field = rng.gamma(0.6, 3.0, (m, n)).astype(dtype)
    field[rng.random((m, n)) < 0.05] = np.nan
    yy, xx = np.indices((m, n), dtype=np.float64)
    ys = yy + rng.normal(0, 6, (m, n))
    xs = xx + rng.normal(0, 6, (m, n))
    expected = scipy_map_coordinates(field, [ys, xs], **kw)
    got = semilagrangian.map_coordinates(field, [ys, xs], **kw)
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected, equal_nan=True)


def _flow_case(h=240, w=260, seed=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((h, w))
    field = np.zeros((h, w), np.float32)
    for _ in range(12):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        s = rng.uniform(6, 20)
        field += rng.uniform(0.5, 9) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
    field[field < 0.05] = 0.0
    field[:4, :] = np.nan
    vy = (1.5 + 0.8 * np.sin(xx / 40.0)).astype(np.float32)
    vx = (2.5 + 0.6 * np.cos(yy / 30.0)).astype(np.float32)
    return field.astype(np.float32), vy, vx


def test_advect_field_series_identical_with_and_without_the_pool(monkeypatch):
    field, vy, vx = _flow_case()
    horizons = [14.7, 19.7, 24.7, 34.7, 44.7, 59.7, 74.7]
    monkeypatch.setattr(semilagrangian, "_ROW_CHUNK_MIN_PIXELS", 1)
    threaded = list(advect_field_series(field, vy, vx, horizons_minutes=horizons, dt_minutes=10.0))
    monkeypatch.setattr(parallel, "_workers", 1)
    serial = list(advect_field_series(field, vy, vx, horizons_minutes=horizons, dt_minutes=10.0))
    for a, b in zip(threaded, serial):
        assert np.array_equal(a, b, equal_nan=True)


# --- STEPS members in parallel ---------------------------------------------


def _to_dbz(r):
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 10.0 * np.log10(200.0) + 16.0 * np.log10(np.maximum(r, 1e-6))
    return d.astype(np.float32)


def test_run_ensemble_parallel_members_equal_serial_synthetic():
    field, vy, vx = _flow_case(h=192, w=208, seed=11)
    field = np.nan_to_num(field)
    frames = [_to_dbz(np.roll(field, (-2 * k, -3 * k), (0, 1))) for k in (2, 1, 0)]
    kw = dict(n_ens_members=6, n_timesteps=5, n_cascade_levels=6,
              downsample_factor=1, pixel_scale_m=1000.0, timestep_min=10.0, seed=42)
    serial = run_ensemble(frames, vy, vx, num_workers=1, **kw)
    threaded = run_ensemble(frames, vy, vx, num_workers=4, **kw)
    assert serial.shape == (6, 5, 192, 208)
    assert np.array_equal(serial, threaded, equal_nan=True)
    # And the members really differ from one another (the check is not
    # vacuous: a degenerate ensemble would be equal by construction).
    assert not np.array_equal(serial[0], serial[1], equal_nan=True)


def test_run_ensemble_prints_nothing(capsys):
    field, vy, vx = _flow_case(h=96, w=96, seed=5)
    field = np.nan_to_num(field)
    frames = [_to_dbz(np.roll(field, (-k, -k), (0, 1))) for k in (2, 1, 0)]
    run_ensemble(frames, vy, vx, n_ens_members=4, n_timesteps=2, n_cascade_levels=4,
                 downsample_factor=1, pixel_scale_m=1000.0, timestep_min=10.0)
    assert capsys.readouterr().out == ""


_REAL = [CORPUS / f"2026/09/dk.com.2026090817{m}.500_max.h5" for m in ("20", "30", "40")]


@pytest.mark.skipif(
    not all(p.exists() for p in _REAL),
    reason="local radar corpus (~/dmi-nowcast-corpus-local) not available",
)
def test_run_ensemble_parallel_members_equal_serial_real_corpus():
    """16 members x 9 steps on the 2026-09-08 17:40 wet cycle, same seed."""
    from dmi_nowcast_core.dense_flow import dense_flow, estimate_motion
    from dmi_nowcast_core.parse import parse_composite
    from dmi_nowcast_core.transform import dbz_to_rain_rate

    cs = [parse_composite(p) for p in _REAL]
    a, b = cs[-2], cs[-1]
    rain = dbz_to_rain_rate(b.reflectivity_dbz)
    m = estimate_motion(
        a.reflectivity_dbz, b.reflectivity_dbz, rain, pixel_km=0.5, dt_min=10.0,
        support_threshold_mm_h=0.5, completion="confidence",
        confidence_window_px=31, confidence_percentile=40.0, texture_percentile=60.0,
        max_px_per_frame=30.0, flow=dense_flow(a.reflectivity_dbz, b.reflectivity_dbz),
    )
    kw = dict(n_timesteps=9, timestep_min=10.0, n_ens_members=16, n_cascade_levels=6,
              threshold_mm_h=0.5, downsample_factor=4)
    dbz = [c.reflectivity_dbz for c in cs]
    serial = run_ensemble(dbz, m.vy, m.vx, num_workers=1, **kw)
    threaded = run_ensemble(dbz, m.vy, m.vx, num_workers=4, **kw)
    assert np.array_equal(serial, threaded, equal_nan=True)
