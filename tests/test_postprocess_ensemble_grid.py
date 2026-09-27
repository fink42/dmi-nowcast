"""``ensemble_grid_features`` is ``ensemble_point_features`` at every pixel.

Review R4a (2026-09-27): the on-demand ``/forecast`` path reads its
``ens_*`` columns off per-cycle grids instead of leaving them NaN. Those
grids are only worth having if every pixel carries the number the cycle's
own point function would have given a served point there — bit for bit,
because the served trees split on these values.
"""
from __future__ import annotations

import numpy as np
import pytest

from dmi_nowcast_core import postprocess as P


def _ensemble(seed: int, *, members=16, steps=9, h=37, w=29) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.6, 2.0, size=(members, steps, h, w)).astype(np.float32)
    # Dry patches, so arrivals vary and the ``>= 4 members`` rule bites.
    base[base < 0.8] = 0.0
    # Off-coverage NaN (whole columns), plus scattered member/timestep NaN
    # so the NaN-skipping reductions see every group size.
    base[:, :, :3, :] = np.nan
    holes = rng.random(base.shape) < 0.05
    base[holes] = np.nan
    return base


KW = dict(
    leads_min=[10, 20, 30, 45, 60],
    threshold_mm_h=0.5,
    timestep_min=10.0,
    frame_age_min=17.3,
)


def _same(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return (a.view(np.int32) == b.view(np.int32)) | (np.isnan(a) & np.isnan(b))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n_points", [1, 2, 7, 150])
def test_grid_equals_point_function_bit_for_bit(seed, n_points):
    ens = _ensemble(seed)
    grid = P.ensemble_grid_features(ens, chunk_rows=5, **KW)
    rng = np.random.default_rng(100 + seed)
    h, w = ens.shape[2:]
    pixels = [
        (int(r), int(c))
        for r, c in zip(rng.integers(0, h, n_points), rng.integers(0, w, n_points))
    ]
    point = P.ensemble_point_features(ens, pixels, **KW)
    assert set(grid) == set(point)
    for name, values in point.items():
        at = np.array([grid[name][r, c] for r, c in pixels], dtype=np.float32)
        assert _same(at, values).all(), name


def test_every_pixel_in_one_pass():
    ens = _ensemble(5, h=11, w=13)
    grid = P.ensemble_grid_features(ens, chunk_rows=4, **KW)
    pixels = [(r, c) for r in range(11) for c in range(13)]
    point = P.ensemble_point_features(ens, pixels, **KW)
    for name, values in point.items():
        assert grid[name].shape == (11, 13)
        assert grid[name].dtype == np.float32
        assert _same(grid[name].reshape(-1), values).all(), name
    # The spread is defined somewhere and NaN somewhere (the >= 4 rule).
    spread = grid["ens_eta_spread_min"]
    assert np.isfinite(spread).any() and np.isnan(spread).any()


def test_nanpercentile_columns_matches_numpy():
    rng = np.random.default_rng(3)
    values = rng.random((16, 4000)).astype(np.float32)
    values[rng.random(values.shape) < 0.3] = np.nan
    values[:, :50] = np.nan
    for q in (25.0, 75.0, 90.0):
        with np.errstate(invalid="ignore"), pytest.warns(RuntimeWarning):
            expected = np.nanpercentile(values, q, axis=0)
        got = P._nanpercentile_columns(values, q)
        assert got.dtype == expected.dtype
        assert _same(got, expected).all()


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        P.ensemble_grid_features(np.zeros((2, 3, 4)), **KW)
    with pytest.raises(ValueError):
        P.ensemble_grid_features(
            np.zeros((2, 3, 4, 5)), **{**KW, "timestep_min": 0.0},
        )
