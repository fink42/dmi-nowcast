"""Tests for disc sampling."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from dmi_nowcast_core.geo import CompositeGeo
from dmi_nowcast_core.parse import parse_composite
from dmi_nowcast_core.sample import DiscStats, disc_pixel_indices, sample_disc

FIXTURE = Path(__file__).parent / "fixtures" / "composite_fullrange.h5"
COPENHAGEN = (12.5645, 55.6726)


@pytest.fixture(scope="module")
def geo() -> CompositeGeo:
    return CompositeGeo(parse_composite(FIXTURE))


def test_one_km_disc_at_500m_grid_is_about_a_dozen_pixels():
    """1 km radius / 500 m pixel = 2 px radius → π·2² ≈ 12.6 pixels (plan §6.4)."""
    rows, cols = disc_pixel_indices((100, 100), 50.0, 50.0, 2.0)
    assert 9 <= rows.size <= 16, f"expected ~12 pixels, got {rows.size}"


def test_disc_clipped_at_grid_corner():
    rows, cols = disc_pixel_indices((100, 100), 0.0, 0.0, 5.0)
    # All returned indices must lie inside the grid.
    assert (rows >= 0).all() and (rows < 100).all()
    assert (cols >= 0).all() and (cols < 100).all()


def test_disc_fully_outside_grid_returns_empty():
    rows, cols = disc_pixel_indices((100, 100), 1000.0, 1000.0, 2.0)
    assert rows.size == 0 and cols.size == 0


def test_sample_disc_max_mean_p90_on_constant_field(geo):
    """A constant rain field must yield max = mean = p90 = the constant."""
    field = np.full(geo.composite.reflectivity_dbz.shape, 2.5, dtype=np.float32)
    stats = sample_disc(field, geo, *COPENHAGEN)
    assert stats.max_mm_h == pytest.approx(2.5)
    assert stats.mean_mm_h == pytest.approx(2.5)
    assert stats.p90_mm_h == pytest.approx(2.5)
    assert stats.n_valid == stats.n_pixels_in_disc
    assert stats.n_pixels_in_disc >= 9


def test_sample_disc_handles_nan_pixels(geo):
    """NaN pixels are ignored, not counted toward statistics."""
    field = np.full(geo.composite.reflectivity_dbz.shape, 1.0, dtype=np.float32)
    # Sprinkle NaN at every other pixel.
    field[::2, ::2] = np.nan
    stats = sample_disc(field, geo, *COPENHAGEN)
    assert stats.max_mm_h == pytest.approx(1.0)
    assert stats.n_valid < stats.n_pixels_in_disc  # NaN pixels dropped
    assert stats.n_valid > 0


def test_sample_disc_returns_all_nan_when_no_valid_pixels(geo):
    field = np.full(geo.composite.reflectivity_dbz.shape, np.nan, dtype=np.float32)
    stats = sample_disc(field, geo, *COPENHAGEN)
    assert math.isnan(stats.max_mm_h)
    assert math.isnan(stats.mean_mm_h)
    assert math.isnan(stats.p90_mm_h)
    assert stats.n_valid == 0


def test_p90_lies_between_mean_and_max(geo):
    """p90 reflects the upper end of the distribution: mean ≤ p90 ≤ max.

    Plant hotspots in >10% of the disc so the 90th percentile actually picks up
    a high value (a single-pixel hotspot is at the 100th percentile, so p90
    could fall below mean — that's correct math, not a bug).
    """
    shape = geo.composite.reflectivity_dbz.shape
    field = np.full(shape, 0.1, dtype=np.float32)
    idx = geo.lonlat_to_grid(*COPENHAGEN)
    rc, cc = int(round(idx.row)), int(round(idx.col))
    # Plant a 3×3 hotspot — guarantees >20% of the ~12-pixel disc is high.
    field[rc - 1 : rc + 2, cc - 1 : cc + 2] = 20.0
    stats = sample_disc(field, geo, *COPENHAGEN, radius_m=1000.0)
    assert stats.max_mm_h == pytest.approx(20.0)
    assert stats.mean_mm_h <= stats.p90_mm_h <= stats.max_mm_h


def test_larger_radius_includes_more_pixels(geo):
    """Sanity: pixel count should roughly scale with πr²."""
    field = np.zeros(geo.composite.reflectivity_dbz.shape, dtype=np.float32)
    small = sample_disc(field, geo, *COPENHAGEN, radius_m=500.0)
    large = sample_disc(field, geo, *COPENHAGEN, radius_m=2000.0)
    # 4× radius² → ~16× pixels, but we use 4× linear → ~16× area.
    assert large.n_pixels_in_disc > 4 * small.n_pixels_in_disc


def test_sample_disc_at_grid_edge_returns_clipped_stats(geo):
    """A center right at the edge has fewer pixels but still produces stats."""
    field = np.ones(geo.composite.reflectivity_dbz.shape, dtype=np.float32)
    # UL corner of the grid is at lon=3, lat=60 in geographic terms.
    ul_lon, ul_lat = geo.composite.corners_lonlat["UL"]
    stats = sample_disc(field, geo, ul_lon, ul_lat, radius_m=1000.0)
    # Some pixels lie outside; the quarter-disc still has a few.
    assert 0 < stats.n_pixels_in_disc < 16
    assert stats.max_mm_h == pytest.approx(1.0)
