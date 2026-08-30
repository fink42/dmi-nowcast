"""Tests for geo.py — pyproj round-trip and grid indexing on a real DMI composite."""
from __future__ import annotations

from pathlib import Path

import pytest

from dmi_nowcast_core.geo import CompositeGeo
from dmi_nowcast_core.parse import parse_composite

FIXTURE = Path(__file__).parent / "fixtures" / "composite_fullrange.h5"

# Known Danish landmarks (lon, lat). Coordinates from public maps.
COPENHAGEN_CENTRAL = (12.5645, 55.6726)
AARHUS_CITY_HALL = (10.2107, 56.1572)
ODENSE_CENTRAL = (10.32, 55.33)
ROEMOE_RADAR = (8.4900, 55.1736)  # one of DMI's own radar sites


@pytest.fixture(scope="module")
def geo() -> CompositeGeo:
    return CompositeGeo(parse_composite(FIXTURE))


@pytest.mark.parametrize(
    "name, lon, lat",
    [
        ("Copenhagen", *COPENHAGEN_CENTRAL),
        ("Aarhus", *AARHUS_CITY_HALL),
        ("Odense", *ODENSE_CENTRAL),
        ("Rømø radar", *ROEMOE_RADAR),
    ],
)
def test_round_trip_is_sub_pixel(geo, name, lon, lat):
    """lat/lon → grid → lat/lon must round-trip to better than half a pixel.

    Half a pixel ≈ 250 m. We assert tighter than that — sub-metre — since the
    pyproj round-trip itself has near-zero numerical error and the only loss
    is float64 precision through the linear grid step.
    """
    idx = geo.lonlat_to_grid(lon, lat)
    lon2, lat2 = geo.grid_to_lonlat(idx.row, idx.col)
    assert abs(lon - lon2) < 1e-6, f"{name}: lon drift {abs(lon - lon2)}"
    assert abs(lat - lat2) < 1e-6, f"{name}: lat drift {abs(lat - lat2)}"


def test_danish_landmarks_fall_inside_the_grid(geo):
    assert geo.is_in_grid(*COPENHAGEN_CENTRAL)
    assert geo.is_in_grid(*AARHUS_CITY_HALL)
    assert geo.is_in_grid(*ODENSE_CENTRAL)
    assert geo.is_in_grid(*ROEMOE_RADAR)


def test_far_outside_is_rejected(geo):
    # North Pole and Sahara — clearly outside any Denmark-centric composite.
    assert not geo.is_in_grid(0.0, 89.0)
    assert not geo.is_in_grid(0.0, 20.0)


def test_ul_corner_maps_to_origin(geo):
    """The geographic UL corner from /where should land on (row, col) ≈ (0, 0)."""
    ul_lon, ul_lat = geo.composite.corners_lonlat["UL"]
    idx = geo.lonlat_to_grid(ul_lon, ul_lat)
    assert abs(idx.row) < 1e-6
    assert abs(idx.col) < 1e-6


def test_grid_geometry_matches_array_shape(geo):
    """The opposite corner should land near (height-ish, width-ish).

    Stereographic projection makes geographic corners not coincide exactly with
    array opposite corners, but they should be within a few percent.
    """
    height, width = geo.composite.reflectivity_dbz.shape
    lr_lon, lr_lat = geo.composite.corners_lonlat["LR"]
    idx = geo.lonlat_to_grid(lr_lon, lr_lat)
    # LR is the geographic bottom-right corner of the data rectangle in projection coords.
    # Row should be near the bottom (height), col should be near the right (width).
    assert idx.row > 0.9 * height
    assert idx.col > 0.9 * width


def test_copenhagen_lands_in_expected_quadrant(geo):
    """Copenhagen is in the south-east of Denmark; should be lower-right quadrant."""
    height, width = geo.composite.reflectivity_dbz.shape
    idx = geo.lonlat_to_grid(*COPENHAGEN_CENTRAL)
    assert idx.row > height / 2, "Copenhagen should be south of grid centre"
    assert idx.col > width / 2, "Copenhagen should be east of grid centre"
