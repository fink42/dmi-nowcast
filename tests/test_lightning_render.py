"""Smoke test for the lightning map renderer — projection + PIL path.

Builds a real CompositeGeo from the fixture, throws synthetic strikes near
home, and checks a valid PNG comes out (no basemap needed)."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dmi_nowcast_core.geo import CompositeGeo
from dmi_nowcast_core.lightning import LightningStrike, summarize_clusters
from dmi_nowcast_core.lightning_render import render_lightning_map
from dmi_nowcast_core.parse import parse_composite

FIXTURE = Path(__file__).parent / "fixtures" / "composite_fullrange.h5"
HOME_LAT, HOME_LON = 55.40, 10.39  # Odense, inside the composite
NOW = datetime(2026, 6, 6, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def geo() -> CompositeGeo:
    return CompositeGeo(parse_composite(FIXTURE))


def _cell():
    def dlon(km):
        return km / (111.320 * math.cos(math.radians(HOME_LAT)))
    return [LightningStrike(HOME_LAT, HOME_LON - dlon(20 + m), NOW - timedelta(minutes=m))
            for m in range(9)]


def test_render_returns_png(geo):
    strikes = _cell()
    clusters = summarize_clusters(strikes, HOME_LAT, HOME_LON, now=NOW)
    png = render_lightning_map(
        geo=geo,
        pixel_scale_m=geo.composite.xscale_m,
        home_lat=HOME_LAT, home_lon=HOME_LON,
        strikes=strikes, clusters=clusters,
        pixel_lat=HOME_LAT, pixel_lon=HOME_LON,
        zoom_km=100.0, output_px=500, basemap=None, now=NOW,
        header="approaching | ETA 8m | 9 strikes",
    )
    assert isinstance(png, (bytes, bytearray)) and len(png) > 1000
    assert png[:8] == b"\x89PNG\r\n\x1a\n"  # PNG signature


def test_render_empty_ok(geo):
    png = render_lightning_map(
        geo=geo, pixel_scale_m=geo.composite.xscale_m,
        home_lat=HOME_LAT, home_lon=HOME_LON,
        strikes=[], clusters=[], now=NOW, basemap=None,
    )
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
