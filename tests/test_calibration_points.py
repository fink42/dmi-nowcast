"""Tests for the committed national calibration point set.

Validates ``src/dmi_nowcast_core/calibration_points_v2.json`` — the fixed,
versioned point set the multi-point calibration corpus samples — and the
determinism of its generator. No network: the land check is a pure-Python
ray cast against the committed boundary GeoJSON, and the generator reruns use
the committed boundary + the test-fixture composite.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# The generator is a script, not a package — same pattern as
# test_corpus_manifest.py.
sys.path.insert(0, str(REPO / "scripts"))

from build_calibration_points import (  # noqa: E402  (after sys.path edit)
    BOUNDARY_PATH,
    CALIBRATION_REGIONS,
    COAST_BANDS,
    DEFAULT_SEED,
    RADAR_BANDS,
    REFERENCE_ID,
    REFERENCE_LAT,
    REFERENCE_LON,
    generate,
)

POINTS_PATH = REPO / "src" / "dmi_nowcast_core" / "calibration_points_v2.json"
FIXTURE_COMPOSITE = REPO / "tests" / "fixtures" / "composite_fullrange.h5"
# The composite the generator itself verified against, when the archive is
# present on this checkout; the committed fixture composite has the same grid.
ARCHIVE_COMPOSITE = REPO / "radar_archive" / "dk.com.202511221750.500_max.h5"


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(POINTS_PATH.read_text())


# --- schema -----------------------------------------------------------------

def test_schema_top_level(payload):
    assert set(payload) == {"version", "generated", "seed", "provenance", "points"}
    assert payload["version"] == 2
    assert isinstance(payload["generated"], str)
    assert isinstance(payload["seed"], int)
    assert isinstance(payload["provenance"], str) and payload["provenance"]
    assert isinstance(payload["points"], list)


def test_schema_points(payload):
    for p in payload["points"]:
        assert set(p) == {"id", "lat", "lon", "region", "strata"}, p
        assert isinstance(p["id"], str) and p["id"]
        assert isinstance(p["lat"], float)
        assert isinstance(p["lon"], float)
        assert p["region"] in CALIBRATION_REGIONS
        assert set(p["strata"]) == {"coast_km_band", "radar_km_band"}, p
        assert p["strata"]["coast_km_band"] in COAST_BANDS
        assert p["strata"]["radar_km_band"] in RADAR_BANDS


def test_point_count_in_band(payload):
    assert 115 <= len(payload["points"]) <= 125


def test_ids_unique(payload):
    ids = [p["id"] for p in payload["points"]]
    assert len(ids) == len(set(ids))


def test_reference_point_present_with_exact_coords(payload):
    """The fixed reference point is the Fyn centroid, rounded to 2 decimals."""
    ref = [p for p in payload["points"] if p["id"] == REFERENCE_ID]
    assert len(ref) == 1
    assert ref[0]["lat"] == REFERENCE_LAT == 55.33
    assert ref[0]["lon"] == REFERENCE_LON == 10.32
    assert ref[0]["region"] == "Fyn"


def test_reference_point_is_the_fyn_centroid(payload):
    """Recompute the centroid from the committed boundary — the reference
    point must be that polygon's area-weighted centroid, rounded to 2 dp.

    Pure shoelace over the Fyn polygon (the one containing the point); no
    shapely, no network.
    """
    fc = json.loads(BOUNDARY_PATH.read_text())
    polygons = fc["features"][0]["geometry"]["coordinates"]
    fyn = next(
        poly for poly in polygons
        if _point_in_ring(REFERENCE_LON, REFERENCE_LAT, poly[0])
    )
    area = cx = cy = 0.0
    for ring in fyn:
        a = sx = sy = 0.0
        n = len(ring)
        for i in range(n):
            x1, y1 = ring[i][0], ring[i][1]
            x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
            cross = x1 * y2 - x2 * y1
            a += cross
            sx += (x1 + x2) * cross
            sy += (y1 + y2) * cross
        a *= 0.5
        area += a
        cx += sx / 6.0
        cy += sy / 6.0
    assert round(cy / area, 2) == REFERENCE_LAT
    assert round(cx / area, 2) == REFERENCE_LON


# --- geometry ---------------------------------------------------------------

def test_all_points_inside_composite_grid(payload):
    """Every point must land inside the DMI composite grid (CompositeGeo)."""
    from dmi_nowcast_core.geo import CompositeGeo
    from dmi_nowcast_core.parse import parse_composite

    composite = ARCHIVE_COMPOSITE if ARCHIVE_COMPOSITE.exists() else FIXTURE_COMPOSITE
    geo = CompositeGeo(parse_composite(composite))
    outside = [
        p["id"] for p in payload["points"] if not geo.is_in_grid(p["lon"], p["lat"])
    ]
    assert not outside, f"points outside {composite.name}: {outside}"


def _point_in_ring(lon: float, lat: float, ring: list) -> bool:
    """Ray-cast point-in-polygon for one GeoJSON ring ([lon, lat] pairs)."""
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > lat) != (yj > lat):
            if lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
                inside = not inside
        j = i
    return inside


def _on_land(lon: float, lat: float, multipolygon: list) -> bool:
    for polygon in multipolygon:
        exterior, holes = polygon[0], polygon[1:]
        if _point_in_ring(lon, lat, exterior) and not any(
            _point_in_ring(lon, lat, h) for h in holes
        ):
            return True
    return False


def test_all_points_on_danish_land(payload):
    """Shapely-free containment re-check against the committed boundary."""
    fc = json.loads(BOUNDARY_PATH.read_text())
    geom = fc["features"][0]["geometry"]
    assert geom["type"] == "MultiPolygon"
    off_land = [
        p["id"]
        for p in payload["points"]
        if not _on_land(p["lon"], p["lat"], geom["coordinates"])
    ]
    assert not off_land, f"points outside the Denmark polygon: {off_land}"


# --- coverage ---------------------------------------------------------------

def test_every_region_covered(payload):
    counts: dict[str, int] = {name: 0 for name in CALIBRATION_REGIONS}
    for p in payload["points"]:
        counts[p["region"]] += 1
    for region, n in counts.items():
        floor = 2 if region == "Bornholm" else 3
        assert n >= floor, f"{region}: only {n} points (need >= {floor})"


# --- generator determinism --------------------------------------------------

def test_generator_is_deterministic_and_matches_committed(payload):
    """Same seed → identical output; and the committed set is what the
    committed generator produces.

    Runs on the committed boundary file and the fixture composite (same grid
    as the archive composite) — no network. If a future DuckDB/PROJ upgrade
    ever shifts a band-edge classification, the fix is to rerun
    ``scripts/build_calibration_points.py`` and re-commit the JSON.
    """
    duckdb = pytest.importorskip("duckdb")
    try:  # the spatial extension is a separate download — skip when offline
        duckdb.connect(":memory:").execute("INSTALL spatial; LOAD spatial;")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"duckdb spatial extension unavailable: {exc}")
    kwargs = dict(
        seed=DEFAULT_SEED,
        boundary_path=BOUNDARY_PATH,
        composite_path=FIXTURE_COMPOSITE,
        generated="2026-01-01T00:00:00+00:00",  # pinned: determinism is
        # everything-but-the-timestamp
    )
    a = generate(**kwargs)
    b = generate(**kwargs)
    assert a == b
    assert a["seed"] == payload["seed"]
    assert a["points"] == payload["points"]
