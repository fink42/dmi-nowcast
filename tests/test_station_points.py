"""Station-points builder: who gets selected, and what the points say.

The builder is a script, not a package — ``scripts/`` goes on
``sys.path`` exactly as ``tests/test_calibration_corpus.py`` does.
Everything here runs against a synthetic store in ``tmp_path``.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_station_points as bsp  # noqa: E402  (after sys.path edit)
from dmi_nowcast_core.metobs import Observation, Station  # noqa: E402
from dmi_nowcast_core.station_store import StationObsStore  # noqa: E402

FROM_DAY = date(2026, 6, 1)
TO_DAY = date(2026, 6, 2)


def _station(station_id: str, lat: float, lon: float, *, country: str = "DNK",
             kind: str = "Synop", name: str = "Test",
             parameters: tuple[str, ...] = ("precip_past10min",)) -> Station:
    return Station(
        station_id=station_id, name=name, kind=kind, lat=lat, lon=lon,
        country=country, operation_from=None, operation_to=None,
        status="Active", parameter_ids=parameters, region_id="6",
    )


def _fill(store: StationObsStore, station_id: str, fraction: float,
          parameter: str = "precip_past10min") -> None:
    """Write at least ``fraction`` of the window's slots for one station.

    Rounds UP, so ``_fill(..., 0.8)`` really does clear a 0.8 floor
    instead of landing a fraction of a slot below it.
    """
    start, end = bsp.window_bounds(FROM_DAY, TO_DAY)
    total = bsp.expected_slots(parameter, start, end)
    n = math.ceil(total * fraction)
    store.append([
        Observation(station_id, start + timedelta(minutes=10 * i), parameter, 0.0)
        for i in range(n)
    ])


# ---------------------------------------------------------------------------
# Window arithmetic
# ---------------------------------------------------------------------------


def test_window_is_the_inclusive_day_range() -> None:
    start, end = bsp.window_bounds(FROM_DAY, TO_DAY)
    assert start == datetime(2026, 6, 1, tzinfo=timezone.utc)
    assert end == datetime(2026, 6, 2, 23, 59, 59, tzinfo=timezone.utc)


def test_expected_slots_uses_each_parameter_s_own_cadence() -> None:
    start, end = bsp.window_bounds(FROM_DAY, TO_DAY)
    # Two days: 288 ten-minute slots (the last stamp is 23:50, hence +1),
    # 48 hours, 2880 minutes.
    assert bsp.expected_slots("precip_past10min", start, end) == 288
    assert bsp.expected_slots("precip_dur_past10min", start, end) == 288
    assert bsp.expected_slots("precip_past1h", start, end) == 48
    assert bsp.expected_slots("precip_past1min", start, end) == 2880


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_selects_only_stations_above_the_coverage_floor(tmp_path: Path) -> None:
    store = StationObsStore(tmp_path)
    stations = [
        _station("full", 55.4, 10.4),
        _station("sparse", 56.0, 9.0),
    ]
    _fill(store, "full", 1.0)
    _fill(store, "sparse", 0.5)
    start, end = bsp.window_bounds(FROM_DAY, TO_DAY)
    counts = bsp.observation_counts(store, start, end, bsp.REPORT_PARAMETERS)

    points, rows = bsp.build_points(stations, counts, start, end, 0.8)
    assert [p["id"] for p in points] == ["full"]
    # Both stations still appear in the availability table — the rejects
    # are the interesting half of it.
    assert {r["station_id"] for r in rows} == {"full", "sparse"}
    assert {r["station_id"]: r["selected"] for r in rows} == {
        "full": True, "sparse": False,
    }


def test_coverage_floor_is_inclusive(tmp_path: Path) -> None:
    store = StationObsStore(tmp_path)
    _fill(store, "exact", 0.8)
    start, end = bsp.window_bounds(FROM_DAY, TO_DAY)
    counts = bsp.observation_counts(store, start, end, bsp.REPORT_PARAMETERS)
    points, _ = bsp.build_points([_station("exact", 55.4, 10.4)], counts, start, end, 0.8)
    assert [p["id"] for p in points] == ["exact"]


def test_non_danish_stations_are_excluded(tmp_path: Path) -> None:
    store = StationObsStore(tmp_path)
    _fill(store, "greenland", 1.0)
    _fill(store, "danish", 1.0)
    start, end = bsp.window_bounds(FROM_DAY, TO_DAY)
    counts = bsp.observation_counts(store, start, end, bsp.REPORT_PARAMETERS)
    stations = [
        _station("greenland", 64.2, -51.7, country="GRL"),
        _station("danish", 55.4, 10.4),
    ]
    points, rows = bsp.build_points(stations, counts, start, end, 0.8)
    assert [p["id"] for p in points] == ["danish"]
    assert [r["station_id"] for r in rows] == ["danish"]


def test_selection_is_on_the_amount_parameter_not_the_duration(tmp_path: Path) -> None:
    """A station reporting only wet-minutes has no gauge amount to verify
    against, so it must not qualify."""
    store = StationObsStore(tmp_path)
    _fill(store, "dur-only", 1.0, parameter="precip_dur_past10min")
    start, end = bsp.window_bounds(FROM_DAY, TO_DAY)
    counts = bsp.observation_counts(store, start, end, bsp.REPORT_PARAMETERS)
    points, rows = bsp.build_points(
        [_station("dur-only", 55.4, 10.4, parameters=("precip_dur_past10min",))],
        counts, start, end, 0.8,
    )
    assert points == []
    assert rows[0]["coverage"]["precip_dur_past10min"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Point contents
# ---------------------------------------------------------------------------


def test_point_carries_the_region_and_the_station_id(tmp_path: Path) -> None:
    store = StationObsStore(tmp_path)
    _fill(store, "06126", 1.0)
    start, end = bsp.window_bounds(FROM_DAY, TO_DAY)
    counts = bsp.observation_counts(store, start, end, bsp.REPORT_PARAMETERS)
    points, _ = bsp.build_points(
        [_station("06126", 55.47, 10.33, name="Odense")], counts, start, end, 0.8,
    )
    point = points[0]
    # The id IS the stationId — that is what makes the gauge join possible.
    assert point["id"] == "06126"
    # ``region`` is the FINE box, so reliability_by_region.sql and the
    # regional-split criterion group a station corpus the same way they
    # group the radar-point corpus.
    assert point["region"] == "Fyn"
    # The coarse regions.py value is preserved, but it is "Denmark" for
    # every station and so cannot be the grouping key.
    assert point["strata"]["country_region"] == "Denmark"
    assert point["lat"] == pytest.approx(55.47)
    assert point["lon"] == pytest.approx(10.33)


def test_regions_span_the_calibration_boxes(tmp_path: Path) -> None:
    """One station per box: the station point set must be groupable by the
    same eight region names the radar point set uses."""
    from build_calibration_points import CALIBRATION_REGIONS

    coords = {
        "Bornholm": (55.10, 14.90),
        "Hovedstaden": (55.70, 12.40),
        "Sjælland": (55.40, 11.80),
        "Nordjylland": (57.10, 9.90),
        "Sønderjylland": (54.90, 9.30),
        "Fyn": (55.40, 10.40),
        "Sydjylland": (55.50, 9.50),
        "Midtjylland": (56.20, 9.60),
    }
    assert set(coords) == set(CALIBRATION_REGIONS)

    store = StationObsStore(tmp_path)
    stations = []
    for i, (name, (lat, lon)) in enumerate(coords.items()):
        station_id = f"s{i:02d}"
        _fill(store, station_id, 1.0)
        stations.append(_station(station_id, lat, lon))
    start, end = bsp.window_bounds(FROM_DAY, TO_DAY)
    counts = bsp.observation_counts(store, start, end, bsp.REPORT_PARAMETERS)
    points, _ = bsp.build_points(stations, counts, start, end, 0.8)
    assert {p["region"] for p in points} == set(CALIBRATION_REGIONS)
    assert all(p["strata"]["country_region"] == "Denmark" for p in points)


def test_a_coordinate_outside_every_box_falls_back_to_the_coarse_region() -> None:
    """An empty region would collapse into its own GROUP BY bucket and
    quietly poison a per-region curve."""
    # Anholt-ish water gap / anything the eight boxes miss: assert the
    # contract on the resolver directly so the test does not depend on
    # which coordinate happens to be uncovered.
    assert bsp.resolve_regions(55.47, 10.33) == ("Fyn", "Denmark")
    assert bsp._calibration_region(0.0, 0.0) == ""
    assert bsp.resolve_regions(0.0, 0.0) == ("Other", "Other")


def test_output_is_a_points_file_the_corpus_builder_accepts(tmp_path: Path) -> None:
    """The whole point of the schema: build_calibration_corpus.py must be
    able to consume this file with no changes."""
    from build_calibration_corpus import load_points

    store = StationObsStore(tmp_path / "corpus")
    store.write_catalogue([
        _station("06126", 55.47, 10.33, name="Odense"),
        _station("06188", 55.88, 12.41, name="Sjælsmark"),
    ])
    _fill(store, "06126", 1.0)
    _fill(store, "06188", 0.9)

    out = tmp_path / "station_points.json"
    md = tmp_path / "availability.md"
    rc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "build_station_points.py"),
         "--corpus-dir", str(tmp_path / "corpus"),
         "--from", FROM_DAY.isoformat(), "--to", TO_DAY.isoformat(),
         "--min-coverage", "0.8", "--out", str(out), "--availability-md", str(md)],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    payload = json.loads(out.read_text())
    assert payload["version"] == 2
    assert {p["id"] for p in payload["points"]} == {"06126", "06188"}

    loaded = load_points(out)
    assert {p.id for p in loaded} == {"06126", "06188"}
    # The corpus builder reads ``region`` straight into its Parquet column,
    # so this is the value reliability_by_region.sql will group by.
    assert {p.region for p in loaded} == {"Fyn", "Hovedstaden"}

    table = md.read_text()
    assert "06126" in table and "Odense" in table
    assert "precip_past10min" in table
    assert "| Fyn |" in table


def test_missing_catalogue_fails_loudly(tmp_path: Path) -> None:
    rc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "build_station_points.py"),
         "--corpus-dir", str(tmp_path), "--from", FROM_DAY.isoformat(),
         "--to", TO_DAY.isoformat(), "--out", str(tmp_path / "p.json")],
        capture_output=True, text=True,
    )
    assert rc.returncode == 2
    assert "catalogue" in rc.stderr


def test_availability_markdown_reports_the_1min_flag(tmp_path: Path) -> None:
    store = StationObsStore(tmp_path)
    _fill(store, "with1", 1.0)
    _fill(store, "no1", 1.0)
    start, end = bsp.window_bounds(FROM_DAY, TO_DAY)
    counts = bsp.observation_counts(store, start, end, bsp.REPORT_PARAMETERS)
    stations = [
        _station("with1", 55.4, 10.4,
                 parameters=("precip_past10min", "precip_past1min")),
        _station("no1", 55.5, 10.5),
    ]
    _, rows = bsp.build_points(stations, counts, start, end, 0.8)
    flags = {r["station_id"]: r["has_1min"] for r in rows}
    assert flags == {"with1": True, "no1": False}
    md = bsp.availability_markdown(rows, start, end, 0.8)
    assert "CC BY 4.0" in md
    assert md.count("| yes |") >= 1
