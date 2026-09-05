"""StationObsStore: idempotent appends, month partitioning, read filters.

The store's whole job is to make re-fetching free — the backfill is
resumable and the sidecar poller re-reads a 40-minute window every 10
minutes, so an append that duplicated rows would quadruple the archive
within a day.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dmi_nowcast_core.metobs import Observation, Station
from dmi_nowcast_core.station_store import (
    StationObsStore,
    catalogue_schema,
    obs_schema,
)

T0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def _obs(station: str, minutes: int, parameter: str = "precip_past10min",
         value: float = 0.0, base: datetime = T0) -> Observation:
    return Observation(
        station_id=station,
        observed_utc=base + timedelta(minutes=minutes),
        parameter_id=parameter,
        value=value,
    )


def _station(station_id: str = "06074", **kw) -> Station:
    defaults = dict(
        name="Testholm", kind="Synop", lat=55.5, lon=10.5, country="DNK",
        operation_from=datetime(2003, 8, 8, tzinfo=timezone.utc),
        operation_to=None, status="Active",
        parameter_ids=("precip_past10min", "precip_dur_past10min"),
        region_id="6",
    )
    defaults.update(kw)
    return Station(station_id=station_id, **defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Append
# ---------------------------------------------------------------------------


def test_append_writes_the_expected_month_partition(tmp_path: Path) -> None:
    store = StationObsStore(tmp_path)
    store.append([_obs("06074", 0), _obs("06074", 10)])
    partition = tmp_path / "stations" / "obs" / "2026" / "06.parquet"
    assert partition.exists()
    assert store.partitions() == [partition]


def test_append_is_idempotent(tmp_path: Path) -> None:
    """Replaying a day must not grow the archive."""
    store = StationObsStore(tmp_path)
    rows = [_obs("06074", m) for m in range(0, 60, 10)]
    first = store.append(rows)
    assert first["new"] == len(rows)

    second = store.append(rows)
    assert second["new"] == 0
    assert store.read(T0, T0 + timedelta(hours=1)).num_rows == len(rows)


def test_append_overwrites_a_corrected_value(tmp_path: Path) -> None:
    """DMI restates late/corrected readings for a slot already stored;
    the newer value must replace the old one, not sit beside it."""
    store = StationObsStore(tmp_path)
    store.append([_obs("06074", 0, value=0.0)])
    store.append([_obs("06074", 0, value=1.4)])
    table = store.read(T0, T0)
    assert table.num_rows == 1
    assert table.column("value").to_pylist() == [pytest.approx(1.4)]


def test_key_is_station_time_and_parameter(tmp_path: Path) -> None:
    """Same instant, same station, two parameters — two rows."""
    store = StationObsStore(tmp_path)
    store.append([
        _obs("06074", 0, "precip_past10min", 0.2),
        _obs("06074", 0, "precip_dur_past10min", 3.0),
        _obs("06188", 0, "precip_past10min", 0.0),
    ])
    assert store.read(T0, T0).num_rows == 3


def test_append_splits_across_month_boundaries(tmp_path: Path) -> None:
    store = StationObsStore(tmp_path)
    june = datetime(2026, 6, 30, 23, 50, tzinfo=timezone.utc)
    july = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)
    store.append([
        Observation("06074", june, "precip_past10min", 0.1),
        Observation("06074", july, "precip_past10min", 0.2),
    ])
    assert (tmp_path / "stations" / "obs" / "2026" / "06.parquet").exists()
    assert (tmp_path / "stations" / "obs" / "2026" / "07.parquet").exists()
    assert store.read(june, july).num_rows == 2


def test_append_rejects_naive_timestamps(tmp_path: Path) -> None:
    store = StationObsStore(tmp_path)
    with pytest.raises(ValueError):
        store.append([Observation("06074", datetime(2026, 6, 1, 12), "p", 0.0)])


def test_stored_schema_is_explicit(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    store = StationObsStore(tmp_path)
    store.append([_obs("06074", 0, value=1.5)])
    schema = pq.read_schema(store.partition_path(2026, 6))
    assert schema.equals(obs_schema())
    field = schema.field("observed_utc")
    assert str(field.type.tz) == "UTC"


def test_trace_values_round_trip_unchanged(tmp_path: Path) -> None:
    """The store archives what DMI reported. -0.1 is normalised at the
    point of use, never on the way in."""
    store = StationObsStore(tmp_path)
    store.append([_obs("06074", 0, value=-0.1)])
    assert store.read(T0, T0).column("value").to_pylist() == [pytest.approx(-0.1)]


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def test_read_window_is_inclusive_at_both_ends(tmp_path: Path) -> None:
    store = StationObsStore(tmp_path)
    store.append([_obs("06074", m) for m in (0, 10, 20)])
    table = store.read(T0, T0 + timedelta(minutes=20))
    assert table.num_rows == 3
    table = store.read(T0 + timedelta(minutes=10), T0 + timedelta(minutes=10))
    assert table.num_rows == 1


def test_read_filters_by_parameter_and_station(tmp_path: Path) -> None:
    store = StationObsStore(tmp_path)
    store.append([
        _obs("06074", 0, "precip_past10min"),
        _obs("06074", 0, "precip_dur_past10min"),
        _obs("06188", 0, "precip_past10min"),
    ])
    end = T0 + timedelta(hours=1)
    assert store.read(T0, end, parameter_ids=["precip_past10min"]).num_rows == 2
    assert store.read(T0, end, station_ids=["06074"]).num_rows == 2
    assert store.read(
        T0, end, parameter_ids=["precip_past10min"], station_ids=["06188"],
    ).num_rows == 1


def test_read_of_an_empty_store_returns_the_typed_empty_table(tmp_path: Path) -> None:
    """Callers never special-case "no data yet"."""
    table = StationObsStore(tmp_path).read(T0, T0 + timedelta(hours=1))
    assert table.num_rows == 0
    assert table.schema.equals(obs_schema())


def test_read_is_sorted_by_time(tmp_path: Path) -> None:
    store = StationObsStore(tmp_path)
    store.append([_obs("06074", m) for m in (20, 0, 10)])
    stamps = store.read(T0, T0 + timedelta(hours=1)).column("observed_utc").to_pylist()
    assert stamps == sorted(stamps)


def test_read_spans_month_partitions(tmp_path: Path) -> None:
    store = StationObsStore(tmp_path)
    for month in (5, 6, 7):
        store.append([
            Observation("06074", datetime(2026, month, 15, tzinfo=timezone.utc),
                        "precip_past10min", float(month))
        ])
    table = store.read(
        datetime(2026, 5, 1, tzinfo=timezone.utc),
        datetime(2026, 7, 31, tzinfo=timezone.utc),
    )
    assert table.column("value").to_pylist() == [5.0, 6.0, 7.0]


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


def test_catalogue_round_trips(tmp_path: Path) -> None:
    store = StationObsStore(tmp_path)
    stations = [_station("06074"), _station("06188", name="Sjælsmark", kind="Pluvio")]
    assert store.write_catalogue(stations) == 2
    read_back = {s.station_id: s for s in store.read_catalogue()}
    assert set(read_back) == {"06074", "06188"}
    assert read_back["06188"].name == "Sjælsmark"
    assert read_back["06188"].kind == "Pluvio"
    assert read_back["06074"].parameter_ids == (
        "precip_past10min", "precip_dur_past10min",
    )
    assert read_back["06074"].operation_from == datetime(2003, 8, 8, tzinfo=timezone.utc)
    assert read_back["06074"].operation_to is None


def test_catalogue_write_is_a_replacement_not_an_append(tmp_path: Path) -> None:
    store = StationObsStore(tmp_path)
    store.write_catalogue([_station("06074"), _station("06188")])
    store.write_catalogue([_station("06074", name="Renamed")])
    catalogue = store.read_catalogue()
    assert [s.station_id for s in catalogue] == ["06074"]
    assert catalogue[0].name == "Renamed"


def test_catalogue_dedupes_by_station_id(tmp_path: Path) -> None:
    store = StationObsStore(tmp_path)
    store.write_catalogue([_station("06074", name="old"), _station("06074", name="new")])
    assert [s.name for s in store.read_catalogue()] == ["new"]


def test_missing_catalogue_reads_as_empty(tmp_path: Path) -> None:
    assert StationObsStore(tmp_path).read_catalogue() == []


def test_catalogue_schema_is_explicit(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    store = StationObsStore(tmp_path)
    store.write_catalogue([_station()])
    assert pq.read_schema(store.catalogue_path).equals(catalogue_schema())


def test_writes_leave_no_temp_files_behind(tmp_path: Path) -> None:
    store = StationObsStore(tmp_path)
    store.append([_obs("06074", 0)])
    store.write_catalogue([_station()])
    assert not list(tmp_path.rglob("*.tmp"))
