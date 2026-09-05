"""Parquet archive of DMI metObs station observations (Phase F).

Sits next to the radar composite archive under the same corpus root, so
one bind-mounted volume holds both halves of the benchmark::

    <root>/
      composites/                 (dmi_nowcast_core.corpus)
      stations/
        catalogue.parquet         station metadata, one row per station
        obs/
          YYYY/MM.parquet         observations, one row per
                                  (station, parameter, instant)

Month partitions keep every rewrite bounded: a day's backfill touches one
file of ~1.4M rows rather than a single ever-growing table. Writes are
atomic (tmp + rename) and idempotent — appending the same day twice is a
no-op, which is what makes the backfill resumable and the sidecar poller's
overlapping 40-minute lookback free.

pyarrow only, deliberately: no pandas anywhere in this package, and the
dedupe/sort work is all vectorised Arrow compute.

Data licence: CC BY 4.0 (DMI Open Data).
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from .metobs import Observation, Station

#: Bump when the on-disk column set changes incompatibly.
SCHEMA_VERSION = 1

#: The dedupe key: one reading per station per parameter per instant.
OBS_KEY = ("station_id", "observed_utc", "parameter_id")


def _pa():
    import pyarrow as pa

    return pa


def obs_schema():
    """Explicit Arrow schema for the observation partitions.

    ``observed_utc`` is a UTC-stamped microsecond timestamp — never a
    naive one, so a reader in any zone gets the same instant. ``value``
    is float32: gauge readings have 0.1 mm resolution and durations are
    whole minutes, so float64 would be four bytes of nothing per row
    across ~17M rows a year.
    """
    pa = _pa()
    return pa.schema([
        ("station_id", pa.string()),
        ("observed_utc", pa.timestamp("us", tz="UTC")),
        ("parameter_id", pa.string()),
        ("value", pa.float32()),
    ])


def catalogue_schema():
    pa = _pa()
    return pa.schema([
        ("station_id", pa.string()),
        ("name", pa.string()),
        ("kind", pa.string()),
        ("lat", pa.float64()),
        ("lon", pa.float64()),
        ("country", pa.string()),
        ("operation_from", pa.timestamp("us", tz="UTC")),
        ("operation_to", pa.timestamp("us", tz="UTC")),
        ("status", pa.string()),
        ("parameter_ids", pa.list_(pa.string())),
        ("region_id", pa.string()),
    ])


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("station store timestamps must be timezone-aware UTC")
    return dt.astimezone(timezone.utc)


def _write_atomic(table, path: Path) -> None:
    """Write ``table`` to ``path`` via tmp + rename in the same directory."""
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        pq.write_table(table, tmp, compression="zstd")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _dedupe_last(table, keys: Sequence[str]):
    """Keep the last row per key group — vectorised, no pandas.

    Adds a row index, groups by the key columns taking ``max`` of that
    index, then ``take``s the winners. Later rows win, so re-fetching a
    slot DMI has since corrected replaces the old value instead of
    duplicating it.
    """
    pa = _pa()

    if table.num_rows == 0:
        return table
    idx = pa.array(range(table.num_rows), type=pa.int64())
    with_idx = table.append_column("__row_idx", idx)
    winners = with_idx.group_by(list(keys)).aggregate([("__row_idx", "max")])
    return table.take(winners.column("__row_idx_max"))


class StationObsStore:
    """Read/write the ``stations/`` half of a corpus directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    # -- paths ------------------------------------------------------------

    @property
    def stations_dir(self) -> Path:
        return self.root / "stations"

    @property
    def obs_dir(self) -> Path:
        return self.stations_dir / "obs"

    @property
    def catalogue_path(self) -> Path:
        return self.stations_dir / "catalogue.parquet"

    def partition_path(self, year: int, month: int) -> Path:
        return self.obs_dir / f"{year:04d}" / f"{month:02d}.parquet"

    def partitions(self) -> list[Path]:
        """Every existing month partition, oldest first."""
        if not self.obs_dir.is_dir():
            return []
        return sorted(self.obs_dir.glob("*/*.parquet"))

    # -- observations -----------------------------------------------------

    def append(self, observations: Iterable[Observation]) -> dict[str, int]:
        """Merge ``observations`` into their month partitions.

        Idempotent on ``(station_id, observed_utc, parameter_id)``: the
        partition is read, concatenated, deduped (last wins), sorted and
        rewritten atomically. Returns ``{"YYYY-MM": rows_written}`` plus
        a ``"new"`` total — the net row growth, which is 0 for a replay.
        """
        pa = _pa()
        import pyarrow.parquet as pq

        by_month: dict[tuple[int, int], list[Observation]] = {}
        for obs in observations:
            ts = _as_utc(obs.observed_utc)
            by_month.setdefault((ts.year, ts.month), []).append(obs)

        written: dict[str, int] = {}
        new_rows = 0
        for (year, month), rows in sorted(by_month.items()):
            path = self.partition_path(year, month)
            before = pq.read_metadata(path).num_rows if path.exists() else 0
            incoming = pa.table(
                {
                    "station_id": pa.array([r.station_id for r in rows], pa.string()),
                    "observed_utc": pa.array(
                        [_as_utc(r.observed_utc) for r in rows],
                        pa.timestamp("us", tz="UTC"),
                    ),
                    "parameter_id": pa.array([r.parameter_id for r in rows], pa.string()),
                    "value": pa.array([float(r.value) for r in rows], pa.float32()),
                },
                schema=obs_schema(),
            )
            if path.exists():
                existing = pq.read_table(path, schema=obs_schema())
                combined = pa.concat_tables([existing, incoming])
            else:
                combined = incoming
            merged = _dedupe_last(combined, OBS_KEY).sort_by([
                ("observed_utc", "ascending"),
                ("station_id", "ascending"),
                ("parameter_id", "ascending"),
            ])
            _write_atomic(merged, path)
            written[f"{year:04d}-{month:02d}"] = merged.num_rows
            new_rows += merged.num_rows - before
        written["new"] = new_rows
        return written

    def read(
        self,
        start_utc: datetime,
        end_utc: datetime,
        parameter_ids: Sequence[str] | None = None,
        station_ids: Sequence[str] | None = None,
    ):
        """Observations in ``[start_utc, end_utc]`` (both ends inclusive).

        Inclusive on both sides to match DMI's own ``datetime`` interval
        semantics, so a window written by the backfill reads back with
        exactly the rows that were requested. Returns an empty table with
        the full schema when nothing matches — callers never have to
        special-case "no data yet".
        """
        pa = _pa()
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        start = _as_utc(start_utc)
        end = _as_utc(end_utc)
        wanted = [
            self.partition_path(y, m)
            for (y, m) in _months_between(start, end)
        ]
        tables = []
        for path in wanted:
            if not path.exists():
                continue
            tbl = pq.read_table(path, schema=obs_schema())
            mask = pc.and_(
                pc.greater_equal(tbl.column("observed_utc"), pa.scalar(start, obs_schema().field("observed_utc").type)),
                pc.less_equal(tbl.column("observed_utc"), pa.scalar(end, obs_schema().field("observed_utc").type)),
            )
            if parameter_ids is not None:
                mask = pc.and_(
                    mask, pc.is_in(tbl.column("parameter_id"), value_set=pa.array(list(parameter_ids), pa.string()))
                )
            if station_ids is not None:
                mask = pc.and_(
                    mask, pc.is_in(tbl.column("station_id"), value_set=pa.array(list(station_ids), pa.string()))
                )
            tables.append(tbl.filter(mask))
        if not tables:
            return obs_schema().empty_table()
        return pa.concat_tables(tables).sort_by([
            ("observed_utc", "ascending"),
            ("station_id", "ascending"),
            ("parameter_id", "ascending"),
        ])

    # -- catalogue --------------------------------------------------------

    def write_catalogue(self, stations: Iterable[Station]) -> int:
        """Replace the catalogue with ``stations`` (deduped by id)."""
        pa = _pa()

        rows = list(stations)
        table = pa.table(
            {
                "station_id": pa.array([s.station_id for s in rows], pa.string()),
                "name": pa.array([s.name for s in rows], pa.string()),
                "kind": pa.array([s.kind for s in rows], pa.string()),
                "lat": pa.array([float(s.lat) for s in rows], pa.float64()),
                "lon": pa.array([float(s.lon) for s in rows], pa.float64()),
                "country": pa.array([s.country for s in rows], pa.string()),
                "operation_from": pa.array(
                    [_opt_utc(s.operation_from) for s in rows],
                    pa.timestamp("us", tz="UTC"),
                ),
                "operation_to": pa.array(
                    [_opt_utc(s.operation_to) for s in rows],
                    pa.timestamp("us", tz="UTC"),
                ),
                "status": pa.array([s.status for s in rows], pa.string()),
                "parameter_ids": pa.array(
                    [list(s.parameter_ids) for s in rows], pa.list_(pa.string())
                ),
                "region_id": pa.array([s.region_id for s in rows], pa.string()),
            },
            schema=catalogue_schema(),
        )
        table = _dedupe_last(table, ("station_id",)).sort_by([
            ("station_id", "ascending"),
        ])
        _write_atomic(table, self.catalogue_path)
        return table.num_rows

    def read_catalogue(self) -> list[Station]:
        """The catalogue as :class:`Station` records (empty if absent)."""
        import pyarrow.parquet as pq

        if not self.catalogue_path.exists():
            return []
        tbl = pq.read_table(self.catalogue_path, schema=catalogue_schema())
        out: list[Station] = []
        for row in tbl.to_pylist():
            out.append(
                Station(
                    station_id=row["station_id"],
                    name=row["name"] or "",
                    kind=row["kind"] or "",
                    lat=row["lat"],
                    lon=row["lon"],
                    country=row["country"] or "",
                    operation_from=_ensure_utc(row["operation_from"]),
                    operation_to=_ensure_utc(row["operation_to"]),
                    status=row["status"] or "",
                    parameter_ids=tuple(row["parameter_ids"] or ()),
                    region_id=row["region_id"] or "",
                )
            )
        return out


def _opt_utc(dt: datetime | None) -> datetime | None:
    return None if dt is None else _as_utc(dt)


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _months_between(start: datetime, end: datetime) -> list[tuple[int, int]]:
    """Every (year, month) the inclusive window touches."""
    out: list[tuple[int, int]] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return out


__all__ = [
    "SCHEMA_VERSION",
    "StationObsStore",
    "obs_schema",
    "catalogue_schema",
]
