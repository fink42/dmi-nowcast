"""Parquet archive of DMI metObs station observations (Phase F).

Sits next to the radar composite archive under the same corpus root, so
one bind-mounted volume holds both halves of the benchmark::

    <root>/
      composites/                 (dmi_nowcast_core.corpus)
      stations/
        catalogue.parquet         station metadata, one row per station
        obs/
          .lock                   flock for every read-modify-write
          YYYY/MM_DD.parquet      one UTC day of observations, one row
                                  per (station, parameter, instant)
          YYYY/MM.parquet         a whole month — the layout before
                                  review R5 (2026-09-27); read, never
                                  written

**Day files** (review R5). Every append rewrites only the day files its
rows fall in — a poll's forty minutes is one day file of at most ~40k
rows (two just after midnight), where it used to be the whole month
partition (~1.4M rows, 0.4-0.6 s and a ~680 MB peak inside the live
process, twice per poll). A day file is final once its day has closed
and a late backfill is the only thing that ever rewrites it again, so
there is nothing to compact: the day file IS the compacted unit. It is
written with row groups of :data:`ROW_GROUP_ROWS`.

**Month files are still read.** An archive written before R5 holds
``YYYY/MM.parquet``, and the month R5 is deployed in has both: its old
month file and the day files written since. Every reader takes the month
file first and the day files after it, and where both carry a key the
DAY FILE WINS — it is the later write, exactly as the old
read-concat-dedupe-last merge would have had it. So no migration step
exists or is needed: the first append after the upgrade writes a day
file beside the month partition and every reader sees the union.

Writes are atomic (tmp + rename) and idempotent — appending the same day
twice is a no-op, which is what makes the backfill resumable and the
sidecar poller's overlapping 40-minute lookback free. The read-modify-
write of a day file runs under an exclusive ``flock`` on ``obs/.lock``,
so a backfill container writing the same day as the live poller cannot
lose either one's rows. Readers take no lock: a day file is replaced by
rename, so a reader sees the old file or the new one, never half of one.

pyarrow only, deliberately: no pandas anywhere in this package, and the
dedupe/sort work is all vectorised Arrow compute.

Data licence: CC BY 4.0 (DMI Open Data).
"""
from __future__ import annotations

import os
import re
import tempfile
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

from .metobs import Observation, Station

#: Bump when the on-disk column set changes incompatibly.
SCHEMA_VERSION = 1

#: The dedupe key: one reading per station per parameter per instant.
OBS_KEY = ("station_id", "observed_utc", "parameter_id")

#: Rows per batch when a partition is streamed rather than materialised.
#: 64k rows of four narrow columns is a couple of megabytes — small enough
#: that a month never lands in memory whole, large enough that the
#: per-batch numpy work still amortises.
STREAM_BATCH_ROWS = 65_536

#: Row-group size of a written day file. A day of every Danish gauge is
#: ~20-40k rows, so a day file is one row group; a backfill that folds
#: several parameters into one day still gets groups the time predicate
#: of :meth:`StationObsStore.read_recent` can skip.
ROW_GROUP_ROWS = 50_000

#: ``MM_DD.parquet`` — a day file's name inside its year directory.
_DAY_FILE = re.compile(r"^(\d{2})_(\d{2})\.parquet$")
#: ``MM.parquet`` — a month file's name (the pre-R5 layout).
_MONTH_FILE = re.compile(r"^(\d{2})\.parquet$")

#: The sort every written file and every read result has.
_SORT = [
    ("observed_utc", "ascending"),
    ("station_id", "ascending"),
    ("parameter_id", "ascending"),
]


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

    ``created_utc`` (added 2026-09-16) is DMI's own publication stamp for
    the reading — the only thing that can measure how far behind real
    time a 10-minute slot becomes available, which is what the gauge
    features' ``gauge_lag_min`` is set from. **Additive and nullable**:
    every read below passes this schema explicitly and pyarrow fills a
    column a partition does not carry with nulls, so a file written
    before it existed reads back exactly as it did — which is why
    :data:`SCHEMA_VERSION` does not move.
    """
    pa = _pa()
    return pa.schema([
        ("station_id", pa.string()),
        ("observed_utc", pa.timestamp("us", tz="UTC")),
        ("parameter_id", pa.string()),
        ("value", pa.float32()),
        ("created_utc", pa.timestamp("us", tz="UTC")),
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


def _write_atomic(table, path: Path, *, row_group_size: int | None = None) -> None:
    """Write ``table`` to ``path`` via tmp + rename in the same directory."""
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        pq.write_table(
            table, tmp, compression="zstd", row_group_size=row_group_size,
        )
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

    @property
    def lock_path(self) -> Path:
        return self.obs_dir / ".lock"

    def partition_path(self, year: int, month: int) -> Path:
        """The MONTH file of the pre-R5 layout (read, never written now)."""
        return self.obs_dir / f"{year:04d}" / f"{month:02d}.parquet"

    def day_path(self, day: date) -> Path:
        """The day file a UTC day's observations are written to."""
        return self.obs_dir / f"{day.year:04d}" / f"{day.month:02d}_{day.day:02d}.parquet"

    def month_files(self, year: int, month: int) -> list[Path]:
        """Every file holding ``year-month`` rows, in precedence order.

        The month file first (when there is one), then the day files in
        day order. A later file wins a key collision, so the order IS the
        merge rule — every reader goes through this list.
        """
        year_dir = self.obs_dir / f"{int(year):04d}"
        out: list[Path] = []
        base = self.partition_path(int(year), int(month))
        if base.is_file():
            out.append(base)
        if year_dir.is_dir():
            out.extend(sorted(
                path for path in year_dir.glob(f"{int(month):02d}_*.parquet")
                if _DAY_FILE.match(path.name) and path.is_file()
            ))
        return out

    def partitions(self) -> list[Path]:
        """Every data file (month files and day files), oldest first.

        Within a month the month file sorts before its day files, which is
        also their precedence order.
        """
        if not self.obs_dir.is_dir():
            return []
        return sorted(
            path for path in self.obs_dir.glob("*/*.parquet")
            if _MONTH_FILE.match(path.name) or _DAY_FILE.match(path.name)
        )

    @staticmethod
    def partition_span(path: Path) -> tuple[datetime, datetime] | None:
        """``[start, end)`` a data file can hold rows for, or None.

        None for a name that is not ``YYYY/MM.parquet`` or
        ``YYYY/MM_DD.parquet`` — retention refuses to date what it does
        not recognise.
        """
        try:
            year = int(path.parent.name)
        except ValueError:
            return None
        try:
            match = _MONTH_FILE.match(path.name)
            if match:
                month = int(match.group(1))
                start = datetime(year, month, 1, tzinfo=timezone.utc)
                end = (
                    datetime(year + 1, 1, 1, tzinfo=timezone.utc) if month == 12
                    else datetime(year, month + 1, 1, tzinfo=timezone.utc)
                )
                return start, end
            match = _DAY_FILE.match(path.name)
            if match:
                start = datetime(
                    year, int(match.group(1)), int(match.group(2)),
                    tzinfo=timezone.utc,
                )
                return start, start + timedelta(days=1)
        except ValueError:
            return None
        return None

    @contextmanager
    def locked(self):
        """Exclusive ``flock`` on ``obs/.lock`` for a read-modify-write.

        Advisory and cross-process: the live poller and a backfill in
        another container on the same volume serialise on it, so neither
        can rewrite a day file from a copy that is missing the other's
        rows. Readers never take it.
        """
        import fcntl

        self.obs_dir.mkdir(parents=True, exist_ok=True)
        with open(self.lock_path, "a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    # -- observations -----------------------------------------------------

    def append(self, observations: Iterable[Observation]) -> dict[str, int]:
        """Merge ``observations`` into their day files.

        Idempotent on ``(station_id, observed_utc, parameter_id)``: each
        touched day file is read, concatenated, deduped (last wins),
        sorted and rewritten atomically, under :meth:`locked`. Returns
        ``{"YYYY-MM-DD": rows_in_that_day_file}`` plus a ``"new"`` total —
        the net growth of the day files, which is 0 for a replay. (On the
        first append into a day an old month file already covers, those
        rows count as new here although a reader already saw them: the
        count is a log figure, the reads are what is exact.)
        """
        pa = _pa()
        import pyarrow.parquet as pq

        by_day: dict[date, list[Observation]] = {}
        for obs in observations:
            ts = _as_utc(obs.observed_utc)
            by_day.setdefault(ts.date(), []).append(obs)

        written: dict[str, int] = {}
        new_rows = 0
        if not by_day:
            written["new"] = 0
            return written
        with self.locked():
            for day, rows in sorted(by_day.items()):
                path = self.day_path(day)
                incoming = _observations_table(rows)
                if path.exists():
                    existing = pq.read_table(path, schema=obs_schema())
                    before = existing.num_rows
                    combined = pa.concat_tables([existing, incoming])
                else:
                    before = 0
                    combined = incoming
                merged = _dedupe_last(combined, OBS_KEY).sort_by(_SORT)
                _write_atomic(merged, path, row_group_size=ROW_GROUP_ROWS)
                written[day.isoformat()] = merged.num_rows
                new_rows += merged.num_rows - before
        written["new"] = new_rows
        return written

    def _window_files(
        self, start: datetime, end: datetime,
    ) -> list[tuple[bool, Path]]:
        """``(is_month_file, path)`` for every file the window can touch.

        Precedence order (see :meth:`month_files`), with day files outside
        ``[start.date(), end.date()]`` skipped by name: they cannot hold a
        row the window wants.
        """
        out: list[tuple[bool, Path]] = []
        first, last = start.date(), end.date()
        for (y, m) in _months_between(start, end):
            for path in self.month_files(y, m):
                span = self.partition_span(path)
                is_month = bool(_MONTH_FILE.match(path.name))
                if not is_month and span is not None:
                    day = span[0].date()
                    if day < first or day > last:
                        continue
                out.append((is_month, path))
        return out

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
        ts_type = obs_schema().field("observed_utc").type
        tables = []
        month_rows = False
        for is_month, path in self._window_files(start, end):
            tbl = pq.read_table(path, schema=obs_schema())
            mask = pc.and_(
                pc.greater_equal(tbl.column("observed_utc"), pa.scalar(start, ts_type)),
                pc.less_equal(tbl.column("observed_utc"), pa.scalar(end, ts_type)),
            )
            if parameter_ids is not None:
                mask = pc.and_(
                    mask, pc.is_in(tbl.column("parameter_id"), value_set=pa.array(list(parameter_ids), pa.string()))
                )
            if station_ids is not None:
                mask = pc.and_(
                    mask, pc.is_in(tbl.column("station_id"), value_set=pa.array(list(station_ids), pa.string()))
                )
            tbl = tbl.filter(mask)
            if tbl.num_rows:
                month_rows = month_rows or is_month
                tables.append(tbl)
        return _merged(tables, month_rows)

    def read_recent(
        self,
        start_utc: datetime,
        end_utc: datetime,
        parameter_ids: Sequence[str] | None = None,
        station_ids: Sequence[str] | None = None,
    ):
        """The same rows as :meth:`read`, with every filter pushed down.

        :meth:`read` decodes each file whole and then masks it. That is
        the right shape for a backfill checking a day it just wrote, and
        the wrong one for the **live cycle**, which asks for the last six
        hours of ~110 stations every time a radar frame lands, inside the
        service that also answers HTTP.

        Here the time window, the parameters and the stations all go into
        the dataset scanner, and only the day files the window touches are
        opened at all (plus a pre-R5 month file, whose row groups the time
        predicate skips by statistics).

        Same schema, same inclusive-both-ends semantics, the same
        month-file-then-day-file precedence and the same sort as
        :meth:`read`, so the two are interchangeable to a caller.
        """
        pa = _pa()
        import pyarrow.dataset as pds

        start = _as_utc(start_utc)
        end = _as_utc(end_utc)
        files = self._window_files(start, end)
        if not files:
            return obs_schema().empty_table()
        predicate = (
            (pds.field("observed_utc") >= pa.scalar(start))
            & (pds.field("observed_utc") <= pa.scalar(end))
        )
        if parameter_ids is not None:
            predicate = predicate & pds.field("parameter_id").isin(
                list(parameter_ids),
            )
        if station_ids is not None:
            predicate = predicate & pds.field("station_id").isin(
                list(station_ids),
            )
        tables = []
        month_rows = False
        # Month files and day files are scanned as two datasets so the
        # precedence (day file wins) survives: one dataset over both would
        # leave the order of the fragments to the scanner.
        for want_month in (True, False):
            paths = [str(p) for is_month, p in files if is_month == want_month]
            if not paths:
                continue
            table = pds.dataset(
                paths, format="parquet", schema=obs_schema(),
            ).to_table(filter=predicate, use_threads=False)
            if table.num_rows:
                month_rows = month_rows or want_month
                tables.append(table)
        return _merged(tables, month_rows)

    def stream_month(
        self,
        year: int,
        month: int,
        parameter_ids: Sequence[str] | None = None,
        station_ids: Sequence[str] | None = None,
        *,
        batch_size: int = STREAM_BATCH_ROWS,
    ):
        """One month as a stream of filtered record batches.

        The bulk path beside :meth:`read`. ``read`` answers "give me this
        window, tidy": it spans every file the window touches,
        concatenates them, sorts the result and hands back one table —
        which is what a caller reading a day wants, and three things a
        caller feeding a vectorised pivot pays for and throws away, since
        the window IS the month there and the pivot sorts by its own key
        anyway.

        Two things keep this bounded where ``read`` is not. The station
        and parameter filters are pushed into the parquet scanner, so the
        other stations' rows are never decoded into Arrow buffers at all;
        and the rows arrive a batch at a time, so a month of 1.1M
        observations costs one batch of resident memory rather than a
        whole materialised month. Reading the ten-month archive whole cost
        about 5.5 GB and got the nightly report killed on the VM.

        **Order is not promised**: the month file's rows come first, then
        the day files'. Where a pre-R5 month file and a day file both
        carry a key, the month file's row is dropped (the day file is the
        later write), so every key is yielded once, with the value
        :meth:`read` would return.

        Yields nothing at all for a month with no files: a window with no
        archive behind it is not an error, exactly as in :meth:`read`.
        """
        import pyarrow.dataset as pds

        files = self.month_files(int(year), int(month))
        if not files:
            return
        predicate = None
        if parameter_ids is not None:
            predicate = pds.field("parameter_id").isin(list(parameter_ids))
        if station_ids is not None:
            stations = pds.field("station_id").isin(list(station_ids))
            predicate = stations if predicate is None else predicate & stations
        base = [p for p in files if _MONTH_FILE.match(p.name)]
        days = [p for p in files if not _MONTH_FILE.match(p.name)]

        def batches(paths):
            # ``schema=`` for the same reason every other read passes it:
            # a file written before ``created_utc`` existed has no such
            # column, and projecting it by name from a schema-less dataset
            # is "No match for FieldRef.Name(created_utc)" — which took the
            # gauge truth join, and with it every fit, down on 2026-09-16.
            dataset = pds.dataset(
                [str(p) for p in paths], format="parquet", schema=obs_schema(),
            )
            # Single-threaded with one batch in flight: this runs inside a
            # worker that has other work to do, and the scanner's default
            # readahead would keep a dozen batches resident to save time
            # the pivot does not need saved.
            yield from dataset.to_batches(
                columns=list(obs_schema().names),
                filter=predicate,
                batch_size=int(batch_size),
                use_threads=False,
                batch_readahead=1,
                fragment_readahead=1,
            )

        if base:
            shadow = None
            if days:
                # The keys the day files carry — only ever a few days' worth
                # beside a month file (the month R5 was deployed in, or a
                # backfill that re-ran old days), so this set is small.
                keys = [_key_strings(b) for b in batches(days) if b.num_rows]
                if keys:
                    import pyarrow as pa

                    shadow = pa.concat_arrays(keys)
            for batch in batches(base):
                if shadow is not None and batch.num_rows:
                    import pyarrow.compute as pc

                    batch = batch.filter(pc.invert(
                        pc.is_in(_key_strings(batch), value_set=shadow),
                    ))
                yield batch
        if days:
            yield from batches(days)

    def station_ids_in_month(self, year: int, month: int) -> list[str]:
        """Every station id a month's files carry, sorted."""
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        found: set[str] = set()
        for path in self.month_files(int(year), int(month)):
            column = pq.read_table(path, columns=["station_id"]).column("station_id")
            found.update(pc.unique(column.combine_chunks()).to_pylist())
        return sorted(str(s) for s in found if s is not None)

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


def _observations_table(rows: Sequence[Observation]):
    """``rows`` as a table in :func:`obs_schema`, in input order."""
    pa = _pa()
    return pa.table(
        {
            "station_id": pa.array([r.station_id for r in rows], pa.string()),
            "observed_utc": pa.array(
                [_as_utc(r.observed_utc) for r in rows],
                pa.timestamp("us", tz="UTC"),
            ),
            "parameter_id": pa.array([r.parameter_id for r in rows], pa.string()),
            "value": pa.array([float(r.value) for r in rows], pa.float32()),
            # Null wherever the source had no publication stamp — a
            # recorded fixture, or a backfill written before the column
            # existed. The dedupe key does not include it, so re-fetching
            # a slot fills it in.
            "created_utc": pa.array(
                [_opt_utc(getattr(r, "created_utc", None)) for r in rows],
                pa.timestamp("us", tz="UTC"),
            ),
        },
        schema=obs_schema(),
    )


def _merged(tables: list, month_rows: bool):
    """Concatenate reads in precedence order; dedupe only where it can matter.

    Day files never share a key with each other (a key's instant names
    its day), so a dedupe is needed only when a pre-R5 month file
    contributed rows beside them — and there the LATER table wins, which
    is the day file.
    """
    pa = _pa()

    if not tables:
        return obs_schema().empty_table()
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    if month_rows and len(tables) > 1:
        table = _dedupe_last(table, OBS_KEY)
    return table.sort_by(_SORT)


def _key_strings(batch):
    """One string per row naming its ``OBS_KEY`` — for a set membership test."""
    pa = _pa()
    import pyarrow.compute as pc

    return pc.binary_join_element_wise(
        batch.column("station_id"),
        batch.column("parameter_id"),
        batch.column("observed_utc").cast(pa.int64()).cast(pa.string()),
        "\x1f",
    )


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
    "ROW_GROUP_ROWS",
    "SCHEMA_VERSION",
    "STREAM_BATCH_ROWS",
    "StationObsStore",
    "obs_schema",
    "catalogue_schema",
]
