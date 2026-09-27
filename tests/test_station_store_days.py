"""Review R5: the gauge store's day files read back exactly as the month files did.

The store used to merge every append into its whole month partition
(read, concat, dedupe-last, sort, rewrite). It now rewrites only the day
files an append touches. These tests replay one sequence of appends
through a copy of the OLD whole-partition merge and through the new
store, and require every reader — ``read``, ``read_recent``,
``stream_month`` (as a set of rows) — to answer identically, including
the dedupe (a later write of a key wins) and the order.

They also pin the upgrade path: an archive with a pre-R5 month file,
appended to by the new code, must read as if the old code had done the
append — the day file wins a key it shares with the month file.
"""
from __future__ import annotations

import multiprocessing as mp
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dmi_nowcast_core.metobs import Observation
from dmi_nowcast_core.station_store import (
    OBS_KEY,
    StationObsStore,
    _dedupe_last,
    _observations_table,
    _write_atomic,
    obs_schema,
)

T0 = datetime(2026, 6, 30, 22, 0, tzinfo=timezone.utc)
PARAMS = ("precip_past10min", "precip_dur_past10min")


def _legacy_append(store: StationObsStore, observations) -> None:
    """The pre-R5 ``append``: merge into the whole month partition."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    by_month: dict = {}
    for obs in observations:
        by_month.setdefault(
            (obs.observed_utc.year, obs.observed_utc.month), [],
        ).append(obs)
    for (year, month), rows in sorted(by_month.items()):
        path = store.partition_path(year, month)
        incoming = _observations_table(rows)
        if path.exists():
            combined = pa.concat_tables([
                pq.read_table(path, schema=obs_schema()), incoming,
            ])
        else:
            combined = incoming
        merged = _dedupe_last(combined, OBS_KEY).sort_by([
            ("observed_utc", "ascending"),
            ("station_id", "ascending"),
            ("parameter_id", "ascending"),
        ])
        _write_atomic(merged, path)


def _polls(seed: int = 7, n_polls: int = 30):
    """Overlapping 40-minute lookbacks across a month end, with corrections.

    Every poll re-reads the last four slots; a value can change between
    polls (DMI correcting a slot), so "later write wins" is exercised on
    every key.
    """
    rng = random.Random(seed)
    stations = [f"06{n:03d}" for n in range(12)]
    polls = []
    for k in range(n_polls):
        now = T0 + timedelta(minutes=10 * k)
        rows = []
        for back in range(4):
            slot = now - timedelta(minutes=10 * back)
            for station in stations:
                for param in PARAMS:
                    if rng.random() < 0.1:
                        continue  # late / missing this poll
                    rows.append(Observation(
                        station, slot, param,
                        rng.choice([0.0, -0.1, 0.1, 0.4, 2.5, 10.0]),
                        created_utc=slot + timedelta(minutes=rng.choice([1, 2, 22])),
                    ))
        rng.shuffle(rows)
        polls.append(rows)
    return polls


def _rowset(batches) -> list:
    rows = []
    for batch in batches:
        rows.extend(batch.to_pylist())
    return sorted(rows, key=lambda r: (r["observed_utc"], r["station_id"], r["parameter_id"]))


def _assert_same_reads(new: StationObsStore, old: StationObsStore) -> None:
    windows = [
        (T0 - timedelta(hours=1), T0 + timedelta(hours=8)),
        (T0 + timedelta(minutes=95), T0 + timedelta(minutes=185)),
        (T0 + timedelta(hours=2), T0 + timedelta(hours=2)),
    ]
    for start, end in windows:
        for kwargs in (
            {},
            {"parameter_ids": ["precip_past10min"]},
            {"station_ids": ["06003", "06007"]},
        ):
            expected = old.read(start, end, **kwargs).to_pylist()
            assert new.read(start, end, **kwargs).to_pylist() == expected
            assert new.read_recent(start, end, **kwargs).to_pylist() == expected
    for (year, month) in ((2026, 6), (2026, 7)):
        assert _rowset(new.stream_month(year, month)) == _rowset(
            old.stream_month(year, month),
        )
        assert _rowset(new.stream_month(
            year, month, ["precip_dur_past10min"], ["06001", "06011"],
        )) == _rowset(old.stream_month(
            year, month, ["precip_dur_past10min"], ["06001", "06011"],
        ))


def test_day_files_read_back_exactly_as_the_month_partition(tmp_path: Path) -> None:
    new = StationObsStore(tmp_path / "new")
    old = StationObsStore(tmp_path / "old")
    for rows in _polls():
        new.append(rows)
        _legacy_append(old, rows)
    # The new store wrote day files only, across the month boundary.
    names = [p.name for p in new.partitions()]
    assert names == ["06_30.parquet", "07_01.parquet"]
    assert [p.name for p in old.partitions()] == ["06.parquet", "07.parquet"]
    _assert_same_reads(new, old)


def test_the_month_r5_is_deployed_in_reads_as_the_old_code_would(
    tmp_path: Path,
) -> None:
    """A pre-R5 month file, then new appends that overlap it.

    The first 12 polls land in the month file (the old code); the rest go
    through the new store into day files — and their 40-minute lookback
    re-writes keys the month file already holds, some with corrected
    values. Every read must equal the old code having done all of it.
    """
    polls = _polls(seed=11)
    new = StationObsStore(tmp_path / "new")
    old = StationObsStore(tmp_path / "old")
    for i, rows in enumerate(polls):
        _legacy_append(old, rows)
        if i < 12:
            _legacy_append(new, rows)
        else:
            new.append(rows)
    assert new.partition_path(2026, 6).is_file()
    assert any(p.name.startswith("06_") for p in new.partitions())
    _assert_same_reads(new, old)


def test_a_day_file_wins_a_key_it_shares_with_a_month_file(tmp_path: Path) -> None:
    store = StationObsStore(tmp_path)
    _legacy_append(store, [Observation("06074", T0, "precip_past10min", 0.4)])
    store.append([Observation("06074", T0, "precip_past10min", 1.2)])
    assert store.read(T0, T0).column("value").to_pylist() == [pytest.approx(1.2)]
    assert store.read_recent(T0, T0).column("value").to_pylist() == [
        pytest.approx(1.2),
    ]
    streamed = _rowset(store.stream_month(2026, 6))
    assert [r["value"] for r in streamed] == [pytest.approx(1.2)]
    assert store.station_ids_in_month(2026, 6) == ["06074"]


def test_appending_the_same_poll_twice_is_a_no_op(tmp_path: Path) -> None:
    store = StationObsStore(tmp_path)
    rows = _polls(n_polls=1)[0]
    first = store.append(rows)
    second = store.append(rows)
    assert first["new"] == len({(r.station_id, r.observed_utc, r.parameter_id) for r in rows})
    assert second["new"] == 0


def _append_worker(root: str, station: str, n: int) -> None:
    store = StationObsStore(Path(root))
    for k in range(n):
        store.append([Observation(
            station, T0 + timedelta(minutes=10 * (k % 12)), "precip_past10min", float(k),
        )])


def test_concurrent_writers_lose_no_rows(tmp_path: Path) -> None:
    """The live poller and a backfill writing the same day at once.

    Without the lock, one writer's read-modify-write can start from a copy
    that predates the other's rename, and its rename then drops those rows.
    """
    ctx = mp.get_context("spawn")
    workers = [
        ctx.Process(target=_append_worker, args=(str(tmp_path), f"06{i:03d}", 12))
        for i in range(4)
    ]
    for proc in workers:
        proc.start()
    for proc in workers:
        proc.join(60)
        assert proc.exitcode == 0
    table = StationObsStore(tmp_path).read(T0, T0 + timedelta(hours=2))
    assert table.num_rows == 4 * 12


def test_retention_spans(tmp_path: Path) -> None:
    store = StationObsStore(tmp_path)
    day = store.partition_span(store.day_path(T0.date()))
    assert day == (
        datetime(2026, 6, 30, tzinfo=timezone.utc),
        datetime(2026, 7, 1, tzinfo=timezone.utc),
    )
    month = store.partition_span(store.partition_path(2026, 12))
    assert month == (
        datetime(2026, 12, 1, tzinfo=timezone.utc),
        datetime(2027, 1, 1, tzinfo=timezone.utc),
    )
    assert store.partition_span(tmp_path / "archive" / "old.parquet") is None
    assert store.partition_span(tmp_path / "2026" / "13_01.parquet") is None
