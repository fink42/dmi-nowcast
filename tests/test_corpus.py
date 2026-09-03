"""Tests for the persistent corpus archiver and its listing index."""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dmi_nowcast_core.corpus import (
    SCAN_TYPE_DOPPLER,
    SCAN_TYPE_FULL_RANGE,
    SCAN_TYPE_UNKNOWN,
    ArchivedFrame,
    ArchiveIndex,
    ArchiveResult,
    CorpusArchiver,
    archive_path_for,
    parse_filename_ts,
    scan_type_from_filename,
)


def _write_blob(path: Path, payload: bytes = b"x" * 1024) -> Path:
    path.write_bytes(payload)
    return path


def test_parse_filename_ts_extracts_utc_timestamp() -> None:
    ts = parse_filename_ts("dk.com.202605210520.500_max.h5")
    assert ts == datetime(2026, 5, 21, 5, 20, tzinfo=timezone.utc)


def test_parse_filename_ts_rejects_non_matching_names() -> None:
    with pytest.raises(ValueError):
        parse_filename_ts("not-a-dmi-file.h5")
    with pytest.raises(ValueError):
        parse_filename_ts("dk.com.202605.500_max.h5")  # missing minutes


def test_archive_path_for_uses_year_month_tree(tmp_path: Path) -> None:
    p = archive_path_for(tmp_path, "dk.com.202602010005.500_max.h5")
    assert p == tmp_path / "composites" / "2026" / "02" / "dk.com.202602010005.500_max.h5"


def test_archive_copies_into_year_month_slot(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    src_dir = tmp_path / "cache"
    src_dir.mkdir()
    src = _write_blob(src_dir / "dk.com.202605210520.500_max.h5", b"hello world")

    archiver = CorpusArchiver(corpus)
    result = archiver.archive(src)

    expected = corpus / "composites" / "2026" / "05" / src.name
    assert result == ArchiveResult(path=expected, archived=True)
    assert expected.is_file()
    assert expected.read_bytes() == b"hello world"


def test_archive_is_idempotent(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    src_dir = tmp_path / "cache"
    src_dir.mkdir()
    src = _write_blob(src_dir / "dk.com.202605210525.500_max.h5", b"payload")

    archiver = CorpusArchiver(corpus)
    first = archiver.archive(src)
    assert first.archived is True

    # Second call should detect the existing file and skip.
    second = archiver.archive(src)
    assert second.archived is False
    assert second.path == first.path


def test_archive_preserves_source_mtime(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    src = _write_blob(tmp_path / "dk.com.202605210530.500_max.h5")
    # Force a deterministic mtime so we can assert it round-trips.
    pinned = time.time() - 3600.0
    os.utime(src, (pinned, pinned))

    archiver = CorpusArchiver(corpus)
    result = archiver.archive(src)

    dst_mtime = result.path.stat().st_mtime
    assert abs(dst_mtime - pinned) < 1.0  # filesystem mtime resolution slack


def test_archive_does_not_disturb_source(tmp_path: Path) -> None:
    """Archiver must copy, not move — the working cache owns the original."""
    src = _write_blob(tmp_path / "dk.com.202605210535.500_max.h5", b"original")
    archiver = CorpusArchiver(tmp_path / "corpus")
    archiver.archive(src)
    assert src.is_file()
    assert src.read_bytes() == b"original"


def test_archive_rejects_unknown_filename(tmp_path: Path) -> None:
    src = _write_blob(tmp_path / "not-a-dmi-file.h5")
    archiver = CorpusArchiver(tmp_path / "corpus")
    with pytest.raises(ValueError):
        archiver.archive(src)


def test_iter_files_returns_archived_in_chronological_order(tmp_path: Path) -> None:
    archiver = CorpusArchiver(tmp_path / "corpus")
    # Out-of-order archive calls; iter_files must still yield chronologically.
    names = [
        "dk.com.202605210510.500_max.h5",
        "dk.com.202605210500.500_max.h5",
        "dk.com.202605210515.500_max.h5",
        "dk.com.202604010000.500_max.h5",
    ]
    for n in names:
        archiver.archive(_write_blob(tmp_path / n))

    timestamps = [ts for ts, _ in archiver.iter_files()]
    assert timestamps == sorted(timestamps)
    assert archiver.count() == 4
    assert archiver.total_bytes() == 4 * 1024


def test_iter_files_ignores_unrelated_files(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    archiver = CorpusArchiver(corpus)
    archiver.archive(_write_blob(tmp_path / "dk.com.202605210520.500_max.h5"))
    # Drop a stray file directly into the tree — must be ignored.
    (corpus / "composites" / "2026" / "05" / "notes.txt").write_text("hi")

    assert archiver.count() == 1
    assert next(iter(archiver.iter_files()))[1].name.startswith("dk.com.")


def test_archive_uses_tempfile_then_rename(tmp_path: Path) -> None:
    """After a successful archive, no leftover ``.tmp`` files."""
    archiver = CorpusArchiver(tmp_path / "corpus")
    archiver.archive(_write_blob(tmp_path / "dk.com.202605210540.500_max.h5"))
    tmps = list((tmp_path / "corpus").rglob("*.tmp"))
    assert tmps == []


# ---------------------------------------------------------------------------
# Scan type from the filename minute (:x0 fullRange / :x5 doppler)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("minute, expected", [
    (0, SCAN_TYPE_FULL_RANGE),
    (10, SCAN_TYPE_FULL_RANGE),
    (30, SCAN_TYPE_FULL_RANGE),
    (50, SCAN_TYPE_FULL_RANGE),
    (5, SCAN_TYPE_DOPPLER),
    (15, SCAN_TYPE_DOPPLER),
    (45, SCAN_TYPE_DOPPLER),
    (55, SCAN_TYPE_DOPPLER),
])
def test_scan_type_from_filename_reads_the_minute(minute: int, expected: str) -> None:
    """The filename minute IS the scan-type marker (verified against DMI's
    own scanType-labelled listing for 12 consecutive frames, 2026-09-02)."""
    assert scan_type_from_filename(
        f"dk.com.202605210{7}{minute:02d}.500_max.h5"
    ) == expected


@pytest.mark.parametrize("minute", [1, 2, 3, 4, 6, 7, 8, 9, 23, 47])
def test_scan_type_from_filename_off_grid_is_unknown_never_fullrange(
    minute: int,
) -> None:
    """An off-grid minute must never be mistaken for fullRange — mixing the
    products alternates between materially different views."""
    got = scan_type_from_filename(f"dk.com.2026052107{minute:02d}.500_max.h5")
    assert got == SCAN_TYPE_UNKNOWN
    assert got != SCAN_TYPE_FULL_RANGE


def test_scan_type_from_filename_rejects_non_composite_names() -> None:
    """A name that is not a composite at all is an error — distinct from an
    off-grid minute, which is a real frame of an unnameable product."""
    with pytest.raises(ValueError):
        scan_type_from_filename("notes.txt")
    with pytest.raises(ValueError):
        scan_type_from_filename("dk.com.202605.500_max.h5")


# ---------------------------------------------------------------------------
# ArchiveIndex — the local listing that reaches past DMI's 180-day items API
# ---------------------------------------------------------------------------


def _plant(corpus: Path, *timestamps: datetime) -> list[Path]:
    """Drop empty composite files into the canonical YYYY/MM tree."""
    planted = []
    for ts in timestamps:
        path = archive_path_for(corpus, f"dk.com.{ts:%Y%m%d%H%M}.500_max.h5")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        planted.append(path)
    return planted


def _every_5_min(start: datetime, n: int) -> list[datetime]:
    return [start + timedelta(minutes=5 * i) for i in range(n)]


def test_archive_index_is_empty_for_a_missing_tree(tmp_path: Path) -> None:
    index = ArchiveIndex(tmp_path / "nope")
    assert len(index) == 0
    assert index.earliest() is None
    assert index.latest() is None
    assert index.list_in_window(
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2027, 1, 1, tzinfo=timezone.utc),
    ) == []


def test_archive_index_lists_window_inclusively(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    base = datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc)
    _plant(corpus, *_every_5_min(base, 7))  # 12:00 … 12:30

    index = ArchiveIndex(corpus)
    assert len(index) == 7

    # BOTH bounds inclusive — same contract as fetch.list_in_window, so the
    # two are drop-in alternatives for the corpus builder.
    got = index.list_in_window(base + timedelta(minutes=5), base + timedelta(minutes=20))
    assert [f.datetime_utc.minute for f in got] == [5, 10, 15, 20]
    # A degenerate window lands on exactly the frame at that instant.
    assert [f.datetime_utc for f in index.list_in_window(base, base)] == [base]
    # A window with nothing in it is empty, not an error.
    assert index.list_in_window(
        base + timedelta(days=1), base + timedelta(days=2)
    ) == []


def test_archive_index_filters_by_scan_type(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    base = datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc)
    _plant(corpus, *_every_5_min(base, 6))        # :00 :05 :10 :15 :20 :25
    _plant(corpus, base + timedelta(minutes=3))   # off-grid → "unknown"

    index = ArchiveIndex(corpus)
    end = base + timedelta(hours=1)
    assert [
        f.datetime_utc.minute
        for f in index.list_in_window(base, end, SCAN_TYPE_FULL_RANGE)
    ] == [0, 10, 20]
    assert [
        f.datetime_utc.minute
        for f in index.list_in_window(base, end, SCAN_TYPE_DOPPLER)
    ] == [5, 15, 25]
    # The off-grid frame is indexed but matches NO product filter.
    assert [f.datetime_utc.minute for f in index.list_in_window(base, end)] == [
        0, 3, 5, 10, 15, 20, 25,
    ]
    assert index.count() == 7
    assert index.count(SCAN_TYPE_FULL_RANGE) == 3
    assert index.count(SCAN_TYPE_DOPPLER) == 3
    assert index.count(SCAN_TYPE_UNKNOWN) == 1


def test_archive_index_orders_across_month_and_year_boundaries(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    # Planted out of order and spanning 2025/12 → 2026/02: the index must
    # sort globally, not per directory (os.scandir order is arbitrary).
    planted = [
        datetime(2026, 1, 31, 23, 50, tzinfo=timezone.utc),
        datetime(2025, 12, 31, 23, 50, tzinfo=timezone.utc),
        datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
    ]
    _plant(corpus, *planted)

    index = ArchiveIndex(corpus)
    assert [
        f.datetime_utc for f in index.list_in_window(planted[1], planted[2])
    ] == sorted(planted)
    # A window straddling the year boundary resolves against the whole
    # archive, not just one year's directory.
    straddle = index.list_in_window(
        datetime(2025, 12, 31, 23, 55, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
    )
    assert [f.datetime_utc for f in straddle] == [
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    ]


def test_archive_index_earliest_and_latest_per_scan_type(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    # The archive both starts and ends on a DOPPLER frame, so a fullRange
    # build must not read the overall span as its own (that is exactly the
    # window --days-back 0 derives).
    _plant(
        corpus,
        datetime(2025, 12, 7, 6, 5, tzinfo=timezone.utc),    # doppler, oldest
        datetime(2025, 12, 7, 6, 10, tzinfo=timezone.utc),   # fullRange, oldest
        datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc),    # fullRange, newest
        datetime(2026, 9, 1, 12, 5, tzinfo=timezone.utc),    # doppler, newest
    )
    index = ArchiveIndex(corpus)

    assert index.earliest() == datetime(2025, 12, 7, 6, 5, tzinfo=timezone.utc)
    assert index.earliest(SCAN_TYPE_FULL_RANGE) == datetime(
        2025, 12, 7, 6, 10, tzinfo=timezone.utc
    )
    assert index.earliest(SCAN_TYPE_DOPPLER) == datetime(
        2025, 12, 7, 6, 5, tzinfo=timezone.utc
    )
    assert index.latest() == datetime(2026, 9, 1, 12, 5, tzinfo=timezone.utc)
    assert index.latest(SCAN_TYPE_FULL_RANGE) == datetime(
        2026, 9, 1, 12, 0, tzinfo=timezone.utc
    )
    # A product with no frames at all reports None rather than guessing.
    assert index.earliest(SCAN_TYPE_UNKNOWN) is None
    assert index.latest(SCAN_TYPE_UNKNOWN) is None


def test_archive_index_skips_and_counts_malformed_filenames(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    base = datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc)
    _plant(corpus, base, base + timedelta(minutes=10))
    month_dir = archive_path_for(
        corpus, f"dk.com.{base:%Y%m%d%H%M}.500_max.h5"
    ).parent
    (month_dir / "notes.txt").write_text("hi")
    (month_dir / "dk.com.2026052.500_max.h5").write_bytes(b"")  # truncated stamp
    # An in-flight archiver write is NOT corruption — skipped silently.
    (month_dir / ".dk.com.202605211220.500_max.h5.abc.tmp").write_bytes(b"")

    index = ArchiveIndex(corpus)
    assert len(index) == 2
    assert index.n_malformed == 2  # notes.txt + the truncated name; not the .tmp


def test_archive_index_rejects_naive_window_bounds(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _plant(corpus, datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="timezone-aware"):
        ArchiveIndex(corpus).list_in_window(
            datetime(2026, 5, 21), datetime(2026, 5, 22)
        )


def test_archive_index_reads_only_filenames(tmp_path: Path, monkeypatch) -> None:
    """No stat, no open: every field comes off the name. An 80k-frame
    archive is indexed on every worker start, so a per-file syscall would
    be the difference between milliseconds and minutes."""
    corpus = tmp_path / "corpus"
    base = datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc)
    _plant(corpus, *_every_5_min(base, 20))

    opened: list = []
    monkeypatch.setattr(
        "builtins.open", lambda *a, **k: opened.append(a) or (_ for _ in ()).throw(
            AssertionError("ArchiveIndex must not open archived files")
        ),
    )
    index = ArchiveIndex(corpus)
    assert len(index) == 20
    assert opened == []


def test_archived_frame_exposes_the_radarfeature_attribute_names(
    tmp_path: Path,
) -> None:
    """Consumers read datetime_utc / scan_type / filename off either kind of
    listed frame; an ArchivedFrame adds ``path`` and has NO download_url, so
    code holding one cannot reach the network."""
    corpus = tmp_path / "corpus"
    ts = datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc)
    (planted,) = _plant(corpus, ts)

    (frame,) = ArchiveIndex(corpus).list_in_window(ts, ts)
    assert isinstance(frame, ArchivedFrame)
    assert frame.datetime_utc == ts
    assert frame.scan_type == SCAN_TYPE_FULL_RANGE
    assert frame.filename == planted.name
    assert frame.path == planted
    assert not hasattr(frame, "download_url")
