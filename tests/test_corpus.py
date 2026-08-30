"""Tests for the persistent corpus archiver."""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dmi_nowcast_core.corpus import (
    ArchiveResult,
    CorpusArchiver,
    archive_path_for,
    parse_filename_ts,
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
