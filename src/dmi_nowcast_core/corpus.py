"""Persistent corpus archive for radar HDF5 composites.

The persistent corpus archive:
the long-term archive lives in a directory separate from the sidecar's
working cache so that benchmarks have a stable source of truth even when
the working cache is wiped by ``docker compose down -v``.

Layout::

    <corpus_dir>/
      composites/
        YYYY/MM/dk.com.YYYYMMDDhhmm.500_max.h5
      manifest.parquet        (built by scripts/build_corpus_manifest.py)

The archiver is intentionally minimal: filenames are immutable (DMI URLs
map 1:1 to filenames) and the embedded ``YYYYMMDDhhmm`` timestamp uniquely
identifies the frame, so we use that to bucket into the year/month tree.
Writes are atomic (tmp + rename) and idempotent (skip when destination
exists with the same size).

:class:`ArchiveIndex` reads that same tree back as a *listing*: DMI's
items API only lists the last 180 days, but the archive is unbounded, so
anything that needs to enumerate frames over a long window (the
calibration corpus builder, backtests) asks the archive first and only
falls back to the API for windows the archive does not cover.
"""
from __future__ import annotations

import bisect
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional


_LOGGER = logging.getLogger(__name__)

# DMI composite filename pattern: dk.com.YYYYMMDDhhmm.500_max.h5
# Anchored to avoid matching garbage files that happen to share the prefix.
_FILENAME_RE = re.compile(r"^dk\.com\.(\d{12})\.500_max\.h5$")

#: The two products DMI interleaves in the ``composite`` collection.
SCAN_TYPE_FULL_RANGE = "fullRange"
SCAN_TYPE_DOPPLER = "doppler"
#: Anything whose filename minute is not on either product's grid. It is a
#: deliberate third value rather than an exception: an odd-minute frame in
#: the tree must be *skippable*, and it must NEVER be mistaken for
#: fullRange (mixing the products alternates between materially different
#: views — doppler covers ~40% of fullRange's area).
SCAN_TYPE_UNKNOWN = "unknown"

#: Minute-of-hour ``% 10`` → product. Composite filenames carry no
#: scan-type marker; the minute IS the marker. Verified against DMI's own
#: ``scanType``-labelled listing for 12 consecutive frames on 2026-09-02:
#: every ``:x0`` frame was fullRange and every ``:x5`` frame doppler.
_SCAN_TYPE_BY_MINUTE_MOD_10 = {0: SCAN_TYPE_FULL_RANGE, 5: SCAN_TYPE_DOPPLER}


def parse_filename_ts(filename: str) -> datetime:
    """Parse the UTC timestamp embedded in a DMI composite filename.

    Raises ``ValueError`` if the filename does not match the expected pattern.
    """
    m = _FILENAME_RE.match(filename)
    if not m:
        raise ValueError(f"not a DMI composite filename: {filename!r}")
    s = m.group(1)
    return datetime(
        int(s[0:4]), int(s[4:6]), int(s[6:8]),
        int(s[8:10]), int(s[10:12]),
        tzinfo=timezone.utc,
    )


def scan_type_from_filename(filename: str) -> str:
    """Which DMI product a composite filename belongs to.

    Returns :data:`SCAN_TYPE_FULL_RANGE` for a ``:x0`` minute,
    :data:`SCAN_TYPE_DOPPLER` for ``:x5``, and :data:`SCAN_TYPE_UNKNOWN`
    for any other minute. Raises ``ValueError`` when the name is not a DMI
    composite filename at all (delegated to :func:`parse_filename_ts`) —
    the two failure modes are different: a non-composite file does not
    belong in the tree, while an off-grid minute is a real composite whose
    product we cannot name.

    ``unknown`` must never be treated as fullRange by a caller: filtering
    is therefore always an equality test against a concrete product, never
    a "not doppler" test.
    """
    return _SCAN_TYPE_BY_MINUTE_MOD_10.get(
        parse_filename_ts(filename).minute % 10, SCAN_TYPE_UNKNOWN
    )


def composites_root(corpus_dir: Path) -> Path:
    return Path(corpus_dir) / "composites"


def archive_path_for(corpus_dir: Path, filename: str) -> Path:
    """Return the canonical archive path for a composite filename."""
    ts = parse_filename_ts(filename)
    return composites_root(corpus_dir) / f"{ts.year:04d}" / f"{ts.month:02d}" / filename


@dataclass(frozen=True)
class ArchiveResult:
    path: Path
    archived: bool   # True if a copy was made, False if already present


class CorpusArchiver:
    """Persistent archive of DMI radar HDF5 frames.

    Designed to be called from the sidecar cycle (per-frame) and from
    bulk-backfill scripts. Either way, the contract is the same: hand it
    a downloaded HDF5 path and it places a copy in the canonical
    ``YYYY/MM`` slot, idempotently.

    Not async — file I/O is brief (~80 KB / frame). Sidecar runtime calls
    via ``asyncio.to_thread``.
    """

    def __init__(self, corpus_dir: Path) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.composites_dir = composites_root(self.corpus_dir)
        self.composites_dir.mkdir(parents=True, exist_ok=True)

    def archive(self, src: Path) -> ArchiveResult:
        """Copy ``src`` into the canonical corpus slot. Idempotent.

        Returns ``ArchiveResult(path, archived)`` where ``archived`` is
        True iff a copy was actually made (False when the destination
        already exists with the same size).
        """
        src = Path(src)
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = archive_path_for(self.corpus_dir, src.name)
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            return ArchiveResult(path=dst, archived=False)

        dst.parent.mkdir(parents=True, exist_ok=True)
        # Atomic: write into the destination dir, then rename.
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{src.name}.", suffix=".tmp", dir=str(dst.parent),
        )
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            shutil.copyfile(src, tmp)
            # Preserve mtime so archive ordering reflects acquisition time,
            # not the moment we copied. Useful for debugging and for any
            # downstream consumer that sorts by mtime.
            st = src.stat()
            os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
            os.replace(tmp, dst)
        except Exception:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            raise
        return ArchiveResult(path=dst, archived=True)

    def iter_files(self):
        """Yield (timestamp, path) for every archived composite."""
        if not self.composites_dir.is_dir():
            return
        for p in sorted(self.composites_dir.rglob("dk.com.*.500_max.h5")):
            try:
                yield parse_filename_ts(p.name), p
            except ValueError:
                continue

    def count(self) -> int:
        return sum(1 for _ in self.iter_files())

    def total_bytes(self) -> int:
        return sum(p.stat().st_size for _, p in self.iter_files())


@dataclass(frozen=True)
class ArchivedFrame:
    """One composite already on disk in the corpus archive.

    Deliberately a distinct type from :class:`dmi_nowcast_core.fetch.
    RadarFeature` and NOT a subclass of it: an archived frame has a
    ``path`` and no ``download_url``, so code that receives one cannot
    accidentally reach the network. The attribute names shared with
    ``RadarFeature`` (``datetime_utc``, ``scan_type``, ``filename``) are
    exactly the ones consumers read, so the two are interchangeable
    everywhere that only *lists* frames.
    """

    datetime_utc: datetime
    scan_type: str
    path: Path

    @property
    def filename(self) -> str:
        return self.path.name


class ArchiveIndex:
    """In-memory listing of the corpus archive: ``(time, scan type, path)``.

    Built once from ``<corpus_dir>/composites/YYYY/MM/`` with
    :func:`os.scandir` and NOTHING else — no ``stat``, no file open, no
    HDF5 parse. Every field comes out of the filename, which is why the
    scan type has to be read off the minute (:func:`scan_type_from_filename`).
    ~80k frames index in well under a second beyond the directory reads.

    This is the long-window substitute for DMI's items API: the API lists
    the last 180 days, the archive grows without bound, so a consumer that
    needs frames from further back reads them here.

    Entries whose name is not a DMI composite filename are skipped and
    counted in :attr:`n_malformed` (logged once at build time). The
    archiver's own in-flight temp files (``.<name>.<rand>.tmp``) and other
    dotfiles are skipped silently — they are not corruption, just a write
    in progress.
    """

    def __init__(self, corpus_dir: Path) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.n_malformed = 0
        frames = sorted(self._scan(), key=lambda f: (f.datetime_utc, f.path.name))
        self._frames: list[ArchivedFrame] = frames
        # Parallel key list so window lookups are a pair of bisects rather
        # than a scan of the whole archive.
        self._times: list[datetime] = [f.datetime_utc for f in frames]
        if self.n_malformed:
            _LOGGER.warning(
                "archive index: skipped %d unparseable filename(s) under %s",
                self.n_malformed, composites_root(self.corpus_dir),
            )

    # -- construction -------------------------------------------------

    def _scan(self) -> Iterator[ArchivedFrame]:
        root = composites_root(self.corpus_dir)
        try:
            with os.scandir(root) as it:
                year_dirs = sorted(e.path for e in it if e.is_dir())
        except FileNotFoundError:
            return
        for year_dir in year_dirs:
            try:
                with os.scandir(year_dir) as it:
                    month_dirs = sorted(e.path for e in it if e.is_dir())
            except FileNotFoundError:  # pragma: no cover - racing rmdir
                continue
            for month_dir in month_dirs:
                try:
                    entries = os.scandir(month_dir)
                except FileNotFoundError:  # pragma: no cover - racing rmdir
                    continue
                with entries:
                    for entry in entries:
                        name = entry.name
                        # In-flight archiver writes are not corruption.
                        if name.startswith(".") or name.endswith(".tmp"):
                            continue
                        try:
                            ts = parse_filename_ts(name)
                        except ValueError:
                            self.n_malformed += 1
                            continue
                        # The minute lookup is inlined rather than going
                        # through scan_type_from_filename() so the name is
                        # regex-matched once, not twice, per file.
                        yield ArchivedFrame(
                            datetime_utc=ts,
                            scan_type=_SCAN_TYPE_BY_MINUTE_MOD_10.get(
                                ts.minute % 10, SCAN_TYPE_UNKNOWN
                            ),
                            path=Path(entry.path),
                        )

    # -- queries ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._frames)

    def count(self, scan_type: Optional[str] = None) -> int:
        """Number of indexed frames, optionally of one product only."""
        if not scan_type:
            return len(self._frames)
        return sum(1 for f in self._frames if f.scan_type == scan_type)

    def list_in_window(
        self,
        start: datetime,
        end: datetime,
        scan_type: Optional[str] = None,
    ) -> list[ArchivedFrame]:
        """Archived frames with ``start <= t <= end``, oldest first.

        Both bounds inclusive — same contract as
        :func:`dmi_nowcast_core.fetch.list_in_window`, so the two are
        drop-in alternatives. ``scan_type`` filters by equality, so
        :data:`SCAN_TYPE_UNKNOWN` frames are returned only when no filter
        is asked for. Naive datetimes are rejected rather than silently
        compared against UTC.
        """
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start/end must be timezone-aware UTC datetimes")
        lo = bisect.bisect_left(self._times, start)
        hi = bisect.bisect_right(self._times, end)
        frames = self._frames[lo:hi]
        if scan_type:
            frames = [f for f in frames if f.scan_type == scan_type]
        return frames

    def earliest(self, scan_type: Optional[str] = None) -> Optional[datetime]:
        """Timestamp of the oldest archived frame (of ``scan_type``), or None."""
        for frame in self._frames:
            if not scan_type or frame.scan_type == scan_type:
                return frame.datetime_utc
        return None

    def latest(self, scan_type: Optional[str] = None) -> Optional[datetime]:
        """Timestamp of the newest archived frame (of ``scan_type``), or None."""
        for frame in reversed(self._frames):
            if not scan_type or frame.scan_type == scan_type:
                return frame.datetime_utc
        return None
