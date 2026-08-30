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
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


# DMI composite filename pattern: dk.com.YYYYMMDDhhmm.500_max.h5
# Anchored to avoid matching garbage files that happen to share the prefix.
_FILENAME_RE = re.compile(r"^dk\.com\.(\d{12})\.500_max\.h5$")


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
