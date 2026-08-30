"""Tests for the disk cache."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from dmi_nowcast_core.cache import CacheConfig, DiskCache


def _write(path: Path, size: int) -> Path:
    path.write_bytes(b"\x00" * size)
    return path


@pytest.fixture
def cache(tmp_path: Path) -> DiskCache:
    return DiskCache(CacheConfig(root=tmp_path, max_bytes=1024))


def test_path_and_has(cache):
    assert not cache.has("dk.com.202605171935.500_max.h5")
    p = cache.path("dk.com.202605171935.500_max.h5")
    p.write_bytes(b"hello")
    assert cache.has("dk.com.202605171935.500_max.h5")


def test_total_bytes_sums_files(cache):
    _write(cache.path("a.h5"), 100)
    _write(cache.path("b.h5"), 200)
    assert cache.total_bytes() == 300


def test_total_bytes_ignores_tmp_files(cache):
    """In-flight downloads (.tmp) should not count toward cache size or be evicted."""
    _write(cache.path("real.h5"), 100)
    _write(cache.path("download.h5.tmp"), 999)
    assert cache.total_bytes() == 100


def test_evict_removes_lru_until_under_limit(tmp_path: Path):
    cache = DiskCache(CacheConfig(root=tmp_path, max_bytes=250))
    a = _write(cache.path("a.h5"), 100)
    b = _write(cache.path("b.h5"), 100)
    c = _write(cache.path("c.h5"), 100)
    # Make a oldest, c newest.
    now = time.time()
    import os
    os.utime(a, (now - 300, now - 300))
    os.utime(b, (now - 200, now - 200))
    os.utime(c, (now - 100, now - 100))

    files_evicted, bytes_evicted = cache.evict()
    assert files_evicted == 1
    assert bytes_evicted == 100
    assert not a.exists()
    assert b.exists()
    assert c.exists()


def test_evict_is_noop_when_under_limit(tmp_path: Path):
    cache = DiskCache(CacheConfig(root=tmp_path, max_bytes=10_000))
    _write(cache.path("a.h5"), 100)
    files, bytes_ = cache.evict()
    assert files == 0 and bytes_ == 0


def test_touch_updates_atime(cache, tmp_path: Path):
    p = _write(cache.path("a.h5"), 50)
    import os
    old_atime = time.time() - 1000
    os.utime(p, (old_atime, old_atime))
    cache.touch("a.h5")
    new_atime = p.stat().st_atime
    assert new_atime > old_atime
