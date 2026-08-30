"""Bounded disk cache for DMI radar files.

LRU by atime. Filenames are immutable (DMI URLs map 1:1 to filenames), so
caching is straightforward: presence + size check, no hashing.

Plan §12.4 sizing:
- HA integration: ~50 MB (one hour of frames).
- Backtest harness: 10–30 GB for 90 days. Reality (recon, plan §0 finding):
  files are 70–90 KB, not 0.5–2 MB, so 90 days × 288 frames × ~80 KB ≈ 2 GB.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_MAX_BYTES = 30 * 1024**3  # 30 GB — way over the realistic 2 GB headroom


@dataclass(frozen=True)
class CacheConfig:
    root: Path
    max_bytes: int = DEFAULT_MAX_BYTES


class DiskCache:
    """A flat directory cache with LRU eviction."""

    def __init__(self, config: CacheConfig) -> None:
        self.root = Path(config.root)
        self.max_bytes = config.max_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, filename: str) -> Path:
        return self.root / filename

    def has(self, filename: str) -> bool:
        return self.path(filename).is_file()

    def touch(self, filename: str) -> None:
        """Update atime to mark this file recently used."""
        try:
            self.path(filename).touch(exist_ok=True)
        except FileNotFoundError:
            pass

    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self._files())

    def evict(self) -> tuple[int, int]:
        """Evict LRU files until total ≤ max_bytes. Returns (files_evicted, bytes_evicted)."""
        files = sorted(self._files(), key=lambda p: p.stat().st_atime)
        total = sum(p.stat().st_size for p in files)
        files_evicted = 0
        bytes_evicted = 0
        for p in files:
            if total <= self.max_bytes:
                break
            try:
                size = p.stat().st_size
                p.unlink()
                total -= size
                files_evicted += 1
                bytes_evicted += size
            except OSError:
                continue
        return files_evicted, bytes_evicted

    def _files(self) -> list[Path]:
        return [p for p in self.root.iterdir() if p.is_file() and not p.name.endswith(".tmp")]
