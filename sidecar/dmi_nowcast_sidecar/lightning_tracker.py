"""Rolling in-memory buffer of recent lightning strikes.

POSTed strikes from HA accumulate here; the ``/lightning/eta`` handler reads
a snapshot and hands it to ``dmi_nowcast_core.lightning.compute_eta``. Guarded
by a ``threading.Lock`` so the (light) work can be offloaded with
``asyncio.to_thread`` without racing concurrent POSTs — matching the sidecar's
async discipline (no blocking work on the event loop).

The buffer is purely a bounded, deduplicated store; time-window and relevance
filtering happen in ``compute_eta`` itself, so stale entries here are harmless
(they're capped by ``max_buffer`` and pruned best-effort).
"""
from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Iterable

from dmi_nowcast_core.lightning import LightningStrike

from .config import LightningConfig
from .strike_archive import StrikeArchive


def _key(s: LightningStrike) -> tuple[float, float, int]:
    return (round(s.lat, 4), round(s.lon, 4), int(s.t.timestamp()))


class LightningTracker:
    def __init__(
        self, config: LightningConfig, archive: StrikeArchive | None = None
    ) -> None:
        self._config = config
        self._archive = archive
        self._lock = threading.Lock()
        self._strikes: deque[LightningStrike] = deque(maxlen=config.max_buffer)
        self._seen: set[tuple[float, float, int]] = set()

    def add(self, strikes: Iterable[LightningStrike]) -> int:
        """Append new strikes, deduped by rounded lat/lon + whole-second time.
        Newly-added strikes are also persisted to the archive. Returns how many
        were newly added."""
        new: list[LightningStrike] = []
        with self._lock:
            for s in strikes:
                k = _key(s)
                if k in self._seen:
                    continue
                self._seen.add(k)
                self._strikes.append(s)
                new.append(s)
            self._prune_locked()
        # Persist outside the buffer lock — file I/O has its own lock.
        if new and self._archive is not None:
            self._archive.append(new)
        return len(new)

    def snapshot(self) -> list[LightningStrike]:
        with self._lock:
            self._prune_locked()
            return list(self._strikes)

    def size(self) -> int:
        with self._lock:
            return len(self._strikes)

    def _prune_locked(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(
            minutes=self._config.buffer_window_min
        )
        if any(s.t < cutoff for s in self._strikes):
            kept = [s for s in self._strikes if s.t >= cutoff]
            self._strikes = deque(kept, maxlen=self._config.max_buffer)
        # Keep the dedup set from growing unbounded across long storms.
        if len(self._seen) > 4 * self._config.max_buffer:
            self._seen = {_key(s) for s in self._strikes}


__all__ = ["LightningTracker"]
