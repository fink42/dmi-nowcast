"""Append-only NDJSON archive of received lightning strikes.

Persists every (deduplicated) strike the sidecar receives so we can later build
a ground-truth dataset for backtesting / calibrating the lightning ETA — the
same self-archive strategy used for radar. (Blitzortung's own raw archive is
participant-gated and non-redistributable; the live stream we already receive
is ours to keep for personal use.)

One NDJSON file per UTC day on the durable corpus bind-mount, e.g.
``strikes_2026-06-07.ndjson`` with lines ``{"lat":..,"lon":..,"t":".."}``.
Append-only and line-buffered so it's crash-safe; dedup is handled upstream by
the LightningTracker, and any rare duplicate lines are cheap to drop on read.
"""
from __future__ import annotations

import json
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator

from dmi_nowcast_core.lightning import LightningStrike
from dmi_nowcast_core.regions import REGIONS as _REGIONS
from dmi_nowcast_core.regions import region_of as _region_of


class StrikeArchive:
    def __init__(self, archive_dir: Path) -> None:
        self._dir = Path(archive_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # TTL cache for the (summary, heat_points) snapshot — a full re-read of
        # the archive at most once per ttl, however often the dashboard is hit.
        self._snap_cache: tuple[dict, list] | None = None
        self._snap_ts: float = 0.0

    def append(self, strikes: Iterable[LightningStrike]) -> int:
        """Append strikes to per-UTC-day NDJSON files. Returns lines written."""
        by_day: dict[str, list[tuple[LightningStrike, object]]] = defaultdict(list)
        for s in strikes:
            ts = s.t.astimezone(timezone.utc)
            by_day[ts.date().isoformat()].append((s, ts))
        if not by_day:
            return 0
        written = 0
        with self._lock:
            for day, items in by_day.items():
                path = self._dir / f"strikes_{day}.ndjson"
                with path.open("a") as f:
                    for s, ts in items:
                        f.write(json.dumps(
                            {"lat": s.lat, "lon": s.lon, "t": ts.isoformat()}
                        ) + "\n")
                        written += 1
        return written

    # --- read side (collection monitoring) ------------------------------- #
    def iter_strikes(self) -> Iterator[tuple[float, float, str]]:
        """Yield (lat, lon, iso_t) over every archived strike, oldest day first,
        **deduplicated** on (lat, lon, t).

        Dedup matters: the HA push re-posts every still-live ``geo_location``
        strike each cycle and the upstream tracker dedups only in memory, so a
        sidecar restart re-appends a batch of identical lines. Dropping them here
        keeps the summary counts and the calibration corpus honest."""
        seen: set[tuple[float, float, str]] = set()
        for f in sorted(self._dir.glob("strikes_*.ndjson")):
            try:
                with f.open() as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                            key = (float(d["lat"]), float(d["lon"]), str(d["t"]))
                        except (ValueError, KeyError):
                            continue
                        if key in seen:
                            continue
                        seen.add(key)
                        yield key
            except OSError:
                continue

    def snapshot(
        self, now: datetime | None = None, ttl_s: float = 60.0,
        max_points: int = 20000,
    ) -> tuple[dict, list]:
        """Aggregate the archive into ``(summary_dict, heat_points)``.

        ``summary_dict`` is small (counts, per-day, per-region, bbox, last
        strike); ``heat_points`` is ``[[lat, lon, weight], …]`` for the map —
        binned to a coarse grid once it exceeds ``max_points`` so the page
        stays light. TTL-cached so repeated dashboard hits re-read at most once
        per ``ttl_s``."""
        now = now or datetime.now(timezone.utc)
        with self._lock:
            if self._snap_cache is not None and (time.monotonic() - self._snap_ts) < ttl_s:
                return self._snap_cache

        total = 0
        per_day: dict[str, int] = defaultdict(int)
        per_region: dict[str, int] = {**{k: 0 for k in _REGIONS}, "Other": 0}
        latmin = latmax = lonmin = lonmax = None
        last_t: str | None = None
        coords: list[tuple[float, float]] = []
        for lat, lon, t in self.iter_strikes():
            total += 1
            per_day[t[:10]] += 1
            per_region[_region_of(lat, lon)] += 1
            latmin = lat if latmin is None else min(latmin, lat)
            latmax = lat if latmax is None else max(latmax, lat)
            lonmin = lon if lonmin is None else min(lonmin, lon)
            lonmax = lon if lonmax is None else max(lonmax, lon)
            if last_t is None or t > last_t:
                last_t = t
            coords.append((round(lat, 4), round(lon, 4)))

        if len(coords) > max_points:
            binned = Counter((round(la, 2), round(lo, 2)) for la, lo in coords)
            points = [[la, lo, n] for (la, lo), n in binned.items()]
        else:
            points = [[la, lo, 1] for la, lo in coords]

        today = now.date().isoformat()
        last7 = {(now - timedelta(days=i)).date().isoformat() for i in range(7)}
        summary = {
            "total": total,
            "today": per_day.get(today, 0),
            "last_7d": sum(n for d, n in per_day.items() if d in last7),
            "days": len(per_day),
            "first_day": min(per_day) if per_day else None,
            "last_day": max(per_day) if per_day else None,
            "last_strike_utc": last_t,
            "per_region": per_region,
            "per_day": dict(sorted(per_day.items())),
            "bbox": [latmin, lonmin, latmax, lonmax] if total else None,
            "n_points": len(points),
            "generated_at": now.isoformat(),
        }
        with self._lock:
            self._snap_cache = (summary, points)
            self._snap_ts = time.monotonic()
        return summary, points


__all__ = ["StrikeArchive"]
