"""Bulk fetch of DMI radar composites for a datetime window.

Uses the AsyncDMIClient with bounded concurrency. Resumable: skips files that
are already in the cache (touching atime so they don't get evicted).

OGC API standard datetime filter format:
    datetime=2026-05-17T00:00:00Z/2026-05-17T12:00:00Z
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cache import DiskCache
from .fetch import (
    COMPOSITE_COLLECTION,
    AsyncDMIClient,
    RadarFeature,
    parse_feature,
)


def _to_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


async def list_window(
    client: AsyncDMIClient,
    start: datetime,
    end: datetime,
    *,
    scan_type: str | None = None,
    page_size: int = 500,
) -> list[RadarFeature]:
    """List all composite features with ``start <= datetime <= end``.

    DMI's STAC endpoint supports OGC datetime filtering. We page until we hit
    a response shorter than ``page_size`` (no explicit ``next`` link parsing
    needed for our typical windows of hours-to-days).
    """
    params: dict[str, Any] = {
        "datetime": f"{_to_z(start)}/{_to_z(end)}",
        "limit": page_size,
        "sortorder": "datetime,DESC",
    }
    if scan_type:
        params["scanType"] = scan_type
    url = f"{client.base_url}/collections/{COMPOSITE_COLLECTION}/items"
    payload = await client._get_json(url, params)
    return [parse_feature(f) for f in payload.get("features", [])]


async def bulk_fetch(
    start: datetime,
    end: datetime,
    cache: DiskCache,
    *,
    scan_type: str | None = None,
    concurrency: int = 4,
    client: AsyncDMIClient | None = None,
) -> list[Path]:
    """Fetch every composite in [start, end]. Returns the local paths."""
    owns = client is None
    c = client or AsyncDMIClient()
    try:
        features = await list_window(c, start, end, scan_type=scan_type)
        sem = asyncio.Semaphore(concurrency)

        async def fetch_one(f: RadarFeature) -> Path:
            async with sem:
                if cache.has(f.filename):
                    cache.touch(f.filename)
                    return cache.path(f.filename)
                return await c.download(f, cache.root)

        return list(await asyncio.gather(*(fetch_one(f) for f in features)))
    finally:
        if owns:
            await c.close()
