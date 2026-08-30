"""Tests for the bulk fetch utility."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from dmi_nowcast_core.bulk_fetch import _to_z, bulk_fetch, list_window
from dmi_nowcast_core.cache import CacheConfig, DiskCache
from dmi_nowcast_core.fetch import AsyncDMIClient


def _make_feature(minute: int) -> dict:
    fname = f"dk.com.202605171{9:01d}{minute:02d}.500_max.h5"
    return {
        "id": fname,
        "properties": {
            "datetime": f"2026-05-17T19:{minute:02d}:00Z",
            "scanType": "fullRange" if minute % 10 == 0 else "doppler",
        },
        "asset": {
            "data": {
                "href": f"https://opendataapi.dmi.dk/v1/radardata/download/{fname}",
            }
        },
    }


def test_to_z_formats_naive_as_utc():
    naive = datetime(2026, 5, 17, 19, 35)
    assert _to_z(naive) == "2026-05-17T19:35:00Z"


def test_to_z_converts_local_to_utc():
    from datetime import timedelta
    local = datetime(2026, 5, 17, 21, 35, tzinfo=timezone(timedelta(hours=2)))
    assert _to_z(local) == "2026-05-17T19:35:00Z"


@pytest.mark.asyncio
async def test_list_window_sends_datetime_filter():
    received: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received["params"] = dict(request.url.params)
        return httpx.Response(200, json={"features": [_make_feature(35)]})

    client = AsyncDMIClient(transport=httpx.MockTransport(handler))
    async with client:
        features = await list_window(
            client,
            datetime(2026, 5, 17, 19, 0, tzinfo=timezone.utc),
            datetime(2026, 5, 17, 20, 0, tzinfo=timezone.utc),
        )
    assert received["params"]["datetime"] == "2026-05-17T19:00:00Z/2026-05-17T20:00:00Z"
    assert received["params"]["sortorder"] == "datetime,DESC"
    assert len(features) == 1


@pytest.mark.asyncio
async def test_bulk_fetch_downloads_only_missing_files(tmp_path: Path):
    features = [_make_feature(30), _make_feature(35), _make_feature(40)]

    # Pre-populate the cache with one of the three files.
    cache = DiskCache(CacheConfig(root=tmp_path, max_bytes=10**9))
    cache.path(features[1]["id"]).write_bytes(b"cached")

    downloaded: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "/items" in path:
            return httpx.Response(200, json={"features": features})
        # Download requests:
        filename = path.rsplit("/", 1)[-1]
        downloaded.append(filename)
        return httpx.Response(200, content=b"fresh")

    client = AsyncDMIClient(transport=httpx.MockTransport(handler))
    async with client:
        paths = await bulk_fetch(
            datetime(2026, 5, 17, 19, 30, tzinfo=timezone.utc),
            datetime(2026, 5, 17, 19, 40, tzinfo=timezone.utc),
            cache,
            client=client,
            concurrency=2,
        )

    assert len(paths) == 3
    # Cached file must not have been re-downloaded.
    assert features[1]["id"] not in downloaded
    assert set(downloaded) == {features[0]["id"], features[2]["id"]}
    # Cached file's contents are still the original.
    assert cache.path(features[1]["id"]).read_bytes() == b"cached"
