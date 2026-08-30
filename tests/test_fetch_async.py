"""Tests for the async DMI client: retry, backoff, rate limiting."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from dmi_nowcast_core.fetch import AsyncDMIClient, parse_feature

SAMPLE_FEATURE = {
    "type": "Feature",
    "id": "dk.com.202605171935.500_max.h5",
    "geometry": {"type": "Polygon", "coordinates": []},
    "properties": {
        "datetime": "2026-05-17T19:35:00Z",
        "scanType": "doppler",
    },
    "asset": {
        "data": {
            "href": "https://opendataapi.dmi.dk/v1/radardata/download/dk.com.202605171935.500_max.h5",
        }
    },
}


class FakeClock:
    """Deterministic monotonic clock for rate-limit tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeSleep:
    """Records sleep calls and (optionally) advances a FakeClock without blocking."""

    def __init__(self, clock: FakeClock | None = None) -> None:
        self.clock = clock
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        if self.clock is not None:
            self.clock.advance(seconds)


@pytest.mark.asyncio
async def test_list_latest_happy_path():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["sortorder"] == "datetime,DESC"
        return httpx.Response(200, json={"features": [SAMPLE_FEATURE]})

    client = AsyncDMIClient(transport=httpx.MockTransport(handler))
    async with client:
        features = await client.list_latest(limit=1)
    assert len(features) == 1
    assert features[0].scan_type == "doppler"


@pytest.mark.asyncio
async def test_list_latest_coerces_float_limit_to_int():
    """DMI rejects ``limit=8.0`` with HTTP 400. The client must coerce floats
    from HA's NumberSelector (which always returns floats, even at step=1)."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["limit"] = request.url.params["limit"]
        return httpx.Response(200, json={"features": []})

    client = AsyncDMIClient(transport=httpx.MockTransport(handler))
    async with client:
        await client.list_latest(limit=8.0)  # type: ignore[arg-type]
    assert captured["limit"] == "8", f"expected integer-shaped string, got {captured['limit']!r}"


@pytest.mark.asyncio
async def test_retries_on_429_then_succeeds():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={})
        return httpx.Response(200, json={"features": [SAMPLE_FEATURE]})

    sleep = FakeSleep()
    client = AsyncDMIClient(
        transport=httpx.MockTransport(handler),
        sleep=sleep,
    )
    async with client:
        features = await client.list_latest()
    assert attempts["n"] == 3
    assert features[0].scan_type == "doppler"
    # Two sleeps for two 429 retries; Retry-After=0 honored.
    assert sleep.calls == [0.0, 0.0]


@pytest.mark.asyncio
async def test_retries_on_5xx_with_exponential_backoff():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503, json={})
        return httpx.Response(200, json={"features": []})

    sleep = FakeSleep()
    client = AsyncDMIClient(transport=httpx.MockTransport(handler), sleep=sleep)
    async with client:
        await client.list_latest()
    assert attempts["n"] == 3
    # Exponential backoff: first ≥ 1s, second ≥ 2s (with jitter).
    assert sleep.calls[0] >= 1.0 and sleep.calls[0] < 2.0
    assert sleep.calls[1] >= 2.0 and sleep.calls[1] < 3.0


@pytest.mark.asyncio
async def test_4xx_other_than_429_does_not_retry():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(404, json={})

    client = AsyncDMIClient(transport=httpx.MockTransport(handler), sleep=FakeSleep())
    async with client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.list_latest()
    assert attempts["n"] == 1, "404 should fail fast, not retry"


@pytest.mark.asyncio
async def test_rate_limiter_sleeps_when_limit_exceeded():
    """With a tiny 3-req / 0.5 s window, the 4th request must wait."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"features": []})

    clock = FakeClock()
    sleep = FakeSleep(clock=clock)
    client = AsyncDMIClient(
        transport=httpx.MockTransport(handler),
        rate_limit_requests=3,
        rate_limit_window_s=0.5,
        sleep=sleep,
        clock=clock,
    )
    async with client:
        for _ in range(4):
            await client.list_latest()
    # 4th request had to wait — the limiter should have called sleep at least once.
    assert any(call > 0 for call in sleep.calls), (
        f"expected a positive sleep when exceeding rate limit, got {sleep.calls}"
    )


@pytest.mark.asyncio
async def test_download_streams_and_renames_atomically(tmp_path: Path):
    body = b"\x89HDF\r\n\x1a\n" + b"x" * 10_000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    feature = parse_feature(SAMPLE_FEATURE)
    client = AsyncDMIClient(transport=httpx.MockTransport(handler), sleep=FakeSleep())
    async with client:
        path = await client.download(feature, tmp_path)
    assert path.read_bytes() == body
    assert not path.with_suffix(path.suffix + ".tmp").exists()


@pytest.mark.asyncio
async def test_download_is_idempotent(tmp_path: Path):
    feature = parse_feature(SAMPLE_FEATURE)
    cached = tmp_path / feature.filename
    cached.write_bytes(b"existing")

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("network must not be hit for cached file")

    client = AsyncDMIClient(transport=httpx.MockTransport(handler), sleep=FakeSleep())
    async with client:
        path = await client.download(feature, tmp_path)
    assert path.read_bytes() == b"existing"
