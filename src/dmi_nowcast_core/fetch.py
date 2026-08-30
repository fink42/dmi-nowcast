"""DMI Open Data radar composite API client.

The new host https://opendataapi.dmi.dk/v1/radardata/ requires no authentication.
Sync functions are kept for scripts/backtest; ``AsyncDMIClient`` is what the
Home Assistant coordinator uses (plan §6.1).

Note: plan §6.1 suggests ``aiohttp``; we use ``httpx`` for both sync and async so
the codebase stays single-library. Functionally equivalent for our needs.
"""
from __future__ import annotations

import asyncio
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://opendataapi.dmi.dk/v1/radardata"
COMPOSITE_COLLECTION = "composite"


@dataclass(frozen=True)
class RadarFeature:
    feature_id: str
    datetime_utc: datetime
    scan_type: str
    download_url: str
    filename: str


def list_latest(
    *,
    limit: int = 12,
    scan_type: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    client: httpx.Client | None = None,
) -> list[RadarFeature]:
    """Return the most recent composite features, newest first."""
    # DMI rejects float `limit` values (HTTP 400). Coerce defensively in case
    # the caller forwarded an int-shaped float (e.g. from HA's NumberSelector).
    params: dict[str, Any] = {"limit": int(limit), "sortorder": "datetime,DESC"}
    if scan_type:
        params["scanType"] = scan_type
    url = f"{base_url.rstrip('/')}/collections/{COMPOSITE_COLLECTION}/items"
    owns = client is None
    c = client or httpx.Client(timeout=30.0)
    try:
        resp = c.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()
    finally:
        if owns:
            c.close()
    return [parse_feature(f) for f in payload.get("features", [])]


def list_in_window(
    start: datetime,
    end: datetime,
    *,
    limit: int = 300,
    scan_type: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    client: httpx.Client | None = None,
) -> list[RadarFeature]:
    """Return composites whose ``datetime`` falls in ``[start, end]``.

    Used by the calibration corpus builder to fetch historical events
    from the 180-day archive. ``limit`` caps a single page; the API
    doesn't paginate transparently for windowed queries (``numberMatched``
    isn't reported by the DMI endpoint), so callers slicing a long range
    should call this once per ~hour-long window and concatenate.

    Both bounds are inclusive on the server side. ``start`` and ``end``
    must be timezone-aware (UTC); naive datetimes are rejected so we
    never silently mis-interpret a local-time argument.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start/end must be timezone-aware UTC datetimes")
    start_utc = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    dt_range = (
        f"{start_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}/"
        f"{end_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )
    params: dict[str, Any] = {
        "datetime": dt_range,
        "limit": int(limit),
        # DMI's API only supports ``sortorder=datetime,DESC`` — ascending
        # isn't accepted (HTTP 400). The corpus builder sorts client-side
        # if it needs ascending order.
        "sortorder": "datetime,DESC",
    }
    if scan_type:
        params["scanType"] = scan_type
    url = f"{base_url.rstrip('/')}/collections/{COMPOSITE_COLLECTION}/items"
    owns = client is None
    c = client or httpx.Client(timeout=30.0)
    try:
        resp = c.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()
    finally:
        if owns:
            c.close()
    feats = [parse_feature(f) for f in payload.get("features", [])]
    # Return ascending (oldest first) for ergonomics — the API returns
    # descending due to the sortorder constraint.
    feats.sort(key=lambda f: f.datetime_utc)
    return feats


def download(
    feature: RadarFeature,
    dest_dir: Path,
    *,
    client: httpx.Client | None = None,
) -> Path:
    """Download feature's HDF5 to dest_dir. Idempotent: skips if file already exists."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / feature.filename
    if dest.exists():
        return dest
    owns = client is None
    c = client or httpx.Client(timeout=120.0)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with c.stream("GET", feature.download_url) as response:
            response.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in response.iter_bytes(chunk_size=65536):
                    fh.write(chunk)
        tmp.rename(dest)
    finally:
        if owns:
            c.close()
    return dest


def parse_feature(feature: dict) -> RadarFeature:
    """Convert a STAC feature dict to a RadarFeature.

    Note: DMI uses ``asset`` (singular) where the STAC spec uses ``assets`` (plural).
    Accept both so we are not surprised if DMI conforms later.
    """
    props = feature.get("properties", {})
    assets = feature.get("asset") or feature.get("assets") or {}
    data_asset = assets.get("data", {})
    href = data_asset.get("href", "")
    feature_id = feature.get("id", "")
    filename = href.rsplit("/", 1)[-1] if href else feature_id
    return RadarFeature(
        feature_id=feature_id,
        datetime_utc=_parse_datetime(props.get("datetime", "")),
        scan_type=props.get("scanType", ""),
        download_url=href,
        filename=filename,
    )


def _parse_datetime(s: str) -> datetime:
    if not s:
        return datetime.min.replace(tzinfo=timezone.utc)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class AsyncDMIClient:
    """Async client with retry, exponential backoff, and a sliding-window rate limiter.

    Plan §3.1 caps usage at 500 requests per 5 seconds. We track timestamps of
    recent requests and sleep if a new one would exceed the cap. Retries are
    exponential with jitter; ``Retry-After`` is honored when present.
    """

    RATE_LIMIT_REQUESTS = 500
    RATE_LIMIT_WINDOW_S = 5.0

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        max_retries: int = 5,
        max_backoff_s: float = 60.0,
        timeout_s: float = 30.0,
        rate_limit_requests: int | None = None,
        rate_limit_window_s: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Any = asyncio.sleep,
        clock: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.max_backoff_s = max_backoff_s
        # Defer ``httpx.AsyncClient`` construction: its SSL setup loads CA
        # certificates from disk synchronously, which HA flags as a blocking
        # call when the integration is constructed on the event loop. We build
        # the client on first use inside an executor; the test transport
        # bypasses this entirely.
        self._timeout_s = timeout_s
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        if transport is not None:
            # Test path: synchronous build is fine because no SSL work happens.
            self._client = httpx.AsyncClient(timeout=timeout_s, transport=transport)
        self._client_init_lock = asyncio.Lock()
        self._lock = asyncio.Lock()
        self._request_times: deque[float] = deque()
        self._rate_limit = rate_limit_requests or self.RATE_LIMIT_REQUESTS
        self._rate_window = rate_limit_window_s or self.RATE_LIMIT_WINDOW_S
        self._sleep = sleep
        # ``clock`` is injectable for deterministic rate-limit tests. Default
        # uses the running event loop's monotonic time.
        self._clock = clock or (lambda: asyncio.get_event_loop().time())

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        async with self._client_init_lock:
            if self._client is None:
                # asyncio.to_thread runs in the default thread pool so the SSL
                # CA-cert load doesn't block the event loop.
                self._client = await asyncio.to_thread(
                    httpx.AsyncClient, timeout=self._timeout_s, transport=self._transport,
                )
        return self._client

    async def __aenter__(self) -> "AsyncDMIClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def list_latest(
        self,
        *,
        limit: int = 12,
        scan_type: str | None = None,
    ) -> list[RadarFeature]:
        # DMI rejects float `limit` (HTTP 400); coerce defensively.
        params: dict[str, Any] = {"limit": int(limit), "sortorder": "datetime,DESC"}
        if scan_type:
            params["scanType"] = scan_type
        url = f"{self.base_url}/collections/{COMPOSITE_COLLECTION}/items"
        payload = await self._get_json(url, params)
        return [parse_feature(f) for f in payload.get("features", [])]

    async def download(self, feature: RadarFeature, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / feature.filename
        if dest.exists():
            return dest
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        last_exc: Exception | None = None
        client = await self._get_client()
        for attempt in range(self.max_retries):
            await self._respect_rate_limit()
            try:
                async with client.stream("GET", feature.download_url) as response:
                    if _should_retry(response.status_code):
                        await self._sleep_for_retry(attempt, response)
                        continue
                    response.raise_for_status()
                    # File I/O must not block the event loop on HA. Buffer the
                    # download in memory (radar files are ~70-90 KB, plan
                    # Phase 0 finding) and offload the write to a thread.
                    buf = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        buf.extend(chunk)
                    await asyncio.to_thread(tmp.write_bytes, bytes(buf))
                await asyncio.to_thread(tmp.rename, dest)
                return dest
            except (httpx.TransportError, httpx.RemoteProtocolError) as e:
                last_exc = e
                if attempt == self.max_retries - 1:
                    raise
                await self._sleep_for_retry(attempt, None)
        raise RuntimeError(
            f"download {feature.filename} failed after {self.max_retries} retries"
        ) from last_exc

    async def _get_json(self, url: str, params: dict) -> dict:
        last_exc: Exception | None = None
        client = await self._get_client()
        for attempt in range(self.max_retries):
            await self._respect_rate_limit()
            try:
                response = await client.get(url, params=params)
                if _should_retry(response.status_code):
                    await self._sleep_for_retry(attempt, response)
                    continue
                response.raise_for_status()
                return response.json()
            except (httpx.TransportError, httpx.RemoteProtocolError) as e:
                last_exc = e
                if attempt == self.max_retries - 1:
                    raise
                await self._sleep_for_retry(attempt, None)
        raise RuntimeError(f"GET {url} failed after {self.max_retries} retries") from last_exc

    async def _respect_rate_limit(self) -> None:
        async with self._lock:
            now = self._clock()
            cutoff = now - self._rate_window
            while self._request_times and self._request_times[0] < cutoff:
                self._request_times.popleft()
            if len(self._request_times) >= self._rate_limit:
                wait = self._rate_window - (now - self._request_times[0])
                if wait > 0:
                    await self._sleep(wait)
            self._request_times.append(self._clock())

    async def _sleep_for_retry(self, attempt: int, response: httpx.Response | None) -> None:
        if response is not None:
            ra = response.headers.get("Retry-After")
            if ra:
                try:
                    await self._sleep(min(float(ra), self.max_backoff_s))
                    return
                except ValueError:
                    pass
        backoff = min(2.0 ** attempt + random.random(), self.max_backoff_s)
        await self._sleep(backoff)


def _should_retry(status: int) -> bool:
    return status == 429 or status >= 500
