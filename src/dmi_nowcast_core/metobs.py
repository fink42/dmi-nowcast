"""DMI Open Data **metObs** client — weather-station observations.

Gauge truth for the nowcast benchmark (Phase F). The radar corpus is
verified against the radar itself; this module supplies the independent
ground truth a rain gauge gives, so "how good are we?" can be answered
without the composite grading its own homework.

Data
----
``https://opendataapi.dmi.dk/v2/metObs`` — GeoJSON ``FeatureCollection``
responses over two collections:

- ``observation``: one feature per (station, parameter, timestamp), with
  ``properties = {parameterId, created, value, observed, stationId}`` and
  ``geometry.coordinates = [lon, lat]``.
- ``station``: the station catalogue (704 rows, ~555 Danish).

The parameters this project cares about:

===========================  ===========================================
``precip_past10min``         mm accumulated in the 10 min **ending** at
                             the stamp; 0.1 mm resolution; ~112 Danish
                             stations reporting per slot.
``precip_dur_past10min``     minutes with precipitation inside that same
                             slot (0-10); ~100 stations since 2023.
``precip_past1min``          mm in the minute ending at the stamp; 26
                             stations.
``precip_past1h``            mm in the hour ending at the stamp.
===========================  ===========================================

DMI's ``past…`` naming means *the interval ending at the stamp*; that is
verified empirically against ``precip_past1h`` in the Phase F pre-flight,
not assumed here.

Traces
------
DMI encodes "traces of precipitation, less than 0.1 kg/m²" as the value
**-0.1**. It is documented for ``precip_past1h`` and applies to the other
precipitation parameters the same way. A negative reading is therefore
never an amount: :func:`normalize_precip_mm` maps it to ``0.0``. This
module stores what the API reported — callers normalise at the point of
use, so nothing is silently rewritten on the way into the archive.

Access
------
API keys are no longer required on ``opendataapi.dmi.dk`` (DMI dropped
the requirement on 2025-12-02). Keyless is the documented policy, not an
accident of the current deployment, so the client defaults to sending no
key — while still accepting one (``X-Gravitee-Api-Key``) for anyone
running against a gated deployment or the retiring legacy host. DMI's
fair-use terms still apply: filter server-side, cache, and don't poll
faster than the data changes.

Licence of the data: **CC BY 4.0** (DMI Open Data). Attribute DMI in
anything derived from it.

Implementation note: like :mod:`dmi_nowcast_core.fetch`, both the sync
and the async client are built on ``httpx``. The repo deliberately keeps
one HTTP library, and ``httpx`` is already a core dependency while
``aiohttp`` is not.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Sequence

import httpx

DEFAULT_BASE_URL = "https://opendataapi.dmi.dk/v2/metObs"
#: The retiring legacy host. Serves the same payloads; kept only as an
#: override for anyone still pinned to it.
LEGACY_BASE_URL = "https://dmigw.govcloud.dk/v2/metObs"

OBSERVATION_COLLECTION = "observation"
STATION_COLLECTION = "station"

#: Header DMI's gateway reads an (optional) API key from.
API_KEY_HEADER = "X-Gravitee-Api-Key"

#: One request returns a whole day of ``precip_past10min`` (~15.7k rows)
#: at this limit, so the day-per-request backfill needs no paging in
#: practice. Paging is still implemented — DMI may cap it server-side.
DEFAULT_LIMIT = 300_000

#: Hard stop on ``rel: next`` following, so a server-side paging bug
#: cannot spin forever.
MAX_PAGES = 500

#: DMI's "traces of precipitation, less than 0.1 kg/m²" sentinel.
TRACE_VALUE_MM = -0.1

#: Gauge resolution: the smallest amount that is an amount at all.
GAUGE_RESOLUTION_MM = 0.1

PRECIP_PAST_10MIN = "precip_past10min"
PRECIP_DUR_PAST_10MIN = "precip_dur_past10min"
PRECIP_PAST_1MIN = "precip_past1min"
PRECIP_PAST_1H = "precip_past1h"

#: What the backfill collects by default: the gauge amount, the wet-slot
#: duration that rescues sub-resolution drizzle, and the 1-min series for
#: the handful of stations that report it.
DEFAULT_PARAMETERS: tuple[str, ...] = (
    PRECIP_PAST_10MIN,
    PRECIP_DUR_PAST_10MIN,
    PRECIP_PAST_1MIN,
)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """One (station, parameter, instant) reading.

    ``value`` is exactly what DMI reported — including the ``-0.1``
    trace sentinel. Use :func:`normalize_precip_mm` before treating it
    as an amount.
    """

    station_id: str
    observed_utc: datetime
    parameter_id: str
    value: float


@dataclass(frozen=True)
class Station:
    """A metObs station catalogue entry.

    ``kind`` is the API's ``type`` field (``Synop``, ``Pluvio``,
    ``Manual snow``, ``GIWS``, …) — renamed because ``type`` shadows the
    builtin everywhere it would be used.
    """

    station_id: str
    name: str
    kind: str
    lat: float
    lon: float
    country: str
    operation_from: datetime | None
    operation_to: datetime | None
    status: str
    parameter_ids: tuple[str, ...]
    region_id: str


@dataclass
class ParseStats:
    """How many features a parse kept and how many it dropped."""

    parsed: int = 0
    skipped: int = 0
    #: Distinct reasons, for a log line that says *why* rows vanished.
    reasons: dict[str, int] = field(default_factory=dict)

    def _skip(self, reason: str) -> None:
        self.skipped += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def merge(self, other: "ParseStats") -> None:
        self.parsed += other.parsed
        self.skipped += other.skipped
        for k, v in other.reasons.items():
            self.reasons[k] = self.reasons.get(k, 0) + v


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------


def normalize_precip_mm(value: float | None) -> float | None:
    """Map DMI's trace sentinel (and any negative) to ``0.0`` mm.

    -0.1 means "traces, less than 0.1 kg/m²": real but unmeasurable
    precipitation. It is not an amount and must never be summed as one,
    yet it is also not missing. Zero is the only honest amount for it —
    the slot can still be classified wet through
    ``precip_dur_past10min``, which is exactly what that parameter is
    for.
    """
    if value is None:
        return None
    return 0.0 if value < 0.0 else float(value)


def is_trace(value: float | None) -> bool:
    """True when ``value`` is a negative (trace) reading."""
    return value is not None and value < 0.0


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_datetime(s: str) -> datetime:
    """Parse an ISO-8601 stamp to an aware UTC datetime."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_observation(feature: Any, stats: ParseStats | None = None) -> Observation | None:
    """Parse one GeoJSON feature into an :class:`Observation`.

    Returns ``None`` — and counts the reason in ``stats`` — for anything
    malformed. A single bad feature must never lose the other 15,699 in
    the response.
    """
    st = stats if stats is not None else ParseStats()
    if not isinstance(feature, dict):
        st._skip("not_an_object")
        return None
    props = feature.get("properties")
    if not isinstance(props, dict):
        st._skip("no_properties")
        return None
    station_id = props.get("stationId")
    parameter_id = props.get("parameterId")
    observed = props.get("observed")
    value = props.get("value")
    if not station_id or not parameter_id or not observed:
        st._skip("missing_field")
        return None
    if value is None:
        # A reported null is a missing measurement, not a malformed row;
        # it carries no information the store wants.
        st._skip("null_value")
        return None
    try:
        observed_utc = parse_datetime(str(observed))
        value_f = float(value)
    except (TypeError, ValueError):
        st._skip("unparseable")
        return None
    st.parsed += 1
    return Observation(
        station_id=str(station_id),
        observed_utc=observed_utc,
        parameter_id=str(parameter_id),
        value=value_f,
    )


def parse_observations(
    features: Iterable[Any], stats: ParseStats | None = None
) -> list[Observation]:
    st = stats if stats is not None else ParseStats()
    out: list[Observation] = []
    for f in features:
        obs = parse_observation(f, st)
        if obs is not None:
            out.append(obs)
    return out


def _opt_datetime(s: Any) -> datetime | None:
    if not s:
        return None
    try:
        return parse_datetime(str(s))
    except (TypeError, ValueError):
        return None


def parse_station(feature: Any, stats: ParseStats | None = None) -> Station | None:
    """Parse one station feature; ``None`` when it is unusable."""
    st = stats if stats is not None else ParseStats()
    if not isinstance(feature, dict):
        st._skip("not_an_object")
        return None
    props = feature.get("properties")
    if not isinstance(props, dict):
        st._skip("no_properties")
        return None
    station_id = props.get("stationId")
    if not station_id:
        st._skip("missing_field")
        return None
    geom = feature.get("geometry")
    coords = geom.get("coordinates") if isinstance(geom, dict) else None
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        st._skip("no_geometry")
        return None
    try:
        lon = float(coords[0])
        lat = float(coords[1])
    except (TypeError, ValueError):
        st._skip("unparseable")
        return None
    raw_params = props.get("parameterId") or []
    if isinstance(raw_params, str):
        raw_params = [raw_params]
    st.parsed += 1
    return Station(
        station_id=str(station_id),
        name=str(props.get("name") or ""),
        kind=str(props.get("type") or ""),
        lat=lat,
        lon=lon,
        country=str(props.get("country") or ""),
        operation_from=_opt_datetime(props.get("operationFrom")),
        operation_to=_opt_datetime(props.get("operationTo")),
        status=str(props.get("status") or ""),
        parameter_ids=tuple(str(p) for p in raw_params if p),
        region_id=str(props.get("regionId") or ""),
    )


def parse_stations(features: Iterable[Any], stats: ParseStats | None = None) -> list[Station]:
    st = stats if stats is not None else ParseStats()
    out: list[Station] = []
    for f in features:
        s = parse_station(f, st)
        if s is not None:
            out.append(s)
    return out


def select_current_station_features(features: Iterable[Any]) -> list[Any]:
    """One feature per ``stationId``: the currently valid version.

    The station collection is **versioned**, not unique: 704 features
    cover 294 stations, one row per ``validFrom``/``validTo`` epoch of
    the same physical station (relocations, sensor swaps, metadata
    corrections). Taking whatever order the API happened to return would
    pick a superseded row's coordinates.

    The current version is the one with no ``validTo``; when every
    version is closed (a decommissioned station) the latest ``validFrom``
    wins. Features without a parseable ``stationId`` pass through
    untouched so :func:`parse_station` can count them as skipped.
    """
    best: dict[str, tuple[int, datetime, Any]] = {}
    passthrough: list[Any] = []
    for feature in features:
        props = feature.get("properties") if isinstance(feature, dict) else None
        station_id = props.get("stationId") if isinstance(props, dict) else None
        if not station_id:
            passthrough.append(feature)
            continue
        valid_to = _opt_datetime(props.get("validTo"))
        valid_from = _opt_datetime(props.get("validFrom")) or datetime.min.replace(
            tzinfo=timezone.utc
        )
        # (open-ended?, validFrom) — an open version beats any closed one.
        rank = (1 if valid_to is None else 0, valid_from)
        key = str(station_id)
        current = best.get(key)
        if current is None or rank > (current[0], current[1]):
            best[key] = (rank[0], rank[1], feature)
    return passthrough + [entry[2] for entry in best.values()]


# ---------------------------------------------------------------------------
# Request shaping (shared by the sync and async clients)
# ---------------------------------------------------------------------------


def _ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def datetime_range(start_utc: datetime, end_utc: datetime) -> str:
    """DMI's ``datetime`` interval string. **Both ends inclusive.**

    Verified against the live API: ``00:00:00Z/00:20:00Z`` returns the
    :00, :10 *and* :20 slots. A day is therefore requested as
    ``00:00:00Z/23:59:59Z``, which stops at 23:50 and cannot double-count
    the next day's midnight slot.
    """
    if start_utc.tzinfo is None or end_utc.tzinfo is None:
        raise ValueError("start/end must be timezone-aware UTC datetimes")
    if end_utc < start_utc:
        raise ValueError("end must not precede start")
    return f"{_ts(start_utc)}/{_ts(end_utc)}"


def observation_params(
    parameter_id: str,
    start_utc: datetime,
    end_utc: datetime,
    *,
    station_id: str | None = None,
    bbox: Sequence[float] | None = None,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "parameterId": parameter_id,
        "datetime": datetime_range(start_utc, end_utc),
        "limit": int(limit),
    }
    if station_id:
        params["stationId"] = str(station_id)
    if bbox is not None:
        if len(tuple(bbox)) != 4:
            raise ValueError("bbox must be (lon_min, lat_min, lon_max, lat_max)")
        params["bbox"] = ",".join(f"{float(v):g}" for v in bbox)
    return params


def next_link(payload: Any) -> str | None:
    """The ``rel: next`` href of a FeatureCollection, if any."""
    if not isinstance(payload, dict):
        return None
    links = payload.get("links")
    if not isinstance(links, list):
        return None
    for link in links:
        if isinstance(link, dict) and link.get("rel") == "next":
            href = link.get("href")
            if isinstance(href, str) and href:
                return href
    return None


def _features(payload: Any) -> list[Any]:
    if not isinstance(payload, dict):
        return []
    feats = payload.get("features")
    return feats if isinstance(feats, list) else []


def _station_id_list(station_ids: Iterable[str] | None) -> list[str | None]:
    """One request per station id — DMI's ``stationId`` takes one value.

    ``None`` (no filter) is the single-element list ``[None]`` so both
    call sites share one loop.
    """
    if station_ids is None:
        return [None]
    ids = [str(s) for s in station_ids if s]
    return ids or [None]


def _headers(api_key: str | None, user_agent: str) -> dict[str, str]:
    headers = {"User-Agent": user_agent, "Accept": "application/geo+json, application/json"}
    if api_key:
        headers[API_KEY_HEADER] = api_key
    return headers


DEFAULT_USER_AGENT = "dmi-nowcast-core/0.1 (metObs gauge truth)"


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------


class MetObsClient:
    """Blocking metObs client — backfill scripts and analysis.

    ``session`` accepts a pre-built ``httpx.Client`` (tests inject one
    with a ``MockTransport``); otherwise one is created per call and
    closed again, matching :mod:`dmi_nowcast_core.fetch`'s style.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        session: httpx.Client | None = None,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self._session = session
        self._user_agent = user_agent
        #: Cumulative parse bookkeeping across every call on this client.
        self.stats = ParseStats()
        #: The most recent call's bookkeeping, for a per-request log line.
        self.last_stats = ParseStats()

    # -- plumbing ---------------------------------------------------------

    def _client(self) -> tuple[httpx.Client, bool]:
        if self._session is not None:
            return self._session, False
        return httpx.Client(timeout=self.timeout_s), True

    def _get_json(self, client: httpx.Client, url: str, params: dict | None) -> Any:
        resp = client.get(url, params=params, headers=_headers(self.api_key, self._user_agent))
        resp.raise_for_status()
        return resp.json()

    def _paged(
        self, client: httpx.Client, url: str, params: dict | None
    ) -> Iterator[Any]:
        """Yield each page's payload, following ``rel: next``."""
        seen: set[str] = set()
        page_url: str | None = url
        page_params: dict | None = params
        for _ in range(MAX_PAGES):
            if page_url is None:
                return
            payload = self._get_json(client, page_url, page_params)
            yield payload
            nxt = next_link(payload)
            if not nxt or nxt in seen:
                return
            seen.add(nxt)
            # The next href is fully qualified and already carries every
            # query parameter, so params must not be re-applied.
            page_url, page_params = nxt, None
        raise RuntimeError(f"metObs paging exceeded {MAX_PAGES} pages for {url}")

    # -- API --------------------------------------------------------------

    def fetch_observations(
        self,
        parameter_id: str,
        start_utc: datetime,
        end_utc: datetime,
        *,
        station_ids: Iterable[str] | None = None,
        bbox: Sequence[float] | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[Observation]:
        """Observations of one parameter in ``[start_utc, end_utc]``.

        Both bounds are inclusive (DMI's convention — see
        :func:`datetime_range`). ``station_ids`` issues one request per
        station because the API's ``stationId`` filter takes a single
        value; leave it ``None`` to get every reporting station in one
        request, which is what the day-per-request backfill does.
        """
        url = f"{self.base_url}/collections/{OBSERVATION_COLLECTION}/items"
        stats = ParseStats()
        out: list[Observation] = []
        client, owns = self._client()
        try:
            for station_id in _station_id_list(station_ids):
                params = observation_params(
                    parameter_id, start_utc, end_utc,
                    station_id=station_id, bbox=bbox, limit=limit,
                )
                for payload in self._paged(client, url, params):
                    out.extend(parse_observations(_features(payload), stats))
        finally:
            if owns:
                client.close()
        self.last_stats = stats
        self.stats.merge(stats)
        return out

    def fetch_stations(self, *, limit: int = 10_000) -> list[Station]:
        """The full station catalogue (704 rows at the time of writing)."""
        url = f"{self.base_url}/collections/{STATION_COLLECTION}/items"
        stats = ParseStats()
        out: list[Station] = []
        client, owns = self._client()
        try:
            features: list[Any] = []
            for payload in self._paged(client, url, {"limit": int(limit)}):
                features.extend(_features(payload))
            out.extend(parse_stations(select_current_station_features(features), stats))
        finally:
            if owns:
                client.close()
        self.last_stats = stats
        self.stats.merge(stats)
        return out


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


class AsyncMetObsClient:
    """Async twin of :class:`MetObsClient` for the sidecar poller.

    Same contract, same parsing, ``async``/``await`` plumbing. As in
    :class:`dmi_nowcast_core.fetch.AsyncDMIClient`, the ``httpx``
    client is constructed lazily inside a thread: building one loads CA
    certificates from disk, which is blocking work that has no business
    on an event loop.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        session: httpx.AsyncClient | None = None,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self._user_agent = user_agent
        self._transport = transport
        self._client: httpx.AsyncClient | None = session
        self._owns_client = session is None
        if session is None and transport is not None:
            # Test path: no SSL context is built, so this is loop-safe.
            self._client = httpx.AsyncClient(timeout=timeout_s, transport=transport)
        self._init_lock = asyncio.Lock()
        self.stats = ParseStats()
        self.last_stats = ParseStats()

    async def __aenter__(self) -> "AsyncMetObsClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        async with self._init_lock:
            if self._client is None:
                self._client = await asyncio.to_thread(
                    httpx.AsyncClient, timeout=self.timeout_s, transport=self._transport,
                )
        return self._client

    async def _get_json(self, url: str, params: dict | None) -> Any:
        client = await self._get_client()
        resp = await client.get(
            url, params=params, headers=_headers(self.api_key, self._user_agent),
        )
        resp.raise_for_status()
        return resp.json()

    async def _pages(self, url: str, params: dict | None) -> list[Any]:
        seen: set[str] = set()
        pages: list[Any] = []
        page_url: str | None = url
        page_params: dict | None = params
        for _ in range(MAX_PAGES):
            if page_url is None:
                return pages
            payload = await self._get_json(page_url, page_params)
            pages.append(payload)
            nxt = next_link(payload)
            if not nxt or nxt in seen:
                return pages
            seen.add(nxt)
            page_url, page_params = nxt, None
        raise RuntimeError(f"metObs paging exceeded {MAX_PAGES} pages for {url}")

    async def fetch_observations(
        self,
        parameter_id: str,
        start_utc: datetime,
        end_utc: datetime,
        *,
        station_ids: Iterable[str] | None = None,
        bbox: Sequence[float] | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[Observation]:
        url = f"{self.base_url}/collections/{OBSERVATION_COLLECTION}/items"
        stats = ParseStats()
        out: list[Observation] = []
        for station_id in _station_id_list(station_ids):
            params = observation_params(
                parameter_id, start_utc, end_utc,
                station_id=station_id, bbox=bbox, limit=limit,
            )
            for payload in await self._pages(url, params):
                out.extend(parse_observations(_features(payload), stats))
        self.last_stats = stats
        self.stats.merge(stats)
        return out

    async def fetch_stations(self, *, limit: int = 10_000) -> list[Station]:
        url = f"{self.base_url}/collections/{STATION_COLLECTION}/items"
        stats = ParseStats()
        out: list[Station] = []
        features: list[Any] = []
        for payload in await self._pages(url, {"limit": int(limit)}):
            features.extend(_features(payload))
        out.extend(parse_stations(select_current_station_features(features), stats))
        self.last_stats = stats
        self.stats.merge(stats)
        return out


__all__ = [
    "DEFAULT_BASE_URL",
    "LEGACY_BASE_URL",
    "API_KEY_HEADER",
    "DEFAULT_LIMIT",
    "DEFAULT_PARAMETERS",
    "TRACE_VALUE_MM",
    "GAUGE_RESOLUTION_MM",
    "PRECIP_PAST_10MIN",
    "PRECIP_DUR_PAST_10MIN",
    "PRECIP_PAST_1MIN",
    "PRECIP_PAST_1H",
    "Observation",
    "Station",
    "ParseStats",
    "MetObsClient",
    "AsyncMetObsClient",
    "normalize_precip_mm",
    "is_trace",
    "parse_observation",
    "parse_observations",
    "parse_station",
    "parse_stations",
    "select_current_station_features",
    "parse_datetime",
    "datetime_range",
    "observation_params",
    "next_link",
]
