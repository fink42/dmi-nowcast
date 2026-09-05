"""metObs client: parsing, paging, headers, and the trace sentinel.

Fixtures under ``tests/fixtures/metobs/`` are real recorded responses
(trimmed to a handful of features) — the parser is only worth testing
against the shapes DMI actually emits, including the versioned station
collection where one ``stationId`` appears three times.

Offline: every request is served by an ``httpx.MockTransport``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from dmi_nowcast_core.metobs import (
    API_KEY_HEADER,
    DEFAULT_BASE_URL,
    AsyncMetObsClient,
    MetObsClient,
    Observation,
    ParseStats,
    datetime_range,
    is_trace,
    next_link,
    normalize_precip_mm,
    observation_params,
    parse_observation,
    parse_observations,
    parse_station,
    parse_stations,
    select_current_station_features,
)

FIXTURES = Path(__file__).parent / "fixtures" / "metobs"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parses_recorded_observation_features() -> None:
    payload = _load("observations_page1.json")
    stats = ParseStats()
    obs = parse_observations(payload["features"], stats)
    assert obs, "fixture should contain features"
    assert stats.skipped == 0
    assert stats.parsed == len(obs)
    for o in obs:
        assert isinstance(o, Observation)
        assert o.parameter_id == "precip_past10min"
        assert o.observed_utc.tzinfo == timezone.utc
        # DMI stamps the 10-min grid.
        assert o.observed_utc.minute % 10 == 0
        assert o.observed_utc.second == 0
        assert o.station_id


def test_parses_recorded_station_features() -> None:
    payload = _load("stations.json")
    stations = parse_stations(select_current_station_features(payload["features"]))
    by_id = {s.station_id: s for s in stations}
    assert "06188" in by_id
    sj = by_id["06188"]
    assert sj.name == "Sjælsmark"
    assert sj.kind == "Synop"
    assert sj.country == "DNK"
    assert "precip_past10min" in sj.parameter_ids
    assert 54.0 < sj.lat < 58.0 and 8.0 < sj.lon < 16.0
    assert sj.operation_from is not None
    assert sj.operation_from.tzinfo == timezone.utc


def test_station_collection_is_versioned_and_current_version_wins() -> None:
    """704 features cover 294 stations; picking the wrong version would
    take a superseded row's metadata."""
    payload = _load("stations.json")
    features = payload["features"]
    ids = [f["properties"]["stationId"] for f in features]
    assert len(ids) > len(set(ids)), "fixture must contain a versioned station"

    current = select_current_station_features(features)
    current_ids = [f["properties"]["stationId"] for f in current]
    assert len(current_ids) == len(set(current_ids))
    for feature in current:
        versions = [
            f for f in features
            if f["properties"]["stationId"] == feature["properties"]["stationId"]
        ]
        open_versions = [f for f in versions if f["properties"].get("validTo") is None]
        if open_versions:
            assert feature["properties"].get("validTo") is None
        else:
            latest = max(f["properties"]["validFrom"] for f in versions)
            assert feature["properties"]["validFrom"] == latest


@pytest.mark.parametrize(
    "feature, reason",
    [
        ("not a dict", "not_an_object"),
        ({"properties": None}, "no_properties"),
        ({"properties": {"parameterId": "p", "observed": "2026-09-01T00:00:00Z", "value": 1}},
         "missing_field"),
        ({"properties": {"stationId": "1", "parameterId": "p",
                         "observed": "2026-09-01T00:00:00Z", "value": None}},
         "null_value"),
        ({"properties": {"stationId": "1", "parameterId": "p",
                         "observed": "not-a-time", "value": 1}},
         "unparseable"),
    ],
)
def test_malformed_features_are_skipped_and_counted(feature, reason: str) -> None:
    stats = ParseStats()
    assert parse_observation(feature, stats) is None
    assert stats.skipped == 1
    assert stats.reasons == {reason: 1}


def test_one_bad_feature_does_not_lose_the_good_ones() -> None:
    good = _load("observations_page1.json")["features"]
    stats = ParseStats()
    obs = parse_observations([{"broken": True}, *good], stats)
    assert len(obs) == len(good)
    assert stats.skipped == 1


def test_station_without_geometry_is_skipped() -> None:
    stats = ParseStats()
    assert parse_station({"properties": {"stationId": "1"}}, stats) is None
    assert stats.reasons == {"no_geometry": 1}


# ---------------------------------------------------------------------------
# Trace sentinel
# ---------------------------------------------------------------------------


def test_trace_sentinel_is_zero_mm_not_an_amount() -> None:
    """DMI's -0.1 means "traces, less than 0.1 kg/m²" — real, but not a
    measurable amount. Summing it would subtract rain from the record."""
    assert normalize_precip_mm(-0.1) == 0.0
    assert is_trace(-0.1)
    # Any negative, not just the documented sentinel.
    assert normalize_precip_mm(-0.5) == 0.0
    assert is_trace(-0.5)
    # Real values pass through untouched, and missing stays missing.
    assert normalize_precip_mm(0.0) == 0.0
    assert normalize_precip_mm(2.3) == pytest.approx(2.3)
    assert normalize_precip_mm(None) is None
    assert not is_trace(0.0)
    assert not is_trace(None)


def test_client_stores_the_raw_trace_value() -> None:
    """The archive keeps what DMI reported; normalisation happens at the
    point of use, never on the way in."""
    stats = ParseStats()
    obs = parse_observation(
        {"properties": {"stationId": "06074", "parameterId": "precip_past10min",
                        "observed": "2026-09-01T12:00:00Z", "value": -0.1}},
        stats,
    )
    assert obs is not None
    assert obs.value == pytest.approx(-0.1)


# ---------------------------------------------------------------------------
# Request shaping
# ---------------------------------------------------------------------------


def test_datetime_range_is_inclusive_at_both_ends() -> None:
    """Verified against the live API: 00:00/00:20 returns :00, :10 AND
    :20. A day is therefore requested as 00:00:00/23:59:59."""
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 1, 23, 59, 59, tzinfo=timezone.utc)
    assert datetime_range(start, end) == "2026-09-01T00:00:00Z/2026-09-01T23:59:59Z"


def test_naive_datetimes_are_rejected() -> None:
    with pytest.raises(ValueError):
        datetime_range(datetime(2026, 9, 1), datetime(2026, 9, 2, tzinfo=timezone.utc))


def test_observation_params_shape() -> None:
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 1, 1, tzinfo=timezone.utc)
    p = observation_params("precip_past10min", start, end,
                           station_id="06074", bbox=(8.0, 54.5, 15.5, 58.0), limit=42)
    assert p["parameterId"] == "precip_past10min"
    assert p["limit"] == 42
    assert p["stationId"] == "06074"
    assert p["bbox"] == "8,54.5,15.5,58"


def test_next_link_found_and_absent() -> None:
    assert next_link(_load("observations_page1.json")) is not None
    assert next_link(_load("observations_page2.json")) is None


# ---------------------------------------------------------------------------
# HTTP behaviour
# ---------------------------------------------------------------------------


def _paging_transport(seen: list[httpx.Request]) -> httpx.MockTransport:
    page1 = _load("observations_page1.json")
    page2 = _load("observations_page2.json")
    next_href = next_link(page1)
    assert next_href

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if str(request.url) == next_href:
            return httpx.Response(200, json=page2)
        return httpx.Response(200, json=page1)

    return httpx.MockTransport(handler)


def test_fetch_observations_follows_the_next_link() -> None:
    seen: list[httpx.Request] = []
    with httpx.Client(transport=_paging_transport(seen)) as session:
        client = MetObsClient(session=session)
        obs = client.fetch_observations(
            "precip_past10min",
            datetime(2026, 9, 1, 12, tzinfo=timezone.utc),
            datetime(2026, 9, 1, 12, 10, tzinfo=timezone.utc),
        )
    expected = (
        len(_load("observations_page1.json")["features"])
        + len(_load("observations_page2.json")["features"])
    )
    assert len(obs) == expected
    assert len(seen) == 2
    # The second request is the server's own href — no params re-applied.
    assert str(seen[1].url) == next_link(_load("observations_page1.json"))


def test_paging_stops_on_a_self_referential_next_link() -> None:
    """A server that returns itself as `next` must not spin forever."""
    payload = _load("observations_page1.json")
    self_href = "https://example.invalid/loop"
    payload = {**payload, "links": [{"rel": "next", "href": self_href}]}
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as session:
        MetObsClient(session=session).fetch_observations(
            "precip_past10min",
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 9, 1, 1, tzinfo=timezone.utc),
        )
    # First page, then the `next` page, then the repeat is recognised.
    assert calls["n"] == 2


def test_api_key_header_sent_only_when_configured() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_load("observations_page2.json"))

    args = (
        "precip_past10min",
        datetime(2026, 9, 1, tzinfo=timezone.utc),
        datetime(2026, 9, 1, 1, tzinfo=timezone.utc),
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as session:
        MetObsClient(session=session).fetch_observations(*args)
        assert API_KEY_HEADER not in seen[-1].headers

        MetObsClient(session=session, api_key="s3cret").fetch_observations(*args)
        assert seen[-1].headers[API_KEY_HEADER] == "s3cret"


def test_station_ids_issue_one_request_each() -> None:
    """DMI's stationId filter takes a single value, so a multi-station
    query is N requests — not one that silently returns the wrong set."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_load("observations_page2.json"))

    with httpx.Client(transport=httpx.MockTransport(handler)) as session:
        MetObsClient(session=session).fetch_observations(
            "precip_past10min",
            datetime(2026, 9, 1, tzinfo=timezone.utc),
            datetime(2026, 9, 1, 1, tzinfo=timezone.utc),
            station_ids=["06074", "06188"],
        )
    assert len(seen) == 2
    assert [r.url.params["stationId"] for r in seen] == ["06074", "06188"]


def test_client_defaults_to_the_new_host() -> None:
    assert MetObsClient().base_url == DEFAULT_BASE_URL.rstrip("/")
    assert "opendataapi.dmi.dk" in DEFAULT_BASE_URL


def test_http_error_propagates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with httpx.Client(transport=httpx.MockTransport(handler)) as session:
        with pytest.raises(httpx.HTTPStatusError):
            MetObsClient(session=session).fetch_observations(
                "precip_past10min",
                datetime(2026, 9, 1, tzinfo=timezone.utc),
                datetime(2026, 9, 1, 1, tzinfo=timezone.utc),
            )


# ---------------------------------------------------------------------------
# Async twin
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_client_matches_the_sync_one() -> None:
    seen: list[httpx.Request] = []
    sync_seen: list[httpx.Request] = []
    args = (
        "precip_past10min",
        datetime(2026, 9, 1, 12, tzinfo=timezone.utc),
        datetime(2026, 9, 1, 12, 10, tzinfo=timezone.utc),
    )

    page1 = _load("observations_page1.json")
    page2 = _load("observations_page2.json")
    next_href = next_link(page1)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=page2 if str(request.url) == next_href else page1)

    client = AsyncMetObsClient(transport=httpx.MockTransport(handler), api_key="k")
    try:
        got = await client.fetch_observations(*args)
    finally:
        await client.aclose()

    with httpx.Client(transport=_paging_transport(sync_seen)) as session:
        expected = MetObsClient(session=session).fetch_observations(*args)

    assert got == expected
    assert seen[0].headers[API_KEY_HEADER] == "k"


@pytest.mark.asyncio
async def test_async_fetch_stations_dedupes_versions() -> None:
    payload = _load("stations.json")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = AsyncMetObsClient(transport=httpx.MockTransport(handler))
    try:
        stations = await client.fetch_stations()
    finally:
        await client.aclose()
    ids = [s.station_id for s in stations]
    assert len(ids) == len(set(ids))
