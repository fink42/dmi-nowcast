"""Unit tests for the DMI API client using mocked httpx transport."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from dmi_nowcast_core.fetch import (
    DEFAULT_BASE_URL,
    RadarFeature,
    download,
    list_latest,
    parse_feature,
)

# A real STAC feature, captured from the live API on 2026-05-17.
SAMPLE_FEATURE = {
    "type": "Feature",
    "id": "dk.com.202605171935.500_max.h5",
    "geometry": {"type": "Polygon", "coordinates": []},
    "properties": {
        "datetime": "2026-05-17T19:35:00Z",
        "created": "2026-05-17T19:42:08.082Z",
        "scanType": "doppler",
    },
    "stac_version": "1.0.0",
    "collection": "composite",
    "asset": {
        "data": {
            "type": "application/x-hdf5",
            "title": "Radar file download resource",
            "roles": ["data"],
            "href": "https://opendataapi.dmi.dk/v1/radardata/download/dk.com.202605171935.500_max.h5",
        }
    },
}


def test_parse_feature_extracts_all_fields():
    f = parse_feature(SAMPLE_FEATURE)
    assert f.feature_id == "dk.com.202605171935.500_max.h5"
    assert f.datetime_utc == datetime(2026, 5, 17, 19, 35, tzinfo=timezone.utc)
    assert f.scan_type == "doppler"
    assert f.download_url.endswith("/download/dk.com.202605171935.500_max.h5")
    assert f.filename == "dk.com.202605171935.500_max.h5"


def test_parse_feature_accepts_plural_assets_per_stac_spec():
    """DMI uses ``asset`` (singular); STAC spec is ``assets`` (plural). Accept both."""
    plural = dict(SAMPLE_FEATURE)
    plural["assets"] = plural.pop("asset")
    f = parse_feature(plural)
    assert f.download_url.endswith(".h5")


def test_parse_feature_handles_missing_datetime():
    bad = dict(SAMPLE_FEATURE)
    bad["properties"] = {"scanType": "fullRange"}
    f = parse_feature(bad)
    assert f.datetime_utc.tzinfo is not None  # always UTC-aware


def _mock_transport(features: list[dict]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/collections/composite/items")
        assert request.url.params["sortorder"] == "datetime,DESC"
        return httpx.Response(200, json={"type": "FeatureCollection", "features": features})

    return httpx.MockTransport(handler)


def test_list_latest_returns_features_sorted_newest_first():
    older = dict(SAMPLE_FEATURE)
    older = {**SAMPLE_FEATURE, "id": "dk.com.202605171930.500_max.h5",
             "properties": {**SAMPLE_FEATURE["properties"], "datetime": "2026-05-17T19:30:00Z",
                            "scanType": "fullRange"}}
    transport = _mock_transport([SAMPLE_FEATURE, older])
    with httpx.Client(transport=transport) as c:
        features = list_latest(client=c, limit=2)
    assert len(features) == 2
    assert features[0].datetime_utc > features[1].datetime_utc
    assert features[0].scan_type == "doppler"
    assert features[1].scan_type == "fullRange"


def test_list_latest_filters_by_scan_type():
    seen_params = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_params.update(request.url.params)
        return httpx.Response(200, json={"features": [SAMPLE_FEATURE]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        list_latest(client=c, scan_type="fullRange")
    assert seen_params["scanType"] == "fullRange"


def test_download_writes_file_atomically(tmp_path: Path):
    body = b"\x89HDF\r\n\x1a\n" + b"\x00" * 1024  # HDF5 magic header + filler

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    feature = parse_feature(SAMPLE_FEATURE)
    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        path = download(feature, tmp_path, client=c)

    assert path.exists()
    assert path.read_bytes() == body
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_download_is_idempotent(tmp_path: Path):
    """Cached files must not be re-downloaded — DMI URLs are immutable per filename."""
    feature = parse_feature(SAMPLE_FEATURE)
    existing = tmp_path / feature.filename
    existing.write_bytes(b"previously downloaded")

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("download() should not hit the network when file already exists")

    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        path = download(feature, tmp_path, client=c)
    assert path.read_bytes() == b"previously downloaded"


def test_default_base_url_is_new_unauthenticated_host():
    """Plan §3.1: new host requires no API key; legacy dmigw retiring 2026-06-30."""
    assert DEFAULT_BASE_URL == "https://opendataapi.dmi.dk/v1/radardata"
