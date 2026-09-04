"""D1 — Web Push backend: store, VAPID, endpoint policy, routes, service.

The decision state machine itself lives in ``test_push_engine.py``; this
module covers everything around it, and in particular the three things
that would be expensive to get wrong in production:

1. **The SSRF guard.** ``POST /api/push/subscribe`` hands the service a
   URL it will later POST to, from a VM that can see the LAN. The policy
   tests are adversarial on purpose (IP literals, suffix spoofing).
2. **Path safety.** The VAPID private key and the subscription database
   now live *inside* ``storage.data_dir``, which is the directory two
   route families read files out of. Every traversal shape is probed, with
   and without the static frontend mounted, and the response body is
   checked for the PEM header and the SQLite magic — not just the status.
3. **Send-once semantics.** The service persists the new state before it
   sends, skips repeated radar timestamps, and refuses to evaluate grids
   that belong to a different frame than the state it was handed.

Fully synthetic and offline: the 64×64 @ 500 m ODIM helper is the one
``test_app_nowcast.py`` uses (copied per this suite's convention that each
module stands alone), and ``push.fanout.send`` is monkeypatched wherever a
delivery would otherwise happen.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pyproj import CRS, Transformer

from dmi_nowcast_core.geo import CompositeGeo
from dmi_nowcast_core.national import NationalProducts
from dmi_nowcast_core.parse import parse_composite
from dmi_nowcast_sidecar.app import (
    _safe_frame_name,
    _safe_nowcast_name,
    create_app,
)
from dmi_nowcast_sidecar.compute import CycleEngine, CycleResult, NationalSnapshot
from dmi_nowcast_sidecar.config import Config, PushConfig
from dmi_nowcast_sidecar.push import fanout as fanout_mod
from dmi_nowcast_sidecar.push import keygen, vapid
from dmi_nowcast_sidecar.push.endpoint_policy import validate_endpoint
from dmi_nowcast_sidecar.push.fanout import SendResult
from dmi_nowcast_sidecar.push.paths import resolved_db_path, resolved_key_path
from dmi_nowcast_sidecar.push.service import PushService
from dmi_nowcast_sidecar.push.store import NewSubscription, PushStore, sub_id

API_KEY = "push-mode-secret"
HOME_LON, HOME_LAT = 10.32, 55.33
SUBJECT = "mailto:ops@example.com"

ENDPOINT_A = "https://fcm.googleapis.com/fcm/send/AAAAAAAAAAA-token-a"
ENDPOINT_B = "https://updates.push.services.mozilla.com/wpush/v2/token-b"
P256DH = "B" + "x" * 86
AUTH = "y" * 22

PEM_HEADER = b"-----BEGIN PRIVATE KEY-----"
SQLITE_MAGIC = b"SQLite format 3"

DMI_PROJ = "+proj=stere +lat_0=56 +lon_0=10.5666 +lat_ts=56 +ellps=WGS84 +units=m +no_defs"
GRID_PX = 64
PIXEL_M = 500.0
GAIN, OFFSET = 0.5, -32.0
NODATA, UNDETECT = 255, 0

DOWNSAMPLE = 4
GRID_DS = GRID_PX // DOWNSAMPLE
NAN_PIXEL = (2, 3)

RADAR_TS = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
RADAR_TS2 = RADAR_TS + timedelta(minutes=10)


# ---------------------------------------------------------------------------
# Synthetic ODIM helper (adapted from test_app_nowcast.py)
# ---------------------------------------------------------------------------

def _corners_lonlat() -> dict[str, tuple[float, float]]:
    crs = CRS.from_proj4(DMI_PROJ)
    to_proj = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x_home, y_home = to_proj.transform(HOME_LON, HOME_LAT)
    half = GRID_PX / 2 * PIXEL_M
    return {
        "UL": to_wgs.transform(x_home - half, y_home + half),
        "UR": to_wgs.transform(x_home + half, y_home + half),
        "LL": to_wgs.transform(x_home - half, y_home - half),
        "LR": to_wgs.transform(x_home + half, y_home - half),
    }


def _write_composite(path: Path, ts: datetime, dbz_value: float) -> None:
    raw_value = int(round((dbz_value - OFFSET) / GAIN))
    raw = np.full((GRID_PX, GRID_PX), raw_value, dtype=np.uint8)
    with h5py.File(path, "w") as h5:
        what = h5.create_group("what")
        what.attrs["gain"] = GAIN
        what.attrs["offset"] = OFFSET
        what.attrs["nodata"] = NODATA
        what.attrs["undetect"] = UNDETECT
        what.attrs["date"] = ts.strftime("%Y%m%d").encode()
        what.attrs["time"] = ts.strftime("%H%M%S").encode()
        what.attrs["product"] = b"DBZH"
        where = h5.create_group("where")
        where.attrs["projdef"] = DMI_PROJ.encode()
        where.attrs["xscale"] = PIXEL_M
        where.attrs["yscale"] = PIXEL_M
        for name, (lon, lat) in _corners_lonlat().items():
            where.attrs[f"{name}_lon"] = lon
            where.attrs[f"{name}_lat"] = lat
        how = h5.create_group("how")
        how.attrs["zr-a"] = 200.0
        how.attrs["zr-b"] = 1.6
        h5.create_group("dataset1").create_group("data1").create_dataset(
            "data", data=raw,
        )


def _grid(value: float) -> np.ndarray:
    g = np.full((GRID_DS, GRID_DS), value, dtype=np.float32)
    g[NAN_PIXEL] = np.nan
    return g


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def products() -> NationalProducts:
    """Uniformly wet (p = 0.9 at every offered lead), one NaN pixel."""
    return NationalProducts(
        p_rain={20: _grid(0.85), 30: _grid(0.9)},
        eta_min=_grid(6.0),
        intensity_mm_h=_grid(2.5),
        leads_min=(20, 30),
        threshold_mm_h=0.5,
        timestep_min=5.0,
        frame_age_min=2.0,
        downsample_factor=DOWNSAMPLE,
        n_members=8,
    )


@pytest.fixture
def geo(tmp_path: Path) -> CompositeGeo:
    path = tmp_path / "geo_composite.h5"
    _write_composite(path, RADAR_TS, 30.0)
    return CompositeGeo(parse_composite(path))


@pytest.fixture
def push_config(minimal_config: Config) -> Config:
    """``minimal_config`` with push on, in public mode, with a key set.

    Public mode with a key is the interesting shape: the three subscriber
    routes must stay anonymous while ``/test`` and ``/stats`` disappear.
    """
    minimal_config.push = PushConfig(enabled=True, vapid_subject=SUBJECT)
    minimal_config.server.public_mode = True
    minimal_config.server.api_key = API_KEY
    return minimal_config


@pytest.fixture
def seeded_engine(
    push_config: Config, geo: CompositeGeo, products: NationalProducts,
) -> CycleEngine:
    """Engine with synthetic geo + national products already 'computed'."""
    eng = CycleEngine(push_config)
    eng._basemap_attempted = True  # never fetch OSM
    eng._geo = geo
    eng._national_latest = (products, RADAR_TS)
    return eng


@pytest.fixture
def client(push_config: Config, seeded_engine: CycleEngine):
    app = create_app(push_config, engine=seeded_engine, auto_start_scheduler=False)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def bare_client(push_config: Config):
    """Push enabled, but no cycle has run — no geo, no products."""
    app = create_app(push_config, auto_start_scheduler=False)
    with TestClient(app) as c:
        yield c


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"}


def _sub_body(endpoint: str = ENDPOINT_A, **over) -> dict:
    body = {
        "subscription": {
            "endpoint": endpoint,
            "keys": {"p256dh": P256DH, "auth": AUTH},
            "expirationTime": None,
        },
        "lat": HOME_LAT,
        "lon": HOME_LON,
        "threshold_pct": 60,
        "lead_min": 30,
        "quiet_hours": {"enabled": False, "start": "22:00", "end": "07:00"},
        "tz": "Europe/Copenhagen",
        "lang": "da",
    }
    body.update(over)
    return body


def _new_sub(endpoint: str = ENDPOINT_A, **over) -> NewSubscription:
    kwargs = {
        "endpoint": endpoint,
        "p256dh": P256DH,
        "auth": AUTH,
        "lat": HOME_LAT,
        "lon": HOME_LON,
        "threshold_pct": 60,
        "lead_min": 30,
        "quiet_enabled": False,
        "quiet_start": "22:00",
        "quiet_end": "07:00",
        "tz": "Europe/Copenhagen",
        "lang": "da",
    }
    kwargs.update(over)
    return NewSubscription(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def test_store_creates_schema_and_parent(tmp_path: Path) -> None:
    db = tmp_path / "nested" / "dir" / "subs.sqlite"
    store = PushStore(db)
    assert db.parent.is_dir()
    assert store.count() == 0
    assert store.stats() == {"subscriptions": 0, "armed": 0}
    store.close()


def test_store_upsert_creates_then_updates(tmp_path: Path) -> None:
    store = PushStore(tmp_path / "subs.sqlite")
    assert store.upsert(_new_sub()) is True
    first = store.get(ENDPOINT_A)
    assert first is not None
    assert (first.lat, first.lon) == (HOME_LAT, HOME_LON)
    assert first.armed is True and first.streak == 0
    assert first.below_since_utc is None and first.last_eval_radar_ts is None
    assert first.created_utc.tzinfo is not None
    assert first.quiet_enabled is False
    assert first.lang == "da"

    # Dirty the state machine, then re-subscribe with different prefs.
    store.update_state(
        ENDPOINT_A,
        armed=False,
        streak=3,
        below_since_utc=RADAR_TS,
        last_eval_radar_ts=RADAR_TS,
        last_notified_utc=RADAR_TS,
    )
    assert store.upsert(_new_sub(threshold_pct=80, lead_min=45)) is False

    second = store.get(ENDPOINT_A)
    assert second is not None
    assert (second.threshold_pct, second.lead_min) == (80, 45)
    # Editing preferences restarts the machine...
    assert second.armed is True
    assert second.streak == 0
    assert second.below_since_utc is None
    assert second.last_eval_radar_ts is None
    # ...but keeps the history that is not the machine's.
    assert second.created_utc == first.created_utc
    assert second.last_notified_utc == RADAR_TS
    assert store.count() == 1
    store.close()


def test_store_update_state_roundtrips_aware_datetimes(tmp_path: Path) -> None:
    store = PushStore(tmp_path / "subs.sqlite")
    store.upsert(_new_sub())
    store.update_state(
        ENDPOINT_A,
        armed=False,
        streak=2,
        below_since_utc=RADAR_TS,
        last_eval_radar_ts=RADAR_TS2,
    )
    row = store.get(ENDPOINT_A)
    assert row is not None
    assert row.armed is False and row.streak == 2
    assert row.below_since_utc == RADAR_TS
    assert row.below_since_utc.tzinfo is not None  # type: ignore[union-attr]
    assert row.last_eval_radar_ts == RADAR_TS2
    # Not passed → untouched, and None is a real value here.
    assert row.last_notified_utc is None
    store.update_state(
        ENDPOINT_A,
        armed=True,
        streak=0,
        below_since_utc=None,
        last_eval_radar_ts=RADAR_TS2,
        last_notified_utc=RADAR_TS2,
    )
    assert store.get(ENDPOINT_A).last_notified_utc == RADAR_TS2  # type: ignore[union-attr]
    store.close()


def test_store_naive_datetime_is_treated_as_utc(tmp_path: Path) -> None:
    """Nothing should hand us a naive datetime, but if something does it
    must not come back as local time."""
    store = PushStore(tmp_path / "subs.sqlite")
    store.upsert(_new_sub())
    store.update_state(
        ENDPOINT_A,
        armed=True,
        streak=0,
        below_since_utc=datetime(2026, 8, 28, 12, 0),  # naive
        last_eval_radar_ts=None,
    )
    assert store.get(ENDPOINT_A).below_since_utc == RADAR_TS  # type: ignore[union-attr]
    store.close()


def test_store_list_delete_and_stats(tmp_path: Path) -> None:
    store = PushStore(tmp_path / "subs.sqlite")
    store.upsert(_new_sub(ENDPOINT_A))
    store.upsert(_new_sub(ENDPOINT_B))
    assert store.count() == 2
    assert {s.endpoint for s in store.list()} == {ENDPOINT_A, ENDPOINT_B}
    assert store.stats() == {"subscriptions": 2, "armed": 2}

    store.update_state(
        ENDPOINT_B, armed=False, streak=0,
        below_since_utc=None, last_eval_radar_ts=None,
    )
    assert store.stats() == {"subscriptions": 2, "armed": 1}

    assert store.delete(ENDPOINT_B) is True
    assert store.delete(ENDPOINT_B) is False
    assert store.count() == 1
    assert store.get(ENDPOINT_B) is None
    store.close()


def test_store_is_thread_safe(tmp_path: Path) -> None:
    """Routes touch the store from the event-loop thread while the cycle
    evaluation touches it from an ``asyncio.to_thread`` worker."""
    store = PushStore(tmp_path / "subs.sqlite")
    errors: list[Exception] = []

    def _hammer(prefix: str) -> None:
        try:
            for i in range(40):
                endpoint = f"https://fcm.googleapis.com/fcm/send/{prefix}{i}"
                store.upsert(_new_sub(endpoint))
                store.update_state(
                    endpoint, armed=False, streak=i,
                    below_since_utc=RADAR_TS, last_eval_radar_ts=RADAR_TS,
                )
                store.count()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=_hammer, args=("a",)),
        threading.Thread(target=_hammer, args=("b",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert store.count() == 80
    store.close()


def test_sub_id_is_short_stable_and_not_the_endpoint() -> None:
    handle = sub_id(ENDPOINT_A)
    assert len(handle) == 10
    assert handle == sub_id(ENDPOINT_A)
    assert handle != sub_id(ENDPOINT_B)
    assert handle not in ENDPOINT_A


# ---------------------------------------------------------------------------
# VAPID keys
# ---------------------------------------------------------------------------

def test_ensure_private_key_creates_0600_then_reuses(tmp_path: Path) -> None:
    path = tmp_path / "push" / "vapid_private.pem"
    pem = vapid.ensure_private_key(path)
    assert path.is_file()
    assert pem.startswith(PEM_HEADER)
    assert (path.stat().st_mode & 0o777) == 0o600
    # Second call must not rotate the key — that would invalidate every
    # stored subscription on every restart.
    assert vapid.ensure_private_key(path) == pem


def test_public_key_is_raw_uncompressed_point_b64url() -> None:
    import base64

    pem = vapid.generate_private_key_pem()
    encoded = vapid.public_key_b64url(pem)
    assert "=" not in encoded
    assert "+" not in encoded and "/" not in encoded
    raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    assert len(raw) == 65
    assert raw[0] == 0x04
    # Deterministic for a given key.
    assert vapid.public_key_b64url(pem) == encoded


def test_generated_pem_loads_in_pywebpush_signer() -> None:
    """The PEM this writes must be usable by the actual sender."""
    import pywebpush

    signer = pywebpush.Vapid.from_pem(vapid.generate_private_key_pem())
    assert signer.public_key is not None


def test_keygen_cli_writes_and_refuses_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    path = tmp_path / "keys" / "vapid_private.pem"
    assert keygen.main([str(path)]) == 0
    original = path.read_bytes()
    out = capsys.readouterr().out
    assert str(path) in out
    assert vapid.public_key_b64url(original) in out
    assert (path.stat().st_mode & 0o777) == 0o600

    assert keygen.main([str(path)]) == 1
    assert path.read_bytes() == original
    assert "refusing to overwrite" in capsys.readouterr().err

    assert keygen.main([str(path), "--force"]) == 0
    assert path.read_bytes() != original


# ---------------------------------------------------------------------------
# Endpoint policy (the SSRF guard)
# ---------------------------------------------------------------------------

ALLOWED = [
    "fcm.googleapis.com",
    "updates.push.services.mozilla.com",
    "push.services.mozilla.com",
    "web.push.apple.com",
    "notify.windows.com",
]


@pytest.mark.parametrize("url", [
    "https://fcm.googleapis.com/fcm/send/abc",
    "https://updates.push.services.mozilla.com/wpush/v2/abc",
    "https://web.push.apple.com/QAbc123",
    "https://notify.windows.com/w/?token=abc",
    # A subdomain of an allowed suffix is fine.
    "https://eu.fcm.googleapis.com/fcm/send/abc",
    # Trailing-dot FQDN and an uppercase host normalise.
    "https://FCM.googleapis.com./fcm/send/abc",
])
def test_endpoint_policy_accepts_vendor_hosts(url: str) -> None:
    assert validate_endpoint(url, ALLOWED) is None


@pytest.mark.parametrize("url", [
    "http://fcm.googleapis.com/fcm/send/abc",          # not https
    "https://192.168.1.10/fcm/send/abc",               # LAN IPv4 literal
    "https://[::1]/fcm/send/abc",                      # bracketed IPv6
    "https://127.0.0.1/fcm/send/abc",
    "https://169.254.169.254/latest/meta-data/",        # cloud metadata
    "https://localhost/fcm/send/abc",
    "https://evil.example/fcm/send/abc",
    "https://fcm.googleapis.com.evil.example/fcm/send/abc",  # suffix spoof
    "https://xfcm.googleapis.com/fcm/send/abc",         # no dot boundary
    "https://user:pass@fcm.googleapis.com/fcm/send/abc",
    "https://fcm.googleapis.com:8443/fcm/send/abc",
    "file:///etc/passwd",
    "",
    "   ",
])
def test_endpoint_policy_rejects(url: str) -> None:
    reason = validate_endpoint(url, ALLOWED)
    assert isinstance(reason, str) and reason


def test_endpoint_policy_reason_is_readable() -> None:
    assert validate_endpoint("http://fcm.googleapis.com/x", ALLOWED) == (
        "endpoint must use https"
    )
    assert "IP address" in str(
        validate_endpoint("https://192.168.1.10/x", ALLOWED),
    )


# ---------------------------------------------------------------------------
# Routes — push enabled, public mode
# ---------------------------------------------------------------------------

def test_config_route_is_anonymous_and_lists_options(client: TestClient) -> None:
    r = client.get("/api/push/config")
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-store"
    body = r.json()
    assert body["enabled"] is True
    assert body["threshold_options_pct"] == [40, 60, 80]
    # national leads [10, 20, 30, 45, 60] filtered by min_lead_min = 20.
    assert body["lead_options_min"] == [20, 30, 45, 60]
    assert body["defaults"] == {
        "threshold_pct": 60,
        "lead_min": 30,
        "quiet_hours": {"enabled": False, "start": "22:00", "end": "07:00"},
    }
    assert body["capacity_reached"] is False
    # The advertised key is the one on disk, in browser form.
    import base64
    raw = base64.urlsafe_b64decode(
        body["vapid_public_key"] + "=" * (-len(body["vapid_public_key"]) % 4),
    )
    assert len(raw) == 65 and raw[0] == 0x04


def test_subscribe_happy_path_writes_the_row(
    client: TestClient, push_config: Config,
) -> None:
    r = client.post("/api/push/subscribe", json=_sub_body())
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "created": True}
    assert r.headers["cache-control"] == "no-store"

    store = PushStore(resolved_db_path(push_config))
    row = store.get(ENDPOINT_A)
    assert row is not None
    assert (row.lat, row.lon) == (HOME_LAT, HOME_LON)
    assert row.threshold_pct == 60 and row.lead_min == 30
    assert row.tz == "Europe/Copenhagen" and row.lang == "da"
    assert row.quiet_enabled is False
    assert row.armed is True and row.streak == 0
    store.close()


def test_resubscribe_updates_prefs_and_resets_state(
    client: TestClient, push_config: Config,
) -> None:
    assert client.post("/api/push/subscribe", json=_sub_body()).json()["created"]
    store = PushStore(resolved_db_path(push_config))
    store.update_state(
        ENDPOINT_A, armed=False, streak=2,
        below_since_utc=RADAR_TS, last_eval_radar_ts=RADAR_TS,
    )
    store.close()

    r = client.post(
        "/api/push/subscribe",
        json=_sub_body(threshold_pct=80, lead_min=45, lang="en"),
    )
    assert r.json() == {"ok": True, "created": False}

    store = PushStore(resolved_db_path(push_config))
    row = store.get(ENDPOINT_A)
    assert row is not None
    assert (row.threshold_pct, row.lead_min, row.lang) == (80, 45, "en")
    assert row.armed is True and row.streak == 0
    assert row.below_since_utc is None and row.last_eval_radar_ts is None
    assert store.count() == 1
    store.close()


@pytest.mark.parametrize("body,expected", [
    (_sub_body(tz="Mars/Olympus"), 400),
    (_sub_body(lead_min=15), 400),          # below min_lead_min
    (_sub_body(lead_min=10), 400),          # a national lead, but not offered
    (_sub_body(threshold_pct=50), 400),     # not an offered option
    (_sub_body(endpoint="https://evil.example/fcm/send/x"), 400),
    (_sub_body(endpoint="http://fcm.googleapis.com/fcm/send/x"), 400),
    (_sub_body(lang="de"), 422),
    (_sub_body(lat=95.0), 422),
    (_sub_body(quiet_hours={"enabled": True, "start": "25:00", "end": "07:00"}), 422),
])
def test_subscribe_validation_failures_write_nothing(
    client: TestClient, push_config: Config, body: dict, expected: int,
) -> None:
    r = client.post("/api/push/subscribe", json=body)
    assert r.status_code == expected, r.text
    if expected == 400:
        assert isinstance(r.json()["detail"], str)
    store = PushStore(resolved_db_path(push_config))
    assert store.count() == 0
    store.close()


def test_subscribe_rejects_oversize_keys_and_endpoint(
    client: TestClient, push_config: Config,
) -> None:
    for over in (
        {"subscription": {
            "endpoint": ENDPOINT_A,
            "keys": {"p256dh": "x" * 513, "auth": AUTH},
        }},
        {"subscription": {
            "endpoint": ENDPOINT_A,
            "keys": {"p256dh": P256DH, "auth": "y" * 513},
        }},
        {"subscription": {
            "endpoint": "https://fcm.googleapis.com/fcm/send/" + "z" * 2100,
            "keys": {"p256dh": P256DH, "auth": AUTH},
        }},
    ):
        r = client.post("/api/push/subscribe", json=_sub_body(**over))
        assert r.status_code == 422, r.text
    store = PushStore(resolved_db_path(push_config))
    assert store.count() == 0
    store.close()


def test_subscribe_off_coverage_point_is_400(
    client: TestClient, push_config: Config,
) -> None:
    """Same wording /forecast uses — a point that can never fire is refused
    at subscribe time rather than silently never notifying."""
    r = client.post("/api/push/subscribe", json=_sub_body(lat=45.0, lon=-30.0))
    assert r.status_code == 400
    assert r.json()["detail"] == "coordinates outside the radar composite grid"
    store = PushStore(resolved_db_path(push_config))
    assert store.count() == 0
    store.close()


def test_subscribe_accepted_before_the_first_cycle(bare_client: TestClient) -> None:
    """No grids yet → nothing to check the point against → accept."""
    r = bare_client.post("/api/push/subscribe", json=_sub_body(lat=45.0, lon=-30.0))
    assert r.status_code == 200
    assert r.json()["created"] is True


def test_subscribe_capacity_is_503_but_existing_rows_still_update(
    push_config: Config, seeded_engine: CycleEngine,
) -> None:
    push_config.push.max_subscriptions = 1
    app = create_app(push_config, engine=seeded_engine, auto_start_scheduler=False)
    with TestClient(app) as c:
        assert c.post("/api/push/subscribe", json=_sub_body(ENDPOINT_A)).status_code == 200
        r = c.post("/api/push/subscribe", json=_sub_body(ENDPOINT_B))
        assert r.status_code == 503
        assert "capacity" in r.json()["detail"]
        # An endpoint already stored may still edit its preferences.
        r = c.post("/api/push/subscribe", json=_sub_body(ENDPOINT_A, threshold_pct=80))
        assert r.status_code == 200
        assert r.json()["created"] is False
        assert c.get("/api/push/config").json()["capacity_reached"] is True


def test_unsubscribe_is_idempotent(client: TestClient) -> None:
    client.post("/api/push/subscribe", json=_sub_body())
    r = client.post("/api/push/unsubscribe", json={"endpoint": ENDPOINT_A})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "deleted": True}
    r = client.post("/api/push/unsubscribe", json={"endpoint": ENDPOINT_A})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "deleted": False}


def test_operator_routes_are_404_without_bearer_and_work_with_it(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def _fake_send(**kwargs) -> SendResult:
        calls.append(kwargs["endpoint"])
        return SendResult(ok=True, gone=False, status=201, error=None)

    monkeypatch.setattr(fanout_mod, "send", _fake_send)
    client.post("/api/push/subscribe", json=_sub_body())

    # Hidden: byte-identical to a route that does not exist.
    for method, url in (("POST", "/api/push/test"), ("GET", "/api/push/stats")):
        r = client.request(method, url, json={})
        assert r.status_code == 404, f"{method} {url} leaked {r.status_code}"
        assert r.json() == {"detail": "Not Found"}
    assert client.post(
        "/api/push/test", json={}, headers={"Authorization": "Bearer wrong"},
    ).status_code == 404

    r = client.post("/api/push/test", json={}, headers=_auth())
    assert r.status_code == 200
    assert r.json() == {"sent": 1, "failed": 0, "removed": 0}
    assert calls == [ENDPOINT_A]

    r = client.get("/api/push/stats", headers=_auth())
    assert r.status_code == 200
    body = r.json()
    assert body["subscriptions"] == 1 and body["armed"] == 1
    assert body["last_evaluated_radar_ts"] is None
    assert body["last_fanout"] is None
    # Never leak an endpoint through the operator view either.
    assert ENDPOINT_A not in r.text


def test_disabled_push_in_default_mode(minimal_config: Config) -> None:
    """LAN instance, feature off: the routes exist and say so."""
    assert minimal_config.push.enabled is False
    app = create_app(minimal_config, auto_start_scheduler=False)
    with TestClient(app) as c:
        r = c.get("/api/push/config")
        assert r.status_code == 200
        assert r.json() == {"enabled": False}
        assert c.post("/api/push/subscribe", json=_sub_body()).status_code == 503
        assert c.post(
            "/api/push/unsubscribe", json={"endpoint": ENDPOINT_A},
        ).status_code == 503
        assert c.post("/api/push/test", json={}).status_code == 503
        assert c.get("/api/push/stats").status_code == 503
    # Nothing was created on disk for a disabled feature.
    assert not resolved_key_path(minimal_config).exists()
    assert not resolved_db_path(minimal_config).exists()


# ---------------------------------------------------------------------------
# Path safety — the key and the DB live under data_dir, which routes read
# ---------------------------------------------------------------------------

_TRAVERSALS = [
    "/nowcast/../push/subscriptions.sqlite",
    "/nowcast/..%2Fpush%2Fsubscriptions.sqlite",
    "/nowcast/push/subscriptions.sqlite",
    "/nowcast/subscriptions.sqlite",
    "/nowcast/../push/vapid_private.pem",
    "/nowcast/..%2Fpush%2Fvapid_private.pem",
    "/nowcast/push/vapid_private.pem",
    "/nowcast/vapid_private.pem",
    "/frames/../push/vapid_private.pem",
    "/frames/..%2Fpush%2Fvapid_private.pem",
    "/frames/../push/subscriptions.sqlite",
    "/frames/push/vapid_private.pem",
    "/push/subscriptions.sqlite",
    "/push/vapid_private.pem",
    "/api/push/subscriptions.sqlite",
    "/api/push/vapid_private.pem",
]


def _assert_no_secret_leak(client: TestClient, paths: list[str]) -> None:
    for path in paths:
        r = client.get(path, headers=_auth())
        assert r.status_code in (400, 404), f"{path} answered {r.status_code}"
        assert PEM_HEADER not in r.content, path
        assert SQLITE_MAGIC not in r.content, path
        assert b"BEGIN PRIVATE KEY" not in r.content, path


def test_push_secrets_are_not_reachable_over_http(
    client: TestClient, push_config: Config,
) -> None:
    """The files really are on disk under data_dir before we probe."""
    client.post("/api/push/subscribe", json=_sub_body())
    key_path = resolved_key_path(push_config)
    db_path = resolved_db_path(push_config)
    assert key_path.is_file() and key_path.read_bytes().startswith(PEM_HEADER)
    assert db_path.is_file()
    assert key_path.parent == push_config.storage.data_dir / "push"
    # The row is really in that file, so a leak would be a real leak.
    with sqlite3.connect(str(db_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0] == 1

    _assert_no_secret_leak(client, _TRAVERSALS)


def test_push_secrets_are_not_reachable_with_the_frontend_mounted(
    push_config: Config, seeded_engine: CycleEngine, tmp_path: Path,
) -> None:
    """The SPA mount is a catch-all at ``/`` — it must not become a file
    server for the data volume either."""
    root = tmp_path / "frontend_build"
    root.mkdir()
    (root / "index.html").write_text("<!doctype html><div id=app></div>")
    push_config.server.frontend_dir = root
    app = create_app(push_config, engine=seeded_engine, auto_start_scheduler=False)
    with TestClient(app) as c:
        c.post("/api/push/subscribe", json=_sub_body())
        assert resolved_key_path(push_config).is_file()
        _assert_no_secret_leak(c, _TRAVERSALS)
        # And the SPA fallback does not turn a secret path into the shell.
        assert c.get("/push/subscriptions.sqlite").status_code == 404


@pytest.mark.parametrize("name", [
    "../push/vapid_private.pem",
    "..%2Fpush%2Fvapid_private.pem",
    "push/subscriptions.sqlite",
    "subscriptions.sqlite",
    "vapid_private.pem",
    "..",
    "../../etc/passwd",
    "p_rain_10min_202608281200.png/../vapid_private.pem",
])
def test_name_guards_reject_traversal_and_secret_suffixes(name: str) -> None:
    assert _safe_nowcast_name(name) is False
    assert _safe_frame_name(name) is False


def test_name_guards_still_accept_the_real_artifacts() -> None:
    assert _safe_nowcast_name("p_rain_10min_202608281200.png") is True
    assert _safe_nowcast_name("manifest_202608281200.json") is True
    assert _safe_frame_name("frame_00.png") is True
    assert _safe_frame_name("loop.png") is True


# ---------------------------------------------------------------------------
# Service — per-cycle evaluation and fan-out
# ---------------------------------------------------------------------------

def _state_with(radar_ts: datetime):
    """A minimal valid State carrying one radar timestamp."""
    from dmi_nowcast_sidecar.state_schema import (
        CalibrationBlock,
        DiagnosticsBlock,
        ForecastBlock,
        HomeBlock,
        MotionBlock,
        NowBlock,
        RadarBlock,
        State,
    )

    return State(
        generated_at=radar_ts,
        radar=RadarBlock(latest_ts=radar_ts, data_age_minutes=2.0),
        home=HomeBlock(lat=HOME_LAT, lon=HOME_LON, radius_km=1.0),
        now=NowBlock(
            rain_rate_mm_h=0.0, rain_rate_p90_mm_h=0.0,
            raining=False, raining_hysteresis_state="dry",
        ),
        forecast=ForecastBlock(
            method="farneback", rain_incoming=False, eta_minutes=None,
            eta_p50_window_min=None, peak_intensity_mm_h=0.0,
            peak_lead_min=0, per_lead=[],
        ),
        motion=MotionBlock(
            dy_px_per_min=0.0, dx_px_per_min=0.0,
            speed_km_per_h=0.0, bearing_deg_from=0.0,
        ),
        confidence=0.5,
        calibration=CalibrationBlock(
            fitted_at=None, n_events=None, brier_before=None, brier_after=None,
        ),
        diagnostics=DiagnosticsBlock(
            cycle_ms=0.0, fetch_ms=0.0, compute_ms=0.0, render_ms=0.0,
        ),
    )


@pytest.fixture
def service(
    push_config: Config, seeded_engine: CycleEngine, geo: CompositeGeo,
) -> PushService:
    """Two subscriptions: one on the wet centre pixel, one on the NaN pixel."""
    store = PushStore(resolved_db_path(push_config))
    nan_lon, nan_lat = geo.grid_to_lonlat(
        NAN_PIXEL[0] * DOWNSAMPLE, NAN_PIXEL[1] * DOWNSAMPLE,
    )
    store.upsert(_new_sub(ENDPOINT_A))                       # wet
    store.upsert(_new_sub(ENDPOINT_B, lat=nan_lat, lon=nan_lon))  # nodata
    return PushService(
        push_config, seeded_engine, store,
        vapid.generate_private_key_pem(), "public-key",
    )


@pytest.fixture
def sends(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Record every delivery instead of making one."""
    calls: list[dict] = []

    def _fake_send(**kwargs) -> SendResult:
        calls.append(kwargs)
        return SendResult(ok=True, gone=False, status=201, error=None)

    monkeypatch.setattr(fanout_mod, "send", _fake_send)
    return calls


async def test_service_persistence_then_one_notification(
    service: PushService, seeded_engine: CycleEngine, sends: list[dict],
) -> None:
    # First observation: over threshold, but persistence is 2.
    await service.after_cycle(CycleResult(state=_state_with(RADAR_TS)))
    assert sends == []
    wet = service.store.get(ENDPOINT_A)
    dry = service.store.get(ENDPOINT_B)
    assert wet is not None and dry is not None
    assert wet.streak == 1 and wet.armed is True
    assert wet.last_eval_radar_ts == RADAR_TS
    assert wet.last_notified_utc is None
    # The NaN pixel is nodata, never "dry with a streak".
    assert dry.streak == 0 and dry.armed is True
    assert service.last_evaluated_radar_ts == RADAR_TS

    # Second observation, new radar frame: fires, once, for the wet point.
    seeded_engine._national_latest = (
        seeded_engine.national_latest[0], RADAR_TS2,  # type: ignore[index]
    )
    await service.after_cycle(CycleResult(state=_state_with(RADAR_TS2)))
    assert len(sends) == 1
    assert sends[0]["endpoint"] == ENDPOINT_A
    assert sends[0]["ttl_s"] == service.config.push.ttl_s
    assert sends[0]["vapid_subject"] == SUBJECT
    payload = sends[0]["payload"]
    assert payload["type"] == "rain_incoming"
    assert payload["lang"] == "da"
    assert payload["p_pct"] == 90
    assert payload["lead_min"] == 30

    wet = service.store.get(ENDPOINT_A)
    assert wet is not None
    assert wet.armed is False            # a push disarms the subscription
    assert wet.streak == 2
    assert wet.last_notified_utc is not None
    assert wet.last_eval_radar_ts == RADAR_TS2
    assert service.last_fanout is not None
    assert service.last_fanout["sent"] == 1
    assert service.last_fanout["notified"] == 1
    assert service.last_fanout["subscriptions"] == 2


async def test_service_does_not_push_into_rain_already_at_the_point(
    service: PushService,
    seeded_engine: CycleEngine,
    products: NationalProducts,
    sends: list[dict],
) -> None:
    """The live bug: the ETA grid says 6 min (a fresh arrival), but the
    radar measures 0.9 mm/h falling on the point this very frame. The
    subscription is consumed silently instead of being told rain is
    incoming while it rains."""
    observed = _grid(0.9)  # >= forecast.rain_threshold_mm_h (0.5)
    seeded_engine._national_latest = NationalSnapshot(
        products, RADAR_TS, observed,
    )
    await service.after_cycle(CycleResult(state=_state_with(RADAR_TS)))
    seeded_engine._national_latest = NationalSnapshot(
        products, RADAR_TS2, observed,
    )
    await service.after_cycle(CycleResult(state=_state_with(RADAR_TS2)))

    assert sends == []
    assert service.last_fanout is not None
    assert service.last_fanout["notified"] == 0
    assert service.last_fanout["actions"]["already_raining"] == 1
    wet = service.store.get(ENDPOINT_A)
    assert wet is not None
    assert wet.armed is False           # the arm is consumed, as if pushed
    assert wet.last_notified_utc is None


async def test_service_still_pushes_when_the_point_is_dry_now(
    service: PushService,
    seeded_engine: CycleEngine,
    products: NationalProducts,
    sends: list[dict],
) -> None:
    """An observed grid that says 0 mm/h at the point is the ordinary
    "rain incoming" case — the new rule must not silence it."""
    observed = _grid(0.0)
    seeded_engine._national_latest = NationalSnapshot(
        products, RADAR_TS, observed,
    )
    await service.after_cycle(CycleResult(state=_state_with(RADAR_TS)))
    seeded_engine._national_latest = NationalSnapshot(
        products, RADAR_TS2, observed,
    )
    await service.after_cycle(CycleResult(state=_state_with(RADAR_TS2)))

    assert len(sends) == 1
    assert sends[0]["endpoint"] == ENDPOINT_A


def test_service_rules_take_the_pipeline_detection_threshold(
    service: PushService,
) -> None:
    """One threshold for the whole pipeline: what counts as rain falling
    here is what counts as rain in Home Assistant's ``raining_now`` and in
    the ensemble exceedance."""
    service.config.forecast.rain_threshold_mm_h = 1.25
    assert service._rules().raining_now_mm_h == pytest.approx(1.25)


async def test_service_ignores_a_repeated_radar_timestamp(
    service: PushService, seeded_engine: CycleEngine, sends: list[dict],
) -> None:
    """The no-new-frame fast path re-emits the same state; evaluating it
    would double-count the streak and fire an observation early."""
    await service.after_cycle(CycleResult(state=_state_with(RADAR_TS)))
    assert service.store.get(ENDPOINT_A).streak == 1  # type: ignore[union-attr]

    await service.after_cycle(CycleResult(state=_state_with(RADAR_TS)))
    assert sends == []
    assert service.store.get(ENDPOINT_A).streak == 1  # type: ignore[union-attr]


async def test_service_skips_when_products_belong_to_another_frame(
    service: PushService, seeded_engine: CycleEngine, sends: list[dict],
) -> None:
    """Held grids stamped RADAR_TS, state stamped RADAR_TS2: skip, and do
    NOT advance the marker — the real frame must still be evaluated."""
    await service.after_cycle(CycleResult(state=_state_with(RADAR_TS2)))
    assert sends == []
    assert service.last_evaluated_radar_ts is None
    row = service.store.get(ENDPOINT_A)
    assert row is not None
    assert row.streak == 0 and row.last_eval_radar_ts is None

    # The matching frame arrives; the observation is not lost.
    seeded_engine._national_latest = (
        seeded_engine.national_latest[0], RADAR_TS2,  # type: ignore[index]
    )
    await service.after_cycle(CycleResult(state=_state_with(RADAR_TS2)))
    assert service.store.get(ENDPOINT_A).streak == 1  # type: ignore[union-attr]


async def test_service_no_state_and_no_products_are_no_ops(
    service: PushService, seeded_engine: CycleEngine, sends: list[dict],
) -> None:
    await service.after_cycle(CycleResult(state=None, error="fetch failed"))
    assert service.last_evaluated_radar_ts is None

    seeded_engine._national_latest = None
    await service.after_cycle(CycleResult(state=_state_with(RADAR_TS)))
    assert service.last_evaluated_radar_ts is None
    assert sends == []


async def test_service_deletes_a_gone_subscription(
    service: PushService, seeded_engine: CycleEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _gone(**kwargs) -> SendResult:
        return SendResult(ok=False, gone=True, status=410, error="gone")

    monkeypatch.setattr(fanout_mod, "send", _gone)
    await service.after_cycle(CycleResult(state=_state_with(RADAR_TS)))
    seeded_engine._national_latest = (
        seeded_engine.national_latest[0], RADAR_TS2,  # type: ignore[index]
    )
    await service.after_cycle(CycleResult(state=_state_with(RADAR_TS2)))

    assert service.store.get(ENDPOINT_A) is None      # row removed
    assert service.store.get(ENDPOINT_B) is not None  # untouched
    assert service.last_fanout["removed"] == 1  # type: ignore[index]
    assert service.last_fanout["sent"] == 0     # type: ignore[index]


async def test_service_survives_a_raising_send(
    service: PushService, seeded_engine: CycleEngine, geo: CompositeGeo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One dead push service must not cost the other subscriber their
    notification, and must not propagate out of the cycle hook."""
    # Both subscriptions on the wet pixel this time, so both should fire.
    service.store.upsert(_new_sub(ENDPOINT_B))
    delivered: list[str] = []

    def _flaky(**kwargs) -> SendResult:
        if kwargs["endpoint"] == ENDPOINT_A:
            raise RuntimeError("connection reset")
        delivered.append(kwargs["endpoint"])
        return SendResult(ok=True, gone=False, status=201, error=None)

    monkeypatch.setattr(fanout_mod, "send", _flaky)
    await service.after_cycle(CycleResult(state=_state_with(RADAR_TS)))
    seeded_engine._national_latest = (
        seeded_engine.national_latest[0], RADAR_TS2,  # type: ignore[index]
    )
    await service.after_cycle(CycleResult(state=_state_with(RADAR_TS2)))

    assert delivered == [ENDPOINT_B]
    assert service.last_fanout["sent"] == 1     # type: ignore[index]
    assert service.last_fanout["failed"] == 1   # type: ignore[index]
    # A failed send still disarmed the row: the state was persisted first.
    assert service.store.get(ENDPOINT_A).armed is False  # type: ignore[union-attr]


async def test_send_test_reaches_every_row_and_one_row(
    service: PushService, sends: list[dict],
) -> None:
    counts = await service.send_test(None)
    assert counts == {"sent": 2, "failed": 0, "removed": 0}
    assert {c["endpoint"] for c in sends} == {ENDPOINT_A, ENDPOINT_B}
    assert sends[0]["payload"]["type"] == "test"

    sends.clear()
    counts = await service.send_test(ENDPOINT_B)
    assert counts == {"sent": 1, "failed": 0, "removed": 0}
    assert [c["endpoint"] for c in sends] == [ENDPOINT_B]

    sends.clear()
    counts = await service.send_test("https://fcm.googleapis.com/fcm/send/nope")
    assert counts == {"sent": 0, "failed": 0, "removed": 0}
    assert sends == []


async def test_fanout_budget_stops_the_cycle_being_held_hostage(
    service: PushService, seeded_engine: CycleEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero budget drops everything queued and says so — the cycle must
    never wait on a slow push service."""
    service.config.push.fanout_budget_s = 1e-9
    monkeypatch.setattr(
        fanout_mod, "send",
        lambda **kw: SendResult(ok=True, gone=False, status=201, error=None),
    )
    await service.after_cycle(CycleResult(state=_state_with(RADAR_TS)))
    seeded_engine._national_latest = (
        seeded_engine.national_latest[0], RADAR_TS2,  # type: ignore[index]
    )
    await service.after_cycle(CycleResult(state=_state_with(RADAR_TS2)))
    assert service.last_fanout["skipped"] == 1  # type: ignore[index]
    assert service.last_fanout["sent"] == 0     # type: ignore[index]
