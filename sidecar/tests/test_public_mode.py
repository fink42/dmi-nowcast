"""P1 — public service mode (website Phase C plan §P1).

Three things are under test, and they are the three ways this package can
fail in production:

1. **The gate.** In ``server.public_mode`` only ``/healthz``,
   ``/nowcast/*``, ``/forecast`` and the static frontend answer; every
   other registered route (``/state.json``, ``/frames/*``,
   ``/lightning/*``, the archive dashboards, ``/docs``) returns a 404 that
   is byte-identical to a nonexistent path's, unless a valid bearer is
   presented. The default mode is asserted *unchanged* with exact
   assertions, because the LAN instance ships from the same code.
2. **The static frontend.** SPA fallback, the cache-header table, and —
   critically — HTTP Range support, which the Protomaps ``.pmtiles``
   basemap depends on (MapLibre never downloads the whole file). Range is
   handled natively by the installed Starlette's ``FileResponse``; the
   test pins that so an upgrade can't silently break the map.
3. **The compute saving.** ``render_frames`` and the OSM basemap fetch
   must not run in public mode: they only feed ``/frames/*``, which is
   hidden. Proven with stubs that record a call and then raise — the raise
   alone would be swallowed by the cycle's render-failure policy, so the
   recording is what makes the assertion honest.

Fully synthetic: the ODIM composite writer is the same 64×64 @ 500 m
helper ``test_app_nowcast.py`` / ``test_compute_ensemble.py`` use (copied
per this suite's convention, so each module stands alone).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pyproj import CRS, Transformer

from dmi_nowcast_sidecar import compute as compute_mod
from dmi_nowcast_sidecar.app import create_app
from dmi_nowcast_sidecar.compute import CycleEngine
from dmi_nowcast_sidecar.config import Config

API_KEY = "public-mode-secret"
HOME_LON, HOME_LAT = 10.32, 55.33
STAMP = "202608281200"

# Tiny valid 1x1 transparent PNG (same bytes the other app tests use).
PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c63000000000200013e9a3a200000000049454e44ae426082"
)

# Every route the gate must hide, as (method, url). Kept as one list so a
# new private route is one line away from being covered in both directions.
HIDDEN_ROUTES: list[tuple[str, str]] = [
    ("GET", "/state.json"),
    ("GET", "/frames/manifest.json"),
    ("GET", "/frames/frame_00.png"),
    ("GET", "/frames/loop.png"),
    ("GET", f"/lightning/eta?lat={HOME_LAT}&lon={HOME_LON}"),
    ("GET", f"/lightning/probability?lat={HOME_LAT}&lon={HOME_LON}"),
    ("GET", f"/lightning/clusters?lat={HOME_LAT}&lon={HOME_LON}"),
    ("GET", "/lightning/map.png"),
    ("GET", "/lightning/archive/summary"),
    ("GET", "/lightning/archive/dashboard.html"),
    ("POST", "/lightning/strikes"),
    # Web Push operator routes. Exact paths, deliberately not the whole
    # ``/api/push/`` prefix: /config, /subscribe and /unsubscribe are on
    # the public allow-list and must keep answering anonymously.
    ("POST", "/api/push/test"),
    ("GET", "/api/push/stats"),
    # The OpenAPI surface names every hidden route — hide it too.
    ("GET", "/docs"),
    ("GET", "/openapi.json"),
]


# ---------------------------------------------------------------------------
# Fixtures — configs, clients, seeded artifacts
# ---------------------------------------------------------------------------

@pytest.fixture
def public_config(minimal_config: Config) -> Config:
    """``minimal_config`` in public mode WITH a key set.

    A key set is the interesting case: it must not lock the public surface
    (anonymous browsers) while still unlocking the hidden one for an
    operator on the LAN.
    """
    minimal_config.server.public_mode = True
    minimal_config.server.api_key = API_KEY
    return minimal_config


@pytest.fixture
def public_client(public_config: Config):
    app = create_app(public_config, auto_start_scheduler=False)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def default_client(minimal_config: Config):
    app = create_app(minimal_config, auto_start_scheduler=False)
    with TestClient(app) as c:
        yield c


def _seed_nowcast(config: Config) -> dict:
    """A stamped artifact + stamped and stable manifests under nowcast/."""
    out = config.storage.data_dir / "nowcast"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "cycle": STAMP, "artifacts": []}
    payload = json.dumps(manifest).encode()
    (out / f"p_rain_10min_{STAMP}.png").write_bytes(PNG_BYTES)
    (out / f"manifest_{STAMP}.json").write_bytes(payload)
    (out / "manifest.json").write_bytes(payload)
    return manifest


def _seed_frames(config: Config) -> dict:
    """A frames.json + one frame PNG under frames/ (the home crop)."""
    frames_dir = config.storage.data_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    (frames_dir / "frame_00.png").write_bytes(PNG_BYTES)
    (frames_dir / "loop.png").write_bytes(PNG_BYTES)
    manifest = {
        "version": 1,
        "generated_at": "2026-08-28T12:00:00Z",
        "frame_count": 1,
        "frames": [{"index": 0, "filename": "frame_00.png"}],
    }
    (frames_dir / "frames.json").write_text(json.dumps(manifest))
    return manifest


def _seed_state(config: Config, *, eta_minutes: float = 12.0) -> None:
    """Persist a state.json — the block the public instance must not leak."""
    from dmi_nowcast_sidecar.state_schema import (
        CalibrationBlock,
        DiagnosticsBlock,
        ForecastBlock,
        HomeBlock,
        MotionBlock,
        NowBlock,
        PerLeadEntry,
        RadarBlock,
        State,
    )
    from dmi_nowcast_sidecar.storage import StateStore

    ts = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
    StateStore(config.storage.data_dir).write(State(
        generated_at=ts,
        radar=RadarBlock(latest_ts=ts, data_age_minutes=2.0),
        home=HomeBlock(lat=HOME_LAT, lon=HOME_LON, radius_km=1.0),
        now=NowBlock(
            rain_rate_mm_h=0.0, rain_rate_p90_mm_h=0.0,
            raining=False, raining_hysteresis_state="dry",
        ),
        forecast=ForecastBlock(
            method="farneback", rain_incoming=True, eta_minutes=eta_minutes,
            eta_p50_window_min=None, peak_intensity_mm_h=2.4, peak_lead_min=15,
            per_lead=[PerLeadEntry(
                lead_min=15, rain_rate_mm_h=2.4, p_rain=1.0, p_calibrated=0.78,
            )],
        ),
        motion=MotionBlock(
            dy_px_per_min=0.0, dx_px_per_min=0.0,
            speed_km_per_h=0.0, bearing_deg_from=0.0,
        ),
        confidence=0.71,
        calibration=CalibrationBlock(
            fitted_at=None, n_events=None, brier_before=None, brier_after=None,
        ),
        diagnostics=DiagnosticsBlock(
            cycle_ms=0.0, fetch_ms=0.0, compute_ms=0.0, render_ms=0.0,
        ),
    ))


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"}


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------

def test_public_mode_and_frontend_dir_default_off(minimal_config: Config) -> None:
    """Additive config: the LAN instance's behaviour is the default."""
    assert minimal_config.server.public_mode is False
    assert minimal_config.server.frontend_dir is None


def test_public_mode_settable_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The compose stack sets it through the env, like every other field."""
    monkeypatch.setenv("DMI_NOWCAST_SERVER__PUBLIC_MODE", "true")
    monkeypatch.setenv("DMI_NOWCAST_SERVER__FRONTEND_DIR", "/app/frontend/build")
    cfg = Config(home={"lat": HOME_LAT, "lon": HOME_LON})  # type: ignore[arg-type]
    assert cfg.server.public_mode is True
    assert cfg.server.frontend_dir == Path("/app/frontend/build")


def test_shipped_public_example_config_is_actually_public() -> None:
    """The committed public stack must not drift into leaking: parse the
    example config the compose file stages and assert its invariants."""
    from dmi_nowcast_sidecar.config import load_config

    path = (
        Path(__file__).resolve().parents[1]
        / "deploy" / "public" / "config.public.example.yaml"
    )
    cfg = load_config(path)
    assert cfg.server.public_mode is True
    assert cfg.server.frontend_dir == Path("/app/frontend/build")
    # Vestigial neutral reference point, no corpus, no lightning archive.
    assert (cfg.home.lat, cfg.home.lon) == (56.0, 10.0)
    assert cfg.storage.corpus_dir is None
    assert cfg.lightning.enabled is False
    assert cfg.lightning.archive_enabled is False
    # The national products are the site's payload — they must stay on.
    assert cfg.forecast.national.enabled is True
    # Web Push is the public instance's only outbound channel (Phase D).
    # Enabled, with a subject the operator is expected to replace — the
    # model rejects ``enabled`` without one, so this cannot silently ship
    # half-configured.
    assert cfg.push.enabled is True
    assert cfg.push.vapid_subject is not None
    assert cfg.push.vapid_subject.startswith(("mailto:", "https:"))
    # Key and DB default into the data volume, not a bind mount.
    assert cfg.push.vapid_private_key_file is None
    assert cfg.push.db_path is None


# ---------------------------------------------------------------------------
# Route matrix — public mode
# ---------------------------------------------------------------------------

def test_public_routes_open_without_bearer(
    public_config: Config, public_client: TestClient,
) -> None:
    """The published surface answers anonymously even with a key set."""
    # Before any cycle: honest 503s, never 401/404.
    assert public_client.get("/healthz").status_code == 200
    # Push is off in this fixture, but the subscriber routes must still be
    # *reachable* without a bearer — the feature flag answers, not the gate.
    assert public_client.get("/api/push/config").json() == {"enabled": False}
    assert public_client.post(
        "/api/push/unsubscribe", json={"endpoint": "https://example.invalid/x"},
    ).status_code == 503
    assert public_client.get("/nowcast/manifest.json").status_code == 503
    assert public_client.get(
        "/forecast", params={"lat": HOME_LAT, "lon": HOME_LON},
    ).status_code == 503

    # With artifacts on disk they serve, with their Phase A cache headers.
    manifest = _seed_nowcast(public_config)
    r = public_client.get("/nowcast/manifest.json")
    assert r.status_code == 200
    assert r.json() == manifest
    assert r.headers["cache-control"] == "public, max-age=30"

    r = public_client.get(f"/nowcast/p_rain_10min_{STAMP}.png")
    assert r.status_code == 200
    assert r.content == PNG_BYTES
    assert r.headers["cache-control"] == "public, max-age=300, immutable"


def test_hidden_routes_404_without_bearer(
    public_config: Config, public_client: TestClient,
) -> None:
    """Everything private is 404 — and the data really is on disk, so this
    is a gate test, not a "nothing to serve yet" test."""
    _seed_state(public_config)
    _seed_frames(public_config)

    for method, url in HIDDEN_ROUTES:
        r = public_client.request(method, url, json={"strikes": []})
        assert r.status_code == 404, f"{method} {url} leaked {r.status_code}"
        assert r.json() == {"detail": "Not Found"}


def test_hidden_404_is_indistinguishable_from_nonexistent(
    public_config: Config, public_client: TestClient,
) -> None:
    """Same status, same body, no auth hint in the headers."""
    _seed_state(public_config)
    hidden = public_client.get("/state.json")
    absent = public_client.get("/no-such-route-at-all")
    assert hidden.status_code == absent.status_code == 404
    assert hidden.json() == absent.json()
    assert "www-authenticate" not in {k.lower() for k in hidden.headers}
    assert hidden.headers.get("content-type") == absent.headers.get("content-type")


def test_hidden_routes_normal_with_bearer(
    public_config: Config, public_client: TestClient,
) -> None:
    """A valid bearer restores each hidden route's ordinary behaviour."""
    _seed_state(public_config, eta_minutes=12.0)
    _seed_frames(public_config)
    auth = _auth()

    r = public_client.get("/state.json", headers=auth)
    assert r.status_code == 200
    assert r.json()["forecast"]["eta_minutes"] == 12.0
    assert r.json()["home"]["lat"] == HOME_LAT

    r = public_client.get("/frames/manifest.json", headers=auth)
    assert r.status_code == 200
    assert r.json()["frame_count"] == 1

    r = public_client.get("/frames/frame_00.png", headers=auth)
    assert r.status_code == 200
    assert r.content == PNG_BYTES

    # Lightning is enabled in the default config: a real answer, not a 404.
    r = public_client.get(
        "/lightning/eta", params={"lat": HOME_LAT, "lon": HOME_LON}, headers=auth,
    )
    assert r.status_code == 200

    # The write endpoint accepts pushes again (and only) behind the bearer.
    r = public_client.post("/lightning/strikes", json={"strikes": []}, headers=auth)
    assert r.status_code == 200
    assert r.json()["accepted"] == 0

    assert public_client.get("/openapi.json", headers=auth).status_code == 200


def test_trailing_slash_does_not_confirm_a_hidden_route(
    public_config: Config, public_client: TestClient,
) -> None:
    """Starlette redirects ``/state.json/`` → ``/state.json`` with a 307,
    and a redirect only exists for routes that exist. The gate covers the
    slash-toggled path so the probe stays a plain 404."""
    _seed_state(public_config)
    for path in ("/state.json/", "/lightning/eta/", "/frames/manifest.json/"):
        r = public_client.get(path, follow_redirects=False)
        assert r.status_code == 404, f"{path} answered {r.status_code}"
        assert r.json() == {"detail": "Not Found"}


def test_wrong_bearer_is_still_404(
    public_config: Config, public_client: TestClient,
) -> None:
    """A bad token must not upgrade the 404 into a 401 (that would confirm
    the route exists)."""
    _seed_state(public_config)
    for headers in (
        {"Authorization": "Bearer wrong"},
        {"Authorization": "Basic " + API_KEY},
        {"Authorization": API_KEY},
    ):
        r = public_client.get("/state.json", headers=headers)
        assert r.status_code == 404
        assert r.json() == {"detail": "Not Found"}


def test_public_mode_without_key_has_no_unlock(minimal_config: Config) -> None:
    """public_mode + no api_key: the hidden surface is simply unreachable —
    a bearer cannot be guessed into existence."""
    minimal_config.server.public_mode = True
    assert minimal_config.server.api_key is None
    _seed_state(minimal_config)
    app = create_app(minimal_config, auto_start_scheduler=False)
    with TestClient(app) as c:
        assert c.get("/state.json").status_code == 404
        assert c.get("/state.json", headers={"Authorization": "Bearer "}).status_code == 404
        assert c.get("/healthz").status_code == 200


# ---------------------------------------------------------------------------
# Route matrix — default (LAN) mode is unchanged
# ---------------------------------------------------------------------------

def test_default_mode_state_json_unchanged(
    minimal_config: Config, default_client: TestClient,
) -> None:
    """Exact assertions on the endpoint the HA integration polls."""
    r = default_client.get("/state.json")
    assert r.status_code == 503
    assert "no nowcast state" in r.json()["detail"]

    _seed_state(minimal_config, eta_minutes=7.5)
    r = default_client.get("/state.json")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/json"
    body = r.json()
    assert body["schema_version"] == 1
    assert body["forecast"]["eta_minutes"] == 7.5
    assert body["forecast"]["rain_incoming"] is True
    assert body["home"] == {"lat": HOME_LAT, "lon": HOME_LON, "radius_km": 1.0}
    assert body["confidence"] == 0.71


def test_default_mode_frames_unchanged(
    minimal_config: Config, default_client: TestClient,
) -> None:
    """/frames/* keeps serving the home crop unauthenticated."""
    assert default_client.get("/frames/manifest.json").status_code == 503

    _seed_frames(minimal_config)
    r = default_client.get("/frames/manifest.json")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/json"
    assert r.json()["frames"][0]["filename"] == "frame_00.png"

    r = default_client.get("/frames/frame_00.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.headers["cache-control"] == "public, max-age=60"
    assert r.content == PNG_BYTES

    # The guards still guard, and a missing frame is still 404 (not gated).
    assert default_client.get("/frames/frame_aa.png").status_code == 400
    assert default_client.get("/frames/frame_99.png").status_code == 404


def test_default_mode_leaves_every_hidden_route_reachable(
    minimal_config: Config, default_client: TestClient,
) -> None:
    """No route disappears when public_mode is off — the gate is not
    installed at all, so nothing can answer 404-by-policy."""
    _seed_state(minimal_config)
    _seed_frames(minimal_config)
    for method, url in HIDDEN_ROUTES:
        r = default_client.request(method, url, json={"strikes": []})
        assert r.status_code != 404, f"{method} {url} unexpectedly hidden"


def test_default_mode_api_key_still_guards_nowcast(
    minimal_config: Config,
) -> None:
    """Outside public mode the §A3 endpoints keep their bearer semantics —
    the public-mode carve-out in require_api_key must not bleed across."""
    minimal_config.server.api_key = API_KEY
    _seed_nowcast(minimal_config)
    app = create_app(minimal_config, auto_start_scheduler=False)
    with TestClient(app) as c:
        assert c.get("/nowcast/manifest.json").status_code == 401
        assert c.get("/nowcast/manifest.json", headers=_auth()).status_code == 200


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

INDEX_HTML = "<!doctype html><title>dmi-nowcast</title><div id=app></div>"
HASHED_JS = "export const x=1;\n"
SW_JS = "self.addEventListener('install',()=>{});\n"
PMTILES = bytes(range(256)) * 8  # 2048 deterministic bytes


@pytest.fixture
def frontend_dir(tmp_path: Path) -> Path:
    """A miniature SvelteKit ``build/`` tree."""
    root = tmp_path / "frontend_build"
    (root / "_app" / "immutable" / "chunks").mkdir(parents=True)
    (root / "index.html").write_text(INDEX_HTML)
    (root / "service-worker.js").write_text(SW_JS)
    (root / "manifest.webmanifest").write_text('{"name":"dmi-nowcast"}')
    (root / "_app" / "immutable" / "chunks" / "entry.abc12345.js").write_text(HASHED_JS)
    (root / "basemap.pmtiles").write_bytes(PMTILES)
    return root


@pytest.fixture
def site_client(public_config: Config, frontend_dir: Path):
    public_config.server.frontend_dir = frontend_dir
    app = create_app(public_config, auto_start_scheduler=False)
    with TestClient(app) as c:
        yield c


def test_index_served_at_root_short_cached(site_client: TestClient) -> None:
    r = site_client.get("/")
    assert r.status_code == 200
    assert r.text == INDEX_HTML
    assert r.headers["content-type"].startswith("text/html")
    assert r.headers["cache-control"] == "no-cache"


def test_spa_fallback_for_client_routes(site_client: TestClient) -> None:
    """Client-side routes (no file extension) resolve to the app shell."""
    for path in ("/about", "/da/om", "/deep/client/route"):
        r = site_client.get(path)
        assert r.status_code == 200, path
        assert r.text == INDEX_HTML
        assert r.headers["cache-control"] == "no-cache"


def test_missing_asset_is_404_not_the_shell(site_client: TestClient) -> None:
    """A missing *file* must not come back as 200 HTML — a broken deploy
    should look broken, not serve HTML where JSON/tiles were expected."""
    for path in ("/missing.json", "/_app/immutable/chunks/gone.deadbeef.js"):
        assert site_client.get(path).status_code == 404, path


def test_cache_header_table(site_client: TestClient) -> None:
    cases = {
        "/_app/immutable/chunks/entry.abc12345.js": "public, max-age=31536000, immutable",
        "/index.html": "no-cache",
        "/service-worker.js": "no-cache",
        "/basemap.pmtiles": "public, max-age=86400",
        "/manifest.webmanifest": "public, max-age=300",
    }
    for path, expected in cases.items():
        r = site_client.get(path)
        assert r.status_code == 200, path
        assert r.headers["cache-control"] == expected, path


def test_pmtiles_range_request_returns_206(site_client: TestClient) -> None:
    """MapLibre reads the basemap by byte range — the whole Protomaps
    decision rests on this working. Starlette's FileResponse implements it
    natively (verified against the installed version); this test pins it."""
    r = site_client.get("/basemap.pmtiles", headers={"Range": "bytes=100-199"})
    assert r.status_code == 206
    assert r.content == PMTILES[100:200]
    assert len(r.content) == 100
    assert r.headers["content-range"] == f"bytes 100-199/{len(PMTILES)}"
    assert r.headers["content-length"] == "100"
    assert r.headers["accept-ranges"] == "bytes"

    # A second, non-adjacent range — the pmtiles reader does many of these.
    r2 = site_client.get("/basemap.pmtiles", headers={"Range": "bytes=0-15"})
    assert r2.status_code == 206
    assert r2.content == PMTILES[0:16]

    # Open-ended and suffix forms both behave.
    r3 = site_client.get("/basemap.pmtiles", headers={"Range": "bytes=2040-"})
    assert r3.status_code == 206
    assert r3.content == PMTILES[2040:]

    # Unsatisfiable range → 416, not a silent full body.
    r4 = site_client.get("/basemap.pmtiles", headers={"Range": "bytes=99999-"})
    assert r4.status_code == 416

    # Without a Range header the file is served whole.
    r5 = site_client.get("/basemap.pmtiles")
    assert r5.status_code == 200
    assert r5.content == PMTILES


def test_frontend_never_shadows_the_api_or_the_gate(
    public_config: Config, site_client: TestClient,
) -> None:
    """With the frontend mounted, public API routes still win over static
    files, and hidden routes are still 404 (not the SPA shell)."""
    _seed_nowcast(public_config)
    _seed_state(public_config)
    assert site_client.get("/healthz").json()["status"] == "ok"
    assert site_client.get("/nowcast/manifest.json").status_code == 200
    r = site_client.get("/state.json")
    assert r.status_code == 404
    assert r.json() == {"detail": "Not Found"}
    assert site_client.get("/state.json", headers=_auth()).status_code == 200


def test_frontend_served_in_default_mode_too(
    minimal_config: Config, frontend_dir: Path,
) -> None:
    """frontend_dir is independent of public_mode (local `npm run` preview
    against the LAN instance)."""
    minimal_config.server.frontend_dir = frontend_dir
    app = create_app(minimal_config, auto_start_scheduler=False)
    with TestClient(app) as c:
        assert c.get("/").text == INDEX_HTML
        assert c.get("/state.json").status_code == 503  # API still first


def test_missing_frontend_dir_is_a_warning_not_a_crash(
    minimal_config: Config, tmp_path: Path,
) -> None:
    """The image builds with the frontend optional; an API-only service is
    better than one that refuses to boot."""
    minimal_config.server.frontend_dir = tmp_path / "never_built"
    app = create_app(minimal_config, auto_start_scheduler=False)
    with TestClient(app) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/").status_code == 404


# ---------------------------------------------------------------------------
# Compute savings — render + basemap skipped in public mode
# ---------------------------------------------------------------------------

DMI_PROJ = "+proj=stere +lat_0=56 +lon_0=10.5666 +lat_ts=56 +ellps=WGS84 +units=m +no_defs"
GRID_PX = 64
PIXEL_M = 500.0
GAIN, OFFSET = 0.5, -32.0
NODATA, UNDETECT = 255, 0


def _corners_lonlat() -> dict[str, tuple[float, float]]:
    """Grid corners such that the reference point sits at the grid centre."""
    crs = CRS.from_proj4(DMI_PROJ)
    to_proj = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x, y = to_proj.transform(HOME_LON, HOME_LAT)
    half = GRID_PX / 2 * PIXEL_M
    return {
        "UL": to_wgs.transform(x - half, y + half),
        "UR": to_wgs.transform(x + half, y + half),
        "LL": to_wgs.transform(x - half, y - half),
        "LR": to_wgs.transform(x + half, y - half),
    }


def _write_composite(path: Path, ts: datetime, dbz_value: float) -> None:
    """Minimal DMI-style ODIM HDF5 composite, uniform ``dbz_value``."""
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


@pytest.fixture
def synthetic_paths(tmp_path: Path) -> list[Path]:
    """Two uniformly wet frames 10 min apart (fullRange cadence)."""
    newest = datetime.now(timezone.utc) - timedelta(minutes=4)
    paths = []
    for i, dbz in enumerate((30.0, 31.0)):
        p = tmp_path / f"public_synthetic_{i}.h5"
        _write_composite(p, newest - timedelta(minutes=10 * (1 - i)), dbz)
        paths.append(p)
    return paths


def _install_tripwires(monkeypatch: pytest.MonkeyPatch) -> tuple[list, list]:
    """Stub render_frames + build_basemap so a call is both recorded and
    fatal-looking. Recording matters: a bare raise would be swallowed by
    the cycle's render-failure policy and prove nothing."""
    render_calls: list[dict] = []
    basemap_calls: list[dict] = []

    def _render(**kwargs):
        render_calls.append(kwargs)
        raise AssertionError("render_frames must not run in public_mode")

    def _basemap(**kwargs):
        basemap_calls.append(kwargs)
        raise AssertionError("build_basemap must not run in public_mode")

    monkeypatch.setattr(compute_mod, "render_frames", _render)
    monkeypatch.setattr(compute_mod, "build_basemap", _basemap)
    return render_calls, basemap_calls


def test_public_mode_skips_render_and_basemap(
    public_config: Config,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public_config.forecast.steps.enabled = False  # deterministic path only
    render_calls, basemap_calls = _install_tripwires(monkeypatch)

    engine = CycleEngine(public_config)
    state = engine._compute_sync(synthetic_paths, fetch_ms=0.0)

    assert render_calls == []
    assert basemap_calls == []
    # The basemap fetch was never even attempted (the lazy-load flag proves
    # it, since _ensure_basemap sets it before trying).
    assert engine._basemap_attempted is False
    assert engine.basemap is None
    assert state.diagnostics.render_ms == 0.0
    # No frame artifacts written — /frames/* would have nothing to serve,
    # which is exactly right since it is hidden.
    frames_dir = public_config.storage.data_dir / "frames"
    assert list(frames_dir.glob("*.png")) == []
    assert not (frames_dir / "frames.json").exists()

    # Everything the public site consumes is untouched: the state is real.
    assert state.now.raining is True
    assert state.forecast.eta_minutes == 5.0
    assert [e.lead_min for e in state.forecast.per_lead] == [
        5, 10, 15, 20, 25, 30, 45, 60,
    ]
    assert state.radar.latest_ts is not None
    assert state.motion is not None


def test_default_mode_still_renders(
    minimal_config: Config,
    synthetic_paths: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control for the test above: with public_mode off the same stubs ARE
    called (the raise is caught by the render-failure policy, so the cycle
    still produces a state — the recorded call is the evidence)."""
    minimal_config.forecast.steps.enabled = False
    render_calls, basemap_calls = _install_tripwires(monkeypatch)

    engine = CycleEngine(minimal_config)
    state = engine._compute_sync(synthetic_paths, fetch_ms=0.0)

    assert len(basemap_calls) == 1
    assert len(render_calls) == 1
    assert render_calls[0]["out_dir"] == minimal_config.storage.data_dir / "frames"
    assert engine._basemap_attempted is True
    assert state.diagnostics.render_ms == 0.0  # the stub raised
    assert state.now.raining is True
