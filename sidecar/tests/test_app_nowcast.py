"""A3 — /nowcast/* artifact serving + /forecast point lookup (Phase A §A3).

Fully synthetic, mirroring the suite's conventions:

- artifact/manifest serving is tested against fake files written straight
  into the nowcast dir (the A2 writer has its own tests),
- /forecast runs against a hand-built :class:`NationalProducts` (no STEPS)
  plus a :class:`CompositeGeo` from the same tiny synthetic ODIM composite
  helper ``test_compute_ensemble.py`` uses (64×64 @ 500 m centred on the
  test home, so the home pixel is exactly the grid centre),
- bearer behaviour matches ``test_app.py``'s ``require_api_key`` coverage:
  no-op when ``server.api_key`` is unset, enforced on the §A3 endpoints
  when set, legacy read endpoints untouched.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pyproj import CRS, Transformer

from dmi_nowcast_core.geo import CompositeGeo
from dmi_nowcast_core.national import NationalProducts
from dmi_nowcast_core.parse import parse_composite
from dmi_nowcast_sidecar.app import _safe_nowcast_name, create_app
from dmi_nowcast_sidecar.compute import CycleEngine, NationalSnapshot
from dmi_nowcast_sidecar.config import Config

# Same home as ``minimal_config`` (conftest.py); the synthetic grid is built
# so this point lands exactly at its centre (adapted from
# test_compute_ensemble.py).
HOME_LON, HOME_LAT = 10.32, 55.33
DMI_PROJ = "+proj=stere +lat_0=56 +lon_0=10.5666 +lat_ts=56 +ellps=WGS84 +units=m +no_defs"
GRID_PX = 64          # native synthetic grid — 64×64 @ 500 m = 32×32 km
PIXEL_M = 500.0
GAIN, OFFSET = 0.5, -32.0
NODATA, UNDETECT = 255, 0

DOWNSAMPLE = 4
GRID_DS = GRID_PX // DOWNSAMPLE           # 16×16 product grids
CENTRE_DS = (GRID_PX // 2) // DOWNSAMPLE  # home pixel on the product grid
NAN_PIXEL = (2, 3)                        # downsampled pixel forced to NaN
OBSERVED_MM_H = 1.25                      # observed rain on the product grid

RADAR_TS = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
STAMP = "202608281200"

# Tiny valid 1x1 transparent PNG (same bytes test_frames.py uses).
PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c63000000000200013e9a3a200000000049454e44ae426082"
)


# ---------------------------------------------------------------------------
# Synthetic ODIM helper (adapted from test_compute_ensemble.py) — only used
# to build a CompositeGeo through the production parse path.
# ---------------------------------------------------------------------------

def _corners_lonlat() -> dict[str, tuple[float, float]]:
    """Grid corners such that home sits exactly at the grid centre."""
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
    """Write a minimal DMI-style ODIM HDF5 composite (uniform ``dbz_value``)."""
    raw_value = int(round((dbz_value - OFFSET) / GAIN))
    assert UNDETECT < raw_value < NODATA
    raw = np.full((GRID_PX, GRID_PX), raw_value, dtype=np.uint8)
    corners = _corners_lonlat()
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
        for name, (lon, lat) in corners.items():
            where.attrs[f"{name}_lon"] = lon
            where.attrs[f"{name}_lat"] = lat
        how = h5.create_group("how")
        how.attrs["zr-a"] = 200.0
        how.attrs["zr-b"] = 1.6
        h5.create_group("dataset1").create_group("data1").create_dataset(
            "data", data=raw,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _grid(value: float) -> np.ndarray:
    g = np.full((GRID_DS, GRID_DS), value, dtype=np.float32)
    g[NAN_PIXEL] = np.nan
    return g


@pytest.fixture
def products() -> NationalProducts:
    """Hand-built products: uniform known values, one NaN pixel."""
    return NationalProducts(
        p_rain={10: _grid(0.625), 20: _grid(0.75)},
        eta_min=_grid(6.0),
        intensity_mm_h=_grid(2.5),
        leads_min=(10, 20),
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
def observed() -> np.ndarray:
    """Observed rain on the same product grid; NaN at the same nodata pixel."""
    return _grid(OBSERVED_MM_H)


@pytest.fixture
def engine(
    minimal_config: Config,
    geo: CompositeGeo,
    products: NationalProducts,
    observed: np.ndarray,
) -> CycleEngine:
    """Engine with synthetic geo + national products already 'computed'."""
    eng = CycleEngine(minimal_config)
    eng._basemap_attempted = True  # never fetch OSM
    eng._geo = geo
    eng._national_latest = NationalSnapshot(products, RADAR_TS, observed)
    return eng


@pytest.fixture
def client(minimal_config: Config, engine: CycleEngine):
    app = create_app(minimal_config, engine=engine, auto_start_scheduler=False)
    with TestClient(app) as c:
        yield c


def _nowcast_dir(config: Config) -> Path:
    return config.storage.data_dir / "nowcast"


def _seed_artifacts(config: Config) -> dict:
    """Write a fake stamped PNG + stamped and stable manifests; return the
    manifest dict."""
    out = _nowcast_dir(config)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "cycle": STAMP,
        "radar_ts_utc": RADAR_TS.isoformat(),
        "artifacts": [{"filename": f"p_rain_10min_{STAMP}.png"}],
    }
    payload = json.dumps(manifest).encode()
    (out / f"p_rain_10min_{STAMP}.png").write_bytes(PNG_BYTES)
    (out / f"manifest_{STAMP}.json").write_bytes(payload)
    (out / "manifest.json").write_bytes(payload)
    return manifest


def _write_minimal_state(config: Config, confidence: float) -> None:
    """Persist a minimal valid state.json so /forecast can source confidence."""
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
    from dmi_nowcast_sidecar.storage import StateStore

    state = State(
        generated_at=RADAR_TS,
        radar=RadarBlock(latest_ts=RADAR_TS, data_age_minutes=2.0),
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
        confidence=confidence,
        calibration=CalibrationBlock(
            fitted_at=None, n_events=None, brier_before=None, brier_after=None,
        ),
        diagnostics=DiagnosticsBlock(
            cycle_ms=0.0, fetch_ms=0.0, compute_ms=0.0, render_ms=0.0,
        ),
    )
    StateStore(config.storage.data_dir).write(state)


# ---------------------------------------------------------------------------
# _safe_nowcast_name — allow-list guard
# ---------------------------------------------------------------------------

def test_safe_nowcast_name_accepts_stamped_artifacts() -> None:
    assert _safe_nowcast_name(f"p_rain_10min_{STAMP}.png")
    assert _safe_nowcast_name(f"eta_{STAMP}.png")
    assert _safe_nowcast_name(f"intensity_{STAMP}.png")
    assert _safe_nowcast_name(f"overlay_now_{STAMP}.png")
    assert _safe_nowcast_name(f"overlay_45min_{STAMP}.png")
    assert _safe_nowcast_name(f"manifest_{STAMP}.json")
    # R2 motion grids — underscores only, so the existing regex serves them.
    assert _safe_nowcast_name(f"motion_east_kmh_{STAMP}.png")
    assert _safe_nowcast_name(f"motion_north_kmh_{STAMP}.png")


def test_safe_nowcast_name_rejects_alias_traversal_and_junk() -> None:
    # The stable alias must never flow through the immutable route.
    assert not _safe_nowcast_name("manifest.json")
    assert not _safe_nowcast_name("../etc/passwd")
    assert not _safe_nowcast_name(f"../p_rain_10min_{STAMP}.png")
    assert not _safe_nowcast_name(f"p_rain_10min_{STAMP}.png/../")
    assert not _safe_nowcast_name(f"p_rain_10min_{STAMP}.PNG")   # uppercase ext
    assert not _safe_nowcast_name(f"p_rain_10min_{STAMP}.exe")   # wrong ext
    assert not _safe_nowcast_name("p_rain_10min_2026082812.png") # 10-digit stamp
    assert not _safe_nowcast_name(f"p_rain_10min_{STAMP}.png.json")
    assert not _safe_nowcast_name(f"_hidden_{STAMP}.png")        # non-alnum lead
    assert not _safe_nowcast_name(f"{STAMP}.png")                # no stem
    assert not _safe_nowcast_name("state.json")
    assert not _safe_nowcast_name("")


# ---------------------------------------------------------------------------
# GET /nowcast/manifest.json — stable alias
# ---------------------------------------------------------------------------

def test_nowcast_manifest_503_before_first_cycle(client: TestClient) -> None:
    r = client.get("/nowcast/manifest.json")
    assert r.status_code == 503
    assert "first national cycle" in r.json()["detail"]


def test_nowcast_manifest_served_with_short_cache(
    minimal_config: Config, client: TestClient,
) -> None:
    manifest = _seed_artifacts(minimal_config)
    r = client.get("/nowcast/manifest.json")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/json"
    assert r.headers["cache-control"] == "public, max-age=30"
    # The alias is rewritten each cycle — it must never be immutable.
    assert "immutable" not in r.headers["cache-control"]
    assert r.json() == manifest


# ---------------------------------------------------------------------------
# GET /nowcast/{filename} — cycle-stamped artifacts
# ---------------------------------------------------------------------------

def test_nowcast_artifact_png_served_immutable(
    minimal_config: Config, client: TestClient,
) -> None:
    _seed_artifacts(minimal_config)
    r = client.get(f"/nowcast/p_rain_10min_{STAMP}.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.headers["cache-control"] == "public, max-age=300, immutable"
    assert r.content == PNG_BYTES


def test_nowcast_stamped_manifest_served_immutable(
    minimal_config: Config, client: TestClient,
) -> None:
    manifest = _seed_artifacts(minimal_config)
    r = client.get(f"/nowcast/manifest_{STAMP}.json")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/json"
    assert r.headers["cache-control"] == "public, max-age=300, immutable"
    assert r.json() == manifest


def test_nowcast_artifact_bad_name_400(
    minimal_config: Config, client: TestClient,
) -> None:
    _seed_artifacts(minimal_config)
    assert client.get("/nowcast/etc..").status_code == 400
    assert client.get(f"/nowcast/p_rain_10min_{STAMP}.exe").status_code == 400
    assert client.get("/nowcast/state.json").status_code == 400
    # Percent-encoded traversal: depending on client/router normalisation this
    # is rejected by the guard (400) or never matches a route (404) — either
    # way the file must not be served.
    r = client.get("/nowcast/%2e%2e%2fstate.json")
    assert r.status_code in (400, 404)


def test_nowcast_artifact_missing_404(
    minimal_config: Config, client: TestClient,
) -> None:
    _seed_artifacts(minimal_config)
    r = client.get("/nowcast/p_rain_60min_209912310000.png")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /forecast — point lookup into the in-memory national products
# ---------------------------------------------------------------------------

def test_forecast_503_before_first_national_cycle(
    minimal_config: Config,
) -> None:
    """Fresh engine: no geo, no products → 503 with a clear detail."""
    app = create_app(minimal_config, auto_start_scheduler=False)
    with TestClient(app) as c:
        r = c.get("/forecast", params={"lat": HOME_LAT, "lon": HOME_LON})
        assert r.status_code == 503
        assert "first national cycle" in r.json()["detail"]


def test_forecast_values_at_known_pixel(client: TestClient) -> None:
    """Home sits at the product-grid centre; every grid has a known value."""
    r = client.get("/forecast", params={"lat": HOME_LAT, "lon": HOME_LON})
    assert r.status_code == 200
    assert r.headers["cache-control"] == "public, max-age=30"
    body = r.json()
    assert body["lat"] == HOME_LAT
    assert body["lon"] == HOME_LON
    # UTC ISO 8601 with explicit offset, equal to the products' radar time.
    assert body["radar_ts_utc"].endswith(("Z", "+00:00"))
    assert datetime.fromisoformat(body["radar_ts_utc"]) == RADAR_TS
    assert body["n_members"] == 8
    assert body["calibrated"] is False
    assert body["per_lead"] == [
        {"lead_min": 10, "p_rain": pytest.approx(0.625)},
        {"lead_min": 20, "p_rain": pytest.approx(0.75)},
    ]
    assert body["eta_min"] == pytest.approx(6.0)
    assert body["intensity_mm_h"] == pytest.approx(2.5)
    # The observation, on the same pixel as every forecast product.
    assert body["observed_mm_h"] == pytest.approx(OBSERVED_MM_H)
    # No state.json written yet → confidence is null, not fabricated.
    assert body["confidence"] is None


def test_forecast_confidence_from_state(
    minimal_config: Config, client: TestClient,
) -> None:
    """Confidence comes from the same store /state.json serves."""
    _write_minimal_state(minimal_config, confidence=0.71)
    r = client.get("/forecast", params={"lat": HOME_LAT, "lon": HOME_LON})
    assert r.status_code == 200
    assert r.json()["confidence"] == pytest.approx(0.71)


def test_forecast_nan_pixel_returns_nulls(
    client: TestClient, geo: CompositeGeo,
) -> None:
    """A NaN (off-composite) pixel yields nulls, never NaN-in-JSON."""
    # Downsampled NAN_PIXEL (r, c) ↔ native (r·f, c·f) exactly.
    lon, lat = geo.grid_to_lonlat(
        NAN_PIXEL[0] * DOWNSAMPLE, NAN_PIXEL[1] * DOWNSAMPLE,
    )
    r = client.get("/forecast", params={"lat": lat, "lon": lon})
    assert r.status_code == 200
    body = r.json()
    assert [e["p_rain"] for e in body["per_lead"]] == [None, None]
    assert body["eta_min"] is None
    assert body["intensity_mm_h"] is None
    # A nodata pixel is an UNKNOWN observation, never a dry one.
    assert body["observed_mm_h"] is None
    # Grid-independent fields still present.
    assert body["n_members"] == 8


def test_forecast_without_an_observed_grid_serves_null(
    minimal_config: Config, engine: CycleEngine, products: NationalProducts,
) -> None:
    """A cycle whose observed reduction failed still serves every forecast
    field — the observation is additive, not load-bearing. The plain tuple
    is the pre-observation snapshot shape, which must keep working."""
    engine._national_latest = (products, RADAR_TS)
    app = create_app(minimal_config, engine=engine, auto_start_scheduler=False)
    with TestClient(app) as c:
        r = c.get("/forecast", params={"lat": HOME_LAT, "lon": HOME_LON})
    assert r.status_code == 200
    body = r.json()
    assert body["observed_mm_h"] is None
    assert body["eta_min"] == pytest.approx(6.0)


def test_forecast_out_of_grid_400(client: TestClient) -> None:
    """Valid lat/lon far off the 32×32 km synthetic composite → 400."""
    r = client.get("/forecast", params={"lat": 45.0, "lon": -30.0})
    assert r.status_code == 400
    assert "outside" in r.json()["detail"]


def test_forecast_invalid_params_422(client: TestClient) -> None:
    assert client.get("/forecast").status_code == 422
    assert client.get("/forecast", params={"lat": HOME_LAT}).status_code == 422
    assert client.get(
        "/forecast", params={"lat": "abc", "lon": HOME_LON},
    ).status_code == 422
    assert client.get(
        "/forecast", params={"lat": 95.0, "lon": HOME_LON},
    ).status_code == 422
    assert client.get(
        "/forecast", params={"lat": HOME_LAT, "lon": 181.0},
    ).status_code == 422


# ---------------------------------------------------------------------------
# Bearer behaviour — same optional require_api_key as the write endpoints
# ---------------------------------------------------------------------------

def test_bearer_enforced_on_all_three_when_key_set(
    minimal_config: Config, engine: CycleEngine,
) -> None:
    minimal_config.server.api_key = "secret-123"
    _seed_artifacts(minimal_config)
    app = create_app(minimal_config, engine=engine, auto_start_scheduler=False)
    auth = {"Authorization": "Bearer secret-123"}
    with TestClient(app) as c:
        for url in (
            "/nowcast/manifest.json",
            f"/nowcast/p_rain_10min_{STAMP}.png",
            f"/forecast?lat={HOME_LAT}&lon={HOME_LON}",
        ):
            assert c.get(url).status_code == 401
            assert c.get(
                url, headers={"Authorization": "Bearer wrong"},
            ).status_code == 401
            assert c.get(url, headers=auth).status_code == 200
        # Legacy read endpoints stay open (HA polls them unauthenticated):
        # 503 (no state yet), never 401.
        assert c.get("/state.json").status_code == 503


def test_bearer_noop_when_key_unset(
    minimal_config: Config, client: TestClient,
) -> None:
    """LAN-trust default: everything already exercised unauthenticated above;
    spot-check one artifact URL explicitly."""
    _seed_artifacts(minimal_config)
    assert minimal_config.server.api_key is None
    assert client.get("/nowcast/manifest.json").status_code == 200
