"""FastAPI app + healthz + state.json + auth dependency."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from dmi_nowcast_sidecar.app import create_app, require_api_key
from dmi_nowcast_sidecar.config import Config


@pytest.fixture
def client(minimal_config: Config):
    """TestClient with lifespan entered so app.state.engine is populated."""
    app = create_app(minimal_config, auto_start_scheduler=False)
    with TestClient(app) as c:
        yield c


def test_healthz_returns_200(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["started_at"]
    assert body["last_cycle"] is None  # no state.json yet
    assert body["server_time"]


def test_app_state_has_config(minimal_config: Config) -> None:
    app = create_app(minimal_config, auto_start_scheduler=False)
    with TestClient(app):
        assert app.state.config is minimal_config
        assert app.state.engine is not None
        assert app.state.last_cycle_at is None


def test_state_json_503_before_any_cycle(client: TestClient) -> None:
    """Sidecar with no state.json on disk returns 503 — HA coordinator
    treats this as unavailable, which is the right semantics."""
    r = client.get("/state.json")
    assert r.status_code == 503
    assert "no nowcast state" in r.json()["detail"]


def test_state_json_returns_persisted_state(minimal_config: Config) -> None:
    """Sidecar boot finds a previously-written state.json and serves it."""
    # Write a state.json manually.
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

    state = State(
        generated_at=datetime(2026, 5, 20, 7, 30, tzinfo=timezone.utc),
        radar=RadarBlock(
            latest_ts=datetime(2026, 5, 20, 7, 25, tzinfo=timezone.utc),
            data_age_minutes=5.0,
        ),
        home=HomeBlock(lat=55.33, lon=10.32, radius_km=1.0),
        now=NowBlock(
            rain_rate_mm_h=0.0, rain_rate_p90_mm_h=0.0,
            raining=False, raining_hysteresis_state="dry",
        ),
        forecast=ForecastBlock(
            method="farneback",
            rain_incoming=True,
            eta_minutes=12.0,
            eta_p50_window_min=None,
            peak_intensity_mm_h=2.4,
            peak_lead_min=15,
            per_lead=[
                PerLeadEntry(lead_min=5, rain_rate_mm_h=0.0,
                             p_rain=0.0, p_calibrated=0.0),
                PerLeadEntry(lead_min=15, rain_rate_mm_h=2.4,
                             p_rain=1.0, p_calibrated=0.78),
            ],
        ),
        motion=MotionBlock(
            dy_px_per_min=-1.2, dx_px_per_min=0.8,
            speed_km_per_h=27.5, bearing_deg_from=215.0,
        ),
        confidence=0.71,
        calibration=CalibrationBlock(
            fitted_at=None, n_events=None,
            brier_before=None, brier_after=None,
        ),
        diagnostics=DiagnosticsBlock(
            cycle_ms=1820, fetch_ms=410, compute_ms=1290, render_ms=120,
        ),
    )
    store = StateStore(minimal_config.storage.data_dir)
    store.write(state)

    app = create_app(minimal_config, auto_start_scheduler=False)
    with TestClient(app) as c:
        r = c.get("/state.json")
        assert r.status_code == 200
        body = r.json()
        assert body["schema_version"] == 1
        assert body["forecast"]["eta_minutes"] == 12.0
        assert body["forecast"]["peak_intensity_mm_h"] == 2.4
        assert len(body["forecast"]["per_lead"]) == 2
        # Healthz now reports last_cycle (seeded from state.json on boot).
        h = c.get("/healthz").json()
        assert h["last_cycle"] is not None


def test_auth_noop_when_api_key_unset(minimal_config: Config) -> None:
    """LAN-trust default: write endpoints work without a token."""
    app = _app_with_protected_route(minimal_config)
    with TestClient(app) as c:
        assert c.post("/_test_protected").status_code == 200


def test_auth_required_when_api_key_set(minimal_config: Config) -> None:
    minimal_config.server.api_key = "secret-123"
    app = _app_with_protected_route(minimal_config)
    with TestClient(app) as c:
        assert c.post("/_test_protected").status_code == 401
        assert c.post("/_test_protected", headers={"Authorization": "Basic xxx"}).status_code == 401
        assert c.post(
            "/_test_protected", headers={"Authorization": "Bearer wrong"}
        ).status_code == 401
        assert c.post(
            "/_test_protected", headers={"Authorization": "Bearer secret-123"}
        ).status_code == 200


def _app_with_protected_route(config: Config) -> FastAPI:
    """Helper: add a test-only protected route so we can exercise the
    auth dependency without committing to a phase-A write endpoint."""
    app = create_app(config, auto_start_scheduler=False)

    @app.post("/_test_protected")
    async def protected(_=Depends(require_api_key)) -> dict:
        return {"ok": True}

    return app
