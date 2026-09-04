"""FastAPI app + routes.

Endpoints (phase B):

- ``GET /healthz`` — liveness; reports ``last_cycle`` timestamp.
- ``GET /state.json`` — latest nowcast state per :class:`State` schema.

Website Phase A (§A3) adds the national surface:

- ``GET /nowcast/manifest.json`` — latest national-cycle manifest.
- ``GET /nowcast/{filename}`` — cycle-stamped artifact PNGs / manifests.
- ``GET /forecast?lat=&lon=`` — point lookup into the national grids.

Website Phase D adds the Web Push surface (see ``push/routes.py``):

- ``GET  /api/push/config`` — feature flag + VAPID public key + options.
- ``POST /api/push/subscribe`` / ``POST /api/push/unsubscribe``.
- ``POST /api/push/test`` and ``GET /api/push/stats`` — operator-only.

Auth: write endpoints check ``Authorization: Bearer <key>`` against
``config.server.api_key`` when set. Legacy read endpoints are always open
since the HA integration polls them unauthenticated; the §A3 endpoints go
through the same optional bearer (``require_api_key`` — a no-op under
LAN trust, enforced when a key is set).

Public mode (website Phase C plan §P1)
--------------------------------------

``server.public_mode: true`` turns this process into the internet-facing
instance. One HTTP middleware — :func:`_public_mode_gate`, installed once
and only in public mode — implements a **default-deny** model over the
whole route table:

1. The **public surface** is an explicit allow-list, open to everyone
   with no bearer even when ``server.api_key`` is set (the key exists to
   unlock the hidden surface, not to lock the public one):

   ``/healthz``, ``/forecast``, anything under ``/nowcast/``, the three
   subscriber-facing push routes (``/api/push/config``,
   ``/api/push/subscribe``, ``/api/push/unsubscribe``) and the static
   frontend (see :func:`_mount_frontend`). Note these are *exact* paths,
   not an ``/api/push/`` prefix: ``/api/push/test`` and
   ``/api/push/stats`` are operator routes and stay hidden.

2. **Everything else that matches a registered route** — ``/state.json``
   (the configured point's block), ``/frames/*`` (the home crop),
   ``/lightning/*`` (including the strike-ingest POST and the archive
   dashboards), ``/api/push/test``, ``/api/push/stats``, ``/docs`` and
   ``/openapi.json`` — answers ``404 {"detail": "Not Found"}``,
   byte-identical to the response for a path that was never registered.
   A request carrying a valid ``Authorization: Bearer <server.api_key>``
   passes through and gets the route's normal behaviour, so an operator
   on the LAN can still reach everything.
   With ``api_key`` unset in public mode the hidden surface is simply
   unreachable — the safe default.

3. Paths matching **no** registered route fall through to the static
   frontend (SPA fallback), exactly as if the gate weren't there.

The route table is snapshotted before the frontend is mounted, so the
catch-all ``/`` mount never counts as "a registered route" for rule 2 —
and any route added later is hidden by default, which is the direction
a security default should fail in.

Public mode also skips the home-crop frame rendering and the OSM basemap
fetch in the cycle (see ``compute.py``): both exist only to feed
``/frames/*``, which is hidden here.
"""
from __future__ import annotations

import asyncio
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import structlog
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.routing import BaseRoute, Match
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

from dataclasses import asdict

from dmi_nowcast_core.lightning import (
    EtaParams,
    LightningStrike,
    compute_eta,
    smooth_eta,
    strike_probability,
    summarize_clusters,
)
from dmi_nowcast_core.lightning_render import render_lightning_map

from .archive_dashboard import render_dashboard_html
from . import prob_calibration

from . import __version__
from .compute import CycleEngine, CycleResult
from .config import Config
from .eta_smoother import EtaSmoother
from .lightning_schema import LightningEtaResponse, StrikesAccepted, StrikesIn
from .national_artifacts import LATEST_MANIFEST_NAME
from .national_sample import finite_or_none, sample_point
from .push.paths import resolved_db_path, resolved_key_path
from .push.routes import build_router as build_push_router
from .scheduler import CycleScheduler
from .state_schema import ForecastPointLead, ForecastPointResponse
from .storage import StateStore

_log = structlog.get_logger(__name__)


class HealthResponse(BaseModel):
    status: str
    version: str
    started_at: str
    last_cycle: str | None
    server_time: str


def create_app(
    config: Config,
    *,
    engine: CycleEngine | None = None,
    scheduler: CycleScheduler | None = None,
    auto_start_scheduler: bool = True,
) -> FastAPI:
    """Build the FastAPI app.

    ``engine`` and ``scheduler`` are injected by tests that want to
    control the cycle directly. The default in-process scheduler can be
    disabled with ``auto_start_scheduler=False`` — useful for unit
    tests that only exercise the HTTP surface.
    """
    if engine is None:
        engine = CycleEngine(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.config = config
        app.state.engine = engine
        app.state.started_at = datetime.now(timezone.utc)
        app.state.last_cycle_at = None
        app.state.last_error = None
        # Seed last_cycle_at from any persisted state.json so /healthz
        # is honest immediately after restart.
        existing = engine.store.load()
        if existing is not None:
            app.state.last_cycle_at = existing.generated_at

        # Build scheduler now that we have an app reference to bind the
        # on-complete callback to.
        local_scheduler = scheduler or CycleScheduler(
            engine,
            interval_min=config.poll.interval_min,
            jitter_sec=config.poll.jitter_sec,
            on_cycle_complete=lambda r: _on_cycle_complete(app, r),
            # Web Push evaluation runs after every cycle (Phase D). None
            # when the feature is off — the scheduler then behaves exactly
            # as it did before.
            after_cycle=(
                push_service.after_cycle if push_service is not None else None
            ),
        )
        app.state.scheduler = local_scheduler
        if auto_start_scheduler:
            await local_scheduler.start(run_immediately=True)
        try:
            yield
        finally:
            if auto_start_scheduler:
                await local_scheduler.shutdown()
            if push_store is not None:
                # Close the SQLite handle explicitly; the WAL files are
                # checkpointed on close, so a restart never inherits a
                # stale -wal alongside the volume snapshot.
                try:
                    push_store.close()
                except Exception as exc:  # noqa: BLE001
                    _log.warning("push_store_close_failed", error=str(exc))

    app = FastAPI(
        title="dmi-nowcast-sidecar",
        version=__version__,
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    # Web Push (Phase D). The store and the VAPID key live in the data
    # volume; the key is generated on first start when missing. Imported
    # lazily — ``push.service`` pulls in pywebpush/requests, which an
    # instance with the feature off should never have to load.
    #
    # An initialisation failure (unwritable volume, unreadable key) is
    # logged and degrades to "disabled": the routes answer 503 and the
    # cycle keeps running. A nowcast service that refuses to boot because
    # notifications are broken has its priorities backwards.
    push_service = None
    push_store = None
    push_public_key = None
    if config.push.enabled:
        try:
            from .push.service import PushService
            from .push.store import PushStore
            from .push.vapid import ensure_private_key, public_key_b64url

            pem = ensure_private_key(resolved_key_path(config))
            push_public_key = public_key_b64url(pem)
            push_store = PushStore(resolved_db_path(config))
            push_service = PushService(
                config, engine, push_store, pem, push_public_key,
            )
            _log.info(
                "push_enabled",
                db=str(resolved_db_path(config)),
                key=str(resolved_key_path(config)),
            )
        except Exception as exc:  # noqa: BLE001
            _log.error("push_init_failed", error=str(exc))
            push_service = None
            push_store = None
            push_public_key = None
    app.state.push_service = push_service
    app.state.push_store = push_store

    @app.get("/healthz", response_model=HealthResponse, tags=["public"])
    async def healthz(request: Request) -> HealthResponse:
        """Liveness probe. Always 200 if the process is up."""
        last_cycle = getattr(request.app.state, "last_cycle_at", None)
        started_at = getattr(request.app.state, "started_at", datetime.now(timezone.utc))
        return HealthResponse(
            status="ok",
            version=__version__,
            started_at=started_at.isoformat(timespec="seconds"),
            last_cycle=last_cycle.isoformat(timespec="seconds") if last_cycle else None,
            server_time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    @app.get("/frames/manifest.json", tags=["public"])
    async def frames_manifest(request: Request) -> Response:
        """Manifest of per-frame PNGs the Lovelace slider card consumes."""
        engine: CycleEngine = request.app.state.engine
        manifest_path = engine.config.storage.data_dir / "frames" / "frames.json"
        if not manifest_path.is_file():
            raise HTTPException(
                status_code=503,
                detail="no frames available yet — first cycle has not completed",
            )
        return Response(
            content=manifest_path.read_bytes(),
            media_type="application/json",
        )

    @app.get("/frames/{filename}", tags=["public"])
    async def frame_png(request: Request, filename: str) -> Response:
        """Serve one frame PNG. Path-traversal protected: only `frame_NN.png`
        (digits) is accepted, so an attacker can't request `/frames/../etc`."""
        if not _safe_frame_name(filename):
            raise HTTPException(status_code=400, detail="invalid frame name")
        engine: CycleEngine = request.app.state.engine
        path = engine.config.storage.data_dir / "frames" / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail="frame not found")
        return Response(
            content=path.read_bytes(),
            media_type="image/png",
            headers={
                # 60s cache — frames refresh on each cycle (~5 min) and HA
                # will revalidate after its own poll cadence.
                "Cache-Control": "public, max-age=60",
            },
        )

    @app.get("/state.json", tags=["public"])
    async def state_json(request: Request) -> Response:
        """Return the latest state.

        503 when no cycle has produced a state yet — clearer than 200
        with a partially-filled object, since the HA coordinator can
        interpret 503 as "unavailable" and surface a real reason
        rather than zeros.
        """
        store: StateStore = request.app.state.engine.store
        state = store.load()
        if state is None:
            raise HTTPException(
                status_code=503,
                detail="no nowcast state available yet — first cycle has not completed",
            )
        return Response(
            content=state.model_dump_json(),
            media_type="application/json",
        )

    # NOTE: registered before /nowcast/{filename} so the literal path wins —
    # the stable alias must never be served with the immutable cache header.
    @app.get("/nowcast/manifest.json", tags=["public"])
    async def nowcast_manifest(
        request: Request, _: None = Depends(require_api_key),
    ) -> Response:
        """Latest national-cycle manifest (stable alias; Phase A plan §A3).

        The manifest names every cycle-stamped artifact plus the grid
        geometry needed to sample them client-side. Short cache only — the
        alias is rewritten every cycle.
        """
        engine: CycleEngine = request.app.state.engine
        path = engine.config.storage.data_dir / "nowcast" / LATEST_MANIFEST_NAME
        if not path.is_file():
            raise HTTPException(
                status_code=503,
                detail="no national artifacts yet — first national cycle has not completed",
            )
        return Response(
            content=path.read_bytes(),
            media_type="application/json",
            headers={"Cache-Control": "public, max-age=30"},
        )

    @app.get("/nowcast/{filename}", tags=["public"])
    async def nowcast_artifact(
        request: Request, filename: str, _: None = Depends(require_api_key),
    ) -> Response:
        """Serve one national artifact: a product/overlay PNG or a stamped
        manifest. Path-traversal protected: only cycle-stamped names
        (``p_rain_20min_202608281200.png``, ``manifest_202608281200.json``)
        are accepted — stamped content never changes under a given name, so
        immutable caching is safe. The stable ``manifest.json`` alias is
        deliberately NOT matched here (the dedicated route above owns it)."""
        if not _safe_nowcast_name(filename):
            raise HTTPException(status_code=400, detail="invalid artifact name")
        engine: CycleEngine = request.app.state.engine
        path = engine.config.storage.data_dir / "nowcast" / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail="artifact not found")
        media_type = "image/png" if filename.endswith(".png") else "application/json"
        return Response(
            content=path.read_bytes(),
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=300, immutable"},
        )

    @app.get("/forecast", response_model=ForecastPointResponse, tags=["public"])
    async def forecast_point(
        request: Request,
        response: Response,
        lat: float = Query(..., ge=-90.0, le=90.0),
        lon: float = Query(..., ge=-180.0, le=180.0),
        _: None = Depends(require_api_key),
    ) -> ForecastPointResponse:
        """Point lookup into the latest in-memory national products (§A3).

        Pure O(1) work — one pyproj point transform + array indexing into
        the held grids, no numpy reductions — so it stays on the event loop
        by design. 503 before the first successful national cycle; 400 for
        coordinates outside the composite grid; 422 for missing/invalid
        query parameters (FastAPI validation).
        """
        engine: CycleEngine = request.app.state.engine
        latest = engine.national_latest
        geo = engine.geo
        if latest is None or geo is None:
            raise HTTPException(
                status_code=503,
                detail="no national forecast yet — first national cycle has not completed",
            )
        products, radar_ts = latest
        # lat/lon → native grid → ÷ downsample factor → nearest product
        # pixel (the aggregate_at_home convention, home.row / f). The
        # arithmetic lives in national_sample.sample_point, shared with the
        # Web Push decision engine (Phase D) so a notification and the
        # panel can never disagree about which pixel a point reads.
        # ``getattr`` because the snapshot's observed grid is additive: a
        # cycle whose observed reduction failed publishes the pair alone.
        sample = sample_point(
            products, geo, lat, lon,
            observed_mm_h=getattr(latest, "observed_mm_h", None),
        )
        if sample is None:
            raise HTTPException(
                status_code=400,
                detail="coordinates outside the radar composite grid",
            )
        per_lead = [
            ForecastPointLead(lead_min=lead, p_rain=p)
            for lead, p in sample.p_rain.items()
        ]
        # Confidence stays the global scalar in Phase A — sourced from the
        # same store /state.json serves; null before the first state exists.
        state = engine.store.load()
        # §B4 truthful flags: the held grids were calibrated with the
        # process's static national curves (loaded once at init, before any
        # cycle), so the calibrated subset of the served leads derives
        # exactly from that curve set. ``calibrated`` is true only when
        # every served lead's grid went through a curve; ``fitted_at`` is
        # null when none did.
        curve_leads = engine.national_curve_leads
        n_calibrated = sum(
            1 for lead in products.leads_min if int(lead) in curve_leads
        )
        response.headers["Cache-Control"] = "public, max-age=30"
        return ForecastPointResponse(
            lat=lat,
            lon=lon,
            radar_ts_utc=radar_ts,
            n_members=products.n_members,
            calibrated=(
                n_calibrated > 0 and n_calibrated == len(products.leads_min)
            ),
            calibration_fitted_at=(
                engine.national_calibration_fitted_at if n_calibrated else None
            ),
            per_lead=per_lead,
            eta_min=sample.eta_min,
            intensity_mm_h=sample.intensity_mm_h,
            observed_mm_h=sample.observed_mm_h,
            confidence=float(state.confidence) if state is not None else None,
        )

    @app.post("/lightning/strikes", response_model=StrikesAccepted, tags=["lightning"])
    async def lightning_strikes(
        payload: StrikesIn,
        request: Request,
        _: None = Depends(require_api_key),
    ) -> StrikesAccepted:
        """Ingest Blitzortung strikes pushed by HA into the rolling buffer.

        First (and only) write endpoint — ``require_api_key`` is a no-op under
        LAN-trust (``server.api_key`` unset), enforced when a key is set.
        """
        cfg: Config = request.app.state.config
        if not cfg.lightning.enabled:
            raise HTTPException(status_code=503, detail="lightning feature disabled")
        engine: CycleEngine = request.app.state.engine
        strikes = [
            LightningStrike(
                lat=s.lat,
                lon=s.lon,
                t=s.t if s.t.tzinfo else s.t.replace(tzinfo=timezone.utc),
            )
            for s in payload.strikes
        ]
        # add() now also persists to the strike archive (file I/O) — off-loop.
        added = await asyncio.to_thread(engine.lightning.add, strikes)
        return StrikesAccepted(accepted=added, buffer=engine.lightning.size())

    @app.get("/lightning/eta", response_model=LightningEtaResponse, tags=["lightning"])
    async def lightning_eta(
        request: Request,
        lat: float,
        lon: float,
        rings: str | None = None,
    ) -> LightningEtaResponse:
        """ETA of the threatening cell's leading edge to each ring of (lat, lon).

        ``rings`` is a comma-separated km list (default from config, e.g. "3,10").
        Compute runs off the event loop via ``asyncio.to_thread``.
        """
        cfg: Config = request.app.state.config
        lc = cfg.lightning
        if not lc.enabled:
            raise HTTPException(status_code=503, detail="lightning feature disabled")
        ring_list = _parse_rings(rings) if rings else list(lc.rings_km)
        params = _eta_params(lc)
        engine: CycleEngine = request.app.state.engine
        strikes = engine.lightning.snapshot()
        result = await asyncio.to_thread(
            compute_eta, strikes, lat, lon, ring_list, params
        )
        # Cross-cycle EMA smoothing (per target) to damp the whipsaw from
        # refitting sparse strikes each cycle. Only here (sensors/alerts) —
        # the debug map/clusters stay raw.
        if lc.smoothing_enabled:
            now = datetime.now(timezone.utc)
            key = EtaSmoother.key_for(lat, lon)
            prior_c, prior_e, alpha = engine.eta_smoother.prior(key, now)
            result, used_c, used_e = smooth_eta(
                result, prior_c, prior_e, alpha, lc.min_closing_kmh
            )
            if used_c is not None:
                engine.eta_smoother.store(key, used_c, used_e, now)
        return LightningEtaResponse.from_result(result, lat, lon)

    @app.get("/lightning/probability", tags=["lightning"])
    async def lightning_probability(
        request: Request, lat: float, lon: float,
    ) -> dict:
        """Calibrated areal probability P(≥1 strike within ring within lead) for
        (lat, lon) — the Phase-2 probabilistic forecast. Returns both the raw
        ensemble probability and the region-calibrated one per ring.

        Leads are the operational ones the calibrator was fit on (3 km→15 min,
        10 km→30 min). 10 km is the primary alert ring; 3 km a closer escalation.
        """
        cfg: Config = request.app.state.config
        lc = cfg.lightning
        if not lc.enabled:
            raise HTTPException(status_code=503, detail="lightning feature disabled")
        params = _eta_params(lc)
        engine: CycleEngine = request.app.state.engine
        strikes = engine.lightning.snapshot()
        ring_leads = prob_calibration.ring_leads()

        def _build() -> tuple[str, list[dict]]:
            # All CPU + the (first-call) calibrator file read run off the loop.
            result = strike_probability(strikes, lat, lon, ring_leads, params)
            region = prob_calibration.region_of(lat, lon)
            rings = [{
                "ring_km": rp.ring_km,
                "lead_min": rp.lead_min,
                "p_raw": round(rp.prob, 3),
                "p": round(prob_calibration.calibrate(region, rp.ring_km, rp.prob), 3),
            } for rp in result]
            return region, rings

        region, rings = await asyncio.to_thread(_build)
        return {
            "target_lat": lat,
            "target_lon": lon,
            "region": region,
            "n_strikes_buffer": len(strikes),  # the snapshot we actually computed on
            "rings": rings,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/lightning/clusters", tags=["lightning"])
    async def lightning_clusters(
        request: Request, lat: float, lon: float, rings: str | None = None,
    ) -> dict:
        """All clusters near (lat, lon) with motion + ETA — JSON for debugging."""
        cfg: Config = request.app.state.config
        lc = cfg.lightning
        if not lc.enabled:
            raise HTTPException(status_code=503, detail="lightning feature disabled")
        ring_list = _parse_rings(rings) if rings else list(lc.rings_km)
        params = _eta_params(lc)
        engine: CycleEngine = request.app.state.engine
        strikes = engine.lightning.snapshot()
        clusters = await asyncio.to_thread(
            summarize_clusters, strikes, lat, lon, params, None, tuple(ring_list),
        )
        return {
            "target_lat": lat,
            "target_lon": lon,
            "n_strikes_buffer": engine.lightning.size(),
            "n_clusters": len(clusters),
            "clusters": [asdict(c) for c in clusters],
        }

    @app.get("/lightning/map.png", tags=["lightning"])
    async def lightning_map(
        request: Request,
        pixel_lat: float | None = None,
        pixel_lon: float | None = None,
    ) -> Response:
        """Home-centred lightning map PNG (strikes + clusters + arrows + rings),
        matching the radar overlay. Pass pixel_lat/lon to mark the Pixel target."""
        cfg: Config = request.app.state.config
        lc = cfg.lightning
        if not lc.enabled:
            raise HTTPException(status_code=503, detail="lightning feature disabled")
        engine: CycleEngine = request.app.state.engine
        geo = engine.geo
        if geo is None:
            raise HTTPException(
                status_code=503, detail="no radar geometry yet — first cycle pending"
            )
        params = _eta_params(lc)
        home_lat, home_lon = cfg.home.lat, cfg.home.lon
        strikes = engine.lightning.snapshot()
        rings = tuple(lc.rings_km) + (params.relevance_radius_km,)

        def _build() -> bytes:
            clusters = summarize_clusters(
                strikes, home_lat, home_lon, params, None, tuple(lc.rings_km)
            )
            eta = compute_eta(strikes, home_lat, home_lon, list(lc.rings_km), params)
            header = (
                f"{eta.state} | cells {eta.n_cells} | strikes {eta.n_strikes}"
                f" | conf {eta.confidence:.2f}"
            )
            if eta.closing_kmh:
                header += f" | {eta.closing_kmh:.0f} km/h"
            return render_lightning_map(
                geo=geo,
                pixel_scale_m=geo.composite.xscale_m,
                home_lat=home_lat, home_lon=home_lon,
                strikes=strikes, clusters=clusters, rings_km=rings,
                pixel_lat=pixel_lat, pixel_lon=pixel_lon,
                zoom_km=100.0, output_px=600, basemap=engine.basemap, header=header,
            )

        png = await asyncio.to_thread(_build)
        return Response(
            content=png, media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    def _archive_or_503(request: Request):
        cfg: Config = request.app.state.config
        if not (cfg.lightning.enabled and cfg.lightning.archive_enabled):
            raise HTTPException(status_code=503, detail="strike archive disabled")
        arch = request.app.state.engine.strike_archive
        if arch is None:
            raise HTTPException(status_code=503, detail="strike archive unavailable")
        return arch

    @app.get("/lightning/archive/summary", tags=["lightning"])
    async def lightning_archive_summary(request: Request) -> dict:
        """Collection stats: totals, today/7d, per-day, per-region, bbox, last strike."""
        arch = _archive_or_503(request)
        summary, _points = await asyncio.to_thread(arch.snapshot)
        return summary

    @app.get("/lightning/archive/dashboard.html", tags=["lightning"])
    async def lightning_archive_dashboard(request: Request) -> Response:
        """Self-contained HTML monitoring dashboard (stat cards + per-day bars +
        Leaflet heat-map of all archived strikes). Mirror to HA /local + iframe."""
        arch = _archive_or_503(request)
        summary, points = await asyncio.to_thread(arch.snapshot)
        html = await asyncio.to_thread(render_dashboard_html, summary, points)
        return Response(content=html, media_type="text/html",
                        headers={"Cache-Control": "no-store"})

    # Registered before the public-mode snapshot below, so the gate sees
    # these routes and hides the two operator ones by default.
    app.include_router(
        build_push_router(
            config,
            store=push_store,
            public_key=push_public_key,
            service=push_service,
        ),
    )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):  # noqa: ARG001
        # Avoid leaking tracebacks over HTTP.
        _log.exception("unhandled_exception", path=request.url.path)
        return JSONResponse(
            {"detail": "internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    # Public mode: install the default-deny gate over the API route table
    # BEFORE the frontend is mounted, so the catch-all "/" mount is not part
    # of the snapshot (see module docstring, rule 3).
    if config.server.public_mode:
        install_public_mode_gate(app, list(app.routes))

    # Static frontend, if one is built into the image (Phase C §P2).
    if config.server.frontend_dir is not None:
        _mount_frontend(app, Path(config.server.frontend_dir))

    _log.info(
        "app_initialized",
        port=config.server.port,
        method=config.forecast.method,
        home=f"{config.home.lat:.4f},{config.home.lon:.4f}",
        public_mode=config.server.public_mode,
        frontend_dir=(
            str(config.server.frontend_dir)
            if config.server.frontend_dir is not None else None
        ),
    )
    return app


def _on_cycle_complete(app: Any, result: CycleResult) -> None:
    """Update app.state after a cycle. Called from the scheduler thread."""
    if result.state is not None:
        app.state.last_cycle_at = result.state.generated_at
        app.state.last_error = None
    else:
        app.state.last_error = result.error


def _parse_rings(rings: str) -> list[float]:
    """Parse a comma-separated km list (e.g. "3,10"). 400 on bad input."""
    try:
        out = [float(x) for x in rings.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid rings parameter")
    if not out:
        raise HTTPException(status_code=400, detail="empty rings parameter")
    return out


def _eta_params(lc: Any) -> EtaParams:
    """Build EtaParams from the LightningConfig section."""
    return EtaParams(
        buffer_window_min=lc.buffer_window_min,
        min_strikes=lc.min_strikes,
        cluster_eps_km=lc.cluster_eps_km,
        relevance_radius_km=lc.relevance_radius_km,
        leading_edge_recent_min=lc.leading_edge_recent_min,
        min_closing_kmh=lc.min_closing_kmh,
        min_fit_span_min=lc.min_fit_span_min,
    )


def require_api_key(request: Request) -> None:
    """Dependency used by write endpoints.

    No-op when ``server.api_key`` is unset (LAN trust). When set, requires
    ``Authorization: Bearer <key>``.

    Public mode carves out the allow-listed public surface (``/forecast``,
    ``/nowcast/*``, the three subscriber-facing ``/api/push/*`` routes):
    those must stay open to anonymous browsers even when a key is
    configured, because in public mode the key's job is to *unlock* the
    hidden surface, not to lock the published one.
    """
    config: Config = request.app.state.config
    if config.server.public_mode and _is_public_path(request.url.path):
        return
    expected = config.server.api_key
    if expected is None:
        return
    header = request.headers.get("Authorization", "")
    prefix = "Bearer "
    if not header.startswith(prefix):
        raise HTTPException(status_code=401, detail="missing bearer token")
    provided = header[len(prefix):]
    if not _consteq(provided, expected):
        raise HTTPException(status_code=401, detail="invalid bearer token")


# ---------------------------------------------------------------------------
# Public mode (website Phase C plan §P1) — see the module docstring
# ---------------------------------------------------------------------------

# The public surface. Exact paths plus prefixes; the static frontend is not
# listed because it is never part of the gated route snapshot (rule 3).
_PUBLIC_PATHS = frozenset({
    "/healthz",
    "/forecast",
    # Web Push, subscriber-facing only. Exact paths, never an
    # ``/api/push/`` prefix — ``/api/push/test`` and ``/api/push/stats``
    # are operator routes and must stay behind the bearer.
    "/api/push/config",
    "/api/push/subscribe",
    "/api/push/unsubscribe",
})
_PUBLIC_PREFIXES = ("/nowcast/",)

# Byte-identical to Starlette's own "no such route" body, so a gated route
# and a nonexistent one are indistinguishable from outside.
_NOT_FOUND_BODY = {"detail": "Not Found"}


def _is_public_path(path: str) -> bool:
    """True for the routes public mode serves without any bearer."""
    return path in _PUBLIC_PATHS or path.startswith(_PUBLIC_PREFIXES)


def _has_valid_bearer(config: Config, request: Request) -> bool:
    """True when the request carries ``Authorization: Bearer <api_key>``.

    False when no key is configured: in public mode that means the hidden
    surface has no unlock at all, which is the safe direction to fail.
    """
    expected = config.server.api_key
    if not expected:
        return False
    header = request.headers.get("Authorization", "")
    prefix = "Bearer "
    if not header.startswith(prefix):
        return False
    return _consteq(header[len(prefix):], expected)


def _matches_any_route(routes: Sequence[BaseRoute], scope: Scope) -> bool:
    """True when ``scope`` hits one of the snapshotted routes.

    Two subtleties, both about not confirming a route's existence through a
    status code other than 404:

    - ``Match.PARTIAL`` (path matches, method doesn't) counts, so probing
      ``GET /lightning/strikes`` gets a 404 rather than a 405.
    - the slash-toggled path counts too, because Starlette's
      ``redirect_slashes`` answers ``/state.json/`` with a 307 to
      ``/state.json`` — a redirect that only exists for real routes.
    """
    if _matches_exactly(routes, scope):
        return True
    path = scope.get("path", "")
    toggled = path[:-1] if path.endswith("/") and len(path) > 1 else path + "/"
    return toggled != path and _matches_exactly(routes, {**scope, "path": toggled})


def _matches_exactly(routes: Sequence[BaseRoute], scope: Scope) -> bool:
    for route in routes:
        match, _child_scope = route.matches(scope)
        if match is not Match.NONE:
            return True
    return False


def install_public_mode_gate(app: FastAPI, gated_routes: Sequence[BaseRoute]) -> None:
    """Install the single default-deny middleware for public mode.

    ``gated_routes`` is the route snapshot taken before the frontend mount.
    One middleware for the whole app — never a per-route repetition of the
    rule, so a route added later cannot forget to be gated.
    """
    routes = list(gated_routes)

    @app.middleware("http")
    async def _public_mode_gate(request: Request, call_next):
        path = request.url.path
        if not _is_public_path(path) and _matches_any_route(routes, request.scope):
            config: Config = request.app.state.config
            if not _has_valid_bearer(config, request):
                # Deliberately silent: no log of the probed path at info
                # level, no hint in the body, no WWW-Authenticate header.
                return JSONResponse(_NOT_FOUND_BODY, status_code=404)
        return await call_next(request)


# ---------------------------------------------------------------------------
# Static frontend (website Phase C plan §P2)
# ---------------------------------------------------------------------------

# SvelteKit's content-hashed build output. Everything under here carries a
# hash in the filename, so it can never change under a given URL.
_IMMUTABLE_PREFIXES = ("_app/",)
_IMMUTABLE_CACHE = "public, max-age=31536000, immutable"
# index.html and the service worker gate every deploy: they must be
# revalidated, or a stale shell pins old asset URLs after a release.
_NO_CACHE_FILES = frozenset({"index.html", "service-worker.js", "sw.js"})
_NO_CACHE = "no-cache"
# The Protomaps basemap: one large, effectively static file with no hash in
# its name. A day of caching keeps MapLibre's Range requests off the origin
# without pinning a stale basemap for a week.
_PMTILES_CACHE = "public, max-age=86400"
# Unhashed static assets (icons, web manifest, robots.txt): short cache, so
# a deploy is picked up within minutes.
_DEFAULT_CACHE = "public, max-age=300"


def _cache_control_for(path: str) -> str:
    """Cache-Control for one frontend asset, keyed on its build-output path."""
    name = path.rsplit("/", 1)[-1]
    if path.startswith(_IMMUTABLE_PREFIXES):
        return _IMMUTABLE_CACHE
    if name in _NO_CACHE_FILES or name.endswith(".html"):
        return _NO_CACHE
    if name.endswith(".pmtiles"):
        return _PMTILES_CACHE
    return _DEFAULT_CACHE


class SpaStaticFiles(StaticFiles):
    """Static files with SPA fallback and per-asset cache headers.

    Range requests are handled by Starlette's own ``FileResponse`` (206 +
    ``Content-Range``), which the ``.pmtiles`` basemap depends on — MapLibre
    reads it by byte range and never downloads the whole file.

    Fallback rule: a miss on an *extensionless* path serves ``index.html``
    (client-side routes like ``/about`` or ``/da/om``, and ``/`` itself,
    which Starlette normalises to ``"."``), while a miss on a path with a
    file suffix stays a 404. A missing asset must not come back as 200
    text/html — that turns a broken deploy into silently corrupt
    tiles/JSON, and it keeps the gate's 404s in company.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        served = path
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404 or Path(path).suffix:
                raise
            served = "index.html"
            response = await super().get_response(served, scope)
        response.headers["Cache-Control"] = _cache_control_for(served)
        return response


def _mount_frontend(app: FastAPI, frontend_dir: Path) -> None:
    """Mount the built frontend at ``/`` (registered last: it matches all).

    A configured-but-absent directory is a warning, not a crash: the image
    is built with the frontend optional (see ``sidecar/deploy/Dockerfile``),
    and an API-only service is far better than a service that won't boot.
    """
    if not frontend_dir.is_dir():
        _log.warning("frontend_dir_missing", path=str(frontend_dir))
        return
    if not (frontend_dir / "index.html").is_file():
        _log.warning("frontend_index_missing", path=str(frontend_dir))
    app.mount(
        "/",
        SpaStaticFiles(directory=frontend_dir, html=False),
        name="frontend",
    )
    _log.info("frontend_mounted", path=str(frontend_dir))


_FRAME_NAME_RE = re.compile(r"^(?:frame_\d{2,4}|loop)\.png$")


def _safe_frame_name(name: str) -> bool:
    """Allow only ``frame_NN.png`` (2-4 digits) or ``loop.png``. Defence-in-
    depth against path traversal even though FastAPI's parameter parser
    already strips ``/``. Anything else returns 400."""
    return bool(_FRAME_NAME_RE.match(name))


# Cycle-stamped national artifact names, mirroring national_artifacts'
# ``_STAMP_RE``: alnum-led stem, ``_YYYYMMDDHHMM`` stamp, .png or .json.
_NOWCAST_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]*_\d{12}\.(?:png|json)$")


def _safe_nowcast_name(name: str) -> bool:
    """Allow only cycle-stamped artifact names (``p_rain_20min_<stamp>.png``,
    ``eta_<stamp>.png``, ``overlay_now_<stamp>.png``,
    ``manifest_<stamp>.json``). Same defence-in-depth role as
    :func:`_safe_frame_name`; the stable ``manifest.json`` alias carries no
    stamp and is intentionally rejected — its dedicated route serves it with
    a short (non-immutable) cache lifetime."""
    return bool(_NOWCAST_NAME_RE.match(name))


# Kept as a module-level alias: the grid → JSON-safe float rule moved to
# national_sample.finite_or_none with the sampling arithmetic (Phase D).
_finite_or_none = finite_or_none


def _consteq(a: str, b: str) -> bool:
    """Constant-time string compare."""
    if len(a) != len(b):
        return False
    acc = 0
    for ca, cb in zip(a, b):
        acc |= ord(ca) ^ ord(cb)
    return acc == 0


__all__ = [
    "create_app",
    "require_api_key",
    "install_public_mode_gate",
    "SpaStaticFiles",
    "HealthResponse",
]
