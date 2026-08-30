"""FastAPI app + routes.

Endpoints (phase B):

- ``GET /healthz`` — liveness; reports ``last_cycle`` timestamp.
- ``GET /state.json`` — latest nowcast state per :class:`State` schema.

Website Phase A (§A3) adds the national surface:

- ``GET /nowcast/manifest.json`` — latest national-cycle manifest.
- ``GET /nowcast/{filename}`` — cycle-stamped artifact PNGs / manifests.
- ``GET /forecast?lat=&lon=`` — point lookup into the national grids.

Auth: write endpoints check ``Authorization: Bearer <key>`` against
``config.server.api_key`` when set. Legacy read endpoints are always open
since the HA integration polls them unauthenticated; the §A3 endpoints go
through the same optional bearer (``require_api_key`` — a no-op under
LAN trust, enforced when a key is set).
"""
from __future__ import annotations

import asyncio
import math
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

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
        )
        app.state.scheduler = local_scheduler
        if auto_start_scheduler:
            await local_scheduler.start(run_immediately=True)
        try:
            yield
        finally:
            if auto_start_scheduler:
                await local_scheduler.shutdown()

    app = FastAPI(
        title="dmi-nowcast-sidecar",
        version=__version__,
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

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
        # lat/lon → native grid → ÷ downsample factor → nearest product pixel,
        # exactly the aggregate_at_home convention (home.row / f).
        idx = geo.lonlat_to_grid(lon, lat)
        f = products.downsample_factor
        row = int(round(idx.row / f))
        col = int(round(idx.col / f))
        h, w = products.eta_min.shape
        if not (0 <= row < h and 0 <= col < w):
            raise HTTPException(
                status_code=400,
                detail="coordinates outside the radar composite grid",
            )
        per_lead = [
            ForecastPointLead(
                lead_min=int(lead),
                p_rain=_finite_or_none(products.p_rain[lead][row, col]),
            )
            for lead in products.leads_min
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
            eta_min=_finite_or_none(products.eta_min[row, col]),
            intensity_mm_h=_finite_or_none(products.intensity_mm_h[row, col]),
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

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):  # noqa: ARG001
        # Avoid leaking tracebacks over HTTP.
        _log.exception("unhandled_exception", path=request.url.path)
        return JSONResponse(
            {"detail": "internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    _log.info(
        "app_initialized",
        port=config.server.port,
        method=config.forecast.method,
        home=f"{config.home.lat:.4f},{config.home.lon:.4f}",
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
    """
    config: Config = request.app.state.config
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


def _finite_or_none(value: Any) -> float | None:
    """Grid sample → JSON-safe float; NaN/±inf (nodata, off-composite) → None."""
    v = float(value)
    return v if math.isfinite(v) else None


def _consteq(a: str, b: str) -> bool:
    """Constant-time string compare."""
    if len(a) != len(b):
        return False
    acc = 0
    for ca, cb in zip(a, b):
        acc |= ord(ca) ^ ord(cb)
    return acc == 0


__all__ = ["create_app", "require_api_key", "HealthResponse"]
