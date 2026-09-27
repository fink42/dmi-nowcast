"""Frame rendering — wraps :func:`dmi_nowcast_core.render.render_loop_png`.

Builds the LoopFrame list (3 observed + N forecasts), calls render_loop_png
with ``individual_frames_dir`` pointing at ``data_dir/frames/`` so per-frame
PNGs + a ``frames.json`` manifest land on disk. The HA integration mirrors
those into ``/config/www/dmi_radar/`` so the existing Lovelace card works
unchanged (Lovelace path option A in plan §5.5).
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import structlog

from dmi_nowcast_core.advect import advect_field_series
from dmi_nowcast_core.parse import RadarComposite
from dmi_nowcast_core.render import LoopFrame, render_loop_png
from dmi_nowcast_core.transform import dbz_to_rain_rate

_log = structlog.get_logger(__name__)

# Forecast leads (minutes from wall-clock now) shown in the animated loop.
# Lead 0 closes the frame-age gap so the loop transitions smoothly from
# "obs" to "now".
LOOP_FORECAST_LEADS_MIN: tuple[int, ...] = (0, 5, 10, 15, 20, 25, 30)


def loop_horizons_minutes(frame_age_min: float) -> list[float]:
    """Advection horizons (minutes from RADAR-FRAME time) of the loop's
    forecast frames: the frame age plus each loop lead.

    The one expression both :func:`render_frames` and its caller use, so
    the caller can tell — by exact float equality — whether the fields its
    own advection series already produced are the loop's frames.
    """
    age = max(0.0, frame_age_min)
    return [age + float(lead) for lead in LOOP_FORECAST_LEADS_MIN]


def render_frames(
    *,
    composites: list[RadarComposite],
    rain_now: np.ndarray,
    vy: np.ndarray,
    vx: np.ndarray,
    dt_min: float,
    frame_age_min: float,
    geo,
    home_lat: float,
    home_lon: float,
    radius_km: float,
    out_dir: Path,
    now_stats_subline: str | None,
    disc_motion_dy_per_min: float,
    disc_motion_dx_per_min: float,
    disc_motion_speed_kmh: float,
    disc_motion_bearing_from: str,
    basemap: Any = None,
    forecast_fields: Mapping[int, np.ndarray] | None = None,
) -> tuple[bytes, float]:
    """Render the animated PNG + per-frame PNGs + manifest.

    Returns ``(apng_bytes, render_ms)``. The manifest and per-frame PNGs
    land in ``out_dir`` as a side effect — the bytes are returned only
    for callers who also want to expose the APNG over HTTP.

    ``forecast_fields`` maps each loop lead (:data:`LOOP_FORECAST_LEADS_MIN`)
    to its already-advected field. The cycle passes the fields its own
    per-lead series produced when that series' first horizons are exactly
    :func:`loop_horizons_minutes` — same field, same flow, same ``dt_min``,
    same horizons, so the same arrays this function would integrate
    (bit-identical; ~5 s of native-grid advection saved per cycle). Missing
    or incomplete → the loop is advected here, as before.
    """
    t0 = time.perf_counter()
    now_utc = datetime.now(timezone.utc)
    composite_now = composites[-1]

    frames: list[LoopFrame] = []

    # Observed frames — oldest first, latest held last.
    for c in composites:
        rr = (
            rain_now if c is composite_now
            else dbz_to_rain_rate(c.reflectivity_dbz, zr_a=c.zr_a, zr_b=c.zr_b)
        )
        is_latest = c is composite_now
        offset_min = (now_utc - c.timestamp_utc).total_seconds() / 60.0
        if is_latest:
            label = "obs"
            duration_ms = 1000
        else:
            # Round to whole minutes so 5/10-min cadence jitter doesn't
            # show as ugly fractional labels.
            label = f"−{round(offset_min)} min"
            duration_ms = 300
        frames.append(LoopFrame(
            rain_rate_mm_h=rr,
            timestamp_utc=c.timestamp_utc,
            label=label, kind="observed",
            duration_ms=duration_ms,
        ))

    # Forecast frames at lead 0, 5, …, 30 min from wall-clock now. The
    # 0-min lead advects by frame_age so playback lands on "now". One
    # ascending series = one trajectory integration for the whole loop.
    last_lead = LOOP_FORECAST_LEADS_MIN[-1]
    if forecast_fields is not None and all(
        lead in forecast_fields for lead in LOOP_FORECAST_LEADS_MIN
    ):
        advected = [forecast_fields[lead] for lead in LOOP_FORECAST_LEADS_MIN]
    else:
        advected = advect_field_series(
            rain_now, vy, vx,
            horizons_minutes=loop_horizons_minutes(frame_age_min),
            dt_minutes=dt_min,
        )
    for lead, field in zip(LOOP_FORECAST_LEADS_MIN, advected):
        ts = now_utc + timedelta(minutes=lead)
        if lead == 0:
            label, duration_ms = "now", 1500
        elif lead == last_lead:
            label, duration_ms = f"+{lead} min", 1000
        else:
            label, duration_ms = f"+{lead} min", 400
        frames.append(LoopFrame(
            rain_rate_mm_h=field,
            timestamp_utc=ts,
            label=label, kind="forecast",
            duration_ms=duration_ms,
        ))

    apng = render_loop_png(
        frames,
        composite=composite_now,
        geo=geo,
        home_lat=home_lat, home_lon=home_lon,
        radius_km=radius_km,
        zoom_km=100.0, output_px=500,
        now_stats_subline=now_stats_subline,
        motion_dy_per_min=disc_motion_dy_per_min,
        motion_dx_per_min=disc_motion_dx_per_min,
        motion_speed_kmh=disc_motion_speed_kmh,
        motion_bearing_from=disc_motion_bearing_from,
        basemap=basemap,
        individual_frames_dir=out_dir,
    )
    render_ms = (time.perf_counter() - t0) * 1000
    return apng, render_ms
