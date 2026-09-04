"""Pydantic models for ``state.json``.

The contract between sidecar and the HA integration. ``SCHEMA_VERSION`` is
bumped whenever a breaking field change ships, and the integration warns
when it sees a version it doesn't recognise. See plan §6 for the human-
readable schema description.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1


class RadarBlock(BaseModel):
    """Provenance of the input radar frame.

    ``data_age_minutes`` is measured at the moment ``state.json`` was
    written; consumers (HA) should not extrapolate freshness from this
    alone — also check ``generated_at`` and apply their own clock to
    detect stale states.
    """
    latest_ts: datetime
    data_age_minutes: float
    source: str = "dmi-radardata-500m-composite"


class HomeBlock(BaseModel):
    lat: float
    lon: float
    radius_km: float


class NowBlock(BaseModel):
    rain_rate_mm_h: float
    rain_rate_p90_mm_h: float
    raining: bool
    # "wet" or "dry" — exposes the hysteresis state explicitly so the HA
    # binary_sensor's last_changed reflects state-machine transitions, not
    # the raw threshold crossing each cycle.
    raining_hysteresis_state: Literal["wet", "dry"]


class PerLeadEntry(BaseModel):
    lead_min: int
    rain_rate_mm_h: float
    # ``p_rain`` is the uncalibrated deterministic Yes/No probability;
    # ``p_calibrated`` is the isotonic-mapped version. They differ only
    # when calibration curves are loaded. Both keep these exact semantics
    # even with the ensemble running — the HA integration reads them.
    p_rain: float
    p_calibrated: float
    # Raw STEPS ensemble exceedance fraction at this lead (fraction of
    # members whose max-in-disc rate crosses the threshold by then).
    # Deliberately NOT passed through the isotonic curves — those were
    # fitted on the binary deterministic forecast and would be a category
    # error here (website Phase A plan §A0, "calibration honesty").
    # Null when the ensemble didn't run this cycle.
    p_ensemble: float | None = None


class ForecastBlock(BaseModel):
    method: Literal["farneback", "tvl1", "mean-motion"]
    rain_incoming: bool
    # Minutes from generated_at; null if no rain expected within horizon.
    eta_minutes: float | None
    # P25/P75 window for ETA — null when ensemble unavailable.
    eta_p50_window_min: tuple[float, float] | None
    peak_intensity_mm_h: float
    peak_lead_min: int
    per_lead: list[PerLeadEntry]


class ProbabilisticBlock(BaseModel):
    """STEPS ensemble summary at home (additive; website Phase A plan §A0).

    Phase B (§B4): when the pooled national curves are loaded, the served
    ``p_ensemble`` fractions are isotonic-calibrated per lead and
    ``calibrated`` flips to True — but only when EVERY served home lead
    went through a curve. Leads without a curve are served raw (never
    interpolated between neighbouring leads' curves); ``calibrated_leads``
    names exactly which leads were calibrated so a partial curve file
    can't lie. Without curves everything stays raw member fractions with
    ``calibrated: false``, exactly the pre-B4 behaviour.
    """
    n_members: int
    calibrated: bool = False
    # P25/P75 first-exceedance window, minutes from ``generated_at``
    # (frame-age corrected); null when no member predicts rain within the
    # horizon. Mirrors ``ForecastBlock.eta_p50_window_min``. NOT calibrated
    # (it's a time window, not a probability).
    eta_p50_window_min: tuple[float, float] | None = None
    # ``fitted_at`` of the national curve file whose curves were applied to
    # ``p_ensemble``; null when no home lead was calibrated this cycle.
    calibration_fitted_at: datetime | None = None
    # The home leads whose ``p_ensemble`` went through a national curve
    # (subset of the served leads, possibly empty). Null when no national
    # curves are loaded at all — i.e. the pre-B4 raw path.
    calibrated_leads: list[int] | None = None


class MotionBlock(BaseModel):
    """Disc-area motion vector, in grid-pixel units per minute.

    Positive ``dy`` means southward, positive ``dx`` eastward (image
    convention). ``speed_km_per_h`` converts via the composite's pixel
    resolution. ``bearing_deg_from`` is the meteorological "wind from"
    direction (0° = from north).
    """
    dy_px_per_min: float
    dx_px_per_min: float
    speed_km_per_h: float
    bearing_deg_from: float


class CalibrationBlock(BaseModel):
    # Null when curves aren't loaded — uncalibrated probabilities flow through.
    fitted_at: datetime | None
    n_events: int | None
    brier_before: float | None
    brier_after: float | None


class DiagnosticsBlock(BaseModel):
    cycle_ms: float
    fetch_ms: float
    compute_ms: float
    render_ms: float
    # Wall time of the STEPS ``run_ensemble`` call; 0.0 when the ensemble
    # was disabled, skipped, or failed (website Phase A plan §A0 budget gate).
    ensemble_ms: float = 0.0
    # National products reduction + artifact write time; 0.0 when national
    # products are disabled or the ensemble didn't run (plan §A5 gate).
    national_ms: float = 0.0
    # Total bytes of national artifacts written this cycle (PNGs + manifests);
    # 0 when nothing was written (plan §A2 size gate).
    artifact_bytes: int = 0


class ForecastPointLead(BaseModel):
    """One per-lead entry in ``GET /forecast`` (website Phase A plan §A3).

    ``p_rain`` is the STEPS ensemble exceedance fraction at the queried
    pixel — the national-grid counterpart of :class:`PerLeadEntry`'s
    ``p_ensemble``. Since Phase B (§B4) the served grid IS the calibrated
    one when national curves cover the lead (the response's ``calibrated``
    flag says whether every served lead was); leads without a curve stay
    raw. Null when the pixel is NaN in the product grid (off-composite /
    nodata).
    """
    lead_min: int
    p_rain: float | None


class ForecastPointResponse(BaseModel):
    """Root payload for ``GET /forecast?lat=&lon=`` (website Phase A plan §A3).

    Point lookup into the latest national product grids; mirrors the
    ``state.json`` forecast dialect so Phase C/D consumers and the HA client
    speak one language. All timestamps UTC ISO 8601 with explicit offset;
    conversion to local time happens at the consumer (CLAUDE.md UTC
    contract).
    """
    # Queried coordinates, echoed back.
    lat: float
    lon: float
    # Radar timestamp of the composite the products were computed from.
    radar_ts_utc: datetime
    n_members: int
    # True only when every served lead's grid went through a national curve
    # (§B4) — mirrors ProbabilisticBlock.calibrated. False on the raw path
    # and under partial curve coverage.
    calibrated: bool = False
    # ``fitted_at`` of the national curve file applied to the served grids;
    # null when no served lead was calibrated.
    calibration_fitted_at: datetime | None = None
    per_lead: list[ForecastPointLead]
    # Minutes from now until rain reaches the pixel; null when no rain is
    # expected within the forecast horizon (NaN in the ETA grid).
    eta_min: float | None
    # Ensemble-median rain rate at the pixel's ETA step; null with no ETA.
    intensity_mm_h: float | None
    # OBSERVED rain rate at the pixel right now (p90 over the ~2 km product
    # block of the newest composite), mm/h. The only non-forecast field
    # here: no ETA/probability product can say whether it is raining at the
    # point *at this moment*, because the ensemble's first timestep is
    # already ~10 min out. Null when the pixel is nodata or the cycle
    # published no observed grid. Additive (default null), so pinned
    # clients are unaffected.
    observed_mm_h: float | None = None
    # Global confidence scalar from the latest ``state.json`` (Phase A keeps
    # confidence global, plan §A1); null when no state is available yet.
    confidence: float | None


class State(BaseModel):
    """Root payload for ``GET /state.json``."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=SCHEMA_VERSION)
    generated_at: datetime
    radar: RadarBlock
    home: HomeBlock
    now: NowBlock
    forecast: ForecastBlock
    # STEPS ensemble summary; null when the ensemble didn't run this cycle
    # (disabled, too few frames, or a run failure). Additive — schema_version
    # stays 1 (website Phase A plan §A0).
    probabilistic: ProbabilisticBlock | None = None
    motion: MotionBlock
    confidence: float
    calibration: CalibrationBlock
    diagnostics: DiagnosticsBlock
    # When the last cycle errored, the sidecar serves the *previous* good
    # state.json with ``last_error`` set on the version we'd-have-emitted.
    # Consumers can render a stale-data badge.
    last_error: str | None = None
