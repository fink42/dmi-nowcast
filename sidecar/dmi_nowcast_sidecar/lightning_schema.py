"""Pydantic models for the lightning ETA endpoints.

Request: a batch of strikes POSTed by HA (pushed from the Blitzortung
``geo_location.lightning_strike_*`` entities). Response: the cell-tracking
ETA for one target point. The actual maths lives in
``dmi_nowcast_core.lightning``; these are just the HTTP contract.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from dmi_nowcast_core.lightning import EtaResult


class StrikeIn(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    t: datetime  # ISO-8601 in JSON; tz-aware UTC expected


class StrikesIn(BaseModel):
    strikes: list[StrikeIn]


class StrikesAccepted(BaseModel):
    accepted: int
    buffer: int


class RingEtaOut(BaseModel):
    ring_km: float
    eta_min: float | None
    inside: bool


class LightningEtaResponse(BaseModel):
    target_lat: float
    target_lon: float
    state: Literal[
        "approaching", "receding", "stalled", "inside_ring", "insufficient_data"
    ]
    rings: list[RingEtaOut]
    leading_edge_km: float | None
    closing_kmh: float | None
    cell_speed_kmh: float | None
    cell_bearing_deg: float | None
    n_strikes: int
    n_cells: int
    confidence: float

    @classmethod
    def from_result(
        cls, result: EtaResult, target_lat: float, target_lon: float
    ) -> "LightningEtaResponse":
        return cls(
            target_lat=target_lat,
            target_lon=target_lon,
            state=result.state,  # type: ignore[arg-type]
            rings=[
                RingEtaOut(ring_km=r.ring_km, eta_min=r.eta_min, inside=r.inside)
                for r in result.rings
            ],
            leading_edge_km=result.leading_edge_km,
            closing_kmh=result.closing_kmh,
            cell_speed_kmh=result.cell_speed_kmh,
            cell_bearing_deg=result.cell_bearing_deg,
            n_strikes=result.n_strikes,
            n_cells=result.n_cells,
            confidence=result.confidence,
        )


__all__ = [
    "StrikeIn", "StrikesIn", "StrikesAccepted", "RingEtaOut", "LightningEtaResponse",
]
