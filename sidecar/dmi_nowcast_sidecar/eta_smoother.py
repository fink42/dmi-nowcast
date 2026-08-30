"""Cross-cycle EMA smoothing state for the lightning ETA.

Holds, per target, the last smoothed ``(closing_kmh, leading_edge_km)`` plus a
timestamp. On each ETA request it returns the prior values and a time-aware
blend weight ``alpha = 1 - exp(-dt/tau)`` (so irregular poll gaps are handled);
the pure :func:`dmi_nowcast_core.lightning.smooth_eta` does the actual blend and
the caller persists the result. Thread-safe — the ETA compute runs in a thread.
"""
from __future__ import annotations

import math
import threading
from datetime import datetime

from .config import LightningConfig


class EtaSmoother:
    def __init__(self, config: LightningConfig) -> None:
        self._c = config
        self._lock = threading.Lock()
        # key -> (closing_kmh, edge_km, ts)
        self._state: dict[str, tuple[float, float, datetime]] = {}

    @staticmethod
    def key_for(lat: float, lon: float) -> str:
        # ~1 km granularity: stable for home and a stationary Pixel; a moving
        # phone gets a new key (and a fresh, geometry-appropriate estimate).
        return f"{round(lat, 2)},{round(lon, 2)}"

    def prior(self, key: str, now: datetime) -> tuple[float | None, float | None, float]:
        """(prior_closing_kmh, prior_edge_km, alpha) for this update.

        Returns ``(None, None, 1.0)`` when there's no usable prior (first sight
        or gap > ``smoothing_max_gap_min``) — i.e. use the raw estimate."""
        with self._lock:
            entry = self._state.get(key)
        if entry is None:
            return None, None, 1.0
        closing, edge, ts = entry
        dt_min = (now - ts).total_seconds() / 60.0
        if dt_min <= 0 or dt_min > self._c.smoothing_max_gap_min:
            return None, None, 1.0
        tau = self._c.smoothing_tau_min
        alpha = 1.0 if tau <= 0 else 1.0 - math.exp(-dt_min / tau)
        return closing, edge, alpha

    def store(self, key: str, closing_kmh: float, edge_km: float, now: datetime) -> None:
        with self._lock:
            self._state[key] = (closing_kmh, edge_km, now)


__all__ = ["EtaSmoother"]
