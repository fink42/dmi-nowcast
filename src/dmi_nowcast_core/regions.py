"""Region boxes — the single source of truth for Denmark / Alps classification.

Used by the archive aggregation (`strike_archive`), the probability calibrator
(`prob_calibration`), and the backtest harness (`lightning_backtest`). Keeping one
copy avoids the silent drift where a strike is *counted* in one region but
*calibrated* with another's curve (review finding M5)."""
from __future__ import annotations

# name -> (lat_min, lat_max, lon_min, lon_max)
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "Denmark": (53.0, 58.5, 6.5, 16.0),
    "Alps": (44.0, 48.0, 5.0, 11.5),
}


def region_of(lat: float, lon: float) -> str:
    """The region whose box contains (lat, lon), or ``"Other"``."""
    for name, (a0, a1, o0, o1) in REGIONS.items():
        if a0 <= lat <= a1 and o0 <= lon <= o1:
            return name
    return "Other"


__all__ = ["REGIONS", "region_of"]
