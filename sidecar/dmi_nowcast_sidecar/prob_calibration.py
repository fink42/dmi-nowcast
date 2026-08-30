"""Load the bundled per-region lightning-probability calibrators and apply them.

``strike_probability`` (Phase 2) produces a *raw* ensemble probability that
discriminates well (high ROC-AUC) but is over-confident (negative raw BSS). The
calibrators here — isotonic curves fit on the backtest, one per (region, ring) —
map raw P → an honest frequency. Bundled as ``lightning_prob_calibration.json``
(fit from ``scripts/lightning_backtest.py`` output).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from dmi_nowcast_core.calibrate import IsotonicCalibrator
from dmi_nowcast_core.regions import region_of  # re-exported for the app

_BUNDLE = Path(__file__).with_name("lightning_prob_calibration.json")
# Fallback leads if the bundle is missing/old (must match what curves were fit on).
_DEFAULT_RING_LEADS = [(3.0, 15.0), (10.0, 30.0)]


@lru_cache(maxsize=1)
def _bundle() -> dict:
    """Parse the bundled JSON once. Missing/malformed → ``{}`` (→ raw passthrough);
    note ``lru_cache`` does not cache exceptions, so an unhandled error here would
    re-raise on every request."""
    try:
        return json.loads(_BUNDLE.read_text()) if _BUNDLE.exists() else {}
    except (OSError, ValueError):
        return {}


@lru_cache(maxsize=1)
def _curves() -> dict[str, dict[float, IsotonicCalibrator]]:
    out: dict[str, dict[float, IsotonicCalibrator]] = {}
    for region, rings in _bundle().get("curves", {}).items():
        try:
            out[region] = {
                float(ring): IsotonicCalibrator(
                    raw_breakpoints=c["raw_breakpoints"],
                    calibrated_values=c["calibrated_values"],
                )
                for ring, c in rings.items()
            }
        except (KeyError, TypeError, ValueError):
            continue
    return out


def ring_leads() -> list[tuple[float, float]]:
    """The (ring_km, lead_min) pairs the bundled calibrators were fit on — the
    single source of truth for what the probability endpoint should request, so it
    can't drift from the curves."""
    leads = _bundle().get("leads", {})
    try:
        pairs = sorted((float(r), float(l)) for r, l in leads.items())
    except (ValueError, AttributeError):
        pairs = []
    return pairs or list(_DEFAULT_RING_LEADS)


def calibrate(region: str, ring_km: float, raw: float) -> float:
    """Calibrated probability for ``raw`` at (region, ring); falls back to the
    pooled curve, then to raw if no calibrator is bundled. Clamped to [0, 1]."""
    curves = _curves()
    cal = curves.get(region, {}).get(float(ring_km)) or curves.get("pooled", {}).get(float(ring_km))
    p = float(cal.predict(float(raw))) if cal is not None else float(raw)
    return max(0.0, min(1.0, p))


__all__ = ["region_of", "ring_leads", "calibrate"]
