"""Reflectivity → rain rate conversion with hail cap.

Plan §6.2 and §14 contract:
- Cap dBZ at 53 to suppress hail contamination (a 60 dBZ hail core would imply
  ~600 mm/h otherwise).
- Cap rain rate at 100 mm/h as a final guard.
- Marshall–Palmer Z–R: ``Z = zr_a * R^zr_b`` → ``R = (Z / zr_a)^(1/zr_b)``.
- DMI publishes ``zr_a=200``, ``zr_b=1.6`` in ``/how`` (see ``parse.py``); these
  are the Marshall–Palmer defaults but we read from the file so any future
  change on DMI's side flows through automatically.

Sentinel handling mirrors ``parse.py``:
- NaN (was ``nodata``) → NaN rain rate (missing stays missing).
- ``-inf`` (was ``undetect``) → 0 mm/h (observed dry).
"""
from __future__ import annotations

import numpy as np

DBZ_HAIL_CAP = 53.0
RAIN_RATE_CAP_MM_H = 100.0


def dbz_to_rain_rate(
    dbz: np.ndarray,
    *,
    zr_a: float = 200.0,
    zr_b: float = 1.6,
    dbz_cap: float = DBZ_HAIL_CAP,
    rain_cap_mm_h: float = RAIN_RATE_CAP_MM_H,
) -> np.ndarray:
    """Convert reflectivity (dBZ) to rain rate (mm/h)."""
    arr = np.asarray(dbz, dtype=np.float32)
    capped = np.minimum(arr, np.float32(dbz_cap))
    # 10**(-inf/10) = 0; NaN propagates.
    with np.errstate(invalid="ignore"):
        z = np.power(np.float32(10.0), capped / np.float32(10.0))
        rate = np.power(z / np.float32(zr_a), np.float32(1.0 / zr_b))
    rate = np.minimum(rate, np.float32(rain_cap_mm_h))
    return rate
