"""ODIM HDF5 parser for DMI radar composites.

DMI deviates from the ODIM convention in three ways:

1. Scaling metadata (``gain``, ``offset``, ``nodata``, ``undetect``) and the
   quantity identifier (``product=DBZH``) live in the root ``/what`` group.
   The ODIM standard puts them in ``/datasetN/dataN/what``; DMI's
   ``/dataset1/data1/what`` is empty.
2. The quantity key is ``product`` rather than ``quantity``.
3. The Marshall–Palmer Z–R coefficients are published in ``/how`` as ``zr-a``
   and ``zr-b``, so we read them from the file instead of hardcoding.

The plan §6.2 hardcodes 200/1.6 — we expose the values from the file so any
future change on DMI's side flows through automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class RadarComposite:
    """A parsed DMI radar composite frame in dBZ.

    ``reflectivity_dbz`` is a 2D float32 array with NaN at ``nodata`` pixels.
    ``undetect`` pixels are set to ``-inf`` so they sort below any valid dBZ
    but remain distinguishable from missing data (callers can promote to NaN
    or a sentinel rain rate as needed).
    """

    reflectivity_dbz: np.ndarray
    timestamp_utc: datetime
    projection: str
    xscale_m: float
    yscale_m: float
    corners_lonlat: dict[str, tuple[float, float]]
    zr_a: float
    zr_b: float
    quantity: str
    source_path: Path


def parse_composite(path: Path) -> RadarComposite:
    """Parse a DMI composite HDF5 file into a RadarComposite."""
    with h5py.File(path, "r") as h5:
        what = _attrs(h5["/what"])
        where = _attrs(h5["/where"])
        how = _attrs(h5.get("/how"))

        # Plan §3.3 contract: always read scaling from the file, never hardcode.
        gain = float(what["gain"])
        offset = float(what["offset"])
        nodata_raw = what["nodata"]
        undetect_raw = what["undetect"]

        ds = h5["/dataset1/data1"]
        raw = ds["data"][...]
        # physical = raw * gain + offset (ODIM convention; values match plan §3.3)
        dbz = raw.astype(np.float32) * np.float32(gain) + np.float32(offset)
        dbz[raw == nodata_raw] = np.nan
        dbz[raw == undetect_raw] = -np.inf

        ts = _parse_timestamp(what["date"], what["time"])
        corners = {
            "LL": (float(where["LL_lon"]), float(where["LL_lat"])),
            "LR": (float(where["LR_lon"]), float(where["LR_lat"])),
            "UL": (float(where["UL_lon"]), float(where["UL_lat"])),
            "UR": (float(where["UR_lon"]), float(where["UR_lat"])),
        }
        return RadarComposite(
            reflectivity_dbz=dbz,
            timestamp_utc=ts,
            projection=str(where["projdef"]),
            xscale_m=float(where["xscale"]),
            yscale_m=float(where["yscale"]),
            corners_lonlat=corners,
            zr_a=float(how.get("zr-a", 200.0)),
            zr_b=float(how.get("zr-b", 1.6)),
            quantity=str(what.get("product", "")),
            source_path=path,
        )


def _attrs(group: h5py.Group | None) -> dict[str, Any]:
    """Decode HDF5 attrs into plain Python: bytes → str, length-1 arrays → scalar."""
    if group is None:
        return {}
    out: dict[str, Any] = {}
    for k, v in group.attrs.items():
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="replace")
        elif isinstance(v, np.ndarray):
            if v.size == 1:
                v = v.item()
            else:
                v = v.tolist()
        out[k] = v
    return out


def _parse_timestamp(date: Any, time: Any) -> datetime:
    """ODIM date is YYYYMMDD; time is HHMMSS (sometimes with trailing 01 in DMI)."""
    date_s = str(date)
    time_s = str(time).zfill(6)
    # DMI sometimes appends 01 seconds to the time (e.g. 193501 for 19:35:00).
    hour = int(time_s[0:2])
    minute = int(time_s[2:4])
    second = int(time_s[4:6])
    return datetime(
        int(date_s[0:4]), int(date_s[4:6]), int(date_s[6:8]),
        hour, minute, second, tzinfo=timezone.utc,
    )
