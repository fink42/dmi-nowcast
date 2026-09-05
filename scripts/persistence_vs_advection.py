"""Persistence vs. advection: which predicts "wet here in 10/20 min" better?

Why this study exists
---------------------
The site's "it is raining here now" headline is read off the newest radar
composite, which is 14-24 min old by the time a viewer sees it (fullRange
composites every 10 min, ~12 min DMI publication delay, a serving cycle up
to 10 min wide). The replacement reads the radar field *advected to
wall-clock now* on the production motion field. Two live events on
2026-09-04 suggested the raw observation beat the advected field on the
**trailing** edge of a rain area (the point was still wet 14 min later
although advection had moved the cell on); 2026-09-05 07:20Z suggested the
opposite. This script settles it against the archive.

The question, per horizon h in {10, 20} min and per product pixel:

    obs(t) wet?  x  advected(t -> t+h) wet?  ->  truth(t+h) wet?

with two cells of particular interest:

    trailing edge:  obs wet, advected dry  ->  P(truth wet)
    leading edge:   obs dry, advected wet  ->  P(truth wet)

If advection loses on the trailing edge that is a bias in the motion field
(or in how the field is completed off the echo), not a presentation bug.

Production parity
-----------------
Every step mirrors ``sidecar/dmi_nowcast_sidecar/compute.py::_compute_sync``
(the deterministic half; STEPS is deliberately not run -- this is about the
field the loop draws):

* ``parse_composite`` -> ``dbz_to_rain_rate`` with each file's own Z-R
  coefficients and the library's reflectivity/rain caps;
* ``dense_flow(prev_dbz, now_dbz)`` (OpenCV Farneback, production defaults);
* ``complete_flow(vy, vx, rain_now, pixel_km=xscale_m/1000,
  support_threshold_mm_h=0.5)`` -- the R5 far-field relaxation toward bulk
  storm motion, which is part of the production flow both consumers see;
* ``nan_to_num`` then clip to +-``MAX_PX_PER_FRAME`` (30.0, copied from
  ``compute.py::_MAX_PX_PER_FRAME``);
* ``advect_field_series(rain_now, vy, vx, horizons_minutes=[10, 20],
  dt_minutes=dt)`` where ``dt`` is the measured spacing of the two input
  frames (10 min on the fullRange-only feed).

One deliberate difference from production: production advects to
``lead + frame_age_min`` because it projects forward from *now*. Here the
horizons are measured from the radar frame time, because the truth frames
are at radar times. That is the same integrator, only a different target.

Reduction and truth
-------------------
Every field -- rain at t, advected +10/+20, and the truth frames -- is
reduced with ``observed_rain_grid(field, downsample_factor=4)``: block-wise
p90 on the x4 product grid, exactly the grid and statistic the service
samples for a point. Wet := >= 0.5 mm/h (the live
``forecast.rain_threshold_mm_h``). Only product pixels finite in *all* five
grids of a case are counted, so pixels whose backward trajectory left the
composite are neither "wet" nor "dry" but excluded.

Usage
-----
Scout a cheap subset for rainy days (one frame every 2 h is plenty)::

    python scripts/persistence_vs_advection.py --archive-dir DIR --scout \\
        --out-json scout.json

Run the study over whole days::

    python scripts/persistence_vs_advection.py --archive-dir DIR \\
        --days 2026-08-11,2026-08-24 --workers 6 \\
        --out-json results.json --out-md report.md

Case studies at a point::

    python scripts/persistence_vs_advection.py --archive-dir DIR \\
        --point 55.352,10.347 --case 202609050720 --case-leads 10,20,23 \\
        --out-json cases.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

# The repo layout puts the algorithm library under src/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from dmi_nowcast_core.advect import advect_field_series  # noqa: E402
from dmi_nowcast_core.dense_flow import complete_flow, dense_flow  # noqa: E402
from dmi_nowcast_core.geo import CompositeGeo  # noqa: E402
from dmi_nowcast_core.national import observed_rain_grid  # noqa: E402
from dmi_nowcast_core.parse import RadarComposite, parse_composite  # noqa: E402
from dmi_nowcast_core.transform import dbz_to_rain_rate  # noqa: E402

# --- production constants, copied so the study does not import the sidecar ---
MAX_PX_PER_FRAME = 30.0          # compute.py::_MAX_PX_PER_FRAME
RAIN_THRESHOLD_MM_H = 0.5        # config.py ForecastConfig.rain_threshold_mm_h
DOWNSAMPLE_FACTOR = 4            # config.py NationalConfig.downsample_factor
HORIZONS_MIN = (10, 20)
FRAME_INTERVAL_MIN = 10          # fullRange cadence
MIN_WET_FRACTION = 0.005         # a case must have >= 0.5 % of the grid wet

_NAME_RE = re.compile(r"dk\.com\.(\d{12})\.500_max\.h5$")


# --------------------------------------------------------------------------
# archive helpers
# --------------------------------------------------------------------------
def frame_path(archive_dir: Path, ts: datetime) -> Path:
    return (
        archive_dir
        / f"{ts.year:04d}"
        / f"{ts.month:02d}"
        / f"dk.com.{ts:%Y%m%d%H%M}.500_max.h5"
    )


def parse_name_ts(path: Path) -> datetime | None:
    m = _NAME_RE.search(path.name)
    if m is None:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)


def full_range_frames(archive_dir: Path, day: datetime) -> list[datetime]:
    """Every fullRange (minute :x0) frame of ``day`` present on disk."""
    out = []
    for i in range(144):
        ts = day + timedelta(minutes=10 * i)
        if frame_path(archive_dir, ts).exists():
            out.append(ts)
    return out


class CompositeCache:
    """Tiny FIFO cache so a day's frames are parsed once, not four times."""

    def __init__(self, archive_dir: Path, maxsize: int = 6) -> None:
        self.archive_dir = archive_dir
        self.maxsize = maxsize
        self._store: dict[datetime, RadarComposite] = {}
        self._order: list[datetime] = []

    def get(self, ts: datetime) -> RadarComposite:
        hit = self._store.get(ts)
        if hit is not None:
            return hit
        comp = parse_composite(frame_path(self.archive_dir, ts))
        self._store[ts] = comp
        self._order.append(ts)
        while len(self._order) > self.maxsize:
            self._store.pop(self._order.pop(0), None)
        return comp


# --------------------------------------------------------------------------
# the production pipeline, once
# --------------------------------------------------------------------------
def rain_of(comp: RadarComposite) -> np.ndarray:
    """mm/h from the composite's own Z-R, with the library's caps."""
    return dbz_to_rain_rate(
        comp.reflectivity_dbz, zr_a=comp.zr_a, zr_b=comp.zr_b,
    )


def production_flow(
    prev: RadarComposite, now: RadarComposite, rain_now: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``(vy, vx)`` in px per inter-frame step -- compute.py lines ~594-638."""
    vy, vx = dense_flow(prev.reflectivity_dbz, now.reflectivity_dbz)
    vy, vx = complete_flow(
        vy, vx, rain_now,
        pixel_km=float(now.xscale_m) / 1000.0,
        support_threshold_mm_h=RAIN_THRESHOLD_MM_H,
    )
    vy = np.nan_to_num(vy, nan=0.0).astype(np.float32)
    vx = np.nan_to_num(vx, nan=0.0).astype(np.float32)
    np.clip(vy, -MAX_PX_PER_FRAME, MAX_PX_PER_FRAME, out=vy)
    np.clip(vx, -MAX_PX_PER_FRAME, MAX_PX_PER_FRAME, out=vx)
    return vy, vx


def bulk_speed_kmh(
    vy: np.ndarray, vx: np.ndarray, rain: np.ndarray,
    *, pixel_m: float, dt_min: float,
) -> float:
    """Rain-weighted mean speed of the completed flow over echo pixels."""
    echo = np.isfinite(rain) & (rain >= RAIN_THRESHOLD_MM_H)
    if not np.any(echo):
        return 0.0
    w = rain[echo].astype(np.float64)
    total = float(w.sum())
    if total <= 0:
        return 0.0
    mvy = float((vy[echo].astype(np.float64) * w).sum() / total)
    mvx = float((vx[echo].astype(np.float64) * w).sum() / total)
    px_per_min = math.hypot(mvy, mvx) / dt_min
    return px_per_min * pixel_m * 60.0 / 1000.0


def reduce_grid(field: np.ndarray) -> np.ndarray:
    return observed_rain_grid(field, downsample_factor=DOWNSAMPLE_FACTOR)


# --------------------------------------------------------------------------
# one case
# --------------------------------------------------------------------------
def run_case(cache: CompositeCache, t: datetime) -> dict[str, Any] | None:
    """Counts for one case, or ``None`` when the case is skipped.

    A case needs frames at t-10, t, t+10 and t+20 with exactly the fullRange
    spacing, and at least ``MIN_WET_FRACTION`` of the product grid wet at t.
    """
    step = timedelta(minutes=FRAME_INTERVAL_MIN)
    prev = cache.get(t - step)
    now = cache.get(t)
    dt_min = (now.timestamp_utc - prev.timestamp_utc).total_seconds() / 60.0
    if not (FRAME_INTERVAL_MIN - 1 <= dt_min <= FRAME_INTERVAL_MIN + 1):
        return None

    rain_now = rain_of(now)
    obs_grid = reduce_grid(rain_now)
    finite_obs = np.isfinite(obs_grid)
    n_finite = int(finite_obs.sum())
    if n_finite == 0:
        return None
    wet_frac = float(np.count_nonzero(obs_grid[finite_obs] >= RAIN_THRESHOLD_MM_H) / n_finite)
    if wet_frac < MIN_WET_FRACTION:
        return None

    truths: dict[int, np.ndarray] = {}
    for h in HORIZONS_MIN:
        tc = cache.get(t + timedelta(minutes=h))
        if abs((tc.timestamp_utc - now.timestamp_utc).total_seconds() / 60.0 - h) > 1.0:
            return None
        truths[h] = reduce_grid(rain_of(tc))

    vy, vx = production_flow(prev, now, rain_now)
    speed = bulk_speed_kmh(
        vy, vx, rain_now, pixel_m=float(now.xscale_m), dt_min=dt_min,
    )
    advected = {
        h: reduce_grid(f)
        for h, f in zip(
            HORIZONS_MIN,
            advect_field_series(
                rain_now, vy, vx,
                horizons_minutes=list(HORIZONS_MIN),
                dt_minutes=dt_min,
            ),
        )
    }

    out: dict[str, Any] = {
        "t": t.strftime("%Y%m%d%H%M"),
        "month": f"{t.year:04d}-{t.month:02d}",
        "wet_fraction": round(wet_frac, 5),
        "bulk_speed_kmh": round(speed, 2),
        "horizons": {},
    }
    for h in HORIZONS_MIN:
        adv, truth = advected[h], truths[h]
        mask = finite_obs & np.isfinite(adv) & np.isfinite(truth)
        o = (obs_grid >= RAIN_THRESHOLD_MM_H) & mask
        a = (adv >= RAIN_THRESHOLD_MM_H) & mask
        y = (truth >= RAIN_THRESHOLD_MM_H) & mask
        cells: dict[str, int] = {}
        for oi in (0, 1):
            om = o if oi else (~o & mask)
            for ai in (0, 1):
                am = a if ai else (~a & mask)
                base = om & am
                cells[f"o{oi}a{ai}t1"] = int(np.count_nonzero(base & y))
                cells[f"o{oi}a{ai}t0"] = int(np.count_nonzero(base & ~y))
        out["horizons"][str(h)] = {"n_pixels": int(mask.sum()), **cells}
    return out


def run_day(
    archive_dir_s: str, day_s: str, stride: int,
) -> tuple[str, list[dict[str, Any]], str | None]:
    """Worker entry point: every case of one day."""
    archive_dir = Path(archive_dir_s)
    day = datetime.strptime(day_s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    try:
        frames = full_range_frames(archive_dir, day)
        cache = CompositeCache(archive_dir)
        cases: list[dict[str, Any]] = []
        for i in range(1, len(frames) - 2, stride):
            t = frames[i]
            need = [frames[i - 1], t, frames[i + 1], frames[i + 2]]
            spacing = [
                (b - a).total_seconds() / 60.0 for a, b in zip(need, need[1:])
            ]
            if any(abs(s - FRAME_INTERVAL_MIN) > 0.5 for s in spacing):
                continue
            try:
                res = run_case(cache, t)
            except Exception as exc:  # one bad frame must not kill the day
                cases.append({"t": t.strftime("%Y%m%d%H%M"), "error": repr(exc)})
                continue
            if res is not None:
                cases.append(res)
        return day_s, cases, None
    except Exception as exc:
        return day_s, [], repr(exc)


# --------------------------------------------------------------------------
# scouting: wet fraction of single frames
# --------------------------------------------------------------------------
def scout_frame(path_s: str) -> tuple[str, float]:
    comp = parse_composite(Path(path_s))
    grid = reduce_grid(rain_of(comp))
    finite = np.isfinite(grid)
    n = int(finite.sum())
    if n == 0:
        return path_s, 0.0
    return path_s, float(np.count_nonzero(grid[finite] >= RAIN_THRESHOLD_MM_H) / n)


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
_CELLS = [f"o{o}a{a}t{y}" for o in (0, 1) for a in (0, 1) for y in (0, 1)]


def empty_counts() -> dict[str, int]:
    return {"n_cases": 0, "n_pixels": 0, **{c: 0 for c in _CELLS}}


def add_case(acc: dict[str, int], hz: dict[str, Any]) -> None:
    acc["n_cases"] += 1
    acc["n_pixels"] += hz["n_pixels"]
    for c in _CELLS:
        acc[c] += hz[c]


def season_of(month: str) -> str:
    m = int(month.split("-")[1])
    if 5 <= m <= 9:
        return "summer (May-Sep)"
    if m in (12, 1, 2, 3):
        return "winter (Dec-Mar)"
    return "shoulder (Apr)"


def scores(acc: dict[str, int]) -> dict[str, Any]:
    """Conditional frequencies plus POD/FAR/CSI for both forecasts."""
    def frac(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    obs_wet = sum(acc[f"o1a{a}t{y}"] for a in (0, 1) for y in (0, 1))
    obs_wet_t = sum(acc[f"o1a{a}t1"] for a in (0, 1))
    adv_wet = sum(acc[f"o{o}a1t{y}"] for o in (0, 1) for y in (0, 1))
    adv_wet_t = sum(acc[f"o{o}a1t1"] for o in (0, 1))
    trail = acc["o1a0t1"] + acc["o1a0t0"]
    lead = acc["o0a1t1"] + acc["o0a1t0"]
    truth_wet = sum(acc[f"o{o}a{a}t1"] for o in (0, 1) for a in (0, 1))

    def skill(hits: int, misses: int, fa: int) -> dict[str, float | None]:
        return {
            "POD": frac(hits, hits + misses),
            "FAR": frac(fa, hits + fa),
            "CSI": frac(hits, hits + misses + fa),
            # forecast wet area / observed wet area; < 1 means the forecast
            # under-produces rain, > 1 that it over-produces it.
            "frequency_bias": frac(hits + fa, hits + misses),
            "hits": hits, "misses": misses, "false_alarms": fa,
        }

    pers = skill(
        hits=obs_wet_t,
        misses=sum(acc[f"o0a{a}t1"] for a in (0, 1)),
        fa=sum(acc[f"o1a{a}t0"] for a in (0, 1)),
    )
    adv = skill(
        hits=adv_wet_t,
        misses=sum(acc[f"o{o}a0t1"] for o in (0, 1)),
        fa=sum(acc[f"o{o}a1t0"] for o in (0, 1)),
    )
    return {
        "n_cases": acc["n_cases"],
        "n_pixels": acc["n_pixels"],
        "base_rate_truth_wet": frac(truth_wet, acc["n_pixels"]),
        "P_truth_wet_given_obs_wet": frac(obs_wet_t, obs_wet),
        "P_truth_wet_given_adv_wet": frac(adv_wet_t, adv_wet),
        "trailing_edge": {
            "n": trail, "P_truth_wet": frac(acc["o1a0t1"], trail),
        },
        "leading_edge": {
            "n": lead, "P_truth_wet": frac(acc["o0a1t1"], lead),
        },
        "both_wet": {
            "n": acc["o1a1t1"] + acc["o1a1t0"],
            "P_truth_wet": frac(acc["o1a1t1"], acc["o1a1t1"] + acc["o1a1t0"]),
        },
        "both_dry": {
            "n": acc["o0a0t1"] + acc["o0a0t0"],
            "P_truth_wet": frac(acc["o0a0t1"], acc["o0a0t1"] + acc["o0a0t0"]),
        },
        "persistence": pers,
        "advection": adv,
        "counts": {c: acc[c] for c in _CELLS},
    }


def aggregate(cases: Sequence[dict[str, Any]], slow_fast_kmh: float = 20.0) -> dict[str, Any]:
    strata: dict[str, dict[str, dict[str, int]]] = {}

    def bucket(name: str, h: str, hz: dict[str, Any]) -> None:
        strata.setdefault(name, {}).setdefault(h, empty_counts())
        add_case(strata[name][h], hz)

    for case in cases:
        if "horizons" not in case:
            continue
        month = case["month"]
        speed = case.get("bulk_speed_kmh", 0.0)
        spd = f"speed < {slow_fast_kmh:g} km/h" if speed < slow_fast_kmh \
            else f"speed >= {slow_fast_kmh:g} km/h"
        for h, hz in case["horizons"].items():
            bucket("pooled", h, hz)
            bucket(f"month {month}", h, hz)
            bucket(season_of(month), h, hz)
            bucket(spd, h, hz)
    return {
        name: {h: scores(acc) for h, acc in per_h.items()}
        for name, per_h in strata.items()
    }


# --------------------------------------------------------------------------
# case studies at a point
# --------------------------------------------------------------------------
def point_pixel(geo: CompositeGeo, lat: float, lon: float) -> tuple[int, int]:
    """Product pixel, exactly as ``national_sample.product_pixel`` does it."""
    idx = geo.lonlat_to_grid(lon, lat)
    f = DOWNSAMPLE_FACTOR
    return int(round(idx.row / f)), int(round(idx.col / f))


def case_study(
    archive_dir: Path, ts: datetime, lat: float, lon: float,
    leads: Sequence[float], window: Sequence[int],
) -> dict[str, Any]:
    """Point read-out around one frame: observations, advection, truth."""
    step = timedelta(minutes=FRAME_INTERVAL_MIN)
    now = parse_composite(frame_path(archive_dir, ts))
    prev = parse_composite(frame_path(archive_dir, ts - step))
    geo = CompositeGeo(now)
    row, col = point_pixel(geo, lat, lon)

    def read(comp: RadarComposite) -> float | None:
        v = float(reduce_grid(rain_of(comp))[row, col])
        return v if math.isfinite(v) else None

    observed: dict[str, float | None] = {}
    for off in window:
        p = frame_path(archive_dir, ts + timedelta(minutes=off))
        observed[f"{off:+d}"] = read(parse_composite(p)) if p.exists() else None

    rain_now = rain_of(now)
    dt_min = (now.timestamp_utc - prev.timestamp_utc).total_seconds() / 60.0
    vy, vx = production_flow(prev, now, rain_now)
    order = sorted(float(x) for x in leads)
    adv = {
        f"+{h:g}": (
            lambda v: v if math.isfinite(v) else None
        )(float(reduce_grid(f)[row, col]))
        for h, f in zip(order, advect_field_series(
            rain_now, vy, vx, horizons_minutes=order, dt_minutes=dt_min,
        ))
    }
    return {
        "frame": ts.strftime("%Y-%m-%dT%H:%MZ"),
        "prev_frame": (ts - step).strftime("%Y-%m-%dT%H:%MZ"),
        "dt_min": dt_min,
        "point": {"lat": lat, "lon": lon},
        "product_pixel": {"row": row, "col": col},
        "bulk_speed_kmh": round(
            bulk_speed_kmh(vy, vx, rain_now,
                           pixel_m=float(now.xscale_m), dt_min=dt_min), 2,
        ),
        "observed_mm_h": observed,
        "advected_mm_h": adv,
        "threshold_mm_h": RAIN_THRESHOLD_MM_H,
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return f"{v:,}"


def markdown_report(agg: dict[str, Any], meta: dict[str, Any]) -> str:
    lines: list[str] = ["# Persistence vs. advection on the DMI archive", ""]
    lines.append(
        f"Frames wet >= {MIN_WET_FRACTION:.1%} of the product grid; wet := "
        f">= {RAIN_THRESHOLD_MM_H} mm/h on the x{DOWNSAMPLE_FACTOR} p90 grid."
    )
    lines.append("")
    for k, v in meta.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    order = ["pooled"] + sorted(
        k for k in agg if k not in ("pooled",)
    )
    for name in order:
        per_h = agg[name]
        lines += [f"## {name}", ""]
        lines.append(
            "| horizon | cases | pixels | base rate | P(wet\\|obs wet) | "
            "P(wet\\|adv wet) | trailing edge P(wet) (n) | leading edge P(wet) (n) | "
            "CSI pers | CSI adv | POD pers | POD adv | FAR pers | FAR adv | "
            "bias pers | bias adv |"
        )
        lines.append("|" + "---|" * 16)
        for h in sorted(per_h, key=int):
            s = per_h[h]
            lines.append(
                f"| +{h} min | {s['n_cases']:,} | {s['n_pixels']:,} | "
                f"{_fmt(s['base_rate_truth_wet'])} | "
                f"{_fmt(s['P_truth_wet_given_obs_wet'])} | "
                f"{_fmt(s['P_truth_wet_given_adv_wet'])} | "
                f"{_fmt(s['trailing_edge']['P_truth_wet'])} "
                f"({s['trailing_edge']['n']:,}) | "
                f"{_fmt(s['leading_edge']['P_truth_wet'])} "
                f"({s['leading_edge']['n']:,}) | "
                f"{_fmt(s['persistence']['CSI'])} | {_fmt(s['advection']['CSI'])} | "
                f"{_fmt(s['persistence']['POD'])} | {_fmt(s['advection']['POD'])} | "
                f"{_fmt(s['persistence']['FAR'])} | {_fmt(s['advection']['FAR'])} | "
                f"{_fmt(s['persistence']['frequency_bias'])} | "
                f"{_fmt(s['advection']['frequency_bias'])} |"
            )
        lines.append("")
        if name == "pooled":
            lines += ["### 2x2x2 table (pooled)", ""]
            lines.append(
                "| horizon | obs | advected | n | truth wet | P(truth wet) |"
            )
            lines.append("|" + "---|" * 6)
            for h in sorted(per_h, key=int):
                c = per_h[h]["counts"]
                for o in (1, 0):
                    for a in (1, 0):
                        n1, n0 = c[f"o{o}a{a}t1"], c[f"o{o}a{a}t0"]
                        n = n1 + n0
                        p = f"{n1 / n:.3f}" if n else "n/a"
                        lines.append(
                            f"| +{h} min | {'wet' if o else 'dry'} | "
                            f"{'wet' if a else 'dry'} | {n:,} | {n1:,} | {p} |"
                        )
            lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _split(arg: str | None) -> list[str]:
    return [x.strip() for x in arg.split(",") if x.strip()] if arg else []


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--archive-dir", required=True, type=Path,
                   help="root holding YYYY/MM/dk.com.*.500_max.h5")
    p.add_argument("--days", help="comma-separated YYYY-MM-DD to run cases on")
    p.add_argument("--days-file", type=Path, help="file with one YYYY-MM-DD per line")
    p.add_argument("--frames", help="comma-separated YYYYMMDDhhmm frames to scout")
    p.add_argument("--scout", action="store_true",
                   help="report the wet fraction of every frame under --archive-dir")
    p.add_argument("--stride", type=int, default=1,
                   help="use every Nth candidate frame t (default 1)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out-json", type=Path)
    p.add_argument("--out-md", type=Path)
    p.add_argument("--point", help="lat,lon for --case")
    p.add_argument("--case", help="comma-separated YYYYMMDDhhmm frames for point studies")
    p.add_argument("--case-leads", default="10,20",
                   help="advection horizons in minutes for --case")
    p.add_argument("--case-window", default="-20,-10,0,10,20,30",
                   help="observed offsets in minutes for --case")
    args = p.parse_args(argv)

    if args.case:
        if not args.point:
            p.error("--case needs --point lat,lon")
        lat, lon = (float(x) for x in args.point.split(","))
        leads = [float(x) for x in _split(args.case_leads)]
        window = [int(x) for x in _split(args.case_window)]
        out = []
        for ts_s in _split(args.case):
            ts = datetime.strptime(ts_s, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
            out.append(case_study(args.archive_dir, ts, lat, lon, leads, window))
        payload: dict[str, Any] = {"cases": out}
        print(json.dumps(payload, indent=2))
        if args.out_json:
            args.out_json.write_text(json.dumps(payload, indent=2))
        return 0

    if args.scout:
        paths = sorted(
            str(q) for q in args.archive_dir.rglob("dk.com.*.500_max.h5")
            if parse_name_ts(q) is not None
        )
        if args.frames:
            keep = set(_split(args.frames))
            paths = [q for q in paths if Path(q).name[7:19] in keep]
        wet: dict[str, float] = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for path_s, frac in pool.map(scout_frame, paths, chunksize=8):
                wet[Path(path_s).name[7:19]] = round(frac, 5)
        payload = {"n_frames": len(wet), "wet_fraction": wet}
        if args.out_json:
            args.out_json.write_text(json.dumps(payload, indent=2))
        else:
            print(json.dumps(payload, indent=2))
        return 0

    days = _split(args.days)
    if args.days_file:
        days += [
            ln.strip() for ln in args.days_file.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    if not days:
        p.error("need --days / --days-file (or --scout / --case)")
    days = sorted(set(days))

    cases: list[dict[str, Any]] = []
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(run_day, str(args.archive_dir), d, args.stride)
            for d in days
        ]
        for i, fut in enumerate(as_completed(futures), 1):
            day_s, day_cases, err = fut.result()
            if err:
                errors.append(f"{day_s}: {err}")
            cases += [c for c in day_cases if "error" not in c]
            errors += [
                f"{day_s} {c['t']}: {c['error']}" for c in day_cases if "error" in c
            ]
            print(f"[{i}/{len(days)}] {day_s}: {len(day_cases)} cases",
                  file=sys.stderr, flush=True)

    agg = aggregate(cases)
    meta = {
        "days": len(days),
        "cases (frames)": len([c for c in cases if "horizons" in c]),
        "stride": args.stride,
        "errors": len(errors),
    }
    payload = {
        "meta": {**meta, "day_list": days, "error_list": errors[:50]},
        "aggregate": agg,
        "cases": cases,
    }
    if args.out_json:
        args.out_json.write_text(json.dumps(payload, indent=2))
    if args.out_md:
        args.out_md.write_text(markdown_report(agg, meta))
    if not args.out_json and not args.out_md:
        print(json.dumps(agg["pooled"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
