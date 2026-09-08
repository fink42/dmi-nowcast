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

The question, per horizon h (10 and 20 min originally; see "Phase H"
below for the longer leads) and per product pixel:

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
* ``variants.production_flow`` -- ``dense_flow`` (OpenCV Farneback,
  production defaults), then the R5 ``complete_flow`` far-field relaxation
  toward bulk storm motion, then ``nan_to_num`` and the +-30 px/frame clip,
  in exactly that order. The registry owns those steps now, so a candidate
  flow can be swapped in with ``--variant`` without this file changing;
* ``advect_field_series(rain_now, vy, vx, horizons_minutes=offsets,
  dt_minutes=dt)`` where ``dt`` is the measured spacing of the two input
  frames (10 min on the fullRange-only feed).

One deliberate difference from production: production advects to
``lead + frame_age_min`` because it projects forward from *now*. Here the
horizons are measured from the radar frame time, because the truth frames
are at radar times. That is the same integrator, only a different target.

Reduction and truth
-------------------
Every field -- rain at t, each advected field, and each truth frame -- is
reduced with ``observed_rain_grid(field, downsample_factor=4)``: block-wise
p90 on the x4 product grid, exactly the grid and statistic the service
samples for a point. Wet := >= 0.5 mm/h (the live
``forecast.rain_threshold_mm_h``). A pixel counts at a horizon only if it is
finite in the observation, that horizon's advected field AND that horizon's
truth, so pixels whose backward trajectory left the composite are neither
"wet" nor "dry" but excluded. The mask is therefore per horizon, which is
why ``n_pixels`` shrinks as the lead grows.

Phase H: the Layer A harness
----------------------------
The plan's Layer A (``forecast_skill_plan.md`` §2) screens candidate
motion fields against the radar's own future. This script is that
harness, so it gained four switches:

* ``--variant NAME`` picks the flow from ``dmi_nowcast_core.variants``.
  Every candidate scores on the *identical* cases, which is the one
  property the protocol insists on; ``production`` is the live pipeline.
* ``--horizons`` extends past +10/+20. The archive is a 10-minute grid,
  so a horizon with no frame (+45) is scored against the nearest one and
  the advection targets that same offset — see :func:`snap_horizon`.
* ``--thresholds`` and ``--fss-scales-km`` add the plan's 0.5 / 1 / 4 mm/h
  categorical scores and the 2-32 km FSS. Both are pooled over cases from
  summed counts, never averaged per case.
* every pair also reports the H-F flow-stall diagnostic on the completed
  flow.

The JSON grows a ``days`` block: per day, per stratum, per horizon and
threshold, the summed contingency table and FSS components. That is what
``scripts/compare_layer_a.py`` resamples for the day-block bootstrap CI
on a candidate-minus-baseline difference.

**Reproducing the archived baseline.** ``--variant production --horizons
10,20`` restores the pre-Phase-H case selection exactly (a case then needs
frames at t-10 … t+20 and no further), and the 0.5 mm/h tables it prints
are byte-for-byte the ones in ``archive/persistence_vs_advection_20260905.md``.
The threshold, FSS and stall sections are appended, not substituted.

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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

# The repo layout puts the algorithm library under src/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from dmi_nowcast_core.advect import advect_field_series  # noqa: E402
from dmi_nowcast_core.benchmark import stall_diagnostic  # noqa: E402
from dmi_nowcast_core.evaluate import (  # noqa: E402
    ContingencyTable, csi, ets, far, frequency_bias, fss_components, pod,
)
from dmi_nowcast_core.geo import CompositeGeo  # noqa: E402
from dmi_nowcast_core.national import observed_rain_grid  # noqa: E402
from dmi_nowcast_core.parse import RadarComposite, parse_composite  # noqa: E402
from dmi_nowcast_core.transform import dbz_to_rain_rate  # noqa: E402
from dmi_nowcast_core.variants import get_variant, list_variants  # noqa: E402

# --- production constants, copied so the study does not import the sidecar ---
RAIN_THRESHOLD_MM_H = 0.5        # config.py ForecastConfig.rain_threshold_mm_h
DOWNSAMPLE_FACTOR = 4            # config.py NationalConfig.downsample_factor
FRAME_INTERVAL_MIN = 10          # fullRange cadence
MIN_WET_FRACTION = 0.005         # a case must have >= 0.5 % of the grid wet
SPEED_SPLIT_KMH = 20.0           # plan §2: the bulk-speed stratum boundary

#: Product-grid spacing in km: the 500 m composite reduced by
#: ``DOWNSAMPLE_FACTOR``. FSS neighbourhoods are quoted in km and converted
#: through this, so "8 km" is 4 product pixels, not 16 native ones.
PRODUCT_PIXEL_KM = 2.0

#: Phase H Layer A defaults (plan §2). The archived 2026-09-05 baseline ran
#: the equivalent of ``--horizons 10,20`` at 0.5 mm/h; passing those
#: reproduces its tables exactly.
DEFAULT_HORIZONS_MIN = (10, 20, 30, 45)
DEFAULT_THRESHOLDS_MM_H = (0.5, 1.0, 4.0)
DEFAULT_FSS_SCALES_KM = (2, 4, 8, 16, 32)

#: The two forecasts every case scores, in report order.
METHODS = ("persistence", "advection")

_NAME_RE = re.compile(r"dk\.com\.(\d{12})\.500_max\.h5$")


# --------------------------------------------------------------------------
# run configuration
# --------------------------------------------------------------------------
def snap_horizon(horizon_min: float, interval_min: int = FRAME_INTERVAL_MIN) -> int:
    """Nearest archive frame offset to ``horizon_min``, ties going LATER.

    The fullRange archive is a 10-minute grid, so a 45-minute horizon has
    no truth frame. Rather than interpolate the truth (which would invent
    rain that no radar saw) the horizon is scored against the nearest real
    frame and the advection targets *that* offset, so forecast and truth
    are always the same lead apart. The mismatch is reported next to every
    such row as ``truth_offset_min``.

    A tie — 45 min is 5 from both 40 and 50 — resolves to the later frame
    because that is the conservative direction: a longer lead is a harder
    forecast, so the reported skill understates rather than overstates.
    """
    steps = math.floor(horizon_min / interval_min + 0.5)
    return max(interval_min, int(steps) * interval_min)


@dataclass(frozen=True)
class CaseSpec:
    """Everything a worker needs to score one case, pickled to the pool.

    Frozen and made of plain tuples so it survives ``ProcessPoolExecutor``
    unchanged, and so a run's configuration can be written verbatim into
    the report's meta block — a Layer A number is meaningless without the
    variant and threshold set that produced it.
    """

    variant: str = "production"
    horizons_min: tuple[int, ...] = DEFAULT_HORIZONS_MIN
    thresholds_mm_h: tuple[float, ...] = DEFAULT_THRESHOLDS_MM_H
    fss_scales_km: tuple[int, ...] = DEFAULT_FSS_SCALES_KM
    #: Filled in by __post_init__ from the horizons; passing it directly
    #: is for tests that want an offset the snapper would not choose.
    offsets_min: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.horizons_min:
            raise ValueError("need at least one horizon")
        if any(h <= 0 for h in self.horizons_min):
            raise ValueError("horizons must be positive")
        if list(self.horizons_min) != sorted(self.horizons_min):
            raise ValueError("horizons must be given in increasing order")
        if not self.offsets_min:
            object.__setattr__(
                self, "offsets_min",
                tuple(snap_horizon(h) for h in self.horizons_min),
            )

    @property
    def frames_ahead(self) -> int:
        """Frames after ``t`` a case needs, from the furthest truth offset."""
        return max(self.offsets_min) // FRAME_INTERVAL_MIN

    @property
    def fss_scales_px(self) -> tuple[int, ...]:
        """Neighbourhood side in PRODUCT pixels for each km scale."""
        return tuple(
            max(1, int(round(km / PRODUCT_PIXEL_KM))) for km in self.fss_scales_km
        )

    def horizon_keys(self) -> tuple[str, ...]:
        return tuple(str(h) for h in self.horizons_min)


def threshold_key(value: float) -> str:
    """JSON/table key for a threshold: ``0.5``, ``1``, ``4``."""
    return f"{value:g}"


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


def variant_flow(
    prev: RadarComposite, now: RadarComposite, rain_now: np.ndarray,
    variant: str = "production",
) -> tuple[np.ndarray, np.ndarray]:
    """``(vy, vx)`` in px per inter-frame step, from the variant registry.

    The registry (``dmi_nowcast_core.variants``) owns the flow so a
    candidate can be scored on identical cases without touching this
    script; ``production`` there is ``compute.py::_compute_sync`` step for
    step, which is what this function used to inline.
    """
    return get_variant(variant)(
        prev.reflectivity_dbz, now.reflectivity_dbz, rain_now,
        pixel_km=float(now.xscale_m) / 1000.0,
    )


def production_flow(
    prev: RadarComposite, now: RadarComposite, rain_now: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The live flow. Kept as a name because case studies read better with it."""
    return variant_flow(prev, now, rain_now, "production")


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
def horizon_metrics(
    obs_grid: np.ndarray,
    adv: np.ndarray,
    truth: np.ndarray,
    spec: CaseSpec,
) -> dict[str, Any]:
    """Every score for one (case, horizon): legacy cells, thresholds, FSS.

    ``obs_grid`` doubles as the persistence forecast — the field at ``t``
    carried forward unchanged — so both forecasts go through the identical
    mask, threshold and neighbourhood arithmetic here.

    The 2x2x2 ``o?a?t?`` cells at 0.5 mm/h are computed exactly as they
    were before Phase H, and independently of ``--thresholds``: they are
    what the archived 2026-09-05 tables are made of, and a run with a
    different threshold list must not move them.
    """
    mask = np.isfinite(obs_grid) & np.isfinite(adv) & np.isfinite(truth)
    n_pixels = int(mask.sum())

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

    thresholds: dict[str, dict[str, list[int]]] = {}
    for thr in spec.thresholds_mm_h:
        truth_wet = (truth >= thr) & mask
        per_method: dict[str, list[int]] = {}
        for name, forecast in (("persistence", obs_grid), ("advection", adv)):
            pred_wet = (forecast >= thr) & mask
            hits = int(np.count_nonzero(pred_wet & truth_wet))
            misses = int(np.count_nonzero(truth_wet & ~pred_wet))
            false_alarms = int(np.count_nonzero(pred_wet & ~truth_wet))
            per_method[name] = [
                hits, misses, false_alarms,
                n_pixels - hits - misses - false_alarms,
            ]
        thresholds[threshold_key(thr)] = per_method

    # FSS wants the case mask expressed as "no rain here", because its
    # fractions are means over the whole grid: NaN reads as below-threshold
    # in ``evaluate.fss_components``. Both forecasts and the truth get the
    # same treatment, so no method gains area the others lack.
    obs_m = np.where(mask, obs_grid, np.nan)
    adv_m = np.where(mask, adv, np.nan)
    truth_m = np.where(mask, truth, np.nan)
    fss_block: dict[str, dict[str, list[float]]] = {}
    for thr in spec.thresholds_mm_h:
        per_scale: dict[str, list[float]] = {}
        for km, px in zip(spec.fss_scales_km, spec.fss_scales_px):
            p_mse, p_ref = fss_components(
                obs_m, truth_m, threshold=thr, neighborhood_px=px,
            )
            a_mse, a_ref = fss_components(
                adv_m, truth_m, threshold=thr, neighborhood_px=px,
            )
            per_scale[f"{km:g}"] = [p_mse, p_ref, a_mse, a_ref]
        fss_block[threshold_key(thr)] = per_scale

    return {
        "n_pixels": n_pixels, **cells,
        "thresholds": thresholds,
        "fss": fss_block,
    }


def run_case(
    cache: CompositeCache, t: datetime, spec: CaseSpec | None = None,
) -> dict[str, Any] | None:
    """Counts for one case, or ``None`` when the case is skipped.

    A case needs a frame at t-10, at t, and at every truth offset in
    ``spec`` with exactly the fullRange spacing, plus at least
    ``MIN_WET_FRACTION`` of the product grid wet at t. With the default
    ``spec`` (horizons 10, 20) that is the pre-Phase-H rule verbatim.
    """
    spec = spec or CaseSpec(horizons_min=(10, 20))
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
    for h, off in zip(spec.horizons_min, spec.offsets_min):
        tc = cache.get(t + timedelta(minutes=off))
        if abs((tc.timestamp_utc - now.timestamp_utc).total_seconds() / 60.0 - off) > 1.0:
            return None
        truths[h] = reduce_grid(rain_of(tc))

    vy, vx = variant_flow(prev, now, rain_now, spec.variant)
    speed = bulk_speed_kmh(
        vy, vx, rain_now, pixel_m=float(now.xscale_m), dt_min=dt_min,
    )
    stall = stall_diagnostic(
        vy, vx, rain_now,
        pixel_km=float(now.xscale_m) / 1000.0, dt_min=dt_min,
        wet_threshold_mm_h=RAIN_THRESHOLD_MM_H,
        bulk_min_kmh=SPEED_SPLIT_KMH,
    )
    advected = {
        h: reduce_grid(f)
        for h, f in zip(
            spec.horizons_min,
            advect_field_series(
                rain_now, vy, vx,
                horizons_minutes=list(spec.offsets_min),
                dt_minutes=dt_min,
            ),
        )
    }

    out: dict[str, Any] = {
        "t": t.strftime("%Y%m%d%H%M"),
        "month": f"{t.year:04d}-{t.month:02d}",
        "wet_fraction": round(wet_frac, 5),
        "bulk_speed_kmh": round(speed, 2),
        "stall": {
            "bulk_kmh": _round_or_none(stall["bulk_kmh"], 2),
            "stalled_share": _round_or_none(stall["stalled_share"], 4),
            "applicable": bool(stall["applicable"]),
            "n_wet": int(stall["n_wet"]),
            "profile_kmh": [_round_or_none(v, 2) for v in stall["profile_kmh"]],
        },
        "horizons": {},
    }
    for h, off in zip(spec.horizons_min, spec.offsets_min):
        block = horizon_metrics(obs_grid, advected[h], truths[h], spec)
        block["truth_offset_min"] = off
        out["horizons"][str(h)] = block
    return out


def run_day(
    archive_dir_s: str, day_s: str, stride: int, spec: CaseSpec | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any], str | None]:
    """Worker entry point: every case of one day, plus its day blocks.

    A candidate ``t`` needs an unbroken 10-minute run of frames from
    ``t-10`` to the furthest truth offset — the pre-Phase-H rule, only
    extended as far as the horizons reach, so ``--horizons 10,20`` selects
    exactly the cases the archived baseline selected.

    Returns ``(day, cases, day_blocks, error)``. ``day_blocks`` is the
    per-stratum sum over the day's cases — the unit the day-block
    bootstrap resamples, and the only place the threshold and FSS detail
    is kept. The per-case dicts are stripped of that detail before they
    travel back: with four horizons, three thresholds and five FSS scales
    it is ~100 numbers per case per horizon, which turns a 3,800-case run
    into tens of megabytes of JSON nobody reads.
    """
    spec = spec or CaseSpec()
    archive_dir = Path(archive_dir_s)
    day = datetime.strptime(day_s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    try:
        frames = full_range_frames(archive_dir, day)
        cache = CompositeCache(archive_dir)
        cases: list[dict[str, Any]] = []
        day_blocks: dict[str, Any] = {}
        ahead = spec.frames_ahead
        for i in range(1, len(frames) - ahead, stride):
            t = frames[i]
            need = frames[i - 1: i + ahead + 1]
            spacing = [
                (b - a).total_seconds() / 60.0 for a, b in zip(need, need[1:])
            ]
            if any(abs(s - FRAME_INTERVAL_MIN) > 0.5 for s in spacing):
                continue
            try:
                res = run_case(cache, t, spec)
            except Exception as exc:  # one bad frame must not kill the day
                cases.append({"t": t.strftime("%Y%m%d%H%M"), "error": repr(exc)})
                continue
            if res is not None:
                accumulate_day_blocks(day_blocks, res)
                cases.append(strip_detail(res))
        return day_s, cases, day_blocks, None
    except Exception as exc:
        return day_s, [], {}, repr(exc)


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


def strata_of(
    case: dict[str, Any], slow_fast_kmh: float = SPEED_SPLIT_KMH,
) -> tuple[str, ...]:
    """The strata a case belongs to, plan §2: never one pooled number.

    One definition shared by the legacy per-case aggregation and the
    per-day blocks, so a stratum cannot mean two different sets of frames
    in the same report.

    Caveat for cross-run comparison: month and season are properties of
    the case, but the speed bucket is read off the VARIANT's own bulk
    motion. Two variants can therefore sort a borderline case into
    different speed strata. Pooled and seasonal comparisons are
    like-for-like; the speed split is diagnostic, and
    ``scripts/compare_layer_a.py`` says so next to its table.
    """
    month = case["month"]
    speed = case.get("bulk_speed_kmh", 0.0)
    spd = f"speed < {slow_fast_kmh:g} km/h" if speed < slow_fast_kmh \
        else f"speed >= {slow_fast_kmh:g} km/h"
    return ("pooled", f"month {month}", season_of(month), spd)


def aggregate(cases: Sequence[dict[str, Any]], slow_fast_kmh: float = SPEED_SPLIT_KMH) -> dict[str, Any]:
    strata: dict[str, dict[str, dict[str, int]]] = {}

    def bucket(name: str, h: str, hz: dict[str, Any]) -> None:
        strata.setdefault(name, {}).setdefault(h, empty_counts())
        add_case(strata[name][h], hz)

    for case in cases:
        if "horizons" not in case:
            continue
        for name in strata_of(case, slow_fast_kmh):
            for h, hz in case["horizons"].items():
                bucket(name, h, hz)
    return {
        name: {h: scores(acc) for h, acc in per_h.items()}
        for name, per_h in strata.items()
    }


# --------------------------------------------------------------------------
# day blocks: the unit the bootstrap resamples
# --------------------------------------------------------------------------
def _round_or_none(value: Any, digits: int) -> float | None:
    """Round for JSON, turning NaN into null — a gap, not a zero."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if math.isfinite(v) else None


def strip_detail(case: dict[str, Any]) -> dict[str, Any]:
    """Drop the per-case threshold / FSS detail once it is in the day block.

    Kept as a separate step (rather than never computing it per case)
    because a day block IS the sum over its cases: the arithmetic has to
    happen per case, only the storage does not.
    """
    for hz in case.get("horizons", {}).values():
        hz.pop("thresholds", None)
        hz.pop("fss", None)
    return case


def empty_block(hz: dict[str, Any]) -> dict[str, Any]:
    """A zeroed accumulator shaped like the horizon block ``hz``."""
    return {
        "truth_offset_min": hz.get("truth_offset_min"),
        "n_cases": 0,
        "n_pixels": 0,
        "thresholds": {
            t: {m: [0, 0, 0, 0] for m in METHODS} for t in hz["thresholds"]
        },
        "fss": {
            t: {s: [0.0, 0.0, 0.0, 0.0] for s in per_scale}
            for t, per_scale in hz["fss"].items()
        },
    }


def add_horizon_block(acc: dict[str, Any], hz: dict[str, Any]) -> None:
    """Sum one case's horizon block into an accumulator of the same shape."""
    acc["n_cases"] += 1
    acc["n_pixels"] += hz["n_pixels"]
    for t, per_method in hz["thresholds"].items():
        for m, counts in per_method.items():
            dst = acc["thresholds"][t][m]
            for i, v in enumerate(counts):
                dst[i] += v
    for t, per_scale in hz["fss"].items():
        for s, comp in per_scale.items():
            dst_f = acc["fss"][t][s]
            for i, v in enumerate(comp):
                dst_f[i] += v


def accumulate_day_blocks(blocks: dict[str, Any], case: dict[str, Any]) -> None:
    """Fold one case into ``blocks[stratum][horizon]``, in place."""
    for name in strata_of(case):
        per_h = blocks.setdefault(name, {})
        for h_key, hz in case["horizons"].items():
            if h_key not in per_h:
                per_h[h_key] = empty_block(hz)
            add_horizon_block(per_h[h_key], hz)


def merge_blocks(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Pool day blocks across days, in place on ``dst``."""
    for name, per_h in src.items():
        dst_h = dst.setdefault(name, {})
        for h_key, acc in per_h.items():
            if h_key not in dst_h:
                dst_h[h_key] = {
                    "truth_offset_min": acc["truth_offset_min"],
                    "n_cases": 0, "n_pixels": 0,
                    "thresholds": {
                        t: {m: [0, 0, 0, 0] for m in per_m}
                        for t, per_m in acc["thresholds"].items()
                    },
                    "fss": {
                        t: {s: [0.0, 0.0, 0.0, 0.0] for s in per_s}
                        for t, per_s in acc["fss"].items()
                    },
                }
            target = dst_h[h_key]
            target["n_cases"] += acc["n_cases"]
            target["n_pixels"] += acc["n_pixels"]
            for t, per_method in acc["thresholds"].items():
                for m, counts in per_method.items():
                    out = target["thresholds"][t][m]
                    for i, v in enumerate(counts):
                        out[i] += v
            for t, per_scale in acc["fss"].items():
                for s, comp in per_scale.items():
                    out_f = target["fss"][t][s]
                    for i, v in enumerate(comp):
                        out_f[i] += v


def table_of(counts: Sequence[int]) -> ContingencyTable:
    """``[hits, misses, false_alarms, correct_negatives]`` → the dataclass."""
    return ContingencyTable(
        hits=int(counts[0]), misses=int(counts[1]),
        false_alarms=int(counts[2]), correct_negatives=int(counts[3]),
    )


def fss_of(components: Sequence[float], method: str) -> float | None:
    """Pooled FSS from summed ``[p_mse, p_ref, a_mse, a_ref]`` components."""
    mse, ref = (components[0], components[1]) if method == "persistence" \
        else (components[2], components[3])
    if ref <= 0:
        return None
    return round(1.0 - mse / ref, 4)


def block_scores(acc: dict[str, Any]) -> dict[str, Any]:
    """Turn one pooled horizon block into reportable scores.

    Ratios are formed from the POOLED counts, never averaged over days or
    cases: a quiet frame and a frontal one contribute their pixels, not
    one vote each.
    """
    def _r(value: float) -> float | None:
        return None if not math.isfinite(value) else round(value, 4)

    thresholds: dict[str, Any] = {}
    for t, per_method in acc["thresholds"].items():
        thresholds[t] = {}
        for m, counts in per_method.items():
            ct = table_of(counts)
            thresholds[t][m] = {
                "CSI": _r(csi(ct)), "POD": _r(pod(ct)), "FAR": _r(far(ct)),
                "frequency_bias": _r(frequency_bias(ct)), "ETS": _r(ets(ct)),
                "hits": ct.hits, "misses": ct.misses,
                "false_alarms": ct.false_alarms,
                "correct_negatives": ct.correct_negatives,
            }
    fss_scores: dict[str, Any] = {
        t: {s: {m: fss_of(comp, m) for m in METHODS} for s, comp in per_scale.items()}
        for t, per_scale in acc["fss"].items()
    }
    return {
        "truth_offset_min": acc["truth_offset_min"],
        "n_cases": acc["n_cases"],
        "n_pixels": acc["n_pixels"],
        "thresholds": thresholds,
        "fss": fss_scores,
    }


def aggregate_blocks(days: dict[str, Any]) -> dict[str, Any]:
    """Pool every day's blocks and score them, per stratum × horizon."""
    pooled: dict[str, Any] = {}
    for block in days.values():
        merge_blocks(pooled, block)
    return {
        name: {h: block_scores(acc) for h, acc in per_h.items()}
        for name, per_h in pooled.items()
    }


def aggregate_stall(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stall diagnostic per stratum (plan §2: every Layer A run reports it).

    Shares and profiles are taken over APPLICABLE pairs only — a bulk of
    at least ``SPEED_SPLIT_KMH``. Below that a slow pixel is not evidence
    of a stall, it is just slow rain, and pooling the two would hide the
    failure inside the winter drizzle that dominates the case count.
    """
    per_stratum: dict[str, dict[str, Any]] = {}
    for case in cases:
        stall = case.get("stall")
        if not stall:
            continue
        for name in strata_of(case):
            acc = per_stratum.setdefault(
                name, {"n_pairs": 0, "n_applicable": 0, "shares": [], "profiles": []},
            )
            acc["n_pairs"] += 1
            if not stall.get("applicable"):
                continue
            acc["n_applicable"] += 1
            share = stall.get("stalled_share")
            if share is not None:
                acc["shares"].append(float(share))
            profile = stall.get("profile_kmh") or []
            if profile and all(v is not None for v in profile):
                acc["profiles"].append([float(v) for v in profile])

    out: dict[str, Any] = {}
    for name, acc in per_stratum.items():
        shares = np.asarray(acc["shares"], dtype=float)
        profiles = np.asarray(acc["profiles"], dtype=float) if acc["profiles"] else None
        out[name] = {
            "n_pairs": acc["n_pairs"],
            "n_applicable": acc["n_applicable"],
            "applicable_share": _round_or_none(
                acc["n_applicable"] / acc["n_pairs"] if acc["n_pairs"] else float("nan"), 4,
            ),
            "stalled_share_median": _round_or_none(
                float(np.median(shares)) if shares.size else float("nan"), 4,
            ),
            "stalled_share_p90": _round_or_none(
                float(np.percentile(shares, 90)) if shares.size else float("nan"), 4,
            ),
            "mean_profile_kmh": (
                [_round_or_none(v, 2) for v in profiles.mean(axis=0)]
                if profiles is not None else []
            ),
        }
    return out


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
    variant: str = "production",
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
    vy, vx = variant_flow(prev, now, rain_now, variant)
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
        "variant": variant,
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


def markdown_report(
    agg: dict[str, Any],
    meta: dict[str, Any],
    blocks: dict[str, Any] | None = None,
    stall: dict[str, Any] | None = None,
) -> str:
    """The legacy tables, unchanged, plus the Phase H Layer A sections.

    Everything the pre-Phase-H script wrote is emitted first and verbatim,
    so the archived 2026-09-05 report stays a like-for-like comparison;
    thresholds, FSS and the stall diagnostic are appended sections.
    """
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
        if blocks and name in blocks:
            lines += _threshold_tables(blocks[name])
    if blocks:
        lines += _fss_tables(blocks)
    if stall:
        lines += _stall_table(stall)
    return "\n".join(lines)


def _horizon_label(h: str, block: dict[str, Any]) -> str:
    """``+45 min (truth +50)`` when the archive has no frame at the horizon."""
    off = block.get("truth_offset_min")
    if off is None or int(off) == int(h):
        return f"+{h} min"
    return f"+{h} min (truth +{int(off)})"


def _threshold_tables(per_h: dict[str, Any]) -> list[str]:
    """CSI / POD / FAR / bias / ETS per horizon × threshold, both forecasts."""
    lines = ["### skill by rain-rate threshold", ""]
    lines.append(
        "| horizon | threshold | method | CSI | POD | FAR | bias | ETS | "
        "hits | misses | false alarms |"
    )
    lines.append("|" + "---|" * 11)
    for h in sorted(per_h, key=int):
        block = per_h[h]
        label = _horizon_label(h, block)
        for t in sorted(block["thresholds"], key=float):
            for m in METHODS:
                s = block["thresholds"][t][m]
                lines.append(
                    f"| {label} | {t} mm/h | {m} | {_fmt(s['CSI'])} | "
                    f"{_fmt(s['POD'])} | {_fmt(s['FAR'])} | "
                    f"{_fmt(s['frequency_bias'])} | {_fmt(s['ETS'])} | "
                    f"{s['hits']:,} | {s['misses']:,} | {s['false_alarms']:,} |"
                )
    lines.append("")
    return lines


def _fss_tables(blocks: dict[str, Any]) -> list[str]:
    """Pooled FSS, one table per stratum. Scales are km on the x4 grid."""
    lines = [
        "## Fractions Skill Score",
        "",
        f"Neighbourhoods in km on the x{DOWNSAMPLE_FACTOR} product grid "
        f"({PRODUCT_PIXEL_KM:g} km per pixel); pooled over cases as "
        "`1 - sum(mse)/sum(mse_ref)`, not averaged per case.",
        "",
    ]
    order = ["pooled"] + sorted(k for k in blocks if k != "pooled")
    for name in order:
        per_h = blocks[name]
        scales = sorted(
            {s for b in per_h.values() for t in b["fss"].values() for s in t},
            key=float,
        )
        if not scales:
            continue
        lines += [f"### {name}", ""]
        lines.append(
            "| horizon | threshold | method | "
            + " | ".join(f"{s} km" for s in scales) + " |"
        )
        lines.append("|" + "---|" * (3 + len(scales)))
        for h in sorted(per_h, key=int):
            block = per_h[h]
            label = _horizon_label(h, block)
            for t in sorted(block["fss"], key=float):
                for m in METHODS:
                    cells = " | ".join(
                        _fmt(block["fss"][t].get(s, {}).get(m)) for s in scales
                    )
                    lines.append(f"| {label} | {t} mm/h | {m} | {cells} |")
        lines.append("")
    return lines


def _stall_table(stall: dict[str, Any]) -> list[str]:
    """The H-F flow-stall diagnostic, per stratum (plan §2)."""
    lines = [
        "## Flow stall diagnostic",
        "",
        f"Share of wet pixels moving slower than 5 km/h while the "
        f"rain-weighted bulk is at least {SPEED_SPLIT_KMH:g} km/h, and the "
        "median along-motion speed from the rear of the echo to its front. "
        "Statistics are over APPLICABLE pairs only — below the bulk "
        "threshold a slow pixel is slow rain, not a stall.",
        "",
    ]
    order = ["pooled"] + sorted(k for k in stall if k != "pooled")
    n_bins = max(
        (len(v.get("mean_profile_kmh") or []) for v in stall.values()), default=0,
    )
    lines.append(
        "| stratum | pairs | applicable | stalled share median | p90 | "
        + " | ".join(f"bin {i + 1}" for i in range(n_bins)) + " |"
    )
    lines.append("|" + "---|" * (5 + n_bins))
    for name in order:
        s = stall[name]
        profile = s.get("mean_profile_kmh") or []
        cells = " | ".join(
            _fmt(profile[i]) if i < len(profile) else "n/a" for i in range(n_bins)
        )
        lines.append(
            f"| {name} | {s['n_pairs']:,} | {_fmt(s['applicable_share'])} "
            f"({s['n_applicable']:,}) | {_fmt(s['stalled_share_median'])} | "
            f"{_fmt(s['stalled_share_p90'])} | {cells} |"
        )
    lines.append("")
    lines.append("Profile bins run rear (upwind) to front, in km/h.")
    lines.append("")
    return lines


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
    p.add_argument("--variant", default="production",
                   choices=list(list_variants()),
                   help="flow variant from dmi_nowcast_core.variants "
                        "(default production = the live pipeline)")
    p.add_argument("--horizons",
                   default=",".join(str(h) for h in DEFAULT_HORIZONS_MIN),
                   help="comma-separated forecast horizons in minutes "
                        f"(default {','.join(str(h) for h in DEFAULT_HORIZONS_MIN)}); "
                        "each is scored against the nearest archive frame")
    p.add_argument("--thresholds",
                   default=",".join(f"{t:g}" for t in DEFAULT_THRESHOLDS_MM_H),
                   help="comma-separated rain-rate thresholds in mm/h "
                        f"(default {','.join(f'{t:g}' for t in DEFAULT_THRESHOLDS_MM_H)})")
    p.add_argument("--fss-scales-km",
                   default=",".join(str(s) for s in DEFAULT_FSS_SCALES_KM),
                   help="comma-separated FSS neighbourhood sizes in km "
                        f"(default {','.join(str(s) for s in DEFAULT_FSS_SCALES_KM)})")
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
            out.append(case_study(
                args.archive_dir, ts, lat, lon, leads, window, args.variant,
            ))
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

    # Sorted and de-duplicated: horizons must increase for
    # ``advect_field_series``, and a repeated threshold would silently
    # collapse onto one JSON key.
    spec = CaseSpec(
        variant=args.variant,
        horizons_min=tuple(sorted({int(x) for x in _split(args.horizons)})),
        thresholds_mm_h=tuple(sorted({float(x) for x in _split(args.thresholds)})),
        fss_scales_km=tuple(sorted({int(x) for x in _split(args.fss_scales_km)})),
    )

    cases: list[dict[str, Any]] = []
    errors: list[str] = []
    day_blocks: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(run_day, str(args.archive_dir), d, args.stride, spec)
            for d in days
        ]
        for i, fut in enumerate(as_completed(futures), 1):
            day_s, day_cases, blocks, err = fut.result()
            if err:
                errors.append(f"{day_s}: {err}")
            cases += [c for c in day_cases if "error" not in c]
            if blocks:
                day_blocks[day_s] = blocks
            errors += [
                f"{day_s} {c['t']}: {c['error']}" for c in day_cases if "error" in c
            ]
            print(f"[{i}/{len(days)}] {day_s}: {len(day_cases)} cases",
                  file=sys.stderr, flush=True)

    agg = aggregate(cases)
    blocks_agg = aggregate_blocks(day_blocks)
    stall_agg = aggregate_stall(cases)
    meta = {
        "days": len(days),
        "cases (frames)": len([c for c in cases if "horizons" in c]),
        "stride": args.stride,
        "errors": len(errors),
        "variant": spec.variant,
        "horizons (min)": ",".join(str(h) for h in spec.horizons_min),
        "truth offsets (min)": ",".join(str(o) for o in spec.offsets_min),
        "thresholds (mm/h)": ",".join(f"{t:g}" for t in spec.thresholds_mm_h),
        "FSS scales (km)": ",".join(str(s) for s in spec.fss_scales_km),
    }
    payload = {
        "meta": {**meta, "day_list": days, "error_list": errors[:50]},
        "aggregate": agg,
        "blocks": blocks_agg,
        "stall": stall_agg,
        # Per-day, per-stratum contingency and FSS sums: the block the
        # day-block bootstrap in scripts/compare_layer_a.py resamples.
        "days": day_blocks,
        "cases": cases,
    }
    if args.out_json:
        args.out_json.write_text(json.dumps(payload, indent=2))
    if args.out_md:
        args.out_md.write_text(
            markdown_report(agg, meta, blocks_agg, stall_agg)
        )
    if not args.out_json and not args.out_md:
        print(json.dumps(agg["pooled"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
