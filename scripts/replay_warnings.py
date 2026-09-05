"""Replay the live push decision rule at every rain gauge, over history.

Phase F ("how good are we?"). The site sends a browser notification when
the calibrated probability of rain at a subscriber's point crosses their
threshold at their lead. Nobody has ever checked those notifications
against a measurement. This script puts a **virtual subscriber at every
DMI rain gauge**, replays the exact production decision rule over the
archived radar frames, and scores every warning it would have sent
against what that gauge actually recorded.

Production parity, step by step
-------------------------------
Each frame runs the live cycle
(``sidecar/dmi_nowcast_sidecar/compute.py::_compute_sync``) minus the
parts that only serve pixels to a browser:

1. frames at ``T-20`` / ``T-10`` / ``T`` off the archive (fullRange, the
   10-min product — the runtime is fullRange-only since Phase B);
2. ``dense_flow`` (Farnebäck) on the last two dBZ frames, then
   ``complete_flow`` at the production support threshold, ``nan_to_num``,
   clip to ±30 px/frame;
3. ``run_ensemble`` — the vendored STEPS, 24 members, 6 cascade levels,
   ×4 downsample, ``ceil(horizon_min / dt)`` timesteps from radar time;
4. ``national_products`` at ``--frame-age-min``, then the same isotonic
   national curves the service serves (``--national-curves``) — the
   decision reads the CALIBRATED probability or it is not the live rule;
5. ``observed_rain_grid`` for the observed arm of "already raining", and
   the lead-0 deterministic field (the newest composite advected to
   ``radar_ts + frame_age``) for ``forecast_now_mm_h``;
6. ``sample_point`` per station — the same sampler ``/forecast`` and the
   push service use, so a replayed warning reads the same pixel a real
   one would have;
7. ``push.engine.evaluate`` per station, carrying ``SubState`` from frame
   to frame.

The simulated frame age is the one number that cannot be recovered from
the archive: live, it is ``now - radar_ts`` at compute time (median 14
min). It moves the products' lead bookkeeping AND the wall clock the
decision runs on, so ``generated_at = radar_ts + frame_age_min``.

Parallelism and the state simplification
----------------------------------------
The per-frame pipeline is state-free; only ``evaluate`` carries state,
and it carries it along one station's sequence of frames. So the work is
parallelised **by day, one process per day**, and — the simplification —
**every day starts armed, with an empty streak**. A subscription
disarmed at 23:50 is armed again at 00:00 the next day, which can only
ever add warnings near a midnight boundary (at most one per station per
day, and only when rain was already firing at midnight). Chaining state
across days would serialise the whole run into one process; at ~22 s per
frame and 144 frames a day that is the difference between hours and
weeks. Frames inside a day are strictly ordered, so the state machine
sees exactly the observation sequence the live service would have seen.

Resume granularity is one day: the progress file records each finished
day (its row count and the per-station end state), and the day's parquet
is written once, atomically, when the day completes. A run interrupted
mid-day redoes that day; a run interrupted between days picks up at the
next one.

Usage
-----
Validate on a couple of hours, one worker, four stations::

    python scripts/replay_warnings.py \\
        --archive-dir /var/lib/dmi-nowcast-corpus/composites \\
        --corpus-dir /var/lib/dmi-nowcast-corpus \\
        --points station_points.json --days 2026-09-05 \\
        --start-utc 06:00 --end-utc 08:00 --workers 1 \\
        --out-dir /tmp/replay

Then the real thing::

    python scripts/replay_warnings.py \\
        --archive-dir /var/lib/dmi-nowcast-corpus/composites \\
        --corpus-dir /var/lib/dmi-nowcast-corpus \\
        --points /var/lib/dmi-nowcast-corpus/stations/station_points.json \\
        --days-file days.txt --workers 6 --frame-age-min 14 \\
        --national-curves /var/lib/dmi-nowcast/national_curves.json \\
        --out-dir /var/lib/dmi-nowcast-corpus/stations/replay \\
        --progress /var/lib/dmi-nowcast-corpus/stations/replay/progress.json

Outputs under ``--out-dir``:

``decisions/YYYY-MM-DD.parquet``
    One row per (frame, station): the sampled forecast and the action the
    engine took. :data:`dmi_nowcast_core.warning_score.DECISION_COLUMNS`.
``events.parquet``
    Every warning the replay sent, with its outcome, matched onset and
    lead error.
``onsets.parquet``
    Every gauge onset, and the warning that claimed it (or "miss").
``summary.json``
    Pooled and per-station scores, plus the run's provenance.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# The repo layout puts the algorithm library under src/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
_SIDECAR = _REPO_ROOT / "sidecar"
if str(_SIDECAR) not in sys.path:
    sys.path.insert(0, str(_SIDECAR))

from dmi_nowcast_core.advect import advect_field_series  # noqa: E402
from dmi_nowcast_core.calibrate import load_calibration_curves  # noqa: E402
from dmi_nowcast_core.geo import CompositeGeo  # noqa: E402
from dmi_nowcast_core.national import (  # noqa: E402
    national_products,
    observed_rain_grid,
)
from dmi_nowcast_core.parse import RadarComposite, parse_composite  # noqa: E402
from dmi_nowcast_core.probabilistic import run_ensemble  # noqa: E402
from dmi_nowcast_core.transform import dbz_to_rain_rate  # noqa: E402
from dmi_nowcast_core.warning_score import (  # noqa: E402
    DEFAULT_DRY_MIN,
    DEFAULT_TOLERANCE_MIN,
    PRECIP_DUR_PARAM,
    PRECIP_PARAM,
    ScoreResult,
    align_decision_table,
    decision_schema,
    decision_table,
    coverage_runs,
    gauge_slots,
    onsets,
    per_lead_columns,
    pooled_summary,
    raining_now_agreement,
    score_warnings,
)

# ---------------------------------------------------------------------------
# Production constants, copied (not imported) so the study does not depend
# on a sidecar Config object. Each cites its source of truth.
# ---------------------------------------------------------------------------
MAX_PX_PER_FRAME = 30.0        # compute.py::_MAX_PX_PER_FRAME
RAIN_THRESHOLD_MM_H = 0.5      # config.py ForecastConfig.rain_threshold_mm_h
DOWNSAMPLE_FACTOR = 4          # config.py StepsConfig.downsample_factor
ENSEMBLE_SIZE = 24             # config.py StepsConfig.ensemble_size
N_CASCADE_LEVELS = 6           # config.py StepsConfig.n_cascade_levels
HORIZON_MIN = 90               # config.py StepsConfig.horizon_min
NATIONAL_LEADS = (10, 20, 30, 45, 60)   # config.py NationalConfig.leads_min
FRAME_INTERVAL_MIN = 10        # fullRange cadence (Phase B addendum)
FRAME_TOLERANCE_S = 60         # a frame is "on the grid" within a minute
#: The live subscriber row this replay reproduces.
DEFAULT_RULES: dict[str, float] = {
    "threshold_pct": 40,
    "lead_min": 30,
    "rearm_after_min": 60,
    "persistence_obs": 1,
    "raining_now_eta_min": 1.5,
    "raining_now_mm_h": RAIN_THRESHOLD_MM_H,
}
#: Live median compute latency — see the module docstring.
DEFAULT_FRAME_AGE_MIN = 14.0
#: Pad each scored day by this much so a 23:5x warning can still find its
#: onset, and so the first slots of a day have their dry evidence.
GAUGE_PAD_MIN = 120


# ---------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StationPoint:
    id: str
    lat: float
    lon: float
    region: str | None = None


def load_points(path: Path) -> tuple[StationPoint, ...]:
    """Read the v2 station points file the Phase F builder writes."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict) or "points" not in raw:
        raise ValueError(f"{path}: expected an object with a 'points' array")
    version = int(raw.get("version", 0))
    if version != 2:
        raise ValueError(f"{path}: unsupported points version {version!r} (want 2)")
    points: list[StationPoint] = []
    seen: set[str] = set()
    for entry in raw["points"]:
        pid = str(entry["id"])
        if pid in seen:
            raise ValueError(f"{path}: duplicate station id {pid!r}")
        seen.add(pid)
        points.append(StationPoint(
            id=pid,
            lat=float(entry["lat"]),
            lon=float(entry["lon"]),
            region=entry.get("region"),
        ))
    if not points:
        raise ValueError(f"{path}: no points")
    return tuple(points)


def parse_rules(spec: str | None) -> dict[str, float]:
    """``k=v,k=v`` over :data:`DEFAULT_RULES`; unknown keys are an error."""
    rules = dict(DEFAULT_RULES)
    if not spec:
        return rules
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, _, value = chunk.partition("=")
        key = key.strip()
        if key not in rules:
            raise ValueError(
                f"unknown rule {key!r}; known: {', '.join(sorted(rules))}"
            )
        rules[key] = float(value)
    if not 0 < rules["threshold_pct"] < 100:
        raise ValueError("threshold_pct must be in (0, 100)")
    if int(rules["lead_min"]) not in NATIONAL_LEADS:
        raise ValueError(
            f"lead_min must be one of the served leads {NATIONAL_LEADS}"
        )
    if rules["persistence_obs"] < 1:
        raise ValueError("persistence_obs must be >= 1")
    return rules


# ---------------------------------------------------------------------------
# Archive access
# ---------------------------------------------------------------------------


def frame_path(archive_dir: Path, ts: datetime) -> Path:
    return (
        Path(archive_dir)
        / f"{ts.year:04d}"
        / f"{ts.month:02d}"
        / f"dk.com.{ts:%Y%m%d%H%M}.500_max.h5"
    )


def full_range_frames(archive_dir: Path, day: date) -> list[datetime]:
    """Every fullRange (minute :x0) frame of ``day`` present on disk."""
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    out: list[datetime] = []
    for i in range(24 * 60 // FRAME_INTERVAL_MIN):
        ts = start + timedelta(minutes=FRAME_INTERVAL_MIN * i)
        if frame_path(archive_dir, ts).exists():
            out.append(ts)
    return out


class CompositeCache:
    """FIFO cache so a day's frames are parsed once, not three times."""

    def __init__(self, archive_dir: Path, maxsize: int = 4) -> None:
        self.archive_dir = Path(archive_dir)
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


# ---------------------------------------------------------------------------
# One frame: the live pipeline, reduced to what a point needs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameSettings:
    """STEPS / product settings for a replay. Defaults mirror production."""

    frame_age_min: float = DEFAULT_FRAME_AGE_MIN
    ensemble_size: int = ENSEMBLE_SIZE
    n_cascade_levels: int = N_CASCADE_LEVELS
    downsample_factor: int = DOWNSAMPLE_FACTOR
    horizon_min: int = HORIZON_MIN
    leads_min: tuple[int, ...] = NATIONAL_LEADS
    threshold_mm_h: float = RAIN_THRESHOLD_MM_H
    national_curves_path: str | None = None


def production_flow(
    prev: RadarComposite, now: RadarComposite, rain_now: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``(vy, vx)`` in px per inter-frame step — compute.py's motion block.

    Falls back to a uniform phase-correlation shift when OpenCV is absent,
    exactly as ``build_calibration_corpus._process_event`` does, so the
    replay still runs (with a cruder motion field) on a box without cv2.
    """
    from dmi_nowcast_core.dense_flow import (
        DenseFlowUnavailable,
        complete_flow,
        dense_flow,
    )

    try:
        vy, vx = dense_flow(prev.reflectivity_dbz, now.reflectivity_dbz)
    except DenseFlowUnavailable:
        from dmi_nowcast_core.motion import phase_correlation_shift

        rain_prev = dbz_to_rain_rate(
            prev.reflectivity_dbz, zr_a=prev.zr_a, zr_b=prev.zr_b,
        )
        dy, dx = phase_correlation_shift(rain_prev, rain_now)
        vy = np.full(rain_now.shape, dy, dtype=np.float32)
        vx = np.full(rain_now.shape, dx, dtype=np.float32)
    vy, vx = complete_flow(
        vy, vx, rain_now,
        pixel_km=float(now.xscale_m) / 1000.0,
        support_threshold_mm_h=RAIN_THRESHOLD_MM_H,
    )
    vy = np.nan_to_num(vy, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    vx = np.nan_to_num(vx, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    np.clip(vy, -MAX_PX_PER_FRAME, MAX_PX_PER_FRAME, out=vy)
    np.clip(vx, -MAX_PX_PER_FRAME, MAX_PX_PER_FRAME, out=vx)
    return vy, vx


#: One isotonic curve set per process, keyed on the file path — the curves
#: are read once per worker rather than once per frame.
_CURVE_CACHE: dict[str, dict[int, Any]] = {}


def _curves(path: str | None) -> dict[int, Any]:
    if not path:
        return {}
    cached = _CURVE_CACHE.get(path)
    if cached is None:
        cached = load_calibration_curves(Path(path))
        _CURVE_CACHE[path] = cached
    return cached


def sample_frame(
    cache: CompositeCache,
    radar_ts: datetime,
    points: Sequence[StationPoint],
    settings: FrameSettings,
) -> list[dict[str, Any]]:
    """Run one frame end-to-end; return one sample dict per station.

    Raises on a missing input frame or a STEPS failure — the day worker
    catches it and records the frame as an error rather than pretending
    the stations were dry.
    """
    from dmi_nowcast_sidecar.national_sample import sample_point

    step = timedelta(minutes=FRAME_INTERVAL_MIN)
    composites = [
        cache.get(radar_ts - 2 * step),
        cache.get(radar_ts - step),
        cache.get(radar_ts),
    ]
    spacing = [
        (b.timestamp_utc - a.timestamp_utc).total_seconds() / 60.0
        for a, b in zip(composites, composites[1:])
    ]
    if any(abs(s - FRAME_INTERVAL_MIN) > 1.0 for s in spacing):
        raise RuntimeError(f"input frames are not on the {FRAME_INTERVAL_MIN}-min grid")
    now = composites[-1]
    dt_min = spacing[-1]
    geo = CompositeGeo(now)
    rain_now = dbz_to_rain_rate(now.reflectivity_dbz, zr_a=now.zr_a, zr_b=now.zr_b)
    vy, vx = production_flow(composites[-2], now, rain_now)

    n_timesteps = max(1, math.ceil(settings.horizon_min / dt_min - 1e-9))
    forecast = run_ensemble(
        [c.reflectivity_dbz for c in composites],
        vy, vx,
        zr_a=now.zr_a, zr_b=now.zr_b,
        n_ens_members=settings.ensemble_size,
        n_timesteps=n_timesteps,
        timestep_min=dt_min,
        n_cascade_levels=settings.n_cascade_levels,
        threshold_mm_h=settings.threshold_mm_h,
        downsample_factor=settings.downsample_factor,
        pixel_scale_m=float(now.xscale_m),
    )
    products = national_products(
        forecast,
        leads_min=settings.leads_min,
        threshold_mm_h=settings.threshold_mm_h,
        timestep_min=dt_min,
        frame_age_min=settings.frame_age_min,
        downsample_factor=settings.downsample_factor,
    )
    del forecast

    # §B4: the calibrated grid REPLACES the raw one, exactly as the cycle
    # does it — a decision taken on a raw ensemble fraction is not the
    # decision the service takes.
    curves = _curves(settings.national_curves_path)
    if curves:
        from dataclasses import replace as _replace

        products = _replace(products, p_rain={
            int(lead): (
                products.p_rain[int(lead)] if curves.get(int(lead)) is None
                else curves[int(lead)].predict(products.p_rain[int(lead)])
            )
            for lead in products.leads_min
        })

    observed_grid = observed_rain_grid(
        rain_now, downsample_factor=settings.downsample_factor,
    )
    # Lead 0 of the deterministic series: the newest composite advected to
    # generated_at. The live cycle computes the whole series for its
    # overlays; a point decision only ever reads lead 0.
    forecast_now_field = next(iter(advect_field_series(
        rain_now, vy, vx,
        horizons_minutes=[settings.frame_age_min],
        dt_minutes=dt_min,
    )))
    forecast_grids = {
        0: observed_rain_grid(
            forecast_now_field, downsample_factor=settings.downsample_factor,
        )
    }

    # The composite's OWN timestamp, not the one in its filename — the
    # live cycle stamps rows with ``composite_now.timestamp_utc`` and the
    # two must agree for a replay row and a live row to be comparable.
    stamped_ts = now.timestamp_utc
    if stamped_ts.tzinfo is None:
        stamped_ts = stamped_ts.replace(tzinfo=timezone.utc)
    generated_at = stamped_ts + timedelta(minutes=settings.frame_age_min)
    out: list[dict[str, Any]] = []
    for point in points:
        sample = sample_point(
            products, geo, point.lat, point.lon,
            observed_mm_h=observed_grid,
            forecast_mm_h=forecast_grids,
        )
        series = sample.forecast_mm_h if sample else None
        out.append({
            "radar_ts": stamped_ts,
            "generated_at": generated_at,
            "station_id": point.id,
            "p_rain": sample.p_rain if sample else {},
            "eta_min": sample.eta_min if sample else None,
            "intensity_mm_h": sample.intensity_mm_h if sample else None,
            "observed_mm_h": sample.observed_mm_h if sample else None,
            "forecast_now_mm_h": series.get(0) if series else None,
        })
    return out


# ---------------------------------------------------------------------------
# One day: frames in order, the state machine carried along
# ---------------------------------------------------------------------------


def _engine():
    from dmi_nowcast_sidecar.push import engine as decision_engine

    return decision_engine


def state_to_json(state: Any) -> dict:
    return {
        "armed": bool(state.armed),
        "streak": int(state.streak),
        "below_since_utc": (
            state.below_since_utc.isoformat() if state.below_since_utc else None
        ),
        "last_eval_radar_ts": (
            state.last_eval_radar_ts.isoformat()
            if state.last_eval_radar_ts else None
        ),
    }


def state_from_json(raw: dict | None):
    eng = _engine()
    if not raw:
        return eng.INITIAL_STATE
    return eng.SubState(
        armed=bool(raw.get("armed", True)),
        streak=int(raw.get("streak", 0)),
        below_since_utc=_parse_iso(raw.get("below_since_utc")),
        last_eval_radar_ts=_parse_iso(raw.get("last_eval_radar_ts")),
    )


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def run_day(args: tuple) -> dict:
    """Worker entry point: one whole day, frames in order. Never raises.

    Every station starts ARMED with an empty streak — the day-parallel
    simplification stated in the module docstring.
    """
    (
        archive_dir_s, day_s, points, settings, rules, out_dir_s,
        start_min, end_min,
    ) = args
    started = time.time()
    day = date.fromisoformat(day_s)
    eng = _engine()
    engine_rules = eng.Rules(
        persistence_obs=int(rules["persistence_obs"]),
        rearm_after_min=int(rules["rearm_after_min"]),
        raining_now_eta_min=float(rules["raining_now_eta_min"]),
        raining_now_mm_h=float(rules["raining_now_mm_h"]),
    )
    threshold_pct = int(rules["threshold_pct"])
    lead = int(rules["lead_min"])

    result: dict[str, Any] = {
        "day": day_s, "rows": 0, "frames": 0, "errors": [],
        "state": {}, "elapsed_s": 0.0, "frame_ms": [], "failed": False,
    }
    try:
        archive_dir = Path(archive_dir_s)
        frames = [
            ts for ts in full_range_frames(archive_dir, day)
            if start_min <= ts.hour * 60 + ts.minute <= end_min
        ]
        cache = CompositeCache(archive_dir)
        states = {p.id: eng.INITIAL_STATE for p in points}
        rows: list[dict[str, Any]] = []
        for radar_ts in frames:
            t0 = time.time()
            try:
                samples = sample_frame(cache, radar_ts, points, settings)
            except Exception as exc:  # noqa: BLE001 — one frame, not the day
                result["errors"].append(
                    f"{radar_ts:%Y-%m-%dT%H:%MZ}: {type(exc).__name__}: {exc}"
                )
                continue
            for sample in samples:
                station = sample["station_id"]
                obs = eng.Observation(
                    radar_ts_utc=sample["radar_ts"],
                    p_rain=sample["p_rain"].get(lead),
                    eta_min=sample["eta_min"],
                    intensity_mm_h=sample["intensity_mm_h"],
                    observed_mm_h=sample["observed_mm_h"],
                    forecast_now_mm_h=sample["forecast_now_mm_h"],
                )
                decision = eng.evaluate(
                    states[station], obs,
                    threshold_pct=threshold_pct,
                    quiet=None,
                    tz="UTC",
                    now_utc=sample["generated_at"],
                    rules=engine_rules,
                )
                states[station] = decision.state
                rows.append({
                    "radar_ts": sample["radar_ts"],
                    "generated_at": sample["generated_at"],
                    "station_id": station,
                    # The rule's lead — the number the decision was taken
                    # on — plus every served lead beside it, so a
                    # threshold/horizon sweep needs no second STEPS run.
                    "p_rain": obs.p_rain,
                    **per_lead_columns(sample["p_rain"]),
                    "eta_min": obs.eta_min,
                    "intensity_mm_h": obs.intensity_mm_h,
                    "observed_mm_h": obs.observed_mm_h,
                    "forecast_now_mm_h": obs.forecast_now_mm_h,
                    "action": decision.action,
                    "armed_after": decision.state.armed,
                    "streak_after": decision.state.streak,
                })
            result["frames"] += 1
            result["frame_ms"].append(round((time.time() - t0) * 1000.0, 1))
        out_path = Path(out_dir_s) / "decisions" / f"{day_s}.parquet"
        write_decisions(out_path, rows, settings.leads_min)
        result["rows"] = len(rows)
        result["state"] = {sid: state_to_json(s) for sid, s in states.items()}
    except Exception as exc:  # noqa: BLE001 — a dead day must not kill the run
        result["errors"].append(f"{day_s}: {type(exc).__name__}: {exc}")
        # Only a day-level failure is unresumable; a frame that threw is
        # recorded and the rest of the day still counts as replayed.
        result["failed"] = True
    result["elapsed_s"] = round(time.time() - started, 2)
    return result


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------


def _write_table_atomic(table, path: Path) -> None:
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_decisions(path: Path, rows: Sequence[dict], leads_min=None) -> None:
    """Write one day's decision rows, atomically, in the shared schema."""
    _write_table_atomic(decision_table(rows, leads_min), path)


def read_decisions(path: Path, leads_min=None) -> list[dict]:
    """Read one day's rows, tolerating a file written under other leads.

    Deliberately NOT ``read_table(..., schema=...)``: a parquet written
    before the per-lead columns existed has only the base columns, and
    pinning the schema on read would refuse it. The file is read as it is
    and then aligned to the union schema, so an old day and a new day are
    the same dict shape by the time anything scores them.
    """
    import pyarrow.parquet as pq

    return align_decision_table(
        pq.read_table(path), leads_min,
    ).to_pylist()


def write_events(path: Path, rows: Sequence[dict]) -> None:
    import pyarrow as pa

    schema = pa.schema([
        ("station_id", pa.string()),
        ("sent_utc", pa.timestamp("us", tz="UTC")),
        ("eta_min", pa.float32()),
        ("outcome", pa.string()),
        ("onset_utc", pa.timestamp("us", tz="UTC")),
        ("lead_error_min", pa.float32()),
    ])
    table = pa.table(
        {
            name: pa.array([r.get(name) for r in rows], type=schema.field(name).type)
            for name in schema.names
        },
        schema=schema,
    )
    _write_table_atomic(table, path)


def write_onsets(path: Path, rows: Sequence[dict]) -> None:
    import pyarrow as pa

    schema = pa.schema([
        ("station_id", pa.string()),
        ("onset_utc", pa.timestamp("us", tz="UTC")),
        ("outcome", pa.string()),
        ("sent_utc", pa.timestamp("us", tz="UTC")),
        ("lead_error_min", pa.float32()),
    ])
    table = pa.table(
        {
            name: pa.array([r.get(name) for r in rows], type=schema.field(name).type)
            for name in schema.names
        },
        schema=schema,
    )
    _write_table_atomic(table, path)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False, default=str)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Gauge truth + scoring
# ---------------------------------------------------------------------------


def day_slots(
    store: Any,
    day: date,
    station_ids: Sequence[str],
    *,
    pad_min: int = GAUGE_PAD_MIN,
) -> dict[str, list[tuple[datetime, bool | None]]]:
    """Gauge slots for one day ± ``pad_min``, per station.

    The pad is what lets a 23:5x warning find its onset after midnight,
    and what gives the first slots of the day the three dry slots the
    onset rule needs behind them.
    """
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc) - timedelta(
        minutes=pad_min
    )
    end = start + timedelta(days=1, minutes=2 * pad_min)
    table = store.read(
        start, end, [PRECIP_PARAM, PRECIP_DUR_PARAM], list(station_ids),
    )
    return {
        sid: gauge_slots(table, sid, start_utc=start, end_utc=end)
        for sid in station_ids
    }


def merge_slots(
    into: dict[str, dict[datetime, bool | None]],
    add: dict[str, list[tuple[datetime, bool | None]]],
) -> None:
    """Union day windows, preferring a determined value over ``None``."""
    for sid, slots in add.items():
        target = into.setdefault(sid, {})
        for ts, wet in slots:
            if wet is not None or ts not in target:
                target[ts] = wet


def score(
    decisions: Sequence[dict],
    slots_by_day: Sequence[dict[str, list[tuple[datetime, bool | None]]]],
    points: Sequence[StationPoint],
    *,
    lead_min: int,
    tolerance_min: int,
    dry_min: int,
    threshold_mm_h: float,
) -> tuple[dict[str, ScoreResult], dict[str, list[tuple[datetime, bool | None]]], dict]:
    """Per-station scores, the merged slot grid, and the pooled agreement.

    Onsets are computed per day window and then **deduplicated by instant**
    across windows: an onset near a window edge is detected in whichever
    window holds its dry evidence, and counted once.
    """
    onset_by_station: dict[str, set[datetime]] = {p.id: set() for p in points}
    for window in slots_by_day:
        for sid, slots in window.items():
            onset_by_station.setdefault(sid, set()).update(onsets(slots, dry_min))

    merged: dict[str, dict[datetime, bool | None]] = {}
    for window in slots_by_day:
        merge_slots(merged, window)
    slot_lists = {
        sid: sorted(grid.items()) for sid, grid in merged.items()
    }

    warnings_by_station: dict[str, list[tuple[datetime, float | None]]] = {
        p.id: [] for p in points
    }
    frames_by_station: dict[str, list[datetime]] = {p.id: [] for p in points}
    for row in decisions:
        if row.get("radar_ts") is not None:
            frames_by_station.setdefault(row["station_id"], []).append(
                row["radar_ts"],
            )
        if row.get("action") != "notify":
            continue
        warnings_by_station.setdefault(row["station_id"], []).append(
            (row["generated_at"], row.get("eta_min"))
        )

    # The intervals each station was actually being watched over. A replay
    # runs contiguous days, so this is normally one run per day-block plus
    # the lead window at its end — but a resumed run with a missing day
    # must not count that day's rain as misses, and the gauge archive
    # reaches back years further than any replay does.
    coverage_by_station = {
        sid: coverage_runs(stamps, extend_min=lead_min + tolerance_min)
        for sid, stamps in frames_by_station.items()
    }

    # The last slot each station actually reported. A warning whose window
    # reaches past it has not come due yet, and neither has an onset within
    # tolerance of it — both come back "pending" rather than being graded
    # on evidence that does not exist. Matters on the trailing edge of a
    # replay run as much as it does live.
    known_until = {
        sid: max(
            (ts for ts, wet in slots if wet is not None),
            default=None,
        )
        for sid, slots in slot_lists.items()
    }
    results = {
        sid: score_warnings(
            warnings_by_station.get(sid, []),
            sorted(onset_by_station.get(sid, ())),
            lead_min=lead_min,
            tolerance_min=tolerance_min,
            dry_min=dry_min,
            known_until=known_until.get(sid),
            coverage=coverage_by_station.get(sid),
        )
        for sid in (p.id for p in points)
    }
    agreement = raining_now_agreement(
        decisions, slot_lists, threshold_mm_h=threshold_mm_h,
    )
    return results, slot_lists, agreement


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _split(arg: str | None) -> list[str]:
    return [x.strip() for x in arg.split(",") if x.strip()] if arg else []


def _hhmm_to_min(value: str | None, default: int) -> int:
    if not value:
        return default
    hh, _, mm = value.partition(":")
    return int(hh) * 60 + int(mm or 0)


def _load_progress(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {"version": 1, "days": {}}
    try:
        raw = json.loads(path.read_text())
    except Exception:  # noqa: BLE001 — a corrupt progress file restarts the run
        return {"version": 1, "days": {}}
    if not isinstance(raw, dict) or "days" not in raw:
        return {"version": 1, "days": {}}
    return raw


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--archive-dir", required=True, type=Path,
                   help="root holding YYYY/MM/dk.com.*.500_max.h5")
    p.add_argument("--corpus-dir", required=True, type=Path,
                   help="station observation store root (holds stations/obs/)")
    p.add_argument("--points", required=True, type=Path,
                   help="station points JSON (version 2)")
    p.add_argument("--days", help="comma-separated YYYY-MM-DD")
    p.add_argument("--days-file", type=Path, help="one YYYY-MM-DD per line")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--frame-age-min", type=float, default=DEFAULT_FRAME_AGE_MIN)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--rules", help="k=v,... over the live subscriber row")
    p.add_argument("--progress", type=Path,
                   help="JSON progress file; a finished day is not redone")
    p.add_argument("--national-curves", type=Path,
                   help="isotonic curve file the service serves (§B4)")
    p.add_argument("--start-utc", help="clip each day at HH:MM (inclusive)")
    p.add_argument("--end-utc", help="clip each day at HH:MM (inclusive)")
    p.add_argument("--tolerance-min", type=int, default=DEFAULT_TOLERANCE_MIN)
    p.add_argument("--dry-min", type=int, default=DEFAULT_DRY_MIN)
    p.add_argument("--ensemble-size", type=int, default=ENSEMBLE_SIZE)
    p.add_argument("--cascade-levels", type=int, default=N_CASCADE_LEVELS)
    p.add_argument("--downsample-factor", type=int, default=DOWNSAMPLE_FACTOR)
    p.add_argument("--horizon-min", type=int, default=HORIZON_MIN)
    p.add_argument("--no-score", action="store_true",
                   help="replay only; skip the gauge scoring pass")
    args = p.parse_args(argv)

    rules = parse_rules(args.rules)
    points = load_points(args.points)
    days = _split(args.days)
    if args.days_file:
        days += [
            ln.strip() for ln in args.days_file.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    if not days:
        p.error("need --days or --days-file")
    days = sorted(set(days))
    start_min = _hhmm_to_min(args.start_utc, 0)
    end_min = _hhmm_to_min(args.end_utc, 24 * 60)

    settings = FrameSettings(
        frame_age_min=float(args.frame_age_min),
        ensemble_size=int(args.ensemble_size),
        n_cascade_levels=int(args.cascade_levels),
        downsample_factor=int(args.downsample_factor),
        horizon_min=int(args.horizon_min),
        leads_min=NATIONAL_LEADS,
        threshold_mm_h=RAIN_THRESHOLD_MM_H,
        national_curves_path=(
            str(args.national_curves) if args.national_curves else None
        ),
    )
    out_dir = Path(args.out_dir)
    (out_dir / "decisions").mkdir(parents=True, exist_ok=True)

    progress = _load_progress(args.progress)
    done = {
        d for d, entry in progress["days"].items()
        if entry.get("status") == "done"
        and (out_dir / "decisions" / f"{d}.parquet").is_file()
    }
    todo = [d for d in days if d not in done]
    if done:
        print(f"resuming: {len(done)} day(s) already done, {len(todo)} to go",
              file=sys.stderr)

    tasks = [
        (str(args.archive_dir), d, points, settings, rules, str(out_dir),
         start_min, end_min)
        for d in todo
    ]
    errors: list[str] = []
    frame_ms: list[float] = []
    if args.workers <= 1:
        results = (run_day(t) for t in tasks)
        for i, res in enumerate(results, 1):
            _absorb(res, progress, args.progress, errors, frame_ms, i, len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_day, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                _absorb(
                    fut.result(), progress, args.progress, errors, frame_ms,
                    i, len(tasks),
                )

    decisions: list[dict] = []
    n_frames = 0
    for d in days:
        path = out_dir / "decisions" / f"{d}.parquet"
        if not path.is_file():
            continue
        rows = read_decisions(path, settings.leads_min)
        decisions += rows
        n_frames += len({r["radar_ts"] for r in rows})

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": {
            "days": days,
            "n_days": len(days),
            "n_frames": n_frames,
            "n_stations": len(points),
            "n_decision_rows": len(decisions),
            "frame_age_min": settings.frame_age_min,
            "rules": rules,
            "steps": {
                "ensemble_size": settings.ensemble_size,
                "n_cascade_levels": settings.n_cascade_levels,
                "downsample_factor": settings.downsample_factor,
                "horizon_min": settings.horizon_min,
                "leads_min": list(settings.leads_min),
                "threshold_mm_h": settings.threshold_mm_h,
            },
            "national_curves": settings.national_curves_path,
            "archive_dir": str(args.archive_dir),
            "corpus_dir": str(args.corpus_dir),
            "points_file": str(args.points),
            "frame_ms": {
                "n": len(frame_ms),
                "mean": round(sum(frame_ms) / len(frame_ms), 1) if frame_ms else None,
                "min": round(min(frame_ms), 1) if frame_ms else None,
                "max": round(max(frame_ms), 1) if frame_ms else None,
            },
            "errors": errors[:100],
            "n_errors": len(errors),
        },
    }

    if args.no_score:
        summary["gauge"] = {"available": False, "reason": "--no-score"}
    else:
        summary.update(_score_and_write(
            out_dir, decisions, days, points, args, rules,
        ))
    _write_json_atomic(out_dir / "summary.json", summary)
    print(json.dumps(summary.get("pooled", summary["run"]), indent=2, default=str))
    return 0


def _absorb(
    res: dict,
    progress: dict,
    progress_path: Path | None,
    errors: list[str],
    frame_ms: list[float],
    index: int,
    total: int,
) -> None:
    """Fold one finished day into the run state and persist progress."""
    errors.extend(res["errors"])
    frame_ms.extend(res["frame_ms"])
    progress["days"][res["day"]] = {
        "status": "failed" if res.get("failed") else "done",
        "rows": res["rows"],
        "frames": res["frames"],
        "elapsed_s": res["elapsed_s"],
        "state": res["state"],
        "n_errors": len(res["errors"]),
    }
    if progress_path is not None:
        _write_json_atomic(progress_path, progress)
    print(
        f"[{index}/{total}] {res['day']}: {res['frames']} frames, "
        f"{res['rows']} rows, {res['elapsed_s']}s"
        + (f", {len(res['errors'])} error(s)" if res["errors"] else ""),
        file=sys.stderr, flush=True,
    )


def _score_and_write(
    out_dir: Path,
    decisions: list[dict],
    days: Sequence[str],
    points: Sequence[StationPoint],
    args: Any,
    rules: dict,
) -> dict:
    """The gauge pass: onsets, scores, events.parquet, onsets.parquet."""
    try:
        from dmi_nowcast_core.station_store import StationObsStore
    except Exception as exc:  # noqa: BLE001 — the store may not be built yet
        return {"gauge": {
            "available": False,
            "reason": f"station store unavailable: {type(exc).__name__}: {exc}",
        }}
    station_ids = [p.id for p in points]
    store = StationObsStore(args.corpus_dir)
    windows = []
    for d in days:
        try:
            windows.append(day_slots(store, date.fromisoformat(d), station_ids))
        except Exception as exc:  # noqa: BLE001 — one unreadable month
            windows.append({})
            print(f"gauge read failed for {d}: {exc}", file=sys.stderr)
    n_known = sum(
        1 for w in windows for slots in w.values() for _, wet in slots
        if wet is not None
    )
    if n_known == 0:
        return {"gauge": {
            "available": False,
            "reason": "no gauge observations in the store for these days",
        }}

    results, slot_lists, agreement = score(
        decisions, windows, points,
        lead_min=int(rules["lead_min"]),
        tolerance_min=int(args.tolerance_min),
        dry_min=int(args.dry_min),
        threshold_mm_h=float(rules["raining_now_mm_h"]),
    )
    events = [
        {
            "station_id": sid,
            "sent_utc": w.sent_utc,
            "eta_min": w.eta_min,
            "outcome": w.outcome,
            "onset_utc": w.onset_utc,
            "lead_error_min": w.lead_error_min,
        }
        for sid, res in results.items() for w in res.warnings
    ]
    events.sort(key=lambda r: (r["sent_utc"], r["station_id"]))
    write_events(out_dir / "events.parquet", events)
    onset_rows = [
        {
            "station_id": sid,
            "onset_utc": o.onset_utc,
            "outcome": o.outcome,
            "sent_utc": o.sent_utc,
            "lead_error_min": o.lead_error_min,
        }
        for sid, res in results.items() for o in res.onsets
    ]
    onset_rows.sort(key=lambda r: (r["onset_utc"], r["station_id"]))
    write_onsets(out_dir / "onsets.parquet", onset_rows)

    by_station: dict[str, dict] = {}
    for point in points:
        res = results[point.id]
        rows = [r for r in decisions if r["station_id"] == point.id]
        by_station[point.id] = {
            "lat": point.lat,
            "lon": point.lon,
            "region": point.region,
            "n_rows": len(rows),
            "n_slots_known": sum(
                1 for _, wet in slot_lists.get(point.id, ()) if wet is not None
            ),
            "warnings": res.summary,
            "raining_now": raining_now_agreement(
                rows, {point.id: slot_lists.get(point.id, [])},
                threshold_mm_h=float(rules["raining_now_mm_h"]),
            ),
        }
    return {
        "gauge": {
            "available": True,
            "dry_min": int(args.dry_min),
            "tolerance_min": int(args.tolerance_min),
            "n_known_slots": n_known,
            "n_stations_with_obs": sum(
                1 for sid in slot_lists
                if any(w is not None for _, w in slot_lists[sid])
            ),
        },
        "pooled": {
            "warnings": pooled_summary(
                results.values(),
                lead_min=int(rules["lead_min"]),
                tolerance_min=int(args.tolerance_min),
                dry_min=int(args.dry_min),
            ),
            "raining_now": agreement,
        },
        "stations": by_station,
    }


if __name__ == "__main__":
    raise SystemExit(main())
