"""Radar-against-radar backtest harness (plan §10).

For each prediction timestamp ``T`` in a window:
1. Use only frames up to ``T``.
2. Run each method (persistence, lagrangian_mean, …) at horizons 5..60 min.
3. Compare predictions to actual radar at ``T+Δ``.
4. Emit one row per ``(T, method, horizon)``.

Output is Parquet (plan §10.1 column schema, abridged for the Phase 2 baselines).
The harness has no Home Assistant dependencies and runs as a standalone CLI.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from .baselines import lagrangian_dense, lagrangian_mean, persistence
from .cache import DiskCache
from .dense_flow import dense_flow
from .evaluate import contingency, csi, far, hss, pod
from .fetch import RadarFeature
from .geo import CompositeGeo
from .parse import parse_composite
from .sample import sample_disc
from .transform import dbz_to_rain_rate

DEFAULT_HORIZONS_MIN: tuple[int, ...] = (5, 10, 15, 20, 30, 45, 60)
DEFAULT_METHODS: tuple[str, ...] = ("persistence", "lagrangian_mean", "lagrangian_dense")


def _floor_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


def _closest_within(
    timestamps: list[datetime], target: datetime, tolerance: timedelta
) -> datetime | None:
    """Return the timestamp in ``timestamps`` closest to ``target`` within ``tolerance``."""
    if not timestamps:
        return None
    best = min(timestamps, key=lambda t: abs(t - target))
    return best if abs(best - target) <= tolerance else None


def run_backtest(
    features: Iterable[RadarFeature],
    cache: DiskCache,
    *,
    lat: float,
    lon: float,
    radius_m: float = 1000.0,
    horizons_min: tuple[int, ...] = DEFAULT_HORIZONS_MIN,
    threshold_mm_h: float = 0.1,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    scan_type_filter: str | None = "fullRange",
) -> list[dict]:
    """Run the backtest. Returns a list of result rows.

    Frames are loaded into memory up front; for a 12-hour window of fullRange
    frames (72 frames × ~14 MB each) that's ~1 GB. For longer windows, stream
    via a generator-based variant (not yet implemented).
    """
    sorted_features = sorted(features, key=lambda f: f.datetime_utc)
    if scan_type_filter:
        sorted_features = [f for f in sorted_features if f.scan_type == scan_type_filter]
    if not sorted_features:
        return []

    # Load dBZ frames keyed by feature timestamp. We keep dBZ (not rain rate)
    # because dense optical flow needs the log-scaled field; rain rate is
    # derived per-sample on demand and is fast.
    dbz_frames: dict[datetime, np.ndarray] = {}
    zr: tuple[float, float] = (200.0, 1.6)
    geo: CompositeGeo | None = None
    for feature in sorted_features:
        path = cache.path(feature.filename)
        if not path.is_file():
            continue
        composite = parse_composite(path)
        dbz_frames[_floor_minute(feature.datetime_utc)] = composite.reflectivity_dbz
        if geo is None:
            geo = CompositeGeo(composite)
            zr = (composite.zr_a, composite.zr_b)
    if geo is None or len(dbz_frames) < 2:
        return []

    sorted_ts = sorted(dbz_frames.keys())
    cadence_min = (sorted_ts[1] - sorted_ts[0]).total_seconds() / 60.0

    rows: list[dict] = []
    needs_dense = "lagrangian_dense" in methods
    for i in range(1, len(sorted_ts)):
        ts = sorted_ts[i]
        dbz_now = dbz_frames[ts]
        dbz_prev = dbz_frames[sorted_ts[i - 1]]
        rain_now = dbz_to_rain_rate(dbz_now, zr_a=zr[0], zr_b=zr[1])
        rain_prev = dbz_to_rain_rate(dbz_prev, zr_a=zr[0], zr_b=zr[1])
        actual_dt_min = (ts - sorted_ts[i - 1]).total_seconds() / 60.0
        # Compute dense flow once per i; reused for every horizon at this timestamp.
        flow_cache: tuple[np.ndarray, np.ndarray] | None = None
        if needs_dense:
            flow_cache = dense_flow(dbz_prev, dbz_now)
        strictly_future = [t for t in sorted_ts if t > ts]
        for horizon in horizons_min:
            future_ts = _closest_within(
                strictly_future, ts + timedelta(minutes=horizon),
                tolerance=timedelta(minutes=cadence_min / 2),
            )
            if future_ts is None:
                continue
            actual_horizon_min = (future_ts - ts).total_seconds() / 60.0
            rain_future = dbz_to_rain_rate(dbz_frames[future_ts], zr_a=zr[0], zr_b=zr[1])
            actual = sample_disc(rain_future, geo, lon, lat, radius_m=radius_m)
            actual_rain = (
                bool(actual.max_mm_h >= threshold_mm_h)
                if math.isfinite(actual.max_mm_h) else None
            )
            for method in methods:
                if method == "persistence":
                    pred = persistence(rain_now, geo, lon, lat, radius_m=radius_m)
                elif method == "lagrangian_mean":
                    pred = lagrangian_mean(
                        rain_now, rain_prev, geo, lon, lat,
                        horizon_minutes=horizon, dt_minutes=actual_dt_min,
                        radius_m=radius_m,
                    )
                elif method == "lagrangian_dense":
                    assert flow_cache is not None
                    vy, vx = flow_cache
                    pred = lagrangian_dense(
                        rain_now, vy, vx, geo, lon, lat,
                        horizon_minutes=horizon, dt_minutes=actual_dt_min,
                        radius_m=radius_m,
                    )
                else:
                    raise ValueError(f"unknown method: {method}")
                pred_rain = (
                    bool(pred.max_mm_h >= threshold_mm_h)
                    if math.isfinite(pred.max_mm_h) else None
                )
                rows.append({
                    "timestamp_utc": ts.isoformat(),
                    "method": method,
                    "horizon_minutes": int(horizon),
                    "actual_horizon_min": float(actual_horizon_min),
                    "predicted_intensity_mm_h": float(pred.max_mm_h)
                        if math.isfinite(pred.max_mm_h) else None,
                    "predicted_rain": pred_rain,
                    "actual_intensity_mm_h": float(actual.max_mm_h)
                        if math.isfinite(actual.max_mm_h) else None,
                    "actual_rain": actual_rain,
                    "n_valid_pred": pred.n_valid,
                    "n_valid_actual": actual.n_valid,
                    "threshold_mm_h": threshold_mm_h,
                    "radius_m": radius_m,
                })
    return rows


def write_parquet(rows: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # pyarrow won't write a Parquet file with zero rows + no schema, so do it explicitly.
        empty = pa.table({"timestamp_utc": pa.array([], type=pa.string())})
        pq.write_table(empty, path)
        return
    pq.write_table(pa.Table.from_pylist(rows), path)


def summarize(rows: list[dict]) -> str:
    by: dict[tuple[str, int], list[tuple[bool, bool]]] = defaultdict(list)
    for r in rows:
        if r["predicted_rain"] is None or r["actual_rain"] is None:
            continue
        by[(r["method"], r["horizon_minutes"])].append(
            (r["predicted_rain"], r["actual_rain"])
        )
    lines = [
        f"{'method':<18} {'horizon':>7} {'N':>5} {'wet%':>5} "
        f"{'CSI':>6} {'POD':>6} {'FAR':>6} {'HSS':>6}"
    ]
    for key in sorted(by.keys()):
        pairs = by[key]
        method, horizon = key
        pred = np.array([p[0] for p in pairs])
        actl = np.array([p[1] for p in pairs])
        ct = contingency(pred, actl)
        wet_pct = 100.0 * (ct.hits + ct.misses) / max(ct.total, 1)
        lines.append(
            f"{method:<18} {horizon:>7d} {ct.total:>5d} {wet_pct:>4.1f}% "
            f"{csi(ct):>6.3f} {pod(ct):>6.3f} {far(ct):>6.3f} {hss(ct):>6.3f}"
        )
    return "\n".join(lines)
