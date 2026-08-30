"""Phase 4 backtest: pysteps STEPS ensemble → probabilities → calibration.

For each sampled timestamp:
1. Run STEPS on the 3 most recent dBZ frames + Farnebäck flow.
2. Aggregate the ensemble at home → P(rain in next 10/30/60 min) + ETA window.
3. Compare against actual radar within the lead window.
4. Write Parquet rows (raw probabilities + outcome) and fit isotonic calibration.

Pysteps STEPS is ~30-70 s per call on a 1728×1984 grid, so the harness
subsamples timestamps (every Nth fullRange frame, default every 2 h) to keep
the wall time reasonable.

Example:
    .venv/bin/python scripts/phase4_backtest.py \\
        --lat 55.33 --lon 10.32 \\
        --start "2026-05-15T20:00:00Z" --end "2026-05-17T20:00:00Z" \\
        --output reports/phase4_calibration.parquet \\
        --calibration-output reports/phase4_calibrator.json \\
        --stride-min 120
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dmi_nowcast_core.calibrate import brier_score, fit_isotonic, reliability_curve  # noqa: E402
from dmi_nowcast_core.dense_flow import dense_flow  # noqa: E402
from dmi_nowcast_core.geo import CompositeGeo  # noqa: E402
from dmi_nowcast_core.parse import parse_composite  # noqa: E402
from dmi_nowcast_core.probabilistic import aggregate_at_home, run_ensemble  # noqa: E402
from dmi_nowcast_core.sample import sample_disc  # noqa: E402
from dmi_nowcast_core.transform import dbz_to_rain_rate  # noqa: E402

ROOT = Path(__file__).parent.parent
LEAD_MINUTES = (10, 30, 60)
TIMESTEP_MIN = 5.0
N_TIMESTEPS = 12  # 60 min forecast
N_ENS_MEMBERS = 10


def _parse_iso(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _filename_timestamp(path: Path) -> datetime | None:
    m = re.search(r"\.com\.(\d{12})\.", path.name)
    if not m:
        return None
    s = m.group(1)
    return datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]),
                    int(s[8:10]), int(s[10:12]), tzinfo=timezone.utc)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--radius-km", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--stride-min", type=float, default=120.0, help="Sample one prediction per stride")
    p.add_argument("--cache-dir", default=str(ROOT / ".recon_cache"))
    p.add_argument("--output", required=True)
    p.add_argument("--calibration-output", required=True)
    args = p.parse_args()

    cache = Path(args.cache_dir)
    start = _parse_iso(args.start)
    end = _parse_iso(args.end)

    # Index fullRange files by timestamp.
    files: dict[datetime, Path] = {}
    for path in cache.glob("dk.com.*.500_max.h5"):
        ts = _filename_timestamp(path)
        if ts is None:
            continue
        if ts.minute % 10 != 0:  # fullRange only
            continue
        if start <= ts <= end + timedelta(minutes=60):
            files[ts] = path
    if len(files) < 4:
        print(f"Not enough cached fullRange frames in window: {len(files)}", file=sys.stderr)
        return 1

    sorted_ts = sorted(files.keys())

    # Pick prediction timestamps every stride_min apart.
    stride = timedelta(minutes=args.stride_min)
    prediction_ts: list[datetime] = []
    next_ok = sorted_ts[2]  # need 2 prior frames
    for ts in sorted_ts:
        if ts >= next_ok and ts >= start:
            prediction_ts.append(ts)
            next_ok = ts + stride
    print(f"Window has {len(sorted_ts)} fullRange frames; sampling {len(prediction_ts)} predictions every {args.stride_min} min")

    # Load geo from any frame (projection is constant).
    sample_composite = parse_composite(files[sorted_ts[0]])
    geo = CompositeGeo(sample_composite)

    rows: list[dict] = []
    for k, ts in enumerate(prediction_ts):
        idx = sorted_ts.index(ts)
        if idx < 2:
            continue
        # 3 input frames: T-2, T-1, T
        f3 = [parse_composite(files[sorted_ts[idx - 2 + j]]) for j in range(3)]
        dbz = [c.reflectivity_dbz for c in f3]

        t0 = time.perf_counter()
        vy, vx = dense_flow(dbz[1], dbz[2])
        forecast = run_ensemble(
            dbz, vy, vx,
            zr_a=f3[2].zr_a, zr_b=f3[2].zr_b,
            n_ens_members=N_ENS_MEMBERS, n_timesteps=N_TIMESTEPS,
            timestep_min=TIMESTEP_MIN, threshold_mm_h=args.threshold,
        )
        result = aggregate_at_home(
            forecast, geo, args.lon, args.lat,
            radius_m=args.radius_km * 1000.0,
            threshold_mm_h=args.threshold,
            timestep_min=TIMESTEP_MIN,
            leads_min=LEAD_MINUTES,
        )
        elapsed = time.perf_counter() - t0

        # Outcomes at each lead.
        for lead, prob in zip(LEAD_MINUTES, result.probability_by_lead):
            # Was there rain at home within `lead` minutes of T?
            wet_by_lead = False
            valid = True
            for future_min in range(int(TIMESTEP_MIN),
                                    int(lead) + 1, int(TIMESTEP_MIN)):
                future_ts = ts + timedelta(minutes=future_min)
                # Snap to nearest cached fullRange (10-min cadence)
                snapped = future_ts.replace(minute=(future_ts.minute // 10) * 10, second=0, microsecond=0)
                if snapped not in files:
                    continue
                composite = parse_composite(files[snapped])
                rain = dbz_to_rain_rate(composite.reflectivity_dbz, zr_a=composite.zr_a, zr_b=composite.zr_b)
                stats = sample_disc(rain, geo, args.lon, args.lat, radius_m=args.radius_km * 1000.0)
                if math.isfinite(stats.max_mm_h) and stats.max_mm_h >= args.threshold:
                    wet_by_lead = True
                    break
            rows.append({
                "timestamp_utc": ts.isoformat(),
                "lead_minutes": int(lead),
                "raw_probability": float(prob),
                "actual_rain": int(wet_by_lead) if valid else None,
                "eta_p25_min": result.eta_p25_min,
                "eta_p50_min": result.eta_p50_min,
                "eta_p75_min": result.eta_p75_min,
                "n_ens_members": result.n_members,
            })

        print(f"[{k+1}/{len(prediction_ts)}] {ts.isoformat()}: "
              f"P(10/30/60)={result.probability_by_lead}, ETA_P50={result.eta_p50_min:.0f} ({elapsed:.1f}s)")

    # Write Parquet
    import pyarrow as pa
    import pyarrow.parquet as pq
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output)
    print(f"\nWrote {output}")

    # Fit isotonic calibration on the combined data (or per-lead — for MVP, per-lead).
    print("\nReliability per lead:")
    for lead in LEAD_MINUTES:
        per_lead = [r for r in rows if r["lead_minutes"] == lead and r["actual_rain"] is not None]
        if len(per_lead) < 4:
            print(f"  {lead}m: only {len(per_lead)} samples, skipping")
            continue
        raw = np.array([r["raw_probability"] for r in per_lead])
        outcomes = np.array([r["actual_rain"] for r in per_lead], dtype=np.float64)
        bs = brier_score(raw, outcomes)
        centers, freq, counts = reliability_curve(raw, outcomes, n_bins=5)
        print(f"  {lead}m: N={len(per_lead)}, wet%={100*outcomes.mean():.1f}%, Brier={bs:.4f}")
        for c, f, n in zip(centers, freq, counts):
            if n > 0:
                print(f"    raw≈{c:.2f}: observed_freq={f:.2f} (n={n})")

    # Fit a global calibrator from all leads pooled (MVP).
    all_pairs = [(r["raw_probability"], r["actual_rain"]) for r in rows if r["actual_rain"] is not None]
    if len(all_pairs) >= 10:
        raw = np.array([p for p, _ in all_pairs])
        outcomes = np.array([o for _, o in all_pairs], dtype=np.float64)
        cal = fit_isotonic(raw, outcomes)
        cal.save(Path(args.calibration_output))
        print(f"\nSaved calibrator → {args.calibration_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
