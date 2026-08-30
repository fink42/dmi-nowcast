"""Phase 2 backtest CLI.

Downloads DMI radar composites for a window, runs persistence and
Lagrangian-mean-motion baselines, and writes Parquet + a console summary.

Example:
    .venv/bin/python scripts/phase2_backtest.py \\
        --lat 55.6726 --lon 12.5645 \\
        --start "2026-05-17T06:00:00Z" --end "2026-05-17T20:00:00Z" \\
        --output reports/backtest_2026-05-17.parquet
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dmi_nowcast_core.backtest import run_backtest, summarize, write_parquet  # noqa: E402
from dmi_nowcast_core.bulk_fetch import bulk_fetch, list_window  # noqa: E402
from dmi_nowcast_core.cache import CacheConfig, DiskCache  # noqa: E402
from dmi_nowcast_core.fetch import AsyncDMIClient  # noqa: E402

ROOT = Path(__file__).parent.parent


def _parse_iso(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


async def amain(args: argparse.Namespace) -> int:
    cache = DiskCache(CacheConfig(root=Path(args.cache_dir), max_bytes=args.cache_bytes))
    start = _parse_iso(args.start)
    end = _parse_iso(args.end)

    async with AsyncDMIClient() as client:
        print(f"Fetching composites {start.isoformat()} .. {end.isoformat()}")
        # bulk_fetch downloads; list_window gives us the metadata for backtest.
        await bulk_fetch(start, end, cache, scan_type=args.scan_type, client=client, concurrency=args.concurrency)
        features = await list_window(client, start, end, scan_type=args.scan_type)
        print(f"  → {len(features)} features in window")

    print("Running backtest…")
    rows = run_backtest(
        features,
        cache,
        lat=args.lat,
        lon=args.lon,
        radius_m=args.radius_km * 1000.0,
        threshold_mm_h=args.threshold,
        scan_type_filter=args.scan_type,
    )
    print(f"  → {len(rows)} prediction rows")

    output = Path(args.output)
    write_parquet(rows, output)
    print(f"Wrote {output.relative_to(ROOT) if output.is_absolute() else output}")

    print()
    print(summarize(rows))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--start", required=True, help="ISO 8601, e.g. 2026-05-17T06:00:00Z")
    p.add_argument("--end", required=True)
    p.add_argument("--radius-km", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.1, help="mm/h")
    p.add_argument("--scan-type", choices=["fullRange", "doppler"], default="fullRange")
    p.add_argument("--cache-dir", default=str(ROOT / ".recon_cache"))
    p.add_argument("--cache-bytes", type=int, default=30 * 1024**3)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--output", required=True, help="Parquet path")
    args = p.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
