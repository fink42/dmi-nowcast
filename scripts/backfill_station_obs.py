#!/usr/bin/env python3
"""Backfill DMI metObs station observations into the corpus (Phase F, F1).

One request per (UTC day × parameter) — a whole day of
``precip_past10min`` is ~15.7k features and comes back in a single
response, so the day is the natural unit of work. Results land in the
corpus's ``stations/obs/YYYY/MM.parquet`` partitions via
:class:`~dmi_nowcast_core.station_store.StationObsStore`, which dedupes
on ``(station_id, observed_utc, parameter_id)``. Re-running is therefore
a no-op for days already stored, and ``--progress`` makes a resumed run
skip the requests too.

Politeness: ``--sleep`` (default 0.5 s) between requests keeps the rate
at DMI's fair-use pace. Keys are not required on ``opendataapi.dmi.dk``
since 2025-12-02; ``--api-key`` exists for gated deployments only.

Usage::

    python scripts/backfill_station_obs.py \
        --corpus-dir /var/lib/dmi-nowcast-corpus \
        --from 2026-06-01 --to 2026-09-05 \
        --progress /var/lib/dmi-nowcast-corpus/stations/backfill_progress.json

Data licence: CC BY 4.0 (DMI Open Data).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dmi_nowcast_core.metobs import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_PARAMETERS,
    MetObsClient,
    is_trace,
)
from dmi_nowcast_core.station_store import StationObsStore  # noqa: E402


def parse_day(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def day_bounds(day: date) -> tuple[datetime, datetime]:
    """``[00:00:00Z, 23:59:59Z]`` — DMI's interval is inclusive at BOTH
    ends, so a naive ``00:00/00:00`` next-day window would double-count
    midnight. 23:59:59 stops cleanly at the 23:50 slot."""
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    return start, start + timedelta(hours=23, minutes=59, seconds=59)


def days_between(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def load_progress(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {"done": [], "days": {}}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"done": [], "days": {}}
    data.setdefault("done", [])
    data.setdefault("days", {})
    return data


def save_progress(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", required=True, type=Path,
                    help="corpus root; observations go under <dir>/stations/")
    ap.add_argument("--from", dest="from_day", required=True, type=parse_day,
                    metavar="YYYY-MM-DD")
    ap.add_argument("--to", dest="to_day", type=parse_day, metavar="YYYY-MM-DD",
                    help="inclusive end day (default: --from, i.e. one day)")
    ap.add_argument("--parameters", default=",".join(DEFAULT_PARAMETERS),
                    help=f"comma-separated parameterIds (default: {','.join(DEFAULT_PARAMETERS)})")
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="seconds between requests (default 0.5 = 2 req/s)")
    ap.add_argument("--progress", type=Path, default=None,
                    help="JSON progress file; makes a resumed run skip finished days")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--api-key", default=None,
                    help="optional X-Gravitee-Api-Key (not required on opendataapi.dmi.dk)")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--skip-catalogue", action="store_true",
                    help="don't refresh <corpus>/stations/catalogue.parquet")
    args = ap.parse_args()

    to_day = args.to_day or args.from_day
    if to_day < args.from_day:
        ap.error("--to must not precede --from")
    parameters = [p.strip() for p in args.parameters.split(",") if p.strip()]
    if not parameters:
        ap.error("--parameters must name at least one parameterId")

    store = StationObsStore(args.corpus_dir)
    client = MetObsClient(base_url=args.base_url, api_key=args.api_key,
                          timeout_s=args.timeout)
    progress = load_progress(args.progress)
    done: set[str] = set(progress["done"])

    if not args.skip_catalogue:
        t0 = time.monotonic()
        stations = client.fetch_stations()
        n = store.write_catalogue(stations)
        print(f"catalogue  stations={n} skipped={client.last_stats.skipped} "
              f"{time.monotonic() - t0:.1f}s", flush=True)
        time.sleep(args.sleep)

    total_new = 0
    for day in days_between(args.from_day, to_day):
        start, end = day_bounds(day)
        for parameter in parameters:
            key = f"{day.isoformat()}|{parameter}"
            if key in done:
                print(f"{day} {parameter:<22} skipped (already in progress file)", flush=True)
                continue
            t0 = time.monotonic()
            try:
                obs = client.fetch_observations(parameter, start, end)
            except Exception as exc:  # noqa: BLE001
                print(f"{day} {parameter:<22} FAILED: {type(exc).__name__}: {exc}",
                      flush=True)
                save_progress(args.progress, {"done": sorted(done), "days": progress["days"]})
                time.sleep(args.sleep)
                continue
            traces = sum(1 for o in obs if is_trace(o.value))
            stations_seen = len({o.station_id for o in obs})
            written = store.append(obs)
            new = written.pop("new", 0)
            total_new += new
            elapsed = time.monotonic() - t0
            print(
                f"{day} {parameter:<22} rows={len(obs):>6} new={new:>6} "
                f"stations={stations_seen:>3} traces={traces:>4} "
                f"skipped={client.last_stats.skipped:>3} {elapsed:5.1f}s",
                flush=True,
            )
            done.add(key)
            progress["days"][key] = {
                "rows": len(obs), "new": new, "stations": stations_seen,
                "traces": traces, "skipped": client.last_stats.skipped,
            }
            save_progress(args.progress, {"done": sorted(done), "days": progress["days"]})
            time.sleep(args.sleep)

    print(f"done: {total_new} new rows into {store.obs_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
