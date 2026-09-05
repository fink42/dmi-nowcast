#!/usr/bin/env python3
"""Build a calibration-points file from DMI gauge stations (Phase F, F2).

The radar corpus is a set of *points*; the gauge benchmark needs those
points to be *stations*, so that ``scripts/join_gauge_truth.py`` can match
a corpus row's ``point_id`` to a ``stationId`` and read the gauge that
actually stood there. This script turns the metObs catalogue plus the
observation archive into a points file the existing
``scripts/build_calibration_corpus.py --points`` accepts unchanged
(schema version 2: ``{"version": 2, "points": [{id, lat, lon, region,
strata?}]}``).

Selection: Danish stations (``country == DNK``) reporting
``precip_past10min`` in at least ``--min-coverage`` of the 10-minute slots
in the window. Coverage, not mere catalogue membership, is the criterion —
a station that lists the parameter but reported it twice in three months is
not truth.

``region`` is the **fine sub-region box** — Sjælland, Nordjylland,
Midtjylland, Sydjylland, Fyn, Sønderjylland, Hovedstaden, Bornholm —
using exactly the boxes ``scripts/build_calibration_points.py``
stratifies the radar point set with. That makes
``sql/reliability_by_region.sql`` and the regional-split criterion in
``scripts/national_calibration_report.py`` work on a station corpus
exactly as they do on the radar-point corpus, region for region.

:mod:`dmi_nowcast_core.regions` stays the project's coarse
classification authority and is preserved as
``strata.country_region``; it is ``"Denmark"`` for every station here,
which is why it cannot be the grouping key. A station outside all eight
boxes falls back to the coarse value rather than carrying an empty
region.

Usage::

    python scripts/build_station_points.py \
        --corpus-dir /var/lib/dmi-nowcast-corpus \
        --from 2026-06-01 --to 2026-09-05 \
        --min-coverage 0.8 \
        --out station_points.json \
        --availability-md reports/station_availability.md

Data licence: CC BY 4.0 (DMI Open Data).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dmi_nowcast_core.metobs import (  # noqa: E402
    PRECIP_DUR_PAST_10MIN,
    PRECIP_PAST_10MIN,
    PRECIP_PAST_1H,
    PRECIP_PAST_1MIN,
    Station,
)
from dmi_nowcast_core.regions import region_of  # noqa: E402
from dmi_nowcast_core.station_store import StationObsStore  # noqa: E402

#: Slot length of each parameter, in seconds — what "one observation" means
#: for that series, and therefore the denominator of its coverage.
SLOT_SECONDS: dict[str, int] = {
    PRECIP_PAST_10MIN: 600,
    PRECIP_DUR_PAST_10MIN: 600,
    PRECIP_PAST_1MIN: 60,
    PRECIP_PAST_1H: 3600,
}
DEFAULT_SLOT_SECONDS = 600

#: Coverage of THIS parameter is what selects a station.
SELECTION_PARAMETER = PRECIP_PAST_10MIN

#: Column order of the availability table.
REPORT_PARAMETERS = (
    PRECIP_PAST_10MIN,
    PRECIP_DUR_PAST_10MIN,
    PRECIP_PAST_1H,
    PRECIP_PAST_1MIN,
)


def parse_day(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def window_bounds(from_day: date, to_day: date) -> tuple[datetime, datetime]:
    """``[from 00:00:00Z, to 23:59:59Z]`` — the same inclusive window the
    backfill writes, so coverage denominators line up with what was asked
    for."""
    start = datetime(from_day.year, from_day.month, from_day.day, tzinfo=timezone.utc)
    end = datetime(to_day.year, to_day.month, to_day.day, tzinfo=timezone.utc) + timedelta(
        hours=23, minutes=59, seconds=59
    )
    return start, end


def expected_slots(parameter_id: str, start: datetime, end: datetime) -> int:
    """How many observations a perfect station would report in the window."""
    step = SLOT_SECONDS.get(parameter_id, DEFAULT_SLOT_SECONDS)
    span = int((end - start).total_seconds())
    return span // step + 1


def _calibration_region(lat: float, lon: float) -> str:
    """The fine sub-region box, or ``""`` when no box contains the point.

    The boxes are priority-ordered and imported from the radar point-set
    generator rather than copied, so the two point sets can never end up
    grouping the same coordinate under different region names.
    """
    try:
        from build_calibration_points import _region_of as _fine_region
    except Exception:  # noqa: BLE001 — the coarse region is a usable fallback
        return ""
    return _fine_region(lat, lon) or ""


def resolve_regions(lat: float, lon: float) -> tuple[str, str]:
    """``(region, country_region)`` for one station.

    ``region`` is the fine box the reliability queries group by; it falls
    back to the coarse value for a coordinate no box contains, because an
    empty region would silently collapse into its own GROUP BY bucket.
    """
    country_region = region_of(lat, lon)
    return _calibration_region(lat, lon) or country_region, country_region


def observation_counts(
    store: StationObsStore, start: datetime, end: datetime, parameters: tuple[str, ...]
) -> dict[str, dict[str, int]]:
    """``{parameter_id: {station_id: n_observations}}`` over the window."""
    table = store.read(start, end, parameter_ids=list(parameters))
    out: dict[str, dict[str, int]] = {p: {} for p in parameters}
    if table.num_rows == 0:
        return out
    grouped = table.group_by(["parameter_id", "station_id"]).aggregate(
        [("value", "count")]
    )
    params = grouped.column("parameter_id").to_pylist()
    stations = grouped.column("station_id").to_pylist()
    counts = grouped.column("value_count").to_pylist()
    for parameter, station_id, n in zip(params, stations, counts):
        out.setdefault(parameter, {})[station_id] = int(n)
    return out


def build_points(
    stations: list[Station],
    counts: dict[str, dict[str, int]],
    start: datetime,
    end: datetime,
    min_coverage: float,
) -> tuple[list[dict], list[dict]]:
    """Return ``(points, availability_rows)``.

    ``availability_rows`` covers every Danish station that either declares
    or actually reports a precipitation parameter, selected or not — the
    rejects are the interesting half of an availability table.
    """
    denom = {p: expected_slots(p, start, end) for p in REPORT_PARAMETERS}
    points: list[dict] = []
    rows: list[dict] = []
    for station in sorted(stations, key=lambda s: s.station_id):
        if station.country != "DNK":
            continue
        coverage = {
            p: (counts.get(p, {}).get(station.station_id, 0) / denom[p] if denom[p] else 0.0)
            for p in REPORT_PARAMETERS
        }
        declares_precip = any(p in station.parameter_ids for p in REPORT_PARAMETERS)
        reports_precip = any(coverage[p] > 0 for p in REPORT_PARAMETERS)
        if not declares_precip and not reports_precip:
            continue
        selected = coverage[SELECTION_PARAMETER] >= min_coverage
        region, country_region = resolve_regions(station.lat, station.lon)
        rows.append({
            "station_id": station.station_id,
            "name": station.name,
            "kind": station.kind,
            "region": region,
            "country_region": country_region,
            "lat": station.lat,
            "lon": station.lon,
            "status": station.status,
            "coverage": coverage,
            "has_1min": PRECIP_PAST_1MIN in station.parameter_ids,
            "selected": selected,
        })
        if not selected:
            continue
        points.append({
            "id": station.station_id,
            "lat": round(float(station.lat), 5),
            "lon": round(float(station.lon), 5),
            "region": region,
            "strata": {
                "station_kind": station.kind,
                "country_region": country_region,
            },
        })
    return points, rows


def availability_markdown(
    rows: list[dict], start: datetime, end: datetime, min_coverage: float
) -> str:
    denom = {p: expected_slots(p, start, end) for p in REPORT_PARAMETERS}
    lines = [
        "# DMI gauge station availability",
        "",
        f"Window: `{start.isoformat()}` – `{end.isoformat()}` (UTC, both ends inclusive).",
        "",
        "Coverage = observations stored / observations a perfectly reporting "
        "station would have produced in the window "
        + ", ".join(f"`{p}`: {denom[p]}" for p in REPORT_PARAMETERS)
        + ".",
        "",
        f"Selection rule: `country == DNK` and `{SELECTION_PARAMETER}` coverage "
        f"≥ {min_coverage:.2f}. "
        f"**{sum(1 for r in rows if r['selected'])} of {len(rows)}** "
        "precipitation-capable Danish stations pass.",
        "",
        "`region` is the fine calibration box (the key "
        "`sql/reliability_by_region.sql` groups by); every station's coarse "
        "`regions.py` region is `Denmark`, preserved in the points file as "
        "`strata.country_region`.",
        "",
        "Source: DMI Open Data metObs, licence CC BY 4.0.",
        "",
        "| station | name | kind | region | "
        + " | ".join(p for p in REPORT_PARAMETERS)
        + " | 1-min | selected |",
        "|---|---|---|---|" + "---|" * len(REPORT_PARAMETERS) + "---|---|",
    ]
    for r in sorted(rows, key=lambda r: (-r["coverage"][SELECTION_PARAMETER], r["station_id"])):
        cov = " | ".join(f"{r['coverage'][p]:.3f}" for p in REPORT_PARAMETERS)
        lines.append(
            f"| `{r['station_id']}` | {r['name']} | {r['kind']} | "
            f"{r['region']} | {cov} | "
            f"{'yes' if r['has_1min'] else 'no'} | "
            f"{'**yes**' if r['selected'] else 'no'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus-dir", required=True, type=Path)
    ap.add_argument("--from", dest="from_day", required=True, type=parse_day,
                    metavar="YYYY-MM-DD")
    ap.add_argument("--to", dest="to_day", required=True, type=parse_day,
                    metavar="YYYY-MM-DD")
    ap.add_argument("--min-coverage", type=float, default=0.8)
    ap.add_argument("--out", required=True, type=Path,
                    help="points JSON, consumable by build_calibration_corpus.py --points")
    ap.add_argument("--availability-md", type=Path, default=None,
                    help="also write the per-station availability table as markdown")
    args = ap.parse_args()

    if args.to_day < args.from_day:
        ap.error("--to must not precede --from")
    if not 0.0 <= args.min_coverage <= 1.0:
        ap.error("--min-coverage must be in [0, 1]")

    start, end = window_bounds(args.from_day, args.to_day)
    store = StationObsStore(args.corpus_dir)
    stations = store.read_catalogue()
    if not stations:
        print(f"no catalogue at {store.catalogue_path} — run backfill_station_obs.py first",
              file=sys.stderr)
        return 2
    counts = observation_counts(store, start, end, REPORT_PARAMETERS)
    points, rows = build_points(stations, counts, start, end, args.min_coverage)

    payload = {
        "version": 2,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "DMI Open Data metObs station observations (CC BY 4.0)",
        "window": {"from": args.from_day.isoformat(), "to": args.to_day.isoformat()},
        "selection": {
            "country": "DNK",
            "parameter": SELECTION_PARAMETER,
            "min_coverage": args.min_coverage,
            "expected_slots": expected_slots(SELECTION_PARAMETER, start, end),
        },
        "region_source": (
            "calibration sub-region boxes from scripts/build_calibration_points.py; "
            "the coarse regions.py value is kept in strata.country_region"
        ),
        "points": points,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(points)} station points -> {args.out} "
          f"(of {len(rows)} precipitation-capable DNK stations)")

    if args.availability_md is not None:
        args.availability_md.parent.mkdir(parents=True, exist_ok=True)
        args.availability_md.write_text(
            availability_markdown(rows, start, end, args.min_coverage)
        )
        print(f"availability table -> {args.availability_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
