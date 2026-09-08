#!/usr/bin/env python3
"""Attach gauge truth to a calibration corpus (Phase F, F2).

The corpus verifies each row against the radar composite. This script adds
the same row's **rain-gauge** verdict, so a reliability curve can be drawn
against an instrument the nowcast never sees — the radar can no longer
grade its own homework.

The verification instant
------------------------
A corpus row's ``outcome`` is verified at ONE instant, the one the served
probability actually describes::

    T + ceil((lead_min + frame_age_min) / timestep_min - 1e-9) * timestep_min

(``build_calibration_corpus.snap_lead_min`` applied to the EFFECTIVE lead;
that function is imported here rather than reimplemented, so the two can
never drift.) The gauge verdict must describe the same instant, so this
script looks up the 10-minute gauge slot **stamped** at it — DMI's
``precip_past10min`` at ``HH:MM`` is the accumulation over the ten minutes
*ending* at ``HH:MM``.

Wet rule
--------
``gauge_outcome = 1`` when ``precip_past10min ≥ --wet-mm`` **or**
``precip_dur_past10min ≥ --wet-dur-min``. The gauge's 0.1 mm / 10 min floor
is 0.6 mm/h, just above the 0.5 mm/h radar threshold, so amount alone would
call genuine light rain dry; the duration channel catches exactly that case.

``gauge_outcome`` is null only when the **amount** slot is missing —
a missing duration alone still leaves a usable dry/wet call from the mm —
or when the station is a **dead gauge**.

Dead gauges
-----------
A station that reported at least ``--min-known-slots`` ten-minute slots
over the corpus window and was never once wet is a bucket stuck at zero,
not a dry corner of Denmark: every wet radar row there becomes a false
alarm nothing could have avoided. The 2026-09-08 product study found
station 06080 doing exactly that — 3 876 known slots on the wettest days
of the year, zero wet — so its rows leave here with a null
``gauge_outcome`` (the raw ``gauge_mm`` / ``gauge_dur_min`` readings are
kept: they are what the station said, and the point is that nobody should
score against them). The rule is
``dmi_nowcast_core.warning_score.dead_gauges``, shared with the threshold
sweep and the nightly quality report so all three exclude the same
stations. ``--min-known-slots 0`` switches it off.

Traces: DMI encodes "traces of precipitation, less than 0.1 kg/m²" as the
value ``-0.1``. It is never an amount, so any negative reading is
normalised to ``0.0`` mm here (in ``gauge_mm`` as well as in the wet test);
the archive keeps the raw value. Such a slot can still be wet through the
duration channel, which is the physically right answer for drizzle.

Usage::

    python scripts/join_gauge_truth.py \
        --corpus reports/station_corpus.parquet \
        --corpus-dir /var/lib/dmi-nowcast-corpus \
        --out reports/station_corpus_gauge.parquet

A corpus built over the UNION of several ``--points`` files (one STEPS
run serving both the radar calibration points and the gauge points)
carries a ``point_set`` column. Only the gauge points can join a gauge
observation — the radar points are grid coordinates, not station ids —
so pass ``--point-set station_points`` and the join runs on, and writes,
exactly the rows a station-only corpus would have held. Without the flag
(``all``, the default, and the only option for a pre-union corpus) every
row is kept and non-station point_ids simply come back with null gauge
columns, as before.

Data licence of the gauge data: CC BY 4.0 (DMI Open Data).
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dmi_nowcast_core.metobs import (  # noqa: E402
    PRECIP_DUR_PAST_10MIN,
    PRECIP_PAST_10MIN,
    normalize_precip_mm,
)
from dmi_nowcast_core.station_store import StationObsStore  # noqa: E402
from dmi_nowcast_core.warning_score import (  # noqa: E402
    DEFAULT_MIN_KNOWN_SLOTS,
    dead_gauge_scan,
    gauge_truth_vectorised,
)

#: The gauge grid the verification instant is snapped onto.
GAUGE_SLOT_MIN = 10

DEFAULT_WET_MM = 0.1
DEFAULT_WET_DUR_MIN = 1.0


def _snap_lead_min_fallback(lead_min: float, timestep_min: float) -> int:
    """Local copy of the builder's rule, used only if the import fails.

    ``tests/test_gauge_join.py`` asserts this agrees with the builder's
    own function across the grid, so a fallback can never silently verify
    against a different instant.
    """
    return int(math.ceil(lead_min / timestep_min - 1e-9) * timestep_min)


try:  # Prefer the builder's own function — one definition, no drift.
    from build_calibration_corpus import snap_lead_min  # noqa: E402
except Exception:  # noqa: BLE001
    snap_lead_min = _snap_lead_min_fallback  # type: ignore[assignment]


def parse_event_time(s: str) -> datetime:
    """The corpus stores ``event_time`` as ISO-8601 UTC."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def verification_instant(
    event_time: datetime, lead_min: float, frame_age_min: float, timestep_min: float
) -> datetime:
    """The instant a corpus row's outcome describes (builder's rule)."""
    return event_time + timedelta(
        minutes=snap_lead_min(float(lead_min) + float(frame_age_min), float(timestep_min))
    )


def gauge_slot(instant: datetime, slot_min: int = GAUGE_SLOT_MIN) -> datetime:
    """Round an instant onto the gauge's 10-minute stamp grid.

    Verification instants are already whole timesteps past an on-grid
    event time, so this is normally the identity; it exists so a corpus
    built on an off-grid cadence still lands on a real gauge stamp
    instead of silently missing every slot.
    """
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    step = slot_min * 60
    secs = (instant - epoch).total_seconds()
    return epoch + timedelta(seconds=round(secs / step) * step)


def wet_outcome(
    mm: float | None, dur_min: float | None, wet_mm: float, wet_dur_min: float
) -> int | None:
    """The gauge verdict for one slot.

    ``None`` when the amount is missing — the row has no gauge truth. A
    present amount with a missing duration is still a verdict: the mm
    channel alone decides it.
    """
    if mm is None:
        return None
    amount = normalize_precip_mm(mm)
    if amount is not None and amount >= wet_mm:
        return 1
    if dur_min is not None and dur_min >= wet_dur_min:
        return 1
    return 0


def load_gauge_index(
    store: StationObsStore,
    start: datetime,
    end: datetime,
    station_ids: list[str] | None,
) -> tuple[dict[tuple[str, datetime], float], dict[tuple[str, datetime], float]]:
    """``({(station, stamp): mm}, {(station, stamp): duration_min})``."""
    parameters = [PRECIP_PAST_10MIN, PRECIP_DUR_PAST_10MIN]
    table = store.read(start, end, parameter_ids=parameters, station_ids=station_ids)
    mm: dict[tuple[str, datetime], float] = {}
    dur: dict[tuple[str, datetime], float] = {}
    if table.num_rows == 0:
        return mm, dur
    stations = table.column("station_id").to_pylist()
    stamps = table.column("observed_utc").to_pylist()
    params = table.column("parameter_id").to_pylist()
    values = table.column("value").to_pylist()
    for station_id, stamp, parameter, value in zip(stations, stamps, params, values):
        if value is None:
            continue
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        key = (station_id, stamp.astimezone(timezone.utc))
        if parameter == PRECIP_PAST_10MIN:
            mm[key] = float(value)
        elif parameter == PRECIP_DUR_PAST_10MIN:
            dur[key] = float(value)
    return mm, dur


def _dead_gauges(
    corpus_dir: Path,
    station_ids: list[str],
    first: datetime,
    last: datetime,
    min_known_slots: int,
) -> set[str]:
    """Stations excluded by the shared dead-gauge rule, named on stdout.

    One extra vectorised pass over the same months the join already reads
    — the grid is a few tens of megabytes and the rule has to see EVERY
    slot, not just the ones a corpus row happens to verify against, or a
    station's verdict would depend on which events were sampled.
    """
    if min_known_slots <= 0 or not station_ids:
        return set()
    truth = gauge_truth_vectorised(
        Path(corpus_dir), first, last, station_ids, log=None,
    )
    rows = dead_gauge_scan(truth, min_known_slots=min_known_slots)
    for row in rows:
        print(
            f"dead gauge {row.station_id}: {row.known_slots} known slot(s), "
            "never wet — gauge_outcome nulled"
        )
    return {row.station_id for row in rows}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", required=True, type=Path, help="input corpus Parquet")
    ap.add_argument("--corpus-dir", required=True, type=Path,
                    help="corpus root holding stations/obs/*")
    ap.add_argument("--out", required=True, type=Path, help="output Parquet")
    ap.add_argument("--point-set", type=str, default="all", metavar="NAME",
                    help="Join (and write) only the rows whose point_set "
                         "column equals NAME — the stem of one of the "
                         "--points files the corpus was built from, e.g. "
                         "station_points. Default 'all' keeps every row; "
                         "required to be 'all' for a corpus with no "
                         "point_set column.")
    ap.add_argument("--min-known-slots", type=int,
                    default=DEFAULT_MIN_KNOWN_SLOTS,
                    help="a station reporting at least this many gauge "
                         "slots over the corpus window and never once wet "
                         "is a dead bucket: its rows get a null "
                         "gauge_outcome and are named on stdout. 0 "
                         f"disables the rule (default {DEFAULT_MIN_KNOWN_SLOTS})")
    ap.add_argument("--wet-mm", type=float, default=DEFAULT_WET_MM,
                    help=f"mm in the 10-min slot that counts as wet (default {DEFAULT_WET_MM})")
    ap.add_argument("--wet-dur-min", type=float, default=DEFAULT_WET_DUR_MIN,
                    help="minutes of precipitation in the slot that count as wet "
                         f"(default {DEFAULT_WET_DUR_MIN})")
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    if not args.corpus.exists():
        print(f"no corpus at {args.corpus}", file=sys.stderr)
        return 2
    table = pq.read_table(args.corpus)
    if args.point_set != "all":
        if "point_set" not in table.schema.names:
            print(
                f"{args.corpus} has no point_set column, so --point-set "
                f"{args.point_set!r} cannot select anything (it predates the "
                "union build) — use --point-set all",
                file=sys.stderr,
            )
            return 2
        import pyarrow.compute as pc

        available = sorted(set(table.column("point_set").to_pylist()))
        table = table.filter(pc.equal(table.column("point_set"), args.point_set))
        if table.num_rows == 0:
            print(
                f"{args.corpus} has no rows with point_set == "
                f"{args.point_set!r} (available: {available})",
                file=sys.stderr,
            )
            return 2
        print(f"point_set={args.point_set}: kept {table.num_rows} rows "
              f"(available sets: {available})")
    n = table.num_rows
    print(f"corpus {args.corpus} rows={n}")

    event_times = table.column("event_time").to_pylist()
    point_ids = table.column("point_id").to_pylist()
    lead_mins = table.column("lead_min").to_pylist()
    frame_ages = table.column("frame_age_min").to_pylist()
    timesteps = table.column("timestep_min").to_pylist()

    # Verification instants first, so the gauge read covers exactly the
    # window the corpus needs — no more, no less.
    slots: list[datetime | None] = []
    for event_time, lead, age, step in zip(event_times, lead_mins, frame_ages, timesteps):
        if event_time is None or lead is None or step in (None, 0):
            slots.append(None)
            continue
        instant = verification_instant(
            parse_event_time(event_time), lead, age if age is not None else 0.0, step,
        )
        slots.append(gauge_slot(instant))

    present = [s for s in slots if s is not None]
    store = StationObsStore(args.corpus_dir)
    dead: set[str] = set()
    if present:
        stations_wanted = sorted({p for p in point_ids if p})
        mm_index, dur_index = load_gauge_index(
            store, min(present), max(present), stations_wanted or None,
        )
        dead = _dead_gauges(
            args.corpus_dir, stations_wanted, min(present), max(present),
            int(args.min_known_slots),
        )
    else:
        mm_index, dur_index = {}, {}
    print(f"gauge slots loaded: mm={len(mm_index)} dur={len(dur_index)} "
          f"from {store.obs_dir}")

    gauge_mm: list[float | None] = []
    gauge_dur: list[float | None] = []
    gauge_outcome: list[int | None] = []
    traces = 0
    per_lead_joined: Counter = Counter()
    per_lead_null: Counter = Counter()
    per_lead_total: Counter = Counter()

    for point_id, lead, slot in zip(point_ids, lead_mins, slots):
        lead_key = int(lead) if lead is not None else -1
        per_lead_total[lead_key] += 1
        if slot is None or not point_id:
            gauge_mm.append(None)
            gauge_dur.append(None)
            gauge_outcome.append(None)
            per_lead_null[lead_key] += 1
            continue
        key = (point_id, slot)
        raw_mm = mm_index.get(key)
        dur = dur_index.get(key)
        if raw_mm is not None and raw_mm < 0.0:
            traces += 1
        mm = normalize_precip_mm(raw_mm)
        # A dead gauge has readings and no truth: keep what it said,
        # refuse to grade anything against it.
        outcome = (
            None if point_id in dead
            else wet_outcome(raw_mm, dur, args.wet_mm, args.wet_dur_min)
        )
        gauge_mm.append(mm)
        gauge_dur.append(dur)
        gauge_outcome.append(outcome)
        if outcome is None:
            per_lead_null[lead_key] += 1
        else:
            per_lead_joined[lead_key] += 1

    out = (
        table
        .append_column("gauge_mm", pa.array(gauge_mm, pa.float32()))
        .append_column("gauge_dur_min", pa.array(gauge_dur, pa.float32()))
        .append_column("gauge_outcome", pa.array(gauge_outcome, pa.int8()))
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    pq.write_table(out, tmp, compression="zstd")
    tmp.replace(args.out)

    joined = sum(per_lead_joined.values())
    print(f"joined={joined} null={n - joined} traces_normalised={traces} "
          f"dead_gauges={len(dead)}")
    print(f"{'lead_min':>9} {'rows':>9} {'joined':>9} {'null':>9}")
    for lead in sorted(per_lead_total):
        print(f"{lead:>9} {per_lead_total[lead]:>9} "
              f"{per_lead_joined[lead]:>9} {per_lead_null[lead]:>9}")
    if joined == 0:
        print("WARNING: no corpus row matched a gauge slot — a corpus built on "
              "the radar calibration points cannot join, because its point_ids "
              "are grid points, not station ids. Rebuild it with "
              "scripts/build_station_points.py output as --points, or — on a "
              "union corpus — select the gauge rows with "
              "--point-set station_points.")
    print(f"wrote {args.out} ({out.num_rows} rows, {out.num_columns} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
