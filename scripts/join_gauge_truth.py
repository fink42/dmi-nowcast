#!/usr/bin/env python3
"""Attach gauge truth to a calibration corpus (Phase F, F2).

The corpus verifies each row against the radar composite. This script adds
the same row's **rain-gauge** verdict, so a reliability curve can be drawn
against an instrument the nowcast never sees — the radar can no longer
grade its own homework.

The verification instant
------------------------
A corpus row's ``outcome`` is verified at the instants the served
probability actually describes. The last of them — the one the lead is
*defined* by — is::

    T + ceil((lead_min + frame_age_min) / timestep_min - 1e-9) * timestep_min

(``build_calibration_corpus.snap_lead_min`` applied to the EFFECTIVE lead;
that function is imported here rather than reimplemented, so the two can
never drift.) The gauge verdict must describe the same instants, so this
script looks up the 10-minute gauge slots **stamped** at them — DMI's
``precip_past10min`` at ``HH:MM`` is the accumulation over the ten minutes
*ending* at ``HH:MM``.

Truth is CUMULATIVE — "rain within L", not "rain at T+L"
--------------------------------------------------------
(``--outcome-rule``, default ``within``; the radar half was fixed on
2026-09-09, this half on 2026-09-11.)

The served quantity is ``P(rain at ANY time within L minutes)``, and
``build_calibration_corpus.py`` has scored its radar ``outcome`` that way
since 2026-09-09: rain in ANY frame at ``T + j·step``,
``j = 1 .. snapped/step``. This script went on verifying the gauge at the
single snapped instant, i.e. ``P(rain AT T+L)`` — a strictly rarer event,
and rarer the longer the lead. The quality page's gauge reliability curve
is scored on ``gauge_outcome``, so it sat below the diagonal by
construction: the forecast was being graded against an outcome nobody
serves.

So under ``--outcome-rule within`` (the default) a row's gauge verdict
reads EVERY instant the lead's window covers — the same
``T + j·step`` the radar builder uses, each snapped onto the gauge's
10-minute stamp grid the way :func:`gauge_slot` has always done — and is

- ``1`` if the gauge is wet (rule below) at ANY of those slots,
- ``0`` if it is wet at none,
- ``null`` when the FINAL slot (the snapped instant's) has no amount.

That missing-data policy mirrors the radar builder's frame policy exactly.
A lead is defined by its final instant, so a hole there is no truth at
all; an INTERMEDIATE slot with no amount is skipped and the rest of the
window still scores, because nulling a whole lead over one 10-minute hole
throws away a good final observation. A skip can only ever turn a 1 into a
0 — a shower that passed through the gap goes unseen — never invent a
positive, so the skips are counted and summarised per run rather than
swallowed.

``--outcome-rule instant`` restores the single-slot scoring bit-for-bit.
It exists to reproduce an older curve; the rule is recorded on every row
in ``gauge_outcome_rule`` (and in the output's Parquet metadata) so a
report can always say which event a curve was scored against.

Wet rule
--------
A slot is wet when ``precip_past10min ≥ --wet-mm`` **or**
``precip_dur_past10min ≥ --wet-dur-min`` — the canonical rule of
``dmi_nowcast_core.warning_score`` (``WET_PRECIP_MM`` / ``WET_DUR_MIN``).
The gauge's 0.1 mm / 10 min floor is 0.6 mm/h, just above the 0.5 mm/h
radar threshold, so amount alone would call genuine light rain dry; the
duration channel catches exactly that case.

A slot counts as OBSERVED when its **amount** is present — a missing
duration alone still leaves a usable dry/wet call from the mm, while a
duration with no amount is treated as no reading (it is the amount that
decides whether a slot was measured at all). :func:`wet_outcome` is that
rule for one slot and :func:`slot_verdicts` its vectorised twin; the tests
assert the two agree.

Amounts
-------
``gauge_mm`` / ``gauge_dur_min`` are the readings at the FINAL (snapped)
instant, unchanged — what the gauge said at the instant the lead names.
A cumulative outcome needs its own amount, so ``gauge_mm_window`` carries
the sum over the window's observed slots (under ``instant`` the window is
one slot and the two amounts agree). Like ``gauge_mm`` it is a *reading*,
not a verdict: it is present whenever any slot in the window was observed,
including on a row whose outcome is null because the final slot was
missing or the gauge is dead.

Traces: DMI encodes "traces of precipitation, less than 0.1 kg/m²" as the
value ``-0.1``. It is never an amount, so any negative reading is
normalised to ``0.0`` mm here (in ``gauge_mm`` and ``gauge_mm_window`` as
well as in the wet test); the archive keeps the raw value. Such a slot can
still be wet through the duration channel, which is the physically right
answer for drizzle.

Dead gauges
-----------
A station that reported at least ``--min-known-slots`` ten-minute slots
over the corpus window and was never once wet is a bucket stuck at zero,
not a dry corner of Denmark: every wet radar row there becomes a false
alarm nothing could have avoided. The 2026-09-08 product study found
station 06080 doing exactly that — 3 876 known slots on the wettest days
of the year, zero wet — so its rows leave here with a null
``gauge_outcome`` (the raw ``gauge_mm`` / ``gauge_dur_min`` /
``gauge_mm_window`` readings are kept: they are what the station said, and
the point is that nobody should score against them). The rule is
``dmi_nowcast_core.warning_score.dead_gauges``, shared with the threshold
sweep and the nightly quality report so all three exclude the same
stations. ``--min-known-slots 0`` switches it off.

Shape
-----
Everything after the gauge read is vectorised over rows: a national corpus
is ~500k rows × up to eight verification instants, and a Python loop over
those pairs is minutes of wall time and gigabytes of boxed floats. The
gauge itself is read into a dense ``(station × 10-min slot)`` grid a
record batch at a time (``StationObsStore.stream_month``), so a ten-month
archive costs one batch of resident memory plus ~35 MB of grid rather than
the several GB the old ``to_pylist`` index needed.

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
    WET_DUR_MIN,
    WET_PRECIP_MM,
    dead_gauge_scan,
    gauge_truth_vectorised,
)

#: The gauge grid the verification instants are snapped onto.
GAUGE_SLOT_MIN = 10

#: How a row's gauge truth is scored. Mirrors
#: ``build_calibration_corpus.OUTCOME_RULE_*`` name for name, because the
#: two must describe the same event for the radar and gauge curves to be
#: comparable at all.
OUTCOME_RULE_WITHIN = "within"
OUTCOME_RULE_INSTANT = "instant"
OUTCOME_RULES = (OUTCOME_RULE_WITHIN, OUTCOME_RULE_INSTANT)
DEFAULT_OUTCOME_RULE = OUTCOME_RULE_WITHIN

#: The canonical wet thresholds, from ``warning_score`` so the join, the
#: threshold sweep and the nightly report cannot drift apart.
DEFAULT_WET_MM = WET_PRECIP_MM
DEFAULT_WET_DUR_MIN = WET_DUR_MIN


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
    """The instant a corpus row's outcome is DEFINED by (builder's rule).

    Under ``--outcome-rule within`` the row's window ends here; under
    ``instant`` this is the only instant read.
    """
    return event_time + timedelta(
        minutes=snap_lead_min(float(lead_min) + float(frame_age_min), float(timestep_min))
    )


def window_offsets_min(
    lead_min: float, frame_age_min: float, timestep_min: float,
    outcome_rule: str = DEFAULT_OUTCOME_RULE,
) -> tuple[int, ...]:
    """Minutes after ``T`` this row's gauge truth reads, ascending.

    The scalar reference for the window — the same offsets
    ``build_calibration_corpus._gather_event_frames`` resolves radar
    frames at: ``step, 2·step, …`` up to but not including the snapped
    target, then the target itself. Written that way (rather than as a
    plain ``range``) so the DEFINING instant is in the list even if a
    non-integer timestep ever made the target a non-multiple of the step;
    on the 10-minute grid the two are the same tuple.

    The vectorised join walks ``j = 1 … len(offsets)`` instead, and
    ``tests/test_gauge_join.py`` pins the two against each other.
    """
    target = snap_lead_min(float(lead_min) + float(frame_age_min), float(timestep_min))
    if outcome_rule == OUTCOME_RULE_INSTANT:
        return (int(target),)
    step = int(round(float(timestep_min)))
    if step <= 0:
        return (int(target),)
    return tuple(range(step, int(target), step)) + (int(target),)


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
    """The gauge verdict for ONE slot.

    ``None`` when the amount is missing — that slot has no gauge truth. A
    present amount with a missing duration is still a verdict: the mm
    channel alone decides it.

    The scalar reference for :func:`slot_verdicts`, which is what the
    join actually runs; the tests assert they agree.
    """
    if mm is None:
        return None
    amount = normalize_precip_mm(mm)
    if amount is not None and amount >= wet_mm:
        return 1
    if dur_min is not None and dur_min >= wet_dur_min:
        return 1
    return 0


def slot_verdicts(raw_mm, dur_min, wet_mm: float, wet_dur_min: float):
    """:func:`wet_outcome` over whole arrays: ``(observed, wet, mm)``.

    ``raw_mm`` / ``dur_min`` are float arrays carrying NaN for a missing
    reading. ``observed`` is "the amount is present" (the slot has a
    verdict at all), ``wet`` is that verdict, and ``mm`` is the amount
    with DMI's negative trace sentinel folded to 0.0 — NaN comparisons
    are False throughout, which is exactly the right answer for both
    channels.
    """
    import numpy as np

    observed = ~np.isnan(raw_mm)
    mm = np.where(raw_mm < 0.0, np.float32(0.0), raw_mm)
    wet = observed & ((mm >= wet_mm) | (dur_min >= wet_dur_min))
    return observed, wet, mm


def snap_lead_min_vec(lead_min, timestep_min):
    """:func:`snap_lead_min` over arrays — same value, row by row.

    ``int(ceil(lead/step - 1e-9) * step)``, including the truncation of
    the product that ``int()`` does (visible only on a non-integer
    timestep). Pinned against the scalar function in the tests.
    """
    import numpy as np

    lead = np.asarray(lead_min, dtype=np.float64)
    step = np.asarray(timestep_min, dtype=np.float64)
    # A zero or absent timestep is a corpus row the join cannot place at
    # all; it is masked out by the caller, so the arithmetic only has to
    # stay quiet rather than be meaningful.
    with np.errstate(divide="ignore", invalid="ignore"):
        steps = np.ceil(lead / step - 1e-9)
        product = np.where(np.isfinite(steps), steps * step, 0.0)
        return np.trunc(np.where(np.isfinite(product), product, 0.0)).astype(np.int64)


def gauge_slot_sec_vec(instant_sec, slot_min: int = GAUGE_SLOT_MIN):
    """:func:`gauge_slot` over arrays of epoch seconds."""
    import numpy as np

    step = int(slot_min) * 60
    return (np.rint(np.asarray(instant_sec, dtype=np.float64) / step) * step).astype(
        np.int64
    )


def _months_in(first_sec: int, last_sec: int) -> list[tuple[int, int]]:
    """Every ``(year, month)`` an inclusive epoch-second window touches."""
    first = datetime.fromtimestamp(int(first_sec), tz=timezone.utc)
    last = datetime.fromtimestamp(int(last_sec), tz=timezone.utc)
    out: list[tuple[int, int]] = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        out.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


class GaugeGrid:
    """A dense ``(station × 10-minute slot)`` view of the two parameters.

    The old index was a pair of ``{(station, stamp): value}`` dicts built
    from ``to_pylist()`` over the whole window — several gigabytes of
    boxed Python objects for a ten-month archive, and a per-row dict
    lookup afterwards. This is the same data as two float32 arrays: ~35 MB
    for ten months of ~100 stations, filled a record batch at a time so a
    month never lands in memory whole, and read by fancy indexing.

    Missing readings are NaN. Raw values are kept exactly as the archive
    has them — the negative trace sentinel included — because the run
    reports how many traces it normalised.
    """

    __slots__ = ("stations", "index", "first_sec", "step_sec", "n_slots", "mm", "dur")

    def __init__(self, stations: list[str], first_sec: int, last_sec: int,
                 slot_min: int = GAUGE_SLOT_MIN) -> None:
        import numpy as np
        import pyarrow as pa

        self.stations = list(stations)
        self.index = pa.array(self.stations, type=pa.string())
        self.step_sec = int(slot_min) * 60
        self.first_sec = int(first_sec)
        self.n_slots = max(1, (int(last_sec) - self.first_sec) // self.step_sec + 1)
        shape = (len(self.stations), self.n_slots)
        self.mm = np.full(shape, np.nan, dtype=np.float32)
        self.dur = np.full(shape, np.nan, dtype=np.float32)

    @property
    def known_slots(self) -> tuple[int, int]:
        """``(amount readings, duration readings)`` the grid holds."""
        import numpy as np

        return (
            int(np.count_nonzero(~np.isnan(self.mm))),
            int(np.count_nonzero(~np.isnan(self.dur))),
        )

    def fill_from(self, store: StationObsStore) -> int:
        """Read every month the grid spans, one record batch at a time."""
        import pyarrow as pa

        if not self.stations:
            return 0
        pool = pa.default_memory_pool()
        rows = 0
        last_sec = self.first_sec + (self.n_slots - 1) * self.step_sec
        for year, month in _months_in(self.first_sec, last_sec):
            for batch in store.stream_month(
                year, month,
                [PRECIP_PAST_10MIN, PRECIP_DUR_PAST_10MIN],
                self.stations,
            ):
                rows += self._absorb(batch)
            # Arrow's allocator keeps freed pages by default, and ten
            # months of that is the difference between 200 MB and 600.
            pool.release_unused()
        return rows

    def _absorb(self, batch) -> int:
        import numpy as np
        import pyarrow as pa
        import pyarrow.compute as pc

        if batch.num_rows == 0:
            return 0
        station = np.asarray(
            pc.fill_null(
                pc.index_in(batch.column("station_id"), value_set=self.index), -1
            ).cast(pa.int64()),
            dtype=np.int64,
        )
        micros = np.asarray(
            batch.column("observed_utc").cast(pa.int64()), dtype=np.int64
        )
        values = np.asarray(
            pc.fill_null(batch.column("value"), float("nan")).cast(pa.float32()),
            dtype=np.float32,
        )
        is_amount = np.asarray(
            pc.fill_null(pc.equal(batch.column("parameter_id"), PRECIP_PAST_10MIN), False),
            dtype=bool,
        )
        # Exact stamp matching, exactly as the old dict lookup did: an
        # observation stamped off the 10-minute grid could never match a
        # verification slot then and is dropped now, rather than being
        # quietly rounded into a neighbouring slot.
        offset = micros // 1_000_000 - self.first_sec
        position = offset // self.step_sec
        good = (
            (station >= 0)
            & (offset % self.step_sec == 0)
            & (position >= 0)
            & (position < self.n_slots)
            & ~np.isnan(values)
        )
        flat = station * self.n_slots + position
        amount = good & is_amount
        duration = good & ~is_amount
        # Last writer wins, as the dicts did; the store dedupes on
        # (station, stamp, parameter), so duplicates are not expected.
        self.mm.reshape(-1)[flat[amount]] = values[amount]
        self.dur.reshape(-1)[flat[duration]] = values[duration]
        return int(batch.num_rows)

    def read(self, station_idx, slot_sec, take):
        """``(mm, dur)`` for the rows in ``take``; NaN off the grid."""
        import numpy as np

        mm = np.full(station_idx.shape, np.nan, dtype=np.float32)
        dur = np.full(station_idx.shape, np.nan, dtype=np.float32)
        if not self.stations:
            return mm, dur
        position = (np.asarray(slot_sec, dtype=np.int64) - self.first_sec) // self.step_sec
        inside = take & (position >= 0) & (position < self.n_slots) & (station_idx >= 0)
        if not inside.any():
            return mm, dur
        flat = station_idx[inside] * self.n_slots + position[inside]
        mm[inside] = self.mm.reshape(-1)[flat]
        dur[inside] = self.dur.reshape(-1)[flat]
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


def _row_inputs(table, np, pa, pc):
    """The five corpus columns the join needs, as numpy arrays.

    ``event_time`` is dictionary-decoded before parsing: a national corpus
    has ~4 000 distinct event times behind ~500 000 rows, so the Python
    datetime parsing runs over the uniques and the rows are a gather.
    """
    n = table.num_rows

    events = table.column("event_time").combine_chunks()
    event_null = np.asarray(pc.is_null(events), dtype=bool)
    uniques = pc.unique(events).drop_null()
    parsed = np.array(
        [int(parse_event_time(s).timestamp()) for s in uniques.to_pylist()],
        dtype=np.int64,
    ) if len(uniques) else np.zeros(0, dtype=np.int64)
    codes = np.asarray(
        pc.fill_null(pc.index_in(events, value_set=uniques), 0).cast(pa.int64()),
        dtype=np.int64,
    )
    event_sec = parsed[codes] if parsed.size else np.zeros(n, dtype=np.int64)

    points = table.column("point_id").combine_chunks()
    stations = sorted(p for p in pc.unique(points).drop_null().to_pylist() if p)
    station_idx = np.asarray(
        pc.fill_null(
            pc.index_in(points, value_set=pa.array(stations, pa.string())), -1
        ).cast(pa.int64()),
        dtype=np.int64,
    )

    leads = table.column("lead_min").combine_chunks()
    lead_null = np.asarray(pc.is_null(leads), dtype=bool)
    lead = np.asarray(
        pc.fill_null(leads, 0).cast(pa.float64()), dtype=np.float64
    )

    age = np.asarray(
        pc.fill_null(table.column("frame_age_min").combine_chunks(), 0.0).cast(
            pa.float64()
        ),
        dtype=np.float64,
    )

    steps = table.column("timestep_min").combine_chunks()
    timestep = np.asarray(
        pc.fill_null(steps, 0.0).cast(pa.float64()), dtype=np.float64
    )

    return stations, event_sec, event_null, station_idx, lead, lead_null, age, timestep


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
    ap.add_argument("--outcome-rule", choices=OUTCOME_RULES,
                    default=DEFAULT_OUTCOME_RULE,
                    help="How a row's gauge truth is scored. 'within' "
                         "(default) matches the served quantity and the "
                         "radar corpus: wet at ANY 10-min slot on the "
                         "instants T + j*timestep up to the lead's snapped "
                         "one. 'instant' is the pre-2026-09-11 rule — the "
                         "snapped slot alone — kept only to reproduce an "
                         "older curve. Recorded in gauge_outcome_rule.")
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

    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
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
        # pc.unique, not set(to_pylist()): a national corpus is 500k rows
        # and the answer is a handful of labels.
        available = sorted(
            pc.unique(table.column("point_set").combine_chunks()).drop_null().to_pylist()
        )
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
    rule = str(args.outcome_rule)
    print(f"corpus {args.corpus} rows={n} outcome_rule={rule}")

    (stations, event_sec, event_null, station_idx,
     lead, lead_null, age, timestep) = _row_inputs(table, np, pa, pc)

    # The verification window, row by row. ``snapped`` is the instant the
    # lead is DEFINED by; ``n_instants`` is how many timestep instants the
    # window covers, which is 1 under --outcome-rule instant.
    snapped = snap_lead_min_vec(lead + age, timestep)
    step_min = np.rint(timestep).astype(np.int64)
    valid = ~event_null & ~lead_null & (timestep != 0.0) & (station_idx >= 0)
    if rule == OUTCOME_RULE_INSTANT:
        n_instants = np.ones(n, dtype=np.int64)
    else:
        # A timestep that rounds to zero minutes has no window to walk.
        valid = valid & (step_min > 0)
        safe_step = np.where(step_min > 0, step_min, 1)
        # ceil(snapped / step), but never fewer than the one defining
        # instant — a zero-minute effective lead still verifies at T+0.
        n_instants = np.maximum(-(-np.maximum(snapped, 0) // safe_step), 1)

    # The gauge read covers exactly the slots the corpus asks for: the
    # earliest FIRST instant of any row's window to the latest final one.
    final_slot = gauge_slot_sec_vec(event_sec + snapped * 60)
    if rule == OUTCOME_RULE_INSTANT:
        first_slot = final_slot
    else:
        first_offset = np.minimum(np.where(step_min > 0, step_min, 0), snapped)
        first_slot = gauge_slot_sec_vec(event_sec + first_offset * 60)

    store = StationObsStore(args.corpus_dir)
    dead: set[str] = set()
    grid = GaugeGrid(stations, 0, 0)
    if valid.any() and stations:
        # The dead-gauge scan keeps the FINAL-instant window it has always
        # had: it names months, and widening it by an hour could pull in a
        # whole extra month of slots and change a station's verdict.
        finals = final_slot[valid]
        dead = _dead_gauges(
            args.corpus_dir, stations,
            datetime.fromtimestamp(int(finals.min()), tz=timezone.utc),
            datetime.fromtimestamp(int(finals.max()), tz=timezone.utc),
            int(args.min_known_slots),
        )
        grid = GaugeGrid(
            stations, int(first_slot[valid].min()), int(final_slot[valid].max()),
        )
        grid.fill_from(store)
    n_mm, n_dur = grid.known_slots
    print(f"gauge slots loaded: mm={n_mm} dur={n_dur} from {store.obs_dir} "
          f"({len(grid.stations)} station(s) x {grid.n_slots} slot(s))")

    # --- the window walk -------------------------------------------------
    # One pass per timestep instant (at most ~8 on the 10-min grid at the
    # served leads), each fully vectorised over rows. Everything that
    # outlives an iteration is one array per row.
    wet_any = np.zeros(n, dtype=bool)
    observed_count = np.zeros(n, dtype=np.int32)
    skipped = np.zeros(n, dtype=np.int32)
    mm_window = np.zeros(n, dtype=np.float64)
    final_observed = np.zeros(n, dtype=bool)
    final_mm = np.full(n, np.nan, dtype=np.float32)
    final_dur = np.full(n, np.nan, dtype=np.float32)

    max_instants = int(n_instants[valid].max()) if valid.any() else 0
    for j in range(1, max_instants + 1):
        active = valid & (j <= n_instants)
        if not active.any():
            continue
        if rule == OUTCOME_RULE_INSTANT:
            slot_sec = final_slot
        else:
            offset = np.minimum(j * step_min, snapped)
            slot_sec = gauge_slot_sec_vec(event_sec + offset * 60)
        raw_mm, dur = grid.read(station_idx, slot_sec, active)
        observed, wet, mm = slot_verdicts(
            raw_mm, dur, float(args.wet_mm), float(args.wet_dur_min)
        )
        is_final = active & (j == n_instants)
        wet_any |= active & wet
        observed_count += (active & observed).astype(np.int32)
        mm_window += np.where(active & observed, mm, 0.0)
        # A missing INTERMEDIATE slot is skipped and counted; a missing
        # FINAL slot is what nulls the row.
        skipped += (active & ~observed & ~is_final).astype(np.int32)
        final_observed |= is_final & observed
        final_mm = np.where(is_final, raw_mm, final_mm)
        final_dur = np.where(is_final, dur, final_dur)

    # --- the columns -----------------------------------------------------
    dead_station = np.array([s in dead for s in stations], dtype=bool)
    if dead_station.size:
        dead_row = (station_idx >= 0) & dead_station[np.maximum(station_idx, 0)]
    else:
        dead_row = np.zeros(n, dtype=bool)

    has_outcome = valid & final_observed & ~dead_row
    traces = int(np.count_nonzero(final_mm < 0.0))

    gauge_mm = pa.array(
        np.where(final_mm < 0.0, np.float32(0.0), final_mm),
        type=pa.float32(), mask=np.isnan(final_mm),
    )
    gauge_dur = pa.array(final_dur, type=pa.float32(), mask=np.isnan(final_dur))
    gauge_outcome = pa.array(
        wet_any.astype(np.int8), type=pa.int8(), mask=~has_outcome,
    )
    gauge_window = pa.array(
        mm_window.astype(np.float32), type=pa.float32(),
        mask=~(valid & (observed_count > 0)),
    )
    gauge_rule = pa.array([rule] * n, type=pa.string())

    out = (
        table
        .append_column("gauge_mm", gauge_mm)
        .append_column("gauge_dur_min", gauge_dur)
        .append_column("gauge_outcome", gauge_outcome)
        .append_column("gauge_mm_window", gauge_window)
        .append_column("gauge_outcome_rule", gauge_rule)
    )
    # The rule again where a reader looks before the rows: which event the
    # gauge column was scored against is not a detail a report may guess.
    metadata = dict(out.schema.metadata or {})
    metadata.update({
        b"gauge_outcome_rule": rule.encode(),
        b"gauge_wet_mm": repr(float(args.wet_mm)).encode(),
        b"gauge_wet_dur_min": repr(float(args.wet_dur_min)).encode(),
        b"gauge_min_known_slots": str(int(args.min_known_slots)).encode(),
        b"gauge_slot_min": str(int(GAUGE_SLOT_MIN)).encode(),
    })
    out = out.replace_schema_metadata(metadata)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    pq.write_table(out, tmp, compression="zstd")
    tmp.replace(args.out)

    joined = int(np.count_nonzero(has_outcome))
    print(f"joined={joined} null={n - joined} traces_normalised={traces} "
          f"dead_gauges={len(dead)}")
    if rule == OUTCOME_RULE_WITHIN:
        print(
            f"window: rule=within slots_scored={int(observed_count.sum())} "
            f"intermediate_slots_missing={int(skipped.sum())} "
            f"rows_with_a_gap={int(np.count_nonzero(skipped > 0))} "
            "— a skipped slot can only hide rain, never invent it"
        )
    else:
        print("window: rule=instant — one slot per row, the snapped instant")

    lead_key = np.where(lead_null, -1, lead.astype(np.int64))
    keys, inverse = np.unique(lead_key, return_inverse=True)
    totals = np.bincount(inverse, minlength=keys.size)
    joins = np.bincount(
        inverse, weights=has_outcome.astype(np.float64), minlength=keys.size
    )
    print(f"{'lead_min':>9} {'rows':>9} {'joined':>9} {'null':>9}")
    for key, total, hit in zip(keys.tolist(), totals.tolist(), joins.tolist()):
        print(f"{key:>9} {total:>9} {int(hit):>9} {total - int(hit):>9}")
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
