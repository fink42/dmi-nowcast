"""Scoring push warnings against DMI rain-gauge observations (Phase F).

The site's Web Push rule is a state machine over calibrated probability
(``push.engine.evaluate``). Asking "how good are the warnings?" means
replaying that rule at points where a *measurement* exists — the gauges —
and asking, for every warning it would have sent, whether rain actually
started at that gauge inside the promised window.

This module is the scoring half. It knows nothing about radar, STEPS or
the sidecar: it takes gauge rows in, decision rows in, and produces
hits / false alarms / misses / lead errors out. That keeps it testable
without a single HDF5 file, and keeps one definition of "a hit" shared by
the historical replay (``scripts/replay_warnings.py``) and the live
station evaluation step in the sidecar.

Definitions, all decided up front so they cannot drift between the two
consumers:

**A gauge slot** is a 10-minute interval stamped at its END — DMI's
``precip_past10min`` at 07:20Z covers (07:10Z, 07:20Z]. A slot is WET
when ``precip_past10min >= 0.1 mm`` OR ``precip_dur_past10min >= 1 min``;
the duration arm catches drizzle that rounds to 0.0 mm but wets a road.
A slot with neither parameter reported is ``None`` — *unknown*, never
dry. DMI's "trace of precipitation" sentinel (a negative amount, see
``metobs.normalize_precip_mm``) is an amount below 0.1 mm, so it is read
as 0.0 here and can only make a slot wet through the duration arm.

**An onset** is the first wet slot after at least ``dry_min`` minutes
(three slots at the default 30) of *known dry* slots. An unknown slot
cannot certify dryness, so it resets the dry run rather than extending
it: no onset is ever declared on the strength of missing data. The onset
timestamp is the slot END, which is the only instant the gauge actually
reports; the true first drop fell somewhere in the preceding 10 minutes,
so the measured gap between a warning and its onset is overstated by up
to 10 minutes and every lead error here is biased that far NEGATIVE (see
the sign convention below — negative reads as "the warning was early").
Stated once, here, rather than hidden in a correction factor.

**Lead error sign**: ``lead_error_min = eta_min − (onset − sent)``, in
minutes. POSITIVE means the rain arrived sooner than the notification
said it would — the warning was LATE and the user got less lead time
than promised, which is the failure that matters. NEGATIVE means the
rain came later than the ETA — the warning was early. The website's
quality page is built on this convention; do not flip it here without
flipping it there.

**Coverage** is the other half of the same honesty. An onset is only a
miss where a decision could have caught it. The gauge archive is
backfilled — months of 10-minute slots at a hundred stations — while
decision rows exist only for the frames the service actually evaluated,
and scoring the first against the second counts every rain event since
December as a miss. The first live report did exactly that: five
warnings, two hits, and 2 088 "misses" from a gauge archive the service
had never been running for.

So pass ``coverage`` — the intervals the decision rows actually span, from
:func:`coverage_runs` — and an unclaimed onset outside them is
``uncovered``: not a hit, not a miss, not in POD. A claimed onset is
settled by the claim, coverage or not — a hit, or ``miss_late`` when the
warning that claimed it was too late to be useful; the claim is evidence
in itself, and the run's tail is where a warning's own window legitimately
reaches past the last frame.

**Pending** is the third outcome, and the reason it exists is that a
report is built while the weather is still happening. A warning sent two
minutes ago promises rain within the next forty; the gauge has not
reported those forty minutes yet, so calling it a false alarm is not a
measurement, it is an accusation the evidence cannot support. Pass
``known_until`` — the last gauge slot end that station actually reported
— and any warning whose window ``sent + lead + tolerance`` reaches past
it, and which has claimed no onset, is PENDING: excluded from hits, false
alarms, POD, FAR and the lead-error quantiles alike, and counted on its
own. The same boundary applies to onsets: an unclaimed onset within
``tolerance`` of ``known_until`` is pending rather than a miss, because
DMI backfills late station reports and a slot near the edge can still
change. Without ``known_until`` nothing is pending and the scoring is
exactly what it was.

A warning that HAS claimed an onset is never pending, even with its
window still open: the onset is in the record, the claim is settled, and
no later slot can unmake it. Demoting it would take a confirmed hit out
of the numerator while leaving its onset in POD's denominator — turning
a hit into a miss, which is worse than the bug this rule fixes.

**A hit** is a warning with an onset in ``(sent, sent + lead + tolerance]``.
The tolerance exists because the warning promises "rain within LEAD
minutes" off a radar frame that is already 14–24 min old; a warning whose
rain lands 8 minutes late kept its promise in every sense the user cares
about. An onset is claimed by at most one warning and the earliest
warning claims it, so two warnings cannot both take credit for one rain
event, and an onset left unclaimed is a **miss**. A warning that claims
nothing is a **false alarm**.

**Late** is the fourth outcome, and it exists because a warning that
arrives while the user is already reaching for the door handle is not a
warning. Pass ``min_useful_lead_min`` and a claimed onset whose realised
lead — ``onset − sent``, in minutes — falls short of it makes the warning
``late`` and its onset ``miss_late``:

* not a hit — nobody was warned in any useful sense;
* still on the recall side — the rain came and the user was not usefully
  told, so it stays in the denominator of POD / recall;
* **not** a false alarm — the rain did arrive, and charging precision for
  it would punish a correct forecast for being a few minutes tight.

Precision therefore is ``hits / (hits + false_alarms)`` and does not see
lates at all, while ``far`` keeps its old denominator (every graded
warning, lates included), so ``far != 1 − precision`` as soon as a late
exists. Both are reported; neither is derived from the other. The claim
itself is unchanged: a late warning still consumes its onset, because the
rain that fell two minutes after the notification IS the rain the
notification was about, and letting a later onset be claimed instead
would flatter a rule that only ever warns at the last moment. The default
``min_useful_lead_min=0.0`` makes nothing late and leaves every number
exactly what it was.

Everything is timezone-aware UTC. A naive datetime is a programming
error and is raised on rather than guessed at.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "SLOT_MIN",
    "WET_PRECIP_MM",
    "WET_DUR_MIN",
    "PRECIP_PARAM",
    "PRECIP_DUR_PARAM",
    "DEFAULT_DRY_MIN",
    "DEFAULT_LEAD_MIN",
    "DEFAULT_TOLERANCE_MIN",
    "DEFAULT_MIN_USEFUL_LEAD_MIN",
    "F_BETAS",
    "DECISION_COLUMNS",
    "DEFAULT_PRODUCT_LEADS_MIN",
    "align_decision_table",
    "concat_decision_tables",
    "decision_columns",
    "decision_leads_in",
    "decision_schema",
    "decision_table",
    "p_rain_column",
    "parse_p_rain_column",
    "per_lead_columns",
    "DEFAULT_COVERAGE_GAP_MIN",
    "slot_end_of",
    "gauge_slots",
    "onsets",
    "coverage_runs",
    "WarningOutcome",
    "OnsetOutcome",
    "ScoreResult",
    "score_warnings",
    "pooled_summary",
    "skill_scores",
    "raining_now_agreement",
]

#: Gauge reporting cadence, minutes. DMI's 10-minute precipitation
#: parameters; the whole slot grid is built on this.
SLOT_MIN = 10

#: Wet-slot thresholds (decided 2026-09-05, Phase F). Either arm is enough.
WET_PRECIP_MM = 0.1
WET_DUR_MIN = 1.0

#: The two metObs parameters the wet rule reads.
PRECIP_PARAM = "precip_past10min"
PRECIP_DUR_PARAM = "precip_dur_past10min"

#: Minutes of known-dry slots that must precede a wet slot for it to be
#: an onset (three slots at the 10-min cadence).
DEFAULT_DRY_MIN = 30
#: The live subscription's lead, and the grace period allowed on top of it.
DEFAULT_LEAD_MIN = 30
DEFAULT_TOLERANCE_MIN = 10

#: Minutes of realised lead below which a warning is ``late`` rather than a
#: hit. Zero here — the scorer's own default changes nothing — because the
#: historical numbers on the quality page were produced without it. The
#: threshold fit (``scripts/sweep_thresholds.py``) passes 5.
DEFAULT_MIN_USEFUL_LEAD_MIN = 0.0

#: The β values ``f_beta`` reports, keyed by their string form: β < 1
#: weights precision (fewer wasted notifications), β > 1 weights recall
#: (less rain missed), β = 1 is F1, the objective the horizon fit maximises.
F_BETAS: tuple[float, ...] = (0.5, 1.0, 2.0)

#: The longest gap between consecutive decision rows that still counts as
#: continuous coverage: two radar cycles at DMI's 10-minute cadence. One
#: missed cycle is a hiccup and the rain either side of it was still being
#: watched; three hours is an outage, and nothing in it was.
DEFAULT_COVERAGE_GAP_MIN = 20


# ---------------------------------------------------------------------------
# The decision row — one shape, two writers
# ---------------------------------------------------------------------------

#: The columns EVERY decision row carries, in order. The historical replay
#: and the live ``station_eval`` step in the sidecar both append rows of
#: this shape, so a replay parquet and a live parquet concatenate without
#: a translation layer and this module can score either.
#:
#: ``p_rain`` is the probability at the *rule's* lead — the number the
#: decision was actually taken on. It is joined on disk by one
#: ``p_rain_<lead>`` column per served lead (:func:`decision_columns`), so
#: a threshold/horizon sweep can be run offline against the gauges without
#: re-running STEPS. This base tuple stays fixed: readers pin it, and files
#: written before the per-lead columns existed have exactly these.
DECISION_COLUMNS: tuple[str, ...] = (
    "radar_ts",
    "generated_at",
    "station_id",
    "p_rain",
    "eta_min",
    "intensity_mm_h",
    "observed_mm_h",
    "forecast_now_mm_h",
    "action",
    "armed_after",
    "streak_after",
)

#: Default per-lead probability columns: the leads the national products
#: publish (mirrors ``national.DEFAULT_LEADS_MIN`` — restated rather than
#: imported so this module's import graph stays dependency-free).
DEFAULT_PRODUCT_LEADS_MIN: tuple[int, ...] = (10, 20, 30, 45, 60)

_P_RAIN_PREFIX = "p_rain_"


def p_rain_column(lead: int) -> str:
    """Column name for the probability at ``lead`` minutes."""
    return f"{_P_RAIN_PREFIX}{int(lead)}"


def parse_p_rain_column(name: str) -> int | None:
    """``"p_rain_30"`` → ``30``; anything else → ``None``."""
    if not name.startswith(_P_RAIN_PREFIX):
        return None
    tail = name[len(_P_RAIN_PREFIX):]
    return int(tail) if tail.isdigit() else None


def _leads(leads_min: Iterable[int] | None) -> tuple[int, ...]:
    if leads_min is None:
        leads_min = DEFAULT_PRODUCT_LEADS_MIN
    return tuple(sorted({int(lead) for lead in leads_min}))


def decision_columns(leads_min: Iterable[int] | None = None) -> tuple[str, ...]:
    """:data:`DECISION_COLUMNS` plus one ``p_rain_<lead>`` per served lead."""
    return DECISION_COLUMNS + tuple(p_rain_column(lead) for lead in _leads(leads_min))


def decision_schema(leads_min: Iterable[int] | None = None):
    """Arrow schema for :func:`decision_columns`.

    pyarrow is imported lazily: this module's scoring functions are pure
    Python and must stay importable in an environment that has no Arrow
    (the core package lists pyarrow as a dev dependency only).

    Every forecast field is nullable float32 — ``None`` from
    ``sample_point`` means "off coverage / nodata", which is emphatically
    not zero, and the null survives to the parquet so a reader cannot
    silently average it as a dry sample. That applies to the per-lead
    columns too: a lead the cycle did not publish reads null, never 0 %.

    ``leads_min`` defaults to :data:`DEFAULT_PRODUCT_LEADS_MIN`. Pass the
    cycle's own ``products.leads_min`` when writing, so a config that
    serves different leads writes the columns it actually has.
    """
    import pyarrow as pa

    fields = [
        ("radar_ts", pa.timestamp("us", tz="UTC")),
        ("generated_at", pa.timestamp("us", tz="UTC")),
        ("station_id", pa.string()),
        ("p_rain", pa.float32()),
        ("eta_min", pa.float32()),
        ("intensity_mm_h", pa.float32()),
        ("observed_mm_h", pa.float32()),
        ("forecast_now_mm_h", pa.float32()),
        ("action", pa.string()),
        ("armed_after", pa.bool_()),
        ("streak_after", pa.int32()),
    ]
    fields += [(p_rain_column(lead), pa.float32()) for lead in _leads(leads_min)]
    return pa.schema(fields)


def per_lead_columns(p_rain: Mapping[int, float | None] | None) -> dict:
    """``sample_point``'s ``p_rain`` dict → the ``p_rain_<lead>`` row fields.

    One place, two writers: the replay and the live step must name and
    populate these identically or their parquet files stop concatenating.
    """
    if not p_rain:
        return {}
    return {p_rain_column(lead): value for lead, value in p_rain.items()}


def decision_table(rows: Sequence[Mapping[str, Any]], leads_min=None):
    """Row dicts → an Arrow table in the decision schema.

    A row missing a column contributes a null, which is what lets a writer
    hand over rows built before a lead was served.
    """
    import pyarrow as pa

    schema = decision_schema(leads_min)
    return pa.table(
        {
            name: pa.array(
                [row.get(name) for row in rows], type=schema.field(name).type,
            )
            for name in schema.names
        },
        schema=schema,
    )


def decision_leads_in(table) -> tuple[int, ...]:
    """The lead times a decision table (or column-name list) carries."""
    names = table.schema.names if hasattr(table, "schema") else list(table)
    found = {parse_p_rain_column(name) for name in names}
    return tuple(sorted(lead for lead in found if lead is not None))


def align_decision_table(table, leads_min: Iterable[int] | None = None):
    """Conform a decision table to the schema, filling absent columns with nulls.

    This is what makes a parquet written before the per-lead columns
    existed readable beside one written after: the target schema is the
    UNION of the requested leads and the leads the file already has, so
    nothing on disk is dropped and nothing missing reads as a value. Types
    are cast rather than assumed, so a file written by an older float64
    build still lines up.
    """
    import pyarrow as pa

    leads = tuple(sorted(set(_leads(leads_min)) | set(decision_leads_in(table))))
    schema = decision_schema(leads)
    present = set(table.schema.names)
    columns = [
        table.column(field.name).cast(field.type)
        if field.name in present
        else pa.nulls(table.num_rows, type=field.type)
        for field in schema
    ]
    return pa.table(columns, schema=schema)


def concat_decision_tables(tables, leads_min: Iterable[int] | None = None):
    """Concatenate decision tables written under different lead sets.

    The report producer reads a directory of per-day parquet files that may
    straddle the day the per-lead columns were added; this aligns every one
    of them to the union schema first, so the concatenation cannot fail on
    a schema mismatch.
    """
    import pyarrow as pa

    tables = list(tables)
    leads = set(_leads(leads_min))
    for table in tables:
        leads |= set(decision_leads_in(table))
    leads = tuple(sorted(leads))
    if not tables:
        return decision_schema(leads).empty_table()
    return pa.concat_tables(
        [align_decision_table(table, leads) for table in tables]
    )


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _as_utc(value: datetime, what: str = "datetime") -> datetime:
    """Reject naive datetimes; normalise everything else to UTC."""
    if not isinstance(value, datetime):
        raise TypeError(f"{what} must be a datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{what} must be timezone-aware (UTC)")
    return value.astimezone(timezone.utc)


def slot_end_of(ts: datetime, *, slot_min: int = SLOT_MIN) -> datetime:
    """The end stamp of the slot containing ``ts``.

    Slots are half-open at the start and closed at the end — an
    observation stamped exactly on a boundary IS that slot's end, it does
    not roll into the next one. Sub-second precision is discarded, which
    is what makes a raw metObs stamp and a synthesised grid instant
    compare equal.
    """
    ts = _as_utc(ts, "ts")
    base = ts.replace(second=0, microsecond=0)
    remainder = base.minute % slot_min
    if remainder == 0 and ts == base:
        return base
    return base - timedelta(minutes=remainder) + timedelta(minutes=slot_min)


def _rows_of(table: Any) -> list[Mapping[str, Any]]:
    """Rows from a pyarrow Table, or from any sequence of mappings.

    Duck-typed on ``to_pylist`` so the scoring tests can hand in plain
    dicts and stay independent of both pyarrow and the station store the
    Phase F backfill writes.
    """
    if hasattr(table, "to_pylist"):
        return list(table.to_pylist())
    return [dict(row) for row in table]


def _amount_mm(value: Any) -> float | None:
    """A precipitation amount, with DMI's trace sentinel folded to 0.0.

    A negative reading is DMI's "traces of precipitation, less than
    0.1 kg/m²" marker (``metobs.normalize_precip_mm``): it is a statement
    that the amount is below the wet threshold, never a negative depth.
    Handled inline rather than by importing ``metobs`` so this module
    keeps its numpy-free, dependency-free import graph.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return 0.0 if v < 0.0 else v


# ---------------------------------------------------------------------------
# Gauge slots and onsets
# ---------------------------------------------------------------------------


def gauge_slots(
    table: Any,
    station_id: str,
    *,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
    slot_min: int = SLOT_MIN,
) -> list[tuple[datetime, bool | None]]:
    """One station's 10-min slots over a contiguous grid: ``(slot_end, wet)``.

    ``table`` is what ``StationObsStore.read`` returns (columns
    ``station_id`` / ``observed_utc`` / ``parameter_id`` / ``value``), or
    any sequence of mappings with those keys.

    The result is a **contiguous** grid — every slot between the first and
    last covered instant appears exactly once, in order — because the
    onset rule counts consecutive dry slots and cannot do that over a list
    with silent holes. A slot the station did not report is ``None``.

    ``wet`` is ``True`` when either arm of the rule fires, ``False`` when
    at least one arm reported and neither fired, and ``None`` when the
    station reported neither parameter for that slot. A station that
    reports only the duration parameter is therefore still scoreable —
    unknown means *nothing was said*, not *one thing was missing*.

    ``start_utc`` / ``end_utc`` pin the grid (both inclusive, snapped to
    slot ends); by default it spans the station's own rows. Pin them when
    several stations must share one grid, or when a caller wants the pad
    around a day.
    """
    if slot_min <= 0:
        raise ValueError("slot_min must be positive")
    wanted = str(station_id)
    #: slot end → [amount arm fired?, duration arm fired?] merged over rows
    seen: dict[datetime, bool] = {}
    for row in _rows_of(table):
        if str(row.get("station_id")) != wanted:
            continue
        observed = row.get("observed_utc")
        if observed is None:
            continue
        slot = slot_end_of(_as_utc(observed, "observed_utc"), slot_min=slot_min)
        param = str(row.get("parameter_id"))
        if param not in (PRECIP_PARAM, PRECIP_DUR_PARAM):
            continue  # some other parameter rode along in the read
        value = _amount_mm(row.get("value"))
        if value is None:
            # Reported but unusable (null / NaN) — says nothing either way,
            # so it must not turn an unknown slot into a dry one.
            continue
        fired = (
            value >= WET_PRECIP_MM
            if param == PRECIP_PARAM
            else value >= WET_DUR_MIN
        )
        seen[slot] = seen.get(slot, False) or fired

    if start_utc is not None:
        first = slot_end_of(start_utc, slot_min=slot_min)
    elif seen:
        first = min(seen)
    else:
        return []
    if end_utc is not None:
        last = slot_end_of(end_utc, slot_min=slot_min)
    elif seen:
        last = max(seen)
    else:
        return []
    if last < first:
        return []

    step = timedelta(minutes=slot_min)
    out: list[tuple[datetime, bool | None]] = []
    cursor = first
    while cursor <= last:
        out.append((cursor, seen.get(cursor)))
        cursor += step
    return out


def onsets(
    slots: Sequence[tuple[datetime, bool | None]],
    dry_min: int = DEFAULT_DRY_MIN,
    *,
    slot_min: int = SLOT_MIN,
) -> list[datetime]:
    """Onset instants: the first wet slot after ``dry_min`` of known dry.

    ``ceil(dry_min / slot_min)`` consecutive dry slots are required — three
    at the defaults, and *exactly* three is enough. An unknown slot resets
    the run (missing data never certifies a dry spell), and so does a gap
    in the grid, so a list stitched from two separate days cannot invent
    an onset across the seam.

    The first slots of a record can never be onsets: there is no evidence
    about what came before them. That is deliberate — it costs the odd
    genuine event at a window edge and buys the guarantee that every onset
    reported here is one the gauge actually witnessed starting.
    """
    need = max(1, math.ceil(dry_min / slot_min))
    step = timedelta(minutes=slot_min)
    out: list[datetime] = []
    dry_run = 0
    previous: datetime | None = None
    for ts, wet in slots:
        ts = _as_utc(ts, "slot end")
        if previous is not None and ts - previous != step:
            dry_run = 0  # a hole in the grid is not a dry spell
        previous = ts
        if wet is None:
            dry_run = 0
            continue
        if wet:
            if dry_run >= need:
                out.append(ts)
            dry_run = 0
        else:
            dry_run += 1
    return out


# ---------------------------------------------------------------------------
# Coverage: the intervals a decision row could actually have caught rain in
# ---------------------------------------------------------------------------


def coverage_runs(
    timestamps: Iterable[datetime],
    *,
    max_gap_min: int = DEFAULT_COVERAGE_GAP_MIN,
    extend_min: float = 0.0,
) -> list[tuple[datetime, datetime]]:
    """Merge decision timestamps into the runs they actually cover.

    ``timestamps`` are one station's decision instants — the ``radar_ts``
    of every evaluated frame. Consecutive stamps no more than
    ``max_gap_min`` apart belong to the same run; a longer gap ends it,
    because nothing was being watched in between and rain that fell there
    was never anybody's to catch.

    Each run's end is pushed out by ``extend_min`` — the caller passes
    ``lead_min + tolerance_min`` — since the last frame of a run makes a
    promise about the following half hour, and an onset that lands inside
    that promise is squarely in scope even though no later frame exists.

    Returns disjoint, sorted ``[(start, end)]``. Empty input, empty list:
    no decisions, no coverage, and nothing to score against.
    """
    if max_gap_min <= 0:
        raise ValueError("max_gap_min must be positive")
    stamps = sorted({_as_utc(ts, "decision timestamp") for ts in timestamps})
    if not stamps:
        return []
    gap = timedelta(minutes=max_gap_min)
    tail = timedelta(minutes=float(extend_min))
    runs: list[tuple[datetime, datetime]] = []
    start = previous = stamps[0]
    for ts in stamps[1:]:
        if ts - previous > gap:
            runs.append((start, previous + tail))
            start = ts
        previous = ts
    runs.append((start, previous + tail))
    return runs


def _covered(ts: datetime, runs: Sequence[tuple[datetime, datetime]]) -> bool:
    """Whether ``ts`` falls inside any run (both ends inclusive)."""
    return any(start <= ts <= end for start, end in runs)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WarningOutcome:
    """One replayed warning and what the gauge said about it."""

    sent_utc: datetime
    eta_min: float | None
    #: ``"hit"``, ``"late"`` (an onset claimed with less than
    #: ``min_useful_lead_min`` of realised lead), ``"false_alarm"``, or
    #: ``"pending"`` — the last only when ``known_until`` says the window
    #: has not closed yet.
    outcome: str
    onset_utc: datetime | None = None
    #: ``eta - (onset - sent)``, minutes. POSITIVE = the rain arrived
    #: SOONER than the ETA said, i.e. the warning was late and the user got
    #: less lead than promised; NEGATIVE = it came later, the warning was
    #: early. ``None`` when the warning missed, or carried no ETA.
    lead_error_min: float | None = None


@dataclass(frozen=True)
class OnsetOutcome:
    """One gauge onset and the warning (if any) that claimed it."""

    onset_utc: datetime
    #: ``"hit"``, ``"miss_late"`` (claimed, but too late to be useful),
    #: ``"miss"``, ``"pending"`` (an unclaimed onset too close to
    #: ``known_until`` for the gauge's word to be final), or ``"uncovered"``
    #: (no decision row was watching that instant).
    outcome: str
    sent_utc: datetime | None = None
    lead_error_min: float | None = None


@dataclass(frozen=True)
class ScoreResult:
    warnings: tuple[WarningOutcome, ...] = ()
    onsets: tuple[OnsetOutcome, ...] = ()
    summary: dict = field(default_factory=dict)


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    """p25 / p50 / p75 with linear interpolation (numpy's default rule).

    Implemented by hand so this module needs no numpy: the samples here
    number in the tens, and an explicit formula is easier to check against
    a hand-worked test than a call into a library's percentile machinery.
    """
    n = len(values)
    if n == 0:
        return {"p25": None, "p50": None, "p75": None, "n": 0}
    ordered = sorted(float(v) for v in values)

    def q(p: float) -> float:
        if n == 1:
            return ordered[0]
        pos = (n - 1) * p
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return ordered[int(pos)]
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

    return {"p25": q(0.25), "p50": q(0.5), "p75": q(0.75), "n": n}


def _f_beta(precision: float | None, recall: float | None, beta: float) -> float | None:
    """``(1 + β²)·P·R / (β²·P + R)``, or ``None`` where it has no value.

    Undefined means undefined: if either rate could not be measured, or if
    both are zero (a rule that caught nothing, warned about nothing, or
    both), there is no harmonic mean to report and a 0.0 would read as a
    measurement rather than an absence.
    """
    if precision is None or recall is None:
        return None
    b2 = float(beta) ** 2
    denom = b2 * precision + recall
    if denom <= 0.0:
        return None
    return (1.0 + b2) * precision * recall / denom


def skill_scores(
    hits: int, false_alarms: int, misses: int, late: int = 0,
) -> dict[str, Any]:
    """Precision / recall / F-scores / CSI from one confusion count.

    One definition, shared by the per-station summary and the pooled one,
    so a national number and a station number cannot drift apart:

    * ``precision = hits / (hits + false_alarms)`` — lates are absent from
      both halves: the rain came, so the notification was not wrong.
    * ``recall = hits / (hits + misses + late)`` — a late warning leaves
      the rain effectively unwarned, so it sits with the misses.
    * ``csi = hits / (hits + misses + late + false_alarms)`` — the
      meteorologists' single number, with late on the miss side exactly as
      in recall.

    Every rate is ``None`` when its denominator is empty. A rate over no
    events is not zero.
    """
    predicted = hits + false_alarms
    actual = hits + misses + late
    precision = (hits / predicted) if predicted else None
    recall = (hits / actual) if actual else None
    denom = hits + misses + late + false_alarms
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f_beta(precision, recall, 1.0),
        "f_beta": {
            f"{beta:g}": _f_beta(precision, recall, beta) for beta in F_BETAS
        },
        "csi": (hits / denom) if denom else None,
    }


def score_warnings(
    warnings: Iterable[tuple[datetime, float | None]],
    onset_times: Sequence[datetime],
    *,
    lead_min: int = DEFAULT_LEAD_MIN,
    tolerance_min: int = DEFAULT_TOLERANCE_MIN,
    dry_min: int = DEFAULT_DRY_MIN,
    known_until: datetime | None = None,
    coverage: Sequence[tuple[datetime, datetime]] | None = None,
    min_useful_lead_min: float = DEFAULT_MIN_USEFUL_LEAD_MIN,
) -> ScoreResult:
    """Match warnings to onsets; return per-warning, per-onset and totals.

    ``warnings`` is ``[(sent_utc, eta_min)]`` for ONE station (an ETA of
    ``None`` is allowed — the warning still counts, it just contributes no
    lead error). ``onset_times`` is that station's onsets from
    :func:`onsets`.

    Matching is a one-to-one greedy assignment in send order: each warning
    claims the earliest still-unclaimed onset inside its window
    ``(sent, sent + lead_min + tolerance_min]``. Warnings that claim
    nothing are false alarms; onsets nothing claimed are misses. Hence
    ``hits + late + false_alarms == warnings`` and ``hits + misses + late
    == n_onsets − pending_onsets − uncovered_onsets`` always hold, which is
    what makes POD and FAR readable side by side.

    ``min_useful_lead_min`` is the shortest realised lead — ``onset −
    sent`` — that still counts as a warning. A claim below it is ``late``
    (the onset ``miss_late``): out of the hits, out of precision's
    denominator, still in recall's. The default 0.0 makes nothing late.
    Lead-error quantiles stay a property of the HITS, so a late warning's
    error is recorded on its own row but does not move the median: the
    spread answers "when we warned in time, how close was the ETA?", and
    folding in the warnings that arrived too late would answer two
    questions with one number.

    ``coverage`` is that station's decision runs from :func:`coverage_runs`.
    Give it and an unclaimed onset outside every run is ``uncovered``
    rather than a miss: the gauge archive runs back to the backfill, the
    decisions only cover the frames the service evaluated, and counting
    the difference as misses measures the archive's depth rather than the
    service's skill. A CLAIMED onset stays a hit regardless — the claim is
    its own evidence.

    ``known_until`` is the last gauge slot end this station reported. Give
    it and the scoring gains a third outcome — see the module docstring:
    a warning whose window has not closed yet, and which has claimed no
    onset, is ``pending`` rather than a false alarm, and an unclaimed
    onset within ``tolerance_min`` of ``known_until`` is ``pending``
    rather than a miss. Both are excluded from every rate. Omit it (the
    default) and nothing is pending, which is the right behaviour for a
    closed historical window.

    A second warning cannot inherit an onset an earlier warning already
    took: two warnings for one rain event means one of them was noise, and
    counting it as a hit would hide exactly the spam this scoring exists
    to detect.

    ``dry_min`` takes no part in the matching — the onsets arrive already
    computed. It is carried into the summary so a stored result records
    the onset definition it was produced under.
    """
    window = timedelta(minutes=lead_min + tolerance_min)
    grace = timedelta(minutes=tolerance_min)
    horizon = _as_utc(known_until, "known_until") if known_until is not None else None
    runs = (
        None if coverage is None
        else [
            (_as_utc(a, "coverage start"), _as_utc(b, "coverage end"))
            for a, b in coverage
        ]
    )
    sent_list = sorted(
        ((_as_utc(sent, "sent_utc"), eta) for sent, eta in warnings),
        key=lambda pair: pair[0],
    )
    onset_list = sorted(_as_utc(o, "onset") for o in onset_times)
    claimed_by: list[datetime | None] = [None] * len(onset_list)
    claimed_error: list[float | None] = [None] * len(onset_list)
    claimed_late: list[bool] = [False] * len(onset_list)

    warning_rows: list[WarningOutcome] = []
    lead_errors: list[float] = []
    hits = 0
    pending = 0
    late = 0
    useful = float(min_useful_lead_min)
    for sent, eta in sent_list:
        pick: int | None = None
        for i, onset in enumerate(onset_list):
            if onset <= sent:
                continue
            if onset > sent + window:
                break
            if claimed_by[i] is not None:
                continue
            pick = i
            break
        if pick is None:
            if horizon is not None and sent + window > horizon:
                # The promise has not come due yet. Grading it now would
                # only measure how recently the report was built.
                pending += 1
                warning_rows.append(WarningOutcome(sent, eta, "pending"))
                continue
            warning_rows.append(WarningOutcome(sent, eta, "false_alarm"))
            continue
        onset = onset_list[pick]
        realised = (onset - sent).total_seconds() / 60.0
        # A claim below the useful lead is still a claim — the onset is
        # consumed either way — but the warning did not do its job.
        in_time = realised >= useful
        error: float | None = None
        if eta is not None:
            # Predicted lead minus delivered lead: positive = the rain beat
            # the ETA, so the warning was late. See the module docstring.
            error = float(eta) - realised
            if in_time:
                lead_errors.append(error)
        claimed_by[pick] = sent
        claimed_error[pick] = error
        claimed_late[pick] = not in_time
        if in_time:
            hits += 1
        else:
            late += 1
        warning_rows.append(WarningOutcome(
            sent, eta, "hit" if in_time else "late", onset, error,
        ))

    onset_rows = tuple(
        OnsetOutcome(
            onset,
            _onset_outcome(
                claimed_by[i], claimed_late[i], onset, horizon, grace, runs,
            ),
            claimed_by[i],
            claimed_error[i],
        )
        for i, onset in enumerate(onset_list)
    )
    n_sent = len(warning_rows)
    scored = n_sent - pending
    false_alarms = scored - hits - late
    pending_onsets = sum(1 for row in onset_rows if row.outcome == "pending")
    uncovered = sum(1 for row in onset_rows if row.outcome == "uncovered")
    misses = len(onset_list) - hits - late - pending_onsets - uncovered
    skill = skill_scores(hits, false_alarms, misses, late)
    summary = {
        # ``warnings`` is the SCORED count, so hits + late + false_alarms
        # adds up to it in the sentence the page writes. ``n_sent`` keeps
        # the raw total honest alongside.
        "warnings": scored,
        "n_sent": n_sent,
        "pending": pending,
        "hits": hits,
        "late": late,
        "false_alarms": false_alarms,
        "misses": misses,
        "pending_onsets": pending_onsets,
        "uncovered_onsets": uncovered,
        "n_onsets": len(onset_list),
        # POD and recall are the same number by construction, kept under
        # both names so a meteorologist and a product decision can read the
        # same summary without translating.
        "pod": skill["recall"],
        "far": (false_alarms / scored) if scored else None,
        "precision": skill["precision"],
        "recall": skill["recall"],
        "f1": skill["f1"],
        "f_beta": skill["f_beta"],
        "csi": skill["csi"],
        "lead_error_min": _quantiles(lead_errors),
        "lead_min": int(lead_min),
        "tolerance_min": int(tolerance_min),
        "dry_min": int(dry_min),
        "min_useful_lead_min": float(min_useful_lead_min),
        "known_until": horizon,
        "coverage_runs": 0 if runs is None else len(runs),
    }
    return ScoreResult(tuple(warning_rows), onset_rows, summary)


def _onset_outcome(
    claimed_by: datetime | None,
    claimed_late: bool,
    onset: datetime,
    horizon: datetime | None,
    grace: timedelta,
    runs: Sequence[tuple[datetime, datetime]] | None,
) -> str:
    """``hit`` / ``miss_late`` / ``miss`` / ``pending`` / ``uncovered``.

    A claimed onset is settled — a hit, or ``miss_late`` when the warning
    that claimed it arrived too late to be useful — and neither coverage
    nor the gauge's horizon downgrades it further. An unclaimed one
    is a miss only where the service could have caught it: inside a
    decision run (else ``uncovered`` — nobody was watching), and far
    enough from the gauge's last word for that word to be final (else
    ``pending`` — DMI backfills late station reports, so a slot within
    ``tolerance`` of the edge can still move, and with it the onset
    instant a warning would have had to match).
    """
    if claimed_by is not None:
        return "miss_late" if claimed_late else "hit"
    if runs is not None and not _covered(onset, runs):
        return "uncovered"
    if horizon is not None and onset + grace > horizon:
        return "pending"
    return "miss"


def pooled_summary(results: Iterable[ScoreResult], **params: Any) -> dict:
    """Totals over several stations' :class:`ScoreResult`.

    Counts add; POD, FAR, precision, recall, the F-scores, CSI and the
    lead-error quantiles are recomputed from the pooled populations rather
    than averaged, because a station with two warnings and a station with
    two hundred must not carry the same weight in a national number.

    Pending and uncovered onsets, and pending warnings, pool as their own
    counts and stay out of every rate, exactly as they do per station — a
    station whose window is still open must not drag the national FAR up
    for the fifteen minutes before its gauge reports, and a station whose
    gauge archive predates the service must not drag POD to zero with
    rain nobody was watching for.
    """
    rows = list(results)
    warnings = [w for r in rows for w in r.warnings]
    onset_rows = [o for r in rows for o in r.onsets]
    hits = sum(1 for w in warnings if w.outcome == "hit")
    late = sum(1 for w in warnings if w.outcome == "late")
    pending = sum(1 for w in warnings if w.outcome == "pending")
    scored = len(warnings) - pending
    false_alarms = scored - hits - late
    misses = sum(1 for o in onset_rows if o.outcome == "miss")
    pending_onsets = sum(1 for o in onset_rows if o.outcome == "pending")
    uncovered = sum(1 for o in onset_rows if o.outcome == "uncovered")
    # The quantiles are over the HITS, exactly as they are per station: a
    # late warning has an error, and it is on its own row, but it is not
    # part of "how close was the ETA when we warned in time?".
    errors = [
        w.lead_error_min for w in warnings
        if w.outcome == "hit" and w.lead_error_min is not None
    ]
    skill = skill_scores(hits, false_alarms, misses, late)
    out = {
        "warnings": scored,
        "n_sent": len(warnings),
        "pending": pending,
        "hits": hits,
        "late": late,
        "false_alarms": false_alarms,
        "misses": misses,
        "pending_onsets": pending_onsets,
        "uncovered_onsets": uncovered,
        "n_onsets": len(onset_rows),
        "pod": skill["recall"],
        "far": (false_alarms / scored) if scored else None,
        "precision": skill["precision"],
        "recall": skill["recall"],
        "f1": skill["f1"],
        "f_beta": skill["f_beta"],
        "csi": skill["csi"],
        "lead_error_min": _quantiles(errors),
    }
    out.update(params)
    return out


# ---------------------------------------------------------------------------
# "Is it raining now" agreement
# ---------------------------------------------------------------------------


def _skill(hits: int, misses: int, false_alarms: int, correct_neg: int) -> dict:
    total = hits + misses + false_alarms + correct_neg
    return {
        "n": total,
        "agreement": ((hits + correct_neg) / total) if total else None,
        "pod": (hits / (hits + misses)) if (hits + misses) else None,
        "far": (false_alarms / (hits + false_alarms))
        if (hits + false_alarms)
        else None,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_neg,
    }


def raining_now_agreement(
    rows: Iterable[Mapping[str, Any]],
    slots_by_station: Mapping[str, Sequence[tuple[datetime, bool | None]]]
    | None = None,
    *,
    threshold_mm_h: float = 0.5,
    slot_min: int = SLOT_MIN,
) -> dict:
    """How well "it is raining here now" matches the gauge, two ways.

    The site answers "is it raining at this point right now" from the
    lead-0 **deterministic forecast** — the newest composite advected to
    wall-clock now — because that composite is 14–24 min old by the time
    anyone reads it. Whether that beats simply reporting the ageing
    observation is an empirical question, and this is the measurement:
    both series are scored against the same gauge slot, so the comparison
    is paired and the difference is the value the advection adds.

    ``rows`` are decision rows (:data:`DECISION_COLUMNS` or any mapping
    with ``generated_at`` / ``station_id`` / ``forecast_now_mm_h`` /
    ``observed_mm_h``). Gauge truth comes from ``slots_by_station`` — the
    slot CONTAINING ``generated_at`` — or from a ``gauge_wet`` key on the
    row itself when the caller resolved it already.

    A row whose gauge slot is unknown, or whose series value is ``None``
    (off coverage / nodata), is skipped **for that series only**: the two
    series therefore report their own ``n``, and a cycle where the
    observed grid failed still contributes its forecast sample.
    """
    lookup: dict[str, dict[datetime, bool | None]] = {}
    for station, slots in (slots_by_station or {}).items():
        lookup[str(station)] = {
            _as_utc(ts, "slot end"): wet for ts, wet in slots
        }

    counts = {
        "forecast_now_mm_h": [0, 0, 0, 0],  # hits, misses, FA, correct neg
        "observed_mm_h": [0, 0, 0, 0],
    }
    n_rows = 0
    n_scored = 0
    n_wet = 0
    for row in rows:
        n_rows += 1
        generated = row.get("generated_at")
        if generated is None:
            continue
        instant = slot_end_of(_as_utc(generated, "generated_at"), slot_min=slot_min)
        if "gauge_wet" in row:
            wet = row.get("gauge_wet")
        else:
            wet = lookup.get(str(row.get("station_id")), {}).get(instant)
        if wet is None:
            continue
        n_scored += 1
        n_wet += 1 if wet else 0
        for key, cell in counts.items():
            value = row.get(key)
            if value is None:
                continue
            try:
                predicted = float(value) >= threshold_mm_h
            except (TypeError, ValueError):
                continue
            if predicted and wet:
                cell[0] += 1
            elif not predicted and wet:
                cell[1] += 1
            elif predicted and not wet:
                cell[2] += 1
            else:
                cell[3] += 1

    return {
        "n_rows": n_rows,
        "n_scored": n_scored,
        "gauge_wet_rate": (n_wet / n_scored) if n_scored else None,
        "threshold_mm_h": float(threshold_mm_h),
        "forecast_now": _skill(*counts["forecast_now_mm_h"]),
        "observed": _skill(*counts["observed_mm_h"]),
    }
