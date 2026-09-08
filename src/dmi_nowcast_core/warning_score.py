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
(six slots at the default 60) of *known dry* slots, and it counts only
where rain actually arrived: the onset slot's depth plus the FOLLOWING
slot's must reach ``onset_min_mm``, 0.2 mm by default. An unknown slot
cannot certify dryness, so it resets the dry run rather than extending
it: no onset is ever declared on the strength of missing data. A
candidate that fails the amount test is dropped as an onset but still
resets the dry run — it rained, so the slot after it is not the start of
a new event either.

The amount is summed over two slots because DMI's 10-minute bins cut a
shower in half as often as not: 0.1 mm and then 0.4 mm is one event of
0.5 mm rather than two drizzles. A depth the station never reported and
DMI's trace sentinel alike contribute zero millimetres, so a
duration-only slot can be wet and still fail the amount test.
``onset_min_mm=0.0`` restores the rule that shipped before 2026-09-07,
where the wet flag alone made an onset.

The per-slot WET rule is untouched by this, deliberately: "is it raining
in this slot" — what certifies a dry spell, and what
:func:`raining_now_agreement` scores — is a different question from "did
a rain event start here".

The onset timestamp is the slot END, which is the only instant the gauge
actually reports; the true first drop fell somewhere in the preceding
10 minutes, so the measured gap between a warning and its onset is
overstated by up to 10 minutes and every lead error here is biased that
far NEGATIVE (see the sign convention below — negative reads as "the
warning was early"). Stated once, here, rather than hidden in a
correction factor.

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
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "SLOT_MIN",
    "WET_PRECIP_MM",
    "WET_DUR_MIN",
    "PRECIP_PARAM",
    "PRECIP_DUR_PARAM",
    "DEFAULT_DRY_MIN",
    "DEFAULT_ONSET_MIN_MM",
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
    "gauge_slot_amounts",
    "onsets",
    "DEFAULT_GAUGE_PAD_MIN",
    "StationSlots",
    "GaugeTruth",
    "gauge_truth_vectorised",
    "DEFAULT_MIN_KNOWN_SLOTS",
    "DeadGauge",
    "dead_gauge_scan",
    "dead_gauges",
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
#: an onset (six slots at the 10-min cadence). Matched to the push rule's
#: own 60-minute re-arm: a "new" rain event the rule could never have
#: warned about a second time is not evidence about the forecast.
DEFAULT_DRY_MIN = 60

#: Millimetres an onset must deliver over the onset slot AND the one
#: after it. Below this the gauge saw drizzle, and a scoreboard that
#: counts drizzle as a rain event measures the gauge's sensitivity rather
#: than the service's skill. ``0.0`` is the pre-2026-09-07 rule: the wet
#: flag alone, whatever it weighed.
DEFAULT_ONSET_MIN_MM = 0.2
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


def gauge_slot_amounts(
    table: Any,
    station_id: str,
    *,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
    slot_min: int = SLOT_MIN,
) -> list[tuple[datetime, bool | None, float | None]]:
    """One station's slots as ``(slot_end, wet, mm)`` over a contiguous grid.

    ``table`` is what ``StationObsStore.read`` returns (columns
    ``station_id`` / ``observed_utc`` / ``parameter_id`` / ``value``), or
    any sequence of mappings with those keys.

    The result is a **contiguous** grid — every slot between the first and
    last covered instant appears exactly once, in order — because the
    onset rule counts consecutive dry slots and cannot do that over a list
    with silent holes. A slot the station did not report is unknown.

    ``wet`` is ``True`` when either arm of the wet rule fires, ``False``
    when at least one arm reported and neither fired, and ``None`` when
    the station reported neither parameter for that slot. A station that
    reports only the duration parameter is therefore still scoreable —
    unknown means *nothing was said*, not *one thing was missing*.

    ``mm`` is the largest ``precip_past10min`` the station reported in the
    slot, with DMI's trace sentinel folded to 0.0, and ``None`` where it
    reported no amount at all. The onset rule reads the two the same way
    (neither contributes millimetres), but they are different statements —
    "the gauge weighed nothing" against "the gauge said nothing" — and
    only this grid can tell them apart.

    ``start_utc`` / ``end_utc`` pin the grid (both inclusive, snapped to
    slot ends); by default it spans the station's own rows. Pin them when
    several stations must share one grid, or when a caller wants the pad
    around a day.
    """
    if slot_min <= 0:
        raise ValueError("slot_min must be positive")
    wanted = str(station_id)
    #: slot end → the largest usable reading of each parameter in it
    amounts: dict[datetime, float] = {}
    durations: dict[datetime, float] = {}
    for row in _rows_of(table):
        if str(row.get("station_id")) != wanted:
            continue
        observed = row.get("observed_utc")
        if observed is None:
            continue
        param = str(row.get("parameter_id"))
        if param not in (PRECIP_PARAM, PRECIP_DUR_PARAM):
            continue  # some other parameter rode along in the read
        value = _amount_mm(row.get("value"))
        if value is None:
            # Reported but unusable (null / NaN) — says nothing either way,
            # so it must not turn an unknown slot into a dry one.
            continue
        slot = slot_end_of(_as_utc(observed, "observed_utc"), slot_min=slot_min)
        target = amounts if param == PRECIP_PARAM else durations
        previous = target.get(slot)
        if previous is None or value > previous:
            target[slot] = value

    reported = amounts.keys() | durations.keys()
    if start_utc is not None:
        first = slot_end_of(start_utc, slot_min=slot_min)
    elif reported:
        first = min(reported)
    else:
        return []
    if end_utc is not None:
        last = slot_end_of(end_utc, slot_min=slot_min)
    elif reported:
        last = max(reported)
    else:
        return []
    if last < first:
        return []

    step = timedelta(minutes=slot_min)
    out: list[tuple[datetime, bool | None, float | None]] = []
    cursor = first
    while cursor <= last:
        mm = amounts.get(cursor)
        dur = durations.get(cursor)
        if mm is None and dur is None:
            wet: bool | None = None
        else:
            wet = (mm is not None and mm >= WET_PRECIP_MM) or (
                dur is not None and dur >= WET_DUR_MIN
            )
        out.append((cursor, wet, mm))
        cursor += step
    return out


def gauge_slots(
    table: Any,
    station_id: str,
    *,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
    slot_min: int = SLOT_MIN,
) -> list[tuple[datetime, bool | None]]:
    """:func:`gauge_slot_amounts` without the depths: ``(slot_end, wet)``.

    The shape every consumer of the wet flag alone wants — the dry-run
    detection, ``raining_now_agreement``, the "was it raining here" lookup.
    A series in this shape cannot answer the onset rule's amount test, and
    :func:`onsets` says so rather than reading a missing depth as zero.
    """
    return [
        (slot, wet)
        for slot, wet, _mm in gauge_slot_amounts(
            table, station_id,
            start_utc=start_utc, end_utc=end_utc, slot_min=slot_min,
        )
    ]


def _depth(mm: Any) -> float:
    """Millimetres a slot contributes to an onset amount.

    A depth the station never reported contributes nothing — a
    duration-only slot measured no depth — and so does DMI's trace
    sentinel, which is "below 0.1 mm" and never a negative amount: summing
    it would let a trace *subtract* from the shower it belongs to.
    """
    if mm is None:
        return 0.0
    try:
        value = float(mm)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value) or value < 0.0:
        return 0.0
    return value


def _two_slot_mm(
    rows: Sequence[Sequence[Any]], index: int, step: timedelta,
) -> float:
    """The onset slot's depth plus the following slot's.

    The neighbour counts only when it really is the next slot: a series
    that ends here, or one with a seam after it, contributes the onset
    slot alone rather than reaching across a gap for millimetres that were
    measured somewhere else entirely.
    """
    total = _depth(rows[index][2])
    if index + 1 >= len(rows):
        return total
    here = _as_utc(rows[index][0], "slot end")
    following = rows[index + 1]
    if _as_utc(following[0], "slot end") - here == step:
        total += _depth(following[2])
    return total


def onsets(
    slots: Sequence[Sequence[Any]],
    dry_min: int = DEFAULT_DRY_MIN,
    *,
    onset_min_mm: float | None = DEFAULT_ONSET_MIN_MM,
    slot_min: int = SLOT_MIN,
) -> list[datetime]:
    """Onset instants: a wet slot after ``dry_min`` dry, that delivered rain.

    ``ceil(dry_min / slot_min)`` consecutive dry slots are required — six
    at the defaults, and *exactly* six is enough. An unknown slot resets
    the run (missing data never certifies a dry spell), and so does a gap
    in the grid, so a list stitched from two separate days cannot invent
    an onset across the seam.

    ``onset_min_mm`` is then the amount test: the onset slot's depth plus
    the following slot's must reach it, an unreported depth and a trace
    sentinel alike counting as zero millimetres. A candidate that fails is
    dropped but still resets the dry run — it rained, so the slot after it
    is not the start of a new event either. ``0.0`` (or ``None``) is the
    older rule, where the wet flag alone made an onset.

    ``slots`` is :func:`gauge_slot_amounts`' ``(slot_end, wet, mm)``. The
    two-element shape :func:`gauge_slots` returns carries no depths, so it
    is accepted only with the amount test off: reading a missing depth as
    zero would silently drop every onset instead.

    The first slots of a record can never be onsets: there is no evidence
    about what came before them. That is deliberate — it costs the odd
    genuine event at a window edge and buys the guarantee that every onset
    reported here is one the gauge actually witnessed starting.
    """
    rows = [tuple(row) for row in slots]
    floor = 0.0 if onset_min_mm is None else float(onset_min_mm)
    if floor > 0.0 and any(len(row) < 3 for row in rows):
        raise ValueError(
            "onset_min_mm needs a slot series carrying millimetres: build it "
            "with gauge_slot_amounts, or pass onset_min_mm=0.0",
        )
    need = max(1, math.ceil(dry_min / slot_min))
    step = timedelta(minutes=slot_min)
    out: list[datetime] = []
    dry_run = 0
    previous: datetime | None = None
    for index, row in enumerate(rows):
        ts = _as_utc(row[0], "slot end")
        wet = row[1]
        if previous is not None and ts - previous != step:
            dry_run = 0  # a hole in the grid is not a dry spell
        previous = ts
        if wet is None:
            dry_run = 0
            continue
        if wet:
            if dry_run >= need and (
                floor <= 0.0 or _two_slot_mm(rows, index, step) >= floor
            ):
                out.append(ts)
            dry_run = 0
        else:
            dry_run += 1
    return out


# ---------------------------------------------------------------------------
# The same rules, vectorised: a month of parquet straight into numpy
# ---------------------------------------------------------------------------
#
# :func:`gauge_slots` and :func:`onsets` above are the reference. They are
# row-at-a-time Python, which is right for a hand-written test series and
# ruinous for the real archive: ten months of ~104 stations is ~11 million
# observation rows, and a caller that rescans a month's table once per
# station — as all three of this module's consumers did — spends about
# half an hour and several gigabytes producing a few thousand onsets.
#
# What follows is the same three rules over numpy arrays instead:
#
# * a slot's value is the LARGEST reading either parameter reported in it,
#   with DMI's negative "trace" sentinel folded to 0.0 first (identical to
#   the ``or``-of-fired-arms in :func:`gauge_slots`, since folding is
#   monotone and both arms threshold a maximum);
# * a slot nobody reported is UNKNOWN, and unknown is never dry;
# * an onset is a wet slot preceded by ``ceil(dry_min / slot_min)``
#   consecutive KNOWN DRY slots, counted by boolean run length rather
#   than by a running counter, and delivering ``onset_min_mm`` over
#   itself and the slot after it.
#
# The grid is built ONCE over the whole window rather than per month, and
# that is not merely faster, it is exactly what the month-by-month callers
# were approximating: reading month M padded either side and unioning the
# onsets found in the overlaps recovers precisely the onsets a full-window
# grid finds, because truncating a window can only SHORTEN a dry run (so a
# padded read never invents an onset), and every instant is interior to
# some padded month (so no onset is lost). The union was the workaround
# for not being able to hold the full grid; a dense array of ~44 000 slots
# × ~104 stations is 40 MB, so the workaround is no longer needed.
#
# numpy and pyarrow are imported inside the functions: this module's
# scoring half must stay importable with neither, and does.

#: Minutes of context read either side of a month partition. The onset
#: rule needs the dry slots in FRONT of an event and a warning at the end
#: of a window needs the slots behind it, so a month is never read alone.
#: Two hours covers the longest dry spell any caller asks for (120 min).
DEFAULT_GAUGE_PAD_MIN = 120


def _months_between(start: datetime, end: datetime) -> list[tuple[int, int]]:
    """Every ``(year, month)`` the inclusive window touches."""
    out: list[tuple[int, int]] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        out.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def _month_window(
    year: int, month: int, pad: timedelta,
) -> tuple[datetime, datetime]:
    """One month's padded read window, as every caller builds it."""
    start = datetime(year, month, 1, tzinfo=timezone.utc) - pad
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) + pad
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc) + pad
    return start, end


@dataclass(frozen=True)
class StationSlots:
    """One station's contiguous slot grid, held as numpy arrays.

    The array form of what :func:`gauge_slots` returns as a list of
    tuples, with the amounts kept rather than collapsed to a boolean —
    the variant analyses threshold them at more than one level.

    * ``slot_end`` — int64 epoch SECONDS, ascending, exactly one slot
      apart. Integers, not datetimes: the whole point is that a run-length
      pass over 44 000 slots costs microseconds.
    * ``mm`` / ``dur`` — float32, ``NaN`` where that parameter reported
      nothing for the slot. Negative readings (DMI's "trace of
      precipitation" sentinel) are already folded to 0.0.
    * ``known`` — either parameter reported something usable.
    * ``wet`` — the shipped rule, ``False`` wherever the slot is unknown.
      Read it beside ``known``: ``wet=False, known=False`` is *unknown*,
      which is emphatically not dry.
    """

    station_id: str
    slot_end: Any
    mm: Any
    dur: Any
    known: Any
    wet: Any
    slot_min: int = SLOT_MIN

    def __len__(self) -> int:
        return int(self.slot_end.size)

    def slots(self) -> list[tuple[datetime, bool | None]]:
        """The reference shape: ``[(slot_end, wet|None)]``.

        For tests and for small windows. Materialising this for a hundred
        stations over a year is exactly the cost this class exists to
        avoid, so nothing on the hot path calls it.
        """
        return [
            (
                datetime.fromtimestamp(int(sec), timezone.utc),
                bool(wet) if known else None,
            )
            for sec, wet, known in zip(self.slot_end, self.wet, self.known)
        ]

    def _index(self, when: datetime) -> int | None:
        """Grid position of the slot containing ``when``, if it has one."""
        if self.slot_end.size == 0:
            return None
        stamp = slot_end_of(when, slot_min=self.slot_min)
        step = self.slot_min * 60
        offset = int(stamp.timestamp()) - int(self.slot_end[0])
        if offset < 0 or offset % step:
            return None
        index = offset // step
        return int(index) if index < self.slot_end.size else None

    def wet_at(self, when: datetime) -> bool | None:
        """``True`` / ``False`` / ``None`` for the slot containing ``when``."""
        index = self._index(when)
        if index is None or not self.known[index]:
            return None
        return bool(self.wet[index])

    def known_until(self) -> datetime | None:
        """The last slot this station actually reported, or ``None``."""
        import numpy as np

        found = np.flatnonzero(self.known)
        if found.size == 0:
            return None
        return datetime.fromtimestamp(
            int(self.slot_end[found[-1]]), timezone.utc,
        )

    def known_between(self, first: datetime, last: datetime) -> int:
        """Known slots in ``[first, last]`` — both snapped to slot ends."""
        import numpy as np

        step = self.slot_min * 60
        base = int(self.slot_end[0]) if self.slot_end.size else 0
        lo = (int(slot_end_of(first, slot_min=self.slot_min).timestamp()) - base) // step
        hi = (int(slot_end_of(last, slot_min=self.slot_min).timestamp()) - base) // step
        lo = max(0, lo)
        hi = min(self.slot_end.size - 1, hi)
        if hi < lo:
            return 0
        return int(np.count_nonzero(self.known[lo:hi + 1]))

    def onsets_with_amounts(
        self,
        dry_min: int = DEFAULT_DRY_MIN,
        *,
        onset_min_mm: float | None = DEFAULT_ONSET_MIN_MM,
    ) -> list[tuple[datetime, float]]:
        """``[(onset, mm over the onset slot and the next)]``.

        :func:`onsets`' rule, by boolean run length. ``dry_run`` at slot
        *i* is the number of consecutive known-dry slots immediately
        before it — reset by a wet slot and by an unknown one alike — so
        the counter the reference carries is the run length ending at
        *i − 1*, and that is a maximum-accumulate away.

        ``onset_min_mm`` is the amount test: the onset slot's depth plus
        the following slot's must reach it, with an unreported depth and a
        trace sentinel alike contributing zero. A candidate that fails
        still resets the dry run, which happens for free — every wet slot
        does, tested or not. ``0.0`` (or ``None``) is the older rule,
        where the wet flag alone made an onset.
        """
        import numpy as np

        n = self.slot_end.size
        if n == 0:
            return []
        need = max(1, math.ceil(dry_min / self.slot_min))
        dry = self.known & ~self.wet
        # Run length of consecutive dry slots ENDING at each position:
        # the last non-dry position at or before i marks where the run
        # started, and max-accumulate finds it in one pass.
        position = np.arange(n, dtype=np.int64)
        started = np.maximum.accumulate(np.where(dry, 0, position + 1))
        run = np.where(dry, position - started + 1, 0)
        before = np.zeros(n, dtype=np.int64)
        before[1:] = run[:-1]
        candidate = self.wet & (before >= need)

        depth = np.nan_to_num(self.mm, nan=0.0, posinf=0.0, neginf=0.0)
        two = depth.astype(np.float64)
        two[:-1] += depth[1:]
        floor = 0.0 if onset_min_mm is None else float(onset_min_mm)
        if floor > 0.0:
            candidate = candidate & (two >= floor)
        found = np.flatnonzero(candidate)
        return [
            (
                datetime.fromtimestamp(int(self.slot_end[i]), timezone.utc),
                float(two[i]),
            )
            for i in found
        ]

    def onsets(
        self,
        dry_min: int = DEFAULT_DRY_MIN,
        *,
        onset_min_mm: float | None = DEFAULT_ONSET_MIN_MM,
    ) -> list[datetime]:
        """The onset instants alone — :func:`onsets` over this grid."""
        return [
            instant for instant, _mm
            in self.onsets_with_amounts(dry_min, onset_min_mm=onset_min_mm)
        ]


@dataclass(frozen=True)
class GaugeTruth:
    """What a scoring run needs from the gauge archive, and nothing else.

    ``onsets`` and ``known_until`` are the two things every consumer
    reads; ``series`` is the grid they were derived from, kept so a second
    onset definition costs a run-length pass rather than a second read of
    the archive, and so ``wet_at`` can answer "was it raining at this
    station at this instant" for the ``raining_now`` comparison.
    """

    series: dict[str, StationSlots] = field(default_factory=dict)
    onsets: dict[str, list[datetime]] = field(default_factory=dict)
    known_until: dict[str, datetime] = field(default_factory=dict)
    #: Known slots summed over the read windows, month by month. The pad
    #: makes consecutive windows overlap, so a slot in a pad sliver is
    #: counted twice — deliberately, because this is the "did the archive
    #: say anything at all" number the month-by-month readers reported and
    #: it is quoted in stored reports.
    known_slots: int = 0
    slot_min: int = SLOT_MIN

    def wet_at(self, station_id: str, when: datetime) -> bool | None:
        """The wet flag at one station's slot, or ``None`` if unknown."""
        series = self.series.get(str(station_id))
        return None if series is None else series.wet_at(when)

    def onsets_for(
        self,
        dry_min: int = DEFAULT_DRY_MIN,
        *,
        onset_min_mm: float | None = DEFAULT_ONSET_MIN_MM,
    ) -> dict[str, list[tuple[datetime, float]]]:
        """Every station's onsets under one alternative definition.

        The archive is read once; a variant is a run-length pass over the
        grid already in memory.
        """
        return {
            station: series.onsets_with_amounts(
                dry_min, onset_min_mm=onset_min_mm,
            )
            for station, series in self.series.items()
        }


def _scatter_max(flat_grid, index, values) -> None:
    """``grid[i] = max(grid[i], v)`` for every ``(i, v)``, vectorised.

    ``np.maximum.at`` does this directly and is an order of magnitude
    slower than sorting: the pairs are ordered by (slot, value) so the
    LAST pair of each slot carries its maximum, and one masked assignment
    settles the lot. Duplicates are rare in practice — a station reports a
    parameter once per slot — but two readings inside one 10-minute bin,
    or a slot straddling two month partitions, both land here.
    """
    import numpy as np

    if index.size == 0:
        return
    order = np.lexsort((values, index))
    ordered_index = index[order]
    ordered_values = values[order]
    last = np.empty(ordered_index.size, dtype=bool)
    last[-1] = True
    np.not_equal(ordered_index[1:], ordered_index[:-1], out=last[:-1])
    target = ordered_index[last]
    flat_grid[target] = np.maximum(flat_grid[target], ordered_values[last])


def _absorb_batch(
    batch,
    station_index,
    mm_flat,
    dur_flat,
    *,
    first_sec: int,
    step_sec: int,
    step_us: int,
    n_slots: int,
) -> int:
    """Fold one record batch of observations into the two value grids.

    Returns the number of readings it used. Everything here is a whole-
    column operation: the station id becomes a row index by hash lookup,
    the instant becomes a slot index by integer ceiling division, and the
    readings are scattered into the grid by maximum.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    if batch.num_rows == 0:
        return 0
    row_station = pc.index_in(
        batch.column("station_id"), value_set=station_index,
    )
    value = batch.column("value")
    keep = pc.and_(pc.is_valid(row_station), pc.is_valid(value))
    keep = pc.and_(keep, pc.is_valid(batch.column("observed_utc")))
    keep = pc.and_(keep, pc.is_valid(batch.column("parameter_id")))
    keep = pc.and_(keep, pc.fill_null(pc.is_finite(value), False))
    row_station = row_station.filter(keep)
    batch = batch.filter(keep)
    if batch.num_rows == 0:
        return 0

    station_of = np.asarray(row_station, dtype=np.int64)
    micros = np.asarray(
        batch.column("observed_utc").cast(pa.int64()), dtype=np.int64,
    )
    # DMI's trace sentinel: a negative amount is "below 0.1 mm", never a
    # negative depth (``_amount_mm``). The fold makes a new array, which
    # is what lets the rest of this work in place — Arrow's own buffers
    # come back read-only.
    readings = np.maximum(
        np.asarray(batch.column("value"), dtype=np.float32), np.float32(0.0),
    )
    is_amount = np.asarray(
        pc.fill_null(pc.equal(batch.column("parameter_id"), PRECIP_PARAM), False),
        dtype=bool,
    )

    # ``slot_end_of`` is a ceiling to the next slot boundary, with an
    # instant already ON a boundary staying put — which is what integer
    # ceiling division does, on microseconds so a sub-second stamp rounds
    # up exactly as the reference's does.
    slot = -(-micros // step_us) * step_us // 1_000_000
    position = (slot - first_sec) // step_sec
    inside = (position >= 0) & (position < n_slots)
    flat = station_of * n_slots + position
    amount = inside & is_amount
    duration = inside & ~is_amount
    _scatter_max(mm_flat, flat[amount], readings[amount])
    _scatter_max(dur_flat, flat[duration], readings[duration])
    return int(batch.num_rows)


def gauge_truth_vectorised(
    store_root: Any,
    start: datetime,
    end: datetime,
    station_ids: Sequence[str] | None = None,
    *,
    wet_mm: float = WET_PRECIP_MM,
    wet_dur_min: float = WET_DUR_MIN,
    dry_min: int = DEFAULT_DRY_MIN,
    onset_min_mm: float | None = DEFAULT_ONSET_MIN_MM,
    slot_min: int = SLOT_MIN,
    pad_min: int = DEFAULT_GAUGE_PAD_MIN,
    log=None,
) -> GaugeTruth:
    """The whole gauge truth for a window, in one vectorised pass.

    ``store_root`` is the corpus root a
    :class:`~dmi_nowcast_core.station_store.StationObsStore` takes.
    ``start`` / ``end`` name the MONTHS to read, exactly as the callers'
    ``_months_between(start, end)`` did; each month partition is read with
    ``pad_min`` minutes of context either side, and the grid spans the
    union of those windows.

    ``station_ids`` restricts the read (and fixes the row order of the
    grid); ``None`` reads every station the partitions hold. Every
    requested station gets an entry in ``onsets`` — an empty list where
    the archive says nothing — while ``known_until`` carries only the
    stations that actually reported, which is the test the callers use to
    decide whether a station can verify anything at all.

    ``dry_min`` / ``onset_min_mm`` are the onset definition, exactly as
    :func:`onsets` reads them.

    Identical in result to ``gauge_slot_amounts`` + ``onsets`` per station
    per month, and about a thousand times faster; the equality is asserted
    against the reference implementation in the tests, on hand-made series
    and on a real month of the archive.
    """
    import numpy as np
    import pyarrow as pa

    from .station_store import StationObsStore

    if slot_min <= 0:
        raise ValueError("slot_min must be positive")
    store = StationObsStore(store_root)
    pad = timedelta(minutes=int(pad_min))
    months = _months_between(_as_utc(start, "start"), _as_utc(end, "end"))
    windows = [_month_window(year, month, pad) for year, month in months]
    grid_first = slot_end_of(windows[0][0], slot_min=slot_min)
    grid_last = slot_end_of(windows[-1][1], slot_min=slot_min)

    step_sec = slot_min * 60
    first_sec = int(grid_first.timestamp())
    n_slots = (int(grid_last.timestamp()) - first_sec) // step_sec + 1
    if n_slots <= 0:
        return GaugeTruth(slot_min=slot_min)

    stations = (
        None if station_ids is None else [str(s) for s in dict.fromkeys(station_ids)]
    )
    if stations is None:
        stations = _stations_in(store, months)
    if not stations:
        return GaugeTruth(slot_min=slot_min)
    station_index = pa.array(stations, type=pa.string())

    # -1.0 is the "nothing reported" sentinel while the grid is filled: a
    # folded reading is never negative, so it is a proper identity for the
    # running maximum. It becomes NaN once the fill is done.
    shape = (len(stations), int(n_slots))
    mm = np.full(shape, -1.0, dtype=np.float32)
    dur = np.full(shape, -1.0, dtype=np.float32)
    mm_flat = mm.reshape(-1)
    dur_flat = dur.reshape(-1)
    step_us = step_sec * 1_000_000

    # The partitions the GRID touches, which is the months asked for plus
    # whatever sliver of their neighbours the pad reaches into. Each is
    # streamed a batch at a time and folded straight into the grid, so
    # the resident cost of a month is one batch and not a month.
    pool = pa.default_memory_pool()
    for year, month in _months_between(grid_first, grid_last):
        rows = 0
        try:
            for batch in store.stream_month(
                year, month, [PRECIP_PARAM, PRECIP_DUR_PARAM], stations,
            ):
                rows += _absorb_batch(
                    batch, station_index, mm_flat, dur_flat,
                    first_sec=first_sec, step_sec=step_sec,
                    step_us=step_us, n_slots=n_slots,
                )
        except Exception as exc:  # noqa: BLE001 — one unreadable month
            if log:
                log(f"gauge read failed for {year}-{month:02d}: {exc}")
            continue
        finally:
            # Arrow's allocator holds freed pages by default, and ten
            # months of that is the difference between 200 MB and 600.
            pool.release_unused()
        if log:
            log(f"gauge month {year}-{month:02d}: {rows} reading(s)")

    mm[mm < 0.0] = np.nan
    dur[dur < 0.0] = np.nan
    slot_ends = first_sec + np.arange(n_slots, dtype=np.int64) * step_sec
    with np.errstate(invalid="ignore"):
        known = ~(np.isnan(mm) & np.isnan(dur))
        wet = (mm >= float(wet_mm)) | (dur >= float(wet_dur_min))

    truth_series: dict[str, StationSlots] = {}
    onsets_by_station: dict[str, list[datetime]] = {}
    known_until: dict[str, datetime] = {}
    for row, station in enumerate(stations):
        series = StationSlots(
            station_id=station,
            slot_end=slot_ends,
            mm=mm[row],
            dur=dur[row],
            known=known[row],
            wet=wet[row],
            slot_min=slot_min,
        )
        truth_series[station] = series
        onsets_by_station[station] = series.onsets(
            dry_min, onset_min_mm=onset_min_mm,
        )
        last = series.known_until()
        if last is not None:
            known_until[station] = last

    known_slots = 0
    for (window_start, window_end) in windows:
        for series in truth_series.values():
            known_slots += series.known_between(window_start, window_end)

    return GaugeTruth(
        series=truth_series,
        onsets=onsets_by_station,
        known_until=known_until,
        known_slots=known_slots,
        slot_min=slot_min,
    )


def _stations_in(store: Any, months: Sequence[tuple[int, int]]) -> list[str]:
    """Every station id the requested partitions carry, sorted.

    Only for ``station_ids=None`` — a caller asking about the whole
    archive. The three consumers all name their stations, because the
    decision rows decide which stations there is anything to score at.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    found: set[str] = set()
    for year, month in months:
        path = store.partition_path(int(year), int(month))
        if not path.exists():
            continue
        column = pq.read_table(path, columns=["station_id"]).column("station_id")
        found.update(pc.unique(column.combine_chunks()).to_pylist())
    return sorted(str(s) for s in found if s is not None)


# ---------------------------------------------------------------------------
# Dead gauges: a bucket that never moves is not a measurement
# ---------------------------------------------------------------------------

#: Known slots a station must contribute before "never wet" is evidence
#: about the GAUGE rather than about the window. 500 ten-minute slots is
#: about 3.5 days of continuous reporting — long enough that a working
#: Danish gauge has almost certainly seen rain, short enough that the rule
#: still fires on a month-long window.
DEFAULT_MIN_KNOWN_SLOTS = 500


@dataclass(frozen=True)
class DeadGauge:
    """One excluded station and the counts that convicted it."""

    station_id: str
    #: Slots the station reported something in, over the evaluated window
    #: (restricted to ``on_days`` when the caller passed one).
    known_slots: int
    #: Wet slots in the same population. Zero, by construction — the field
    #: exists so a report can state the evidence rather than assert it.
    wet_slots: int


def _epoch_day(value: date | datetime) -> int:
    """Days since 1970-01-01 for a date, or for a UTC instant's date."""
    if isinstance(value, datetime):
        value = _as_utc(value, "on_days entry").date()
    if not isinstance(value, date):
        raise TypeError(
            f"on_days entries must be dates, got {type(value).__name__}"
        )
    return value.toordinal() - date(1970, 1, 1).toordinal()


def dead_gauge_scan(
    truth: Any,
    *,
    min_known_slots: int = DEFAULT_MIN_KNOWN_SLOTS,
    on_days: Iterable[date | datetime] | None = None,
) -> list[DeadGauge]:
    """The stations the dead-gauge rule excludes, with their counts.

    :func:`dead_gauges` is the rule; this is the evidence behind it, kept
    separate so a report can print "06080: 3 876 known slots, never wet"
    rather than a bare id.

    ``truth`` is a :class:`GaugeTruth` or any mapping of station id to
    :class:`StationSlots`. Sorted by station id; empty when nothing
    qualifies, and empty when ``min_known_slots <= 0``, which switches the
    rule off.
    """
    import numpy as np

    series: Mapping[str, StationSlots] = getattr(truth, "series", truth)
    floor = int(min_known_slots)
    day_set = None
    if on_days is not None:
        wanted = sorted({_epoch_day(day) for day in on_days})
        day_set = np.array(wanted, dtype=np.int64)

    rows: list[DeadGauge] = []
    any_wet = False
    for station, slots in series.items():
        known = slots.known
        # ``wet`` is already False wherever the slot is unknown, but the
        # conjunction is free and makes the invariant local.
        wet = slots.wet & known
        if day_set is not None:
            on_day = np.isin(slots.slot_end // 86_400, day_set)
            known = known & on_day
            wet = wet & on_day
        n_wet = int(np.count_nonzero(wet))
        any_wet = any_wet or n_wet > 0
        rows.append(DeadGauge(str(station), int(np.count_nonzero(known)), n_wet))

    if floor <= 0 or not any_wet:
        # No station saw rain: the WINDOW was dry, and excluding every
        # gauge in it would be the rule diagnosing the weather.
        return []
    return sorted(
        (row for row in rows if row.wet_slots == 0 and row.known_slots >= floor),
        key=lambda row: row.station_id,
    )


def dead_gauges(
    truth: Any,
    *,
    min_known_slots: int = DEFAULT_MIN_KNOWN_SLOTS,
    on_days: Iterable[date | datetime] | None = None,
    log: Any = None,
) -> list[str]:
    """Stations whose gauge reported plenty and never once reported rain.

    A gauge that is wired up but broken is worse for scoring than one that
    is missing. A missing station has no truth, so every consumer already
    leaves it out of the pool (``known_until`` carries only the stations
    that reported). A gauge stuck at zero looks like a station that is
    working and permanently dry, so every radar-wet slot there becomes a
    false alarm and every warning a wrong one.

    Found in the 2026-09-08 product study
    (``archive/product_study_20260908/``): station 06080 reported
    ``precip_past10min = 0`` and ``precip_dur_past10min = 0`` in all 3 876
    known slots of the wettest days of the year, including days when its
    neighbours measured double figures, and contributed 553 false alarms
    to that study alone. The rule is written as a general property —
    *known often, wet never* — rather than as an id, because the next
    gauge to die will have a different number and nobody will notice.

    ``min_known_slots`` is the evidence floor
    (:data:`DEFAULT_MIN_KNOWN_SLOTS`): below it "never wet" says more
    about the window than about the station, and a genuinely dry fortnight
    must not cost a working gauge. ``0`` switches the rule off. As a
    second guard, a window in which NO station was ever wet excludes
    nothing at all.

    ``on_days`` restricts the counting to a set of UTC dates — the study's
    "on the wettest days" reading, which is the strictest form of the
    test: a gauge that stayed dry while the country was wet is dead beyond
    argument. A slot belongs to the date its END falls on. ``None``, the
    default, uses the whole evaluated window.

    ``log`` is an optional ``log(message)`` callable; every exclusion is
    reported through it with the count behind it, because a station
    silently dropped from a scoreboard is a station nobody can audit.
    """
    found = dead_gauge_scan(
        truth, min_known_slots=min_known_slots, on_days=on_days,
    )
    if log:
        for row in found:
            log(
                f"dead gauge {row.station_id}: {row.known_slots} known slot(s), "
                "never wet — excluded from scoring"
            )
    return [row.station_id for row in found]


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
    onset_min_mm: float | None = DEFAULT_ONSET_MIN_MM,
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

    ``dry_min`` and ``onset_min_mm`` take no part in the matching — the
    onsets arrive already computed. They are carried into the summary so a
    stored result records the onset definition it was produced under.
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
        "onset_min_mm": 0.0 if onset_min_mm is None else float(onset_min_mm),
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
    slots_by_station: Mapping[str, Sequence[Sequence[Any]]] | None = None,
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
        # ``(slot_end, wet)`` or ``(slot_end, wet, mm)``: the depth is the
        # onset rule's business, and this only ever asks the wet flag.
        lookup[str(station)] = {
            _as_utc(row[0], "slot end"): row[1] for row in slots
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
