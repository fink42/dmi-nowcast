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

**A hit** is a warning with an onset in ``(sent, sent + lead + tolerance]``.
The tolerance exists because the warning promises "rain within LEAD
minutes" off a radar frame that is already 14–24 min old; a warning whose
rain lands 8 minutes late kept its promise in every sense the user cares
about. An onset is claimed by at most one warning and the earliest
warning claims it, so two warnings cannot both take credit for one rain
event, and an onset left unclaimed is a **miss**. A warning that claims
nothing is a **false alarm**.

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
    "DECISION_COLUMNS",
    "decision_schema",
    "slot_end_of",
    "gauge_slots",
    "onsets",
    "WarningOutcome",
    "OnsetOutcome",
    "ScoreResult",
    "score_warnings",
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


# ---------------------------------------------------------------------------
# The decision row — one shape, two writers
# ---------------------------------------------------------------------------

#: Columns of one evaluated (frame, station) decision. The historical
#: replay and the live ``station_eval`` step in the sidecar append rows of
#: exactly this shape, so a replay parquet and a live parquet concatenate
#: without a translation layer and this module can score either.
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


def decision_schema():
    """Arrow schema for :data:`DECISION_COLUMNS`.

    pyarrow is imported lazily: this module's scoring functions are pure
    Python and must stay importable in an environment that has no Arrow
    (the core package lists pyarrow as a dev dependency only).

    Every forecast field is nullable float32 — ``None`` from
    ``sample_point`` means "off coverage / nodata", which is emphatically
    not zero, and the null survives to the parquet so a reader cannot
    silently average it as a dry sample.
    """
    import pyarrow as pa

    return pa.schema([
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
    ])


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
# Scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WarningOutcome:
    """One replayed warning and what the gauge said about it."""

    sent_utc: datetime
    eta_min: float | None
    outcome: str  # "hit" | "false_alarm"
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
    outcome: str  # "hit" | "miss"
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


def score_warnings(
    warnings: Iterable[tuple[datetime, float | None]],
    onset_times: Sequence[datetime],
    *,
    lead_min: int = DEFAULT_LEAD_MIN,
    tolerance_min: int = DEFAULT_TOLERANCE_MIN,
    dry_min: int = DEFAULT_DRY_MIN,
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
    ``hits + false_alarms == len(warnings)`` and
    ``hits + misses == len(onsets)`` always hold, which is what makes POD
    and FAR readable side by side.

    A second warning cannot inherit an onset an earlier warning already
    took: two warnings for one rain event means one of them was noise, and
    counting it as a hit would hide exactly the spam this scoring exists
    to detect.

    ``dry_min`` takes no part in the matching — the onsets arrive already
    computed. It is carried into the summary so a stored result records
    the onset definition it was produced under.
    """
    window = timedelta(minutes=lead_min + tolerance_min)
    sent_list = sorted(
        ((_as_utc(sent, "sent_utc"), eta) for sent, eta in warnings),
        key=lambda pair: pair[0],
    )
    onset_list = sorted(_as_utc(o, "onset") for o in onset_times)
    claimed_by: list[datetime | None] = [None] * len(onset_list)
    claimed_error: list[float | None] = [None] * len(onset_list)

    warning_rows: list[WarningOutcome] = []
    lead_errors: list[float] = []
    hits = 0
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
            warning_rows.append(WarningOutcome(sent, eta, "false_alarm"))
            continue
        onset = onset_list[pick]
        error: float | None = None
        if eta is not None:
            # Predicted lead minus delivered lead: positive = the rain beat
            # the ETA, so the warning was late. See the module docstring.
            error = float(eta) - (onset - sent).total_seconds() / 60.0
            lead_errors.append(error)
        claimed_by[pick] = sent
        claimed_error[pick] = error
        hits += 1
        warning_rows.append(WarningOutcome(sent, eta, "hit", onset, error))

    onset_rows = tuple(
        OnsetOutcome(
            onset,
            "hit" if claimed_by[i] is not None else "miss",
            claimed_by[i],
            claimed_error[i],
        )
        for i, onset in enumerate(onset_list)
    )
    n_warnings = len(warning_rows)
    false_alarms = n_warnings - hits
    misses = len(onset_list) - hits
    summary = {
        "warnings": n_warnings,
        "hits": hits,
        "false_alarms": false_alarms,
        "misses": misses,
        "n_onsets": len(onset_list),
        "pod": (hits / len(onset_list)) if onset_list else None,
        "far": (false_alarms / n_warnings) if n_warnings else None,
        "lead_error_min": _quantiles(lead_errors),
        "lead_min": int(lead_min),
        "tolerance_min": int(tolerance_min),
        "dry_min": int(dry_min),
    }
    return ScoreResult(tuple(warning_rows), onset_rows, summary)


def pooled_summary(results: Iterable[ScoreResult], **params: Any) -> dict:
    """Totals over several stations' :class:`ScoreResult`.

    Counts add; POD, FAR and the lead-error quantiles are recomputed from
    the pooled populations rather than averaged, because a station with
    two warnings and a station with two hundred must not carry the same
    weight in a national number.
    """
    rows = list(results)
    warnings = [w for r in rows for w in r.warnings]
    onset_rows = [o for r in rows for o in r.onsets]
    hits = sum(1 for w in warnings if w.outcome == "hit")
    false_alarms = len(warnings) - hits
    misses = sum(1 for o in onset_rows if o.outcome == "miss")
    errors = [w.lead_error_min for w in warnings if w.lead_error_min is not None]
    out = {
        "warnings": len(warnings),
        "hits": hits,
        "false_alarms": false_alarms,
        "misses": misses,
        "n_onsets": len(onset_rows),
        "pod": (hits / len(onset_rows)) if onset_rows else None,
        "far": (false_alarms / len(warnings)) if warnings else None,
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
