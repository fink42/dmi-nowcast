"""Sweep the push rule's (lead, threshold) grid against the rain gauges.

The site sends one browser notification per subscription when the
calibrated probability of rain within LEAD minutes crosses THRESHOLD, and
today LEAD is 30 and THRESHOLD is 40 %. Those two numbers were chosen
before there was any measurement to choose them with. This script answers
the question the gauges can now answer: *for each horizon, which threshold
would have been the best one to ship?*

Why it needs no STEPS run
-------------------------
Every decision row written by ``scripts/replay_warnings.py`` (history) and
by the sidecar's ``station_eval`` step (live) carries ``p_rain_<lead>`` for
**every served lead**, beside the ``eta_min`` / ``observed_mm_h`` fields
the rule also reads. And ``push.engine.evaluate`` is a pure function of
(state, observation, rules). So an arbitrary (lead, threshold) rule can be
replayed over the stored rows exactly as the live service would have run
it — same state machine, same arming, same "already raining" silence — for
the cost of a dict lookup per row. The expensive half (radar, flow, the
ensemble, the isotonic curves) is already on disk.

What is replayed, per (lead, threshold)
---------------------------------------
Per station, rows in ``radar_ts`` order through ``evaluate`` with
``Rules(persistence_obs, rearm_after_min, raining_now_mm_h)``, carrying
``SubState`` from row to row. The state is reset to ``INITIAL_STATE`` — a
fresh, armed subscription — at the start of every **coverage run**: a gap
of more than ``--coverage-gap-min`` (20, two radar cycles) means the
service was not watching, and carrying an arm across an outage would
invent a decision nobody could have taken. Those are the same runs
``warning_score.coverage_runs`` uses to decide which onsets were ever
catchable. A row whose ``p_rain_<lead>`` is null is skipped for that lead
alone: it is off coverage or an unserved lead, which is not "dry", so it
neither warns nor advances the machine.

Quiet hours are off throughout. A virtual subscriber at a rain gauge has
no time zone and no bedtime, and a suppressed notification would be scored
as silence the rule never intended.

Every ``notify`` becomes a warning ``(generated_at, eta_min)`` and is
scored against that station's gauge onsets by
``warning_score.score_warnings`` with ``known_until=`` and ``coverage=``,
exactly as ``replay_warnings.score`` does, then pooled across stations
with ``pooled_summary``. Stations the gauge store has nothing for are left
out of the pool entirely: without a measurement every warning there would
score as a false alarm, which measures the archive rather than the rule.

The objective: F1, with a minimum useful lead
---------------------------------------------
The subscriber chooses a horizon. The threshold that horizon warns at is
fitted here, and "best" is **F1** — the harmonic mean of

* precision = ``hits / (hits + false alarms)``: of the notifications sent,
  how many were followed by rain;
* recall = ``hits / (hits + misses + late)``: of the rain that came, how
  much was usefully warned about.

"Usefully" is ``--min-useful-lead-min`` (5). A warning whose rain arrives
less than five minutes later is **late**: not a hit, still on the recall
side (the rain came and nobody was told in time), and NOT a false alarm
(the rain did come — precision is not charged for it). That asymmetry is
the whole point of the knob: a rule that fires thirty seconds before the
first drop must not be able to buy a high score with warnings nobody could
act on. See ``warning_score`` for the definitions themselves.

The pick per lead is the **plateau midpoint**, not the argmax. F1 across a
threshold column is flat near its top and the argmax of a flat curve is
noise: the set of thresholds scoring at least ``--plateau-frac`` (0.95) of
the lead's best F1 is the plateau, and the pick is its midpoint rounded to
the nearest 5 (ties upward, toward the quieter rule). The bounds are
reported, so a reader can see how wide the flat region was — a one-cell
plateau is a warning sign in itself. Max CSI and the FAR-capped pick stay
in the output as secondary information for meteorologists.

A lead whose scored warnings across the entire grid number fewer than
``--min-warnings`` (30) gets NO pick and is marked ``insufficient``: the
service falls back to the shipping threshold rather than to a number
fitted on a handful of events.

The radar set: a self-consistency check, not a second truth
-----------------------------------------------------------
``--radar-decisions-dir`` takes the same replay run over the *radar
calibration points*, where the truth is the corpus ``outcome`` — the radar
observing itself — rather than a gauge. Onsets there are derived from the
decision rows' own ``observed_mm_h`` (≥ 0.5 mm/h is wet, on 10-minute
slots at ``radar_ts``) through the same :func:`onsets` rule, and swept
identically. Its plateau per lead is reported beside the gauge one, with
``agrees_with_radar`` saying whether the gauge pick falls inside it.

**This is not independent evidence.** The radar produced both the forecast
and its truth, so it shares every bias in the composite — column-max
reflectivity, bright band, virga that never reaches the ground. Agreement
means the fit is not an artefact of the ~100 gauge points; disagreement
means the two instruments disagree about what rain is. Neither promotes
the radar number over the gauge number, and the gauge pick is always the
one that ships.

Two callers, one implementation
-------------------------------
This module is the fit. :func:`run_fit` takes a :class:`SweepOptions` and
returns the whole payload, ``payload["thresholds"]`` being the small
document the service reads.

* ``scripts/sweep_thresholds.py`` is a thin CLI over it — the manual run,
  with ``--out-json`` / ``--out-md`` / ``--out-csv`` for the report and
  ``--previous`` for the stability guard.
* ``quality_report.QualityReportTask`` runs it nightly in its executor and
  writes the guarded table where the running service reads it.

It lives in the sidecar package rather than in ``dmi_nowcast_core``
because the replay is ``push.engine.evaluate`` itself — the fit must use
the shipped decision machinery, not a copy of it.

Later ``decisions_dirs`` entries win a ``(radar_ts, station_id)``
collision: the live row is the decision the service actually took, the
replay's is a reconstruction of what it would have taken.

Offline and read-only: parquet in, a document out. No DMI calls, no
network, no event loop — every caller runs it in a worker.
"""
from __future__ import annotations

import csv
import math
import multiprocessing as mp
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from dmi_nowcast_core.push_thresholds import (
    DEFAULT_FALLBACK_THRESHOLD_PCT,
    SCHEMA_VERSION as THRESHOLDS_SCHEMA_VERSION,
    validate_thresholds,
)
from dmi_nowcast_core.warning_score import (
    DEFAULT_COVERAGE_GAP_MIN,
    DEFAULT_DRY_MIN,
    DEFAULT_PRODUCT_LEADS_MIN,
    DEFAULT_TOLERANCE_MIN,
    PRECIP_DUR_PARAM,
    PRECIP_PARAM,
    SLOT_MIN,
    align_decision_table,
    coverage_runs,
    decision_leads_in,
    gauge_slots,
    p_rain_column,
    pooled_summary,
    score_warnings,
    slot_end_of,
)
from dmi_nowcast_core.warning_score import onsets as gauge_onsets

from .push import engine as decision_engine

#: The live rule this sweep is looking for a replacement for.
CURRENT_LEAD_MIN = 30
CURRENT_THRESHOLD_PCT = 40

#: Detection threshold shared with ``forecast.rain_threshold_mm_h`` and the
#: engine's own ``Rules.raining_now_mm_h`` default.
RAIN_THRESHOLD_MM_H = 0.5

#: Pad the gauge read either side of the decision window, so an onset at
#: the very start has its three dry slots behind it and a warning sent at
#: the very end can still find the rain it promised.
GAUGE_PAD_MIN = 120

#: Above this share of wrong notifications a rule is not shippable.
DEFAULT_FAR_CAP = 0.30

#: A warning whose rain arrives sooner than this is late, not a hit. The
#: scorer's own default is 0.0 — nothing late — because the historical
#: quality numbers were produced before this existed; the FIT asks for
#: five minutes, and this is the fit.
FIT_MIN_USEFUL_LEAD_MIN = 5.0

#: A threshold scoring at least this share of the lead's best F1 is on the
#: plateau, and the pick is the plateau's midpoint.
DEFAULT_PLATEAU_FRAC = 0.95

#: Fewer scored warnings than this across the whole grid at one lead and
#: the lead gets no pick at all.
DEFAULT_MIN_WARNINGS = 30

#: Picks are rounded to whole multiples of this many percent. The grid is
#: usually stepped by 5 too, so a rounded midpoint is a cell that was
#: actually measured.
PICK_ROUNDING_PCT = 5

#: The columns of the per-cell CSV, in order.
CSV_COLUMNS = (
    "lead_min", "threshold_pct", "warnings", "n_sent", "pending", "hits",
    "false_alarms", "late", "misses", "pending_onsets", "uncovered_onsets",
    "n_onsets", "pod", "far", "precision", "recall", "f1", "f_beta_0.5",
    "f_beta_2", "csi", "lead_error_p25", "lead_error_p50",
    "lead_error_p75", "lead_error_n", "warnings_per_station_day",
    "n_stations", "n_days", "n_rows",
)


# ---------------------------------------------------------------------------
# CLI argument shapes
# ---------------------------------------------------------------------------


def parse_leads(spec: str | None) -> tuple[int, ...]:
    """``"10,20,30"`` → ``(10, 20, 30)``; empty → the served product leads."""
    if spec is None or not spec.strip():
        return tuple(DEFAULT_PRODUCT_LEADS_MIN)
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        lead = int(chunk)
        if lead <= 0:
            raise ValueError(f"lead must be positive, got {lead}")
        out.append(lead)
    if not out:
        raise ValueError("no leads given")
    return tuple(sorted(set(out)))


def parse_thresholds(spec: str | None) -> tuple[int, ...]:
    """``"20:80:5"`` → 20, 25, … 80 (both ends inclusive).

    A comma-separated list is accepted too, for a hand-picked grid. The
    step defaults to 5 when a range omits it. Thresholds are whole
    percents in (0, 100): the engine compares ``p_rain >= pct / 100``, and
    0 % would fire on every frame while 100 % could never fire at all.
    """
    if spec is None or not spec.strip():
        raise ValueError("no thresholds given")
    spec = spec.strip()
    if ":" in spec:
        parts = [p.strip() for p in spec.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError(f"bad threshold range {spec!r}; want lo:hi[:step]")
        lo, hi = int(parts[0]), int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 5
        if step <= 0:
            raise ValueError("threshold step must be positive")
        if hi < lo:
            raise ValueError(f"bad threshold range {spec!r}: hi < lo")
        values = list(range(lo, hi + 1, step))
    else:
        values = [int(chunk) for chunk in spec.split(",") if chunk.strip()]
    out = sorted({int(v) for v in values})
    if not out:
        raise ValueError("no thresholds given")
    for value in out:
        if not 0 < value < 100:
            raise ValueError(f"threshold {value} out of range (0, 100)")
    return tuple(out)


# ---------------------------------------------------------------------------
# Loading decision rows
# ---------------------------------------------------------------------------

#: A file must have these to be a decision parquet. Anything else under a
#: given directory (``events.parquet``, ``onsets.parquet``, a stray export)
#: is skipped with a note rather than crashing the run.
_REQUIRED_COLUMNS = ("radar_ts", "station_id", "action")


def decision_parquets(directory: Path) -> list[Path]:
    """Every parquet under ``directory``, recursively, in a stable order."""
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.parquet") if p.is_file())


def _schema_leads(path: Path) -> tuple[bool, tuple[int, ...]]:
    """``(is a decision file, its lead columns)`` without reading the data."""
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    names = set(schema.names)
    if not all(column in names for column in _REQUIRED_COLUMNS):
        return False, ()
    return True, decision_leads_in(list(schema.names))


def load_decisions(
    directories: Sequence[Path],
    *,
    leads_min: Iterable[int] | None = None,
    log=None,
) -> tuple[list[dict], tuple[int, ...], dict[str, int]]:
    """Read every decision parquet under ``directories`` into one row list.

    Rows are deduplicated on ``(radar_ts, station_id)`` with the LATER
    directory winning, so the caller orders its ``--decisions-dir`` flags
    from least to most authoritative (replay first, live second).

    All files are aligned to the union of the requested leads and every
    lead any file carries, so a day written before ``p_rain_45`` existed
    reads back with a null there instead of a missing key.
    """
    import pyarrow.parquet as pq

    counts: dict[str, int] = {"files": 0, "skipped": 0, "rows": 0, "duplicates": 0}
    paths_by_dir: list[list[Path]] = []
    leads: set[int] = set(int(lead) for lead in (leads_min or ()))
    for directory in directories:
        usable: list[Path] = []
        for path in decision_parquets(directory):
            try:
                ok, file_leads = _schema_leads(path)
            except Exception as exc:  # noqa: BLE001 — one unreadable file
                counts["skipped"] += 1
                if log:
                    log(f"skipping {path.name}: {type(exc).__name__}: {exc}")
                continue
            if not ok:
                counts["skipped"] += 1
                if log:
                    log(f"skipping {path.name}: not a decision table")
                continue
            leads |= set(file_leads)
            usable.append(path)
        paths_by_dir.append(usable)

    union = tuple(sorted(leads))
    merged: dict[tuple[Any, str], dict] = {}
    for paths in paths_by_dir:
        for path in paths:
            try:
                table = align_decision_table(pq.read_table(path), union)
            except Exception as exc:  # noqa: BLE001 — one unreadable file
                counts["skipped"] += 1
                if log:
                    log(f"skipping {path.name}: {type(exc).__name__}: {exc}")
                continue
            counts["files"] += 1
            rows = table.to_pylist()
            counts["rows"] += len(rows)
            for row in rows:
                key = (row.get("radar_ts"), str(row.get("station_id")))
                if key in merged:
                    counts["duplicates"] += 1
                merged[key] = row
            if log:
                log(f"read {path.name}: {len(rows)} rows")
    ordered = sorted(
        merged.values(),
        key=lambda r: (r["radar_ts"], str(r.get("station_id"))),
    )
    return ordered, union, counts


# ---------------------------------------------------------------------------
# Tracks: the compact per-station form the sweep replays
# ---------------------------------------------------------------------------

#: One row, reduced to what ``evaluate`` reads plus its coverage run index:
#: ``(radar_ts, generated_at, eta_min, intensity_mm_h, observed_mm_h,
#: forecast_now_mm_h, run_id, p_rain_by_lead)``. A tuple, not a dataclass:
#: this is the structure a million rows are held in, and it is passed to
#: worker processes.
_RADAR_TS, _GENERATED, _ETA, _INTENSITY, _OBSERVED, _FORECAST, _RUN, _P = range(8)


def build_tracks(
    rows: Sequence[dict],
    leads: Sequence[int],
    *,
    coverage_gap_min: int = DEFAULT_COVERAGE_GAP_MIN,
) -> tuple[dict[str, list[tuple]], dict[str, list[datetime]]]:
    """Per-station tracks in ``radar_ts`` order, plus the raw frame stamps.

    The run index is assigned here, once, from the FULL frame sequence, so
    every lead in the sweep splits its state at exactly the same instants
    the coverage runs do — even at a lead whose probability column is
    mostly null.
    """
    by_station: dict[str, list[dict]] = {}
    for row in rows:
        radar_ts = row.get("radar_ts")
        if radar_ts is None:
            continue
        by_station.setdefault(str(row.get("station_id")), []).append(row)

    gap = timedelta(minutes=coverage_gap_min)
    columns = [p_rain_column(lead) for lead in leads]
    tracks: dict[str, list[tuple]] = {}
    frames: dict[str, list[datetime]] = {}
    for station, station_rows in by_station.items():
        station_rows.sort(key=lambda r: r["radar_ts"])
        track: list[tuple] = []
        stamps: list[datetime] = []
        run = 0
        previous: datetime | None = None
        for row in station_rows:
            radar_ts = row["radar_ts"]
            if previous is not None and radar_ts - previous > gap:
                run += 1
            previous = radar_ts
            stamps.append(radar_ts)
            generated = row.get("generated_at") or radar_ts
            track.append((
                radar_ts,
                generated,
                _opt_float(row.get("eta_min")),
                _opt_float(row.get("intensity_mm_h")),
                _opt_float(row.get("observed_mm_h")),
                _opt_float(row.get("forecast_now_mm_h")),
                run,
                tuple(_opt_float(row.get(column)) for column in columns),
            ))
        tracks[station] = track
        frames[station] = stamps
    return tracks, frames


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out  # NaN is missing, not a probability


# ---------------------------------------------------------------------------
# Gauge truth
# ---------------------------------------------------------------------------


def gauge_truth(
    corpus_dir: Path,
    station_ids: Sequence[str],
    window: tuple[datetime, datetime],
    *,
    dry_min: int = DEFAULT_DRY_MIN,
    log=None,
) -> tuple[dict[str, list[datetime]], dict[str, datetime], int]:
    """``(onsets per station, known_until per station, known slot count)``.

    Read month by month with a pad either side, exactly as
    ``quality_report._gauge_truth`` does, so the full slot grid for a year
    of a hundred stations is never held at once. Onsets found in the
    overlap of two padded windows are deduplicated by instant.

    None of this depends on the lead or the threshold, so it is computed
    once and shared by every cell of the sweep.
    """
    from dmi_nowcast_core.station_store import StationObsStore

    store = StationObsStore(Path(corpus_dir))
    pad = timedelta(minutes=GAUGE_PAD_MIN)
    start, end = window
    onset_sets: dict[str, set[datetime]] = {}
    known_until: dict[str, datetime] = {}
    known_slots = 0
    for year, month in _months_between(start - pad, end + pad):
        month_start = datetime(year, month, 1, tzinfo=timezone.utc) - pad
        if month == 12:
            month_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) + pad
        else:
            month_end = datetime(year, month + 1, 1, tzinfo=timezone.utc) + pad
        try:
            table = store.read(
                month_start, month_end,
                [PRECIP_PARAM, PRECIP_DUR_PARAM], list(station_ids),
            )
        except Exception as exc:  # noqa: BLE001 — one unreadable month
            if log:
                log(f"gauge read failed for {year}-{month:02d}: {exc}")
            continue
        for station in station_ids:
            slots = gauge_slots(
                table, station, start_utc=month_start, end_utc=month_end,
            )
            if not slots:
                continue
            onset_sets.setdefault(station, set()).update(
                gauge_onsets(slots, dry_min)
            )
            for stamp, wet in slots:
                if wet is None:
                    continue
                known_slots += 1
                previous = known_until.get(station)
                if previous is None or stamp > previous:
                    known_until[station] = stamp
        if log:
            log(f"gauge month {year}-{month:02d}: {known_slots} known slots so far")
    onsets_by_station = {
        station: sorted(values) for station, values in onset_sets.items()
    }
    return onsets_by_station, known_until, known_slots


def radar_truth(
    tracks: Mapping[str, Sequence[tuple]],
    *,
    dry_min: int = DEFAULT_DRY_MIN,
    slot_min: int = SLOT_MIN,
    threshold_mm_h: float = RAIN_THRESHOLD_MM_H,
) -> tuple[dict[str, list[datetime]], dict[str, datetime]]:
    """``(onsets per point, known_until per point)`` from the radar itself.

    The radar decision set has no gauge behind it. Its truth is the rain
    rate the composite observed at the point — ``observed_mm_h``, the same
    column the engine's "already raining" silence reads — binned onto the
    same 10-minute slot grid the gauges report on and pushed through the
    same :func:`onsets` rule, so the two sweeps differ in their truth and
    in nothing else.

    A null ``observed_mm_h`` is off-coverage or nodata: it leaves the slot
    UNKNOWN, exactly as an unreported gauge slot does, and an unknown slot
    resets the dry run rather than certifying it. The grid is filled
    contiguously between the first and last frame so a gap in the rows
    cannot be mistaken for a dry spell.

    This is a self-consistency check. The forecast and the truth come from
    the same instrument, so agreement is evidence that the gauge fit is
    not an artefact of a hundred points — not evidence that either number
    is right.
    """
    step = timedelta(minutes=slot_min)
    onsets_by_point: dict[str, list[datetime]] = {}
    known_until: dict[str, datetime] = {}
    for point, track in tracks.items():
        seen: dict[datetime, bool] = {}
        for record in track:
            observed = record[_OBSERVED]
            if observed is None:
                continue
            slot = slot_end_of(record[_RADAR_TS], slot_min=slot_min)
            seen[slot] = seen.get(slot, False) or (observed >= threshold_mm_h)
        if not seen:
            continue
        slots: list[tuple[datetime, bool | None]] = []
        cursor, last = min(seen), max(seen)
        while cursor <= last:
            slots.append((cursor, seen.get(cursor)))
            cursor += step
        onsets_by_point[point] = gauge_onsets(slots, dry_min, slot_min=slot_min)
        known_until[point] = last
    return onsets_by_point, known_until


def _months_between(start: datetime, end: datetime) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        out.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


# ---------------------------------------------------------------------------
# One cell of the grid
# ---------------------------------------------------------------------------


def _engine():
    """The shipped decision engine. A function, so a test can swap it."""
    return decision_engine


def replay_station(
    track: Sequence[tuple],
    lead_index: int,
    threshold_pct: int,
    *,
    persistence_obs: int,
    rearm_after_min: int,
    raining_now_mm_h: float = RAIN_THRESHOLD_MM_H,
) -> list[tuple[datetime, float | None]]:
    """One station's warnings under one (lead, threshold) rule.

    The subscription starts armed at the head of every coverage run, and a
    row with no probability at this lead is passed over without touching
    the state. Everything else is ``push.engine.evaluate`` — the arming,
    the persistence streak, the 60-minute re-arm and the "already raining"
    silence are the engine's, not this script's.
    """
    eng = _engine()
    rules = eng.Rules(
        persistence_obs=int(persistence_obs),
        rearm_after_min=int(rearm_after_min),
        raining_now_mm_h=float(raining_now_mm_h),
    )
    state = eng.INITIAL_STATE
    run: int | None = None
    warnings: list[tuple[datetime, float | None]] = []
    for record in track:
        if record[_RUN] != run:
            run = record[_RUN]
            state = eng.INITIAL_STATE
        p_rain = record[_P][lead_index]
        if p_rain is None:
            continue
        decision = eng.evaluate(
            state,
            eng.Observation(
                radar_ts_utc=record[_RADAR_TS],
                p_rain=p_rain,
                eta_min=record[_ETA],
                intensity_mm_h=record[_INTENSITY],
                observed_mm_h=record[_OBSERVED],
                forecast_now_mm_h=record[_FORECAST],
            ),
            threshold_pct=int(threshold_pct),
            quiet=None,
            tz="UTC",
            now_utc=record[_GENERATED],
            rules=rules,
        )
        state = decision.state
        if decision.action == "notify":
            warnings.append((record[_GENERATED], record[_ETA]))
    return warnings


def score_cell(shared: dict, lead: int, threshold_pct: int | None) -> dict:
    """Replay and score one grid cell; ``threshold_pct=None`` = do nothing.

    The do-nothing row is scored through exactly the same machinery with
    an empty warning list, so its miss count is the same population every
    real cell's POD is measured against.
    """
    lead_index = shared["leads"].index(int(lead))
    results = []
    n_sent = 0
    for station in shared["stations"]:
        if threshold_pct is None:
            warnings: list[tuple[datetime, float | None]] = []
        else:
            warnings = replay_station(
                shared["tracks"][station],
                lead_index,
                int(threshold_pct),
                persistence_obs=shared["persistence_obs"],
                rearm_after_min=shared["rearm_after_min"],
                raining_now_mm_h=shared["raining_now_mm_h"],
            )
        n_sent += len(warnings)
        results.append(score_warnings(
            warnings,
            shared["onsets"].get(station, ()),
            lead_min=int(lead),
            tolerance_min=shared["tolerance_min"],
            dry_min=shared["dry_min"],
            known_until=shared["known_until"].get(station),
            coverage=shared["coverage"][int(lead)].get(station, ()),
            min_useful_lead_min=shared["min_useful_lead_min"],
        ))
    pooled = pooled_summary(results)
    return _cell(pooled, lead, threshold_pct, shared)


def _cell(pooled: dict, lead: int, threshold_pct: int | None, shared: dict) -> dict:
    """A pooled summary, plus the derived numbers the report ranks on.

    Every rate comes from ``pooled_summary``, which recomputes it from the
    pooled counts: precision and recall as the fit's objective reads them
    (late warnings out of precision, on the miss side of recall), CSI with
    late on the miss side too, and FAR over every graded warning — so FAR
    is NOT ``1 − precision`` once a lead has lates, and both are printed.
    """
    station_days = max(1, shared["station_days"])
    spread = pooled["lead_error_min"]
    f_beta = pooled["f_beta"]
    return {
        "lead_min": int(lead),
        "threshold_pct": None if threshold_pct is None else int(threshold_pct),
        "warnings": pooled["warnings"],
        "n_sent": pooled["n_sent"],
        "pending": pooled["pending"],
        "hits": pooled["hits"],
        "false_alarms": pooled["false_alarms"],
        "late": pooled["late"],
        "misses": pooled["misses"],
        "pending_onsets": pooled["pending_onsets"],
        "uncovered_onsets": pooled["uncovered_onsets"],
        "n_onsets": pooled["n_onsets"],
        "pod": pooled["pod"],
        "far": pooled["far"],
        "precision": pooled["precision"],
        "recall": pooled["recall"],
        "f1": pooled["f1"],
        "f_beta_0.5": f_beta.get("0.5"),
        "f_beta_2": f_beta.get("2"),
        "csi": pooled["csi"],
        "lead_error_min": spread,
        "warnings_per_station_day": pooled["n_sent"] / station_days,
        "n_stations": len(shared["stations"]),
        "n_days": shared["n_days"],
        "n_rows": shared["n_rows"],
    }


# ---------------------------------------------------------------------------
# The grid, in parallel
# ---------------------------------------------------------------------------

#: Set once per worker process. Under ``fork`` the children inherit it for
#: free; under ``spawn`` the pool initializer plants it. Either way the
#: rows are loaded once in the parent and never rebuilt per cell.
_SHARED: dict | None = None


def _init_worker(shared: dict) -> None:
    global _SHARED
    _SHARED = shared


def release_shared() -> None:
    """Drop the parent's payload reference between two sweeps.

    ``run_sweep`` plants the payload in a module global so a forked worker
    inherits it for free; that global also keeps the whole gauge set alive
    after the sweep is over, and the radar cross-check would then load a
    second one beside it. Dropping the local name is not enough — this is.
    """
    global _SHARED
    _SHARED = None


def _worker(task: tuple[int, int | None]) -> dict:
    lead, threshold = task
    if _SHARED is None:  # pragma: no cover — a pool wired without a payload
        raise RuntimeError("worker started without a shared payload")
    return score_cell(_SHARED, lead, threshold)


def run_sweep(
    shared: dict,
    leads: Sequence[int],
    thresholds: Sequence[int],
    *,
    workers: int = 1,
    log=None,
) -> tuple[list[dict], dict[str, dict]]:
    """Score every ``(lead, threshold)`` cell plus one do-nothing per lead."""
    global _SHARED
    _SHARED = shared
    tasks: list[tuple[int, int | None]] = [
        (lead, threshold) for lead in leads for threshold in thresholds
    ]
    tasks += [(lead, None) for lead in leads]
    done = 0
    total = len(tasks)
    results: list[dict] = []

    def absorb(cell: dict) -> None:
        nonlocal done
        done += 1
        if log:
            label = (
                "none" if cell["threshold_pct"] is None
                else f"{cell['threshold_pct']}%"
            )
            log(
                f"[{done}/{total}] lead {cell['lead_min']} @ {label}: "
                f"{cell['n_sent']} warnings, POD {_fmt(cell['pod'])}, "
                f"FAR {_fmt(cell['far'])}, CSI {_fmt(cell['csi'])}"
            )
        results.append(cell)

    if workers <= 1:
        for task in tasks:
            absorb(_worker(task))
    else:
        ctx = (
            mp.get_context("fork")
            if "fork" in mp.get_all_start_methods()
            else mp.get_context("spawn")
        )
        kwargs: dict[str, Any] = {}
        if ctx.get_start_method() != "fork":
            # Under spawn the payload has to cross the process boundary;
            # the initializer sends it once per worker rather than once
            # per cell.
            kwargs = {"initializer": _init_worker, "initargs": (shared,)}
        with ProcessPoolExecutor(
            max_workers=int(workers), mp_context=ctx, **kwargs,
        ) as pool:
            for cell in pool.map(_worker, tasks, chunksize=1):
                absorb(cell)

    cells = [c for c in results if c["threshold_pct"] is not None]
    cells.sort(key=lambda c: (c["lead_min"], c["threshold_pct"]))
    do_nothing = {
        str(c["lead_min"]): c for c in results if c["threshold_pct"] is None
    }
    return cells, do_nothing


# ---------------------------------------------------------------------------
# Picking
# ---------------------------------------------------------------------------


def scored_warnings(cells: Sequence[dict]) -> int:
    """Total graded warnings a lead produced across the whole grid.

    The evidence test for a pick. Summed over thresholds, so it double
    counts the same rain seen at 40 % and at 45 % — deliberately: it is not
    a sample size, it is the question "did this horizon ever do anything
    worth fitting on?", and a lead that fired twice at every threshold
    should fail it.
    """
    return sum(int(c.get("warnings") or 0) for c in cells)


def round_to(value: float, step: int = PICK_ROUNDING_PCT) -> int:
    """Nearest multiple of ``step``, halves rounded UP (the quieter rule)."""
    return int(math.floor(value / step + 0.5) * step)


def pick_plateau(
    cells: Sequence[dict], plateau_frac: float = DEFAULT_PLATEAU_FRAC,
) -> dict | None:
    """The F1 plateau and its midpoint — the pick that ships.

    Every threshold scoring at least ``plateau_frac`` of the lead's best
    F1 is on the plateau; the pick is the midpoint of its bounds, rounded
    to :data:`PICK_ROUNDING_PCT` with halves going up. Taking the midpoint
    rather than the argmax is the whole point: two thresholds five percent
    apart whose F1 differs in the third decimal are not distinguishable on
    this much data, and the middle of the flat region is the choice that
    survives the next month of it.

    ``cell`` is the measured cell the reported metrics come from: the one
    at the pick when the grid has it (a step-5 grid always does), else the
    nearest cell on the plateau, so no number in the output is an
    interpolation. The pick is clamped into the plateau bounds, which only
    ever bites on a grid whose thresholds are not multiples of five — the
    edge of a measured plateau beats a round number outside it. ``None``
    when no cell at this lead has an F1 at all.
    """
    scored = [c for c in cells if c.get("f1") is not None]
    if not scored:
        return None
    best = max(float(c["f1"]) for c in scored)
    if best <= 0.0:
        # Every cell scored zero: there is no plateau to be in the middle
        # of, and a "pick" here would be an arbitrary threshold dressed up
        # as a measurement.
        return None
    on_plateau = [
        c for c in scored if float(c["f1"]) >= plateau_frac * best
    ]
    thresholds = sorted(int(c["threshold_pct"]) for c in on_plateau)
    lo, hi = thresholds[0], thresholds[-1]
    pick = round_to((lo + hi) / 2.0)
    pick = min(max(pick, lo), hi)
    cell = min(
        on_plateau,
        key=lambda c: (abs(int(c["threshold_pct"]) - pick), -int(c["threshold_pct"])),
    )
    return {
        "threshold_pct": pick,
        "plateau": [lo, hi],
        "plateau_frac": float(plateau_frac),
        "max_f1": best,
        "n_thresholds": len(thresholds),
        "cell": cell,
    }


def pick_max_csi(cells: Sequence[dict]) -> dict | None:
    """The highest-CSI cell; ties go to the higher threshold (less noise)."""
    scored = [c for c in cells if c["csi"] is not None]
    if not scored:
        return None
    return max(scored, key=lambda c: (c["csi"], c["threshold_pct"]))


def pick_max_pod_under_far(
    cells: Sequence[dict], far_cap: float,
) -> dict | None:
    """The highest-POD cell whose FAR is at or under the cap.

    A cell that sent nothing has no FAR — an undefined rate is not a
    passing one — so it cannot win here by default; the do-nothing row is
    reported separately as the floor.
    """
    eligible = [
        c for c in cells
        if c["far"] is not None and c["far"] <= far_cap and c["pod"] is not None
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda c: (c["pod"], c["threshold_pct"]))


def radar_sweep(
    directories: Sequence[Path],
    leads: Sequence[int],
    thresholds: Sequence[int],
    *,
    coverage_gap_min: int = DEFAULT_COVERAGE_GAP_MIN,
    tolerance_min: int = DEFAULT_TOLERANCE_MIN,
    dry_min: int = DEFAULT_DRY_MIN,
    persistence_obs: int = 1,
    rearm_after_min: int = 60,
    min_useful_lead_min: float = FIT_MIN_USEFUL_LEAD_MIN,
    raining_now_mm_h: float = RAIN_THRESHOLD_MM_H,
    workers: int = 1,
    log=None,
) -> tuple[list[dict], dict]:
    """The same grid over the radar calibration points, against the radar.

    Identical machinery to the gauge sweep — same loader, same tracks,
    same replay, same scorer, same thresholds — with exactly one thing
    swapped: the onsets come from :func:`radar_truth` rather than from the
    gauge store. That is what makes the two plateaus comparable at all.

    Returns ``(cells, info)``; ``info`` describes the set even when it is
    empty, so a caller can say "no radar rows" rather than "no agreement".
    """
    rows, file_leads, counts = load_decisions(directories, leads_min=(), log=log)
    info: dict[str, Any] = {
        "dirs": [str(d) for d in directories],
        "rows": len(rows),
        "files": counts["files"],
        "points": 0,
        "onsets": 0,
        "leads": [],
    }
    if not rows:
        return [], info
    usable = [lead for lead in leads if lead in file_leads]
    missing = [lead for lead in leads if lead not in file_leads]
    if missing and log:
        log(
            "radar set has no column for lead(s): "
            + ", ".join(str(lead) for lead in missing)
        )
    if not usable:
        return [], info
    tracks, frames = build_tracks(rows, usable, coverage_gap_min=coverage_gap_min)
    n_rows = len(rows)
    del rows
    onsets_by_point, known_until = radar_truth(tracks, dry_min=dry_min)
    points = sorted(p for p in tracks if p in known_until)
    days_by_point = {
        point: {record[_GENERATED].date() for record in tracks[point]}
        for point in points
    }
    days = set().union(*days_by_point.values()) if days_by_point else set()
    info.update({
        "points": len(points),
        "onsets": sum(len(onsets_by_point.get(p, ())) for p in points),
        "leads": list(usable),
        "days": len(days),
        "rows": n_rows,
    })
    if not points:
        return [], info
    shared = {
        "leads": list(usable),
        "stations": points,
        "tracks": {p: tracks[p] for p in points},
        "onsets": onsets_by_point,
        "known_until": known_until,
        "coverage": {
            lead: {
                point: coverage_runs(
                    frames[point],
                    max_gap_min=coverage_gap_min,
                    extend_min=lead + tolerance_min,
                )
                for point in points
            }
            for lead in usable
        },
        "tolerance_min": int(tolerance_min),
        "dry_min": int(dry_min),
        "min_useful_lead_min": float(min_useful_lead_min),
        "persistence_obs": int(persistence_obs),
        "rearm_after_min": int(rearm_after_min),
        "raining_now_mm_h": float(raining_now_mm_h),
        "station_days": sum(len(days_by_point[p]) for p in points),
        "n_days": len(days),
        "n_rows": n_rows,
    }
    del tracks, frames
    cells, _ = run_sweep(shared, usable, thresholds, workers=workers, log=log)
    return cells, info


def build_picks(
    cells: Sequence[dict],
    leads: Sequence[int],
    *,
    far_cap: float = DEFAULT_FAR_CAP,
    plateau_frac: float = DEFAULT_PLATEAU_FRAC,
    min_warnings: int = DEFAULT_MIN_WARNINGS,
    radar_cells: Sequence[dict] | None = None,
) -> dict[str, dict]:
    """Every pick for every lead, plus the radar cross-check.

    The plateau pick is the one that ships and it is withheld — no pick,
    ``insufficient`` true — when the lead's whole grid produced fewer than
    ``min_warnings`` graded warnings. Max CSI and the FAR-capped pick are
    reported regardless: they are secondary information, and a reader
    comparing them against a withheld plateau learns something about how
    thin the evidence was.

    ``agrees_with_radar`` is true when the gauge pick lands inside the
    radar plateau, false when it does not, and null when either sweep had
    no plateau to compare — never quietly true.
    """
    out: dict[str, dict] = {}
    for lead in leads:
        lead_cells = [c for c in cells if c["lead_min"] == lead]
        total = scored_warnings(lead_cells)
        insufficient = total < int(min_warnings)
        plateau = (
            None if insufficient else pick_plateau(lead_cells, plateau_frac)
        )
        radar_plateau = None
        if radar_cells is not None:
            radar_plateau = pick_plateau(
                [c for c in radar_cells if c["lead_min"] == lead], plateau_frac,
            )
        agrees: bool | None = None
        if plateau is not None and radar_plateau is not None:
            lo, hi = radar_plateau["plateau"]
            agrees = lo <= plateau["threshold_pct"] <= hi
        out[str(lead)] = {
            "plateau": plateau,
            "insufficient": insufficient,
            "scored_warnings": total,
            "min_warnings": int(min_warnings),
            "max_csi": pick_max_csi(lead_cells),
            "max_pod_far_capped": pick_max_pod_under_far(lead_cells, far_cap),
            "radar_plateau": radar_plateau,
            "agrees_with_radar": agrees,
        }
    return out


def build_thresholds_document(
    payload: dict,
    *,
    fallback_threshold_pct: int = DEFAULT_FALLBACK_THRESHOLD_PCT,
) -> dict:
    """The small file the service reads: one threshold per horizon.

    Everything the sweep learned, reduced to what a request-time lookup
    needs plus enough provenance to argue with it: the objective and the
    rule constants it was fitted under, the window of evidence, and per
    lead the pick, the counts behind it, the plateau, and the radar
    cross-check. ``push_thresholds.validate_thresholds`` is the contract;
    ``push_thresholds.effective_threshold`` is the reader.

    A lead with no pick carries null rates and zero counts: those fields
    describe the picked cell, and where there is no pick there is no cell
    to describe. ``insufficient`` says which of the two reasons applies —
    too little evidence, or a grid on which nothing scored at all.
    """
    settings = payload["settings"]
    window = payload["window"]
    leads: dict[str, dict] = {}
    for lead in payload["leads"]:
        pick = payload["picks"].get(str(lead), {})
        plateau = pick.get("plateau")
        cell = (plateau or {}).get("cell") or {}
        radar = pick.get("radar_plateau")
        leads[str(lead)] = {
            "threshold_pct": None if plateau is None else int(
                plateau["threshold_pct"]
            ),
            "insufficient": bool(pick.get("insufficient")),
            "f1": cell.get("f1"),
            "precision": cell.get("precision"),
            "recall": cell.get("recall"),
            "far": cell.get("far"),
            "csi": cell.get("csi"),
            "warnings": int(cell.get("warnings") or 0),
            "hits": int(cell.get("hits") or 0),
            "false_alarms": int(cell.get("false_alarms") or 0),
            "misses": int(cell.get("misses") or 0),
            "late": int(cell.get("late") or 0),
            "plateau": None if plateau is None else list(plateau["plateau"]),
            "radar_plateau": None if radar is None else list(radar["plateau"]),
            "agrees_with_radar": pick.get("agrees_with_radar"),
        }
    return {
        "schema_version": THRESHOLDS_SCHEMA_VERSION,
        "fitted_at_utc": payload["generated_at_utc"],
        "objective": {
            "metric": "f1",
            "min_useful_lead_min": float(settings["min_useful_lead_min"]),
            "plateau_frac": float(settings["plateau_frac"]),
            "min_warnings": int(settings["min_warnings"]),
            "rearm_after_min": int(settings["rearm_after_min"]),
            "persistence_obs": int(settings["persistence_obs"]),
            "tolerance_min": int(settings["tolerance_min"]),
            "dry_min": int(settings["dry_min"]),
        },
        "window": {
            "from": window["from"],
            "to": window["to"],
            "days": int(window["days"]),
            # The stations that carried gauge truth — the ones the fit
            # actually stands on, not every station with a decision row.
            "stations": int(window["stations_scored"]),
            "rows": int(window["rows"]),
        },
        "fallback_threshold_pct": int(fallback_threshold_pct),
        "leads": leads,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 2) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:.0f} %"


def _minutes(value: float | None) -> str:
    return "–" if value is None else f"{value:+.0f}"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _threshold_of(cell: Mapping[str, Any] | None) -> int | None:
    return None if cell is None else cell.get("threshold_pct")


def _find(cells: Sequence[dict], lead: int, threshold: int) -> dict | None:
    for cell in cells:
        if cell["lead_min"] == lead and cell["threshold_pct"] == threshold:
            return cell
    return None


def render_markdown(payload: dict) -> str:
    """The human-readable half: one table and one paragraph per lead."""
    settings = payload["settings"]
    window = payload["window"]
    cells = payload["cells"]
    lines: list[str] = []
    lines.append("# Push threshold sweep")
    lines.append("")
    lines.append(
        f"Generated {payload['generated_at_utc']}. Every (lead, threshold) "
        "pair below was replayed through the live push state machine over "
        "stored decision rows and scored against the DMI rain gauges."
    )
    lines.append("")
    lines.append(
        f"- Window: {window['from']} → {window['to']} "
        f"({window['days']} day(s), {window['stations_scored']} of "
        f"{window['stations']} stations with gauge data, "
        f"{window['rows']} decision rows, {window['station_days']} station-days)"
    )
    lines.append(
        f"- Rule constants: persistence {settings['persistence_obs']} "
        f"observation(s), re-arm after {settings['rearm_after_min']} min, "
        f"silenced above {settings['raining_now_mm_h']} mm/h observed, "
        "quiet hours off"
    )
    lines.append(
        f"- Scoring: {settings['tolerance_min']} min tolerance, "
        f"{settings['dry_min']} min dry run before an onset, coverage gap "
        f"{settings['coverage_gap_min']} min"
    )
    lines.append(
        f"- Objective: max F1, plateau ≥ "
        f"{settings['plateau_frac'] * 100:.0f} % of the lead's best, "
        f"minimum useful lead {settings['min_useful_lead_min']:.0f} min, "
        f"at least {settings['min_warnings']} scored warnings per lead"
    )
    lines.append(f"- FAR cap for the secondary pick: {_pct(settings['far_cap'])}")
    if payload.get("radar"):
        radar = payload["radar"]
        lines.append(
            f"- Radar cross-check: {radar['points']} calibration point(s), "
            f"{radar['rows']} decision row(s) — a self-consistency check "
            "against the composite's own `observed_mm_h`, NOT independent "
            "truth. The gauge pick is the one that ships."
        )
    lines.append("")
    lines.append(
        "Precision is the share of warnings rain followed usefully; recall "
        "(= POD) the share of covered onsets usefully warned about; F1 "
        "their harmonic mean, and the objective. A **late** warning — rain "
        f"less than {settings['min_useful_lead_min']:.0f} min after it was "
        "sent — is not a hit and not a false alarm: it stays on the recall "
        "side only, so FAR is not `1 − precision` wherever the late column "
        "is non-zero. CSI (`hits / (hits + misses + late + false alarms)`) "
        "is kept for comparison with the meteorological literature. Lead "
        "error is `eta − (onset − sent)` in minutes over the hits: positive "
        "means the rain beat the ETA. Every column adds up: `hits + late + "
        "false alarms + pending = sent`, and a pending warning is one whose "
        "window the gauge record does not yet cover, held out of every rate "
        "rather than graded on evidence that does not exist."
    )
    lines.append("")

    for lead in payload["leads"]:
        lead_cells = [c for c in cells if c["lead_min"] == lead]
        if not lead_cells:
            continue
        nothing = payload["do_nothing"].get(str(lead))
        picks = payload["picks"].get(str(lead), {})
        lines.append(f"## Lead {lead} min")
        lines.append("")
        lines.append(
            "| threshold | sent | hits | late | false alarms | pending | "
            "misses | precision | recall (POD) | F1 | F0.5 | F2 | FAR | CSI "
            "| lead err p50 | sent / station-day |"
        )
        lines.append(
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        if nothing:
            lines.append(
                f"| no rule | 0 | 0 | 0 | 0 | 0 | {nothing['misses']} | – | "
                f"{_fmt(nothing['pod'])} | – | – | – | – | "
                f"{_fmt(nothing['csi'])} | – | 0.00 |"
            )
        for cell in lead_cells:
            current = (
                lead == CURRENT_LEAD_MIN
                and cell["threshold_pct"] == CURRENT_THRESHOLD_PCT
            )
            label = f"{cell['threshold_pct']} %"
            if current:
                label = f"**{label}** (shipping today)"
            lines.append(
                f"| {label} | {cell['n_sent']} | {cell['hits']} | "
                f"{cell['late']} | {cell['false_alarms']} | {cell['pending']} | "
                f"{cell['misses']} | "
                f"{_fmt(cell['precision'])} | {_fmt(cell['recall'])} | "
                f"{_fmt(cell['f1'])} | {_fmt(cell['f_beta_0.5'])} | "
                f"{_fmt(cell['f_beta_2'])} | {_fmt(cell['far'])} | "
                f"{_fmt(cell['csi'])} | "
                f"{_minutes(cell['lead_error_min'].get('p50'))} | "
                f"{cell['warnings_per_station_day']:.2f} |"
            )
        lines.append("")
        lines.extend(_plateau_lines(picks, settings))
        best_csi = picks.get("max_csi")
        best_far = picks.get("max_pod_far_capped")
        if best_csi:
            lines.append(
                f"- **Best CSI:** {best_csi['threshold_pct']} % — CSI "
                f"{_fmt(best_csi['csi'])}, POD {_fmt(best_csi['pod'])}, FAR "
                f"{_fmt(best_csi['far'])}, "
                f"{best_csi['warnings_per_station_day']:.2f} warnings per "
                "station-day."
            )
        else:
            lines.append("- **Best CSI:** no cell scored — no onsets in range.")
        if best_far:
            lines.append(
                f"- **Best POD with FAR ≤ {_pct(settings['far_cap'])}:** "
                f"{best_far['threshold_pct']} % — POD {_fmt(best_far['pod'])}, "
                f"FAR {_fmt(best_far['far'])}, CSI {_fmt(best_far['csi'])}."
            )
        else:
            lines.append(
                f"- **Best POD with FAR ≤ {_pct(settings['far_cap'])}:** none. "
                "No threshold on this grid kept the false-alarm rate under "
                "the cap."
            )
        lines.append("")
        lines.append(_lead_paragraph(lead, lead_cells, nothing, picks, payload))
        lines.append("")
    return "\n".join(lines) + "\n"


def _plateau_lines(picks: Mapping[str, Any], settings: Mapping[str, Any]) -> list[str]:
    """The shipping pick for one lead: the plateau, or why there is none."""
    plateau = picks.get("plateau")
    if plateau is None:
        if picks.get("insufficient"):
            return [
                f"- **Pick: none.** Only {picks.get('scored_warnings', 0)} "
                f"scored warning(s) across the whole grid at this lead, "
                f"under the {picks.get('min_warnings', DEFAULT_MIN_WARNINGS)} "
                "this fit requires. The horizon keeps the fallback threshold.",
            ]
        return [
            "- **Pick: none.** No threshold at this lead produced an F1 at "
            "all — nothing was both warned about and verifiable.",
        ]
    lo, hi = plateau["plateau"]
    cell = plateau["cell"]
    out = [
        f"- **Pick: {plateau['threshold_pct']} %** — the midpoint of the F1 "
        f"plateau [{lo} %, {hi} %] "
        f"({plateau['n_thresholds']} threshold(s) within "
        f"{settings['plateau_frac'] * 100:.0f} % of the best F1 "
        f"{_fmt(plateau['max_f1'])}). At the pick: precision "
        f"{_fmt(cell['precision'])}, recall {_fmt(cell['recall'])}, F1 "
        f"{_fmt(cell['f1'])}, {cell['hits']} hit(s), {cell['late']} late, "
        f"{cell['false_alarms']} false alarm(s), {cell['misses']} miss(es).",
    ]
    radar = picks.get("radar_plateau")
    agrees = picks.get("agrees_with_radar")
    if radar is not None:
        rlo, rhi = radar["plateau"]
        out.append(
            f"- **Radar cross-check:** plateau [{rlo} %, {rhi} %] "
            f"(pick {radar['threshold_pct']} %) — the gauge pick "
            + ("is inside it" if agrees else "falls OUTSIDE it")
            + ". Same instrument produced forecast and truth, so this is "
            "consistency, not confirmation."
        )
    return out


def _lead_paragraph(
    lead: int,
    lead_cells: Sequence[dict],
    nothing: dict | None,
    picks: dict,
    payload: dict,
) -> str:
    """Plain language for one lead: what the column says, in sentences."""
    settings = payload["settings"]
    lowest, highest = lead_cells[0], lead_cells[-1]
    parts: list[str] = []
    parts.append(
        f"At a {lead}-minute horizon the grid runs from "
        f"{lowest['threshold_pct']} % ({_plural(lowest['n_sent'], 'warning')}, "
        f"POD {_fmt(lowest['pod'])}, FAR {_fmt(lowest['far'])}) to "
        f"{highest['threshold_pct']} % ({_plural(highest['n_sent'], 'warning')}, "
        f"POD {_fmt(highest['pod'])}, FAR {_fmt(highest['far'])})."
    )
    best_csi = picks.get("max_csi")
    if best_csi:
        parts.append(
            f"The best single number is at {best_csi['threshold_pct']} %: "
            f"{best_csi['hits']} of "
            f"{_plural(best_csi['n_sent'], 'warning')} landed inside the "
            f"promised window and {_plural(best_csi['misses'], 'onset')} went "
            f"unwarned (CSI {_fmt(best_csi['csi'])})."
        )
    best_far = picks.get("max_pod_far_capped")
    if best_far:
        parts.append(
            "Holding false alarms under "
            f"{_pct(settings['far_cap'])} means moving to "
            f"{best_far['threshold_pct']} %, which still catches "
            f"{_pct(best_far['pod'])} of the rain at "
            f"{best_far['warnings_per_station_day']:.2f} notifications per "
            "subscriber-day."
        )
    else:
        parts.append(
            "No threshold on this grid held false alarms under "
            f"{_pct(settings['far_cap'])}, so at this horizon the honest "
            "answer is that the second objective is out of reach with "
            "today's calibration."
        )
    if nothing:
        parts.append(
            "Sending nothing at all would have left all "
            f"{_plural(nothing['misses'], 'covered onset')} unwarned, which "
            "is the floor every row above has to beat."
        )
    current = _find(payload["cells"], CURRENT_LEAD_MIN, CURRENT_THRESHOLD_PCT)
    if lead == CURRENT_LEAD_MIN:
        if current:
            parts.append(
                f"The rule shipping today — {CURRENT_LEAD_MIN} min at "
                f"{CURRENT_THRESHOLD_PCT} % — sits at POD "
                f"{_fmt(current['pod'])}, FAR {_fmt(current['far'])}, CSI "
                f"{_fmt(current['csi'])}, "
                f"{current['warnings_per_station_day']:.2f} notifications per "
                f"subscriber-day, with a median lead error of "
                f"{_minutes(current['lead_error_min'].get('p50'))} min."
            )
        else:
            parts.append(
                f"The rule shipping today — {CURRENT_LEAD_MIN} min at "
                f"{CURRENT_THRESHOLD_PCT} % — is not on this grid; add "
                f"{CURRENT_THRESHOLD_PCT} to --thresholds to place it."
            )
    return " ".join(parts)


def write_csv(path: Path, cells: Sequence[dict], do_nothing: dict) -> None:
    """Every cell as one CSV row; the do-nothing rows have a blank threshold."""
    rows = list(cells) + [
        do_nothing[key] for key in sorted(do_nothing, key=int)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for cell in rows:
            spread = cell["lead_error_min"]
            flat = {name: cell.get(name) for name in CSV_COLUMNS}
            flat["lead_error_p25"] = spread.get("p25")
            flat["lead_error_p50"] = spread.get("p50")
            flat["lead_error_p75"] = spread.get("p75")
            flat["lead_error_n"] = spread.get("n")
            writer.writerow(flat)


def write_atomic(path: Path, text: str) -> None:
    """tmp + rename in the target directory; a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


class SweepError(RuntimeError):
    """The fit cannot run on what it was given.

    Not a bug and not a crash: no decision rows, no lead with a
    probability column, no gauge observation over the window. Every one of
    them means "there is nothing here to fit on", which the CLI turns into
    an exit code and the nightly task turns into one log line and
    yesterday's table.
    """


@dataclass(frozen=True)
class SweepOptions:
    """Everything :func:`run_fit` needs. The CLI's flags, as a value.

    Defaults are the shipped fit: the objective's five-minute useful lead,
    the 0.95 plateau, 30 scored warnings of evidence, one observation of
    persistence and a 60-minute re-arm — the live rule's own constants,
    because a replay under different constants would be measuring a
    different service.
    """

    decisions_dirs: Sequence[Path]
    corpus_dir: Path
    radar_decisions_dirs: Sequence[Path] | None = None
    leads: Sequence[int] = DEFAULT_PRODUCT_LEADS_MIN
    thresholds: Sequence[int] = (20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80)
    rearm_after_min: int = 60
    persistence_obs: int = 1
    tolerance_min: int = DEFAULT_TOLERANCE_MIN
    dry_min: int = DEFAULT_DRY_MIN
    coverage_gap_min: int = DEFAULT_COVERAGE_GAP_MIN
    far_cap: float = DEFAULT_FAR_CAP
    min_useful_lead_min: float = FIT_MIN_USEFUL_LEAD_MIN
    plateau_frac: float = DEFAULT_PLATEAU_FRAC
    min_warnings: int = DEFAULT_MIN_WARNINGS
    fallback_threshold_pct: int = DEFAULT_FALLBACK_THRESHOLD_PCT
    workers: int = 1


def run_fit(options: SweepOptions, *, log=None) -> dict:
    """Replay the whole grid and reduce it to a fitted table.

    The payload is the full record — every cell, the do-nothing rows, the
    picks, the radar cross-check, the window — with
    ``payload["thresholds"]`` the small document the service reads. The
    caller decides what to write and whether to guard it against the table
    already in service (``push_thresholds.apply_stability_guard``).

    Raises :class:`SweepError` when there is nothing to fit on. Blocking
    and CPU-bound by construction: never call it on an event loop.
    """
    requested_leads = tuple(sorted({int(lead) for lead in options.leads}))
    thresholds = tuple(sorted({int(t) for t in options.thresholds}))

    rows, file_leads, counts = load_decisions(
        options.decisions_dirs, leads_min=(), log=log,
    )
    if log:
        log(
            f"loaded {len(rows)} unique decision rows from {counts['files']}"
            f" file(s) ({counts['duplicates']} duplicate key(s),"
            f" {counts['skipped']} file(s) skipped)"
        )
    if not rows:
        raise SweepError("no decision rows found")

    leads = tuple(lead for lead in requested_leads if lead in file_leads)
    missing = [lead for lead in requested_leads if lead not in file_leads]
    if missing and log:
        log(
            "skipping lead(s) with no column in the rows: "
            + ", ".join(str(lead) for lead in missing)
        )
    if not leads:
        raise SweepError(
            "none of the requested leads has a p_rain_<lead> column",
        )

    tracks, frames = build_tracks(
        rows, leads, coverage_gap_min=options.coverage_gap_min,
    )
    # Everything downstream reads the compact tracks, so the row dicts —
    # the biggest object in the process, and the one a forked worker would
    # otherwise page in — are released before the grid runs.
    n_rows = len(rows)
    del rows
    station_ids = sorted(tracks)
    if not station_ids:
        raise SweepError("no decision row carries a radar_ts")
    stamps = [record[_RADAR_TS] for track in tracks.values() for record in track]
    days_by_station = {
        station: {record[_GENERATED].date() for record in track}
        for station, track in tracks.items()
    }
    days = set().union(*days_by_station.values()) if days_by_station else set()
    if log:
        log(
            f"{len(station_ids)} station(s), {len(days)} day(s), leads "
            + ",".join(str(lead) for lead in leads)
        )

    window_from, window_to = min(stamps), max(stamps)
    del stamps
    onsets_by_station, known_until, known_slots = gauge_truth(
        Path(options.corpus_dir), station_ids,
        (window_from, window_to), dry_min=options.dry_min, log=log,
    )
    if known_slots == 0:
        raise SweepError("the gauge store has no observations over this window")
    # A station the gauge store says nothing about cannot verify anything:
    # every warning there would be a false alarm by default, which measures
    # the archive's depth rather than the rule. Leave it out of the pool.
    scored_stations = [s for s in station_ids if s in known_until]
    dropped = len(station_ids) - len(scored_stations)
    if dropped and log:
        log(f"{dropped} station(s) have no gauge observations; left unscored")
    if not scored_stations:
        raise SweepError("no station has gauge observations")

    station_days = sum(len(days_by_station[s]) for s in scored_stations)
    coverage = {
        lead: {
            station: coverage_runs(
                frames[station],
                max_gap_min=options.coverage_gap_min,
                extend_min=lead + options.tolerance_min,
            )
            for station in scored_stations
        }
        for lead in leads
    }
    shared = {
        "leads": list(leads),
        "stations": scored_stations,
        "tracks": {s: tracks[s] for s in scored_stations},
        "onsets": onsets_by_station,
        "known_until": known_until,
        "coverage": coverage,
        "tolerance_min": int(options.tolerance_min),
        "dry_min": int(options.dry_min),
        "min_useful_lead_min": float(options.min_useful_lead_min),
        "persistence_obs": int(options.persistence_obs),
        "rearm_after_min": int(options.rearm_after_min),
        "raining_now_mm_h": RAIN_THRESHOLD_MM_H,
        "station_days": station_days,
        "n_days": len(days),
        "n_rows": n_rows,
    }
    del tracks, frames

    cells, do_nothing = run_sweep(
        shared, leads, thresholds, workers=int(options.workers), log=log,
    )
    del shared
    release_shared()

    radar_cells: list[dict] | None = None
    radar_info: dict | None = None
    if options.radar_decisions_dirs:
        if log:
            log("sweeping the radar cross-check set")
        radar_cells, radar_info = radar_sweep(
            options.radar_decisions_dirs, leads, thresholds,
            coverage_gap_min=options.coverage_gap_min,
            tolerance_min=options.tolerance_min,
            dry_min=options.dry_min,
            persistence_obs=options.persistence_obs,
            rearm_after_min=options.rearm_after_min,
            min_useful_lead_min=float(options.min_useful_lead_min),
            workers=int(options.workers),
            log=log,
        )
        if log:
            log(
                f"radar set: {radar_info['points']} point(s), "
                f"{radar_info['rows']} row(s), {radar_info['onsets']} onset(s), "
                f"{len(radar_cells)} cell(s)"
            )
        if not radar_cells:
            radar_cells = None

    picks = build_picks(
        cells, leads,
        far_cap=float(options.far_cap),
        plateau_frac=float(options.plateau_frac),
        min_warnings=int(options.min_warnings),
        radar_cells=radar_cells,
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "settings": {
            "decisions_dirs": [str(d) for d in options.decisions_dirs],
            "corpus_dir": str(options.corpus_dir),
            "leads_requested": list(requested_leads),
            "leads_used": list(leads),
            "leads_in_rows": list(file_leads),
            "thresholds": list(thresholds),
            "persistence_obs": int(options.persistence_obs),
            "rearm_after_min": int(options.rearm_after_min),
            "raining_now_mm_h": RAIN_THRESHOLD_MM_H,
            "tolerance_min": int(options.tolerance_min),
            "dry_min": int(options.dry_min),
            "coverage_gap_min": int(options.coverage_gap_min),
            "far_cap": float(options.far_cap),
            "min_useful_lead_min": float(options.min_useful_lead_min),
            "plateau_frac": float(options.plateau_frac),
            "min_warnings": int(options.min_warnings),
            "fallback_threshold_pct": int(options.fallback_threshold_pct),
            "radar_decisions_dirs": [
                str(d) for d in (options.radar_decisions_dirs or ())
            ],
            "quiet_hours": False,
        },
        "window": {
            "from": window_from.isoformat(),
            "to": window_to.isoformat(),
            "days": len(days),
            "stations": len(station_ids),
            "stations_scored": len(scored_stations),
            "rows": n_rows,
            "station_days": station_days,
            "known_gauge_slots": known_slots,
            "files": counts["files"],
            "duplicate_keys": counts["duplicates"],
        },
        "leads": list(leads),
        "thresholds": list(thresholds),
        "cells": cells,
        "do_nothing": do_nothing,
        "picks": picks,
        "radar": None if radar_info is None else {
            **radar_info, "cells": radar_cells or [],
        },
        "current_rule": {
            "lead_min": CURRENT_LEAD_MIN,
            "threshold_pct": CURRENT_THRESHOLD_PCT,
            "cell": _find(cells, CURRENT_LEAD_MIN, CURRENT_THRESHOLD_PCT),
        },
    }

    thresholds_doc = build_thresholds_document(
        payload, fallback_threshold_pct=int(options.fallback_threshold_pct),
    )
    problems = validate_thresholds(thresholds_doc)
    if log:
        for problem in problems:
            log(f"thresholds schema: {problem}")
    payload["thresholds"] = thresholds_doc
    payload["thresholds_schema_problems"] = problems
    return payload


__all__ = [
    "CSV_COLUMNS",
    "CURRENT_LEAD_MIN",
    "CURRENT_THRESHOLD_PCT",
    "DEFAULT_FAR_CAP",
    "DEFAULT_MIN_WARNINGS",
    "DEFAULT_PLATEAU_FRAC",
    "FIT_MIN_USEFUL_LEAD_MIN",
    "PICK_ROUNDING_PCT",
    "RAIN_THRESHOLD_MM_H",
    "SweepError",
    "SweepOptions",
    "build_picks",
    "build_thresholds_document",
    "build_tracks",
    "decision_parquets",
    "gauge_truth",
    "load_decisions",
    "parse_leads",
    "parse_thresholds",
    "pick_max_csi",
    "pick_max_pod_under_far",
    "pick_plateau",
    "radar_sweep",
    "radar_truth",
    "release_shared",
    "render_markdown",
    "replay_station",
    "round_to",
    "run_fit",
    "run_sweep",
    "score_cell",
    "scored_warnings",
    "write_atomic",
    "write_csv",
]
