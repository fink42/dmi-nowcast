#!/usr/bin/env python3
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

Reading the output
------------------
POD and FAR move in opposite directions across a threshold column and one
number cannot rank them, so two picks are reported per lead:

* **max CSI** — ``hits / (hits + misses + false alarms)``, the usual
  single-number summary of a warning system, ties broken toward the
  higher threshold (fewer notifications for the same skill);
* **max POD subject to FAR ≤ cap** — the product question: given that
  more than ``--far-cap`` of notifications being wrong is not shippable,
  how much rain can still be caught?

Both sit beside the "do nothing" row — no warnings, no false alarms, every
covered onset a miss — which is the floor any rule has to beat.

Usage (on the VM that holds the corpus)::

    python scripts/sweep_thresholds.py \\
        --decisions-dir /var/lib/dmi-nowcast-corpus/stations/replay/decisions \\
        --decisions-dir /var/lib/dmi-nowcast-corpus/stations/eval \\
        --corpus-dir /var/lib/dmi-nowcast-corpus \\
        --leads 10,20,30,45,60 --thresholds 20:80:5 --workers 8 \\
        --out-json sweep.json --out-md sweep.md --out-csv sweep.csv

Later ``--decisions-dir`` wins on a ``(radar_ts, station_id)`` collision:
the live row is the decision the service actually took, the replay's is a
reconstruction of what it would have taken.

Offline and read-only: parquet in, files out. No DMI calls, no network.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
_SIDECAR = _REPO_ROOT / "sidecar"
if str(_SIDECAR) not in sys.path:
    sys.path.insert(0, str(_SIDECAR))

from dmi_nowcast_core.warning_score import (  # noqa: E402
    DEFAULT_COVERAGE_GAP_MIN,
    DEFAULT_DRY_MIN,
    DEFAULT_PRODUCT_LEADS_MIN,
    DEFAULT_TOLERANCE_MIN,
    PRECIP_DUR_PARAM,
    PRECIP_PARAM,
    align_decision_table,
    coverage_runs,
    decision_leads_in,
    gauge_slots,
    p_rain_column,
    pooled_summary,
    score_warnings,
)
from dmi_nowcast_core.warning_score import onsets as gauge_onsets  # noqa: E402

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

#: The columns of the per-cell CSV, in order.
CSV_COLUMNS = (
    "lead_min", "threshold_pct", "warnings", "n_sent", "pending", "hits",
    "false_alarms", "misses", "pending_onsets", "uncovered_onsets",
    "n_onsets", "pod", "far", "csi", "lead_error_p25", "lead_error_p50",
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
    from dmi_nowcast_sidecar.push import engine as decision_engine

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
        ))
    pooled = pooled_summary(results)
    return _cell(pooled, lead, threshold_pct, shared)


def _cell(pooled: dict, lead: int, threshold_pct: int | None, shared: dict) -> dict:
    """A pooled summary, plus the derived numbers the report ranks on."""
    hits = pooled["hits"]
    denom = hits + pooled["misses"] + pooled["false_alarms"]
    station_days = max(1, shared["station_days"])
    spread = pooled["lead_error_min"]
    return {
        "lead_min": int(lead),
        "threshold_pct": None if threshold_pct is None else int(threshold_pct),
        "warnings": pooled["warnings"],
        "n_sent": pooled["n_sent"],
        "pending": pooled["pending"],
        "hits": hits,
        "false_alarms": pooled["false_alarms"],
        "misses": pooled["misses"],
        "pending_onsets": pooled["pending_onsets"],
        "uncovered_onsets": pooled["uncovered_onsets"],
        "n_onsets": pooled["n_onsets"],
        "pod": pooled["pod"],
        "far": pooled["far"],
        "csi": (hits / denom) if denom else None,
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
    lines.append(f"- FAR cap for the second pick: {_pct(settings['far_cap'])}")
    lines.append("")
    lines.append(
        "POD is the share of covered rain onsets a warning caught; FAR the "
        "share of warnings no rain followed; CSI the two together "
        "(`hits / (hits + misses + false alarms)`). Lead error is "
        "`eta − (onset − sent)` in minutes: positive means the rain beat "
        "the ETA and the warning was late. Every column adds up: "
        "`hits + false alarms + pending = sent`, and a pending warning is "
        "one whose window the gauge record does not yet cover, held out of "
        "both rates rather than graded on evidence that does not exist."
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
            "| threshold | sent | hits | false alarms | pending | misses | "
            "POD | FAR | CSI | lead err p50 | sent / station-day |"
        )
        lines.append(
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
            "| ---: | ---: |"
        )
        if nothing:
            lines.append(
                f"| no rule | 0 | 0 | 0 | 0 | {nothing['misses']} | "
                f"{_fmt(nothing['pod'])} | – | {_fmt(nothing['csi'])} | – | 0.00 |"
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
                f"{cell['false_alarms']} | {cell['pending']} | "
                f"{cell['misses']} | "
                f"{_fmt(cell['pod'])} | {_fmt(cell['far'])} | "
                f"{_fmt(cell['csi'])} | "
                f"{_minutes(cell['lead_error_min'].get('p50'))} | "
                f"{cell['warnings_per_station_day']:.2f} |"
            )
        lines.append("")
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


def _write_atomic(path: Path, text: str) -> None:
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
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sweep (lead, threshold) for the push rule against gauges.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--decisions-dir", type=Path, nargs="+", required=True,
                   action="extend", dest="decisions_dirs",
                   help="directory tree of decision parquet files; repeatable, "
                        "later directories win a (radar_ts, station_id) tie")
    p.add_argument("--corpus-dir", type=Path, required=True,
                   help="gauge store root (the directory holding stations/)")
    p.add_argument("--leads", default=",".join(
        str(lead) for lead in DEFAULT_PRODUCT_LEADS_MIN),
        help="comma-separated lead times; leads with no column are skipped")
    p.add_argument("--thresholds", default="20:80:5",
                   help="lo:hi[:step] (inclusive) or a comma-separated list")
    p.add_argument("--rearm-after-min", type=int, default=60)
    p.add_argument("--persistence-obs", type=int, default=1)
    p.add_argument("--tolerance-min", type=int, default=DEFAULT_TOLERANCE_MIN)
    p.add_argument("--dry-min", type=int, default=DEFAULT_DRY_MIN)
    p.add_argument("--coverage-gap-min", type=int,
                   default=DEFAULT_COVERAGE_GAP_MIN,
                   help="a longer gap between frames ends a coverage run and "
                        "restarts every subscription armed")
    p.add_argument("--far-cap", type=float, default=DEFAULT_FAR_CAP,
                   help="the ceiling the second pick's false-alarm rate obeys")
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-md", type=Path, default=None)
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--workers", type=int, default=1,
                   help="processes over (lead, threshold) cells")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    try:
        requested_leads = parse_leads(args.leads)
        thresholds = parse_thresholds(args.thresholds)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.persistence_obs < 1:
        print("error: --persistence-obs must be >= 1", file=sys.stderr)
        return 2
    if not 0.0 < args.far_cap <= 1.0:
        print("error: --far-cap must be in (0, 1]", file=sys.stderr)
        return 2

    rows, file_leads, counts = load_decisions(
        args.decisions_dirs, leads_min=(), log=log,
    )
    log(
        f"loaded {len(rows)} unique decision rows from {counts['files']} file(s)"
        f" ({counts['duplicates']} duplicate key(s), {counts['skipped']} file(s)"
        " skipped)"
    )
    if not rows:
        print("error: no decision rows found", file=sys.stderr)
        return 2

    leads = tuple(lead for lead in requested_leads if lead in file_leads)
    missing = [lead for lead in requested_leads if lead not in file_leads]
    if missing:
        log(
            "skipping lead(s) with no column in the rows: "
            + ", ".join(str(lead) for lead in missing)
        )
    if not leads:
        print(
            "error: none of the requested leads has a p_rain_<lead> column",
            file=sys.stderr,
        )
        return 2

    tracks, frames = build_tracks(
        rows, leads, coverage_gap_min=args.coverage_gap_min,
    )
    # Everything downstream reads the compact tracks, so the row dicts —
    # the biggest object in the process, and the one a forked worker would
    # otherwise page in — are released before the grid runs.
    n_rows = len(rows)
    del rows
    station_ids = sorted(tracks)
    if not station_ids:
        print("error: no decision row carries a radar_ts", file=sys.stderr)
        return 2
    stamps = [record[_RADAR_TS] for track in tracks.values() for record in track]
    days_by_station = {
        station: {record[_GENERATED].date() for record in track}
        for station, track in tracks.items()
    }
    days = set().union(*days_by_station.values()) if days_by_station else set()
    log(
        f"{len(station_ids)} station(s), {len(days)} day(s), leads "
        + ",".join(str(lead) for lead in leads)
    )

    window_from, window_to = min(stamps), max(stamps)
    del stamps
    onsets_by_station, known_until, known_slots = gauge_truth(
        Path(args.corpus_dir), station_ids,
        (window_from, window_to), dry_min=args.dry_min, log=log,
    )
    if known_slots == 0:
        print(
            "error: the gauge store has no observations over this window",
            file=sys.stderr,
        )
        return 2
    # A station the gauge store says nothing about cannot verify anything:
    # every warning there would be a false alarm by default, which measures
    # the archive's depth rather than the rule. Leave it out of the pool.
    scored_stations = [s for s in station_ids if s in known_until]
    dropped = len(station_ids) - len(scored_stations)
    if dropped:
        log(f"{dropped} station(s) have no gauge observations; left unscored")
    if not scored_stations:
        print("error: no station has gauge observations", file=sys.stderr)
        return 2

    station_days = sum(len(days_by_station[s]) for s in scored_stations)
    coverage = {
        lead: {
            station: coverage_runs(
                frames[station],
                max_gap_min=args.coverage_gap_min,
                extend_min=lead + args.tolerance_min,
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
        "tolerance_min": int(args.tolerance_min),
        "dry_min": int(args.dry_min),
        "persistence_obs": int(args.persistence_obs),
        "rearm_after_min": int(args.rearm_after_min),
        "raining_now_mm_h": RAIN_THRESHOLD_MM_H,
        "station_days": station_days,
        "n_days": len(days),
        "n_rows": n_rows,
    }
    del tracks, frames

    cells, do_nothing = run_sweep(
        shared, leads, thresholds, workers=int(args.workers), log=log,
    )
    picks = {
        str(lead): {
            "max_csi": pick_max_csi(
                [c for c in cells if c["lead_min"] == lead]
            ),
            "max_pod_far_capped": pick_max_pod_under_far(
                [c for c in cells if c["lead_min"] == lead], float(args.far_cap),
            ),
        }
        for lead in leads
    }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "settings": {
            "decisions_dirs": [str(d) for d in args.decisions_dirs],
            "corpus_dir": str(args.corpus_dir),
            "leads_requested": list(requested_leads),
            "leads_used": list(leads),
            "leads_in_rows": list(file_leads),
            "thresholds": list(thresholds),
            "persistence_obs": int(args.persistence_obs),
            "rearm_after_min": int(args.rearm_after_min),
            "raining_now_mm_h": RAIN_THRESHOLD_MM_H,
            "tolerance_min": int(args.tolerance_min),
            "dry_min": int(args.dry_min),
            "coverage_gap_min": int(args.coverage_gap_min),
            "far_cap": float(args.far_cap),
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
        "current_rule": {
            "lead_min": CURRENT_LEAD_MIN,
            "threshold_pct": CURRENT_THRESHOLD_PCT,
            "cell": _find(cells, CURRENT_LEAD_MIN, CURRENT_THRESHOLD_PCT),
        },
    }

    if args.out_json:
        _write_atomic(
            args.out_json, json.dumps(payload, indent=1, default=str) + "\n",
        )
        log(f"wrote {args.out_json}")
    if args.out_md:
        _write_atomic(args.out_md, render_markdown(payload))
        log(f"wrote {args.out_md}")
    if args.out_csv:
        write_csv(Path(args.out_csv), cells, do_nothing)
        log(f"wrote {args.out_csv}")

    log(f"swept {len(cells)} cell(s) in {time.time() - started:.1f}s")
    print(json.dumps({
        "window": payload["window"],
        "picks": {
            lead: {
                name: None if cell is None else {
                    "threshold_pct": cell["threshold_pct"],
                    "pod": cell["pod"],
                    "far": cell["far"],
                    "csi": cell["csi"],
                    "warnings": cell["n_sent"],
                }
                for name, cell in per_lead.items()
            }
            for lead, per_lead in picks.items()
        },
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
