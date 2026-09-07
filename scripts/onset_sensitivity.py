#!/usr/bin/env python3
"""How much of the push rule's miss count is the onset DEFINITION?

The first threshold sweep (``scripts/sweep_thresholds.py``) fitted a
threshold per horizon and came back with recall near 0.2 at every lead.
Before that number is read as "the nowcast misses four rain events in
five", two arithmetic facts have to be taken off the table:

1. **The gauge onset rule is generous.** An onset is the first wet slot
   after 30 dry minutes, and a slot is wet at 0.1 mm *or* one minute of
   precipitation. That yields roughly two onsets per station-day — a
   drizzle burst at 09:10 and another at 09:50 are two separate events to
   the scorer, and the second one is a "miss" the moment the first is
   warned about.
2. **The rule can only warn once an hour.** ``push.engine`` disarms on
   every notification and re-arms only after 60 continuous minutes below
   threshold. A second onset inside that hour was never catchable: no
   threshold, no horizon and no better forecast could have produced a
   second notification. Those misses are *shadowed* — they are the price
   of the anti-spam rule, not evidence about the forecast.

This script quantifies both, on the SAME decision rows and the SAME
warnings the sweep produced. Nothing is re-fitted: each lead is replayed
once at the threshold the sweep picked, and that one fixed set of
warnings is then scored against five different onset definitions.

The variants
------------

======  ==========  ===============================  ==========================
name    dry spell   onset amount                     what it asks
======  ==========  ===============================  ==========================
V0      30 min      none (the wet rule alone)        today's definition
V1      60 min      none                             …with the re-arm's clock
V2      60 min      ≥ 0.2 mm over two slots          rain, not drizzle
V3      60 min      ≥ 0.5 mm over two slots          rain worth a notification
V4      120 min     ≥ 0.2 mm over two slots          a new event, not a shower
======  ==========  ===============================  ==========================

Every variant keeps the shipped WET rule (``≥ 0.1 mm`` or ``≥ 1 min``)
for deciding which slots are dry, because that is what certifies a dry
spell; they differ in how long the dry spell must be and in how much rain
the onset itself must deliver. The amount is summed over the onset slot
AND the following one, because DMI's 10-minute bins cut a shower in half
as often as not, and a first slot of 0.1 mm followed by 0.4 mm is one
event of 0.5 mm rather than a drizzle.

A candidate that fails the amount test is dropped as an onset but still
resets the dry run — it rained, so the next slot is not the start of a
new event either.

The shadow
----------

For every (variant, lead) the scoring is exactly the sweep's — the same
``score_warnings`` with the same tolerance, coverage runs, ``known_until``
and five-minute useful lead. On top of it, each missed onset is asked one
extra question: *was there a HIT warning within the previous
``--rearm-after-min`` minutes?* If there was, the rule was disarmed and no
second notification was possible, so the miss says nothing about the
forecast. The count, its share of the misses, and

    recall excluding shadowed = hits / (hits + misses + late − shadowed)

are reported beside the ordinary recall. That is a CONSERVATIVE shadow:
the engine re-arms after 60 minutes *below threshold*, so a subscription
sitting in continuous rain stays disarmed longer than the window counted
here.

Offline and read-only: parquet in, a markdown and a JSON out. No DMI
calls, no network, no radar, no STEPS — the engine replay is
``threshold_sweep.replay_station``, the scoring is
``warning_score.score_warnings``, and neither is reimplemented here.

Usage::

    python scripts/onset_sensitivity.py \\
        --corpus-dir /var/lib/dmi-nowcast-corpus \\
        --decisions-dir /var/lib/dmi-nowcast-corpus/stations/replay/decisions \\
        --decisions-dir /var/lib/dmi-nowcast-corpus/stations/eval \\
        --picks push_thresholds.json \\
        --out-md onset_sensitivity.md --out-json onset_sensitivity.json \\
        --workers 8
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
_SIDECAR = _REPO_ROOT / "sidecar"
if str(_SIDECAR) not in sys.path:
    sys.path.insert(0, str(_SIDECAR))

from dmi_nowcast_core.push_thresholds import (  # noqa: E402
    effective_threshold,
    load_thresholds,
)
from dmi_nowcast_core.warning_score import (  # noqa: E402
    DEFAULT_COVERAGE_GAP_MIN,
    DEFAULT_TOLERANCE_MIN,
    PRECIP_DUR_PARAM,
    PRECIP_PARAM,
    SLOT_MIN,
    WET_DUR_MIN,
    WET_PRECIP_MM,
    ScoreResult,
    pooled_summary,
    score_warnings,
    slot_end_of,
)
from dmi_nowcast_sidecar.threshold_sweep import (  # noqa: E402
    FIT_MIN_USEFUL_LEAD_MIN,
    GAUGE_PAD_MIN,
    RAIN_THRESHOLD_MM_H,
    SweepError,
    build_shared,
    build_tracks,
    load_decisions,
    replay_station,
    write_atomic,
)
# Private, and deliberately so: the two track-record indices are the
# layout ``build_tracks`` writes, which must not be guessed at a second
# time.
from dmi_nowcast_sidecar.threshold_sweep import (  # noqa: E402
    _GENERATED,
    _RADAR_TS,
)

__all__ = [
    "Variant",
    "VARIANTS",
    "MM_BUCKETS",
    "slot_amounts",
    "variant_onsets",
    "gauge_truth_variants",
    "shadowed_misses",
    "score_variant_lead",
    "render_markdown",
    "run",
    "main",
]


# ---------------------------------------------------------------------------
# The variants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """One onset definition: how long the dry spell, how much the rain.

    ``min_mm_two_slots`` of ``None`` means no amount test at all — the wet
    rule alone decides, which is what ships today.
    """

    name: str
    dry_min: int
    min_mm_two_slots: float | None
    note: str

    @property
    def label(self) -> str:
        amount = (
            "wet rule only" if self.min_mm_two_slots is None
            else f"≥ {self.min_mm_two_slots:g} mm / 2 slots"
        )
        return f"dry ≥ {self.dry_min} min, {amount}"

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "dry_min": int(self.dry_min),
            "min_mm_two_slots": self.min_mm_two_slots,
            "label": self.label,
            "note": self.note,
        }


VARIANTS: tuple[Variant, ...] = (
    Variant("V0", 30, None, "the shipped definition the sweep was fitted on"),
    Variant("V1", 60, None, "same wet rule, dry spell matched to the re-arm"),
    Variant("V2", 60, 0.2, "an onset must deliver 0.2 mm over two slots"),
    Variant("V3", 60, 0.5, "an onset must deliver 0.5 mm over two slots"),
    Variant("V4", 120, 0.2, "a new rain event rather than the next shower"),
)

#: Buckets for "how much rain did this onset actually deliver", in mm over
#: the onset slot and the one after it. The edges are V2's and V3's
#: thresholds, so the histogram reads directly as "what V2 drops" and
#: "what V3 keeps".
MM_BUCKETS: tuple[str, ...] = ("<0.2", "0.2-0.5", ">=0.5")


def _bucket_of(mm: float) -> str:
    if mm < 0.2:
        return MM_BUCKETS[0]
    if mm < 0.5:
        return MM_BUCKETS[1]
    return MM_BUCKETS[2]


def _empty_buckets() -> dict[str, int]:
    return {name: 0 for name in MM_BUCKETS}


# ---------------------------------------------------------------------------
# Gauge slots, with the amounts kept
# ---------------------------------------------------------------------------


def _amount(value: Any) -> float | None:
    """A gauge reading, with DMI's trace sentinel folded to 0.0.

    The same rule ``warning_score._amount_mm`` applies — a negative value
    is "traces, below 0.1 mm", never a negative depth — restated here
    because this script needs the number itself and not just whether it
    crossed a threshold.
    """
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return 0.0 if out < 0.0 else out


def slot_amounts(
    rows: Iterable[Mapping[str, Any]],
    *,
    start_utc: datetime,
    end_utc: datetime,
    slot_min: int = SLOT_MIN,
) -> list[tuple[datetime, float | None, float | None]]:
    """One station's 10-min grid as ``(slot_end, mm, dur_min)``.

    The amount-carrying sibling of ``warning_score.gauge_slots``: the same
    contiguous grid over the same slot ends, the same trace handling, and
    the same "a slot nobody reported is unknown, never dry" rule — but the
    values survive instead of collapsing to a boolean, because the
    variants threshold them at more than one level.

    ``rows`` are ONE station's observation rows (``observed_utc`` /
    ``parameter_id`` / ``value``); the caller groups by station, which is
    what keeps a month of a hundred stations from being rescanned once per
    station. A slot with several rows for one parameter takes the largest,
    matching the OR the boolean version does over its arms.

    A slot is *known* when either parameter reported a usable value; a
    slot with only a duration is known, and its ``mm`` stays ``None``
    (which the amount tests read as contributing nothing, not as zero
    evidence — see :func:`variant_onsets`).
    """
    if slot_min <= 0:
        raise ValueError("slot_min must be positive")
    amounts: dict[datetime, float] = {}
    durations: dict[datetime, float] = {}
    for row in rows:
        observed = row.get("observed_utc")
        if observed is None:
            continue
        param = str(row.get("parameter_id"))
        if param not in (PRECIP_PARAM, PRECIP_DUR_PARAM):
            continue
        value = _amount(row.get("value"))
        if value is None:
            continue
        slot = slot_end_of(observed, slot_min=slot_min)
        target = amounts if param == PRECIP_PARAM else durations
        previous = target.get(slot)
        if previous is None or value > previous:
            target[slot] = value

    first = slot_end_of(start_utc, slot_min=slot_min)
    last = slot_end_of(end_utc, slot_min=slot_min)
    if last < first:
        return []
    step = timedelta(minutes=slot_min)
    out: list[tuple[datetime, float | None, float | None]] = []
    cursor = first
    while cursor <= last:
        out.append((cursor, amounts.get(cursor), durations.get(cursor)))
        cursor += step
    return out


def _depth(mm: float | None) -> float:
    """Millimetres this slot contributes to an onset amount.

    An unreported depth contributes nothing — a duration-only slot
    measured no depth — and so does DMI's trace sentinel: a negative
    reading is "below 0.1 mm", never a negative amount, and summing it
    would let a trace *subtract* from the shower it belongs to.
    :func:`slot_amounts` folds the sentinel already; this repeats the fold
    so a grid built by hand, or by some other reader, cannot smuggle one
    through.
    """
    if mm is None or mm < 0.0:
        return 0.0
    return float(mm)


def _is_wet(mm: float | None, dur: float | None) -> bool | None:
    """The shipped wet rule over one slot: ``True`` / ``False`` / unknown."""
    if mm is None and dur is None:
        return None
    return (mm is not None and mm >= WET_PRECIP_MM) or (
        dur is not None and dur >= WET_DUR_MIN
    )


def variant_onsets(
    grid: Sequence[tuple[datetime, float | None, float | None]],
    variant: Variant,
    *,
    slot_min: int = SLOT_MIN,
) -> list[tuple[datetime, float]]:
    """``[(onset instant, mm over the onset slot and the next)]``.

    The dry-run bookkeeping is ``warning_score.onsets``' own, restated over
    the amount-carrying grid: ``ceil(dry_min / slot_min)`` consecutive
    KNOWN dry slots must precede a wet one, an unknown slot resets the run
    rather than extending it, and the first slots of a record can never be
    onsets because nothing is known about what came before them. With
    ``dry_min=30`` and no amount test the result is exactly what the
    shipped rule produces — asserted in the tests.

    The amount test then drops the candidates that did not deliver: the
    onset slot's ``mm`` plus the next slot's, an absent value and a trace
    sentinel alike counting as zero millimetres (:func:`_depth`). A dropped
    candidate still resets the dry run — it rained, so the slot after it
    is not the start of a new event either.
    """
    need = max(1, math.ceil(variant.dry_min / slot_min))
    floor = variant.min_mm_two_slots
    step = timedelta(minutes=slot_min)
    out: list[tuple[datetime, float]] = []
    dry_run = 0
    previous: datetime | None = None
    for index, (ts, mm, dur) in enumerate(grid):
        if previous is not None and ts - previous != step:
            dry_run = 0  # a hole in the grid is not a dry spell
        previous = ts
        wet = _is_wet(mm, dur)
        if wet is None:
            dry_run = 0
            continue
        if not wet:
            dry_run += 1
            continue
        if dry_run >= need:
            total = _depth(mm)
            if index + 1 < len(grid):
                total += _depth(grid[index + 1][1])
            if floor is None or total >= floor:
                out.append((ts, total))
        dry_run = 0
    return out


def gauge_truth_variants(
    corpus_dir: Path,
    station_ids: Sequence[str],
    window: tuple[datetime, datetime],
    *,
    variants: Sequence[Variant] = VARIANTS,
    log=None,
) -> tuple[dict[str, dict[str, dict[datetime, float]]], dict[str, datetime], int]:
    """Every variant's onsets, in one pass over the gauge archive.

    Returns ``(onsets[variant][station][instant] -> two-slot mm,
    known_until[station], known slot count)``.

    The archive is read once, vectorised
    (``warning_score.gauge_truth_vectorised``), and each variant is then a
    boolean run-length pass over the grid already in memory — five onset
    definitions for the price of one read, where the row-at-a-time version
    below paid for a month of Python dict work per station per month.

    ``slot_amounts`` and :func:`variant_onsets` stay as the reference the
    vectorised derivation is tested against: the amount test, the dry-run
    reset on a failed candidate and the trace fold are stated there in
    plain Python and asserted equal to the array version.
    """
    from dmi_nowcast_core.warning_score import gauge_truth_vectorised

    pad = timedelta(minutes=GAUGE_PAD_MIN)
    start, end = window
    loaded = gauge_truth_vectorised(
        Path(corpus_dir), start - pad, end + pad, list(station_ids),
        pad_min=GAUGE_PAD_MIN, log=log,
    )
    found: dict[str, dict[str, dict[datetime, float]]] = {
        variant.name: {
            station: dict(pairs)
            for station, pairs in loaded.onsets_for(
                variant.dry_min, min_mm_two_slots=variant.min_mm_two_slots,
            ).items()
        }
        for variant in variants
    }
    if log:
        log(
            f"gauge truth: {loaded.known_slots} known slot(s) over "
            f"{len(loaded.known_until)} reporting station(s); "
            + ", ".join(
                f"{v.name} {sum(len(s) for s in found[v.name].values())}"
                for v in variants
            )
            + " onset(s)"
        )
    return found, loaded.known_until, loaded.known_slots


# ---------------------------------------------------------------------------
# The re-arm shadow
# ---------------------------------------------------------------------------


def shadowed_misses(
    result: ScoreResult,
    *,
    rearm_after_min: int = 60,
    outcomes: Sequence[str] = ("hit",),
) -> int:
    """Missed onsets that fell inside a warning's disarmed hour.

    A notification disarms the subscription, so no second one could be
    sent for ``rearm_after_min`` minutes afterwards. A miss inside that
    window was not catchable under this rule at any threshold — it is the
    anti-spam rule's cost, not the forecast's error.

    ``outcomes`` selects which warnings cast a shadow. The headline number
    counts only HITS, because a hit is a warning the user demonstrably
    received about rain that demonstrably came: the second onset behind it
    is a bookkeeping artefact. Passing every graded outcome instead
    answers the wider question — how many misses fell behind ANY
    notification — and is reported beside it.

    Conservative by construction: the engine re-arms after 60 minutes
    *below threshold*, so a point in continuous rain stays disarmed longer
    than this window.
    """
    allowed = frozenset(outcomes)
    sent = sorted(
        warning.sent_utc for warning in result.warnings
        if warning.outcome in allowed
    )
    if not sent:
        return 0
    window = timedelta(minutes=float(rearm_after_min))
    count = 0
    for onset in result.onsets:
        if onset.outcome != "miss":
            continue
        index = bisect_left(sent, onset.onset_utc)
        if index == 0:
            continue
        previous = sent[index - 1]
        if previous < onset.onset_utc <= previous + window:
            count += 1
    return count


# ---------------------------------------------------------------------------
# One (variant, lead) cell
# ---------------------------------------------------------------------------


def score_variant_lead(
    shared: Mapping[str, Any],
    variant: Variant,
    lead: int,
    warnings_by_station: Mapping[str, Sequence[tuple[datetime, float | None]]],
    onsets_by_station: Mapping[str, Mapping[datetime, float]],
    *,
    rearm_after_min: int = 60,
) -> dict:
    """Score one fixed warning set against one onset definition.

    Every station goes through ``warning_score.score_warnings`` with the
    sweep's own parameters — the coverage runs from ``build_shared``, that
    station's ``known_until``, the 10-minute tolerance and the five-minute
    useful lead — and the stations are pooled by ``pooled_summary``, so a
    cell here is comparable line for line with a cell of the sweep.

    The do-nothing row is the same scoring with an empty warning list: it
    is the miss count of a service that never warns at all, and therefore
    the denominator every recall on this row is measured against.
    """
    coverage = shared["coverage"][int(lead)]
    known_until = shared["known_until"]
    results: list[ScoreResult] = []
    do_nothing: list[ScoreResult] = []
    shadow = 0
    shadow_any = 0
    buckets = _empty_buckets()
    for station in shared["stations"]:
        onsets = onsets_by_station.get(station, {})
        instants = sorted(onsets)
        common = {
            "lead_min": int(lead),
            "tolerance_min": shared["tolerance_min"],
            "dry_min": variant.dry_min,
            "known_until": known_until.get(station),
            "coverage": coverage.get(station, ()),
            "min_useful_lead_min": shared["min_useful_lead_min"],
        }
        result = score_warnings(
            warnings_by_station.get(station, ()), instants, **common,
        )
        results.append(result)
        do_nothing.append(score_warnings((), instants, **common))
        shadow += shadowed_misses(result, rearm_after_min=rearm_after_min)
        shadow_any += shadowed_misses(
            result, rearm_after_min=rearm_after_min,
            outcomes=("hit", "late", "false_alarm"),
        )
        for row in result.onsets:
            if row.outcome in ("hit", "miss_late", "miss"):
                buckets[_bucket_of(onsets.get(row.onset_utc, 0.0))] += 1

    pooled = pooled_summary(results)
    nothing = pooled_summary(do_nothing)
    station_days = max(1, int(shared["station_days"]))
    hits = pooled["hits"]
    misses = pooled["misses"]
    late = pooled["late"]
    covered = hits + misses + late
    unshadowed = covered - shadow
    return {
        "variant": variant.name,
        "dry_min": variant.dry_min,
        "min_mm_two_slots": variant.min_mm_two_slots,
        "lead_min": int(lead),
        "threshold_pct": int(shared["thresholds"][int(lead)]),
        "n_onsets": pooled["n_onsets"],
        "covered_onsets": covered,
        "uncovered_onsets": pooled["uncovered_onsets"],
        "pending_onsets": pooled["pending_onsets"],
        "onsets_per_station_day": covered / station_days,
        "warnings": pooled["warnings"],
        "n_sent": pooled["n_sent"],
        "pending": pooled["pending"],
        "hits": hits,
        "false_alarms": pooled["false_alarms"],
        "late": late,
        "misses": misses,
        "precision": pooled["precision"],
        "recall": pooled["recall"],
        "f1": pooled["f1"],
        "far": pooled["far"],
        "csi": pooled["csi"],
        "do_nothing_misses": nothing["misses"],
        "shadowed_misses": shadow,
        "shadow_share": (shadow / misses) if misses else None,
        "shadowed_by_any_warning": shadow_any,
        "recall_excl_shadowed": (hits / unshadowed) if unshadowed > 0 else None,
        "mm_buckets": buckets,
        "station_days": station_days,
        "n_stations": len(shared["stations"]),
    }


# ---------------------------------------------------------------------------
# Running it: replay once per lead, then score every variant
# ---------------------------------------------------------------------------

#: Set once per worker process; under ``fork`` the children inherit it and
#: under ``spawn`` the initializer plants it. Same shape as the sweep's.
_SHARED: dict | None = None


def _init_worker(shared: dict) -> None:
    global _SHARED
    _SHARED = shared


def _pool_context():
    if "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return mp.get_context("spawn")


def _replay_worker(lead: int) -> tuple[int, dict[str, list]]:
    if _SHARED is None:  # pragma: no cover — a pool wired without a payload
        raise RuntimeError("worker started without a shared payload")
    return lead, _replay_lead(_SHARED, lead)


def _replay_lead(shared: Mapping[str, Any], lead: int) -> dict[str, list]:
    """One lead's warnings per station, at that lead's fitted threshold."""
    lead_index = list(shared["leads"]).index(int(lead))
    threshold = int(shared["thresholds"][int(lead)])
    return {
        station: replay_station(
            shared["tracks"][station],
            lead_index,
            threshold,
            persistence_obs=shared["persistence_obs"],
            rearm_after_min=shared["rearm_after_min"],
            raining_now_mm_h=shared["raining_now_mm_h"],
        )
        for station in shared["stations"]
    }


def _score_worker(task: tuple[int, int]) -> dict:
    if _SHARED is None:  # pragma: no cover — a pool wired without a payload
        raise RuntimeError("worker started without a shared payload")
    variant_index, lead = task
    variant = _SHARED["variants"][variant_index]
    return score_variant_lead(
        _SHARED,
        variant,
        lead,
        _SHARED["warnings"][lead],
        _SHARED["variant_onsets"][variant.name],
        rearm_after_min=_SHARED["rearm_after_min"],
    )


def _map(tasks, worker, shared: dict, workers: int, log=None):
    """Run ``worker`` over ``tasks``, in this process or in a pool."""
    global _SHARED
    _SHARED = shared
    if workers <= 1:
        for task in tasks:
            yield worker(task)
        return
    ctx = _pool_context()
    kwargs: dict[str, Any] = {}
    if ctx.get_start_method() != "fork":
        kwargs = {"initializer": _init_worker, "initargs": (shared,)}
    with ProcessPoolExecutor(
        max_workers=int(workers), mp_context=ctx, **kwargs,
    ) as pool:
        for out in pool.map(worker, tasks, chunksize=1):
            yield out
    if log:
        log(f"pool of {workers} finished {len(tasks)} task(s)")


def run(
    *,
    corpus_dir: Path,
    decisions_dirs: Sequence[Path],
    picks_path: Path,
    leads: Sequence[int] | None = None,
    tolerance_min: int = DEFAULT_TOLERANCE_MIN,
    coverage_gap_min: int = DEFAULT_COVERAGE_GAP_MIN,
    min_useful_lead_min: float = FIT_MIN_USEFUL_LEAD_MIN,
    persistence_obs: int = 1,
    rearm_after_min: int = 60,
    variants: Sequence[Variant] = VARIANTS,
    workers: int = 1,
    log=None,
) -> dict:
    """The whole analysis: rows in, one payload out.

    Raises :class:`SweepError` on the same three "nothing to measure here"
    conditions the fit raises on — no decision rows, no lead with a
    probability column, no gauge observation over the window.
    """
    picks_doc = load_thresholds(picks_path)
    if picks_doc is None:
        raise SweepError(f"no usable thresholds document at {picks_path}")
    wanted = (
        sorted(int(lead) for lead in picks_doc.get("leads", {}))
        if leads is None else sorted({int(lead) for lead in leads})
    )
    if not wanted:
        raise SweepError("the picks document names no lead")

    rows, file_leads, counts = load_decisions(decisions_dirs, leads_min=(), log=log)
    if not rows:
        raise SweepError("no decision rows found")
    if log:
        log(
            f"loaded {len(rows)} unique decision rows from {counts['files']}"
            f" file(s) ({counts['duplicates']} duplicate key(s))"
        )
    used = tuple(lead for lead in wanted if lead in file_leads)
    if not used:
        raise SweepError("none of the picked leads has a p_rain_<lead> column")
    thresholds = {lead: effective_threshold(picks_doc, lead) for lead in used}

    tracks = build_tracks(rows, used, coverage_gap_min=coverage_gap_min)[0]
    n_rows = len(rows)
    del rows
    station_ids = sorted(tracks)
    if not station_ids:
        raise SweepError("no decision row carries a radar_ts")
    stamps = [record[_RADAR_TS] for track in tracks.values() for record in track]
    window_from, window_to = min(stamps), max(stamps)
    del stamps
    days = {
        record[_GENERATED].date()
        for track in tracks.values() for record in track
    }
    if log:
        log(
            f"{len(station_ids)} station(s), {len(days)} day(s), leads "
            + ", ".join(f"{lead}@{thresholds[lead]}%" for lead in used)
        )

    onsets, known_until, known_slots = gauge_truth_variants(
        Path(corpus_dir), station_ids, (window_from, window_to),
        variants=variants, log=log,
    )
    if known_slots == 0:
        raise SweepError("the gauge store has no observations over this window")
    scored_stations = [s for s in station_ids if s in known_until]
    if not scored_stations:
        raise SweepError("no station has gauge observations")
    if log and len(scored_stations) != len(station_ids):
        log(
            f"{len(station_ids) - len(scored_stations)} station(s) have no "
            "gauge observations; left unscored"
        )

    shared = build_shared(
        {s: tracks[s] for s in scored_stations}, scored_stations, used,
        onsets={},
        known_until=known_until,
        coverage_gap_min=coverage_gap_min,
        tolerance_min=tolerance_min,
        dry_min=VARIANTS[0].dry_min,
        min_useful_lead_min=min_useful_lead_min,
        persistence_obs=persistence_obs,
        rearm_after_min=rearm_after_min,
        n_days=len(days),
        n_rows=n_rows,
    )
    del tracks
    shared["thresholds"] = thresholds
    # ``build_shared`` carries the onsets of ONE definition; this analysis
    # swaps five in and out, so the per-variant sets ride beside it and
    # ``shared["onsets"]`` stays deliberately empty.
    shared["variant_onsets"] = onsets
    shared["variants"] = list(variants)

    started = time.time()
    warnings: dict[int, dict[str, list]] = {}
    for lead, per_station in _map(
        list(used), _replay_worker, shared, workers, log=log,
    ):
        warnings[lead] = per_station
        if log:
            total = sum(len(v) for v in per_station.values())
            log(
                f"lead {lead} @ {thresholds[lead]}%: {total} warning(s) "
                f"in {time.time() - started:.0f}s"
            )
    shared["warnings"] = warnings

    tasks = [
        (index, lead)
        for index, _variant in enumerate(variants)
        for lead in used
    ]
    cells: list[dict] = []
    for cell in _map(tasks, _score_worker, shared, workers, log=log):
        cells.append(cell)
        if log:
            log(
                f"{cell['variant']} @ lead {cell['lead_min']}: "
                f"{cell['covered_onsets']} covered onset(s), "
                f"recall {cell['recall']:.3f}, "
                f"shadow {cell['shadowed_misses']}/{cell['misses']}"
                if cell["recall"] is not None else
                f"{cell['variant']} @ lead {cell['lead_min']}: no rate"
            )
    cells.sort(key=lambda c: (c["lead_min"], c["variant"]))
    global _SHARED
    _SHARED = None

    all_buckets = _empty_buckets()
    for station_onsets in onsets[VARIANTS[0].name].values():
        for total in station_onsets.values():
            all_buckets[_bucket_of(total)] += 1

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds",
        ),
        "settings": {
            "corpus_dir": str(corpus_dir),
            "decisions_dirs": [str(d) for d in decisions_dirs],
            "picks": str(picks_path),
            "leads": list(used),
            "thresholds_pct": {str(k): v for k, v in thresholds.items()},
            "tolerance_min": int(tolerance_min),
            "coverage_gap_min": int(coverage_gap_min),
            "min_useful_lead_min": float(min_useful_lead_min),
            "persistence_obs": int(persistence_obs),
            "rearm_after_min": int(rearm_after_min),
            "raining_now_mm_h": RAIN_THRESHOLD_MM_H,
        },
        "window": {
            "from": window_from.isoformat(),
            "to": window_to.isoformat(),
            "days": len(days),
            "stations": len(station_ids),
            "stations_scored": len(scored_stations),
            "rows": n_rows,
            "station_days": shared["station_days"],
            "known_gauge_slots": known_slots,
        },
        "variants": [variant.as_dict() for variant in variants],
        "cells": cells,
        "v0_mm_buckets_all_onsets": all_buckets,
    }


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    """A share as a percentage, to one decimal.

    One decimal rather than none because the shadow shares land in the
    low single digits, and a table full of "0 %" would read as "none"
    when it means "three in a thousand".
    """
    return "—" if value is None else f"{value * 100:.1f} %"


def _share(part: int, whole: int) -> str:
    return "—" if not whole else _pct(part / whole)


def render_markdown(payload: Mapping[str, Any]) -> str:
    """The report: one table per lead, rows = variants, plus a paragraph."""
    settings = payload["settings"]
    window = payload["window"]
    variants = {v["name"]: v for v in payload["variants"]}
    cells = list(payload["cells"])
    leads = sorted({int(c["lead_min"]) for c in cells})

    out: list[str] = []
    out.append("# Onset sensitivity: how much of the miss count is definitional?\n")
    out.append(
        f"Generated {payload['generated_at_utc']} over "
        f"{window['rows']:,} decision rows — {window['stations_scored']} "
        f"scored station(s), {window['days']} day(s), "
        f"{window['station_days']:,} station-days, "
        f"{window['from'][:10]} → {window['to'][:10]}.\n"
    )
    out.append(
        "The warnings are FIXED: each lead was replayed once at the "
        "threshold the sweep picked ("
        + ", ".join(
            f"{lead} min → {settings['thresholds_pct'][str(lead)]} %"
            for lead in leads
        )
        + "), and that one set was then scored against every onset "
        "definition below. Nothing is refitted, so a row-to-row difference "
        "is the definition and nothing else.\n"
    )
    out.extend(_headline(cells))
    out.append("## The variants\n")
    out.append("| variant | rule | asks |")
    out.append("| --- | --- | --- |")
    for name in sorted(variants):
        variant = variants[name]
        out.append(f"| {name} | {variant['label']} | {variant['note']} |")
    out.append("")
    out.append(
        "Every variant keeps the shipped WET rule (≥ 0.1 mm or ≥ 1 min in "
        "the 10-minute slot) for certifying a dry spell; they differ in how "
        "long that spell must be and in how much rain the onset itself must "
        "deliver, summed over the onset slot and the one after it.\n"
    )

    for lead in leads:
        threshold = settings["thresholds_pct"][str(lead)]
        out.append(f"## Lead {lead} min, warning at {threshold} %\n")
        out.append(
            "| variant | covered onsets | per station-day | TP | FP | FN | "
            "late | precision | recall | F1 | do-nothing misses | shadowed | "
            "shadow share | recall excl. shadowed |"
        )
        out.append("| --- |" + " ---: |" * 13)
        rows = [c for c in cells if int(c["lead_min"]) == lead]
        rows.sort(key=lambda c: c["variant"])
        for cell in rows:
            out.append(
                f"| {cell['variant']} | {cell['covered_onsets']:,} | "
                f"{cell['onsets_per_station_day']:.2f} | {cell['hits']:,} | "
                f"{cell['false_alarms']:,} | {cell['misses']:,} | "
                f"{cell['late']:,} | {_fmt(cell['precision'])} | "
                f"{_fmt(cell['recall'])} | {_fmt(cell['f1'])} | "
                f"{cell['do_nothing_misses']:,} | "
                f"{cell['shadowed_misses']:,} | "
                f"{_pct(cell['shadow_share'])} | "
                f"{_fmt(cell['recall_excl_shadowed'])} |"
            )
        out.append("")
        out.extend(_lead_paragraph(rows, lead, threshold))
        out.append("")

    out.append("## How much of V0 is drizzle\n")
    out.append(
        "Millimetres over the onset slot and the following one, for every "
        "onset the current definition finds. The bucket edges are V2's and "
        "V3's thresholds, so the first column is what V2 discards.\n"
    )
    out.append("| population | < 0.2 mm | 0.2–0.5 mm | ≥ 0.5 mm | n |")
    out.append("| --- | ---: | ---: | ---: | ---: |")
    rows_out = [("all onsets in the gauge window", payload["v0_mm_buckets_all_onsets"])]
    for lead in leads:
        cell = next(
            (c for c in cells
             if int(c["lead_min"]) == lead and c["variant"] == VARIANTS[0].name),
            None,
        )
        if cell:
            rows_out.append((f"covered at lead {lead} min", cell["mm_buckets"]))
    for label, buckets in rows_out:
        total = sum(buckets.values())
        out.append(
            f"| {label} | {_share(buckets[MM_BUCKETS[0]], total)} | "
            f"{_share(buckets[MM_BUCKETS[1]], total)} | "
            f"{_share(buckets[MM_BUCKETS[2]], total)} | {total:,} |"
        )
    out.append("")
    out.append("## Reading the columns\n")
    out.append(
        "* **covered onsets** — onsets a decision row was actually watching "
        "for (`hits + misses + late`); uncovered and pending onsets are out "
        "of every rate here exactly as they are in the sweep.\n"
        "* **do-nothing misses** — the miss count of a service that never "
        "warns. It is the denominator recall is measured against, so a "
        "variant that halves it has halved the size of the problem, not "
        "improved the forecast.\n"
        "* **shadowed** — missed onsets inside the "
        f"{settings['rearm_after_min']}-minute window after a warning that "
        "was itself a hit. The subscription was disarmed, so no second "
        "notification was possible at any threshold. Two things bound it "
        "from below: the engine re-arms after 60 minutes BELOW threshold, "
        "so a point in continuous rain stays disarmed longer than this; "
        "and only hits are counted, while a false alarm disarms the "
        "subscription exactly as well (the JSON's "
        "`shadowed_by_any_warning`, quoted in each paragraph, is that "
        "wider count). Note that the shadow is ZERO by construction for "
        "every variant with a 60-minute dry spell: two onsets 60 minutes "
        "apart cannot both exist when an onset needs 60 dry minutes "
        "behind it.\n"
        "* **recall excl. shadowed** — `hits / (hits + misses + late − "
        "shadowed)`: what recall would read if the misses the anti-spam "
        "rule made impossible were taken out of the denominator. Lates stay "
        "in it, as they do in the ordinary recall.\n"
    )
    return "\n".join(out) + "\n"


def _range(values: Sequence[float | None], render) -> str:
    """``lo–hi`` over the leads, or the single value when they agree."""
    good = [v for v in values if v is not None]
    if not good:
        return "—"
    lo, hi = render(min(good)), render(max(good))
    return lo if lo == hi else f"{lo}–{hi}"


def _headline(cells: Sequence[Mapping[str, Any]]) -> list[str]:
    """The two-sentence answer, computed from the grid rather than typed.

    The whole report exists to separate one cause from another, and a
    reader who opens it should not have to derive which one won from five
    tables. Rendered from the cells, so it cannot drift from them.
    """
    base = [c for c in cells if c["variant"] == VARIANTS[0].name]
    if not base:
        return []
    strict = [c for c in cells if c["variant"] == "V2"]
    out = ["## What it says\n"]
    out.append(
        "**The re-arm shadow is small; the definition is not.** Across the "
        "leads scored here the shadow accounts for "
        + _range([c["shadow_share"] for c in base], _pct)
        + " of the misses ("
        + _range(
            [
                (c["shadowed_by_any_warning"] / c["misses"]) if c["misses"]
                else None
                for c in base
            ],
            _pct,
        )
        + " counting false alarms, which disarm the subscription just as "
        "well), so removing them moves recall from "
        + _range([c["recall"] for c in base], _fmt)
        + " to "
        + _range([c["recall_excl_shadowed"] for c in base], _fmt)
        + " — nothing a reader would notice.\n"
    )
    if strict:
        by_lead = {c["lead_min"]: c for c in base}
        removed = [
            1.0 - (c["covered_onsets"] / by_lead[c["lead_min"]]["covered_onsets"])
            for c in strict
            if by_lead.get(c["lead_min"], {}).get("covered_onsets")
        ]
        out.append(
            "The onset definition, by contrast, moves everything. Asking an "
            "onset for 0.2 mm over two slots and a 60-minute dry spell (V2) "
            "removes " + _range(removed, _pct) + " of the onsets and lifts "
            "recall to " + _range([c["recall"] for c in strict], _fmt)
            + " on the SAME notifications — while cutting precision from "
            + _range([c["precision"] for c in base], _fmt) + " to "
            + _range([c["precision"] for c in strict], _fmt)
            + ", because a notification that was a hit on a drizzle onset "
            "becomes a false alarm. Neither number is the forecast getting "
            "better or worse; both are the question changing.\n"
        )
    return out


def _lead_paragraph(
    rows: Sequence[Mapping[str, Any]], lead: int, threshold: int,
) -> list[str]:
    """One plain-language paragraph about this lead's table."""
    by_name = {row["variant"]: row for row in rows}
    base = by_name.get("V0")
    if base is None:
        return ["No V0 row at this lead, so there is nothing to compare against."]
    lines: list[str] = []
    lines.append(
        f"At {lead} minutes the rule sent {base['n_sent']:,} notifications "
        f"and the current definition gave it {base['covered_onsets']:,} "
        f"onsets to catch — {base['onsets_per_station_day']:.2f} per "
        f"station-day — for a recall of {_fmt(base['recall'])}. "
        f"Of the {base['misses']:,} misses, {base['shadowed_misses']:,} "
        f"({_pct(base['shadow_share'])}) fell inside the hour a warning "
        f"that hit had already disarmed, so no second notification was "
        f"possible for them; taking those out lifts recall to "
        f"{_fmt(base['recall_excl_shadowed'])}. Counting every "
        f"notification rather than only the hits — a false alarm disarms "
        f"the subscription just as effectively — the figure is "
        f"{base['shadowed_by_any_warning']:,} "
        f"({_share(base['shadowed_by_any_warning'], base['misses'])})."
    )
    for name in ("V2", "V4"):
        row = by_name.get(name)
        if row is None:
            continue
        drop = base["covered_onsets"] - row["covered_onsets"]
        lines.append(
            f"{name} ({row['dry_min']} min dry, ≥ "
            f"{row['min_mm_two_slots']:g} mm over two slots) removes "
            f"{drop:,} of those onsets "
            f"({_share(drop, base['covered_onsets'])} of them), leaving "
            f"{row['onsets_per_station_day']:.2f} per station-day; with the "
            f"same {row['n_sent']:,} notifications precision moves to "
            f"{_fmt(row['precision'])}, recall to {_fmt(row['recall'])} "
            f"({_fmt(row['recall_excl_shadowed'])} excluding the "
            f"{row['shadowed_misses']:,} shadowed) and F1 to "
            f"{_fmt(row['f1'])}."
        )
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Score one fixed set of push warnings against several gauge "
            "onset definitions, and measure the re-arm shadow."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--corpus-dir", type=Path, required=True,
                   help="gauge store root (the directory holding stations/)")
    p.add_argument("--decisions-dir", type=Path, nargs="+", required=True,
                   action="extend", dest="decisions_dirs",
                   help="directory tree of decision parquet files; repeatable, "
                        "later directories win a (radar_ts, station_id) tie")
    p.add_argument("--picks", type=Path, required=True,
                   help="the fitted push_thresholds.json; each lead is "
                        "replayed once at its own pick")
    p.add_argument("--leads", default=None,
                   help="comma-separated leads to score; default is every "
                        "lead the picks document carries")
    p.add_argument("--tolerance-min", type=int, default=DEFAULT_TOLERANCE_MIN)
    p.add_argument("--coverage-gap-min", type=int,
                   default=DEFAULT_COVERAGE_GAP_MIN)
    p.add_argument("--min-useful-lead-min", type=float,
                   default=FIT_MIN_USEFUL_LEAD_MIN)
    p.add_argument("--persistence-obs", type=int, default=1)
    p.add_argument("--rearm-after-min", type=int, default=60,
                   help="the engine's re-arm, and therefore the width of the "
                        "shadow window a hit casts over later onsets")
    p.add_argument("--out-md", type=Path, default=None)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--workers", type=int, default=1,
                   help="processes over the leads, then over the "
                        "(variant, lead) cells")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    leads: list[int] | None = None
    if args.leads:
        try:
            leads = [int(part) for part in str(args.leads).split(",") if part.strip()]
        except ValueError:
            print("error: --leads must be a comma-separated list of integers",
                  file=sys.stderr)
            return 2
    if args.persistence_obs < 1:
        print("error: --persistence-obs must be >= 1", file=sys.stderr)
        return 2
    if args.rearm_after_min <= 0:
        print("error: --rearm-after-min must be positive", file=sys.stderr)
        return 2

    try:
        payload = run(
            corpus_dir=Path(args.corpus_dir),
            decisions_dirs=list(args.decisions_dirs),
            picks_path=Path(args.picks),
            leads=leads,
            tolerance_min=int(args.tolerance_min),
            coverage_gap_min=int(args.coverage_gap_min),
            min_useful_lead_min=float(args.min_useful_lead_min),
            persistence_obs=int(args.persistence_obs),
            rearm_after_min=int(args.rearm_after_min),
            workers=int(args.workers),
            log=log,
        )
    except SweepError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.out_json:
        write_atomic(
            Path(args.out_json), json.dumps(payload, indent=1, default=str) + "\n",
        )
        log(f"wrote {args.out_json}")
    if args.out_md:
        write_atomic(Path(args.out_md), render_markdown(payload))
        log(f"wrote {args.out_md}")

    log(f"scored {len(payload['cells'])} cell(s) in {time.time() - started:.1f}s")
    print(json.dumps({
        "window": payload["window"],
        "v0_mm_buckets_all_onsets": payload["v0_mm_buckets_all_onsets"],
        "cells": [
            {
                key: cell[key] for key in (
                    "variant", "lead_min", "covered_onsets",
                    "onsets_per_station_day", "precision", "recall", "f1",
                    "shadowed_misses", "shadow_share", "recall_excl_shadowed",
                )
            }
            for cell in payload["cells"]
        ],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
