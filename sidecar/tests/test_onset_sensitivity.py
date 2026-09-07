"""The onset-definition sensitivity analysis.

``scripts/onset_sensitivity.py`` answers one question: of the misses the
threshold sweep counted, how many are the gauge ONSET DEFINITION and how
many are the engine's own 60-minute re-arm? It reuses the sweep's replay
and the core library's scorer, so what is left to test is the part it
adds: deriving onsets under five definitions from the raw slot amounts,
and counting the misses a hit warning shadowed.

Tested from the sidecar suite for the same reason ``test_sweep_thresholds``
is: the script imports both ``dmi_nowcast_sidecar.threshold_sweep`` and
``dmi_nowcast_core.warning_score``, and this environment is the only one
that has both.

Fully offline and fully synthetic: one hand-written slot series with every
onset worked out on paper, and a hand-written warning/onset pair for the
shadow. No parquet, no radar, no network.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import onset_sensitivity as sens  # noqa: E402  (after the sys.path edit)

from dmi_nowcast_core.station_store import StationObsStore  # noqa: E402
from dmi_nowcast_core.metobs import Observation  # noqa: E402
from dmi_nowcast_core.warning_score import (  # noqa: E402
    gauge_slots,
    gauge_truth_vectorised,
    onsets as gauge_onsets,
    score_warnings,
)

BASE = datetime(2026, 6, 1, tzinfo=timezone.utc)


def at(index: int) -> datetime:
    """The END of slot ``index`` — the first slot ends at 00:10."""
    return BASE + timedelta(minutes=10 * (index + 1))


#: A hand-built slot series as ``(mm, duration minutes)``; ``None`` for a
#: parameter the station did not report. Worked out below, slot by slot:
#:
#: * 0–2   dry, then **3** wet (0.1 mm) — three dry slots behind it, so an
#:         onset at 30 min of dry but not at 60. Two-slot amount 0.1 mm:
#:         drizzle, dropped by every amount test.
#: * 4–10  seven dry slots, then **11** wet (0.3 mm) — an onset at 30 and
#:         at 60 min of dry; 0.3 mm clears V2 and fails V3.
#: * 12–24 thirteen dry slots, then **25** wet (0.6 mm) — clears every dry
#:         spell including 120 min, and every amount test.
#: * 26–31 dry but for the UNKNOWN slot at 29, which resets the run, so
#:         **32** (0.2 mm) has only two dry slots behind it: no onset at
#:         all, under any variant.
#: * 33–38 six dry slots, then **39** (0.15 mm) followed by **40**
#:         (0.1 mm): the two-slot sum is 0.25 mm, so V2 keeps an onset the
#:         first slot alone would not have carried. 40 itself is wet with
#:         a wet slot behind it — a continuation, never an onset.
#: * 41–47 seven dry slots, then **48**: DMI's trace sentinel (−0.1 mm)
#:         with two minutes of duration. Wet through the duration arm, and
#:         worth 0.0 mm — the case that separates the wet rule from the
#:         amount rule.
SERIES: tuple[tuple[float | None, float | None], ...] = (
    (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),          # 0–2   dry
    (0.1, 1.0),                                   # 3     wet, 0.1 mm
    (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
    (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),           # 4–10  dry ×7
    (0.3, 1.0),                                   # 11    wet, 0.3 mm
    (0.0, 0.0),                                   # 12    dry
    (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
    (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
    (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),  # 13–24 dry ×12
    (0.6, 1.0),                                   # 25    wet, 0.6 mm
    (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),           # 26–28 dry
    (None, None),                                 # 29    UNKNOWN
    (0.0, 0.0), (0.0, 0.0),                       # 30–31 dry
    (0.2, 0.0),                                   # 32    wet, too soon
    (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
    (0.0, 0.0), (0.0, 0.0),                       # 33–38 dry ×6
    (0.15, 0.0),                                  # 39    wet, 0.15 mm
    (0.1, 0.0),                                   # 40    still wet
    (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
    (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),           # 41–47 dry ×7
    (-0.1, 2.0),                                  # 48    trace + duration
    (0.0, 0.0),                                   # 49    dry
)

#: Slot indices each variant should call an onset. Every one is argued for
#: in the ``SERIES`` comment above.
EXPECTED: dict[str, tuple[int, ...]] = {
    "V0": (3, 11, 25, 39, 48),
    "V1": (11, 25, 39, 48),
    "V2": (11, 25, 39),
    "V3": (25,),
    "V4": (25,),
}


def grid() -> list[tuple[datetime, float | None, float | None]]:
    """``SERIES`` as the ``(slot end, mm, duration)`` grid the script reads."""
    return [(at(i), mm, dur) for i, (mm, dur) in enumerate(SERIES)]


def variant(name: str) -> sens.Variant:
    return next(v for v in sens.VARIANTS if v.name == name)


# ---------------------------------------------------------------------------
# Variant onset derivation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_variant_finds_exactly_its_own_onsets(name: str) -> None:
    found = sens.variant_onsets(grid(), variant(name))
    assert [ts for ts, _mm in found] == [at(i) for i in EXPECTED[name]]


def test_v0_reproduces_the_shipped_onset_rule() -> None:
    """V0 must BE the definition the sweep was fitted on, not a lookalike.

    Same series, once through ``warning_score.gauge_slots`` +
    ``warning_score.onsets`` (booleans) and once through the script's
    amount-carrying path. A drift between the two would silently move the
    baseline every other variant is compared against.
    """
    rows = []
    for i, (mm, dur) in enumerate(SERIES):
        if mm is not None:
            rows.append({
                "station_id": "06180", "observed_utc": at(i),
                "parameter_id": "precip_past10min", "value": mm,
            })
        if dur is not None:
            rows.append({
                "station_id": "06180", "observed_utc": at(i),
                "parameter_id": "precip_dur_past10min", "value": dur,
            })
    slots = gauge_slots(rows, "06180", start_utc=at(0), end_utc=at(len(SERIES) - 1))
    shipped = gauge_onsets(slots, 30)
    ours = [ts for ts, _mm in sens.variant_onsets(grid(), variant("V0"))]
    assert ours == shipped


def test_the_two_slot_amount_spans_the_onset_slot_and_the_next() -> None:
    found = dict(sens.variant_onsets(grid(), variant("V0")))
    assert found[at(3)] == pytest.approx(0.1)    # 0.1 + 0.0
    assert found[at(39)] == pytest.approx(0.25)  # 0.15 + 0.1
    assert found[at(48)] == pytest.approx(0.0)   # trace + dry


def test_a_failed_amount_test_still_resets_the_dry_run() -> None:
    """Drizzle is not an onset, but it is not a dry spell either.

    Six dry slots, a 0.1 mm drizzle at slot 6, six more dry slots, then
    0.5 mm at slot 13. V0 calls both wet slots onsets. V2 drops the
    drizzle on the amount test and keeps slot 13.

    V4 is the assertion that matters: it needs 120 minutes of dry, and
    slot 13 has only the six slots since the drizzle, so V4 finds nothing.
    Had the *dropped* candidate not reset the run, slot 13 would sit on
    twelve dry slots and V4 would report an onset that never happened.
    """
    series = (
        [(0.0, 0.0)] * 6 + [(0.1, 0.0)] + [(0.0, 0.0)] * 6 + [(0.5, 1.0)]
    )
    cells = [(at(i), mm, dur) for i, (mm, dur) in enumerate(series)]
    assert [ts for ts, _ in sens.variant_onsets(cells, variant("V0"))] == [
        at(6), at(13),
    ]
    assert [ts for ts, _ in sens.variant_onsets(cells, variant("V2"))] == [at(13)]
    assert sens.variant_onsets(cells, variant("V4")) == []


def test_a_hole_in_the_grid_is_not_a_dry_spell() -> None:
    """Two days stitched together cannot invent an onset across the seam."""
    cells = [(at(i), 0.0, 0.0) for i in range(6)]
    cells.append((at(6) + timedelta(days=1), 1.0, 5.0))
    assert sens.variant_onsets(cells, variant("V0")) == []


# ---------------------------------------------------------------------------
# The vectorised load derives the same variants from the archive
# ---------------------------------------------------------------------------


def stored_series(root: Path) -> None:
    """``SERIES`` written to a real parquet store, as DMI would deliver it."""
    rows = []
    for i, (mm, dur) in enumerate(SERIES):
        if mm is not None:
            rows.append(Observation("06180", at(i), "precip_past10min", mm))
        if dur is not None:
            rows.append(Observation("06180", at(i), "precip_dur_past10min", dur))
    StationObsStore(root).append(rows)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_the_vectorised_load_finds_each_variant_s_own_onsets(
    name: str, tmp_path: Path,
) -> None:
    """The archive path and the hand-worked grid must not drift apart.

    ``gauge_truth_variants`` no longer walks the grid in Python — it reads
    each month partition into numpy and derives every variant by boolean
    run length over the same array. The same series, written to a store
    and read back, has to give the same onsets AND the same two-slot
    amounts as :func:`variant_onsets` reading the grid directly; the
    amounts matter because the mm histogram in the report is built from
    them.
    """
    stored_series(tmp_path)
    spec = variant(name)
    truth = gauge_truth_vectorised(
        tmp_path, at(0), at(len(SERIES) - 1), ["06180"],
        dry_min=spec.dry_min, min_mm_two_slots=spec.min_mm_two_slots,
    )
    want = sens.variant_onsets(grid(), spec)
    assert truth.onsets["06180"] == [ts for ts, _mm in want]
    found = truth.series["06180"].onsets_with_amounts(
        spec.dry_min, min_mm_two_slots=spec.min_mm_two_slots,
    )
    assert [mm for _ts, mm in found] == [pytest.approx(mm) for _ts, mm in want]


def test_one_archive_read_serves_every_variant(tmp_path: Path) -> None:
    """Five definitions, one load — the whole point of the rewrite."""
    stored_series(tmp_path)
    truth = gauge_truth_vectorised(
        tmp_path, at(0), at(len(SERIES) - 1), ["06180"],
    )
    for spec in sens.VARIANTS:
        derived = truth.onsets_for(
            spec.dry_min, min_mm_two_slots=spec.min_mm_two_slots,
        )
        assert [ts for ts, _mm in derived["06180"]] == [
            at(i) for i in EXPECTED[spec.name]
        ]


# ---------------------------------------------------------------------------
# The slot table
# ---------------------------------------------------------------------------


def test_slot_amounts_keeps_unknown_unknown_and_folds_the_trace() -> None:
    rows = [
        {"observed_utc": at(0), "parameter_id": "precip_past10min", "value": -0.1},
        {"observed_utc": at(0), "parameter_id": "precip_dur_past10min", "value": 2.0},
        # Slot 1 reports only a duration: known, but with no depth.
        {"observed_utc": at(1), "parameter_id": "precip_dur_past10min", "value": 0.0},
        # Slot 2 reports nothing at all, and slot 3 something unrelated.
        {"observed_utc": at(3), "parameter_id": "temp_dry", "value": 12.0},
    ]
    cells = sens.slot_amounts(rows, start_utc=at(0), end_utc=at(3))
    assert cells == [
        (at(0), 0.0, 2.0),      # trace folded to 0 mm, duration kept
        (at(1), None, 0.0),     # duration-only: known, no depth
        (at(2), None, None),    # nothing reported: unknown
        (at(3), None, None),    # a parameter the rule does not read
    ]


def test_slot_amounts_fills_the_grid_contiguously() -> None:
    rows = [
        {"observed_utc": at(0), "parameter_id": "precip_past10min", "value": 1.0},
        {"observed_utc": at(5), "parameter_id": "precip_past10min", "value": 2.0},
    ]
    cells = sens.slot_amounts(rows, start_utc=at(0), end_utc=at(5))
    assert [ts for ts, _mm, _d in cells] == [at(i) for i in range(6)]


# ---------------------------------------------------------------------------
# The re-arm shadow
# ---------------------------------------------------------------------------

#: A window wide enough that nothing in these tests is uncovered, and a
#: gauge horizon far enough out that nothing is pending.
COVERAGE = [(BASE, BASE + timedelta(hours=12))]
KNOWN_UNTIL = BASE + timedelta(hours=12)


def score(warnings, onset_times, **kwargs):
    return score_warnings(
        warnings, onset_times,
        lead_min=30, tolerance_min=10,
        known_until=KNOWN_UNTIL, coverage=COVERAGE,
        min_useful_lead_min=5.0, **kwargs,
    )


def hhmm(hour: int, minute: int) -> datetime:
    return BASE + timedelta(hours=hour, minutes=minute)


def test_a_miss_inside_the_hour_after_a_hit_is_shadowed() -> None:
    """One warning, three onsets: a hit, a shadowed miss and a plain one.

    The warning at 09:50 promises rain within 30 min (+10 tolerance), so
    it claims the 10:00 onset — a hit, with ten minutes of realised lead.
    10:20 is unclaimed: a miss, but it lands 30 minutes after a warning
    that fired, and the engine cannot fire again for an hour. 11:30 is
    100 minutes out — the subscription had long re-armed, so that miss is
    the forecast's own.
    """
    result = score(
        [(hhmm(9, 50), 30.0)],
        [hhmm(10, 0), hhmm(10, 20), hhmm(11, 30)],
    )
    assert result.summary["hits"] == 1
    assert result.summary["misses"] == 2
    assert sens.shadowed_misses(result, rearm_after_min=60) == 1


def test_a_miss_behind_a_false_alarm_is_not_shadowed_by_default() -> None:
    """Only a HIT casts the headline shadow; every warning casts the wide one.

    The warning at 08:00 claims nothing (its window closes at 08:40) and
    is a false alarm; the 08:50 onset is 50 minutes behind it. The rule
    was just as disarmed, but a false alarm is not evidence that the user
    was told about this rain, so the headline count leaves it out and the
    ``outcomes=`` form picks it up.
    """
    result = score([(hhmm(8, 0), 30.0)], [hhmm(8, 50)])
    assert result.summary["false_alarms"] == 1
    assert result.summary["misses"] == 1
    assert sens.shadowed_misses(result, rearm_after_min=60) == 0
    assert sens.shadowed_misses(
        result, rearm_after_min=60,
        outcomes=("hit", "late", "false_alarm"),
    ) == 1


def test_an_onset_exactly_on_the_rearm_boundary_is_shadowed() -> None:
    """The window is ``(sent, sent + rearm]`` — closed at the far end.

    A warning at 09:50 hits the 10:00 onset. The next onset at 10:50 is
    exactly sixty minutes out: the arm settles at the first observation
    that finds the dry spell 60 minutes old, so that instant is still on
    the disarmed side and the miss is shadowed. One minute later it is
    not.
    """
    inside = score(
        [(hhmm(9, 50), 30.0)], [hhmm(10, 0), hhmm(10, 50)],
    )
    assert sens.shadowed_misses(inside, rearm_after_min=60) == 1
    outside = score(
        [(hhmm(9, 50), 30.0)], [hhmm(10, 0), hhmm(10, 51)],
    )
    assert sens.shadowed_misses(outside, rearm_after_min=60) == 0


def test_an_onset_before_every_warning_is_never_shadowed() -> None:
    result = score([(hhmm(10, 0), 30.0)], [hhmm(9, 0), hhmm(10, 20)])
    assert result.summary["hits"] == 1
    assert result.summary["misses"] == 1
    assert sens.shadowed_misses(result, rearm_after_min=60) == 0


def test_no_warnings_means_no_shadow() -> None:
    result = score([], [hhmm(10, 0), hhmm(10, 20)])
    assert result.summary["misses"] == 2
    assert sens.shadowed_misses(result, rearm_after_min=60) == 0


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_the_markdown_renders_the_numbers_it_was_given() -> None:
    """A hand-made payload through the renderer, asserting the numbers land.

    Cheap insurance: the analysis run itself takes tens of minutes, and a
    format error discovered at the end of it costs the whole run.
    """
    cell = {
        "variant": "V0", "dry_min": 30, "min_mm_two_slots": None,
        "lead_min": 30, "threshold_pct": 40,
        "n_onsets": 40, "covered_onsets": 20, "uncovered_onsets": 15,
        "pending_onsets": 5, "onsets_per_station_day": 2.0,
        "warnings": 10, "n_sent": 10, "pending": 0, "hits": 4,
        "false_alarms": 6, "late": 0, "misses": 16,
        "precision": 0.4, "recall": 0.2, "f1": 0.266, "far": 0.6, "csi": 0.15,
        "do_nothing_misses": 20, "shadowed_misses": 8, "shadow_share": 0.5,
        "shadowed_by_any_warning": 9, "recall_excl_shadowed": 0.333,
        "mm_buckets": {"<0.2": 10, "0.2-0.5": 5, ">=0.5": 5},
        "station_days": 10, "n_stations": 2,
    }
    other = dict(
        cell, variant="V2", dry_min=60, min_mm_two_slots=0.2,
        covered_onsets=9, onsets_per_station_day=0.9, misses=5,
        recall=0.444, precision=0.4, f1=0.42, shadowed_misses=2,
        shadow_share=0.4, recall_excl_shadowed=0.571,
    )
    payload = {
        "generated_at_utc": "2026-09-06T18:00:00+00:00",
        "settings": {
            "thresholds_pct": {"30": 40}, "rearm_after_min": 60,
        },
        "window": {
            "from": "2025-12-08T00:00:01+00:00", "to": "2026-09-06T14:30:01+00:00",
            "days": 54, "stations": 2, "stations_scored": 2, "rows": 100,
            "station_days": 10, "known_gauge_slots": 999,
        },
        "variants": [v.as_dict() for v in sens.VARIANTS],
        "cells": [cell, other],
        "v0_mm_buckets_all_onsets": {"<0.2": 20, "0.2-0.5": 10, ">=0.5": 10},
    }
    text = sens.render_markdown(payload)
    assert "## Lead 30 min, warning at 40 %" in text
    assert "| V0 |" in text and "| V2 |" in text
    # The V0 paragraph must name the shadow and what removing it does.
    assert "8 (50.0 %) fell inside the hour a warning that hit" in text
    assert "recall to 0.333" in text
    # …and the wider count, so the hit-only shadow is never read alone.
    assert "the figure is 9 (56.2 %)" in text
    # And the drizzle table must render the share, not the count.
    assert "| all onsets in the gauge window | 50.0 % | 25.0 % | 25.0 % | 40 |" in text
