"""The review bundle's builder: population, sampling, per-event detail.

The claim under test, above all others: **the review measures the rule
that shipped**. :func:`~dmi_nowcast_sidecar.review.replay_station_traced`
is a copy of ``threshold_sweep.replay_station`` with the state machine
written down, and a copy is only worth having while it is provably the
same rule — so the first test replays several hand-built tracks through
both and asserts the warnings match row for row, including a coverage-run
boundary, an already-raining silence and a long showery spell. Every tag
counted off this bundle is about that rule or about nothing.

After that, in the order a wrong answer would do the most damage:

* the arm / re-arm arithmetic, hand-worked, including the case that makes
  misses structurally unreachable — an over-threshold row while disarmed
  RESETS the dry clock, so the 60 minutes never accumulate;
* the sample: seeded and byte-reproducible, floors honoured, class
  targets hit exactly, and a ``population_hash`` that moves when the
  population moves and not when the seed does;
* ``pending`` never drawn, ``uncovered`` only as the requested control,
  dead gauges excluded;
* the decision-directory precedence, which is the OPPOSITE of the quality
  page's and would silently judge events on the curve scale if it were
  not;
* the feature gap, measured rather than assumed;
* dual truth over both window definitions — the same rain is a different
  verdict depending on which anchor the window hangs from;
* ``generated_at = radar_ts + frame_age`` and the slot-end convention;
* null never coerced to zero, anywhere in the output;
* decision gaps, and the ``prologue`` that explains an arm state set
  hours before the window starts.

Everything is synthetic and offline. The parquet tests build a real
station store and two real decision trees in ``tmp_path``; the rest needs
neither.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from dmi_nowcast_core.metobs import Observation
from dmi_nowcast_core.review_schema import (
    DUAL_TRUTH_CLASSES,
    EVENT_FLAGS,
    INDEX_FIELDS,
    OUTCOME_CLASSES,
    STRATA_KEYS,
    event_id,
)
from dmi_nowcast_core.warning_score import (
    StationSlots,
    dead_gauges,
    p_rain_column,
    slot_end_of,
)
from dmi_nowcast_sidecar import review
from dmi_nowcast_sidecar.threshold_sweep import build_tracks, replay_station

UTC = timezone.utc
DAY = datetime(2026, 6, 1, tzinfo=UTC)
LEAD = 30
LEADS = (10, 20, 30, 45, 60)
CURVE = p_rain_column(LEAD)
STATION = "06180"
FRAME_AGE_MIN = 14.0
#: The shipped timing, restated here so a test that changes it says so.
PERSISTENCE = 1
REARM = 60


# ---------------------------------------------------------------------------
# Row and track helpers
# ---------------------------------------------------------------------------


def _row(
    minute: int,
    p: float | None,
    *,
    observed: float = 0.0,
    station: str = STATION,
    start: datetime = DAY,
    frame_age: float = FRAME_AGE_MIN,
    eta: float | None = 20.0,
    features: bool = True,
) -> dict:
    """One decision row in the shared schema, ``minute`` after ``start``."""
    radar_ts = start + timedelta(minutes=minute)
    row: dict = {
        "radar_ts": radar_ts,
        "generated_at": radar_ts + timedelta(minutes=frame_age),
        "station_id": station,
        "p_rain": p,
        "eta_min": eta,
        "intensity_mm_h": None if p is None else 0.4 + 3.0 * p,
        "observed_mm_h": observed,
        "forecast_now_mm_h": observed,
        "action": "none",
        "armed_after": True,
        "streak_after": 0,
        "threshold_pct": 40,
    }
    for lead in LEADS:
        row[p_rain_column(lead)] = p
    if features:
        row["obs_max_5km_mm_h"] = None if p is None else 2.0 * p
        row["frame_age_min"] = float(frame_age)
    return row


def _track(rows: list[dict], *, coverage_gap_min: int = 20) -> list[tuple]:
    """Rows → the one station's track, through the REAL ``build_tracks``.

    The run index — the thing the replay resets its state on — is assigned
    by ``build_tracks`` from the frame spacing, so a gap in ``rows`` is a
    coverage-run boundary here exactly as it is in production. Hand-built
    tuples would let a test invent a boundary the loader would never
    produce.
    """
    tracks, _frames = build_tracks(
        rows, [LEAD], coverage_gap_min=coverage_gap_min,
        column_for=lambda _lead: CURVE,
    )
    station = rows[0]["station_id"]
    return tracks[station]


def _steady(p: float, *, first: int, last: int, step: int = 10, **kw) -> list[dict]:
    return [_row(minute, p, **kw) for minute in range(first, last + 1, step)]


# The tracks the equality test runs. Each one is a different way for the
# two implementations to disagree.
def _case_quiet() -> list[tuple]:
    return _track(_steady(0.05, first=0, last=300))


def _case_single_warning() -> list[tuple]:
    rows = _steady(0.05, first=0, last=100)
    rows += _steady(0.9, first=110, last=150)
    rows += _steady(0.05, first=160, last=400)
    return _track(rows)


def _case_coverage_boundary() -> list[tuple]:
    """Two runs with a three-hour outage between them.

    The replay resets to ``INITIAL_STATE`` at the head of the second run,
    handing the station a re-arm the live service never had. Both
    implementations must hand out the same one.
    """
    rows = _steady(0.9, first=0, last=60)
    rows += _steady(0.05, first=70, last=120)
    rows += _steady(0.9, first=300, last=360)
    return _track(rows)


def _case_already_raining() -> list[tuple]:
    """Over threshold with the rain already measured at the point.

    ``evaluate`` consumes the arm silently: no warning, and the
    subscription disarms as if it had pushed. A tracing variant that
    counted the action as a notify would produce one warning too many.
    """
    rows = _steady(0.05, first=0, last=60)
    rows += _steady(0.9, first=70, last=120, observed=2.0)
    rows += _steady(0.05, first=130, last=400)
    return _track(rows)


def _case_showery_spell() -> list[tuple]:
    """Six hours of probability bouncing over and under the threshold.

    The dry clock is reset by every over-threshold row while disarmed, so
    the station stays disarmed for hours after the first push. This is the
    mechanism behind ``miss_disarmed_rearm``, and the case where an
    off-by-one in the state copy would show up as a whole extra warning.
    """
    rows = _steady(0.05, first=0, last=50)
    rows.append(_row(60, 0.95))
    minute = 70
    for cycle in range(18):
        rows += [_row(minute + 10 * i, 0.1) for i in range(4)]
        rows.append(_row(minute + 40, 0.8))
        minute += 50
    rows += _steady(0.05, first=minute, last=minute + 300)
    return _track(rows)


def _case_null_probabilities() -> list[tuple]:
    """Rows with no probability at the rule's lead: skipped, not dry."""
    rows = _steady(0.9, first=0, last=20)
    rows += [_row(minute, None) for minute in range(30, 120, 10)]
    rows += _steady(0.9, first=120, last=200)
    return _track(rows)


ALL_CASES = {
    "quiet": _case_quiet,
    "single_warning": _case_single_warning,
    "coverage_boundary": _case_coverage_boundary,
    "already_raining": _case_already_raining,
    "showery_spell": _case_showery_spell,
    "null_probabilities": _case_null_probabilities,
}


# ---------------------------------------------------------------------------
# 1. The traced replay IS ``replay_station``
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(ALL_CASES))
@pytest.mark.parametrize(
    "threshold,persistence,rearm", [(50, 1, 60), (30, 1, 60), (70, 2, 30)],
)
def test_traced_replay_equals_replay_station(
    case: str, threshold: int, persistence: int, rearm: int,
) -> None:
    """The whole guard: same track, same rule, same warnings, row for row.

    If this ever fails, the bundle is measuring a rule nobody ran and
    every mechanism counted off it is about that other rule.
    """
    track = ALL_CASES[case]()
    expected = replay_station(
        track, 0, threshold,
        persistence_obs=persistence, rearm_after_min=rearm,
    )
    trace = review.replay_station_traced(
        track, 0, threshold,
        persistence_obs=persistence, rearm_after_min=rearm,
    )
    assert len(trace) == len(track), "one traced row per track record"
    assert review.traced_warnings(track, trace) == expected
    assert review.traced_warnings(track, trace, with_probability=True) == (
        replay_station(
            track, 0, threshold,
            persistence_obs=persistence, rearm_after_min=rearm,
            with_probability=True,
        )
    )


def test_the_population_s_warnings_are_replay_station_s(
) -> None:
    """The wiring, not just the loop: right column, right lead, right rule.

    ``build_population`` picks the probability column, fills the engine's
    per-row fallback, builds the tracks and replays them. A mistake in any
    of those produces a different warning list while the traced replay
    itself stays correct — so the fixture's events are checked against a
    direct ``replay_station`` run over the same tracks.
    """
    population = review.synthetic_population()
    rule = population.rule
    for station, track in population.tracks.items():
        expected = replay_station(
            track, 0, rule.threshold_pct,
            persistence_obs=rule.persistence_obs,
            rearm_after_min=rule.rearm_after_min,
            raining_now_mm_h=rule.raining_now_mm_h,
            raining_now_eta_min=rule.raining_now_eta_min,
        )
        trace = population.traces[station]
        assert review.traced_warnings(track, trace) == expected
        sent = {
            record.sent_utc for record in population.records
            if record.station_id == station and record.sent_utc is not None
        }
        assert sent <= {stamp for stamp, _eta in expected}


def test_traced_replay_honours_raining_now_overrides() -> None:
    """The served rule names every constant; the trace must pass them on."""
    track = _case_already_raining()
    for eta_floor in (1.5, 25.0):
        expected = replay_station(
            track, 0, 50, persistence_obs=1, rearm_after_min=60,
            raining_now_mm_h=5.0, raining_now_eta_min=eta_floor,
        )
        trace = review.replay_station_traced(
            track, 0, 50, persistence_obs=1, rearm_after_min=60,
            raining_now_mm_h=5.0, raining_now_eta_min=eta_floor,
        )
        assert review.traced_warnings(track, trace) == expected


def test_traced_rows_carry_the_skip_reason_not_an_action() -> None:
    """A row the engine never saw has no action — ``"none"`` would lie."""
    track = _case_null_probabilities()
    trace = review.replay_station_traced(
        track, 0, 50, persistence_obs=1, rearm_after_min=60,
    )
    skipped = [row for row in trace if row.skipped_reason is not None]
    assert skipped, "the fixture has rows with no probability"
    for row in skipped:
        assert row.action is None
        assert row.p_decision is None
        assert row.skipped_reason == review.SKIP_NO_PROBABILITY


def test_run_boundary_rearm_is_only_flagged_when_it_re_armed() -> None:
    """A reset that discards a streak is not a free re-arm; one that
    discards a DISARMED machine is, and a reviewer has to be told."""
    track = _case_coverage_boundary()
    trace = review.replay_station_traced(
        track, 0, 50, persistence_obs=1, rearm_after_min=60,
    )
    rearms = review.run_boundary_rearms(trace)
    assert len(rearms) == 1
    boundary = next(row for row in trace if row.run_id == 1)
    assert rearms == (boundary.radar_ts,)
    previous = trace[trace.index(boundary) - 1]
    assert previous.armed_after is False and boundary.armed_before is True

    # The quiet case has one run and therefore no boundary at all.
    assert review.run_boundary_rearms(
        review.replay_station_traced(
            _case_quiet(), 0, 50, persistence_obs=1, rearm_after_min=60,
        )
    ) == ()


# ---------------------------------------------------------------------------
# 2. The arm / re-arm arithmetic, hand-worked
# ---------------------------------------------------------------------------


def test_an_over_threshold_row_while_disarmed_resets_the_dry_clock() -> None:
    """Plan §4.4, in one track.

    00:00 fires and disarms. The dry clock starts at 00:10 and reaches 40
    minutes. At 01:00 one over-threshold row resets it to ``None`` — not
    to 60 minutes, not to zero elapsed: the clock is not running. The
    spell restarts at 01:10, so the re-arm lands at 02:10 and the machine
    fires again there. A reading that let the first spell stand would have
    re-armed at 01:10, an hour early, and turned a structurally
    unreachable miss into a forecast failure.
    """
    rows = [_row(0, 0.9)]
    rows += [_row(minute, 0.1) for minute in range(10, 60, 10)]
    rows.append(_row(60, 0.9))          # resets below_since to None
    rows += [_row(minute, 0.1) for minute in range(70, 130, 10)]
    rows.append(_row(130, 0.9))         # 01:10 + 60 min = 02:10
    track = _track(rows)
    trace = review.replay_station_traced(
        track, 0, 50, persistence_obs=PERSISTENCE, rearm_after_min=REARM,
    )
    by_minute = {
        int((row.radar_ts - DAY).total_seconds() // 60): row for row in trace
    }

    assert by_minute[0].action == "notify"
    assert by_minute[0].armed_after is False
    assert by_minute[0].below_since_utc is None

    assert by_minute[10].below_since_utc == DAY + timedelta(minutes=10)
    assert by_minute[50].below_since_utc == DAY + timedelta(minutes=10)
    assert by_minute[50].action == "none"

    # The reset. Null, not zero, and not a notify.
    assert by_minute[60].action == "none"
    assert by_minute[60].armed_after is False
    assert by_minute[60].below_since_utc is None

    assert by_minute[70].below_since_utc == DAY + timedelta(minutes=70)
    assert by_minute[120].action == "none", "50 min of dry is not 60"
    assert by_minute[130].action == "notify", "60 min after the RESTARTED spell"

    assert [
        minute for minute, row in sorted(by_minute.items())
        if row.action == "notify"
    ] == [0, 130]


def test_arm_state_at_reports_a_stopped_clock_as_null() -> None:
    """``minutes_to_rearm`` is ``None`` when the clock is not running.

    Zero would read as "about to re-arm" and sixty as "an hour to go";
    both are fabrications. The honest answer is that no dry spell is
    accumulating at all.
    """
    rows = [_row(0, 0.9)]
    rows += [_row(minute, 0.1) for minute in range(10, 60, 10)]
    rows.append(_row(60, 0.9))
    rows += [_row(minute, 0.1) for minute in range(70, 130, 10)]
    track = _track(rows)
    trace = review.replay_station_traced(
        track, 0, 50, persistence_obs=PERSISTENCE, rearm_after_min=REARM,
    )
    age = timedelta(minutes=FRAME_AGE_MIN)

    running = review.arm_state_at(
        trace, DAY + timedelta(minutes=50) + age, rearm_after_min=REARM,
    )
    assert running.armed is False
    assert running.below_since_utc == DAY + timedelta(minutes=10)
    assert running.minutes_to_rearm == pytest.approx(20.0)

    # Exactly at the resetting row, the ENTERING state is reported: the
    # clock the engine still had when it judged that observation, with ten
    # minutes left to run.
    entering = review.arm_state_at(
        trace, DAY + timedelta(minutes=60) + age, rearm_after_min=REARM,
    )
    assert entering.entering is True
    assert entering.below_since_utc == DAY + timedelta(minutes=10)
    assert entering.minutes_to_rearm == pytest.approx(10.0)

    # Five minutes later — after that row — the clock is gone. Not zero,
    # not sixty: not running.
    stopped = review.arm_state_at(
        trace, DAY + timedelta(minutes=65) + age, rearm_after_min=REARM,
    )
    assert stopped.entering is False
    assert stopped.armed is False
    assert stopped.below_since_utc is None
    assert stopped.minutes_to_rearm is None

    # An instant before anything was evaluated: the replay's own start,
    # reported as "no decision seen" rather than as a measurement.
    empty = review.arm_state_at(
        trace, DAY - timedelta(hours=1), rearm_after_min=REARM,
    )
    assert (empty.armed, empty.at_utc, empty.minutes_to_rearm) == (True, None, None)


def test_arm_state_at_a_warning_reports_the_state_that_decided_it() -> None:
    """At the exact instant of a row, the ENTERING state is the answer.

    A warning's anchor is the row's own ``generated_at``, and the machine
    was armed as it judged that row — reporting the disarmed state it left
    behind would put ``arm_state_at_anchor: disarmed`` on every warning
    ever sent.
    """
    track = _track(_steady(0.05, first=0, last=50) + _steady(0.9, first=60, last=120))
    trace = review.replay_station_traced(
        track, 0, 50, persistence_obs=PERSISTENCE, rearm_after_min=REARM,
    )
    sent = DAY + timedelta(minutes=60 + FRAME_AGE_MIN)
    state = review.arm_state_at(trace, sent, rearm_after_min=REARM)
    assert state.entering is True
    assert state.armed is True
    assert state.minutes_to_rearm is None


# ---------------------------------------------------------------------------
# 3. Sampling
# ---------------------------------------------------------------------------


def _record(
    event_class: str,
    *,
    station: str = STATION,
    minute: int = 0,
    season: str = "summer",
    region: str = "Fyn",
    band: str = "moderate",
    control: bool | None = None,
    sent: datetime | None = None,
    onset: datetime | None = None,
) -> review.EventRecord:
    """A minimal :class:`EventRecord` for the allocator's sake."""
    anchor = DAY + timedelta(minutes=minute)
    window = review.warning_window(anchor, lead_min=LEAD, tolerance_min=10)
    return review.EventRecord(
        event_id=event_id(event_class, station, anchor),
        event_class=event_class,
        station_id=station,
        anchor_utc=anchor,
        sent_utc=anchor if sent is None else sent,
        onset_utc=onset,
        eta_min=20.0,
        p_decision=0.6,
        p_decision_source="curve",
        threshold_pct=50,
        lead_error_min=None,
        season=season,
        region=region,
        intensity_mm_h=2.0,
        intensity_band=band,
        intensity_band_source="forecast_intensity_mm_h",
        onset_two_slot_mm=None,
        dual_truth="both_dry",
        gauge_wet_in_window=False,
        radar_wet_in_window=False,
        neighbour_wet_in_window=None,
        neighbour_n_known=0,
        arm_state_at_anchor="armed",
        minutes_to_rearm_at_anchor=None,
        flags=(),
        control=(review.group_of(event_class) in review.CONTROL_GROUPS)
        if control is None
        else control,
        window_used=window,
    )


def _population(n_per_cell: int = 12) -> list[review.EventRecord]:
    """A population spread over every stratum the allocator nests."""
    out: list[review.EventRecord] = []
    minute = 0
    for event_class in ("false_alarm", "miss", "hit", "uncovered", "late"):
        for season in ("summer", "winter"):
            for region in ("Fyn", "Sjælland", "Nordjylland"):
                for band in ("light", "moderate", "heavy"):
                    for i in range(n_per_cell):
                        minute += 10
                        out.append(_record(
                            event_class,
                            station=f"061{i:02d}",
                            minute=minute,
                            season=season,
                            region=region,
                            band=band,
                        ))
    return out


def test_sampling_is_byte_reproducible_and_hits_the_targets() -> None:
    records = _population()
    targets = {
        "false_alarm": 12, "miss": 10, "late": 4, "hit": 6, "uncovered": 3,
    }
    first = review.stratify(records, seed=7, class_targets=targets)
    second = review.stratify(list(reversed(records)), seed=7, class_targets=targets)

    ids = [record.event_id for record in first.records]
    assert ids == [record.event_id for record in second.records], (
        "same population and seed must give the same events in the same order, "
        "whatever order the population arrived in"
    )
    assert len(ids) == sum(targets.values())
    for group, want in targets.items():
        assert first.drawn[group] == want, group

    # A different seed draws a different set, or the seeding does nothing.
    other = review.stratify(records, seed=8, class_targets=targets)
    assert [r.event_id for r in other.records] != ids
    assert other.population_hash == first.population_hash, (
        "the hash is a property of the population, not of the draw"
    )


def test_sampling_honours_the_floor_per_cell() -> None:
    """Every non-empty cell is represented before proportion takes over."""
    records = [
        *[_record("false_alarm", minute=10 * i, region="Fyn") for i in range(100)],
        *[
            _record("false_alarm", minute=5000 + 10 * i, region="Bornholm")
            for i in range(2)
        ],
    ]
    sample = review.stratify(
        records, seed=3, class_targets={"false_alarm": 20}, floor_per_cell=2,
    )
    cells = {cell.stratum: cell for cell in sample.cells if cell.population}
    assert len(cells) == 2
    for cell in cells.values():
        assert cell.drawn >= min(2, cell.population), cell.stratum
    assert sum(cell.drawn for cell in cells.values()) == 20


def test_a_group_smaller_than_its_target_contributes_all_of_it() -> None:
    records = [_record("miss", minute=10 * i) for i in range(4)]
    sample = review.stratify(records, seed=1, class_targets={"miss": 100})
    assert len(sample.records) == 4
    assert sample.drawn["miss"] == 4


def test_population_hash_moves_only_when_the_population_does() -> None:
    records = _population(2)
    base = review.population_hash(records)
    assert review.population_hash(list(reversed(records))) == base
    grown = records + [_record("false_alarm", station="09999", minute=99_999)]
    assert review.population_hash(grown) != base


def test_the_control_group_is_shuffled_into_the_list() -> None:
    """A reviewer who can infer the class from the position is not blind."""
    records = _population()
    sample = review.stratify(
        records, seed=11,
        class_targets={"false_alarm": 30, "miss": 30, "hit": 30},
    )
    classes = [record.event_class for record in sample.records]
    # A list that was concatenated group by group has exactly two changes
    # of class; an interleaved one has many.
    switches = sum(1 for a, b in zip(classes, classes[1:]) if a != b)
    assert switches > 10, classes
    assert all(
        record.control is (record.event_class == "hit")
        for record in sample.records
    )


def test_late_and_miss_late_are_collapsed_to_one_event_per_episode() -> None:
    """``score_warnings`` emits the pair; a reviewer must see it once."""
    sent = DAY + timedelta(minutes=100)
    onset = DAY + timedelta(minutes=104)
    pair = [
        _record("late", minute=100, sent=sent, onset=onset),
        _record("miss_late", minute=104, sent=sent, onset=onset),
    ]
    sample = review.stratify(pair, seed=2, class_targets={"late": 10})
    assert [r.event_class for r in sample.records] == ["late"]
    assert sample.collapsed_late_pairs == 1


def test_target_and_class_targets_are_never_mixed_silently() -> None:
    records = _population(2)
    with pytest.raises(ValueError, match="never mixed silently"):
        review.stratify(
            records, seed=1, target=99, class_targets={"false_alarm": 10},
        )
    scaled = review.stratify(records, seed=1, target=30)
    assert len(scaled.records) == 30
    with pytest.raises(ValueError, match="unknown sampling group"):
        review.stratify(records, seed=1, class_targets={"nonsense": 1})
    with pytest.raises(ValueError, match="unknown stratum key"):
        review.stratify(records, seed=1, strata_keys=("weather",))


def test_sample_groups_cover_every_outcome_class() -> None:
    """A class nothing allocates would be built and never drawn."""
    covered = {
        event_class
        for classes in review.SAMPLE_GROUPS.values()
        for event_class in classes
    }
    assert covered == set(OUTCOME_CLASSES)
    assert set(review.DEFAULT_CLASS_TARGETS) == set(review.SAMPLE_GROUPS)
    assert sum(review.DEFAULT_CLASS_TARGETS.values()) == 300


# ---------------------------------------------------------------------------
# 4. Pending, uncovered, dead gauges
# ---------------------------------------------------------------------------


def test_pending_is_dropped_and_uncovered_survives_as_a_control() -> None:
    """``pending``'s verdict is "ask again later"; ``uncovered``'s is
    "nobody was watching", and only one of those is worth a reviewer's
    time — but the other is the only audit the coverage rule ever gets."""
    population = review.synthetic_population()
    assert {record.event_class for record in population.records} == {
        "false_alarm", "hit", "late", "miss", "miss_late", "uncovered",
    }
    assert "pending" not in {record.event_class for record in population.records}
    uncovered = population.by_class("uncovered")
    assert len(uncovered) == 1
    assert uncovered[0].control is True

    sample = review.stratify(
        population.records, seed=5,
        class_targets={
            "false_alarm": 1, "miss": 1, "late": 1, "hit": 1, "uncovered": 1,
        },
    )
    assert sum(1 for r in sample.records if r.event_class == "uncovered") == 1


def test_a_warning_whose_window_has_not_closed_is_pending_not_an_event() -> None:
    """``known_until`` inside the promise means the gauge has not spoken."""
    rows = _steady(0.05, first=0, last=100) + _steady(0.9, first=110, last=140)
    station = review.station_meta(STATION, "Fixture", 55.33, 10.32)
    stamps = [
        slot_end_of(DAY + timedelta(minutes=m), slot_min=10)
        for m in range(10, 140, 10)
    ]
    truth = _truth_from_slots({STATION: (stamps, set())})
    population = review.build_population(
        rows=rows,
        stations=[station],
        truth=truth,
        rule=_curve_rule(),
    )
    assert population.records == ()
    assert population.excluded["pending_warnings"] == 1


def test_a_dead_gauge_contributes_no_events() -> None:
    """A gauge stuck at zero makes every radar-wet slot a false alarm.

    The rule is ``warning_score.dead_gauges``' — known often, wet never —
    and this asserts the wiring: the station it names is absent from
    ``known_until`` and so produces nothing at all, rather than a day of
    manufactured failures.
    """
    dry, wet = "06080", "06081"
    rows = [
        *(_steady(0.9, first=0, last=200, station=dry)),
        *(_steady(0.9, first=0, last=200, station=wet)),
    ]
    stamps = [
        slot_end_of(DAY + timedelta(minutes=m), slot_min=10)
        for m in range(0, 400, 10)
    ]
    wet_slots = {stamp for stamp in stamps if 200 <= (stamp - DAY).total_seconds() / 60 <= 240}
    truth = _truth_from_slots({dry: (stamps, set()), wet: (stamps, wet_slots)})
    assert dead_gauges(truth, min_known_slots=10) == [dry]

    stations = [
        review.station_meta(dry, "Dry", 55.33, 10.32),
        review.station_meta(wet, "Wet", 57.05, 9.90),
    ]
    population = review.build_population(
        rows=rows,
        stations=stations,
        truth=truth,
        rule=_curve_rule(),
        dead=[dry],
        onsets={
            station: list(truth.onsets.get(station, ()))
            for station in (wet,)
        },
        known_until={wet: truth.known_until[wet]},
    )
    assert {record.station_id for record in population.records} == {wet}
    assert population.excluded["stations_without_gauge"] == 1


# ---------------------------------------------------------------------------
# Small builders the population tests share
# ---------------------------------------------------------------------------


def _curve_rule(**kw) -> review.ReviewRule:
    """The fixture rule: the curve column, 50 %, the shipped timing."""
    options = {
        "lead_min": LEAD,
        "threshold_pct": 50,
        "probability": "curve",
        "min_useful_lead_min": 5.0,
        **kw,
    }
    return review.ReviewRule(**options)


def _truth_from_slots(
    spec: dict[str, tuple[list[datetime], set[datetime]]],
    *,
    unknown: dict[str, set[datetime]] | None = None,
):
    """A :class:`GaugeTruth` from ``{station: (slot grid, wet slots)}``."""
    from dmi_nowcast_core.warning_score import GaugeTruth

    series, onsets, known_until = {}, {}, {}
    for station, (grid, wet) in spec.items():
        silent = (unknown or {}).get(station, set())
        stamps = np.array([int(g.timestamp()) for g in grid], dtype=np.int64)
        mm = np.array(
            [
                np.nan if g in silent else (1.0 if g in wet else 0.0)
                for g in grid
            ],
            dtype=np.float32,
        )
        dur = np.array(
            [
                np.nan if g in silent else (10.0 if g in wet else 0.0)
                for g in grid
            ],
            dtype=np.float32,
        )
        known = np.array([g not in silent for g in grid], dtype=bool)
        wet_flags = np.array([g in wet for g in grid], dtype=bool)
        slots = StationSlots(
            station_id=station, slot_end=stamps, mm=mm, dur=dur,
            known=known, wet=wet_flags,
        )
        series[station] = slots
        onsets[station] = slots.onsets(60, onset_min_mm=0.2)
        reported = [g for g in grid if g not in silent]
        if reported:
            known_until[station] = reported[-1]
    return GaugeTruth(
        series=series, onsets=onsets, known_until=known_until,
        known_slots=sum(len(v[0]) for v in spec.values()),
    )


# ---------------------------------------------------------------------------
# 5. Decision-directory precedence
# ---------------------------------------------------------------------------


def _write_tree(directory: Path, rows: list[dict]) -> Path:
    """One day of decision rows as a parquet tree, features and all."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from dmi_nowcast_core import postprocess as pp
    from dmi_nowcast_core.warning_score import decision_table

    table = decision_table(rows, LEADS)
    for field in pp.feature_schema(LEADS):
        table = table.append_column(field, pa.array(
            [row.get(field.name) for row in rows], type=field.type,
        ))
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, directory / "day.parquet")
    return directory


def _write_gauge(corpus_dir: Path, wet_minutes: range | tuple = ()) -> Path:
    """A station store whose only station rains over ``wet_minutes``."""
    pytest.importorskip("pyarrow")
    from dmi_nowcast_core.station_store import StationObsStore

    observations = [
        Observation(
            station_id=STATION,
            observed_utc=DAY + timedelta(minutes=minute),
            parameter_id="precip_past10min",
            value=0.5 if minute in wet_minutes else 0.0,
        )
        for minute in range(0, 24 * 60, 10)
    ]
    StationObsStore(corpus_dir).append(observations)
    return corpus_dir


def test_the_last_decisions_dir_wins_and_the_duplicates_are_recorded(
    tmp_path: Path,
) -> None:
    """The precedence is the caller's, and it is the opposite of the page's.

    ``load_decisions`` lets the LATER directory win, so the caller passes
    the least authoritative tree FIRST. Here the first tree says the
    probability never crossed and the second says it did: with the honest
    tree last there is a warning, with it first there is none, and the
    duplicate count says how many rows changed hands.
    """
    pytest.importorskip("pyarrow")
    quiet = _write_tree(tmp_path / "live", _steady(0.05, first=0, last=200))
    loud = _write_tree(
        tmp_path / "replay",
        _steady(0.05, first=0, last=100) + _steady(0.95, first=110, last=200),
    )
    corpus = _write_gauge(tmp_path / "corpus", range(130, 180, 10))
    stations = [review.station_meta(STATION, "Fixture", 55.33, 10.32)]
    window = (DAY, DAY + timedelta(days=1))

    honest = review.load_population(
        decisions_dirs=[quiet, loud], decisions_labels=["live", "replay"],
        corpus_dir=corpus, stations=stations, rule=_curve_rule(),
        window=window,
    )
    backwards = review.load_population(
        decisions_dirs=[loud, quiet], corpus_dir=corpus,
        stations=stations, rule=_curve_rule(), window=window,
    )

    assert [r.event_class for r in honest.records] == ["hit"]
    assert [r.event_class for r in backwards.records] == ["miss"]

    corpus_block = honest.provenance["corpus"]
    assert corpus_block["decisions_dirs"] == [str(quiet), str(loud)]
    assert "LAST directory wins" in corpus_block["decisions_precedence"]
    assert corpus_block["duplicates_dropped"] == len(
        _steady(0.05, first=0, last=200)
    )
    assert corpus_block["rows_read"] == 2 * corpus_block["duplicates_dropped"]
    assert corpus_block["decisions_labels"] == ["live", "replay"]

    # And every row says which tree it came out of, because "was this
    # judged on a replay row with features or a live row from the gap?" is
    # the question the feature gap makes unavoidable.
    assert set(honest.row_sources.values()) == {"replay"}
    doc = review.build_event(honest.records[0], honest)
    assert {entry["row_source"] for entry in doc["decisions"]} == {"replay"}


def test_gauge_truth_for_matches_the_threshold_sweep(tmp_path: Path) -> None:
    """The review's truth is the sweep's truth, with the grid kept.

    Same call, same pad, same dead-gauge rule — the only difference is
    that the :class:`GaugeTruth` survives, because the review needs the
    slot grid and reading the archive twice to get it would double the
    build.
    """
    pytest.importorskip("pyarrow")
    from dmi_nowcast_sidecar.threshold_sweep import gauge_truth

    corpus = _write_gauge(tmp_path / "corpus", range(130, 180, 10))
    window = (DAY, DAY + timedelta(days=1))
    expected = gauge_truth(corpus, [STATION], window)
    truth, onsets, known_until, dead = review.gauge_truth_for(
        corpus, [STATION], window,
    )
    assert onsets == expected[0]
    assert known_until == expected[1]
    assert truth.known_slots == expected[2]
    assert dead == expected[3]
    assert truth.series[STATION].slots(), "the grid itself is what this adds"


# ---------------------------------------------------------------------------
# 6. The feature gap
# ---------------------------------------------------------------------------


def test_feature_gap_days_are_measured_per_day() -> None:
    """Counted, not assumed: the repository cannot say when the fix landed."""
    good = [_row(minute, 0.1) for minute in range(0, 600, 10)]
    bad = [
        _row(minute, 0.1, start=DAY + timedelta(days=1), features=False)
        for minute in range(0, 600, 10)
    ]
    half = [
        _row(minute, 0.1, start=DAY + timedelta(days=2), features=minute % 20 == 0)
        for minute in range(0, 600, 10)
    ]
    counts, gaps = review.feature_gap_days(good + bad + half)
    assert gaps == (DAY.date() + timedelta(days=1),)
    assert counts[DAY.date()] == (60, 60)
    assert counts[DAY.date() + timedelta(days=1)] == (60, 0)
    # Exactly half missing is not "more than half": the boundary is stated.
    assert counts[DAY.date() + timedelta(days=2)] == (60, 30)


def test_a_feature_gap_day_is_excluded_by_default_and_flagged_when_kept() -> None:
    excluded = review.synthetic_population(feature_gap_hours=(0, 24))
    assert excluded.records == ()
    assert excluded.excluded["feature_gap_events"] > 0
    assert excluded.provenance["features"]["gap_days"] == [DAY.date().isoformat()]

    kept = review.synthetic_population(
        feature_gap_hours=(0, 24), allow_feature_gap=True,
    )
    assert len(kept.records) == excluded.excluded["feature_gap_events"]
    assert all("feature_gap" in record.flags for record in kept.records)
    assert "feature_gap" in EVENT_FLAGS


# ---------------------------------------------------------------------------
# 7. Dual truth, under both window definitions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gauge,radar,expected",
    [
        (True, True, "both_wet"),
        (True, False, "gauge_wet_radar_dry"),
        (False, True, "radar_wet_gauge_dry"),
        (False, False, "both_dry"),
        (None, True, None),
        (True, None, None),
        (None, None, None),
    ],
)
def test_dual_truth_table(gauge, radar, expected) -> None:
    """The 2×2, plus the two ways an instrument can say nothing."""
    assert review.dual_truth_of(gauge, radar) == expected
    if expected is not None:
        assert expected in DUAL_TRUTH_CLASSES


def test_the_verdict_window_differs_by_side_and_changes_the_answer() -> None:
    """Same rain, same station, two anchors, two verdicts.

    A warning's window looks FORWARD over the promise it made; an onset's
    window sits on the rain event itself. Rain 35 minutes after a warning
    is inside the promise and nowhere near an onset 3 hours later — and an
    event that did not carry ``window_used`` could not tell the two
    readings apart.
    """
    sent = DAY + timedelta(minutes=60)
    onset = DAY + timedelta(minutes=240)
    warning = review.warning_window(sent, lead_min=LEAD, tolerance_min=10)
    onset_win = review.onset_window(onset, tolerance_min=10)
    assert warning.kind == "warning" and onset_win.kind == "onset"
    assert warning.to_utc == sent + timedelta(minutes=40)
    assert onset_win.from_utc == onset - timedelta(minutes=20)
    assert onset_win.to_utc == onset + timedelta(minutes=20)

    grid = [
        slot_end_of(DAY + timedelta(minutes=m), slot_min=10)
        for m in range(10, 400, 10)
    ]
    near_the_warning = {DAY + timedelta(minutes=90)}
    near_the_onset = {DAY + timedelta(minutes=240)}
    for wet, in_warning, in_onset in (
        (near_the_warning, True, False),
        (near_the_onset, False, True),
    ):
        truth = _truth_from_slots({STATION: (grid, wet)})
        slots = truth.series[STATION]
        assert review.gauge_wet_between(
            slots, warning.from_utc, warning.to_utc,
        ) is in_warning
        assert review.gauge_wet_between(
            slots, onset_win.from_utc, onset_win.to_utc,
        ) is in_onset

    # And the radar disc, read over the same two windows off the rows.
    rows = [
        _row(minute, 0.1, observed=2.0 if 80 <= minute <= 100 else 0.0)
        for minute in range(0, 400, 10)
    ]
    radar = review.radar_slots_of(_track(rows), STATION)
    assert radar.wet_between(warning.from_utc, warning.to_utc) is True
    assert radar.wet_between(onset_win.from_utc, onset_win.to_utc) is False


def test_a_silent_gauge_is_not_a_dry_one() -> None:
    """``both_dry`` is the most damaging verdict available; it must be
    earned, not defaulted to when the station said nothing."""
    grid = [
        slot_end_of(DAY + timedelta(minutes=m), slot_min=10)
        for m in range(10, 400, 10)
    ]
    truth = _truth_from_slots(
        {STATION: (grid, set())}, unknown={STATION: set(grid)},
    )
    window = review.warning_window(
        DAY + timedelta(minutes=60), lead_min=LEAD, tolerance_min=10,
    )
    assert review.gauge_wet_between(
        truth.series[STATION], window.from_utc, window.to_utc,
    ) is None
    assert review.dual_truth_of(None, False) is None


# ---------------------------------------------------------------------------
# 8. Stamps: frame age and slot ends
# ---------------------------------------------------------------------------


def test_generated_at_is_radar_ts_plus_frame_age() -> None:
    """The decision instant is derived, and the stored column agrees."""
    population = review.synthetic_population()
    record = population.by_class("false_alarm")[0]
    doc = review.build_event(record, population)
    assert doc["decisions"], "the window has decisions in it"
    for entry in doc["decisions"]:
        radar_ts = datetime.fromisoformat(entry["radar_ts_utc"])
        generated = datetime.fromisoformat(entry["generated_at_utc"])
        assert generated == radar_ts + timedelta(minutes=entry["frame_age_min"])
        assert entry["frame_age_min"] == pytest.approx(
            entry["frame_age_min_stored"],
        )
    assert not [
        note for note in doc["builder_notes"] if "disagrees" in note
    ], "a fixture whose stamps disagree would be a fixture bug"


def test_gauge_slots_are_stamped_at_their_end() -> None:
    """An onset instant IS a slot end, and the slot covers the 10 minutes
    before it — which is why every measured lead is biased negative."""
    population = review.synthetic_population()
    record = population.by_class("miss")[0]
    doc = review.build_event(record, population)
    onset = datetime.fromisoformat(doc["gauge"]["onset_utc"])
    assert onset == slot_end_of(onset, slot_min=10)
    stamps = [
        datetime.fromisoformat(slot["slot_end_utc"])
        for slot in doc["gauge"]["slots"]
    ]
    assert stamps == sorted(stamps)
    assert all(
        (b - a) == timedelta(minutes=10) for a, b in zip(stamps, stamps[1:])
    )
    wet = next(
        slot for slot in doc["gauge"]["slots"]
        if slot["slot_end_utc"] == doc["gauge"]["onset_utc"]
    )
    assert wet["wet"] is True and wet["known"] is True


def test_the_latest_estimate_is_older_than_the_frame_it_sits_on() -> None:
    """Plan §4.5: the picture and the words come from different cycles."""
    population = review.synthetic_population()
    record = population.by_class("false_alarm")[0]
    doc = review.build_event(record, population)
    estimates = [
        entry["latest_estimate"] for entry in doc["decisions"]
        if entry["latest_estimate"] is not None
    ]
    assert estimates
    for entry, estimate in zip(doc["decisions"], estimates):
        assert estimate["age_min"] >= 0
        assert datetime.fromisoformat(estimate["generated_at_utc"]) <= (
            datetime.fromisoformat(entry["radar_ts_utc"])
        )


# ---------------------------------------------------------------------------
# 9. Null is never zero
# ---------------------------------------------------------------------------


def test_nulls_survive_into_the_detail_document() -> None:
    """Every ``None`` here is a measurement that was not made."""
    rows = _steady(0.05, first=0, last=60)
    rows += [
        _row(minute, None, observed=None, eta=None, features=False)
        for minute in range(70, 130, 10)
    ]
    rows += _steady(0.9, first=140, last=200)
    grid = [
        slot_end_of(DAY + timedelta(minutes=m), slot_min=10)
        for m in range(0, 400, 10)
    ]
    truth = _truth_from_slots(
        {STATION: (grid, set())},
        # A band of silence AFTER the warning window has closed, so the
        # event is graded and the detail still carries unknown slots.
        unknown={
            STATION: {
                g for g in grid
                if 220 <= (g - DAY).total_seconds() / 60 <= 260
            }
        },
    )
    station = review.station_meta(STATION, "Fixture", 55.33, 10.32)
    population = review.build_population(
        rows=rows, stations=[station], truth=truth, rule=_curve_rule(),
    )
    record = population.by_class("false_alarm")[0]
    doc = review.build_event(record, population)

    skipped = [
        entry for entry in doc["decisions"]
        if entry["replay"]["skipped_reason"] == review.SKIP_NO_PROBABILITY
    ]
    assert skipped
    for entry in skipped:
        assert entry["p_decision"] is None
        assert entry["over_threshold"] is None
        assert entry["observed_mm_h"] is None
        assert entry["eta_min"] is None
        assert entry["replay"]["action"] is None
        assert entry["features"]["obs_max_5km_mm_h"] is None
        # A null probability is not a zero one, and a null feature is not
        # a dry pixel: a builder that coerced either would have written 0.
        assert 0 not in (
            entry["p_decision"], entry["observed_mm_h"],
            entry["features"]["obs_max_5km_mm_h"],
        )
    unknown_slots = [
        slot for slot in doc["gauge"]["slots"] if not slot["known"]
    ]
    assert unknown_slots
    for slot in unknown_slots:
        # ``review_schema`` fixes the encoding: {"wet": false, "known":
        # false} is UNKNOWN. What must never be invented is the DEPTH —
        # a gauge that said nothing did not weigh zero millimetres.
        assert slot["wet"] is False
        assert slot["known"] is False
        assert slot["mm"] is None
    assert "no_probability" in record.flags
    assert json.dumps(doc), "the document must survive a JSON round trip"


def test_an_uncovered_event_has_no_radar_verdict_at_all() -> None:
    """No decision row was watching, so there is no disc reading to read —
    and ``radar_wet_in_window`` is null rather than false."""
    population = review.synthetic_population()
    record = population.by_class("uncovered")[0]
    assert record.radar_wet_in_window is None
    assert record.dual_truth is None
    doc = review.build_event(record, population)
    assert doc["dual_truth"]["class"] is None
    assert doc["dual_truth"]["radar_wet"] is None
    assert any("null" in note for note in doc["builder_notes"])


# ---------------------------------------------------------------------------
# 10. Decision gaps
# ---------------------------------------------------------------------------


def test_a_decision_gap_inside_the_window_is_reported() -> None:
    population = review.synthetic_population()
    record = population.by_class("uncovered")[0]
    doc = review.build_event(record, population)
    gaps = doc["decision_gaps"]
    assert gaps, "the uncovered role's window sits inside a four-hour outage"
    assert any(gap["coverage_break"] for gap in gaps)
    assert any(
        gap["edge"] in ("leading", "trailing", "whole_window") for gap in gaps
    ), "the outage runs to the window's edge, which is where it shows"
    assert "coverage_gap" in record.flags


def test_a_single_missed_cycle_is_a_gap_but_not_a_coverage_break() -> None:
    rows = _steady(0.05, first=0, last=100)
    rows += _steady(0.9, first=110, last=140)
    # One missing cycle at 180, then the rest.
    rows += [_row(minute, 0.05) for minute in range(150, 300, 10) if minute != 180]
    track = _track(rows)
    gaps = review._decision_gaps(
        track,
        DAY,
        DAY + timedelta(minutes=300),
        gap_min=10,
        coverage_gap_min=20,
    )
    holes = [gap for gap in gaps if gap["edge"] is None]
    assert len(holes) == 1
    assert holes[0]["minutes"] == pytest.approx(20.0)
    assert holes[0]["coverage_break"] is False


# ---------------------------------------------------------------------------
# 11. The prologue
# ---------------------------------------------------------------------------


def test_the_prologue_reports_a_disarming_action_outside_the_window() -> None:
    """The showery spell, end to end.

    The station fires at 00:00 and then bounces over the threshold every
    forty minutes for four hours, so the dry clock never accumulates and
    the rain at 05:00 arrives at a subscription that has been disarmed
    since before the window began. Without the prologue the reviewer sees
    a station that inexplicably never fires and blames the forecast for
    the hysteresis.
    """
    rows = [_row(0, 0.95)]
    minute = 10
    while minute < 300:
        rows += [_row(minute + 10 * i, 0.1) for i in range(4)]
        rows.append(_row(minute + 40, 0.8))
        minute += 50
    rows += _steady(0.1, first=minute, last=minute + 200, observed=2.0)

    grid = [
        slot_end_of(DAY + timedelta(minutes=m), slot_min=10)
        for m in range(0, 600, 10)
    ]
    wet = {g for g in grid if 300 <= (g - DAY).total_seconds() / 60 <= 340}
    truth = _truth_from_slots({STATION: (grid, wet)})
    station = review.station_meta(STATION, "Fixture", 55.33, 10.32)
    population = review.build_population(
        rows=rows, stations=[station], truth=truth, rule=_curve_rule(),
    )
    misses = population.by_class("miss")
    assert misses, "the rain at 05:00 was never warned about"
    record = misses[0]
    assert record.arm_state_at_anchor == "disarmed"

    doc = review.build_event(record, population)
    prologue = doc["prologue"]
    assert prologue["armed_at_window_start"] is False
    assert prologue["at_anchor"]["armed"] is False
    last_notify = prologue["last_notify_utc"]
    assert last_notify is not None, "the disarming push must be reported"
    assert datetime.fromisoformat(last_notify) < (
        datetime.fromisoformat(doc["window"]["from_utc"])
    ), (
        "the whole point: the action that explains this miss lies outside "
        "the ±90 minute window"
    )
    assert 0 < len(prologue["recent_actions"]) <= 5
    assert prologue["rearm_after_min"] == REARM


def test_the_prologue_reports_an_already_raining_disarm() -> None:
    """The arm can be consumed silently, and then nothing ever fires.

    ``evaluate`` disarms on ``already_raining`` exactly as it does on
    ``notify``, with no notification to show for it. Here the disc is
    measuring rain at 01:00 while the gauge stays dry — virga, or a shower
    that missed the gauge — and the arm is spent there. Four hours later
    real rain arrives at a subscription that has been disarmed ever since,
    and only the prologue can say why. ``miss_arm_consumed_already_raining``
    is a first-class tag precisely because of this shape.
    """
    rows = _steady(0.05, first=0, last=50)
    rows += _steady(0.9, first=60, last=70, observed=2.0)  # silent disarm
    minute = 80
    while minute < 290:
        rows += [_row(minute + 10 * i, 0.1) for i in range(4)]
        rows.append(_row(minute + 40, 0.8))  # keeps the dry clock reset
        minute += 50
    rows += _steady(0.1, first=minute, last=minute + 200)

    grid = [
        slot_end_of(DAY + timedelta(minutes=m), slot_min=10)
        for m in range(0, 600, 10)
    ]
    wet = {g for g in grid if 300 <= (g - DAY).total_seconds() / 60 <= 340}
    truth = _truth_from_slots({STATION: (grid, wet)})
    station = review.station_meta(STATION, "Fixture", 55.33, 10.32)
    population = review.build_population(
        rows=rows, stations=[station], truth=truth, rule=_curve_rule(),
    )
    trace = population.traces[STATION]
    assert [row.action for row in trace].count("already_raining") == 1
    assert "notify" not in [row.action for row in trace], (
        "the arm was consumed without a notification"
    )

    misses = population.by_class("miss")
    assert misses
    doc = review.build_event(misses[0], population)
    silent = doc["prologue"]["last_already_raining_utc"]
    assert silent is not None
    assert datetime.fromisoformat(silent) < (
        datetime.fromisoformat(doc["window"]["from_utc"])
    ), "the arm was consumed before the window even began"
    assert doc["prologue"]["last_notify_utc"] is None
    assert doc["prologue"]["armed_at_window_start"] is False


def test_a_free_re_arm_at_a_run_boundary_is_flagged_on_the_event() -> None:
    """The replay's re-arm is not the service's, and it must say so.

    The station fires at 00:00 and stays disarmed — every later row is
    over threshold, so the dry clock never starts. Three hours of outage
    later the replay resets to ``INITIAL_STATE`` and the very first row of
    the new coverage run fires again. The live service, carrying its state
    across the outage, would have sent nothing. Without the flag a
    reviewer reads an impossible notification as a bug in the tool.
    """
    rows = [_row(0, 0.9)]
    rows += _steady(0.9, first=10, last=120)
    rows += _steady(0.9, first=300, last=420)
    grid = [
        slot_end_of(DAY + timedelta(minutes=m), slot_min=10)
        for m in range(0, 600, 10)
    ]
    truth = _truth_from_slots({STATION: (grid, set())})
    station = review.station_meta(STATION, "Fixture", 55.33, 10.32)
    population = review.build_population(
        rows=rows, stations=[station], truth=truth, rule=_curve_rule(),
    )
    trace = population.traces[STATION]
    notifies = [row for row in trace if row.action == "notify"]
    assert len(notifies) == 2, "one per coverage run"
    second = notifies[1]
    assert second.radar_ts in review.run_boundary_rearms(trace)

    record = next(
        r for r in population.records
        if r.sent_utc == second.generated_at
    )
    assert "run_boundary_rearm" in record.flags
    doc = review.build_event(record, population)
    flagged = [
        entry for entry in doc["decisions"]
        if entry["replay"]["run_boundary_rearm"]
    ]
    assert [entry["radar_ts_utc"] for entry in flagged] == [
        second.radar_ts.isoformat()
    ]
    assert doc["prologue"]["run_boundary_rearms_utc"] == [
        second.radar_ts.isoformat()
    ]


# ---------------------------------------------------------------------------
# The document's own shape
# ---------------------------------------------------------------------------


def test_the_fixture_seed_moves_the_numbers_and_not_the_classes() -> None:
    """A fixture whose class mix depended on the RNG could assert nothing."""
    first = review.synthetic_population(seed=1)
    second = review.synthetic_population(seed=2)
    classes = lambda pop: sorted(  # noqa: E731 — one expression, read twice
        (record.event_class, record.station_id) for record in pop.records
    )
    assert classes(first) == classes(second)
    assert [r.p_decision for r in sorted(first.records, key=lambda r: r.event_id)] != (
        [r.p_decision for r in sorted(second.records, key=lambda r: r.event_id)]
    )
    again = review.synthetic_population(seed=1)
    assert [r.p_decision for r in sorted(again.records, key=lambda r: r.event_id)] == (
        [r.p_decision for r in sorted(first.records, key=lambda r: r.event_id)]
    )


def test_the_index_row_is_exactly_the_schema_s_fields_in_order() -> None:
    population = review.synthetic_population()
    record = population.records[0]
    row = review.index_row(record, population.stations[record.station_id])
    assert tuple(row) == INDEX_FIELDS


def test_the_detail_document_has_every_block_the_schema_promises() -> None:
    population = review.synthetic_population()
    for record in population.records:
        doc = review.build_event(record, population)
        for block in (
            "index", "station", "window", "decisions", "decision_gaps",
            "prologue", "gauge", "radar_disc", "neighbours", "dual_truth",
            "notifications", "frames", "flags", "builder_notes",
        ):
            assert block in doc, f"{record.event_id} is missing {block}"
        assert doc["index"]["event_id"] == record.event_id
        assert doc["window"]["verdict"]["kind"] in ("warning", "onset")
        # The frame window reaches back past the decision window, because a
        # decision at its left edge stands on an older composite.
        assert datetime.fromisoformat(doc["window"]["frames_from_utc"]) < (
            datetime.fromisoformat(doc["window"]["from_utc"])
        )
        assert json.dumps(doc)


#: The fields the browser's own copy of the contract
#: (``frontend/src/lib/review/schema.ts``) reads out of each block. Listed
#: here rather than parsed out of the TypeScript so this suite stays a
#: Python suite — and pinned, because four consumers sharing one contract
#: is the entire reason ``review_schema`` exists.
_CONTRACT_FIELDS: dict[str, tuple[str, ...]] = {
    "station": (
        "station_id", "name", "lat", "lon", "region", "station_radar_km",
        "grid", "neighbours",
    ),
    "window": (
        "anchor_utc", "from_utc", "to_utc", "frames_from_utc",
        "slot_from_utc", "slot_to_utc", "known_until_utc",
    ),
    "prologue": (
        "run_id", "run_start_utc", "armed_at_window_start",
        "streak_at_window_start", "below_since_utc",
        "minutes_to_rearm_at_window_start", "last_notify_utc",
        "last_already_raining_utc", "recent_actions", "rearm_after_min",
        "at_anchor", "run_boundary_rearms_utc", "note",
    ),
    "gauge": (
        "slot_min", "slots", "known_until_utc", "wet_slots_in_window",
        "onsets",
    ),
    "radar_disc": (
        "disc_radius_m", "statistic", "threshold_mm_h", "series", "slots",
        "wet_in_window", "first_wet_utc",
    ),
    "neighbours": (
        "radius_km", "any_wet_in_window", "n_known", "n_wet", "stations",
    ),
    "dual_truth": (
        "class", "gauge_wet", "radar_wet", "neighbour_wet", "window_used",
        "rule",
    ),
}

_DECISION_FIELDS = (
    "radar_ts_utc", "generated_at_utc", "frame_age_min", "frame_age_source",
    "frame_ref", "row_source", "p_rain", "p_post", "p_decision",
    "p_decision_source", "p_decision_lead_min", "threshold_pct",
    "over_threshold", "eta_min", "eta_arrival_utc", "intensity_mm_h",
    "observed_mm_h", "forecast_now_mm_h", "features", "features_present",
    "replay", "stored",
)


def test_the_detail_document_carries_the_field_names_the_browser_reads() -> None:
    """The bundle is a contract between four programs, not a dict.

    ``frontend/src/lib/review/schema.ts`` is the consumer that has to
    parse this, and a builder that renamed ``replay`` to ``engine`` or
    wrote ``stratum`` as a joined string would produce a bundle the page
    cannot render — with no error anywhere near the cause.
    """
    population = review.synthetic_population()
    doc = review.build_event(population.records[0], population)
    assert doc["event_id"] == population.records[0].event_id
    assert "bundle_id" in doc
    for block, fields in _CONTRACT_FIELDS.items():
        missing = [name for name in fields if name not in doc[block]]
        assert not missing, f"{block} is missing {missing}"
    for entry in doc["decisions"]:
        missing = [name for name in _DECISION_FIELDS if name not in entry]
        assert not missing, f"a decision is missing {missing}"
        for name in (
            "run_id", "run_boundary_rearm", "armed_before", "streak_before",
            "armed_after", "streak_after", "below_since_utc", "action",
            "skipped_reason",
        ):
            assert name in entry["replay"], name
        for name in (
            "action", "armed_after", "streak_after", "threshold_pct",
            "p_rain_at_rule_lead",
        ):
            assert name in entry["stored"], name
    for gap in doc["decision_gaps"]:
        assert {"from_utc", "to_utc", "minutes", "reason"} <= set(gap)
    for slot in doc["gauge"]["slots"]:
        assert {"slot_end_utc", "mm", "known", "wet"} <= set(slot)
        assert isinstance(slot["wet"], bool), "wet is a bool beside known"
    for marker in doc["notifications"]:
        assert {"kind", "generated_at_utc", "action", "is_event_warning"} <= set(
            marker
        )
        assert marker["kind"] in ("replayed", "stored")
    for frame in doc["frames"]:
        assert {"radar_ts_utc", "stamp", "overlay", "observed", "present"} <= set(
            frame
        )
    # The index row's stratum is a mapping the browser facets on.
    assert doc["index"]["stratum"] == {
        key: population.records[0].stratum_value(key) for key in STRATA_KEYS
    }
    # One spelling per fact in the prologue. Two names for one instant is
    # how the two start disagreeing after an edit, and this is the block a
    # reviewer leans on to tell an unreachable miss from a real one.
    assert set(doc["prologue"]) == set(_CONTRACT_FIELDS["prologue"])
    for entry in doc["prologue"]["recent_actions"]:
        assert set(entry) == {"generated_at_utc", "action", "p_decision"}
    assert set(doc["prologue"]["at_anchor"]) == {
        "armed", "streak", "minutes_to_rearm",
    }


def test_the_nullable_contract_fields_are_really_null_when_unknown() -> None:
    """Six fields the browser types as nullable, and why each one is.

    Nullability is not a formality here: every one of these is a place
    where a plausible number would read as a measurement. A gauge that
    reported nothing is not a dry gauge, a row the engine never evaluated
    was never under the threshold, a station with no product grid has no
    pixel, and a disc read off a stored column has no pixel counts behind
    it.
    """
    # 1-3. A silent gauge: no verdict, no quadrant, and the index carries
    # the null too — it is the field the list filters on.
    grid = [
        slot_end_of(DAY + timedelta(minutes=m), slot_min=10)
        for m in range(0, 400, 10)
    ]
    silent = _truth_from_slots(
        {STATION: (grid, set())}, unknown={STATION: set(grid)},
    )
    # ``known_until`` is pinned by hand: a station that reported nothing
    # has none, and without one nothing here would be scored at all.
    rows = _steady(0.05, first=0, last=100) + _steady(0.9, first=110, last=200)
    station = review.station_meta(STATION, "Fixture", 55.33, 10.32)
    population = review.build_population(
        rows=rows, stations=[station], truth=silent, rule=_curve_rule(),
        known_until={STATION: grid[-1]}, onsets={STATION: []},
    )
    record = population.by_class("false_alarm")[0]
    assert record.gauge_wet_in_window is None
    assert record.dual_truth is None
    doc = review.build_event(record, population)
    assert doc["index"]["gauge_wet_in_window"] is None
    assert doc["index"]["dual_truth"] is None
    assert doc["dual_truth"]["class"] is None
    assert doc["dual_truth"]["gauge_wet"] is None

    # 4. A row with no probability was never compared with the threshold.
    skipped = review.build_event(
        *_population_with_skipped_rows(),
    )
    assert [
        entry["over_threshold"] for entry in skipped["decisions"]
        if entry["replay"]["skipped_reason"] is not None
    ] and all(
        entry["over_threshold"] is None
        for entry in skipped["decisions"]
        if entry["replay"]["skipped_reason"] is not None
    )

    # 5-6. The grid, the disc's pixel counts and the frames: all the frame
    # writer's to fill, and null until it does.
    assert doc["station"]["grid"] == {"row": None, "col": None}
    assert doc["radar_disc"]["series"]
    for entry in doc["radar_disc"]["series"]:
        assert entry["n_pixels"] is None and entry["n_valid"] is None
        assert entry["max_mm_h"] is None and entry["mean_mm_h"] is None
    assert doc["frames"]
    for frame in doc["frames"]:
        assert frame["product"] is None
        assert frame["present"] is None


def _population_with_skipped_rows() -> tuple[review.EventRecord, review.Population]:
    """A population whose window holds rows the engine never evaluated."""
    rows = _steady(0.05, first=0, last=60)
    rows += [
        _row(minute, None, observed=None, eta=None, features=False)
        for minute in range(70, 130, 10)
    ]
    rows += _steady(0.9, first=140, last=200)
    grid = [
        slot_end_of(DAY + timedelta(minutes=m), slot_min=10)
        for m in range(0, 400, 10)
    ]
    truth = _truth_from_slots({STATION: (grid, set())})
    station = review.station_meta(STATION, "Fixture", 55.33, 10.32)
    population = review.build_population(
        rows=rows, stations=[station], truth=truth, rule=_curve_rule(),
    )
    return population.by_class("false_alarm")[0], population


def test_every_flag_the_builder_sets_is_documented_in_the_schema() -> None:
    population = review.synthetic_population(
        feature_gap_hours=(0, 24), allow_feature_gap=True,
    )
    used = {flag for record in population.records for flag in record.flags}
    assert used, "the fixture raises at least one flag"
    assert used <= set(EVENT_FLAGS)


def test_the_frame_window_is_the_frame_planner_s_own_edge() -> None:
    """One edge, computed once — or the reviewer scrubs into a blank map.

    ``review_frames.frame_plan`` pads the past edge by ``window + pad +
    cadence`` because an anchor is almost never on the 10-minute grid.
    A detail document that named ``window + pad`` instead would promise a
    frame the bundle never rendered, so the two are pinned together here —
    on a deliberately off-grid anchor, which is the case that separates
    them.
    """
    from dmi_nowcast_sidecar.review_frames import frame_plan, stamp_of

    population = review.synthetic_population()
    record = population.by_class("false_alarm")[0]
    assert record.anchor_utc.minute % 10 != 0, "an off-grid anchor is the case"

    plan = frame_plan([(record.event_id, record.anchor_utc)])
    planned = plan.by_event_id()[record.event_id]
    doc = review.build_event(record, population)
    assert doc["window"]["frames_from_utc"] == planned.frames_from_utc.isoformat()
    assert doc["window"]["frames_to_utc"] == planned.frames_to_utc.isoformat()
    assert doc["window"]["from_utc"] == planned.window_from_utc.isoformat()
    assert doc["window"]["to_utc"] == planned.window_to_utc.isoformat()

    # Handed the planner's own event, the document repeats its stamps
    # rather than re-deriving a second list from the decision rows.
    with_plan = review.build_event(record, population, event_frames=planned)
    assert [frame["source"] for frame in with_plan["frames"]] == (
        ["frame planner"] * planned.n_frames
    )
    assert [frame["stamp"] for frame in with_plan["frames"]] == list(planned.stamps)
    assert with_plan["index"]["frames"] == planned.n_frames
    assert all(
        frame["stamp"] == stamp_of(datetime.fromisoformat(frame["radar_ts_utc"]))
        for frame in with_plan["frames"]
    ), "the builder's stamp spelling is the frame writer's"
    assert with_plan["frames"][0]["overlay"].endswith(".overlay.png")

    # Whether a frame was RENDERED is the writer's word, and it is not the
    # same fact as whether a decision stood on it.
    assert all(frame["present"] is None for frame in with_plan["frames"])
    absent = datetime.fromisoformat(with_plan["frames"][2]["radar_ts_utc"])
    written = review.build_event(
        record, population, event_frames=planned, missing_stamps=[absent],
    )
    missing = [f for f in written["frames"] if f["present"] is False]
    assert [f["radar_ts_utc"] for f in missing] == [absent.isoformat()]
    assert written["index"]["frames_missing"] == 1
    assert missing[0]["present_source"] == "frame writer"


def test_the_station_pixel_is_null_until_the_frame_writer_fills_it() -> None:
    """A guess would be worse than a null: the map would misplace the dot."""
    population = review.synthetic_population()
    record = population.records[0]
    plain = review.build_event(record, population)
    assert plain["station"]["grid"] == {"row": None, "col": None}
    placed = review.build_event(
        record, population, station_pixel=lambda lat, lon: (12.5, 7.25),
    )
    assert placed["station"]["grid"] == {"row": 12.5, "col": 7.25}


# ---------------------------------------------------------------------------
# The rule object
# ---------------------------------------------------------------------------


def test_served_and_lomo_are_never_mixed_silently() -> None:
    with pytest.raises(ValueError, match="fold_thresholds"):
        review.ReviewRule(rule_source=review.RULE_LOMO)
    with pytest.raises(ValueError, match="never mixed"):
        review.ReviewRule(
            rule_source=review.RULE_SERVED,
            fold_thresholds=review.fold_thresholds_of({(2026, 6): 50}),
        )
    served = review.ReviewRule()
    assert served.held_out is False
    assert served.document()["held_out"] is False

    lomo = review.ReviewRule(
        rule_source=review.RULE_LOMO,
        fold_thresholds=review.fold_thresholds_of({(2026, 6): 55, (2026, 7): 65}),
    )
    assert lomo.held_out is True
    assert lomo.threshold_for(DAY) == 55
    assert lomo.threshold_for(DAY + timedelta(days=40)) == 65
    assert lomo.document()["fold_thresholds"] == {"2026-06": 55, "2026-07": 65}


def test_a_leave_one_month_out_rule_replays_each_month_under_its_own_threshold() -> None:
    """Each month is a self-contained slice: its own runs, its own arming.

    The June threshold here is low enough to fire and July's is not, so a
    rule that quietly used one number for both would show up as two
    warnings or none.
    """
    june = _steady(0.6, first=0, last=200)
    july = _steady(0.6, first=0, last=200, start=datetime(2026, 7, 1, tzinfo=UTC))
    grid = [
        slot_end_of(DAY + timedelta(minutes=m), slot_min=10)
        for m in range(0, 600, 10)
    ] + [
        slot_end_of(datetime(2026, 7, 1, tzinfo=UTC) + timedelta(minutes=m), slot_min=10)
        for m in range(0, 600, 10)
    ]
    truth = _truth_from_slots({STATION: (grid, set())})
    station = review.station_meta(STATION, "Fixture", 55.33, 10.32)
    rule = _curve_rule(
        rule_source=review.RULE_LOMO,
        fold_thresholds=review.fold_thresholds_of({(2026, 6): 50, (2026, 7): 90}),
    )
    population = review.build_population(
        rows=june + july, stations=[station], truth=truth, rule=rule,
    )
    sent = [record.sent_utc for record in population.records]
    assert len(sent) == 1
    assert sent[0].month == 6
    assert population.records[0].threshold_pct == 50
    assert population.provenance["rule"]["held_out"] is True

    with pytest.raises(KeyError, match="2026-07"):
        review.build_population(
            rows=june + july, stations=[station], truth=truth,
            rule=_curve_rule(
                rule_source=review.RULE_LOMO,
                fold_thresholds=review.fold_thresholds_of({(2026, 6): 50}),
            ),
        )


def test_rule_from_served_options_borrows_the_page_s_rule() -> None:
    """The review's rule comes from the same object the page decides with."""
    from dmi_nowcast_sidecar.served_rule import ServedRuleOptions

    options = ServedRuleOptions(
        lead_min=45, probability_source="postprocess",
        persistence_obs=1, rearm_after_min=60, fallback_threshold_pct=70,
    )
    rule = review.rule_from_served_options(options)
    assert rule.lead_min == 45
    assert rule.threshold_pct == 70
    assert rule.threshold_source == "config"
    assert rule.column == "p_post_45"
    assert rule.curve_column == "p_rain_45"


def test_the_engine_s_per_row_fallback_is_applied_before_the_replay() -> None:
    """A row the model could not speak for decides on the curve, and says so.

    That is ``Observation.p_decision``'s rule, applied as a column exactly
    as ``served_rule`` applies it — and the event has to record WHICH
    scale it was judged on, because the thresholds were fitted on the
    post-processed one.
    """
    post = "p_post_30"
    quiet = _steady(0.05, first=0, last=100)
    loud = _steady(0.9, first=110, last=200)
    for row in quiet:
        # The model spoke for the quiet rows and not for the loud ones, so
        # the warning itself is decided on the curve — which is the row the
        # thresholds were NOT fitted for, and the event has to say so.
        row[post] = row["p_rain"]
    rows = quiet + loud
    grid = [
        slot_end_of(DAY + timedelta(minutes=m), slot_min=10)
        for m in range(0, 400, 10)
    ]
    truth = _truth_from_slots({STATION: (grid, set())})
    station = review.station_meta(STATION, "Fixture", 55.33, 10.32)
    population = review.build_population(
        rows=rows, stations=[station], truth=truth,
        rule=_curve_rule(probability="postprocess"),
    )
    assert population.records, "the curve fallback must not silence the rule"
    record = population.records[0]
    assert record.p_decision_source == "curve"
    assert record.p_decision == pytest.approx(0.9)
    assert len(population.curve_fallback_keys) == len(loud)


# ---------------------------------------------------------------------------
# The region boxes, pinned against their source
# ---------------------------------------------------------------------------


def test_the_region_boxes_match_the_calibration_points_script() -> None:
    """Copied, not imported — so a test holds the two together.

    ``scripts/build_calibration_points.py`` is a CLI, and a library
    importing one by ``sys.path`` surgery is worse than eight boxes
    duplicated. What is not acceptable is the two drifting apart, which is
    what this asserts.
    """
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts"))
    try:
        import build_calibration_points as bcp
    finally:
        sys.path.remove(str(repo_root / "scripts"))
    assert review.REGION_BOXES == bcp.CALIBRATION_REGIONS
    assert list(review.REGION_BOXES) == list(bcp.CALIBRATION_REGIONS), (
        "the boxes overlap, so the PRIORITY order is part of the definition"
    )
    for lat, lon, expected in (
        (55.33, 10.32, "Fyn"),
        (55.68, 12.30, "Hovedstaden"),
        (55.11, 14.70, "Bornholm"),
        (0.0, 0.0, review.REGION_OTHER),
    ):
        assert review.region_of(lat, lon) == expected


def test_intensity_bands_and_the_unknown_one() -> None:
    assert review.intensity_band(0.2) == "trace"
    assert review.intensity_band(0.5) == "light"
    assert review.intensity_band(1.0) == "moderate"
    assert review.intensity_band(40.0) == "heavy"
    assert review.intensity_band(None) == review.UNKNOWN_BAND
    assert review.UNKNOWN_BAND not in {name for name, _lo, _hi in
                                       __import__(
                                           "dmi_nowcast_core.review_schema",
                                           fromlist=["INTENSITY_BANDS"],
                                       ).INTENSITY_BANDS}
    # An onset's two-slot depth becomes a rate over those two slots.
    assert review.two_slot_rate_mm_h(0.2) == pytest.approx(0.6)
    assert review.two_slot_rate_mm_h(None) is None


def test_strata_keys_are_all_answerable_by_a_record() -> None:
    record = _record("false_alarm")
    assert [record.stratum_value(key) for key in STRATA_KEYS] == [
        "false_alarm", "summer", "Fyn", "moderate",
    ]
    assert record.stratum() == "false_alarm|summer|Fyn|moderate"
