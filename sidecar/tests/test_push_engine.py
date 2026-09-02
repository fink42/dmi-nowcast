"""The push decision state machine — the part that spams or stays silent.

Every sequence below is written on the radar clock at the 10-minute
fullRange cadence, because that is the clock the machine actually
advances on.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from dmi_nowcast_sidecar.push.engine import (
    INITIAL_STATE,
    Decision,
    Observation,
    QuietHours,
    Rules,
    SubState,
    evaluate,
    in_quiet_hours,
)

CADENCE = timedelta(minutes=10)
#: A June day: Copenhagen is CEST (UTC+2), far from any quiet window.
T0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
CPH = "Europe/Copenhagen"
UTC = timezone.utc


def _obs(
    step: int,
    p: float | None,
    *,
    eta: float | None = None,
    mm: float | None = None,
    base: datetime = T0,
) -> Observation:
    return Observation(
        radar_ts_utc=base + step * CADENCE,
        p_rain=p,
        eta_min=eta,
        intensity_mm_h=mm,
    )


def _drive(
    probs,
    *,
    threshold_pct: int = 60,
    state: SubState = INITIAL_STATE,
    quiet: QuietHours | None = None,
    tz: str = CPH,
    rules: Rules = Rules(),
    eta: float | None = None,
    base: datetime = T0,
    now_at_radar: bool = False,
):
    """Feed a probability sequence; return (actions, states, observations)."""
    actions: list[str] = []
    states: list[SubState] = []
    observations: list[Observation] = []
    for i, p in enumerate(probs):
        obs = _obs(i, p, eta=eta, base=base)
        now_utc = (
            obs.radar_ts_utc
            if now_at_radar
            else obs.radar_ts_utc + timedelta(minutes=2)
        )
        decision = evaluate(
            state,
            obs,
            threshold_pct=threshold_pct,
            quiet=quiet,
            tz=tz,
            now_utc=now_utc,
            rules=rules,
        )
        state = decision.state
        actions.append(decision.action)
        states.append(state)
        observations.append(obs)
    return actions, states, observations


# --------------------------------------------------------------------------
# Persistence: one frame of clutter never pushes
# --------------------------------------------------------------------------


def test_initial_state_is_armed_and_empty() -> None:
    assert INITIAL_STATE == SubState(
        armed=True, streak=0, below_since_utc=None, last_eval_radar_ts=None
    )


def test_one_wet_observation_does_not_fire() -> None:
    actions, states, _ = _drive([0.9])
    assert actions == ["none"]
    assert states[-1].armed is True
    assert states[-1].streak == 1


def test_two_consecutive_wet_fire_exactly_once() -> None:
    actions, states, _ = _drive([0.9, 0.9])
    assert actions == ["none", "notify"]
    assert states[-1].armed is False
    assert states[-1].below_since_utc is None


def test_dry_observation_between_two_wet_resets_the_streak() -> None:
    actions, states, _ = _drive([0.9, 0.1, 0.9])
    assert actions == ["none", "none", "none"]
    assert states[1].streak == 0
    assert states[-1].streak == 1
    assert states[-1].armed is True


def test_probability_exactly_at_threshold_fires() -> None:
    actions, _, _ = _drive([0.60, 0.60], threshold_pct=60)
    assert actions == ["none", "notify"]


def test_probability_just_below_threshold_never_fires() -> None:
    actions, states, _ = _drive([0.59] * 6, threshold_pct=60)
    assert "notify" not in actions
    assert states[-1].streak == 0


def test_none_probability_never_counts_as_over() -> None:
    actions, states, _ = _drive([None] * 6)
    assert actions == ["none"] * 6
    assert states[-1].streak == 0


def test_none_probability_breaks_a_streak() -> None:
    # A lead that went missing is not evidence of rain, and not evidence
    # of dry either — but it must not carry a streak across the gap.
    actions, states, _ = _drive([0.9, None, 0.9])
    assert "notify" not in actions
    assert states[-1].streak == 1


def test_persistence_one_fires_on_the_first_wet_observation() -> None:
    actions, _, _ = _drive([0.9, 0.9], rules=Rules(persistence_obs=1))
    assert actions == ["notify", "none"]


def test_persistence_three_needs_three_consecutive() -> None:
    actions, _, _ = _drive(
        [0.9, 0.9, 0.1, 0.9, 0.9, 0.9], rules=Rules(persistence_obs=3)
    )
    assert actions == ["none", "none", "none", "none", "none", "notify"]


# --------------------------------------------------------------------------
# Hysteresis: disarm on push, re-arm only after 60 dry minutes
# --------------------------------------------------------------------------


def test_wet_observations_after_firing_keep_it_disarmed() -> None:
    actions, states, _ = _drive([0.9] * 10)
    assert actions.count("notify") == 1
    assert actions.index("notify") == 1
    assert states[-1].armed is False
    assert states[-1].below_since_utc is None


def test_rearm_happens_at_exactly_sixty_dry_minutes() -> None:
    # index 0,1 wet -> notify at 1. Index 2 is the first dry observation,
    # so the re-arm clock starts there: +50 is still disarmed, +60 armed.
    probs = [0.9, 0.9] + [0.0] * 8
    actions, states, obs = _drive(probs)
    assert actions == ["none", "notify"] + ["none"] * 8

    first_dry = obs[2].radar_ts_utc
    assert states[2].below_since_utc == first_dry
    by_offset = {
        int((o.radar_ts_utc - first_dry).total_seconds() // 60): s
        for o, s in zip(obs[2:], states[2:])
    }
    assert by_offset[50].armed is False
    assert by_offset[60].armed is True
    assert by_offset[60].below_since_utc is None
    assert by_offset[60].streak == 0


def test_a_wet_observation_restarts_the_dry_clock() -> None:
    # Dry from index 2; a wet blip at +30 min (index 5) clears
    # below_since, so the clock restarts at index 6.
    probs = [0.9, 0.9] + [0.0, 0.0, 0.0, 0.9] + [0.0] * 8
    actions, states, obs = _drive(probs)
    assert actions.count("notify") == 1
    assert states[5].below_since_utc is None
    assert states[6].below_since_utc == obs[6].radar_ts_utc
    # 50 min after the restart the sub is still disarmed...
    assert states[11].armed is False
    assert (obs[11].radar_ts_utc - obs[6].radar_ts_utc) == timedelta(minutes=50)
    # ...and re-arms only at the full hour.
    assert states[12].armed is True


def test_it_fires_again_after_rearming() -> None:
    probs = [0.9, 0.9] + [0.0] * 8 + [0.9, 0.9]
    actions, states, _ = _drive(probs)
    assert actions.count("notify") == 2
    assert [i for i, a in enumerate(actions) if a == "notify"] == [1, 11]
    assert states[-1].armed is False


def test_over_threshold_while_disarmed_clears_below_since() -> None:
    probs = [0.9, 0.9, 0.0, 0.0, 0.9]
    _, states, _ = _drive(probs)
    assert states[3].below_since_utc is not None
    assert states[4].below_since_utc is None
    assert states[4].armed is False


def test_over_threshold_at_the_mark_rearms_and_fires_at_persistence_one() -> None:
    # The arm is settled before the probability is looked at: at +70 the
    # dry spell that started at +10 is exactly 60 minutes old, so the
    # subscription re-arms and this same observation — over threshold,
    # persistence 1 — fires.
    probs = [0.9] + [0.0] * 6 + [0.9]
    actions, states, obs = _drive(probs, rules=Rules(persistence_obs=1))
    assert actions == ["notify"] + ["none"] * 6 + ["notify"]
    assert states[1].below_since_utc == obs[1].radar_ts_utc
    assert obs[7].radar_ts_utc - obs[1].radar_ts_utc == timedelta(minutes=60)
    assert states[7].streak == 1
    assert states[7].armed is False  # consumed by the second push
    assert states[7].below_since_utc is None


def test_over_threshold_before_the_mark_stays_disarmed() -> None:
    # Ten minutes short of the mark nothing changes: the observation at
    # +60 finds a 50-minute spell, so it only clears the dry clock — and
    # with the clock cleared, +70 has nothing to re-arm on either.
    probs = [0.9] + [0.0] * 5 + [0.9, 0.9]
    actions, states, obs = _drive(probs, rules=Rules(persistence_obs=1))
    assert actions == ["notify"] + ["none"] * 7
    assert obs[6].radar_ts_utc - obs[1].radar_ts_utc == timedelta(minutes=50)
    assert states[6].armed is False
    assert states[6].below_since_utc is None
    assert states[7].armed is False


def test_over_threshold_at_the_mark_starts_the_streak_at_persistence_two() -> None:
    # Same re-arm, but persistence 2: the observation at the mark re-arms
    # from a clean slate and becomes streak 1, and the next wet one fires.
    probs = [0.9, 0.9] + [0.0] * 6 + [0.9, 0.9]
    actions, states, obs = _drive(probs)
    assert actions == ["none", "notify"] + ["none"] * 7 + ["notify"]
    assert states[2].below_since_utc == obs[2].radar_ts_utc
    assert obs[8].radar_ts_utc - obs[2].radar_ts_utc == timedelta(minutes=60)
    assert states[8].armed is True
    assert states[8].streak == 1
    assert states[8].below_since_utc is None


def test_a_dry_observation_at_the_mark_still_rearms_silently() -> None:
    # The dry path through the re-arm check is unchanged, persistence 1
    # included: the observation that finds the spell 60 minutes old
    # re-arms, and being dry it fires nothing.
    probs = [0.9] + [0.0] * 7
    actions, states, obs = _drive(probs, rules=Rules(persistence_obs=1))
    assert actions == ["notify"] + ["none"] * 7
    assert states[1].below_since_utc == obs[1].radar_ts_utc
    assert states[6].armed is False  # +60: a 50-minute spell
    assert states[7].armed is True  # +70: the full hour
    assert states[7].streak == 0
    assert states[7].below_since_utc is None


def test_regression_the_shower_that_arrived_with_no_push() -> None:
    # Seen live at the user's point: disarmed at 05:40; dry observations
    # from 06:00 on, so the re-arm clock started at 06:00; at 07:00 the
    # calibrated probability crossed the 40 % threshold with the spell
    # exactly 60 minutes old. The machine used to clear the clock and
    # stay disarmed, so 07:00, 07:10 and 07:20 could not fire and the
    # shower reached the point at 07:30 unannounced.
    rules = Rules(persistence_obs=1)
    state = SubState(
        armed=False,
        streak=1,
        below_since_utc=None,
        last_eval_radar_ts=datetime(2026, 8, 31, 5, 40, tzinfo=UTC),
    )
    sequence = [
        (datetime(2026, 8, 31, 6, 0, tzinfo=UTC), 0.05),
        (datetime(2026, 8, 31, 6, 10, tzinfo=UTC), 0.07),
        (datetime(2026, 8, 31, 6, 20, tzinfo=UTC), 0.04),
        (datetime(2026, 8, 31, 6, 30, tzinfo=UTC), 0.11),
        (datetime(2026, 8, 31, 6, 40, tzinfo=UTC), 0.16),
        (datetime(2026, 8, 31, 6, 50, tzinfo=UTC), 0.29),
        (datetime(2026, 8, 31, 7, 0, tzinfo=UTC), 0.62),
    ]
    actions: list[str] = []
    states: list[SubState] = []
    for radar_ts, p in sequence:
        decision = evaluate(
            state,
            Observation(
                radar_ts_utc=radar_ts,
                p_rain=p,
                eta_min=33.0,
                intensity_mm_h=0.8,
            ),
            threshold_pct=40,
            quiet=None,
            tz=CPH,
            now_utc=radar_ts + timedelta(minutes=2),
            rules=rules,
        )
        state = decision.state
        actions.append(decision.action)
        states.append(state)

    assert states[0].below_since_utc == datetime(2026, 8, 31, 6, 0, tzinfo=UTC)
    assert all(s.armed is False for s in states[:6])
    assert actions == ["none"] * 6 + ["notify"]


# --------------------------------------------------------------------------
# "Already raining" consumes the arm silently
# --------------------------------------------------------------------------


@pytest.mark.parametrize("eta", [0.0, 1.0, 1.5])
def test_already_raining_when_eta_at_or_below_the_limit(eta: float) -> None:
    actions, states, _ = _drive([0.9, 0.9], eta=eta)
    assert actions == ["none", "already_raining"]
    assert states[-1].armed is False
    assert states[-1].below_since_utc is None


def test_eta_just_above_the_limit_still_notifies() -> None:
    actions, _, _ = _drive([0.9, 0.9], eta=1.6)
    assert actions == ["none", "notify"]


def test_eta_none_still_notifies() -> None:
    # No ETA within the horizon is the "window wording" case downstream,
    # not a reason to stay silent.
    actions, _, _ = _drive([0.9, 0.9], eta=None)
    assert actions == ["none", "notify"]


def test_already_raining_disarms_for_the_full_hour_too() -> None:
    probs = [0.9, 0.9] + [0.0] * 5
    actions, states, _ = _drive(probs, eta=0.5)
    assert actions[1] == "already_raining"
    assert all(s.armed is False for s in states[1:])


# --------------------------------------------------------------------------
# Quiet hours defer, they do not disarm
# --------------------------------------------------------------------------

QUIET = QuietHours(start="22:00", end="07:00")
#: 2026-06-01 23:00Z is 2026-06-02 01:00 in Copenhagen — inside the window.
T_QUIET = datetime(2026, 6, 1, 23, 0, tzinfo=UTC)


def test_deferred_keeps_it_armed_and_the_streak_grows() -> None:
    actions, states, _ = _drive(
        [0.9] * 4, quiet=QUIET, base=T_QUIET, now_at_radar=True
    )
    assert actions == ["none", "deferred_quiet", "deferred_quiet", "deferred_quiet"]
    assert all(s.armed for s in states)
    assert [s.streak for s in states] == [1, 2, 3, 4]


def test_first_observation_after_the_window_fires() -> None:
    # Local 06:20 .. 07:00 on a CEST day: 04:20Z .. 05:00Z. The window
    # ends at local 07:00, so the 05:00Z observation is the first one
    # allowed to deliver.
    base = datetime(2026, 6, 2, 4, 20, tzinfo=UTC)
    actions, states, obs = _drive(
        [0.9] * 5, quiet=QUIET, base=base, now_at_radar=True
    )
    assert obs[4].radar_ts_utc.astimezone(ZoneInfo(CPH)).hour == 7
    assert actions == [
        "none",
        "deferred_quiet",
        "deferred_quiet",
        "deferred_quiet",
        "notify",
    ]
    assert states[-1].armed is False


def test_event_that_ends_inside_the_window_fires_nothing() -> None:
    actions, states, _ = _drive(
        [0.9, 0.9, 0.9, 0.0, 0.0], quiet=QUIET, base=T_QUIET, now_at_radar=True
    )
    assert "notify" not in actions
    assert actions == [
        "none",
        "deferred_quiet",
        "deferred_quiet",
        "none",
        "none",
    ]
    assert states[-1].armed is True
    assert states[-1].streak == 0


def test_quiet_hours_take_precedence_over_already_raining() -> None:
    actions, states, _ = _drive(
        [0.9, 0.9], quiet=QUIET, base=T_QUIET, now_at_radar=True, eta=0.2
    )
    assert actions == ["none", "deferred_quiet"]
    assert states[-1].armed is True


def test_no_quiet_window_never_defers() -> None:
    actions, _, _ = _drive([0.9, 0.9], quiet=None, base=T_QUIET, now_at_radar=True)
    assert actions == ["none", "notify"]


# --------------------------------------------------------------------------
# The observation clock is monotonic
# --------------------------------------------------------------------------


def test_replayed_observation_leaves_the_state_untouched() -> None:
    state = SubState(
        armed=True,
        streak=1,
        below_since_utc=None,
        last_eval_radar_ts=T0 + 5 * CADENCE,
    )
    for step in (5, 4, 0):
        decision = evaluate(
            state,
            _obs(step, 0.99),
            threshold_pct=60,
            quiet=None,
            tz=CPH,
            now_utc=T0 + 6 * CADENCE,
        )
        assert decision == Decision(state, "none")
        assert decision.state is state


def test_a_repeated_observation_cannot_complete_a_streak() -> None:
    # The no-new-frame fast path, replayed: the same wet frame twice must
    # not look like two consecutive observations.
    state = INITIAL_STATE
    for _ in range(5):
        state = evaluate(
            state,
            _obs(0, 0.99),
            threshold_pct=60,
            quiet=None,
            tz=CPH,
            now_utc=T0,
        ).state
    assert state.streak == 1
    assert state.armed is True


def test_every_evaluated_observation_advances_last_eval_radar_ts() -> None:
    _, states, obs = _drive([0.9, 0.1, None, 0.9, 0.9, 0.0])
    assert [s.last_eval_radar_ts for s in states] == [
        o.radar_ts_utc for o in obs
    ]


# --------------------------------------------------------------------------
# Property: two pushes require a full dry hour between them
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
def test_notify_never_repeats_without_a_dry_hour(seed: int) -> None:
    rng = random.Random(seed)
    probs = [
        None if rng.random() < 0.05 else round(rng.random(), 3)
        for _ in range(240)
    ]
    etas = [None if rng.random() < 0.5 else rng.uniform(0, 40) for _ in probs]

    state = INITIAL_STATE
    fired: list[int] = []
    observations: list[Observation] = []
    for i, (p, eta) in enumerate(zip(probs, etas)):
        obs = _obs(i, p, eta=eta)
        observations.append(obs)
        decision = evaluate(
            state,
            obs,
            threshold_pct=60,
            quiet=None,
            tz=CPH,
            now_utc=obs.radar_ts_utc,
        )
        state = decision.state
        if decision.action == "notify":
            fired.append(i)

    def dry(i: int) -> bool:
        p = probs[i]
        return p is None or p < 0.6

    for previous, current in zip(fired, fired[1:]):
        # Between two pushes there must be a dry spell that reached the
        # full re-arm hour, measured the way the machine measures it:
        # from the first dry observation to the observation that finds
        # the spell 60 minutes old. That observation is the one that
        # re-arms, and it counts whether it is itself dry or wet — a wet
        # one re-arms and is then evaluated normally, so it can be
        # `current` itself. Hence the range runs to `current` inclusive
        # and the span is taken before the run is broken.
        longest = timedelta(0)
        run_start: int | None = None
        for i in range(previous + 1, current + 1):
            if run_start is not None:
                span = (
                    observations[i].radar_ts_utc
                    - observations[run_start].radar_ts_utc
                )
                longest = max(longest, span)
            if dry(i):
                if run_start is None:
                    run_start = i
            else:
                run_start = None
        assert longest >= timedelta(minutes=60), (
            f"seed {seed}: pushes at {previous} and {current} "
            f"with only {longest} of consecutive dry between them"
        )


# --------------------------------------------------------------------------
# in_quiet_hours
# --------------------------------------------------------------------------


def _at(local: datetime, tz: str = CPH) -> datetime:
    """A local wall-clock time as the UTC instant it corresponds to."""
    return local.replace(tzinfo=ZoneInfo(tz)).astimezone(UTC)


def test_no_window_is_never_quiet() -> None:
    assert in_quiet_hours(T0, CPH, None) is False


@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [(7, 59, False), (8, 0, True), (8, 59, True), (9, 0, False)],
)
def test_normal_window(hour: int, minute: int, expected: bool) -> None:
    quiet = QuietHours(start="08:00", end="09:00")
    moment = _at(datetime(2026, 6, 1, hour, minute))
    assert in_quiet_hours(moment, CPH, quiet) is expected


@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [
        (21, 59, False),
        (22, 0, True),
        (3, 0, True),
        (6, 59, True),
        (7, 0, False),
    ],
)
def test_wrapped_window(hour: int, minute: int, expected: bool) -> None:
    moment = _at(datetime(2026, 6, 1, hour, minute))
    assert in_quiet_hours(moment, CPH, QUIET) is expected


@pytest.mark.parametrize("hour", [0, 7, 8, 12, 23])
def test_equal_start_and_end_is_an_empty_window(hour: int) -> None:
    quiet = QuietHours(start="08:00", end="08:00")
    assert in_quiet_hours(_at(datetime(2026, 6, 1, hour, 0)), CPH, quiet) is False


def test_spring_forward_skipped_hour() -> None:
    # 2026-03-29: 02:00 CET -> 03:00 CEST at 01:00Z. Local 02:30 never
    # happens; a 02:30-03:30 window still catches local 03:00.
    quiet = QuietHours(start="02:30", end="03:30")
    before = datetime(2026, 3, 29, 0, 30, tzinfo=UTC)  # local 01:30 CET
    after = datetime(2026, 3, 29, 1, 0, tzinfo=UTC)  # local 03:00 CEST
    past = datetime(2026, 3, 29, 1, 30, tzinfo=UTC)  # local 03:30 CEST
    assert before.astimezone(ZoneInfo(CPH)).strftime("%H:%M") == "01:30"
    assert after.astimezone(ZoneInfo(CPH)).strftime("%H:%M") == "03:00"
    assert in_quiet_hours(before, CPH, quiet) is False
    assert in_quiet_hours(after, CPH, quiet) is True
    assert in_quiet_hours(past, CPH, quiet) is False


def test_spring_forward_window_end_is_local_seven() -> None:
    # On 2026-03-29 Copenhagen is CEST after the change: local 07:00 = 05:00Z.
    assert in_quiet_hours(datetime(2026, 3, 29, 4, 59, tzinfo=UTC), CPH, QUIET) is True
    assert in_quiet_hours(datetime(2026, 3, 29, 5, 0, tzinfo=UTC), CPH, QUIET) is False


def test_fall_back_repeated_hour_both_passes() -> None:
    # 2026-10-25: 03:00 CEST -> 02:00 CET at 01:00Z, so local 02:30
    # happens twice. A 02:00-03:00 window covers both passes.
    quiet = QuietHours(start="02:00", end="03:00")
    first = datetime(2026, 10, 25, 0, 30, tzinfo=UTC)  # 02:30 CEST
    second = datetime(2026, 10, 25, 1, 30, tzinfo=UTC)  # 02:30 CET
    assert first.astimezone(ZoneInfo(CPH)).strftime("%H:%M") == "02:30"
    assert second.astimezone(ZoneInfo(CPH)).strftime("%H:%M") == "02:30"
    assert in_quiet_hours(first, CPH, quiet) is True
    assert in_quiet_hours(second, CPH, quiet) is True


def test_fall_back_window_end_is_local_seven() -> None:
    # Back on CET: local 07:00 = 06:00Z.
    assert in_quiet_hours(datetime(2026, 10, 25, 5, 59, tzinfo=UTC), CPH, QUIET) is True
    assert in_quiet_hours(datetime(2026, 10, 25, 6, 0, tzinfo=UTC), CPH, QUIET) is False


def test_non_hour_offset_timezone() -> None:
    # Asia/Kolkata is UTC+05:30 — the half-hour offset must not be lost.
    tz = "Asia/Kolkata"
    quiet_moment = datetime(2026, 6, 1, 20, 0, tzinfo=UTC)  # local 01:30
    awake_moment = datetime(2026, 6, 1, 1, 35, tzinfo=UTC)  # local 07:05
    edge_moment = datetime(2026, 6, 1, 1, 25, tzinfo=UTC)  # local 06:55
    assert quiet_moment.astimezone(ZoneInfo(tz)).strftime("%H:%M") == "01:30"
    assert in_quiet_hours(quiet_moment, tz, QUIET) is True
    assert in_quiet_hours(awake_moment, tz, QUIET) is False
    assert in_quiet_hours(edge_moment, tz, QUIET) is True


@pytest.mark.parametrize("tz", ["Mars/Olympus_Mons", "", "Europe/Kopenhagen"])
def test_unknown_timezone_is_not_quiet_and_does_not_raise(tz: str) -> None:
    assert in_quiet_hours(T0, tz, QUIET) is False


def test_malformed_window_is_not_quiet() -> None:
    assert in_quiet_hours(T0, CPH, QuietHours(start="bogus", end="07:00")) is False


def test_naive_now_is_a_programming_error() -> None:
    naive = datetime(2026, 6, 1, 23, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        in_quiet_hours(naive, CPH, QUIET)
    with pytest.raises(ValueError, match="timezone-aware"):
        in_quiet_hours(naive, CPH, None)


def test_evaluate_propagates_the_naive_now_error() -> None:
    # Only reachable on the branch that consults the window, but it must
    # not be swallowed into a silent "not quiet".
    state = SubState(
        armed=True, streak=1, below_since_utc=None, last_eval_radar_ts=None
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        evaluate(
            state,
            _obs(1, 0.99),
            threshold_pct=60,
            quiet=QUIET,
            tz=CPH,
            now_utc=datetime(2026, 6, 1, 23, 0),
        )
