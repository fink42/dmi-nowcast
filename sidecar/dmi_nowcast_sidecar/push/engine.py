"""The per-subscription decision state machine.

One pure function, ``evaluate``, advanced once per *radar observation* —
not once per poll. Everything it needs arrives as arguments: the stored
state, the sampled observation, the subscriber's rules, and the wall
clock. No I/O, no clock reads, no imports from the rest of the sidecar,
so the whole spam-or-silence surface is unit-testable in isolation.

The contract mirrors the Home Assistant integration:

- **persistence** — the calibrated probability must sit at or above the
  threshold for ``rules.persistence_obs`` consecutive observations before
  anything fires. That is ONE observation as shipped (decided 2026-09-13,
  DECIDE-14: a second observation costs ~10 minutes of radar cadence on
  top of a 13–18 minute composite age, and lost F1 at three of the four
  served horizons). The number itself lives in
  ``dmi_nowcast_core.push_rules.DEFAULT_PERSISTENCE_OBS``, overridable
  once, under ``push.persistence_obs``; at two or more, a single frame of
  clutter cannot push.
- **hysteresis** — a push *disarms* the subscription. It re-arms only
  after 60 consecutive minutes below threshold, measured on the radar
  clock (not on wall time, and not on the poll cadence). The arm is
  settled at the *first observation that finds the dry spell 60 minutes
  old*, before that observation's own probability is looked at; the
  observation is then evaluated by the armed rules from a clean slate.
  So an observation back over the threshold at or after the 60-minute
  mark re-arms and starts a streak (at ``persistence_obs = 1`` it fires
  straight away) instead of merely clearing the dry clock. Only an
  over-threshold observation *before* the mark restarts the spell.
- **quiet hours defer, they do not disarm.** Delivery is suppressed and
  the machine keeps running, still armed; an event still over threshold
  when the window ends fires then.
- **"already raining" consumes the arm silently.** If the trigger fires
  but the rain is already at the point, "rain incoming" would be noise:
  no push, and the subscription disarms as if it had pushed. Two
  independent things count as "already raining": an ETA at or below
  ``raining_now_eta_min``, *or* an OBSERVED rain rate at or above
  ``raining_now_mm_h``. The observation is not a refinement of the ETA
  test, it is the case the ETA cannot see: the ensemble's first timestep
  is ~10 min out, so a point under a shower that clears within those 10
  minutes and gets the next cell at +30 reads ETA ≈ 16 min — "rain
  incoming", sent into falling rain. Two of the first four live pushes
  were exactly that.
- **the all-clear retracts a pushed warning, once.** While disarmed after
  a ``notify`` (never after an "already raining" consumption — nothing
  was sent), consecutive observations whose decision probability is
  below the threshold are counted; the ``rules.allclear_readings``-th
  (two as shipped, ``dmi_nowcast_core.push_rules``) returns
  ``"all_clear"`` and marks it sent. An over-threshold observation resets
  the count, as it restarts the dry clock; an observation with no
  probability neither counts nor resets it. After an all-clear nothing
  more happens until the re-arm, and the re-arm itself is untouched: the
  all-clear changes no arming decision, so every notify the machine
  makes is the one it made before the all-clear existed. Quiet hours do
  not defer it — it is delivered silently.

Since Phase H the *probability* the rule reads is a choice, not a
constant: ``Observation.p_source`` selects between the served
curve-calibrated ``p_rain`` and the gauge-trained post-processed
``p_post``, with a per-observation fallback to the former. ``evaluate``
itself is unchanged by that — it compares ``obs.p_decision`` to a number,
as it always did, and never learns where either came from.

Since S11 a lead may be on the **onset AND rule** instead: fire when
``p_onset >= onset_threshold_pct AND p_decision >= threshold_pct``, where
``p_onset`` is the onset-target model's probability that rain STARTS within
the lead (``Observation.p_onset``). ``threshold_pct`` may then be 0 —
onset alone decides. An observation with no ``p_onset`` falls back to the
single-threshold rule at ``single_threshold_pct``. Everything downstream of
"is this observation over?" — persistence, re-arm, quiet hours, already
raining, the all-clear — reads the same one boolean, so it follows the
rule unchanged (:func:`is_over`).

Timestamps are UTC everywhere; the subscriber's IANA time zone is used
for exactly one thing, the quiet-hours comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

import structlog
from dmi_nowcast_core.push_rules import (
    DEFAULT_ALLCLEAR_ENABLED,
    DEFAULT_ALLCLEAR_READINGS,
    DEFAULT_PERSISTENCE_OBS,
    DEFAULT_REARM_AFTER_MIN,
)

_log = structlog.get_logger(__name__)

__all__ = [
    "SubState",
    "INITIAL_STATE",
    "Observation",
    "Rules",
    "QuietHours",
    "Action",
    "Decision",
    "in_quiet_hours",
    "evaluate",
    "is_over",
]


@dataclass(frozen=True)
class SubState:
    """Everything the machine remembers between observations.

    Persisted per subscription so a service restart never re-counts the
    observation it had already evaluated.
    """

    armed: bool
    #: Consecutive observations at or above the threshold.
    streak: int
    #: First below-threshold observation since the subscription disarmed;
    #: ``None`` while armed or while the probability is still over.
    below_since_utc: datetime | None
    #: Radar timestamp of the last observation actually evaluated.
    last_eval_radar_ts: datetime | None
    # The all-clear's memory. Additive with defaults, so a state stored
    # before it existed loads unchanged — and, lacking ``notified``, can
    # never retract a push it has no record of.
    #: Disarmed because a warning was PUSHED (``notify``), as opposed to
    #: armed, or disarmed by an "already raining" consumption.
    notified: bool = False
    #: Consecutive below-threshold observations since the push.
    below_streak: int = 0
    #: The one all-clear for this push has been issued.
    all_clear_sent: bool = False


#: A new (or edited) subscription starts armed with an empty streak.
INITIAL_STATE = SubState(
    armed=True, streak=0, below_since_utc=None, last_eval_radar_ts=None
)


@dataclass(frozen=True)
class Observation:
    """One sample of the calibrated grids at the subscription's point."""

    #: The radar composite timestamp — the observation clock.
    radar_ts_utc: datetime
    #: P(rain within the subscription's lead). ``None`` means nodata, off
    #: coverage, or a lead this cycle did not serve — never "dry".
    p_rain: float | None
    #: Minutes until rain reaches the point; ``None`` = none within the horizon.
    eta_min: float | None
    intensity_mm_h: float | None
    #: OBSERVED rain rate at the point right now, mm/h, from the newest
    #: composite. ``None`` means nodata or an unpublished observed grid —
    #: never "dry", so an absent observation can only leave the ETA test
    #: to decide, exactly as before this field existed.
    observed_mm_h: float | None = None
    #: The DETERMINISTIC forecast rain rate at the point valid NOW, mm/h —
    #: the lead-0 entry of the cycle's forecast series, i.e. the newest
    #: composite advected forward by its own age. Carried for logging and
    #: for whatever a future rule wants to do with it; it deliberately
    #: takes NO part in the decision below. The already-raining test stays
    #: on measurement (``observed_mm_h``) and the ETA grid, because
    #: silencing a notification on an extrapolation would mean the user
    #: hears nothing when the extrapolation is wrong.
    forecast_now_mm_h: float | None = None
    #: P(rain within the lead) from the gauge-trained POST-PROCESSING
    #: model (Phase H, H-P) — the same features the offline replay wrote,
    #: scored per cycle. ``None`` means the model had nothing to say about
    #: this observation: no fitted model, a lead it does not carry, a
    #: point this cycle did not score, or a row it could not score.
    p_post: float | None = None
    #: WHICH probability this observation is to be judged on.
    #: ``"curve"`` — the default, and the behaviour that shipped — is the
    #: served isotonic-calibrated ``p_rain``. ``"postprocess"`` asks for
    #: ``p_post``, and falls back to ``p_rain`` when it is None, per
    #: observation: one point off coverage for the model must not silence
    #: it, and one model outage must not silence everyone.
    p_source: Literal["postprocess", "curve"] = "curve"
    #: P(rain STARTS within the lead) from the onset-target model (S11).
    #: Set only when the subscription's lead is on the onset AND rule;
    #: ``None`` means the rule falls back to the single threshold.
    p_onset: float | None = None

    @property
    def p_decision(self) -> float | None:
        """The probability the rule actually compares to the threshold.

        One definition, read by ``evaluate`` and by the caller that logs
        the decision and writes the message — three places that must not
        be able to disagree about which number was used.
        """
        if self.p_source == "postprocess" and self.p_post is not None:
            return self.p_post
        return self.p_rain

    @property
    def p_decision_source(self) -> Literal["postprocess", "curve", "onset_and"]:
        """Where :attr:`p_decision` came from, after the per-row fallback.

        ``"onset_and"`` when the observation carries ``p_onset`` — the
        caller sets it only for a lead on the onset AND rule, so the
        decision read ``p_onset`` beside ``p_decision``.
        """
        if self.p_onset is not None and self.p_decision is not None:
            return "onset_and"
        if self.p_source == "postprocess" and self.p_post is not None:
            return "postprocess"
        return "curve"


@dataclass(frozen=True)
class Rules:
    """Tuning constants. Defaults are the shipped product contract.

    The two timing constants are imported, not restated: every replay of
    this rule (the station scoreboard, the quality page's served-rule
    hook, the threshold sweep, the historical replay, the benchmark) has
    to agree with the engine, and one number in one module is the only way
    that holds. See :mod:`dmi_nowcast_core.push_rules`.
    """

    #: Consecutive over-threshold observations required to fire.
    persistence_obs: int = DEFAULT_PERSISTENCE_OBS
    #: Minutes of continuous below-threshold radar time before re-arming.
    rearm_after_min: int = DEFAULT_REARM_AFTER_MIN
    #: An ETA at or below this means the rain is already at the point.
    raining_now_eta_min: float = 1.5
    #: An OBSERVED rain rate at or above this means the same thing, and
    #: says it about *now* rather than about the forecast. Default matches
    #: ``forecast.rain_threshold_mm_h`` — the detection threshold the rest
    #: of the pipeline (and Home Assistant's ``raining_now``) uses.
    raining_now_mm_h: float = 0.5
    #: Retract a pushed warning with one silent all-clear (see module
    #: docstring). Defaults in :mod:`dmi_nowcast_core.push_rules`.
    allclear_enabled: bool = DEFAULT_ALLCLEAR_ENABLED
    #: Consecutive below-threshold observations after the push that
    #: trigger it.
    allclear_readings: int = DEFAULT_ALLCLEAR_READINGS


@dataclass(frozen=True)
class QuietHours:
    """A local-time window in which delivery is suppressed.

    ``start``/``end`` are ``"HH:MM"`` in the subscription's time zone.
    ``end <= start`` wraps midnight; ``start == end`` is an empty window
    (never quiet), not a 24-hour one.
    """

    start: str
    end: str


Action = Literal[
    "none", "notify", "deferred_quiet", "already_raining", "all_clear",
]


@dataclass(frozen=True)
class Decision:
    state: SubState
    action: Action


def _parse_hhmm(value: str) -> tuple[int, int]:
    hh, _, mm = value.partition(":")
    return int(hh), int(mm)


def in_quiet_hours(
    now_utc: datetime, tz: str, quiet: QuietHours | None
) -> bool:
    """Is ``now_utc`` inside the subscriber's quiet window?

    ``now_utc`` must be timezone-aware — a naive datetime here is a
    programming error (the UTC-internally rule), and is raised rather
    than guessed at. An *unknown* time zone, by contrast, is a data
    problem in one row and must never take the cycle down: it is logged
    and treated as "not quiet".
    """
    if now_utc.tzinfo is None or now_utc.utcoffset() is None:
        raise ValueError("in_quiet_hours() requires a timezone-aware now_utc")
    if quiet is None:
        return False

    try:
        local = now_utc.astimezone(ZoneInfo(tz))
    except Exception as exc:  # unknown tz, bad string, missing tzdata
        _log.warning(
            "push_quiet_hours_bad_tz", tz=tz, error=type(exc).__name__
        )
        return False

    try:
        start = _parse_hhmm(quiet.start)
        end = _parse_hhmm(quiet.end)
    except (AttributeError, ValueError):
        _log.warning(
            "push_quiet_hours_bad_window", start=quiet.start, end=quiet.end
        )
        return False

    now = (local.hour, local.minute)
    if start == end:
        return False
    if start < end:
        return start <= now < end
    # Wrapped window, e.g. 22:00–07:00.
    return now >= start or now < end


def is_over(
    obs: Observation,
    threshold_pct: int,
    onset_threshold_pct: int | None = None,
    single_threshold_pct: int | None = None,
) -> bool:
    """Is this observation over the rule? The one predicate of the machine.

    Single-threshold rule (``onset_threshold_pct`` None): ``p_decision >=
    threshold_pct``. Onset AND rule: ``p_onset >= onset_threshold_pct AND
    p_decision >= threshold_pct``; an observation with no ``p_onset`` is
    judged on the single rule at ``single_threshold_pct`` (else
    ``threshold_pct``). No ``p_decision`` is never over.
    """
    p = obs.p_decision
    if p is None:
        return False
    if onset_threshold_pct is None or obs.p_onset is None:
        single = threshold_pct
        if onset_threshold_pct is not None and single_threshold_pct is not None:
            single = single_threshold_pct
        return p >= single / 100
    return obs.p_onset >= onset_threshold_pct / 100 and p >= threshold_pct / 100


def evaluate(
    state: SubState,
    obs: Observation,
    *,
    threshold_pct: int,
    quiet: QuietHours | None,
    tz: str,
    now_utc: datetime,
    rules: Rules = Rules(),
    onset_threshold_pct: int | None = None,
    single_threshold_pct: int | None = None,
) -> Decision:
    """Advance the machine by one radar observation.

    Returns the state to persist and what the caller should do about it.
    The caller sends a warning for ``"notify"`` and the silent
    retraction for ``"all_clear"``; ``"already_raining"`` and
    ``"deferred_quiet"`` are state transitions with no delivery.

    ``onset_threshold_pct`` / ``single_threshold_pct`` switch the lead to
    the onset AND rule (see :func:`is_over`); omitted, the rule is the
    single threshold it always was.
    """
    # An observation is evaluated exactly once. Restarts, the no-new-frame
    # fast path and replays all arrive here as a timestamp we have seen.
    if (
        state.last_eval_radar_ts is not None
        and obs.radar_ts_utc <= state.last_eval_radar_ts
    ):
        return Decision(state, "none")

    # The rule reads ONE number, whichever source produced it: the whole
    # H-P change on this path is which probability ``p_decision`` returns,
    # and ``evaluate`` stays pure and unaware of the choice.
    decision_p = obs.p_decision
    over = is_over(obs, threshold_pct, onset_threshold_pct, single_threshold_pct)

    if not state.armed:
        # Disarmed. Settle the arm *before* judging this observation: a
        # dry spell that has reached the re-arm mark has reached it
        # whether or not the probability came back up at that very
        # moment.
        rearmed = (
            state.below_since_utc is not None
            and obs.radar_ts_utc - state.below_since_utc
            >= timedelta(minutes=rules.rearm_after_min)
        )
        if not rearmed:
            # The spell is still too short: an over-threshold observation
            # restarts the dry clock, a dry one starts or keeps it.
            below_streak = state.below_streak
            all_clear = False
            if (
                state.notified
                and not state.all_clear_sent
                and rules.allclear_enabled
            ):
                # Only a push can be retracted, and only once. A reading
                # with no probability is no reading: it neither counts
                # toward the all-clear nor breaks the run (the replay
                # study skipped such rows the same way).
                if over:
                    below_streak = 0
                elif decision_p is not None:
                    below_streak += 1
                all_clear = below_streak >= max(1, int(rules.allclear_readings))
            return Decision(
                SubState(
                    armed=False,
                    streak=state.streak + 1 if over else 0,
                    below_since_utc=(
                        None
                        if over
                        else (state.below_since_utc or obs.radar_ts_utc)
                    ),
                    last_eval_radar_ts=obs.radar_ts_utc,
                    notified=state.notified,
                    below_streak=below_streak,
                    all_clear_sent=state.all_clear_sent or all_clear,
                ),
                "all_clear" if all_clear else "none",
            )
        # Re-armed as of this observation, which now goes through the
        # armed rules from a clean slate.
        state = SubState(
            armed=True,
            streak=0,
            below_since_utc=None,
            last_eval_radar_ts=state.last_eval_radar_ts,
        )

    streak = state.streak + 1 if over else 0

    if streak >= rules.persistence_obs:
        if in_quiet_hours(now_utc, tz, quiet):
            # Suppressed, not consumed: still armed, streak intact, so
            # the first observation after the window can fire.
            return Decision(
                SubState(
                    armed=True,
                    streak=streak,
                    below_since_utc=state.below_since_utc,
                    last_eval_radar_ts=obs.radar_ts_utc,
                ),
                "deferred_quiet",
            )
        already_raining = (
            obs.eta_min is not None
            and obs.eta_min <= rules.raining_now_eta_min
        ) or (
            obs.observed_mm_h is not None
            and obs.observed_mm_h >= rules.raining_now_mm_h
        )
        if already_raining:
            # Rain is already here — forecast to arrive within the next
            # breath, or measured falling at the point this very frame.
            # "Incoming" would be noise. Consume the arm silently.
            return Decision(
                SubState(
                    armed=False,
                    streak=streak,
                    below_since_utc=None,
                    last_eval_radar_ts=obs.radar_ts_utc,
                ),
                "already_raining",
            )
        return Decision(
            SubState(
                armed=False,
                streak=streak,
                below_since_utc=None,
                last_eval_radar_ts=obs.radar_ts_utc,
                # A push happened: until the re-arm it may be retracted,
                # once, by an all-clear.
                notified=True,
            ),
            "notify",
        )
    return Decision(
        SubState(
            armed=True,
            streak=streak,
            below_since_utc=state.below_since_utc,
            last_eval_radar_ts=obs.radar_ts_utc,
        ),
        "none",
    )
