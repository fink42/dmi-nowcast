"""The per-subscription decision state machine.

One pure function, ``evaluate``, advanced once per *radar observation* —
not once per poll. Everything it needs arrives as arguments: the stored
state, the sampled observation, the subscriber's rules, and the wall
clock. No I/O, no clock reads, no imports from the rest of the sidecar,
so the whole spam-or-silence surface is unit-testable in isolation.

The contract mirrors the Home Assistant integration:

- **persistence** — the calibrated probability must sit at or above the
  threshold for two consecutive observations before anything fires, so a
  single frame of clutter can never push.
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
  but the rain is already at the point (ETA at or below
  ``raining_now_eta_min``), "rain incoming" would be noise: no push, and
  the subscription disarms as if it had pushed.

Timestamps are UTC everywhere; the subscriber's IANA time zone is used
for exactly one thing, the quiet-hours comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

import structlog

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


@dataclass(frozen=True)
class Rules:
    """Tuning constants. Defaults are the shipped product contract."""

    #: Consecutive over-threshold observations required to fire.
    persistence_obs: int = 2
    #: Minutes of continuous below-threshold radar time before re-arming.
    rearm_after_min: int = 60
    #: An ETA at or below this means the rain is already at the point.
    raining_now_eta_min: float = 1.5


@dataclass(frozen=True)
class QuietHours:
    """A local-time window in which delivery is suppressed.

    ``start``/``end`` are ``"HH:MM"`` in the subscription's time zone.
    ``end <= start`` wraps midnight; ``start == end`` is an empty window
    (never quiet), not a 24-hour one.
    """

    start: str
    end: str


Action = Literal["none", "notify", "deferred_quiet", "already_raining"]


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


def evaluate(
    state: SubState,
    obs: Observation,
    *,
    threshold_pct: int,
    quiet: QuietHours | None,
    tz: str,
    now_utc: datetime,
    rules: Rules = Rules(),
) -> Decision:
    """Advance the machine by one radar observation.

    Returns the state to persist and what the caller should do about it.
    The caller sends a push only for ``"notify"``; ``"already_raining"``
    and ``"deferred_quiet"`` are state transitions with no delivery.
    """
    # An observation is evaluated exactly once. Restarts, the no-new-frame
    # fast path and replays all arrive here as a timestamp we have seen.
    if (
        state.last_eval_radar_ts is not None
        and obs.radar_ts_utc <= state.last_eval_radar_ts
    ):
        return Decision(state, "none")

    over = obs.p_rain is not None and obs.p_rain >= threshold_pct / 100

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
                ),
                "none",
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
        if (
            obs.eta_min is not None
            and obs.eta_min <= rules.raining_now_eta_min
        ):
            # Rain is already here; "incoming" would be noise. Consume
            # the arm silently.
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
