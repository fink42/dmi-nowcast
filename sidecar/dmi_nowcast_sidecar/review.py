"""The review bundle's builder half: population, sample, per-event detail.

:mod:`dmi_nowcast_core.review_schema` is the contract — the classes, the
flags, the vocabulary, the field lists, the window geometry. This module
is what fills it: it re-decides the stored decision rows under a stated
rule, scores them against the gauges, turns every graded outcome into an
:class:`EventRecord`, draws a seeded stratified sample, and assembles the
per-event detail document the browser reads.

Why it lives in the sidecar package
-----------------------------------

The bundle is *about* a rule, and the rule is only spelled out here:
:mod:`~dmi_nowcast_sidecar.threshold_sweep` reads the decision trees and
builds the tracks, :mod:`~dmi_nowcast_sidecar.served_rule` resolves the
served threshold and the probability column, and
:mod:`~dmi_nowcast_sidecar.push.engine` is the state machine itself. Core
must never import the sidecar, so a builder that imports all three cannot
live in core. ``review_schema`` stays in core because it is pure data.

The one claim this module has to be able to defend
--------------------------------------------------

**The review measures the rule that shipped.** Every warning in the
population comes out of :func:`replay_station_traced`, which is
:func:`~dmi_nowcast_sidecar.threshold_sweep.replay_station` with the state
written down — same ``evaluate`` call, same arguments, same
``INITIAL_STATE`` reset at every coverage-run boundary — and
``sidecar/tests/test_review.py`` pins the two against each other row for
row. If that test ever fails, the bundle is measuring something else and
every tag counted off it is about a rule nobody ran.

``replay_station`` itself is deliberately NOT modified. It is on the
nightly fit's hot path (a million rows × a grid of cells) and a tracing
variant there would cost memory the VM does not have; and a shared
function quietly gaining a second output is exactly how two callers start
disagreeing about what they measured.

Things that would silently corrupt a review, and what is done about them
------------------------------------------------------------------------

*Decision-directory precedence.*
:func:`~dmi_nowcast_sidecar.threshold_sweep.load_decisions` lets the LATER
directory win a ``(radar_ts, station_id)`` tie. For the live scoreboard
that is right — the live row is the service's own word. For a review it is
backwards: the out-of-fold replay row carries features and an honest
``p_post``, and a live row from the feature-gap window carries neither, so
letting the live row win would judge the event on the curve scale against
a threshold fitted on the ``p_post`` scale. The caller therefore passes
``decisions_dirs`` **least authoritative first** (the live tree first, the
out-of-fold replay tree last), the effective order is recorded in the
manifest, and so is the number of rows that lost a tie.

*The feature gap is measured, never assumed.* Live rows lost their Phase-H
features from 2026-09-05 until a fix landed; the end date is not derivable
from this repository, so :func:`feature_gap_days` counts non-null
``obs_max_5km_mm_h`` per UTC day and the days where more than
:data:`FEATURE_GAP_SHARE` of the rows lack features are excluded from the
sampling frame. ``allow_feature_gap=True`` keeps them, flagged.

*The 60-minute re-arm.* ``evaluate`` disarms on every ``notify`` **and**
every ``already_raining``, and re-arms only after 60 minutes of radar time
below threshold measured from ``below_since_utc`` — which any
over-threshold row resets to ``None`` while disarmed. In a showery spell
the dry clock never accumulates and a station can sit disarmed for hours,
which makes every onset in that spell a miss the rule could not have
caught. The disarming action is usually outside the ±90 minute window, so
:func:`build_event` always writes a ``prologue`` block: the arm state
entering the window, the dry clock, the minutes still owed, and the last
non-``none`` actions however far back they lie.

*The replay's free re-arm.* State is reset to ``INITIAL_STATE`` at the
head of every coverage run, so a gap longer than ``coverage_gap_min``
hands the station an arm the live service never had. Rows where that
actually re-armed a disarmed machine carry ``run_boundary_rearm``.

*Null is not zero.* Every verdict here is a three-valued one: ``True``,
``False``, or ``None`` for "the instrument said nothing". A gauge with no
known slot in the window is not a dry gauge, a window with no
``observed_mm_h`` is not a dry radar, and an event whose gauge or radar is
silent gets ``dual_truth = None`` rather than a fabricated quadrant.

Time and units follow ``review_schema``: UTC everywhere, slots stamped at
their END, ``generated_at = radar_ts + frame_age_min``, and
``lead_error_min = eta_min - (onset - sent)`` with POSITIVE meaning the
warning was late.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from dmi_nowcast_core import postprocess as core_postprocess
from dmi_nowcast_core.lightning import haversine_km
from dmi_nowcast_core.push_rules import (
    DEFAULT_PERSISTENCE_OBS,
    DEFAULT_REARM_AFTER_MIN,
)
from dmi_nowcast_core.review_schema import (
    DEFAULT_FRAME_PAD_MIN,
    DEFAULT_WINDOW_MIN,
    INDEX_FIELDS,
    INTENSITY_BANDS,
    NEIGHBOUR_RADIUS_KM,
    RADAR_DISC_RADIUS_M,
    REVIEW_SCHEMA_VERSION,
    STRATA_KEYS,
    WARNING_SIDE_CLASSES,
    event_id,
)
from dmi_nowcast_core.warning_score import (
    DEFAULT_COVERAGE_GAP_MIN,
    DEFAULT_DRY_MIN,
    DEFAULT_LEAD_MIN,
    DEFAULT_MIN_KNOWN_SLOTS,
    DEFAULT_ONSET_MIN_MM,
    DEFAULT_PRODUCT_LEADS_MIN,
    DEFAULT_TOLERANCE_MIN,
    SLOT_MIN,
    GaugeTruth,
    StationSlots,
    coverage_runs,
    p_rain_column,
    score_warnings,
    slot_end_of,
)

from . import served_rule as served_rule_module
from .push import engine as decision_engine
from .threshold_sweep import (
    FIT_MIN_USEFUL_LEAD_MIN,
    GAUGE_PAD_MIN,
    RAIN_THRESHOLD_MM_H,
    build_tracks,
    filter_tracks_by_months,
    load_decisions,
    season_of,
)

# The track tuple's field positions, imported rather than written as
# literals: ``build_tracks`` owns that layout and a hardcoded ``4`` here
# would silently read the wrong column the day it grows a field.
from .threshold_sweep import (  # noqa: PLC2701 — deliberate, see above
    _ETA,
    _FORECAST,
    _GENERATED,
    _INTENSITY,
    _OBSERVED,
    _P,
    _RADAR_TS,
    _RUN,
)

UTC = timezone.utc

# ---------------------------------------------------------------------------
# Constants this module owns
# ---------------------------------------------------------------------------

#: ``rule_source`` values. ``served`` replays the fitted threshold table on
#: every month it was fitted on — in-sample in the thresholds, which is a
#: four-number leak over ten months and is declared rather than hidden.
#: ``lomo`` replays each month under a threshold fitted without it.
RULE_SERVED = "served"
RULE_LOMO = "lomo"

#: A day more than this share of whose decision rows carry no Phase-H
#: features is a feature-gap day: the rows there fell back to the curve
#: scale while the threshold was fitted on the ``p_post`` scale, so the
#: events are judged by a different rule wearing the same number.
FEATURE_GAP_SHARE = 0.5

#: The feature column whose nullity decides whether a row "has features".
#: One column, named once: the fit, the sweep and this module all have to
#: answer that question the same way (``postprocess.feature_only_columns``
#: is the full list; this is its cheapest representative and the one the
#: plan's day-count reconnaissance uses).
FEATURE_PROBE_COLUMN = "obs_max_5km_mm_h"

#: Minutes between consecutive decision rows above which the window has a
#: hole worth showing. The fullRange cadence is 10 minutes, so anything
#: longer is a missed cycle; :data:`DEFAULT_COVERAGE_GAP_MIN` is the
#: separate, larger line at which the coverage rule stops watching.
DECISION_GAP_MIN = SLOT_MIN

#: Why a row was passed over without touching the engine's state. The one
#: reason ``replay_station`` itself has: no probability at the rule's lead.
SKIP_NO_PROBABILITY = "no_probability"

#: Sampling groups: the target counts are quoted per group, and a group
#: may cover two classes. ``late`` does, because ``late`` (the warning)
#: and ``miss_late`` (its onset) are the two sides of ONE episode —
#: ``score_warnings`` emits them in pairs — and a reviewer shown both
#: would judge the same rain twice.
SAMPLE_GROUPS: dict[str, tuple[str, ...]] = {
    "false_alarm": ("false_alarm",),
    "miss": ("miss",),
    "late": ("late", "miss_late"),
    "hit": ("hit",),
    "uncovered": ("uncovered",),
}

#: Groups drawn as a CONTROL: they are not failures, they are the base
#: rate without which a tag tally says nothing (plan §4.7). ``uncovered``
#: is a control for a second reason: the coverage rule removes it from
#: POD's denominator and nothing else in the system ever audits that.
CONTROL_GROUPS: frozenset[str] = frozenset({"hit", "uncovered"})

#: The first bundle's allocation — 300 events (plan [DECIDE-3]).
DEFAULT_CLASS_TARGETS: dict[str, int] = {
    "false_alarm": 120,
    "miss": 100,
    "late": 20,
    "hit": 50,
    "uncovered": 10,
}

#: Events per non-empty cell before proportional allocation starts, so a
#: rare (season, region, band) combination is represented at all. A pooled
#: proportional draw would give the whole winter shoulder of Bornholm zero
#: events and the review would never discover it behaves differently.
DEFAULT_FLOOR_PER_CELL = 1

#: The band an event falls in when the value its band is read from is
#: null. Not one of :data:`~dmi_nowcast_core.review_schema.INTENSITY_BANDS`
#: on purpose: it is the absence of a measurement, and folding it into
#: ``trace`` would turn "we do not know" into "almost nothing".
UNKNOWN_BAND = "unknown"

#: The value ``dual_truth``'s window carries to say which anchor it was
#: built from. The window is NOT the same for the two sides (a warning
#: promises a future interval; an onset names an instant), and an event
#: whose window is unstated cannot be read at all.
WINDOW_WARNING = "warning"
WINDOW_ONSET = "onset"

#: Priority-ordered station regions, a verbatim copy of
#: ``scripts/build_calibration_points.CALIBRATION_REGIONS``. Copied rather
#: than imported because that is a CLI script and a library importing one
#: by ``sys.path`` surgery is worse than eight boxes duplicated —
#: ``test_review.py`` pins the two against each other so they cannot
#: drift. First box containing the point wins.
REGION_BOXES: dict[str, tuple[float, float, float, float]] = {
    "Bornholm": (54.90, 55.35, 14.50, 15.30),
    "Hovedstaden": (55.50, 56.15, 11.90, 12.75),
    "Sjælland": (54.50, 56.05, 10.85, 12.70),
    "Nordjylland": (56.60, 57.80, 8.00, 11.65),
    "Sønderjylland": (54.55, 55.15, 8.00, 10.10),
    "Fyn": (54.60, 55.65, 9.72, 10.90),
    "Sydjylland": (55.15, 55.95, 8.00, 10.90),
    "Midtjylland": (55.95, 56.60, 8.00, 11.60),
}

#: What a point outside every box is called. A station off the boxes is a
#: real thing (Anholt, a coastal sliver, a station the boxes never saw)
#: and it must land in a stratum rather than vanish from the sample.
REGION_OTHER = "Other"


def region_of(lat: float, lon: float) -> str:
    """The first :data:`REGION_BOXES` entry containing the point."""
    for name, (lat_lo, lat_hi, lon_lo, lon_hi) in REGION_BOXES.items():
        if lat_lo <= lat <= lat_hi and lon_lo <= lon <= lon_hi:
            return name
    return REGION_OTHER


def intensity_band(value: float | None) -> str:
    """The band ``value`` (mm/h) falls in, or :data:`UNKNOWN_BAND`."""
    if value is None or not math.isfinite(float(value)):
        return UNKNOWN_BAND
    rate = float(value)
    for name, low, high in INTENSITY_BANDS:
        if low <= rate < high:
            return name
    return INTENSITY_BANDS[-1][0]


def two_slot_rate_mm_h(depth_mm: float | None, *, slot_min: int = SLOT_MIN) -> float | None:
    """An onset's two-slot depth as a rain RATE, so it can be banded.

    The onset amount test sums the onset slot and the one after it — 20
    minutes at the 10-minute cadence — and
    :data:`~dmi_nowcast_core.review_schema.INTENSITY_BANDS` is in mm/h, so
    the depth is spread across those two slots: ``mm * 60 / (2 * slot)``.
    Stated here rather than inline because it is the only place an onset's
    band and a forecast's band are made comparable, and they are not the
    same measurement — which is exactly why every event carries
    ``intensity_band_source``.
    """
    if depth_mm is None:
        return None
    return float(depth_mm) * 60.0 / (2.0 * float(slot_min))


def _as_utc(value: datetime, what: str = "datetime") -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{what} must be a datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{what} must be timezone-aware (UTC)")
    return value.astimezone(UTC)


def _iso(value: datetime | None) -> str | None:
    """An instant as the bundle spells it, or ``None`` — never ``""``."""
    return None if value is None else _as_utc(value).isoformat()


def _minutes(delta: timedelta) -> float:
    return delta.total_seconds() / 60.0


# ---------------------------------------------------------------------------
# Stations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StationMeta:
    """One gauge point, as the bundle's ``stations.json`` describes it.

    ``region`` is a sampling stratum, not a fact about the station, so it
    is a field rather than a lookup: a caller with a better regionalisation
    (the calibration corpus's, say) can supply it and the bundle records
    what it used. :func:`station_meta` fills it from :func:`region_of`.
    """

    station_id: str
    name: str
    lat: float
    lon: float
    region: str

    def document(self) -> dict:
        return {
            "station_id": self.station_id,
            "station_name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "region": self.region,
        }


def station_meta(
    station_id: str, name: str, lat: float, lon: float, region: str | None = None,
) -> StationMeta:
    """A :class:`StationMeta` with the region derived when not given."""
    return StationMeta(
        station_id=str(station_id),
        name=str(name),
        lat=float(lat),
        lon=float(lon),
        region=region_of(float(lat), float(lon)) if region is None else str(region),
    )


@dataclass(frozen=True)
class Neighbour:
    """A gauge close enough to speak to "was this gauge the odd one out".

    Never a third vote on the same question: at
    :data:`~dmi_nowcast_core.review_schema.NEIGHBOUR_RADIUS_KM` a wet
    neighbour says rain existed in the AREA, not at the event's station.
    The UI has to say so beside the badge.
    """

    station_id: str
    name: str
    distance_km: float

    def document(self) -> dict:
        return {
            "station_id": self.station_id,
            "station_name": self.name,
            "distance_km": round(self.distance_km, 3),
        }


def neighbours_of(
    station: StationMeta,
    stations: Sequence[StationMeta],
    *,
    radius_km: float = NEIGHBOUR_RADIUS_KM,
) -> tuple[Neighbour, ...]:
    """Every other station within ``radius_km``, nearest first."""
    found = [
        Neighbour(
            other.station_id,
            other.name,
            haversine_km(station.lat, station.lon, other.lat, other.lon),
        )
        for other in stations
        if other.station_id != station.station_id
    ]
    return tuple(
        sorted(
            (n for n in found if n.distance_km <= radius_km),
            key=lambda n: (n.distance_km, n.station_id),
        )
    )


# ---------------------------------------------------------------------------
# The verdict windows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerdictWindow:
    """The interval a dual-truth verdict was read over, and its rule.

    The window is NOT the same on the two sides of the scoreboard, and an
    event that does not carry the one it used cannot be interpreted:

    * a **warning** claims an interval — ``(sent, sent + lead +
      tolerance]``, the very window :func:`~dmi_nowcast_core.warning_score.
      score_warnings` matched it in, so "was the radar wet in the window"
      asks about the promise the warning actually made;
    * an **onset** names an instant, and the instant is a slot END, so the
      first drop fell in the preceding slot and the two-slot amount test
      reaches one slot past it. The window is therefore
      ``(onset - slot - tolerance, onset + slot + tolerance]``: the rain
      event the onset names, plus the same grace the matching allows for
      the radar and the gauge disagreeing about when it started.
    """

    kind: str
    from_utc: datetime
    to_utc: datetime
    rule: str

    def document(self) -> dict:
        return {
            "kind": self.kind,
            "from_utc": _iso(self.from_utc),
            "to_utc": _iso(self.to_utc),
            "rule": self.rule,
        }


def warning_window(
    sent: datetime, *, lead_min: float, tolerance_min: float,
) -> VerdictWindow:
    """``(sent, sent + lead + tolerance]`` — the promise itself."""
    sent = _as_utc(sent, "sent_utc")
    return VerdictWindow(
        WINDOW_WARNING,
        sent,
        sent + timedelta(minutes=float(lead_min) + float(tolerance_min)),
        f"(sent, sent + {lead_min:g} min lead + {tolerance_min:g} min tolerance]",
    )


def onset_window(
    onset: datetime, *, tolerance_min: float, slot_min: int = SLOT_MIN,
) -> VerdictWindow:
    """``(onset - slot - tol, onset + slot + tol]`` — the rain event."""
    onset = _as_utc(onset, "onset_utc")
    span = timedelta(minutes=float(slot_min) + float(tolerance_min))
    return VerdictWindow(
        WINDOW_ONSET,
        onset - span,
        onset + span,
        f"(onset -/+ {slot_min:g} min slot + {tolerance_min:g} min tolerance]",
    )


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewRule:
    """The rule the population is re-decided under, stated in full.

    Nothing here is defaulted from somewhere far away: a bundle drawn
    under a rule it cannot name is a scoreboard of an unknown thing, and
    the whole point of the review is to attribute failures to a rule that
    actually ran. The shipped values live in
    :mod:`dmi_nowcast_core.push_rules` and are imported, not restated.

    ``rule_source`` is the honesty knob:

    * :data:`RULE_SERVED` — the fitted threshold table, applied to the
      months it was fitted on. ``held_out`` is ``False`` and the manifest
      says so: four numbers fitted over ten months is a small leak, but it
      is a leak.
    * :data:`RULE_LOMO` — one threshold per ``(year, month)``, each fitted
      without that month, applied to that month alone through
      :func:`~dmi_nowcast_sidecar.threshold_sweep.filter_tracks_by_months`.
      ``held_out`` is ``True``. The fold thresholds are an INPUT: fitting
      them is a sweep per fold, which belongs to the caller that can
      afford it, not to a bundle builder.

    The two are never mixed: a lomo rule without fold thresholds, or a
    served rule carrying them, raises in :meth:`validate`.
    """

    lead_min: int = DEFAULT_LEAD_MIN
    threshold_pct: int = 40
    threshold_source: str = served_rule_module.SOURCE_CONFIG
    #: ``"postprocess"`` decides on ``p_post_<lead>`` with the engine's
    #: per-row fallback to ``p_rain_<lead>``; ``"curve"`` decides on the
    #: curve and fills nothing. Same two words ``push.probability_source``
    #: uses — the page must not invent a third.
    probability: str = served_rule_module.PROBABILITY_POSTPROCESS
    #: Where the probability in the rows came from: ``"out_of_fold"`` (a
    #: ``fit_postprocess --write-back`` tree — the honest default),
    #: ``"stored"`` (whatever the rows carry) or ``"model_fill"`` (filled
    #: here from a model fitted on all months, which is in-sample).
    probability_provenance: str = "stored"
    persistence_obs: int = DEFAULT_PERSISTENCE_OBS
    rearm_after_min: int = DEFAULT_REARM_AFTER_MIN
    raining_now_mm_h: float = RAIN_THRESHOLD_MM_H
    raining_now_eta_min: float = 1.5
    coverage_gap_min: int = DEFAULT_COVERAGE_GAP_MIN
    tolerance_min: int = DEFAULT_TOLERANCE_MIN
    dry_min: int = DEFAULT_DRY_MIN
    onset_min_mm: float = DEFAULT_ONSET_MIN_MM
    min_useful_lead_min: float = FIT_MIN_USEFUL_LEAD_MIN
    rule_source: str = RULE_SERVED
    #: ``(((year, month), percent), ...)`` for :data:`RULE_LOMO`. A tuple
    #: rather than a mapping so the rule stays frozen and hashable, and so
    #: the manifest records it in one stable order.
    fold_thresholds: tuple[tuple[tuple[int, int], int], ...] = ()

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.rule_source not in (RULE_SERVED, RULE_LOMO):
            raise ValueError(f"unknown rule_source {self.rule_source!r}")
        if self.rule_source == RULE_LOMO and not self.fold_thresholds:
            raise ValueError(
                "rule_source='lomo' needs fold_thresholds: one threshold per "
                "(year, month), each fitted without that month"
            )
        if self.rule_source == RULE_SERVED and self.fold_thresholds:
            raise ValueError(
                "rule_source='served' must not carry fold_thresholds — the "
                "two rules are never mixed silently"
            )
        if self.probability not in (
            served_rule_module.PROBABILITY_POSTPROCESS,
            served_rule_module.PROBABILITY_CURVE,
        ):
            raise ValueError(f"unknown probability {self.probability!r}")

    @property
    def held_out(self) -> bool:
        """Whether the THRESHOLD never saw the month it is applied to."""
        return self.rule_source == RULE_LOMO

    @property
    def column(self) -> str:
        """The column the replay decides on."""
        if self.probability == served_rule_module.PROBABILITY_POSTPROCESS:
            return core_postprocess.post_column(self.lead_min)
        return p_rain_column(self.lead_min)

    @property
    def curve_column(self) -> str:
        return p_rain_column(self.lead_min)

    @property
    def fold_table(self) -> dict[tuple[int, int], int]:
        return {(int(y), int(m)): int(pct) for (y, m), pct in self.fold_thresholds}

    def threshold_for(self, when: datetime) -> int:
        """The percent this instant is decided at."""
        if self.rule_source == RULE_SERVED:
            return int(self.threshold_pct)
        when = _as_utc(when)
        try:
            return self.fold_table[(when.year, when.month)]
        except KeyError:  # pragma: no cover — build_population checks first
            raise KeyError(
                f"no leave-one-month-out threshold for {when.year}-{when.month:02d}"
            ) from None

    def document(self) -> dict:
        """The manifest's ``rule`` block, minus what the loader fills in."""
        return {
            "source": self.rule_source,
            "held_out": self.held_out,
            "lead_min": int(self.lead_min),
            "threshold_pct": (
                int(self.threshold_pct) if self.rule_source == RULE_SERVED else None
            ),
            "threshold_source": self.threshold_source,
            "fold_thresholds": (
                None
                if self.rule_source == RULE_SERVED
                else {f"{y:04d}-{m:02d}": pct for (y, m), pct in self.fold_thresholds}
            ),
            "probability": self.probability,
            "probability_column": self.column,
            "probability_provenance": self.probability_provenance,
            "persistence_obs": int(self.persistence_obs),
            "rearm_after_min": int(self.rearm_after_min),
            "raining_now_mm_h": float(self.raining_now_mm_h),
            "raining_now_eta_min": float(self.raining_now_eta_min),
            "coverage_gap_min": int(self.coverage_gap_min),
            "tolerance_min": int(self.tolerance_min),
            "min_useful_lead_min": float(self.min_useful_lead_min),
        }


def rule_from_served_options(
    options: served_rule_module.ServedRuleOptions,
    *,
    rule_source: str = RULE_SERVED,
    fold_thresholds: Mapping[tuple[int, int], int] | None = None,
    probability_provenance: str = "stored",
    min_useful_lead_min: float = FIT_MIN_USEFUL_LEAD_MIN,
    tolerance_min: int = DEFAULT_TOLERANCE_MIN,
    dry_min: int = DEFAULT_DRY_MIN,
    onset_min_mm: float = DEFAULT_ONSET_MIN_MM,
) -> ReviewRule:
    """A :class:`ReviewRule` from the quality page's own options object.

    The point of entry, and the reason this module invents no rule: the
    threshold comes from
    :func:`~dmi_nowcast_sidecar.served_rule.resolve_threshold` (the same
    document the running service reads, with the same fallback), and the
    timing comes from the ``ServedRuleOptions`` the page builds from
    ``push.persistence_obs`` / ``push.rearm_after_min``. A bundle and the
    scoreboard therefore cannot be about two different rules.
    """
    if served_rule_module.resolve_onset_rule(options) is not None:
        # S11: this tool replays ONE probability column; graded against the
        # table's p_post half alone it would review a rule nobody is on.
        raise ValueError(
            f"{options.thresholds_path}: lead {int(options.lead_min)} is on "
            "the onset AND rule (onset_threshold_pct), which the review "
            "bundle does not replay yet",
        )
    threshold, source = served_rule_module.resolve_threshold(options)
    return ReviewRule(
        lead_min=int(options.lead_min),
        threshold_pct=int(threshold),
        threshold_source=source,
        probability=options.probability_source,
        probability_provenance=probability_provenance,
        persistence_obs=int(options.persistence_obs),
        rearm_after_min=int(options.rearm_after_min),
        raining_now_mm_h=float(options.raining_now_mm_h),
        raining_now_eta_min=float(options.raining_now_eta_min),
        coverage_gap_min=int(options.coverage_gap_min),
        tolerance_min=int(tolerance_min),
        dry_min=int(dry_min),
        onset_min_mm=float(onset_min_mm),
        min_useful_lead_min=float(min_useful_lead_min),
        rule_source=rule_source,
        fold_thresholds=fold_thresholds_of(fold_thresholds),
    )


def fold_thresholds_of(
    table: Mapping[tuple[int, int], int] | None,
) -> tuple[tuple[tuple[int, int], int], ...]:
    """A ``{(year, month): pct}`` mapping in the rule's frozen shape."""
    if not table:
        return ()
    return tuple(
        ((int(year), int(month)), int(pct))
        for (year, month), pct in sorted(table.items())
    )


# ---------------------------------------------------------------------------
# The traced replay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TracedRow:
    """One row of a replay, with the state machine written down.

    Exactly one of these per track record — including records the replay
    passed over — so a trace and a track line up by index and a decision
    block can be assembled without a second search.

    ``armed_before`` / ``streak_before`` are the state the engine actually
    JUDGED this row with, i.e. after any coverage-run reset this row
    triggered. The free re-arm the reset may have handed out is therefore
    not hidden inside them: it shows as ``run_id`` changing while the
    PREVIOUS row's ``armed_after`` was ``False``, which is what
    :func:`run_boundary_rearms` looks for.

    ``below_since_utc`` is the value AFTER the row, on the radar clock —
    the instant the current dry spell started, or ``None`` while armed or
    while an over-threshold observation has just reset it. ``None`` here
    is emphatically not "zero minutes of dry": it means the 60-minute
    clock is not running at all, which is how a station stays disarmed for
    hours in a showery spell.

    ``skipped_reason`` is :data:`SKIP_NO_PROBABILITY` where the row had no
    probability at the rule's lead — off coverage, an unserved lead, or a
    feature gap with no curve behind it — and ``None`` otherwise.
    ``action`` is ``None`` for exactly those rows: the engine never saw
    them, so they have no action, and calling that ``"none"`` would make a
    row the rule never evaluated indistinguishable from one it evaluated
    and dismissed.
    """

    radar_ts: datetime
    generated_at: datetime
    action: str | None
    armed_before: bool
    armed_after: bool
    streak_before: int
    streak_after: int
    below_since_utc: datetime | None
    run_id: int
    skipped_reason: str | None
    p_decision: float | None

    @property
    def notified(self) -> bool:
        return self.action == "notify"


def replay_station_traced(
    track: Sequence[tuple],
    lead_index: int,
    threshold_pct: int,
    *,
    persistence_obs: int,
    rearm_after_min: int,
    raining_now_mm_h: float = RAIN_THRESHOLD_MM_H,
    raining_now_eta_min: float | None = None,
) -> list[TracedRow]:
    """:func:`threshold_sweep.replay_station`, with the state kept.

    Deliberately a copy of that loop rather than a refactor of it: the
    sweep's version runs a million rows per cell in a worker pool and must
    not grow an allocation per row, and a shared function with a "return
    more" flag is how two callers start disagreeing about what they
    measured. The copy is kept honest by the equality test in
    ``sidecar/tests/test_review.py``, which asserts that the warnings this
    produces are the warnings ``replay_station`` produces, row for row, on
    the same tracks and the same rule — including a coverage-run boundary,
    an already-raining silence and a long showery spell.

    Every argument is passed to ``evaluate`` exactly as ``replay_station``
    passes it, ``raining_now_eta_min=None`` included (which leaves the
    engine's own default in place rather than naming a second one).
    """
    eng = decision_engine
    rules = eng.Rules(
        persistence_obs=int(persistence_obs),
        rearm_after_min=int(rearm_after_min),
        raining_now_mm_h=float(raining_now_mm_h),
        **(
            {}
            if raining_now_eta_min is None
            else {"raining_now_eta_min": float(raining_now_eta_min)}
        ),
    )
    state = eng.INITIAL_STATE
    run: int | None = None
    out: list[TracedRow] = []
    for record in track:
        if record[_RUN] != run:
            run = record[_RUN]
            state = eng.INITIAL_STATE
        entering = state
        p_rain = record[_P][lead_index]
        if p_rain is None:
            out.append(TracedRow(
                radar_ts=record[_RADAR_TS],
                generated_at=record[_GENERATED],
                action=None,
                armed_before=entering.armed,
                armed_after=state.armed,
                streak_before=entering.streak,
                streak_after=state.streak,
                below_since_utc=state.below_since_utc,
                run_id=int(record[_RUN]),
                skipped_reason=SKIP_NO_PROBABILITY,
                p_decision=None,
            ))
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
        out.append(TracedRow(
            radar_ts=record[_RADAR_TS],
            generated_at=record[_GENERATED],
            action=decision.action,
            armed_before=entering.armed,
            armed_after=state.armed,
            streak_before=entering.streak,
            streak_after=state.streak,
            below_since_utc=state.below_since_utc,
            run_id=int(record[_RUN]),
            skipped_reason=None,
            p_decision=float(p_rain),
        ))
    return out


def traced_warnings(
    track: Sequence[tuple],
    trace: Sequence[TracedRow],
    *,
    with_probability: bool = False,
) -> list[tuple]:
    """The warnings a trace holds, in ``replay_station``'s own shape.

    ``(generated_at, eta_min)``, or ``(generated_at, eta_min, p)`` with
    ``with_probability``. The ETA comes from the track because a
    :class:`TracedRow` deliberately does not duplicate it — one column,
    one owner — and the two line up by index.
    """
    if len(track) != len(trace):
        raise ValueError(
            f"trace has {len(trace)} row(s) for a track of {len(track)}"
        )
    out: list[tuple] = []
    for record, row in zip(track, trace):
        if record[_RADAR_TS] != row.radar_ts:
            # The pairing is by index, so a trace assembled out of order
            # (a leave-one-month-out merge, say) would quietly attach one
            # row's ETA to another row's warning.
            raise ValueError(
                f"trace row {row.radar_ts} does not line up with track row "
                f"{record[_RADAR_TS]}"
            )
        if row.action != "notify":
            continue
        out.append(
            (record[_GENERATED], record[_ETA], row.p_decision)
            if with_probability
            else (record[_GENERATED], record[_ETA])
        )
    return out


def run_boundary_rearms(trace: Sequence[TracedRow]) -> tuple[datetime, ...]:
    """``radar_ts`` of every row the replay handed a free re-arm.

    ``replay_station`` resets to ``INITIAL_STATE`` at the head of every
    coverage run, so a gap longer than ``coverage_gap_min`` re-arms a
    subscription the live service would have left disarmed. Only the rows
    where the machine was ACTUALLY disarmed on the way in count: a reset
    that discards a streak of two changes nothing a reviewer would
    misread.
    """
    out: list[datetime] = []
    previous: TracedRow | None = None
    for row in trace:
        if (
            previous is not None
            and row.run_id != previous.run_id
            and not previous.armed_after
            and row.armed_before
        ):
            out.append(row.radar_ts)
        previous = row
    return tuple(out)


@dataclass(frozen=True)
class ArmState:
    """The state machine as of one instant, and how long it owes.

    ``minutes_to_rearm`` is ``None`` when the subscription is armed (there
    is nothing to wait for) **and** when it is disarmed with no dry clock
    running: an over-threshold row has just reset ``below_since_utc``, so
    the re-arm is not "0 minutes away" or "60 minutes away" — it has not
    started, and a number there would be a fabrication. That is the exact
    state plan §4.4 is about.
    """

    armed: bool
    streak: int
    below_since_utc: datetime | None
    minutes_to_rearm: float | None
    at_utc: datetime | None
    radar_ts: datetime | None
    run_id: int | None
    #: ``True`` when the snapshot is the state ENTERING the row at the
    #: instant asked about, ``False`` when it is the state left behind by
    #: the last row before it.
    entering: bool

    @property
    def label(self) -> str:
        return "armed" if self.armed else "disarmed"

    # No ``document()``: the prologue names the fields it publishes, and a
    # second serialisation of the same state is the drift this block was
    # just cleaned of.


def arm_state_at(
    trace: Sequence[TracedRow], when: datetime, *, rearm_after_min: int,
) -> ArmState:
    """The engine's state at ``when``, read off a trace.

    A row stamped exactly at ``when`` (every warning anchor is) reports
    the state the engine ENTERED that row with — the state that decided
    it. Anywhere else, the state the last evaluated row left behind. The
    re-arm countdown is measured on the RADAR clock from that row, because
    that is the arithmetic ``evaluate`` itself does; it does not tick
    between frames, and pretending it does would put a reviewer minutes
    away from a re-arm that had not moved.

    With nothing evaluated before ``when`` the answer is the replay's own
    starting point — armed, no streak, no clock — reported with
    ``at_utc=None`` so it is readable as "no decision was seen" rather
    than as a measurement.
    """
    when = _as_utc(when, "when")
    index: int | None = None
    for i, row in enumerate(trace):
        if _as_utc(row.generated_at) <= when:
            index = i
        else:
            break
    if index is None:
        return ArmState(
            armed=True,
            streak=0,
            below_since_utc=None,
            minutes_to_rearm=None,
            at_utc=None,
            radar_ts=None,
            run_id=None,
            entering=True,
        )
    row = trace[index]
    exact = _as_utc(row.generated_at) == when
    if exact:
        previous = trace[index - 1] if index > 0 else None
        below = (
            previous.below_since_utc
            if previous is not None and previous.run_id == row.run_id
            else None
        )
        armed, streak = row.armed_before, row.streak_before
    else:
        below = row.below_since_utc
        armed, streak = row.armed_after, row.streak_after
    remaining: float | None = None
    if not armed and below is not None:
        elapsed = _minutes(_as_utc(row.radar_ts) - _as_utc(below))
        remaining = max(0.0, float(rearm_after_min) - elapsed)
    return ArmState(
        armed=armed,
        streak=streak,
        below_since_utc=below,
        minutes_to_rearm=remaining,
        at_utc=_as_utc(row.generated_at),
        radar_ts=_as_utc(row.radar_ts),
        run_id=row.run_id,
        entering=exact,
    )


# ---------------------------------------------------------------------------
# The radar disc, read off the stored rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RadarSlots:
    """The radar's word at one station, on the gauge's own slot grid.

    ``observed_mm_h`` in a decision row IS the production "raining now"
    reading — the p90 over the ~1 km disc
    (:data:`~dmi_nowcast_core.review_schema.RADAR_DISC_RADIUS_M`,
    ``sample.sample_disc``) that the engine's already-raining test reads —
    so the second opinion is computed from the number the service itself
    acted on, and the bundle needs no HDF5 to state it.

    Binned exactly as
    :func:`~dmi_nowcast_sidecar.threshold_sweep.radar_truth` bins it: the
    slot's largest rate wins, a null rate leaves the slot UNKNOWN (never
    dry), and the grid is contiguous so a hole in the rows cannot pass for
    a dry spell.
    """

    station_id: str
    #: ``(slot_end, rate_mm_h | None)``, ascending, one slot apart.
    slots: tuple[tuple[datetime, float | None], ...]
    threshold_mm_h: float = RAIN_THRESHOLD_MM_H
    slot_min: int = SLOT_MIN

    def between(
        self, start: datetime, end: datetime,
    ) -> tuple[tuple[datetime, float | None], ...]:
        """Slots whose END falls in ``(start, end]``."""
        start, end = _as_utc(start), _as_utc(end)
        return tuple(
            (stamp, rate) for stamp, rate in self.slots if start < stamp <= end
        )

    def wet_between(self, start: datetime, end: datetime) -> bool | None:
        """``True`` / ``False`` / ``None`` (nothing known) over a window."""
        known = [rate for _stamp, rate in self.between(start, end) if rate is not None]
        if not known:
            return None
        return any(rate >= self.threshold_mm_h for rate in known)

    def series(self, start: datetime, end: datetime) -> list[dict]:
        """The window's slots as the bundle writes them.

        Shaped like a gauge slot so one component can draw both, with
        ``mm`` derived the way
        :func:`~dmi_nowcast_sidecar.threshold_sweep.radar_truth` derives
        it — the slot's rate held across the slot — and ``mm_h`` kept
        beside it, because the radar reports a RATE and the gauge a depth.
        """
        hours = self.slot_min / 60.0
        return [
            {
                "slot_end_utc": _iso(stamp),
                "known": rate is not None,
                "wet": rate is not None and rate >= self.threshold_mm_h,
                "mm": None if rate is None else rate * hours,
                "mm_h": rate,
            }
            for stamp, rate in self.between(start, end)
        ]


def radar_slots_of(
    track: Sequence[tuple],
    station_id: str,
    *,
    threshold_mm_h: float = RAIN_THRESHOLD_MM_H,
    slot_min: int = SLOT_MIN,
) -> RadarSlots:
    """One station's :class:`RadarSlots` from its track."""
    seen: dict[datetime, float] = {}
    for record in track:
        observed = record[_OBSERVED]
        if observed is None:
            continue
        slot = slot_end_of(_as_utc(record[_RADAR_TS]), slot_min=slot_min)
        rate = float(observed)
        if seen.get(slot) is None or rate > seen[slot]:
            seen[slot] = rate
    if not seen:
        return RadarSlots(station_id, (), threshold_mm_h, slot_min)
    step = timedelta(minutes=slot_min)
    cursor, last = min(seen), max(seen)
    grid: list[tuple[datetime, float | None]] = []
    while cursor <= last:
        grid.append((cursor, seen.get(cursor)))
        cursor += step
    return RadarSlots(station_id, tuple(grid), threshold_mm_h, slot_min)


# ---------------------------------------------------------------------------
# The gauge, read three-valued
# ---------------------------------------------------------------------------


def gauge_series(
    slots: StationSlots | None, start: datetime, end: datetime,
) -> list[dict]:
    """A station's slots over ``(start, end]``, ``known`` beside ``wet``.

    ``{"wet": false, "known": false}`` is UNKNOWN, not dry — the
    distinction :class:`~dmi_nowcast_core.warning_score.StationSlots`
    exists to keep and the one a reader collapsing them turns into
    evidence of a false alarm.
    """
    if slots is None or len(slots) == 0:
        return []
    start, end = _as_utc(start), _as_utc(end)
    out: list[dict] = []
    for i in range(len(slots)):
        stamp = datetime.fromtimestamp(int(slots.slot_end[i]), UTC)
        if stamp <= start:
            continue
        if stamp > end:
            break
        known = bool(slots.known[i])
        mm = float(slots.mm[i]) if slots.mm[i] == slots.mm[i] else None
        dur = float(slots.dur[i]) if slots.dur[i] == slots.dur[i] else None
        out.append({
            "slot_end_utc": _iso(stamp),
            "known": known,
            # ``review_schema`` fixes this encoding: {"wet": false,
            # "known": false} is UNKNOWN, not dry — the same split
            # ``StationSlots`` keeps. ``wet`` is only meaningful beside
            # ``known``, and ``mm`` stays null where nothing was weighed.
            "wet": bool(slots.wet[i]) and known,
            "mm": mm,
            "dur_min": dur,
        })
    return out


def gauge_wet_between(
    slots: StationSlots | None, start: datetime, end: datetime,
) -> bool | None:
    """``True`` / ``False`` / ``None`` over ``(start, end]``.

    ``None`` means the station reported nothing in the window at all. The
    difference matters: a false alarm over a silent gauge is
    ``fa_gauge_unreported``, which is not a forecast failure, and a bundle
    that called it ``both_dry`` would have invented the evidence.
    """
    if slots is None or len(slots) == 0:
        return None
    start, end = _as_utc(start), _as_utc(end)
    wet = False
    known_any = False
    for i in range(len(slots)):
        stamp = datetime.fromtimestamp(int(slots.slot_end[i]), UTC)
        if stamp <= start:
            continue
        if stamp > end:
            break
        if not bool(slots.known[i]):
            continue
        known_any = True
        if bool(slots.wet[i]):
            wet = True
    return wet if known_any else None


def gauge_known_between(
    slots: StationSlots | None, start: datetime, end: datetime,
) -> int:
    """How many slots the station actually reported in ``(start, end]``."""
    if slots is None or len(slots) == 0:
        return 0
    start, end = _as_utc(start), _as_utc(end)
    count = 0
    for i in range(len(slots)):
        stamp = datetime.fromtimestamp(int(slots.slot_end[i]), UTC)
        if stamp <= start:
            continue
        if stamp > end:
            break
        if bool(slots.known[i]):
            count += 1
    return count


def dual_truth_of(gauge_wet: bool | None, radar_wet: bool | None) -> str | None:
    """The 2×2 verdict, or ``None`` when an instrument said nothing.

    Deliberately three-valued although
    :data:`~dmi_nowcast_core.review_schema.DUAL_TRUTH_CLASSES` names four
    quadrants: a quadrant asserts two facts, and an event whose gauge was
    silent or whose radar had no reading in the window supports neither.
    ``null`` there is the measurement; ``both_dry`` would be a fabrication
    and the most damaging one available, since it reads as "the forecast
    invented rain".
    """
    if gauge_wet is None or radar_wet is None:
        return None
    if gauge_wet and radar_wet:
        return "both_wet"
    if radar_wet and not gauge_wet:
        return "radar_wet_gauge_dry"
    if gauge_wet and not radar_wet:
        return "gauge_wet_radar_dry"
    return "both_dry"


# ---------------------------------------------------------------------------
# The event record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventRecord:
    """One candidate event, before any sampling.

    Everything a stratum, a filter or a badge needs, and nothing that
    costs a file read. The detail document (:func:`build_event`) repeats
    this under ``"index"`` so an event file means something on its own.

    ``window_used`` rides along because the dual-truth verdict is read
    over an interval whose definition DEPENDS ON THE CLASS (see
    :class:`VerdictWindow`), and a verdict whose window is unstated cannot
    be compared with another event's.
    """

    event_id: str
    event_class: str
    station_id: str
    anchor_utc: datetime
    sent_utc: datetime | None
    onset_utc: datetime | None
    eta_min: float | None
    p_decision: float | None
    p_decision_source: str
    threshold_pct: int | None
    lead_error_min: float | None
    season: str
    region: str
    intensity_mm_h: float | None
    intensity_band: str
    intensity_band_source: str
    onset_two_slot_mm: float | None
    dual_truth: str | None
    #: Three-valued, like every other verdict in the bundle: ``None`` is a
    #: gauge that reported nothing in the window. ``False`` there would be
    #: the exact collapse ``review_schema`` forbids — a gap in the record
    #: read as evidence of a false alarm — and it is the field the index
    #: filters on, so the null has to survive that far.
    gauge_wet_in_window: bool | None
    radar_wet_in_window: bool | None
    neighbour_wet_in_window: bool | None
    neighbour_n_known: int
    arm_state_at_anchor: str
    minutes_to_rearm_at_anchor: float | None
    flags: tuple[str, ...]
    control: bool
    window_used: VerdictWindow

    @property
    def warning_side(self) -> bool:
        return self.event_class in WARNING_SIDE_CLASSES

    @property
    def group(self) -> str:
        return group_of(self.event_class)

    @property
    def hour_utc(self) -> int:
        return int(_as_utc(self.anchor_utc).hour)

    @property
    def pair_key(self) -> tuple:
        """What identifies the EPISODE rather than the row.

        A ``late`` warning and its ``miss_late`` onset are one episode seen
        from both ends; this is what :func:`stratify` collapses them on.
        """
        return (self.station_id, _iso(self.sent_utc), _iso(self.onset_utc))

    def stratum_value(self, key: str) -> str:
        if key == "outcome_class":
            return self.event_class
        if key == "season":
            return self.season
        if key == "region":
            return self.region
        if key == "intensity_band":
            return self.intensity_band
        raise ValueError(f"unknown stratum key {key!r}")

    def stratum(self, keys: Sequence[str] = STRATA_KEYS) -> str:
        return "|".join(self.stratum_value(key) for key in keys)


def group_of(event_class: str) -> str:
    """The sampling group one outcome class belongs to."""
    for name, classes in SAMPLE_GROUPS.items():
        if event_class in classes:
            return name
    raise ValueError(f"unknown event class {event_class!r}")


def index_row(
    record: EventRecord,
    station: StationMeta,
    *,
    frames: int | None = None,
    frames_missing: int | None = None,
    strata_keys: Sequence[str] = STRATA_KEYS,
) -> dict:
    """The ``events.json`` row — exactly :data:`INDEX_FIELDS`, in order.

    ``frames`` / ``frames_missing`` are counts the frame plan supplies;
    :func:`build_event` passes the plan's own numbers and says in the
    ``frames`` block which source they came from. Left out they are
    ``None``, not 0 — a bundle built before the frames were planned has
    not counted zero frames, it has not counted.
    """
    eta_arrival = (
        None
        if record.sent_utc is None or record.eta_min is None
        else _as_utc(record.sent_utc) + timedelta(minutes=float(record.eta_min))
    )
    row = {
        "event_id": record.event_id,
        "class": record.event_class,
        "control": record.control,
        "station_id": record.station_id,
        "station_name": station.name,
        "region": record.region,
        "lat": station.lat,
        "lon": station.lon,
        "anchor_utc": _iso(record.anchor_utc),
        "sent_utc": _iso(record.sent_utc),
        "onset_utc": _iso(record.onset_utc),
        "eta_min": record.eta_min,
        "eta_arrival_utc": _iso(eta_arrival),
        "p_decision": record.p_decision,
        "p_decision_source": record.p_decision_source,
        "threshold_pct": record.threshold_pct,
        "lead_error_min": record.lead_error_min,
        "dual_truth": record.dual_truth,
        "gauge_wet_in_window": record.gauge_wet_in_window,
        "radar_wet_in_window": record.radar_wet_in_window,
        "neighbour_wet_in_window": record.neighbour_wet_in_window,
        "neighbour_n_known": record.neighbour_n_known,
        "season": record.season,
        "hour_utc": record.hour_utc,
        "intensity_band": record.intensity_band,
        "intensity_mm_h": record.intensity_mm_h,
        "intensity_band_source": record.intensity_band_source,
        "onset_two_slot_mm": record.onset_two_slot_mm,
        "arm_state_at_anchor": record.arm_state_at_anchor,
        "minutes_to_rearm_at_anchor": record.minutes_to_rearm_at_anchor,
        "frames": frames,
        "frames_missing": frames_missing,
        "flags": list(record.flags),
        # A mapping, not a joined label: the browser facets on the
        # individual keys, and splitting a string on "|" in four places is
        # how a region called "Nordjylland|Læsø" would break the UI.
        "stratum": {
            key: record.stratum_value(key) for key in strata_keys
        },
        "detail": f"events/{record.event_id}.json",
    }
    missing = [name for name in INDEX_FIELDS if name not in row]
    if missing:  # pragma: no cover — a schema change without a builder change
        raise AssertionError(f"index row is missing {missing}")
    return {name: row[name] for name in INDEX_FIELDS}


# ---------------------------------------------------------------------------
# The population
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Population:
    """Every gradeable event under one rule, plus what a detail needs.

    Held together in one object because :func:`build_event` needs the
    tracks, the traces, the stored rows and the truth that produced a
    record, and handing those around separately is how a detail document
    ends up describing a different replay from the one the record came
    out of.
    """

    records: tuple[EventRecord, ...]
    rule: ReviewRule
    stations: Mapping[str, StationMeta]
    neighbours: Mapping[str, tuple[Neighbour, ...]]
    rows: Mapping[tuple[Any, str], Mapping[str, Any]]
    #: ``(radar_ts, station_id) -> which decision tree the row came from``.
    #: Empty where the caller did not say: a bundle built from one tree
    #: knows the answer trivially, and a guess would be worse than a null.
    row_sources: Mapping[tuple[Any, str], str]
    tracks: Mapping[str, Sequence[tuple]]
    traces: Mapping[str, Sequence[TracedRow]]
    coverage: Mapping[str, Sequence[tuple[datetime, datetime]]]
    truth: GaugeTruth
    onsets: Mapping[str, Sequence[datetime]]
    onset_amounts: Mapping[tuple[str, datetime], float]
    known_until: Mapping[str, datetime]
    radar: Mapping[str, RadarSlots]
    dead_gauges: tuple[str, ...]
    suspect_months: frozenset[tuple[str, tuple[int, int]]]
    curve_fallback_keys: frozenset[tuple[Any, str]]
    feature_gap_dates: tuple[date, ...]
    day_feature_counts: Mapping[date, tuple[int, int]]
    leads: tuple[int, ...]
    design_leads: tuple[int, ...]
    window_min: int
    provenance: Mapping[str, Any]
    excluded: Mapping[str, int]

    def by_class(self, event_class: str) -> tuple[EventRecord, ...]:
        return tuple(r for r in self.records if r.event_class == event_class)

    def get(self, event: str) -> EventRecord:
        for record in self.records:
            if record.event_id == event:
                return record
        raise KeyError(event)


def feature_gap_days(
    rows: Iterable[Mapping[str, Any]],
    *,
    probe: str = FEATURE_PROBE_COLUMN,
    share: float = FEATURE_GAP_SHARE,
) -> tuple[dict[date, tuple[int, int]], tuple[date, ...]]:
    """``({day: (rows, with features)}, the gap days)`` — measured, not assumed.

    The live writer lost its Phase-H features for a stretch in September
    2026 and the repository cannot say when the fix landed on the VM, so
    the only honest way to find the gap is to count it. The day is the
    ``radar_ts``'s UTC date — the frame's own day, which is how the replay
    partitions its parquet — and a day where more than ``share`` of the
    rows carry no ``obs_max_5km_mm_h`` is a gap day.

    A gap day is not corrupt data; it is data judged by a different rule.
    Its rows fall back to the curve while the thresholds were fitted on
    the ``p_post`` scale, so sampling them beside honest rows would mix
    two rules under one name. They are excluded by default and available
    with ``allow_feature_gap=True``, flagged ``feature_gap``.
    """
    counts: dict[date, list[int]] = {}
    for row in rows:
        stamp = row.get("radar_ts")
        if stamp is None:
            continue
        day = _as_utc(stamp, "radar_ts").date()
        entry = counts.setdefault(day, [0, 0])
        entry[0] += 1
        if row.get(probe) is not None:
            entry[1] += 1
    summary = {day: (total, filled) for day, (total, filled) in sorted(counts.items())}
    gaps = tuple(
        day
        for day, (total, filled) in summary.items()
        if total > 0 and (total - filled) > share * total
    )
    return summary, gaps


def _suspect_months(
    truth: GaugeTruth, *, min_known_slots: int,
) -> frozenset[tuple[str, tuple[int, int]]]:
    """``(station, (year, month))`` pairs that fail the dead-gauge rule.

    :func:`~dmi_nowcast_core.warning_score.dead_gauges` is computed over
    the WHOLE window, so a gauge that died in June still contributes five
    honest months and one month of manufactured false alarms. This is the
    same rule — reported often, wet never — applied month by month, which
    is what the ``suspect_gauge_month`` flag warns a reviewer about.
    """
    import numpy as np

    floor = int(min_known_slots)
    if floor <= 0:
        return frozenset()
    out: set[tuple[str, tuple[int, int]]] = set()
    for station, series in truth.series.items():
        if len(series) == 0:
            continue
        # Month of each slot END, without a datetime per slot: seconds to
        # datetime64, truncated to month resolution, is one numpy cast.
        months = np.asarray(series.slot_end, dtype="datetime64[s]").astype(
            "datetime64[M]",
        )
        for month in np.unique(months):
            mask = months == month
            known = int(np.count_nonzero(series.known[mask]))
            wet = int(np.count_nonzero(series.wet[mask] & series.known[mask]))
            if wet == 0 and known >= floor:
                stamp = month.astype(object)
                out.add((str(station), (int(stamp.year), int(stamp.month))))
    return frozenset(out)


def gauge_truth_for(
    corpus_dir: Path,
    station_ids: Sequence[str],
    window: tuple[datetime, datetime],
    *,
    dry_min: int = DEFAULT_DRY_MIN,
    onset_min_mm: float = DEFAULT_ONSET_MIN_MM,
    min_known_slots: int = DEFAULT_MIN_KNOWN_SLOTS,
    log: Callable[[str], None] | None = None,
) -> tuple[GaugeTruth, dict[str, list[datetime]], dict[str, datetime], list[str]]:
    """:func:`threshold_sweep.gauge_truth`, keeping the slot grid.

    Identical inputs, identical rules, identical results — the same
    ``gauge_truth_vectorised`` call with the same pad, the same
    ``dead_gauges`` exclusion — with one difference: the
    :class:`~dmi_nowcast_core.warning_score.GaugeTruth` itself comes back
    rather than being reduced to onsets and ``known_until``. The review
    needs the slot grid to answer "was the gauge wet in this window" and
    "did it report at all", and reading the archive a second time to get
    it would double the most expensive step of the build.

    ``sidecar/tests/test_review.py`` asserts the onsets, the horizons and
    the dead-gauge list match ``threshold_sweep.gauge_truth``'s on the
    same store, so this cannot drift into a second definition of truth.
    """
    from dmi_nowcast_core.warning_score import dead_gauges, gauge_truth_vectorised

    pad = timedelta(minutes=GAUGE_PAD_MIN)
    start, end = window
    truth = gauge_truth_vectorised(
        Path(corpus_dir), start - pad, end + pad, list(station_ids),
        dry_min=dry_min, onset_min_mm=onset_min_mm,
        pad_min=GAUGE_PAD_MIN, log=log,
    )
    dead = dead_gauges(truth, min_known_slots=int(min_known_slots), log=log)
    excluded = set(dead)
    onsets = {s: list(v) for s, v in truth.onsets.items() if s not in excluded}
    known_until = {s: v for s, v in truth.known_until.items() if s not in excluded}
    return truth, onsets, known_until, list(dead)


def _prepare_rows(
    rows: Sequence[Mapping[str, Any]], rule: ReviewRule,
) -> tuple[list[dict], frozenset[tuple[Any, str]]]:
    """Apply ``Observation.p_decision``'s fallback as a column.

    ``replay_station`` sees ONE probability column, and the engine's own
    rule is "take ``p_post`` where it exists and ``p_rain`` where it does
    not, per observation" — one point off coverage for the model must not
    silence it. :mod:`~dmi_nowcast_sidecar.served_rule` does this before
    its replay; so does this, on the same terms, counting the rows that
    took the fallback so the manifest and the per-event
    ``p_decision_source`` can both say which scale the event was judged
    on.
    """
    column, curve = rule.column, rule.curve_column
    prepared: list[dict] = []
    fallback: set[tuple[Any, str]] = set()
    for row in rows:
        out = dict(row)
        key = (out.get("radar_ts"), str(out.get("station_id")))
        if column != curve and out.get(column) is None and out.get(curve) is not None:
            out[column] = out[curve]
            fallback.add(key)
        prepared.append(out)
    return prepared, frozenset(fallback)


def _trace_station(
    track: Sequence[tuple], rule: ReviewRule,
) -> list[TracedRow]:
    """One station's trace under the rule, served or leave-one-month-out.

    Under :data:`RULE_LOMO` each ``(year, month)`` is replayed on its own
    through :func:`~dmi_nowcast_sidecar.threshold_sweep.filter_tracks_by_
    months`, under the threshold fitted without it, exactly as the
    seasonal strata are replayed: the coverage runs, the arming and the
    re-arm clock start fresh inside the slice rather than being carried
    across months that were cut away. The folds partition the track, so
    concatenating them in ``radar_ts`` order rebuilds the full timeline —
    with a seam at each month boundary, which is recorded as a caveat and
    shows up in the trace as a ``run_id`` change.
    """
    if rule.rule_source == RULE_SERVED:
        return replay_station_traced(
            track, 0, rule.threshold_pct,
            persistence_obs=rule.persistence_obs,
            rearm_after_min=rule.rearm_after_min,
            raining_now_mm_h=rule.raining_now_mm_h,
            raining_now_eta_min=rule.raining_now_eta_min,
        )
    months = sorted({
        (_as_utc(record[_RADAR_TS]).year, _as_utc(record[_RADAR_TS]).month)
        for record in track
    })
    table = rule.fold_table
    unknown = [m for m in months if m not in table]
    if unknown:
        raise KeyError(
            "leave-one-month-out needs a threshold for every month in the "
            f"window; missing {['%04d-%02d' % m for m in unknown]}"
        )
    out: list[TracedRow] = []
    offset = 0
    for month in months:
        sliced = filter_tracks_by_months(
            {"_": track}, [month], coverage_gap_min=rule.coverage_gap_min,
        ).get("_", [])
        if not sliced:
            continue
        trace = replay_station_traced(
            sliced, 0, table[month],
            persistence_obs=rule.persistence_obs,
            rearm_after_min=rule.rearm_after_min,
            raining_now_mm_h=rule.raining_now_mm_h,
            raining_now_eta_min=rule.raining_now_eta_min,
        )
        # Every fold re-derives its run index from zero; shifting keeps the
        # merged trace's run ids distinct, so a fold seam reads as the
        # boundary it is rather than as a return to an earlier run.
        bump = offset
        out.extend(
            TracedRow(
                radar_ts=row.radar_ts,
                generated_at=row.generated_at,
                action=row.action,
                armed_before=row.armed_before,
                armed_after=row.armed_after,
                streak_before=row.streak_before,
                streak_after=row.streak_after,
                below_since_utc=row.below_since_utc,
                run_id=row.run_id + bump,
                skipped_reason=row.skipped_reason,
                p_decision=row.p_decision,
            )
            for row in trace
        )
        offset = max((row.run_id for row in out), default=-1) + 1
    out.sort(key=lambda row: _as_utc(row.radar_ts))
    return out


def build_population(
    *,
    rows: Sequence[Mapping[str, Any]],
    stations: Sequence[StationMeta],
    truth: GaugeTruth,
    rule: ReviewRule,
    onsets: Mapping[str, Sequence[datetime]] | None = None,
    known_until: Mapping[str, datetime] | None = None,
    dead: Sequence[str] = (),
    window_min: int = DEFAULT_WINDOW_MIN,
    allow_feature_gap: bool = False,
    min_known_slots: int = DEFAULT_MIN_KNOWN_SLOTS,
    design_leads: Sequence[int] = DEFAULT_PRODUCT_LEADS_MIN,
    neighbour_radius_km: float = NEIGHBOUR_RADIUS_KM,
    row_sources: Mapping[tuple[Any, str], str] | None = None,
    corpus: Mapping[str, Any] | None = None,
    log: Callable[[str], None] | None = None,
) -> Population:
    """Decision rows + gauge truth + a rule → every gradeable event.

    The pure half of :func:`load_population`: no parquet, no archive, no
    network, so the fixture mode (:func:`synthetic_population`) and the
    real build are the same code with different inputs. ``rows`` are
    decision rows in the shared schema; ``truth`` is the gauge archive
    reduced to slot grids, onsets and horizons.

    ``pending`` is dropped entirely — DMI backfills late reports, so the
    label can still move and a human reviewing it learns nothing.
    ``uncovered`` is kept, labelled ``control``, because the coverage rule
    removes it from POD's denominator and nothing else ever audits that.
    """
    rule.validate()
    say = log or (lambda _message: None)
    by_id = {station.station_id: station for station in stations}
    prepared, fallback_keys = _prepare_rows(rows, rule)

    day_counts, gap_days = feature_gap_days(prepared)
    gap_set = set(gap_days)

    tracks, frames = build_tracks(
        prepared, [rule.lead_min],
        coverage_gap_min=rule.coverage_gap_min,
        column_for=lambda _lead: rule.column,
    )
    coverage = {
        station: coverage_runs(
            stamps,
            max_gap_min=rule.coverage_gap_min,
            extend_min=rule.lead_min + rule.tolerance_min,
        )
        for station, stamps in frames.items()
    }
    rows_by_key = {
        (row.get("radar_ts"), str(row.get("station_id"))): row for row in prepared
    }

    onsets = (
        {s: list(v) for s, v in truth.onsets.items() if s not in set(dead)}
        if onsets is None
        else {str(s): list(v) for s, v in onsets.items()}
    )
    known_until = (
        {s: v for s, v in truth.known_until.items() if s not in set(dead)}
        if known_until is None
        else {str(s): v for s, v in known_until.items()}
    )
    amounts = {
        (station, instant): mm
        for station, pairs in truth.onsets_for(
            rule.dry_min, onset_min_mm=rule.onset_min_mm,
        ).items()
        for instant, mm in pairs
    }
    suspect = _suspect_months(truth, min_known_slots=min_known_slots)

    traces: dict[str, list[TracedRow]] = {}
    radar: dict[str, RadarSlots] = {}
    neighbours = {
        station.station_id: neighbours_of(
            station, stations, radius_km=neighbour_radius_km,
        )
        for station in stations
    }

    excluded: dict[str, int] = {
        "pending_warnings": 0,
        "pending_onsets": 0,
        "feature_gap_events": 0,
        "stations_without_gauge": 0,
        "stations_without_meta": 0,
        "duplicate_event_ids": 0,
    }
    records: list[EventRecord] = []
    seen_ids: set[str] = set()

    for station_id in sorted(tracks):
        track = tracks[station_id]
        meta = by_id.get(station_id)
        if meta is None:
            excluded["stations_without_meta"] += 1
            continue
        radar[station_id] = radar_slots_of(track, station_id)
        traces[station_id] = _trace_station(track, rule)
        if station_id not in known_until:
            # No gauge, or a dead one: every warning here would score as a
            # false alarm, which measures the archive rather than the rule.
            excluded["stations_without_gauge"] += 1
            continue
        trace = traces[station_id]
        warnings = traced_warnings(track, trace)
        result = score_warnings(
            warnings,
            onsets.get(station_id, ()),
            lead_min=rule.lead_min,
            tolerance_min=rule.tolerance_min,
            dry_min=rule.dry_min,
            onset_min_mm=rule.onset_min_mm,
            known_until=known_until.get(station_id),
            coverage=coverage.get(station_id, ()),
            min_useful_lead_min=rule.min_useful_lead_min,
        )
        by_generated = {
            _as_utc(row.generated_at): (record, row)
            for record, row in zip(track, trace)
        }
        for outcome in result.warnings:
            if outcome.outcome == "pending":
                excluded["pending_warnings"] += 1
                continue
            record = _warning_record(
                outcome=outcome,
                station=meta,
                rule=rule,
                track=track,
                trace=trace,
                by_generated=by_generated,
                truth=truth,
                amounts=amounts,
                known_until=known_until,
                radar=radar[station_id],
                neighbours=neighbours[meta.station_id],
                fallback_keys=fallback_keys,
                rows_by_key=rows_by_key,
                gap_days=gap_set,
                suspect=suspect,
                window_min=window_min,
            )
            records.append(record)
        for outcome in result.onsets:
            if outcome.outcome in ("pending", "hit"):
                if outcome.outcome == "pending":
                    excluded["pending_onsets"] += 1
                continue
            record = _onset_record(
                outcome=outcome,
                station=meta,
                rule=rule,
                track=track,
                trace=trace,
                by_generated=by_generated,
                truth=truth,
                amounts=amounts,
                known_until=known_until,
                radar=radar[station_id],
                neighbours=neighbours[meta.station_id],
                fallback_keys=fallback_keys,
                rows_by_key=rows_by_key,
                gap_days=gap_set,
                suspect=suspect,
                window_min=window_min,
            )
            records.append(record)

    kept: list[EventRecord] = []
    for record in sorted(records, key=lambda r: (r.anchor_utc, r.event_id)):
        if not allow_feature_gap and _as_utc(record.anchor_utc).date() in gap_set:
            excluded["feature_gap_events"] += 1
            continue
        if record.event_id in seen_ids:
            # Two events cannot share an id: annotations are keyed on it,
            # and a collision would merge two reviewers' judgements.
            excluded["duplicate_event_ids"] += 1
            continue
        seen_ids.add(record.event_id)
        kept.append(record)

    provenance = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "corpus": dict(corpus or {}),
        "rule": rule.document(),
        "truth": {
            "dry_min": int(rule.dry_min),
            "onset_min_mm": float(rule.onset_min_mm),
            "tolerance_min": int(rule.tolerance_min),
            "lead_min": int(rule.lead_min),
            "min_useful_lead_min": float(rule.min_useful_lead_min),
            "radar_disc_radius_m": float(RADAR_DISC_RADIUS_M),
            "radar_threshold_mm_h": float(RAIN_THRESHOLD_MM_H),
            "radar_source": "observed_mm_h in the decision rows",
            "neighbour_radius_km": float(neighbour_radius_km),
            "dead_gauges": sorted(str(s) for s in dead),
            "min_known_slots": int(min_known_slots),
            "known_until": {
                station: _iso(instant) for station, instant in sorted(known_until.items())
            },
            "suspect_gauge_months": sorted(
                f"{station}:{year:04d}-{month:02d}"
                for station, (year, month) in suspect
            ),
        },
        "features": {
            "probe_column": FEATURE_PROBE_COLUMN,
            "gap_share": FEATURE_GAP_SHARE,
            "gap_days": [day.isoformat() for day in gap_days],
            "allowed": bool(allow_feature_gap),
            "rows_by_day": {
                day.isoformat(): {"rows": total, "with_features": filled}
                for day, (total, filled) in day_counts.items()
            },
            "documentation": core_postprocess.feature_documentation(design_leads),
        },
        "window": {
            "decision_min": int(window_min),
            "frame_pad_min": int(DEFAULT_FRAME_PAD_MIN),
        },
        "caveats": _caveats(rule, gap_days, allow_feature_gap),
    }
    say(
        f"review population: {len(kept)} event(s) over {len(traces)} station(s); "
        f"{excluded['pending_warnings']} pending warning(s) and "
        f"{excluded['pending_onsets']} pending onset(s) dropped; "
        f"{len(gap_days)} feature-gap day(s) "
        f"({'kept' if allow_feature_gap else 'excluded'})"
    )
    return Population(
        records=tuple(kept),
        rule=rule,
        stations=by_id,
        neighbours=neighbours,
        rows=rows_by_key,
        row_sources=dict(row_sources or {}),
        tracks=tracks,
        traces=traces,
        coverage=coverage,
        truth=truth,
        onsets=onsets,
        onset_amounts=amounts,
        known_until=known_until,
        radar=radar,
        dead_gauges=tuple(sorted(str(s) for s in dead)),
        suspect_months=suspect,
        curve_fallback_keys=fallback_keys,
        feature_gap_dates=tuple(gap_days),
        day_feature_counts=day_counts,
        leads=tuple(sorted({int(rule.lead_min), *[int(x) for x in design_leads]})),
        design_leads=tuple(int(x) for x in design_leads),
        window_min=int(window_min),
        provenance=provenance,
        excluded=excluded,
    )


def _caveats(
    rule: ReviewRule, gap_days: Sequence[date], allow_feature_gap: bool,
) -> list[str]:
    """The sentences that must ride with the bundle, not only in a plan."""
    out = [
        "The radar verdict comes from the same instrument the forecast was "
        "made from, so agreement is a consistency check and not an "
        "independent opinion.",
        "The composite is column-max reflectivity, which biases the rate "
        "high: virga and bright band read as rain the gauge never sees.",
        "A wet neighbour says rain existed in the area, not at this gauge.",
        "An onset instant is a slot END, so the first drop fell in the "
        "preceding 10 minutes and every measured lead is biased that far "
        "negative.",
        "The replay resets the engine at every coverage-run boundary, which "
        "hands out a re-arm the live service never had; those rows carry "
        "run_boundary_rearm.",
    ]
    if not rule.held_out:
        out.append(
            "The threshold table was fitted on these same months: the "
            "thresholds are in-sample (held_out=false)."
        )
    else:
        out.append(
            "Leave-one-month-out thresholds: each month is replayed as a "
            "self-contained slice, so every month boundary is a seam where "
            "the state machine starts armed."
        )
    if gap_days:
        out.append(
            f"{len(gap_days)} day(s) lack Phase-H features on more than "
            f"{FEATURE_GAP_SHARE:.0%} of their rows and are "
            + ("INCLUDED, flagged feature_gap." if allow_feature_gap else "excluded.")
        )
    return out


def _window_flags(
    *,
    station: StationMeta,
    rule: ReviewRule,
    track: Sequence[tuple],
    trace: Sequence[TracedRow],
    anchor: datetime,
    window_min: int,
    rows_by_key: Mapping[tuple[Any, str], Mapping[str, Any]],
    gap_days: set[date],
    suspect: frozenset[tuple[str, tuple[int, int]]],
    known_until: Mapping[str, datetime],
    verdict: VerdictWindow,
) -> tuple[str, ...]:
    """Every reason this event could be misread if nobody said so."""
    anchor = _as_utc(anchor)
    span = timedelta(minutes=window_min)
    start, end = anchor - span, anchor + span
    flags: set[str] = set()

    inside = [
        (record, row)
        for record, row in zip(track, trace)
        if start <= _as_utc(record[_RADAR_TS]) <= end
    ]
    for record, row in inside:
        key = (record[_RADAR_TS], station.station_id)
        stored = rows_by_key.get(key, {})
        if stored.get(FEATURE_PROBE_COLUMN) is None:
            flags.add("feature_gap")
        if row.skipped_reason == SKIP_NO_PROBABILITY:
            flags.add("no_probability")
    if _as_utc(anchor).date() in gap_days:
        flags.add("feature_gap")

    # The window's own edges count. A window whose rows stop forty minutes
    # in — or that has none at all — has a coverage gap in it just as
    # surely as one with a hole in the middle, and only the edges can say
    # so: "nothing fired" would otherwise read as "nothing was over
    # threshold".
    stamps = [_as_utc(record[_RADAR_TS]) for record, _row in inside]
    edges = [start, *stamps, end]
    for previous, current in zip(edges, edges[1:]):
        if _minutes(current - previous) > rule.coverage_gap_min:
            flags.add("coverage_gap")
    if any(start <= stamp <= end for stamp in run_boundary_rearms(trace)):
        flags.add("run_boundary_rearm")

    if (station.station_id, (anchor.year, anchor.month)) in suspect:
        flags.add("suspect_gauge_month")

    horizon = known_until.get(station.station_id)
    if horizon is not None:
        edge = _as_utc(horizon) - timedelta(minutes=rule.tolerance_min)
        if _as_utc(verdict.to_utc) > edge:
            flags.add("near_known_until")
    return tuple(sorted(flags))


def _peak_probability(
    track: Sequence[tuple],
    trace: Sequence[TracedRow],
    start: datetime,
    end: datetime,
) -> tuple[float | None, datetime | None]:
    """The highest decision probability in ``(start, end]`` and its frame.

    What an onset-side event's ``p_decision`` is: a miss has no warning to
    read a probability off, and the question a reviewer asks about it is
    "how close did it come?" — ``miss_below_threshold`` against
    ``miss_threshold_marginal``. ``None`` where no row in the window
    carried a probability at all, which is ``miss_no_probability`` and is
    emphatically not the same thing as a low one.
    """
    start, end = _as_utc(start), _as_utc(end)
    best: float | None = None
    best_ts: datetime | None = None
    for record, row in zip(track, trace):
        stamp = _as_utc(record[_GENERATED])
        if stamp <= start or stamp > end:
            continue
        if row.p_decision is None:
            continue
        if best is None or row.p_decision > best:
            best = float(row.p_decision)
            best_ts = _as_utc(record[_RADAR_TS])
    return best, best_ts


def _probability_source(
    rule: ReviewRule,
    station_id: str,
    radar_ts: Any,
    fallback_keys: frozenset[tuple[Any, str]],
) -> str:
    """Which scale a row's decision probability was on, after the fallback.

    With no row to point at — an onset-side event whose window carried no
    probability at all — the answer is the scale the rule WOULD have
    judged it on, which is what ``p_decision: null`` beside it already
    says has not happened.
    """
    if rule.probability != served_rule_module.PROBABILITY_POSTPROCESS:
        return served_rule_module.PROBABILITY_CURVE
    if (radar_ts, station_id) in fallback_keys:
        return served_rule_module.PROBABILITY_CURVE
    return served_rule_module.PROBABILITY_POSTPROCESS


def _verdicts(
    *,
    station: StationMeta,
    truth: GaugeTruth,
    radar: RadarSlots,
    neighbours: Sequence[Neighbour],
    window: VerdictWindow,
) -> tuple[bool | None, bool | None, bool | None, int]:
    """``(gauge, radar, neighbour, neighbours that reported)`` over a window."""
    series = truth.series.get(station.station_id)
    gauge = gauge_wet_between(series, window.from_utc, window.to_utc)
    radar_wet = radar.wet_between(window.from_utc, window.to_utc)
    neighbour_wet: bool | None = None
    known = 0
    for neighbour in neighbours:
        other = truth.series.get(neighbour.station_id)
        verdict = gauge_wet_between(other, window.from_utc, window.to_utc)
        if verdict is None:
            continue
        known += 1
        neighbour_wet = bool(neighbour_wet) or verdict
    return gauge, radar_wet, (neighbour_wet if known else None), known


def _warning_record(
    *,
    outcome,
    station: StationMeta,
    rule: ReviewRule,
    track: Sequence[tuple],
    trace: Sequence[TracedRow],
    by_generated: Mapping[datetime, tuple[tuple, TracedRow]],
    truth: GaugeTruth,
    amounts: Mapping[tuple[str, datetime], float],
    known_until: Mapping[str, datetime],
    radar: RadarSlots,
    neighbours: Sequence[Neighbour],
    fallback_keys: frozenset[tuple[Any, str]],
    rows_by_key: Mapping[tuple[Any, str], Mapping[str, Any]],
    gap_days: set[date],
    suspect: frozenset[tuple[str, tuple[int, int]]],
    window_min: int,
) -> EventRecord:
    """One :class:`WarningOutcome` as an event. Anchor: ``sent_utc``."""
    sent = _as_utc(outcome.sent_utc)
    pair = by_generated.get(sent)
    record_row, traced = pair if pair is not None else (None, None)
    radar_ts = None if record_row is None else record_row[_RADAR_TS]
    intensity = None if record_row is None else record_row[_INTENSITY]
    window = warning_window(
        sent, lead_min=rule.lead_min, tolerance_min=rule.tolerance_min,
    )
    gauge, radar_wet, neighbour_wet, n_known = _verdicts(
        station=station, truth=truth, radar=radar,
        neighbours=neighbours, window=window,
    )
    onset = None if outcome.onset_utc is None else _as_utc(outcome.onset_utc)
    arm = arm_state_at(trace, sent, rearm_after_min=rule.rearm_after_min)
    event_class = outcome.outcome
    return EventRecord(
        event_id=event_id(event_class, station.station_id, sent),
        event_class=event_class,
        station_id=station.station_id,
        anchor_utc=sent,
        sent_utc=sent,
        onset_utc=onset,
        eta_min=None if outcome.eta_min is None else float(outcome.eta_min),
        p_decision=None if traced is None else traced.p_decision,
        p_decision_source=_probability_source(
            rule, station.station_id, radar_ts, fallback_keys,
        ),
        threshold_pct=rule.threshold_for(sent),
        lead_error_min=outcome.lead_error_min,
        season=season_of(sent),
        region=station.region,
        intensity_mm_h=None if intensity is None else float(intensity),
        intensity_band=intensity_band(intensity),
        # The prediction, because a false alarm has no onset to read.
        intensity_band_source="forecast_intensity_mm_h",
        onset_two_slot_mm=(
            None if onset is None
            else amounts.get((station.station_id, onset))
        ),
        dual_truth=dual_truth_of(gauge, radar_wet),
        gauge_wet_in_window=gauge,
        radar_wet_in_window=radar_wet,
        neighbour_wet_in_window=neighbour_wet,
        neighbour_n_known=n_known,
        arm_state_at_anchor=arm.label,
        minutes_to_rearm_at_anchor=arm.minutes_to_rearm,
        flags=_window_flags(
            station=station, rule=rule, track=track, trace=trace, anchor=sent,
            window_min=window_min, rows_by_key=rows_by_key, gap_days=gap_days,
            suspect=suspect, known_until=known_until, verdict=window,
        ),
        control=group_of(event_class) in CONTROL_GROUPS,
        window_used=window,
    )


def _onset_record(
    *,
    outcome,
    station: StationMeta,
    rule: ReviewRule,
    track: Sequence[tuple],
    trace: Sequence[TracedRow],
    by_generated: Mapping[datetime, tuple[tuple, TracedRow]],
    truth: GaugeTruth,
    amounts: Mapping[tuple[str, datetime], float],
    known_until: Mapping[str, datetime],
    radar: RadarSlots,
    neighbours: Sequence[Neighbour],
    fallback_keys: frozenset[tuple[Any, str]],
    rows_by_key: Mapping[tuple[Any, str], Mapping[str, Any]],
    gap_days: set[date],
    suspect: frozenset[tuple[str, tuple[int, int]]],
    window_min: int,
) -> EventRecord:
    """One :class:`OnsetOutcome` as an event. Anchor: ``onset_utc``."""
    onset = _as_utc(outcome.onset_utc)
    sent = None if outcome.sent_utc is None else _as_utc(outcome.sent_utc)
    window = onset_window(onset, tolerance_min=rule.tolerance_min)
    gauge, radar_wet, neighbour_wet, n_known = _verdicts(
        station=station, truth=truth, radar=radar,
        neighbours=neighbours, window=window,
    )
    claim = None if sent is None else by_generated.get(sent)
    claim_row, claim_trace = claim if claim is not None else (None, None)
    peak, peak_ts = _peak_probability(
        track, trace, onset - timedelta(minutes=window_min), onset,
    )
    if claim_trace is not None:
        probability = claim_trace.p_decision
        source_ts = None if claim_row is None else claim_row[_RADAR_TS]
        eta = None if claim_row is None else claim_row[_ETA]
        intensity = None if claim_row is None else claim_row[_INTENSITY]
    else:
        probability = peak
        source_ts = peak_ts
        eta = None
        intensity = None
    depth = amounts.get((station.station_id, onset))
    arm = arm_state_at(trace, onset, rearm_after_min=rule.rearm_after_min)
    event_class = outcome.outcome
    return EventRecord(
        event_id=event_id(event_class, station.station_id, onset),
        event_class=event_class,
        station_id=station.station_id,
        anchor_utc=onset,
        sent_utc=sent,
        onset_utc=onset,
        eta_min=None if eta is None else float(eta),
        p_decision=probability,
        p_decision_source=_probability_source(
            rule, station.station_id, source_ts, fallback_keys,
        ),
        threshold_pct=rule.threshold_for(onset),
        lead_error_min=outcome.lead_error_min,
        season=season_of(onset),
        region=station.region,
        intensity_mm_h=None if intensity is None else float(intensity),
        # The onset's own depth, spread across its two slots: a miss has no
        # prediction to band, and banding it on one anyway would stratify
        # the misses by a number the rule never produced.
        intensity_band=intensity_band(two_slot_rate_mm_h(depth)),
        intensity_band_source="onset_two_slot_mm",
        onset_two_slot_mm=None if depth is None else float(depth),
        dual_truth=dual_truth_of(gauge, radar_wet),
        gauge_wet_in_window=gauge,
        radar_wet_in_window=radar_wet,
        neighbour_wet_in_window=neighbour_wet,
        neighbour_n_known=n_known,
        arm_state_at_anchor=arm.label,
        minutes_to_rearm_at_anchor=arm.minutes_to_rearm,
        flags=_window_flags(
            station=station, rule=rule, track=track, trace=trace, anchor=onset,
            window_min=window_min, rows_by_key=rows_by_key, gap_days=gap_days,
            suspect=suspect, known_until=known_until, verdict=window,
        ),
        control=group_of(event_class) in CONTROL_GROUPS,
        window_used=window,
    )


def load_population(
    *,
    decisions_dirs: Sequence[Path],
    decisions_labels: Sequence[str] | None = None,
    corpus_dir: Path,
    stations: Sequence[StationMeta],
    rule: ReviewRule,
    window: tuple[datetime, datetime],
    window_min: int = DEFAULT_WINDOW_MIN,
    allow_feature_gap: bool = False,
    min_known_slots: int = DEFAULT_MIN_KNOWN_SLOTS,
    design_leads: Sequence[int] = DEFAULT_PRODUCT_LEADS_MIN,
    postprocess_model: Path | None = None,
    neighbour_radius_km: float = NEIGHBOUR_RADIUS_KM,
    log: Callable[[str], None] | None = None,
) -> Population:
    """Read the corpus and build the population under ``rule``.

    ``decisions_dirs`` are passed to
    :func:`~dmi_nowcast_sidecar.threshold_sweep.load_decisions` **in
    caller order**, and that function lets the LAST directory win a
    ``(radar_ts, station_id)`` tie. So the caller orders them LEAST
    AUTHORITATIVE FIRST — for this tool the live tree first and the
    out-of-fold replay tree last, the opposite of the quality page's
    order, because a live row from the feature-gap window carries no
    ``p_post`` and would otherwise beat a replay row that does. The
    effective order and the number of rows that lost a tie are recorded in
    the returned ``provenance["corpus"]`` for the manifest.

    ``postprocess_model`` fills ``p_post_<lead>`` from the feature columns
    the way
    :class:`~dmi_nowcast_sidecar.postprocess_fit.ProbabilityFiller` does
    for the quality page. It is OFF by default and should stay off: the
    served model is fitted on every month, so filling with it scores the
    rule on rows the model already saw. The honest input is a
    ``fit_postprocess --write-back`` tree whose ``p_post`` is out of fold,
    declared through ``rule.probability_provenance``.
    """
    say = log or (lambda _message: None)
    dirs = [Path(d) for d in decisions_dirs]
    labels = (
        [str(label) for label in decisions_labels]
        if decisions_labels is not None
        else [d.name for d in dirs]
    )
    if len(labels) != len(dirs):
        raise ValueError(
            f"{len(labels)} label(s) for {len(dirs)} decision directory(ies)"
        )
    column = rule.column
    curve = rule.curve_column
    filler = _filler(rule, postprocess_model, design_leads, say)
    rows, leads, counts = load_decisions(
        dirs,
        leads_min=(rule.lead_min,),
        extra_columns=(
            tuple(
                name for name in (
                    column,
                    *core_postprocess.feature_only_columns(design_leads),
                )
                if name != curve
            )
        ),
        derive=filler,
        log=log,
    )
    say(
        f"review: {counts['rows']} row(s) from {counts['files']} file(s), "
        f"{counts['duplicates']} duplicate key(s) resolved in favour of the "
        "later directory"
    )
    row_sources = _row_sources(dirs, labels)
    station_ids = [station.station_id for station in stations]
    truth, onsets, known_until, dead = gauge_truth_for(
        Path(corpus_dir), station_ids, window,
        dry_min=rule.dry_min, onset_min_mm=rule.onset_min_mm,
        min_known_slots=min_known_slots, log=log,
    )
    corpus = {
        "decisions_dirs": [str(d) for d in dirs],
        "decisions_labels": labels,
        "decisions_precedence": (
            "caller order, least authoritative first; the LAST directory "
            "wins a (radar_ts, station_id) tie"
        ),
        "corpus_dir": str(corpus_dir),
        "files": int(counts.get("files", 0)),
        "files_skipped": int(counts.get("skipped", 0)),
        "rows_read": int(counts.get("rows", 0)),
        "rows_kept": len(rows),
        "duplicates_dropped": int(counts.get("duplicates", 0)),
        "leads": [int(lead) for lead in leads],
        "window_from_utc": _iso(window[0]),
        "window_to_utc": _iso(window[1]),
    }
    if filler is not None:
        corpus["probability_fill"] = {
            key: int(value) for key, value in filler.counts.items()
        }
    return build_population(
        rows=rows,
        stations=stations,
        truth=truth,
        rule=rule,
        onsets=onsets,
        known_until=known_until,
        dead=dead,
        window_min=window_min,
        allow_feature_gap=allow_feature_gap,
        min_known_slots=min_known_slots,
        design_leads=design_leads,
        neighbour_radius_km=neighbour_radius_km,
        row_sources=row_sources,
        corpus=corpus,
        log=log,
    )


def _row_sources(
    dirs: Sequence[Path], labels: Sequence[str],
) -> dict[tuple[Any, str], str]:
    """``(radar_ts, station_id) -> the tree the winning row came from``.

    Two columns off every decision parquet, in the same order
    ``load_decisions`` reads them, with the LAST directory winning exactly
    as it does there. It costs one cheap pass over two columns and buys a
    per-row answer to the question the feature gap makes unavoidable: was
    this event judged on a replay row that carries features, or on a live
    row from the gap that fell back to the curve? A reviewer looking at a
    surprising decision needs to know which tree it came out of.
    """
    import pyarrow.parquet as pq

    from .threshold_sweep import decision_parquets

    out: dict[tuple[Any, str], str] = {}
    for directory, label in zip(dirs, labels):
        for path in decision_parquets(directory):
            try:
                table = pq.read_table(path, columns=["radar_ts", "station_id"])
            except Exception:  # noqa: BLE001 — load_decisions skips it too
                continue
            for stamp, station in zip(
                table.column("radar_ts").to_pylist(),
                table.column("station_id").to_pylist(),
            ):
                out[(stamp, str(station))] = label
    return out


def _filler(
    rule: ReviewRule,
    model_path: Path | None,
    design_leads: Sequence[int],
    say: Callable[[str], None],
):
    """A ``ProbabilityFiller`` for this run, or ``None``.

    Built exactly as ``served_rule._filler`` builds it — rows it cannot
    fill are KEPT, because the engine's per-row fallback has an answer for
    them and dropping them would silence warnings the service would have
    sent.
    """
    if (
        rule.probability != served_rule_module.PROBABILITY_POSTPROCESS
        or model_path is None
    ):
        return None
    from dmi_nowcast_core.postprocess import PostprocessModel

    from .postprocess_fit import ProbabilityFiller

    model = None
    try:
        model = PostprocessModel.loads(Path(model_path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — every way a file can be junk
        say(
            f"review: cannot read {model_path} ({type(exc).__name__}: {exc}); "
            "using the probabilities the rows already carry"
        )
    return ProbabilityFiller(
        model,
        (int(rule.lead_min),),
        tuple(int(lead) for lead in design_leads),
        lambda _lead: rule.column,
        drop_unfilled=False,
    )


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleCell:
    """One (class, season, region, band) cell: what it had, what was drawn."""

    stratum: str
    keys: tuple[str, ...]
    group: str
    population: int
    drawn: int

    def document(self) -> dict:
        return {
            "stratum": self.stratum,
            "keys": list(self.keys),
            "group": self.group,
            "population": self.population,
            "drawn": self.drawn,
        }


@dataclass(frozen=True)
class Sample:
    """The drawn events, the allocation behind them, and the population's hash."""

    records: tuple[EventRecord, ...]
    cells: tuple[SampleCell, ...]
    population_hash: str
    seed: int
    strata_keys: tuple[str, ...]
    floor_per_cell: int
    targets: Mapping[str, int]
    drawn: Mapping[str, int]
    population: Mapping[str, int]
    collapsed_late_pairs: int

    def document(self) -> dict:
        """The manifest's ``sampling`` block."""
        return {
            "seed": self.seed,
            "strata": list(self.strata_keys),
            "floor_per_cell": self.floor_per_cell,
            "targets": dict(self.targets),
            "drawn": dict(self.drawn),
            "population": dict(self.population),
            "total_drawn": len(self.records),
            "population_hash": self.population_hash,
            "collapsed_late_pairs": self.collapsed_late_pairs,
            "control_groups": sorted(CONTROL_GROUPS),
            "cells": [cell.document() for cell in self.cells],
        }


def population_hash(records: Iterable[EventRecord]) -> str:
    """sha256 over the sorted ``station|anchor|class`` lines.

    The identity of the POPULATION, not of the draw: a second bundle can
    prove it sampled the same events by quoting the same hash, and prove
    it did not when the corpus has grown underneath it. Deliberately blind
    to the rule and to the sample — those are separate manifest fields, and
    a hash that moved when the seed changed could not answer the question
    it exists for.
    """
    lines = sorted(
        f"{record.station_id}|{_iso(record.anchor_utc)}|{record.event_class}"
        for record in records
    )
    digest = hashlib.sha256()
    digest.update("\n".join(lines).encode("utf-8"))
    return digest.hexdigest()


def _allocate(
    sizes: Mapping[tuple[str, ...], int], total: int, floor: int,
) -> dict[tuple[str, ...], int]:
    """Proportional allocation with a floor, settled by largest remainder.

    Deterministic to the byte: cells are visited in sorted key order, the
    floor is honoured until the target runs out, the remainder is
    apportioned by the classic Hamilton rule, and a tie in the remainder
    goes to the lower key rather than to whatever the hash table offered
    first. Capacity is respected — a cell can never be asked for more
    events than it has — and whatever a capped cell gives back is
    reallocated in the same way until the target is met or the population
    is exhausted.
    """
    keys = sorted(sizes)
    take = {key: 0 for key in keys}
    remaining = int(total)
    for key in keys:
        if remaining <= 0:
            break
        give = min(int(floor), sizes[key], remaining)
        take[key] += give
        remaining -= give
    while remaining > 0:
        capacity = {key: sizes[key] - take[key] for key in keys}
        free = sum(capacity.values())
        if free <= 0:
            break
        share = min(remaining, free)
        quota = {key: share * capacity[key] / free for key in keys}
        base = {key: min(int(math.floor(q)), capacity[key]) for key, q in quota.items()}
        handed = sum(base.values())
        order = sorted(
            keys,
            key=lambda k: (-(quota[k] - math.floor(quota[k])), k),
        )
        i = 0
        while handed < share and i < len(order):
            key = order[i]
            if base[key] < capacity[key]:
                base[key] += 1
                handed += 1
            i += 1
        if handed == 0:
            # Nothing could be handed out under the proportional rule (a
            # single event left over a hundred cells): give it to the
            # largest cell with room, in key order, so the loop ends.
            for key in sorted(keys, key=lambda k: (-capacity[k], k)):
                if capacity[key] > 0:
                    base[key] += 1
                    handed += 1
                    break
        if handed == 0:  # pragma: no cover — free > 0 guarantees a hand-out
            break
        for key in keys:
            take[key] += base[key]
        remaining -= handed
    return take


def stratify(
    records: Sequence[EventRecord],
    *,
    seed: int,
    target: int | None = None,
    strata_keys: Sequence[str] = STRATA_KEYS,
    floor_per_cell: int = DEFAULT_FLOOR_PER_CELL,
    class_targets: Mapping[str, int] | None = None,
) -> Sample:
    """Draw a seeded, stratified, byte-reproducible sample.

    Same records and same seed, same event ids in the same order — which
    is what lets a rebuild be compared with a review already half done.
    Every random choice comes from one :class:`random.Random` seeded here;
    the global RNG is never touched, and nothing iterates a set or a dict
    without sorting it first.

    Allocation is per GROUP (:data:`SAMPLE_GROUPS`): proportional to each
    cell's population, with :data:`DEFAULT_FLOOR_PER_CELL` per non-empty
    cell first so a rare stratum exists at all, then largest remainder so
    the group's target is hit exactly. A group with fewer events than its
    target contributes all of them.

    ``late`` and ``miss_late`` are collapsed to one candidate per episode
    (the warning side, which carries the ETA the reviewer is judging)
    because ``score_warnings`` emits them in pairs and showing a reviewer
    both would count one piece of rain twice.

    The drawn list is SHUFFLED with the same RNG, so the control groups
    (``hit`` and ``uncovered``, marked ``control=True``) are interleaved
    with the failures. The UI hides the class behind a reveal toggle, and
    a reviewer who can infer the class from an event's position in the
    list is not blind.
    """
    keys = tuple(str(k) for k in strata_keys)
    known_keys = set(STRATA_KEYS)
    unknown_keys = [key for key in keys if key not in known_keys]
    if unknown_keys:
        raise ValueError(
            f"unknown stratum key(s) {unknown_keys}; known: {sorted(known_keys)}"
        )
    targets = dict(DEFAULT_CLASS_TARGETS if class_targets is None else class_targets)
    unknown = sorted(set(targets) - set(SAMPLE_GROUPS))
    if unknown:
        raise ValueError(f"unknown sampling group(s): {unknown}")
    if target is not None and class_targets is not None:
        if int(target) != sum(targets.values()):
            raise ValueError(
                f"target {target} does not match the class targets' sum "
                f"{sum(targets.values())} — the two are never mixed silently"
            )
    if target is not None and class_targets is None:
        targets = _scale_targets(targets, int(target))

    rng = random.Random(int(seed))
    pool: dict[str, list[EventRecord]] = {name: [] for name in SAMPLE_GROUPS}
    collapsed = 0
    # A ``late`` warning and its ``miss_late`` onset are one episode. The
    # warning side is kept — it carries the ETA whose shortfall is what
    # the reviewer is judging — and the onset side is dropped only where
    # its partner is actually present, so an unpaired record (which
    # ``score_warnings`` should never produce, but a hand-built population
    # can) is still reviewable rather than silently gone.
    preferred: dict[str, set[tuple]] = {
        name: {
            record.pair_key
            for record in records
            if record.event_class == classes[0]
        }
        for name, classes in SAMPLE_GROUPS.items()
        if len(classes) > 1
    }
    for record in sorted(records, key=_sort_key):
        group = record.group
        if group in preferred and record.event_class != SAMPLE_GROUPS[group][0]:
            if record.pair_key in preferred[group]:
                collapsed += 1
                continue
        pool[group].append(record)

    cells: list[SampleCell] = []
    drawn: list[EventRecord] = []
    drawn_counts: dict[str, int] = {}
    population_counts: dict[str, int] = {}
    for group in sorted(SAMPLE_GROUPS):
        candidates = pool[group]
        population_counts[group] = len(candidates)
        want = min(int(targets.get(group, 0)), len(candidates))
        buckets: dict[tuple[str, ...], list[EventRecord]] = {}
        for record in candidates:
            cell = tuple(record.stratum_value(key) for key in keys)
            buckets.setdefault(cell, []).append(record)
        allocation = _allocate(
            {cell: len(rows) for cell, rows in buckets.items()}, want, floor_per_cell,
        )
        taken = 0
        for cell in sorted(buckets):
            rows = sorted(buckets[cell], key=_sort_key)
            n = allocation.get(cell, 0)
            picked = rng.sample(rows, n) if n else []
            picked.sort(key=_sort_key)
            drawn.extend(picked)
            taken += len(picked)
            cells.append(SampleCell(
                stratum="|".join(cell),
                keys=cell,
                group=group,
                population=len(rows),
                drawn=len(picked),
            ))
        drawn_counts[group] = taken

    drawn.sort(key=_sort_key)
    rng.shuffle(drawn)
    return Sample(
        records=tuple(drawn),
        cells=tuple(sorted(cells, key=lambda c: (c.group, c.stratum))),
        population_hash=population_hash(records),
        seed=int(seed),
        strata_keys=keys,
        floor_per_cell=int(floor_per_cell),
        targets=targets,
        drawn=drawn_counts,
        population=population_counts,
        collapsed_late_pairs=collapsed,
    )


def _sort_key(record: EventRecord) -> tuple:
    """The only ordering this module ever draws from: total and stable."""
    return (_iso(record.anchor_utc), record.event_class, record.station_id)


def _scale_targets(targets: Mapping[str, int], total: int) -> dict[str, int]:
    """Rescale the default allocation to a different total, exactly.

    Largest remainder again, so the parts add up to the whole and the
    shape of the default (a third false alarms, a third misses, a sixth
    control) survives a bundle that asks for 60 events instead of 300.
    """
    current = sum(targets.values())
    if current <= 0 or total <= 0:
        return {name: 0 for name in targets}
    exact = {name: total * value / current for name, value in targets.items()}
    out = {name: int(math.floor(value)) for name, value in exact.items()}
    short = total - sum(out.values())
    order = sorted(
        targets, key=lambda name: (-(exact[name] - math.floor(exact[name])), name),
    )
    for name in order[:short]:
        out[name] += 1
    return out


# ---------------------------------------------------------------------------
# The detail document
# ---------------------------------------------------------------------------


def build_event(
    record: EventRecord,
    population: Population,
    *,
    bundle_id: str | None = None,
    window_min: int | None = None,
    frame_pad_min: int = DEFAULT_FRAME_PAD_MIN,
    cadence_min: int = SLOT_MIN,
    gap_min: int = DECISION_GAP_MIN,
    station_pixel: Callable[[float, float], tuple[float, float]] | None = None,
    event_frames: Any | None = None,
    planned_stamps: Mapping[str, Any] | None = None,
    missing_stamps: Sequence[datetime] | None = None,
    strata_keys: Sequence[str] = STRATA_KEYS,
) -> dict:
    """The ``events/<event_id>.json`` document for one event.

    Everything a reviewer needs to judge the event without a second file:
    the index row repeated, the station and its neighbours, the three
    windows, every decision in the window with the engine's state beside
    it, the holes in those decisions, the ``prologue`` that explains an
    arm state set hours earlier, the gauge and radar slot series read
    three-valued, the dual-truth verdict with the window it was read over,
    the notifications, the frame list and the flags.

    The field names are the browser's
    (``frontend/src/lib/review/schema.ts``), which is the fourth consumer
    of ``review_schema``'s contract and the one that has to parse this.
    Where this builder cannot know a field — the product-grid position, a
    disc's pixel counts — it writes ``null`` rather than a plausible
    number, and the block says which source it did have.

    ``station_pixel`` maps ``(lat, lon)`` to a fractional product-grid
    ``(row, col)``; the grid belongs to the frame writer
    (:mod:`~dmi_nowcast_sidecar.review_frames`), so without one the
    station's ``grid`` carries explicit nulls. A guess there would put the
    marker in the wrong place on every map in the bundle.

    ``event_frames`` is that module's ``EventFrames`` for this event — its
    stamps and its two window edges — and when it is given, ITS edges are
    what the document names: the bundle's frame edge must be the edge
    frames were actually rendered from, or the reviewer scrubs to a stamp
    the bundle does not contain. Without one the edge is computed the way
    ``review_frames.frame_plan`` computes it (``window + pad + cadence``
    back from the anchor, because an anchor is almost never on the
    10-minute grid and flooring onto it can cost a whole cadence), and the
    stamps fall back to what the decision rows imply.

    ``planned_stamps`` is ``{stamp: PlannedStamp}`` from the same plan, for
    the product and the file names; ``missing_stamps`` is the writer's
    knowledge of which of them the archive could not supply. Both are
    optional and both are labelled in the output, because "the frames this
    event needs" and "the frames the bundle has" are different facts and
    must never wear one label.
    """
    rule = population.rule
    window_min = population.window_min if window_min is None else int(window_min)
    station = population.stations[record.station_id]
    track = population.tracks.get(record.station_id, ())
    trace = population.traces.get(record.station_id, ())
    anchor = _as_utc(record.anchor_utc)
    span = timedelta(minutes=window_min)
    start, end = anchor - span, anchor + span
    frames_from = anchor - timedelta(
        minutes=window_min + frame_pad_min + cadence_min,
    )
    frames_to = end
    if event_frames is not None:
        frames_from = _as_utc(event_frames.frames_from_utc)
        frames_to = _as_utc(event_frames.frames_to_utc)
    slot_from = frames_from
    slot_to = end + timedelta(minutes=rule.lead_min + rule.tolerance_min)
    horizon = population.known_until.get(record.station_id)
    verdict = record.window_used

    inside = [
        (index, rec, row)
        for index, (rec, row) in enumerate(zip(track, trace))
        if start <= _as_utc(rec[_RADAR_TS]) <= end
    ]
    decisions = [
        _decision_entry(
            record_row=rec,
            traced=row,
            previous=trace[index - 1] if index > 0 else None,
            population=population,
            station=station,
            track=track,
            trace=trace,
        )
        for index, rec, row in inside
    ]
    gauge_slots = population.truth.series.get(record.station_id)
    radar = population.radar.get(
        record.station_id, RadarSlots(record.station_id, ()),
    )
    frames = _frame_list(
        track, frames_from, frames_to, event_frames, planned_stamps,
        missing_stamps,
    )
    pixel = None if station_pixel is None else station_pixel(station.lat, station.lon)
    # The three-valued gauge verdict. The index row carries a bool — the
    # question the list filters on is "was the gauge wet" — and the detail
    # keeps the null a silent gauge actually earns.
    gauge_verdict = gauge_wet_between(gauge_slots, verdict.from_utc, verdict.to_utc)
    neighbours = _neighbour_block(record, population, verdict, slot_from, slot_to)

    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "event_id": record.event_id,
        "index": index_row(
            record, station,
            frames=len(frames),
            frames_missing=sum(1 for frame in frames if frame["present"] is False),
            strata_keys=strata_keys,
        ),
        "station": {
            "station_id": station.station_id,
            "name": station.name,
            "lat": station.lat,
            "lon": station.lon,
            "region": station.region,
            "station_radar_km": _station_radar_km(population, record.station_id),
            # Null, not a guess: the product grid is the frame writer's.
            "grid": {
                "row": None if pixel is None else pixel[0],
                "col": None if pixel is None else pixel[1],
            },
            "known_until_utc": _iso(horizon),
            "dead_gauge": record.station_id in population.dead_gauges,
            "neighbours": [
                neighbour.document()
                for neighbour in population.neighbours.get(record.station_id, ())
            ],
        },
        "window": {
            "anchor_utc": _iso(anchor),
            "from_utc": _iso(start),
            "to_utc": _iso(end),
            "decision_min": window_min,
            "frames_from_utc": _iso(frames_from),
            "frames_to_utc": _iso(frames_to),
            "frame_pad_min": int(frame_pad_min),
            "frame_cadence_min": int(cadence_min),
            "slot_from_utc": _iso(slot_from),
            "slot_to_utc": _iso(slot_to),
            "known_until_utc": _iso(horizon),
            "verdict": verdict.document(),
        },
        "decisions": decisions,
        "decision_gaps": _decision_gaps(
            [rec for _i, rec, _r in inside], start, end,
            gap_min=gap_min, coverage_gap_min=rule.coverage_gap_min,
        ),
        "prologue": _prologue(
            trace=trace, start=start, anchor=anchor, rule=rule,
        ),
        "gauge": _gauge_block(
            record, population, gauge_slots, slot_from, slot_to, verdict,
        ),
        "radar_disc": _radar_block(radar, slot_from, slot_to, verdict),
        "neighbours": neighbours,
        "dual_truth": {
            "class": record.dual_truth,
            "gauge_wet": gauge_verdict,
            "radar_wet": record.radar_wet_in_window,
            "neighbour_wet": record.neighbour_wet_in_window,
            "neighbour_n_known": record.neighbour_n_known,
            "window_used": {
                "kind": verdict.kind,
                "from_utc": _iso(verdict.from_utc),
                "to_utc": _iso(verdict.to_utc),
                "definition": verdict.rule,
            },
            "rule": (
                "gauge: a wet slot in the window; radar: the disc at or above "
                f"{RAIN_THRESHOLD_MM_H} mm/h in the window; neighbour: any "
                f"gauge within {NEIGHBOUR_RADIUS_KM:g} km wet in the window"
            ),
            "caveat": (
                "The radar verdict comes from the instrument that made the "
                "forecast; agreement is a consistency check, not a second "
                "opinion. A wet neighbour says rain existed in the area."
            ),
        },
        "notifications": _notifications(record, population, inside, rule),
        "frames": frames,
        "flags": list(record.flags),
        "builder_notes": _builder_notes(record, population, inside),
    }


def _station_radar_km(population: Population, station_id: str) -> float | None:
    """Kilometres to the nearest DMI radar, off any row that carries it.

    A per-station constant the replay already wrote as a feature
    (``product_pairs.nearest_radar_km``), and one a reviewer needs: the
    composite under-detects close to a radar and over-reads column max far
    from one, which is half of ``fa_clutter_or_bright_band`` and
    ``miss_radar_saw_nothing``. Null where no row carried it rather than
    recomputed here, so the bundle quotes the number the decision stood on.
    """
    for (_ts, station), row in population.rows.items():
        if station == station_id:
            value = row.get("station_radar_km")
            if value is not None:
                return float(value)
    return None


def _decision_entry(
    *,
    record_row: tuple,
    traced: TracedRow,
    previous: TracedRow | None,
    population: Population,
    station: StationMeta,
    track: Sequence[tuple],
    trace: Sequence[TracedRow],
) -> dict:
    """One frame in the window: the numbers, the state, and the stored row."""
    rule = population.rule
    radar_ts = _as_utc(record_row[_RADAR_TS])
    generated = _as_utc(record_row[_GENERATED])
    key = (record_row[_RADAR_TS], station.station_id)
    stored = population.rows.get(key, {})
    threshold = rule.threshold_for(generated)
    features = {
        name: stored.get(name)
        for name in core_postprocess.feature_only_columns(population.design_leads)
    }
    # ``season`` and ``hour_utc`` are derived from the instant rather than
    # read: ``feature_source_columns`` says so, and a row written before
    # the columns existed would otherwise read as featureless.
    features["season"] = season_of(generated)
    features["hour_utc"] = int(generated.hour)
    # generated_at IS radar_ts + frame_age, so the age is derived and the
    # stored column is kept beside it: a disagreement is a fact about the
    # archive and must be visible, not averaged away.
    frame_age = _minutes(generated - radar_ts)
    stored_age = stored.get("frame_age_min")
    eta = record_row[_ETA]
    return {
        "radar_ts_utc": _iso(radar_ts),
        "generated_at_utc": _iso(generated),
        "frame_age_min": frame_age,
        "frame_age_source": (
            "feature"
            if stored_age is not None and abs(float(stored_age) - frame_age) < 1e-6
            else "derived"
        ),
        "frame_age_min_stored": None if stored_age is None else float(stored_age),
        "frame_ref": _stamp(radar_ts),
        "row_source": population.row_sources.get(key),
        "p_rain": {
            str(lead): stored.get(p_rain_column(lead))
            for lead in population.leads
        },
        "p_post": {
            str(lead): stored.get(core_postprocess.post_column(lead))
            for lead in population.leads
        },
        "p_decision": traced.p_decision,
        "p_decision_source": _probability_source(
            rule, station.station_id, record_row[_RADAR_TS],
            population.curve_fallback_keys,
        ),
        "p_decision_lead_min": int(rule.lead_min),
        "threshold_pct": threshold,
        # Null on a row the engine never evaluated: it was not under the
        # threshold, it was never compared with it.
        "over_threshold": (
            None if traced.p_decision is None
            else traced.p_decision >= threshold / 100
        ),
        "eta_min": eta,
        "eta_arrival_utc": (
            None if eta is None
            else _iso(generated + timedelta(minutes=float(eta)))
        ),
        "intensity_mm_h": record_row[_INTENSITY],
        "observed_mm_h": record_row[_OBSERVED],
        "forecast_now_mm_h": record_row[_FORECAST],
        "features": features,
        "features_present": stored.get(FEATURE_PROBE_COLUMN) is not None,
        "replay": {
            "action": traced.action,
            "skipped_reason": traced.skipped_reason,
            "armed_before": traced.armed_before,
            "armed_after": traced.armed_after,
            "streak_before": traced.streak_before,
            "streak_after": traced.streak_after,
            "below_since_utc": _iso(traced.below_since_utc),
            "run_id": traced.run_id,
            "run_boundary_rearm": bool(
                previous is not None
                and traced.run_id != previous.run_id
                and not previous.armed_after
                and traced.armed_before
            ),
        },
        "stored": {
            "action": stored.get("action"),
            "armed_after": stored.get("armed_after"),
            "streak_after": stored.get("streak_after"),
            "threshold_pct": stored.get("threshold_pct"),
            "p_rain_at_rule_lead": stored.get(p_rain_column(rule.lead_min)),
        },
        "latest_estimate": _latest_estimate(track, trace, radar_ts),
    }


def _stamp(ts: datetime) -> str:
    """``YYYYMMDDhhmm`` — the archive's and the bundle's frame name.

    Spelled out rather than imported from
    :mod:`~dmi_nowcast_sidecar.review_frames`, whose import pulls in h5py
    and pyproj: this module is also the fixture mode, and a frontend
    fixture must not need a radar stack. ``test_review.py`` pins the two
    spellings against each other.
    """
    return _as_utc(ts).strftime("%Y%m%d%H%M")


def _latest_estimate(
    track: Sequence[tuple],
    trace: Sequence[TracedRow],
    radar_ts: datetime,
) -> dict | None:
    """What the service was saying when this frame appeared.

    Plan §4.5: the picture and the words come from different cycles. A
    composite published at ``radar_ts`` is 13–24 minutes old by the time a
    cycle stands on it, so the numbers a user could see the moment that
    frame existed came from an EARLIER cycle — the newest row whose
    ``generated_at`` is at or before this frame's nominal time. Printing
    the row's own numbers there instead would show the reviewer rain the
    service could not yet have seen, and make every false alarm look
    better than it was.
    """
    radar_ts = _as_utc(radar_ts)
    best: tuple[tuple, TracedRow] | None = None
    for record, row in zip(track, trace):
        if _as_utc(record[_GENERATED]) <= radar_ts:
            best = (record, row)
        else:
            break
    if best is None:
        return None
    record, row = best
    return {
        "radar_ts_utc": _iso(record[_RADAR_TS]),
        "generated_at_utc": _iso(record[_GENERATED]),
        "age_min": _minutes(radar_ts - _as_utc(record[_GENERATED])),
        "p_decision": row.p_decision,
        "eta_min": record[_ETA],
        "intensity_mm_h": record[_INTENSITY],
        "observed_mm_h": record[_OBSERVED],
        "action": row.action,
    }


def _decision_gaps(
    rows: Sequence[tuple],
    start: datetime,
    end: datetime,
    *,
    gap_min: int,
    coverage_gap_min: int,
) -> list[dict]:
    """Holes in the decision sequence inside the window.

    Two sizes, reported as one list with a reason: a missed cycle (longer
    than the 10-minute cadence) is a hiccup the reviewer should see, and a
    gap longer than ``coverage_gap_min`` is where the coverage rule stops
    counting and the replay hands out a free re-arm. The window's own
    edges count: a window that simply has no rows before the anchor is a
    gap, not a silence, and leaving it out would let a reviewer read
    "nothing fired" as "nothing was over threshold".
    """
    start, end = _as_utc(start), _as_utc(end)

    def entry(a: datetime, b: datetime, edge: str | None) -> dict:
        minutes = _minutes(b - a)
        breaks = minutes > coverage_gap_min
        where = {
            "leading": "before the first decision in the window",
            "trailing": "after the last decision in the window",
            "whole_window": "no decision row in the window at all",
            None: "between two decisions",
        }[edge]
        return {
            "from_utc": _iso(a),
            "to_utc": _iso(b),
            "minutes": minutes,
            "coverage_break": breaks,
            "edge": edge,
            "reason": (
                f"{'coverage break' if breaks else 'missed cycle'} "
                f"({minutes:g} min, {where})"
            ),
        }

    stamps = [_as_utc(row[_RADAR_TS]) for row in rows]
    if not stamps:
        return [entry(start, end, "whole_window")]
    out: list[dict] = []
    if _minutes(stamps[0] - start) > gap_min:
        out.append(entry(start, stamps[0], "leading"))
    for previous, current in zip(stamps, stamps[1:]):
        if _minutes(current - previous) > gap_min:
            out.append(entry(previous, current, None))
    if _minutes(end - stamps[-1]) > gap_min:
        out.append(entry(stamps[-1], end, "trailing"))
    return out


def _prologue(
    *,
    trace: Sequence[TracedRow],
    start: datetime,
    anchor: datetime,
    rule: ReviewRule,
) -> dict:
    """The engine state entering the window — mandatory, never optional.

    ``evaluate`` disarms on every ``notify`` AND every ``already_raining``
    and re-arms only after 60 minutes of radar time below threshold, from
    a ``below_since_utc`` that any over-threshold row resets while
    disarmed. The disarming action is therefore usually OUTSIDE a ±90
    minute window, and a reviewer who cannot see it reads a station that
    inexplicably never fires — and blames the forecast for the
    hysteresis. So the last ``notify``, the last ``already_raining`` and
    the last five non-``none`` actions are reported however far back they
    lie, beside the arm state, the dry clock and the minutes still owed.

    One spelling per fact. This is the block a reviewer leans on to tell
    an unreachable miss from a real one, and two names for the same
    instant is how the two start disagreeing after an edit — so the keys
    here are exactly the ones ``frontend/src/lib/review/schema.ts``
    declares, and nothing is repeated in a second shape. Whether an action
    fell inside the window, and whether it is this event's own push, are
    both a comparison away from ``window.from_utc`` and ``index.sent_utc``
    in the same document; they are not stored again here.
    """
    start = _as_utc(start)
    anchor = _as_utc(anchor)
    entering = arm_state_at(trace, start, rearm_after_min=rule.rearm_after_min)
    at_anchor = arm_state_at(trace, anchor, rearm_after_min=rule.rearm_after_min)
    rearms = [_as_utc(stamp) for stamp in run_boundary_rearms(trace)]
    history = [
        {
            "generated_at_utc": _iso(row.generated_at),
            "action": row.action,
            "p_decision": row.p_decision,
        }
        for row in trace
        if row.action not in (None, "none")
        and _as_utc(row.generated_at) <= anchor
    ]
    last_of = {
        action: next(
            (
                entry["generated_at_utc"] for entry in reversed(history)
                if entry["action"] == action
            ),
            None,
        )
        for action in ("notify", "already_raining")
    }
    run_start = next(
        (
            _iso(row.radar_ts) for row in trace
            if entering.run_id is not None and row.run_id == entering.run_id
        ),
        None,
    )
    return {
        "run_id": entering.run_id,
        "run_start_utc": run_start,
        "armed_at_window_start": entering.armed,
        "streak_at_window_start": entering.streak,
        "below_since_utc": _iso(entering.below_since_utc),
        "minutes_to_rearm_at_window_start": entering.minutes_to_rearm,
        "last_notify_utc": last_of["notify"],
        "last_already_raining_utc": last_of["already_raining"],
        "recent_actions": history[-5:],
        "rearm_after_min": int(rule.rearm_after_min),
        # The same three questions asked at the OTHER end of the window.
        # Not a second spelling of the state at the window start: a
        # subscription can be disarmed entering the window and armed at
        # the anchor, and which of the two a miss sat under is the whole
        # question ``miss_disarmed_rearm`` asks.
        "at_anchor": {
            "armed": at_anchor.armed,
            "streak": at_anchor.streak,
            "minutes_to_rearm": at_anchor.minutes_to_rearm,
        },
        # Bounded on purpose: a station's whole trace can hold hundreds of
        # run boundaries over a season, and repeating them in 300 event
        # files would cost more than the frames do. These are the ones
        # that touch this window — the free re-arms a reviewer would
        # otherwise read as a bug in the tool.
        "run_boundary_rearms_utc": [
            _iso(stamp) for stamp in rearms if start <= stamp <= anchor
        ],
        "note": (
            "A disarmed subscription re-arms only after "
            f"{rule.rearm_after_min} minutes of radar time below threshold, "
            "and any over-threshold row resets that clock while disarmed. "
            "minutes_to_rearm is null when the clock is not running at all."
        ),
    }


def _gauge_block(
    record: EventRecord,
    population: Population,
    slots: StationSlots | None,
    slot_from: datetime,
    slot_to: datetime,
    verdict: VerdictWindow,
) -> dict:
    """The gauge's own record over the window, read three-valued."""
    rule = population.rule
    series = gauge_series(slots, slot_from, slot_to)
    anchor = _as_utc(record.anchor_utc)
    return {
        "station_id": record.station_id,
        "slot_min": SLOT_MIN,
        "wet_rule": "precip_past10min >= 0.1 mm OR precip_dur_past10min >= 1 min",
        "onset_rule": (
            f"first wet slot after {rule.dry_min} min known dry, delivering "
            f"{rule.onset_min_mm} mm over it and the slot after it"
        ),
        "known_until_utc": _iso(population.known_until.get(record.station_id)),
        "onset_utc": _iso(record.onset_utc),
        "onset_two_slot_mm": record.onset_two_slot_mm,
        "wet_slots_in_window": sum(
            1 for slot in series if slot["known"] and slot["wet"]
        ),
        "known_slots_in_window": sum(1 for slot in series if slot["known"]),
        "onsets": [
            {
                "onset_utc": _iso(instant),
                "two_slot_mm": population.onset_amounts.get(
                    (record.station_id, _as_utc(instant)),
                ),
                "in_window": (
                    verdict.from_utc <= _as_utc(instant) <= verdict.to_utc
                ),
                "is_event": _iso(instant) == _iso(record.onset_utc),
            }
            for instant in population.onsets.get(record.station_id, ())
            if slot_from <= _as_utc(instant) <= slot_to
        ],
        "suspect_month": (
            (record.station_id, (anchor.year, anchor.month))
            in population.suspect_months
        ),
        "slots": series,
    }


def _radar_block(
    radar: RadarSlots,
    slot_from: datetime,
    slot_to: datetime,
    verdict: VerdictWindow,
) -> dict:
    """The disc's word, and exactly how much of it the bundle has.

    ``series`` carries the p90 the decision rows stored — the number the
    service acted on — and nulls for the statistics only a composite can
    answer (``max``, ``mean``, the pixel counts). A builder that filled
    those from the p90 would invent a spread the disc never reported.
    """
    slots = radar.series(slot_from, slot_to)
    first_wet = next((slot["slot_end_utc"] for slot in slots if slot["wet"]), None)
    return {
        "disc_radius_m": float(RADAR_DISC_RADIUS_M),
        "statistic": "p90 over the disc, as the service samples it",
        "source_column": "observed_mm_h",
        "threshold_mm_h": float(RAIN_THRESHOLD_MM_H),
        "same_instrument_as_the_forecast": True,
        "series": [
            {
                "radar_ts_utc": slot["slot_end_utc"],
                "p90_mm_h": slot["mm_h"],
                "max_mm_h": None,
                "mean_mm_h": None,
                "n_pixels": None,
                "n_valid": None,
            }
            for slot in slots
        ],
        "slots": slots,
        "wet_in_window": radar.wet_between(verdict.from_utc, verdict.to_utc),
        "first_wet_utc": first_wet,
    }


def _neighbour_block(
    record: EventRecord,
    population: Population,
    verdict: VerdictWindow,
    slot_from: datetime,
    slot_to: datetime,
) -> dict:
    """Each neighbour's verdict over the event's own window.

    A wet neighbour says rain existed in the AREA, not at this gauge —
    which is the difference between ``fa_gauge_missed_it`` and
    ``fa_virga_or_aloft``, and the reason the radius and the counts travel
    with the answer instead of a bare boolean.
    """
    stations: list[dict] = []
    n_wet = 0
    n_known = 0
    for neighbour in population.neighbours.get(record.station_id, ()):
        series = population.truth.series.get(neighbour.station_id)
        wet = gauge_wet_between(series, verdict.from_utc, verdict.to_utc)
        slots = gauge_series(series, slot_from, slot_to)
        if wet is not None:
            n_known += 1
            n_wet += int(wet)
        stations.append({
            **neighbour.document(),
            "wet_in_window": wet,
            "first_wet_utc": next(
                (
                    slot["slot_end_utc"] for slot in slots
                    if slot["known"] and slot["wet"]
                    and verdict.from_utc
                    <= datetime.fromisoformat(slot["slot_end_utc"])
                    <= verdict.to_utc
                ),
                None,
            ),
            "known_slots": gauge_known_between(
                series, verdict.from_utc, verdict.to_utc,
            ),
            "onsets_in_window_utc": [
                _iso(instant)
                for instant in population.onsets.get(neighbour.station_id, ())
                if verdict.from_utc <= _as_utc(instant) <= verdict.to_utc
            ],
            "slots": slots,
        })
    return {
        "radius_km": float(NEIGHBOUR_RADIUS_KM),
        "any_wet_in_window": record.neighbour_wet_in_window,
        "n_known": n_known,
        "n_wet": n_wet,
        "caveat": (
            "A wet neighbour says rain existed in the area, not that it "
            "rained at this station."
        ),
        "stations": stations,
    }


def _notifications(
    record: EventRecord,
    population: Population,
    inside: Sequence[tuple[int, tuple, TracedRow]],
    rule: ReviewRule,
) -> list[dict]:
    """Every non-silent action in the window, replayed and as stored.

    Both, and labelled, because they are two different rules: the tree's
    stored ``action`` was decided at a fixed 40 % on the curve, and the
    replayed one at the served threshold on the probability the service
    decides with. Showing only the replay would hide that the archive
    disagrees; showing only the stored one is the mistake this whole
    module exists to correct.
    """
    out: list[dict] = []
    for _index, rec, row in inside:
        generated = _as_utc(rec[_GENERATED])
        eta = rec[_ETA]
        arrival = (
            None if eta is None
            else _iso(generated + timedelta(minutes=float(eta)))
        )
        if row.action not in (None, "none"):
            out.append({
                "kind": "replayed",
                "generated_at_utc": _iso(generated),
                "radar_ts_utc": _iso(rec[_RADAR_TS]),
                "action": row.action,
                "p_decision": row.p_decision,
                "eta_min": eta,
                "eta_arrival_utc": arrival,
                "threshold_pct": rule.threshold_for(generated),
                "is_event_warning": _iso(generated) == _iso(record.sent_utc),
            })
        stored = population.rows.get((rec[_RADAR_TS], record.station_id), {})
        if stored.get("action") not in (None, "none"):
            out.append({
                "kind": "stored",
                "generated_at_utc": _iso(generated),
                "radar_ts_utc": _iso(rec[_RADAR_TS]),
                "action": stored.get("action"),
                "p_decision": stored.get("p_rain"),
                "eta_min": eta,
                "eta_arrival_utc": arrival,
                "threshold_pct": stored.get("threshold_pct"),
                "is_event_warning": False,
            })
    return out


def _frame_list(
    track: Sequence[tuple],
    frames_from: datetime,
    frames_to: datetime,
    event_frames: Any | None,
    planned_stamps: Mapping[str, Any] | None,
    missing_stamps: Sequence[datetime] | None,
) -> list[dict]:
    """The radar frames this event needs, and whether the bundle has them.

    With an ``EventFrames`` from
    :func:`~dmi_nowcast_sidecar.review_frames.frame_plan` the stamps are
    the planner's, because those are the frames that were rendered.
    Without one they are what the decision rows imply — every ``radar_ts``
    in the frame window — which is the honest answer before the archive
    has been opened, and each entry says which of the two it is.

    ``present`` is likewise two different facts kept apart: with
    ``missing_stamps`` it is the writer's word on what the archive could
    supply; without it, whether a decision row stood on that frame. A
    reviewer scrubbing to a frame with no numbers beside it and a reviewer
    scrubbing to a frame that was never rendered are looking at different
    problems.
    """
    frames_from, frames_to = _as_utc(frames_from), _as_utc(frames_to)
    with_rows = {
        _as_utc(record[_RADAR_TS])
        for record in track
        if frames_from <= _as_utc(record[_RADAR_TS]) <= frames_to
    }
    if event_frames is not None:
        stamps = sorted(
            datetime.strptime(stamp, "%Y%m%d%H%M").replace(tzinfo=UTC)
            for stamp in event_frames.stamps
        )
        source = "frame planner"
    else:
        stamps = sorted(with_rows)
        source = "decision rows"
    missing = (
        None if missing_stamps is None
        else {_as_utc(stamp) for stamp in missing_stamps}
    )
    out: list[dict] = []
    for ts in stamps:
        stamp = _stamp(ts)
        planned = None if planned_stamps is None else planned_stamps.get(stamp)
        out.append({
            "radar_ts_utc": _iso(ts),
            "stamp": stamp,
            "product": None if planned is None else planned.product,
            "overlay": f"frames/{stamp}.overlay.png",
            "observed": f"frames/{stamp}.observed.png",
            "present": (ts not in missing) if missing is not None else None,
            "has_decision_row": ts in with_rows,
            "source": source,
            "present_source": "frame writer" if missing is not None else None,
        })
    return out


def _builder_notes(
    record: EventRecord,
    population: Population,
    inside: Sequence[tuple[int, tuple, TracedRow]],
) -> list[str]:
    """What the builder knows about this event that a field cannot hold."""
    notes = list(population.provenance.get("caveats", ()))
    if record.dual_truth is None:
        notes.append(
            "dual_truth is null: the gauge or the radar said nothing in this "
            "event's window, so no quadrant is supported by the evidence."
        )
    if record.event_class == "uncovered":
        notes.append(
            "Control group: no decision row was watching this onset, so the "
            "coverage rule removed it from POD's denominator. Nothing else "
            "in the system audits that."
        )
    if record.control and record.event_class == "hit":
        notes.append(
            "Control group: a hit, offered with the full vocabulary so the "
            "tag tally has a base rate."
        )
    mismatched = [
        _iso(rec[_RADAR_TS])
        for _i, rec, _row in inside
        if (
            (stored := population.rows.get((rec[_RADAR_TS], record.station_id), {}))
            .get("frame_age_min") is not None
            and abs(
                float(stored["frame_age_min"])
                - _minutes(_as_utc(rec[_GENERATED]) - _as_utc(rec[_RADAR_TS]))
            )
            > 1e-6
        )
    ]
    if mismatched:
        notes.append(
            "generated_at - radar_ts disagrees with the stored frame_age_min "
            f"on {len(mismatched)} row(s): {mismatched[:3]}"
        )
    return notes


# ---------------------------------------------------------------------------
# Fixture mode: a whole bundle with no VM, no parquet and no HDF5
# ---------------------------------------------------------------------------

#: The fixture's stations: one per region the boxes actually separate, so
#: a stratified draw over ``region`` has something to separate.
_FIXTURE_STATIONS: tuple[tuple[str, str, float, float], ...] = (
    ("06180", "Fixture Køge", 55.45, 12.10),
    ("06181", "Fixture Odense", 55.33, 10.32),
    ("06182", "Fixture Aalborg", 57.05, 9.90),
    ("06183", "Fixture Esbjerg", 55.53, 8.55),
    ("06184", "Fixture Tønder", 54.95, 8.87),
    # ~11 km from Odense, so the false-alarm station HAS a neighbour: the
    # "wet next door, dry here" reading is a real mechanism
    # (``fa_gauge_missed_it``) and a fixture with no neighbour pair could
    # never produce it.
    ("06185", "Fixture Ringe", 55.24, 10.40),
)

#: One scripted day per station, chosen so the fixture contains every
#: outcome class the sampler has to allocate. A fixture that only produces
#: false alarms cannot exercise a stratified draw, and a fixture whose
#: class mix depends on an RNG cannot be asserted on at all — so the
#: script is explicit and the seed moves nothing about which class a
#: station produces.
_FIXTURE_ROLES: tuple[str, ...] = (
    "hit", "false_alarm", "miss", "uncovered", "late",
)


def synthetic_stations() -> tuple[StationMeta, ...]:
    """The fixture's station list, regions derived like any other."""
    return tuple(
        station_meta(station_id, name, lat, lon)
        for station_id, name, lat, lon in _FIXTURE_STATIONS
    )


def synthetic_population(
    *,
    seed: int = 20260915,
    start: datetime = datetime(2026, 6, 1, tzinfo=UTC),
    hours: int = 24,
    rule: ReviewRule | None = None,
    stations: Sequence[StationMeta] | None = None,
    window_min: int = DEFAULT_WINDOW_MIN,
    allow_feature_gap: bool = False,
    feature_gap_hours: tuple[int, int] | None = None,
    **kwargs: Any,
) -> Population:
    """A complete population in memory: no VM, no parquet, no HDF5.

    The frontend has to be buildable before the next VPN-free window and
    the tests have to run in CI, so the builder carries a fixture mode
    that produces the same :class:`Population` the real one does — same
    replay, same scorer, same dual truth, same flags — from a scripted day
    of rain.

    Each station plays one role from :data:`_FIXTURE_ROLES`, so the
    population contains a hit, a false alarm, a miss, an ``uncovered``
    onset (rain during a decision outage) and a ``late`` / ``miss_late``
    pair. ``feature_gap_hours`` blanks the feature columns over a span of
    hours, so the feature-gap measurement has something to measure; note
    it takes the DECISION column with it where the rule decides on
    ``p_post``, which is what the live writer's gap actually did.

    ``seed`` jitters the probabilities by at most a point, which is enough
    for two fixtures to look different and far too little to move any of
    them across the threshold: every scripted value is at or below 0.1 or
    at or above 0.85, so which class a station produces is a property of
    the code and not of the seed. A fixture whose class mix moved with the
    RNG would be a fixture whose tests could assert nothing.
    """
    rule = rule or ReviewRule(
        lead_min=30,
        threshold_pct=50,
        probability=served_rule_module.PROBABILITY_CURVE,
        probability_provenance="stored",
        min_useful_lead_min=5.0,
    )
    stations = tuple(stations or synthetic_stations())
    rows, wet_slots = _synthetic_rows(
        stations, start, hours, rule, feature_gap_hours, int(seed),
    )
    truth = synthetic_truth(stations, wet_slots, start, hours, rule)
    return build_population(
        rows=rows,
        stations=stations,
        truth=truth,
        rule=rule,
        window_min=window_min,
        allow_feature_gap=allow_feature_gap,
        **kwargs,
    )


def _synthetic_rows(
    stations: Sequence[StationMeta],
    start: datetime,
    hours: int,
    rule: ReviewRule,
    feature_gap_hours: tuple[int, int] | None,
    seed: int,
) -> tuple[list[dict], dict[str, list[datetime]]]:
    """Scripted decision rows, and the gauge slots they are judged by.

    The gauge is filled from the SCRIPT, not from the rows, so rain during
    the ``uncovered`` role's decision outage exists at the gauge with no
    decision row anywhere near it — which is the whole point of that role
    and would be impossible if the truth were derived from the rows.
    """
    frames = int(hours * 60 / SLOT_MIN)
    column = rule.column
    curve = rule.curve_column
    rng = random.Random(int(seed))
    rows: list[dict] = []
    wet: dict[str, list[datetime]] = {s.station_id: [] for s in stations}
    for index, station in enumerate(stations):
        role = _FIXTURE_ROLES[index % len(_FIXTURE_ROLES)]
        for step in range(frames):
            radar_ts = start + timedelta(minutes=SLOT_MIN * step)
            p, observed, rains, frame_age, has_row = _synthetic_profile(
                role, radar_ts,
            )
            # A point of jitter: enough to tell two fixtures apart, never
            # enough to cross a threshold (see the docstring).
            p = min(1.0, max(0.0, p + rng.uniform(-0.01, 0.01)))
            if rains:
                wet[station.station_id].append(radar_ts)
            if not has_row:
                continue
            generated = radar_ts + timedelta(minutes=frame_age)
            featured = not (
                feature_gap_hours is not None
                and feature_gap_hours[0] <= radar_ts.hour < feature_gap_hours[1]
            )
            row: dict[str, Any] = {
                "radar_ts": radar_ts,
                "generated_at": generated,
                "station_id": station.station_id,
                "p_rain": p,
                "eta_min": 20.0,
                "intensity_mm_h": 0.4 + 3.0 * p,
                "observed_mm_h": observed,
                "forecast_now_mm_h": observed,
                # The stored decision is the trees' own 40 % rule, and is
                # never what the review reads: a builder that fell back to
                # it would disagree with the replay immediately.
                "action": "none",
                "armed_after": True,
                "streak_after": 0,
                "threshold_pct": 40,
                curve: p,
            }
            for lead in DEFAULT_PRODUCT_LEADS_MIN:
                row.setdefault(p_rain_column(lead), p)
            if column != curve:
                row[column] = p if featured else None
            if featured:
                row.update({
                    FEATURE_PROBE_COLUMN: 2.0 * p,
                    "up_max_20km_mm_h": 3.0 * p,
                    "up_max_40km_mm_h": 4.0 * p,
                    "up_dist_km": 30.0 * (1.0 - p),
                    "up_wet_frac_40km": p,
                    "bulk_kmh": 32.0,
                    "bulk_dir_deg": 250.0,
                    "local_speed_kmh": 28.0,
                    "stalled_share": 0.07,
                    "frame_age_min": float(frame_age),
                    "station_radar_km": 45.0,
                })
            rows.append(row)
    return rows, wet


def _synthetic_profile(
    role: str, radar_ts: datetime,
) -> tuple[float, float, bool, float, bool]:
    """``(p, observed mm/h, the gauge is wet, frame age, a row exists)``.

    Every number here is load-bearing. The ``late`` role's frame age is 16
    minutes rather than 14 because the realised lead has to land BELOW
    ``min_useful_lead_min`` while the onset stays on the 10-minute slot
    grid: at 14 minutes the nearest onset is 6 minutes after the warning,
    which is a hit. At 16 it is 4 minutes, which is what ``late`` means.
    """
    minute = radar_ts.hour * 60 + radar_ts.minute
    if role == "hit":
        # Warns at 03:00 (sent 03:14); the rain arrives 03:40 — 26 minutes
        # of realised lead, inside (sent, sent + 30 + 10].
        if 180 <= minute < 220:
            return 0.85, 0.0, False, 14.0, True
        if 220 <= minute < 270:
            return 0.9, 2.0, True, 14.0, True
        return 0.05, 0.0, False, 14.0, True
    if role == "false_alarm":
        if 180 <= minute < 220:
            return 0.9, 0.0, False, 14.0, True
        return 0.05, 0.0, False, 14.0, True
    if role == "miss":
        # Rain the rule never came close to warning about.
        if 720 <= minute < 760:
            return 0.1, 1.5, True, 14.0, True
        return 0.02, 0.0, False, 14.0, True
    if role == "uncovered":
        # Rain inside a four-hour decision outage: no row was watching, so
        # the coverage rule takes the onset out of POD's denominator.
        has_row = not (360 <= minute < 600)
        if 420 <= minute < 460:
            return 0.05, 1.5, True, 14.0, has_row
        return 0.03, 0.0, False, 14.0, has_row
    if role == "late":
        if 300 <= minute < 320:
            return 0.9, 0.0, False, 16.0, True
        if 320 <= minute < 380:
            return 0.9, 2.0, True, 16.0, True
        return 0.04, 0.0, False, 16.0, True
    raise ValueError(f"unknown fixture role {role!r}")  # pragma: no cover


def synthetic_truth(
    stations: Sequence[StationMeta],
    wet_slots: Mapping[str, Sequence[datetime]],
    start: datetime,
    hours: int,
    rule: ReviewRule,
) -> GaugeTruth:
    """A :class:`GaugeTruth` built straight from numpy — no archive at all.

    The real one comes out of ``gauge_truth_vectorised``, which needs a
    parquet station store; this builds the same arrays by hand so the
    fixture mode has a gauge without one. The onset rule is not
    reimplemented: :meth:`StationSlots.onsets_with_amounts` is asked, which
    is the same call the real path makes.
    """
    import numpy as np

    step = timedelta(minutes=SLOT_MIN)
    grid: list[datetime] = []
    cursor = slot_end_of(start, slot_min=SLOT_MIN)
    last = start + timedelta(hours=hours)
    while cursor <= last:
        grid.append(cursor)
        cursor += step
    stamps = np.array([int(g.timestamp()) for g in grid], dtype=np.int64)
    series: dict[str, StationSlots] = {}
    onsets: dict[str, list[datetime]] = {}
    known_until: dict[str, datetime] = {}
    for station in stations:
        wet_at = {
            slot_end_of(_as_utc(stamp), slot_min=SLOT_MIN)
            for stamp in wet_slots.get(station.station_id, ())
        }
        mm = np.array(
            [1.0 if g in wet_at else 0.0 for g in grid], dtype=np.float32,
        )
        dur = np.array(
            [10.0 if g in wet_at else 0.0 for g in grid], dtype=np.float32,
        )
        known = np.ones(stamps.size, dtype=bool)
        wet = mm >= 0.1
        slots = StationSlots(
            station_id=station.station_id,
            slot_end=stamps,
            mm=mm,
            dur=dur,
            known=known,
            wet=wet,
        )
        series[station.station_id] = slots
        onsets[station.station_id] = slots.onsets(
            rule.dry_min, onset_min_mm=rule.onset_min_mm,
        )
        known_until[station.station_id] = grid[-1] if grid else start
    return GaugeTruth(
        series=series,
        onsets=onsets,
        known_until=known_until,
        known_slots=int(stamps.size * len(stations)),
    )


def synthetic_event(
    *,
    event_class: str = "false_alarm",
    population: Population | None = None,
    **kwargs: Any,
) -> dict:
    """One detail document from the fixture population.

    The frontend's fixture: a real :func:`build_event` output, assembled
    from the synthetic population, so a page can be built and tested
    against the true document shape before a bundle exists.
    """
    population = population or synthetic_population(**kwargs)
    found = population.by_class(event_class)
    if not found:
        raise LookupError(
            f"the fixture population has no {event_class!r} event; it has "
            + ", ".join(
                sorted({record.event_class for record in population.records})
            )
        )
    return build_event(found[0], population)


__all__ = [
    "ArmState",
    "CONTROL_GROUPS",
    "DECISION_GAP_MIN",
    "DEFAULT_CLASS_TARGETS",
    "DEFAULT_FLOOR_PER_CELL",
    "EventRecord",
    "FEATURE_GAP_SHARE",
    "FEATURE_PROBE_COLUMN",
    "Neighbour",
    "Population",
    "REGION_BOXES",
    "REGION_OTHER",
    "RULE_LOMO",
    "RULE_SERVED",
    "RadarSlots",
    "ReviewRule",
    "SAMPLE_GROUPS",
    "SKIP_NO_PROBABILITY",
    "Sample",
    "SampleCell",
    "StationMeta",
    "TracedRow",
    "UNKNOWN_BAND",
    "VerdictWindow",
    "arm_state_at",
    "build_event",
    "build_population",
    "dual_truth_of",
    "feature_gap_days",
    "fold_thresholds_of",
    "gauge_known_between",
    "gauge_series",
    "gauge_truth_for",
    "gauge_wet_between",
    "group_of",
    "index_row",
    "intensity_band",
    "load_population",
    "neighbours_of",
    "onset_window",
    "population_hash",
    "radar_slots_of",
    "region_of",
    "replay_station_traced",
    "rule_from_served_options",
    "run_boundary_rearms",
    "station_meta",
    "stratify",
    "synthetic_event",
    "synthetic_population",
    "synthetic_stations",
    "synthetic_truth",
    "traced_warnings",
    "two_slot_rate_mm_h",
    "warning_window",
]
