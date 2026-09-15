"""The review bundle's frozen contract — one definition, four consumers.

A **review bundle** is a self-contained directory describing a few hundred
hand-picked warning events, everything needed to judge each one, and the
radar imagery to look at while judging. It is built on the VM (where the
corpus lives), copied to a laptop, and served read-only by
``scripts/review_server.py`` to a dev-only page in the website frontend.

Four pieces of code must agree on its shape: the builder
(``dmi_nowcast_sidecar.review``), the frame writer
(``dmi_nowcast_sidecar.review_frames``), the server
(``scripts/review_server.py``) and the browser
(``frontend/src/lib/review/schema.ts``). This module is what they agree on,
so a field is renamed here or not at all.

The two builder halves live in the SIDECAR package rather than in core
because they need ``threshold_sweep``, ``served_rule``, ``push.engine`` and
``national_artifacts``; core must never import sidecar. This contract module
stays in core precisely because it imports nothing at all.

Why the bundle exists at all: the served rule scores POD 0.38 / FAR 0.67
against gauge onsets, and nobody has ever looked at an individual bad
warning. A pooled F1 cannot tell a forecast that invented rain from a
shower that passed four kilometres north of the gauge, and those two need
opposite fixes. Reviewing events one at a time under a controlled
vocabulary turns one number into a ranked list of named mechanisms.

Layout
------

::

    <bundle>/manifest.json               provenance, rule, truth, sampling, grid
            /events.json                 the INDEX: one compact row per event
            /events/<event_id>.json      the DETAIL: one file per event
            /stations.json               station metadata + the neighbour graph
            /tags.json                   the controlled vocabulary (this module)
            /frames/<stamp>.overlay.png  RGBA colormapped, product grid
            /frames/<stamp>.observed.png grayscale8 mm/h, for cursor sampling
            /exports/                    written by the server, never by the builder

Index and detail are split because the browser must filter and sort all
~300 rows instantly (140 KB, parsed once) while a detail record is ~35 KB
and only three are ever needed at a time. The split also makes the builder
resumable and a ``--deepen <event_id>`` re-run cheap.

Time and units, stated once and honoured everywhere
---------------------------------------------------

- Every timestamp is ISO 8601 with an explicit ``+00:00``. No naive
  stamps, no local time anywhere in the bundle. The browser converts at
  the render boundary and nowhere earlier.
- ``radar_ts_utc`` is the composite's nominal time (the filename stamp).
- ``generated_at_utc = radar_ts_utc + frame_age_min`` is the decision
  instant AND the wall clock the engine ran on. A warning's ``sent_utc``
  equals it exactly (``threshold_sweep.replay_station`` returns
  ``record[_GENERATED]``).
- Gauge and radar slots are stamped at their **END**: ``slot_end_utc``
  names the slot covering ``(slot_end - 10 min, slot_end]``. An onset
  instant is such a slot end, so the true first drop fell somewhere in the
  preceding ten minutes and every measured lead is biased that far
  negative. Stated here rather than hidden in a correction factor.
- ``lead_error_min = eta_min - (onset - sent)``. **POSITIVE means the rain
  arrived sooner than the ETA said — the warning was LATE.** The website's
  quality page is built on this convention; do not flip it.
- Suffixes: ``_utc`` instant, ``_min`` minutes, ``_mm_h`` rate, ``_mm``
  depth, ``_km`` distance, ``_pct`` integer percent 0-100, ``_deg``
  degrees. A bare probability is a 0-1 fraction.
- **``null`` means "not known / not computed". It never means zero.** Slot
  series therefore carry ``known`` beside ``wet``, exactly as
  :class:`~dmi_nowcast_core.warning_score.StationSlots` does:
  ``{"wet": false, "known": false}`` is *unknown*, not dry. A reader that
  collapses the two has silently turned a gap in the gauge record into
  evidence of a false alarm.

The documents
-------------

``manifest.json`` carries, at minimum: ``schema_version``, ``bundle_id``,
``built_at_utc``, and the blocks ``builder`` (script, git commit, dirty
flag, argv, host, versions), ``corpus`` (roots, the decision directories
**in effective precedence order**, rows loaded, duplicates dropped,
``population_hash``), ``window``, ``rule`` (source, probability column and
its provenance, the post-processing model's identity and how many rows it
filled, the threshold table with ``held_out``, the engine's own constants,
``rows_fallback_to_curve``), ``truth`` (the onset rule, the radar-disc
rule, the neighbour rule, dead gauges excluded, and ``known_until`` pinned
per station), ``sampling`` (seed, strata, allocation, populations, draws,
exclusions), ``grid`` (the :data:`GRID_BLOCK_FIELDS` block), ``frames``
(``products`` — a list, because ``--include-doppler`` can mix two — cadence,
count, bytes, per-encoding metadata, BOTH window edges, and missing stamps),
``feature_doc``, and ``caveats``.

``known_until`` is pinned in the manifest on purpose: the gauge archive
grows, so a later rebuild would otherwise silently re-label events that
were ``pending`` when the bundle was drawn.

``events.json`` is ``{"schema_version", "bundle_id", "events": [row]}``
where each row carries the fields in :data:`INDEX_FIELDS` — enough to
filter, sort, facet and label without touching a detail file.

``events/<event_id>.json`` repeats its index row under ``"index"`` so the
file is meaningful alone, then adds: ``station`` (with fractional product-
grid row/col and the neighbour list), ``window`` (the decision window, the
wider frame window, the slot window, and this station's ``known_until``),
``decisions`` (one entry per frame in the window — every lead's ``p_rain``
and ``p_post``, the decision probability and its source, the threshold and
whether it was crossed, ETA, intensity, observed, the Phase-H features,
the traced engine state, the stored row for contrast, and the
``latest_estimate`` as of that instant), ``decision_gaps``, ``prologue``
(the engine state entering the window — mandatory, see below), ``gauge``,
``radar_disc``, ``neighbours``, ``dual_truth``, ``notifications``,
``frames``, ``flags`` and ``builder_notes``.

Why ``prologue`` is mandatory
-----------------------------

``push.engine.evaluate`` disarms on every ``notify`` **and** every
``already_raining``, and re-arms only after 60 minutes of radar time below
threshold — measured from ``below_since_utc``, which is reset to ``None``
by any over-threshold observation while disarmed. In a showery spell the
probability bounces over the threshold every few frames, the dry clock
never accumulates, and the station can stay disarmed for hours. Every
onset in that spell is a miss the rule could not have caught, and counting
it against POD measures the hysteresis, not the forecast.

The disarming action frequently lies outside a +/-90 minute window, so
without ``prologue`` the reviewer sees a station that inexplicably never
fires. Widening the window instead would double the bundle to hide a
summary that costs a few hundred bytes.

Second order: ``threshold_sweep.replay_station`` resets to
``INITIAL_STATE`` at the head of every coverage run, so a gap longer than
``coverage_gap_min`` hands the station a free re-arm the live service
never had. Rows where that happened carry ``run_boundary_rearm``, or a
reviewer will read an impossible notify as a bug in the tool.
"""
from __future__ import annotations

from typing import Final

#: Bumped when the bundle's shape changes incompatibly. The browser refuses
#: a bundle it was not written against rather than guessing.
REVIEW_SCHEMA_VERSION: Final[int] = 1

#: Bumped when a tag is added, removed or redefined. Stored on every
#: annotation so a later aggregation can tell which vocabulary a human was
#: looking at — renaming a tag under saved annotations silently rewrites
#: history otherwise.
REVIEW_VOCAB_VERSION: Final[int] = 1

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

#: Outcome classes drawn into a bundle. ``pending`` is deliberately absent:
#: its verdict is "ask again later" — DMI backfills late station reports,
#: so the label can still move and a human reviewing it learns nothing.
#: ``uncovered`` IS drawn, as a small labelled control group, because it is
#: silently removed from POD's denominator by the coverage rule and nothing
#: else in the system ever audits that.
OUTCOME_CLASSES: Final[tuple[str, ...]] = (
    "false_alarm",   # a warning that claimed no onset, window closed
    "miss",          # an onset no warning claimed
    "miss_late",     # claimed, but with less than min_useful_lead realised
    "late",          # the warning side of a miss_late
    "hit",           # control group
    "uncovered",     # control group: no decision row was watching
)

#: Classes whose anchor is a warning (``sent_utc``) rather than an onset.
WARNING_SIDE_CLASSES: Final[frozenset[str]] = frozenset({"false_alarm", "late", "hit"})

#: The dual-truth verdict. Gauge defines the event; the radar disc at the
#: same point and the gauges within 20 km are the second and third
#: opinions. The radar is NOT independent evidence — forecast and truth
#: come from the same instrument, so ``both_agree`` is a consistency check.
#: That sentence belongs beside the badge in the UI, not only here.
DUAL_TRUTH_CLASSES: Final[tuple[str, ...]] = (
    "both_wet",             # gauge wet, radar wet
    "radar_wet_gauge_dry",  # representativeness, virga, bright band, near miss
    "gauge_wet_radar_dry",  # radar blind: low-level growth, overshoot, clutter filter
    "both_dry",             # the forecast invented rain
)

#: Event-level flags. Each one marks a reason a reviewer's reading could be
#: wrong if they did not know.
EVENT_FLAGS: Final[dict[str, str]] = {
    "feature_gap": (
        "At least one decision in the window has null Phase-H features, so "
        "p_post is absent and the engine fell back to the curve-calibrated "
        "p_rain. The thresholds were fitted on the p_post scale, so this "
        "row was judged by a different rule wearing the same number."
    ),
    "suspect_gauge_month": (
        "This station's month fails the dead-gauge scan (reported often, "
        "never wet). dead_gauges() is computed window-wide, so a gauge that "
        "died mid-window still contributes half a window of false alarms."
    ),
    "coverage_gap": "A gap longer than coverage_gap_min falls inside the window.",
    "run_boundary_rearm": (
        "The replay reset engine state at a coverage-run boundary inside the "
        "window, handing this station a re-arm the live service never had."
    ),
    "near_known_until": (
        "The window reaches within tolerance of this station's last reported "
        "slot, so the gauge's word on it is not final."
    ),
    "no_probability": (
        "Rows in the window carry neither p_post nor p_rain at the rule's "
        "lead, so the engine passed over them without touching its state."
    ),
}

# ---------------------------------------------------------------------------
# The controlled vocabulary
# ---------------------------------------------------------------------------

#: The per-event judgement. Exactly one, required before an event counts as
#: reviewed. This single field is the headline result: it splits "the
#: forecast was wrong" from "the scoring rule called it wrong", which is
#: the question the whole exercise exists to answer.
VERDICTS: Final[dict[str, str]] = {
    "real_failure": (
        "The forecast was wrong in a way a better model or rule could fix."
    ),
    "metric_artefact": (
        "The forecast was defensible; the label came from the scoring rule — "
        "the onset definition, the matching window, the re-arm, a low-catch "
        "gauge, or coverage."
    ),
    "unclear": "The evidence in the bundle does not settle it.",
}

#: Why a warning fired with no rain behind it. Every code names a mechanism
#: that actually exists in this pipeline, with the evidence that shows it.
FALSE_ALARM_TAGS: Final[dict[str, str]] = {
    "fa_cell_died": (
        "Echo existed upstream (up_max_20/40km_mm_h > 0) and decayed before "
        "arrival. STEPS advects; it carries no deterministic growth or decay."
    ),
    "fa_cell_diverted": (
        "The echo passed to one side: the completed flow at the station "
        "(local_speed_kmh, bulk_dir_deg) differed from the cell's own motion."
    ),
    "fa_arrived_late": (
        "The rain did come, but after sent + lead + tolerance, so the greedy "
        "matching window would not let this warning claim the onset."
    ),
    "fa_arrived_early": (
        "The rain came before the warning could be useful; a neighbouring "
        "claim carries a large positive lead_error_min."
    ),
    "fa_virga_or_aloft": (
        "Column-max reflectivity over a dry gauge — the documented composite "
        "bias. Radar wet, gauge dry, neighbours dry."
    ),
    "fa_clutter_or_bright_band": (
        "Stationary echo across frames, or a high stalled_share; small "
        "station_radar_km (near-radar clutter) or large (melting layer)."
    ),
    "fa_drizzle_below_gauge_floor": (
        "Radar 0.5-1 mm/h; the gauge stayed below 0.1 mm per slot or below "
        "the 0.2 mm two-slot onset floor. Real rain, not a countable onset."
    ),
    "fa_gauge_missed_it": (
        "Neighbours wet, this gauge dry with known=true: wind loss, low "
        "catch, or a gauge not yet dead by the dead_gauges rule."
    ),
    "fa_gauge_unreported": (
        "The window's slots are known=false. There is no truth here, and "
        "known_until / pending did not catch it."
    ),
    "fa_already_raining": (
        "Rain was already falling before the warning, so the onset rule's 60 "
        "dry minutes was never satisfied. The engine's already-raining test "
        "reads only the current row."
    ),
    "fa_probability_saturated": (
        "raw_frac_<lead> pinned at 1.0 — ensemble saturation; visible as "
        "p_rain much greater than p_post, or both pinned."
    ),
    "fa_threshold_marginal": (
        "p_decision within 5 points of the threshold: a coin flip, not a "
        "mechanism. Tag it so it does not pollute the mechanism counts."
    ),
    "fa_stale_frame": (
        "frame_age_min >= 20, or a coverage gap immediately before: the "
        "decision stood on an old composite."
    ),
    "fa_edge_of_coverage": (
        "observed_mm_h null, few valid pixels in the disc, or the station "
        "near a composite or radar edge."
    ),
    "fa_other": "Something else. Requires a note.",
}

#: Why rain arrived with no warning behind it.
MISS_TAGS: Final[dict[str, str]] = {
    "miss_disarmed_rearm": (
        "Disarmed at the onset with the 60-minute re-arm unexpired: "
        "structurally unreachable. If this is a large share of misses, the "
        "finding is a rule change, not a model change."
    ),
    "miss_arm_consumed_already_raining": (
        "The arm was consumed silently by the already-raining branch before "
        "the onset — no push, still disarmed."
    ),
    "miss_below_threshold": (
        "p_decision peaked below the threshold across the whole pre-onset "
        "window: sharpness or calibration, not plumbing."
    ),
    "miss_no_probability": (
        "Both p_post and p_rain were null at the rule's lead, so the engine "
        "passed over those rows (off coverage, unserved lead, feature gap)."
    ),
    "miss_convective_initiation": (
        "Nothing upstream at -30 min (up_max_40km_mm_h near zero, up_dist_km "
        "NaN): the rain grew in place, which advection cannot see."
    ),
    "miss_too_fast": (
        "Rain was upstream but arrived sooner than the lead could cover: "
        "small up_dist_km with high bulk_kmh."
    ),
    "miss_frame_age_ate_the_lead": (
        "frame_age_min plus the arrival time exceeded the lead. A fresher "
        "anchor would have warned."
    ),
    "miss_coverage_gap": (
        "Decision rows are missing around the onset even though the coverage "
        "rule did not call it uncovered."
    ),
    "miss_radar_saw_nothing": (
        "The disc stayed dry through the onset: beam overshoot or shallow "
        "rain, typically at large station_radar_km."
    ),
    "miss_gauge_spurious_onset": (
        "An isolated 0.2 mm with no radar and no neighbour: a suspect onset "
        "(heated gauge, dew, a bumped bucket)."
    ),
    "miss_snow_or_sleet": (
        "Winter, long precip duration with tiny depth, weak radar: the Z-R "
        "relation is rain-tuned."
    ),
    "miss_onset_definition_artefact": (
        "The 'new' onset is a continuation whose dry run was reset by an "
        "UNKNOWN slot rather than a dry one."
    ),
    "miss_threshold_marginal": "Peak p_decision within 5 points below the threshold.",
    "miss_other": "Something else. Requires a note.",
}

#: Tags offered for every class, whichever side the anchor sits on.
COMMON_TAGS: Final[dict[str, str]] = {
    "needs_better_imagery": (
        "The +/-90 minute window or the 2 km product grid was not enough to "
        "judge this one. Candidate for a --deepen re-run."
    ),
    "interesting": "Worth coming back to, or worth showing someone.",
}

#: Tags whose meaning is "I could not decide", which must never be counted
#: as a mechanism in the aggregation.
NON_MECHANISM_TAGS: Final[frozenset[str]] = frozenset({
    "fa_threshold_marginal",
    "miss_threshold_marginal",
    "fa_other",
    "miss_other",
    "needs_better_imagery",
    "interesting",
})


def tags_for_class(event_class: str) -> dict[str, str]:
    """The cause tags offered for one outcome class, plus the common ones.

    A ``hit`` gets BOTH cause lists. That is deliberate: the hit control
    group exists to supply a base rate, and a base rate is only useful if
    the same vocabulary is available. If ``fa_cell_died`` describes 40 % of
    the hits too, it explains nothing about the false alarms — and the only
    way to discover that is to let a reviewer tag it on a hit.
    """
    if event_class in ("false_alarm", "late"):
        return {**FALSE_ALARM_TAGS, **COMMON_TAGS}
    if event_class in ("miss", "miss_late", "uncovered"):
        return {**MISS_TAGS, **COMMON_TAGS}
    return {**FALSE_ALARM_TAGS, **MISS_TAGS, **COMMON_TAGS}


def all_tag_codes() -> frozenset[str]:
    """Every code the server will accept. Anything else is a 422."""
    return frozenset(FALSE_ALARM_TAGS) | frozenset(MISS_TAGS) | frozenset(COMMON_TAGS)


def tags_document() -> dict:
    """``tags.json`` — the vocabulary as the bundle ships it.

    Written by the builder and served to the browser, so the page renders
    the vocabulary the bundle was drawn under rather than whatever the
    frontend was compiled with. The browser's built-in copy is a fallback
    for a bundle too old to carry one, and a test pins the two together.
    """
    return {
        "vocab_version": REVIEW_VOCAB_VERSION,
        "verdicts": [
            {"code": code, "description": text} for code, text in VERDICTS.items()
        ],
        "tag_groups": [
            {
                "group": "false_alarm",
                "label": "Why did it warn with no rain?",
                "tags": [
                    {"code": c, "description": d}
                    for c, d in FALSE_ALARM_TAGS.items()
                ],
            },
            {
                "group": "miss",
                "label": "Why was there no warning?",
                "tags": [{"code": c, "description": d} for c, d in MISS_TAGS.items()],
            },
            {
                "group": "common",
                "label": "Either way",
                "tags": [{"code": c, "description": d} for c, d in COMMON_TAGS.items()],
            },
        ],
        "non_mechanism": sorted(NON_MECHANISM_TAGS),
        "classes": {
            event_class: sorted(tags_for_class(event_class))
            for event_class in OUTCOME_CLASSES
        },
    }


# ---------------------------------------------------------------------------
# Field lists the four consumers share
# ---------------------------------------------------------------------------

#: The grid block, byte-identical in shape to what
#: ``national_artifacts._grid_entry`` writes and what the frontend's
#: ``lib/nowcast/manifest.ts`` already parses — which is the whole reason
#: the review map can reuse ``map/warp.ts`` and ``nowcast/sampler.ts``
#: unchanged. Browser-side inverse:
#: ``col = (x - x_ul_m) / pixel_scale_x_m``, ``row = (y_ul_m - y) / pixel_scale_y_m``.
GRID_BLOCK_FIELDS: Final[tuple[str, ...]] = (
    "proj4", "x_ul_m", "y_ul_m",
    "pixel_scale_x_m", "pixel_scale_y_m", "shape", "downsample_factor",
)

#: Every field of an ``events.json`` row. The browser filters, sorts and
#: facets on these alone, so adding one here is how a new filter becomes
#: possible; a detail-only field can never be filtered on.
INDEX_FIELDS: Final[tuple[str, ...]] = (
    "event_id", "class", "control",
    "station_id", "station_name", "region", "lat", "lon",
    "anchor_utc", "sent_utc", "onset_utc",
    "eta_min", "eta_arrival_utc",
    "p_decision", "p_decision_source", "threshold_pct",
    "lead_error_min",
    "dual_truth", "gauge_wet_in_window", "radar_wet_in_window",
    "neighbour_wet_in_window", "neighbour_n_known",
    "season", "hour_utc",
    "intensity_band", "intensity_mm_h", "intensity_band_source",
    "onset_two_slot_mm",
    "arm_state_at_anchor", "minutes_to_rearm_at_anchor",
    "frames", "frames_missing",
    "flags", "stratum", "detail",
)

#: Intensity bands, used as a sampling stratum. The band is read from the
#: PREDICTION for warning-side events (a false alarm has no onset intensity
#: to read) and from the onset's two-slot depth otherwise, so every event
#: carries ``intensity_band_source`` and the stratification stays auditable.
INTENSITY_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("trace", 0.0, 0.5),
    ("light", 0.5, 1.0),
    ("moderate", 1.0, 4.0),
    ("heavy", 4.0, float("inf")),
)

#: Sampling strata, in the order the allocator nests them.
STRATA_KEYS: Final[tuple[str, ...]] = (
    "outcome_class", "season", "region", "intensity_band",
)

# ---------------------------------------------------------------------------
# Window geometry
# ---------------------------------------------------------------------------

#: Minutes of DECISION history either side of the anchor.
DEFAULT_WINDOW_MIN: Final[int] = 90

#: Extra minutes of RADAR FRAMES before the decision window. A composite is
#: 13-24 minutes old by the time a cycle stands on it, so a decision at the
#: left edge of the decision window refers to a frame from before that
#: edge. Without this pad the reviewer scrubs to the moment that matters
#: and finds a blank map. The two edges are separate fields in the bundle
#: (``window.from_utc`` vs ``window.frames_from_utc``) precisely so nobody
#: conflates them later.
#:
#: ``review_frames.frame_plan`` subtracts ``window + pad + ONE CADENCE``,
#: not ``window + pad``. Stamps snap to the 10-minute grid, so an anchor
#: that is not itself on the grid — every ``sent_utc``, since it carries a
#: fractional frame age — would otherwise round forward past the frame the
#: earliest decision in the window actually stood on. The extra cadence is
#: what makes ``earliest_stamp <= window_from - pad`` true for any anchor.
DEFAULT_FRAME_PAD_MIN: Final[int] = 20

#: Radius of the disc the radar verdict is read over — the production
#: "raining now" geometry (``sample.sample_disc``, p90 over ~1 km), so the
#: second opinion is computed the same way the service computes the number
#: it acts on.
RADAR_DISC_RADIUS_M: Final[float] = 1000.0

#: How far a neighbouring gauge may be and still speak to "was this gauge
#: the odd one out". 20 km comfortably exceeds a Danish summer shower's
#: diameter, which is the point: a wet neighbour says rain existed in the
#: area, NOT that it rained at the event's station. Label it that way in
#: the UI or it will be read as a third vote on the same question.
NEIGHBOUR_RADIUS_KM: Final[float] = 20.0


def event_id(event_class: str, station_id: str, anchor_utc) -> str:
    """``"fa-06074-20260612T1340Z"`` — stable, sortable, filename-safe.

    Stable across rebuilds of the same population, because annotations are
    keyed on it: a reviewer's two hundred judgements must survive a bundle
    rebuild that adds events. Hence it is derived from the event's identity
    (class, station, instant) and never from its position in a list.
    """
    abbrev = {
        "false_alarm": "fa", "miss": "miss", "miss_late": "misslate",
        "late": "late", "hit": "hit", "uncovered": "unc",
    }.get(event_class, event_class)
    stamp = anchor_utc.strftime("%Y%m%dT%H%MZ")
    return f"{abbrev}-{station_id}-{stamp}"
