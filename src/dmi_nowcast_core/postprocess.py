"""Gauge-trained post-processing of the rain probability (Phase H, H-P).

What is wrong with the number we serve
--------------------------------------
``p_rain[L]`` is the fraction of STEPS members whose cumulative max rain
rate crosses the threshold within ``L`` minutes, mapped through a per-lead
isotonic curve fitted on the archive. With 16 members sharing **one**
motion field that fraction saturates: at lead 30 roughly a third of the
non-zero rows sit in the top bin, so a 13 mm/h band 13 km upstream and a
drizzle edge that will die before it arrives both read "1.0 raw" and both
come out at the same calibrated ≈ 0.55 (``archive/calibration_target_20260909/``).
The curve cannot separate them because the fraction is the only thing it
is allowed to look at.

The cycle already knows the difference. It has the observed field, the
completed flow, and therefore how much rain sits upstream along that
flow, how far away it is, how fast it is coming, and how organised it is —
plus the season, the hour, the frame age and the station's distance to a
radar. This module fits a small model on those, per lead, against the same
gauge truth Layer B scores (``benchmark_report``: the gauge was wet at
some point in ``(t, t + L]``), leave-one-(year, month)-out, and calibrates
its output isotonically so the served number stays reliable.

Three things live here
----------------------
1. **The feature catalogue and the grid extraction.**
   :func:`feature_columns` defines every column the replay writes, with
   the definition that has to reach ``summary.json``; :func:`station_features`
   computes the grid-derived ones for a whole station list at once, from
   the field and flow the cycle already has in memory. No STEPS re-run, no
   per-pixel Python.
2. **The transform.** :func:`build_design` turns the stored columns into a
   design matrix — log1p on rain rates and distances, sin/cos on the two
   angles, one-hot on the season, an explicit indicator wherever "missing"
   is itself informative (no echo upwind, no ETA inside the horizon) —
   and :class:`Standardiser` stores the means and scales that go in the
   artefact.
3. **The model.** :func:`fit_logistic` is an L2-regularised logistic
   regression by ``scipy.optimize.minimize`` (L-BFGS-B); scikit-learn is
   deliberately not a dependency of this project. :class:`PostprocessModel`
   holds one fit per lead plus its isotonic recalibration and round-trips
   through JSON in the same shape the calibration curves and the push
   threshold table use. :func:`leave_one_month_out` is the honest
   evaluation: out-of-fold predictions only, scored against the curve
   baseline on exactly the same rows, with a paired day-block bootstrap on
   the differences.

What post-processing v2 added, and what it did not
--------------------------------------------------
Everything below is switchable and every default is the model that has
been in service since 2026-09-11, so a caller that asks for nothing gets
the same design matrix and the same coefficients it always did
(``tests/test_postprocess_model.py`` pins that against the pre-v2 code).

* **:class:`FitSettings`** is the one value that says what is being
  fitted — family, design, recalibration, station effects — and it is
  passed down through the fit, the folds, the learning curve and the
  ablation, so what was evaluated out of fold is what gets shipped.
* **Four families** (:data:`MODEL_KINDS`): a logistic or a boosted
  ensemble, each either per lead or ``-shared`` across leads with the
  lead itself in the design. Trees are fitted offline only — see
  :mod:`dmi_nowcast_core.postprocess_trees` — and served through a numpy
  evaluator.
* **:class:`DesignSpec`** records how the design was built, including
  which catalogue columns the fit discovered, which of them it decided to
  log, and where the spline knots are. Fit time resolves; serving time
  replays. It is what lets the v2 design read the feature catalogue at
  run time without a model losing the columns it was fitted on.
* **The lead ordering is enforced.** P(rain within 60) cannot be below
  P(rain within 45), and independently fitted per-lead curves are free to
  cross, so every multi-lead prediction goes through the same running max
  (``national.enforce_lead_monotonic``) the served ``p_rain`` does — on
  the way out of the model AND on the out-of-fold predictions the report
  scores.
* **The evaluation grew a second axis.** Every comparison is reported on
  all rows and on the onset-relevant :data:`DRY` subset, the baseline can
  itself be a model refitted in the same folds, and
  :func:`learning_curve` / :func:`ablation` answer "how much archive does
  this need" and "which feature family earns its keep".

Conventions
-----------
* numpy in, numpy / dicts out. No pandas, no scikit-learn, no Arrow.
* Missing is NaN, never zero. A feature that is NaN after the transform is
  imputed with the **training mean** (0 after standardisation) and, where
  the absence carries information, an indicator column says so.
* Angles are compass bearings in degrees — 0 = north, 90 = east — of the
  direction the rain is *heading toward*, matching the served motion
  arrows (``national.motion_grids_kmh``: north = ``-vy``, east = ``vx``).
* Rain rates are mm/h on the native 500 m composite grid; distances are
  km; speeds are km/h.
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .benchmark import (
    brier_decomposition,
    paired_block_bootstrap,
    pr_auc,
    reliability_bins,
    roc_auc,
)
from .calibrate import IsotonicCalibrator, pava_weighted
# The serving-side guard on p_rain across leads, reused rather than
# restated: "rain within 60 min" contains "rain within 45 min", and this
# module's per-lead isotonic curves are as free to violate that as the
# national curves were. One implementation of the fix, one docstring
# explaining it.
from .national import enforce_lead_monotonic
# The lead → timestep bucket arithmetic, imported rather than restated:
# ``ens_mean_<lead>`` is read off the very ensemble ``raw_frac_<lead>``
# is reduced from, and two implementations of "which timestep is lead L"
# would eventually put the two in different buckets.
from .national import _steps_in_lead

__all__ = [
    "SCHEMA_VERSION",
    "FEATURE_DOC",
    "SCALAR_FEATURE_COLUMNS",
    "SCALAR_FEATURE_COLUMNS_V2",
    "SCALAR_FEATURE_COLUMNS_NG",
    "GaugeSlotTable",
    "neighbour_gauge_features",
    "PROTOCOLS",
    "PROTOCOL_AT_GAUGE",
    "PROTOCOL_RANDOM_POINT",
    "mask_own_gauge",
    "distance_bins",
    "distance_bin_labels",
    "DISTANCE_COLUMN",
    "DISTANCE_BIN_EDGES",
    "random_point_distance_weights",
    "gauge_distance_weights",
    "random_point_expectation",
    "station_groups",
    "group_codes",
    "FoldPlan",
    "DESIGN_SOURCE_COLUMNS",
    "RAW_FRACTION_PREFIX",
    "raw_fraction_column",
    "ENS_MEAN_PREFIX",
    "ENS_P90_PREFIX",
    "ens_mean_column",
    "ens_p90_column",
    "upstream_bin_column",
    "POST_PREFIX",
    "POST_COLUMN_TEMPLATE",
    "post_column",
    "TARGET_WET",
    "TARGET_ONSET",
    "TARGETS",
    "ONSET_PREFIX",
    "ONSET_COLUMN_TEMPLATE",
    "target_column_template",
    "post_schema",
    "SHARED_SOURCE_COLUMNS",
    "feature_source_columns",
    "feature_only_columns",
    "feature_schema",
    "feature_documentation",
    "feature_row",
    "finite_or_none",
    "feature_columns",
    "SEASONS",
    "SEASON_MONTHS",
    "season_of_month",
    "seasons_from_epoch",
    "hours_from_epoch",
    "months_from_epoch",
    "year_months_from_epoch",
    "month_labels",
    "WET_MM_H",
    "OBS_DISC_KM",
    "UPSTREAM_HALF_WIDTH_KM",
    "UPSTREAM_NEAR_KM",
    "UPSTREAM_FAR_KM",
    "UPSTREAM_STEP_KM",
    "UPSTREAM_BIN_KM",
    "UPSTREAM_BINS",
    "WET_FRAC_DISC_KM",
    "MIN_FLOW_PX_PER_FRAME",
    "ETA_CAP_MIN",
    "UP_DIST_CAP_KM",
    "DEFAULT_GAUGE_LAG_MIN",
    "GAUGE_WINDOWS_MIN",
    "GAUGE_DRY_WINDOW_MIN",
    "GAUGE_SINCE_CAP_MIN",
    "disc_max",
    "station_features",
    "station_gauge_features",
    "ensemble_point_features",
    "design_columns",
    "build_design",
    "DESIGN_V1",
    "DESIGN_V2",
    "DESIGN_VERSIONS",
    "DesignSpec",
    "design_spec",
    "natural_spline_basis",
    "spline_knots",
    "FEATURE_FAMILIES",
    "feature_family",
    "family_columns",
    "Standardiser",
    "fit_logistic",
    "DEFAULT_ISOTONIC_BINS",
    "MIN_SEASON_ISOTONIC_ROWS",
    "fit_isotonic_binned",
    "fit_isotonic_by_season",
    "DRY_BEFORE_MIN",
    "DRY_COLUMN",
    "DRY",
    "dry_before",
    "dry_subset",
    "FAMILY_NAMES",
    "KIND_LOGISTIC",
    "KIND_LOGISTIC_SHARED",
    "KIND_TREES",
    "KIND_TREES_SHARED",
    "MODEL_KINDS",
    "is_tree_kind",
    "is_shared_kind",
    "LEAD_COLUMN",
    "LOG_LEAD_COLUMN",
    "own_column",
    "ISOTONIC_POOLED",
    "ISOTONIC_PER_SEASON",
    "ISOTONIC_MODES",
    "STATION_L2_MULTIPLE",
    "FitSettings",
    "LeadModel",
    "PostprocessModel",
    "fit_lead",
    "fit_postprocess",
    "out_of_fold_predictions",
    "leave_one_month_out",
    "subset_scores",
    "CURVE_ORDER_RANDOM",
    "CURVE_ORDER_DATE",
    "CURVE_ORDERS",
    "DEFAULT_CURVE_SEED",
    "learning_curve",
    "ablation",
    "POOLED",
    "N_BINS",
    "enforce_lead_monotonic",
]

#: Version of the ``postprocess.json`` document this module writes and reads.
#:
#: Deliberately NOT bumped for the model families added in post-processing
#: v2. Every one of them is additive with a default equal to today's
#: behaviour: a document without ``kind`` is a logistic, without ``design``
#: is v1, without ``station_offsets`` has none. That keeps the two
#: directions of a deploy skew safe — an old build reading a new default
#: document gets exactly what it always got, and a new build reading an
#: old document likewise. A build meeting a document that really does use
#: a newer design fails the column count in
#: :meth:`Standardiser.transform`, which every caller already catches by
#: falling back to the served curve.
SCHEMA_VERSION = 1

#: The model families. ``kind`` on the document says which, and the
#: default is the one that shipped.
#:
#: A ``-shared`` kind fits ONE model over the rows of every served lead
#: stacked, with the lead itself in the design. The alternative — one
#: model per lead — spends a quarter of the archive on each and lets four
#: independent fits disagree about how the answer bends with the horizon.
#: A shared fit cannot disagree with itself, and it is the only way the
#: 20-minute rows get to inform the 60-minute answer.
KIND_LOGISTIC = "logistic"
KIND_LOGISTIC_SHARED = "logistic-shared"
KIND_TREES = "trees"
KIND_TREES_SHARED = "trees-shared"
MODEL_KINDS: tuple[str, ...] = (
    KIND_LOGISTIC, KIND_LOGISTIC_SHARED, KIND_TREES, KIND_TREES_SHARED,
)

#: Suffix that marks a kind as fitted across leads.
SHARED_SUFFIX = "-shared"


def is_tree_kind(kind: str) -> bool:
    """Is this kind a boosted ensemble rather than a logistic?"""
    return str(kind) in (KIND_TREES, KIND_TREES_SHARED)


def is_shared_kind(kind: str) -> bool:
    """Is this kind one model over every lead, with the lead as a feature?"""
    return str(kind).endswith(SHARED_SUFFIX)

#: Recalibration modes. ``pooled`` is one curve per lead — what shipped.
ISOTONIC_POOLED = "pooled"
ISOTONIC_PER_SEASON = "per-season"
ISOTONIC_MODES: tuple[str, ...] = (ISOTONIC_POOLED, ISOTONIC_PER_SEASON)


# ---------------------------------------------------------------------------
# The feature catalogue
# ---------------------------------------------------------------------------

#: Prefix of the UNcalibrated ensemble-fraction columns. The calibrated
#: ``p_rain_<lead>`` stays exactly as it was — it is the baseline this work
#: has to beat, so it must not be overwritten by the thing under test.
RAW_FRACTION_PREFIX = "raw_frac_"


def raw_fraction_column(lead: int) -> str:
    """Column holding the raw ensemble fraction at ``lead`` minutes."""
    return f"{RAW_FRACTION_PREFIX}{int(lead)}"


#: Every feature column the replay adds, with the definition that must
#: survive into ``summary.json`` and the report. Order is the order the
#: columns are written in. ``raw_frac_<lead>`` is added per served lead by
#: :func:`feature_columns`.
#:
#: Columns the decision schema ALREADY carries and the model also reads —
#: ``observed_mm_h`` (the block-p90 observed rain at the station's product
#: pixel, i.e. the "obs_p90_mm_h" of the plan), ``eta_min``,
#: ``intensity_mm_h`` and ``p_rain_<lead>`` — are deliberately NOT
#: duplicated here. One column, one writer.
SCALAR_FEATURE_COLUMNS: tuple[tuple[str, str], ...] = (
    (
        "obs_max_5km_mm_h",
        "Maximum observed rain rate (mm/h) over the native-grid disc of "
        "radius 5 km centred on the station, from the same anchor field "
        "the cycle forecast from. NaN when no pixel of the disc is inside "
        "the composite.",
    ),
    (
        "up_max_20km_mm_h",
        "Maximum observed rain rate (mm/h) in the upwind corridor 6 km "
        "wide and 0-20 km long, laid along the LOCAL completed flow at "
        "the station (upwind = opposite the direction the cells move). "
        "NaN when the corridor has no pixel inside the composite or the "
        "flow gives no usable direction.",
    ),
    (
        "up_max_40km_mm_h",
        "The same maximum over the 0-40 km corridor.",
    ),
    (
        "up_dist_km",
        "Distance along the corridor to the nearest pixel at or above "
        "0.5 mm/h, in km, 0 when the station itself is wet. NaN when the "
        "40 km corridor holds no such pixel ('no echo upwind') — the "
        "model turns that into an indicator plus a capped value rather "
        "than a large number.",
    ),
    (
        "up_wet_frac_40km",
        "Share of the 40 km corridor's in-composite pixels at or above "
        "0.5 mm/h. Separates a solid band from a ragged edge at the same "
        "maximum intensity.",
    ),
    (
        "bulk_kmh",
        "Speed of the cycle's bulk storm motion (km/h) — the target the "
        "flow completion relaxes toward (dense_flow.MotionEstimate."
        "bulk_vy/bulk_vx), converted with the frame's own pixel size and "
        "spacing.",
    ),
    (
        "bulk_dir_deg",
        "Compass bearing the bulk motion is heading TOWARD: 0 = north, "
        "90 = east. NaN when the bulk motion is below the usable-speed "
        "floor.",
    ),
    (
        "local_speed_kmh",
        "Speed (km/h) of the COMPLETED flow at the station's own pixel — "
        "the field the forecast is advected with there, which off the "
        "echo is the completion's fill rather than a measurement.",
    ),
    (
        "stalled_share",
        "Share of wet pixels whose RAW flow estimate is below 5 km/h "
        "(dense_flow.MotionEstimate.stalled_share). A per-cycle health "
        "signal for the H-F stall defect, constant across the cycle's "
        "stations.",
    ),
    (
        "season",
        "'summer' (May-Sep), 'winter' (Dec-Mar) or 'shoulder' (Apr, Oct, "
        "Nov), from the decision instant. The same split the threshold "
        "sweep and the Layer B strata use.",
    ),
    (
        "hour_utc",
        "Hour of the decision instant in UTC, 0-23. The model uses it as "
        "sin/cos so 23 and 0 are adjacent.",
    ),
    (
        "frame_age_min",
        "Minutes between the anchor frame's nominal time and the decision "
        "instant — the cycle's own simulated latency under the anchor "
        "policy, not a constant.",
    ),
    (
        "station_radar_km",
        "Great-circle km from the station to the nearest DMI radar "
        "(product_pairs.nearest_radar_km). Constant per station; the "
        "composite under-detects close to a radar and over-reads column "
        "max far from one.",
    ),
)


#: The v2 feature block (post-processing v2, 2026-09-16). Appended after
#: the v1 columns and never mixed into them: the schema is additive, a row
#: written before this existed reads these as null, and the order of the
#: columns that were already there must not move.
#:
#: Four families, all nullable and all computed by the SAME functions the
#: replay and the live cycle share — ``station_gauge_features`` (the
#: ``g_*`` block), ``station_features`` (the radar-history and corridor
#: block) and ``ensemble_point_features`` (the ``ens_*`` block).
SCALAR_FEATURE_COLUMNS_V2: tuple[tuple[str, str], ...] = (
    # -- F1: the station's own gauge, as of what the cycle could see ------
    (
        "g_mm_10",
        "Millimetres the gauge at this point measured in the newest "
        "VISIBLE 10-minute slot. Visible means the slot ENDED at or "
        "before the decision instant minus the availability lag "
        "(``gauge_lag_min``, 10 min by default), so this is what the "
        "service could actually have read — never what the archive knows "
        "now. Null at a point that is not a gauge, and at a gauge that "
        "reported no amount in the window.",
    ),
    (
        "g_mm_30",
        "The same sum over the newest three visible slots (30 minutes "
        "back from the visibility horizon).",
    ),
    (
        "g_mm_60",
        "The same sum over the newest six visible slots (60 minutes).",
    ),
    (
        "g_min_since_wet",
        "Minutes from the DECISION INSTANT back to the end of the most "
        "recent visible WET slot (>= 0.1 mm or >= 1 min of precipitation "
        "— ``warning_score``'s rule, unchanged), capped at 360. The lag "
        "is included, so the smallest value this can take is the lag "
        "itself. 360 means 'known, and dry for at least six hours'; null "
        "means the gauge said nothing at all in that window.",
    ),
    (
        "g_dry_60",
        "1 when no visible slot in the last 60 minutes was wet, 0 when "
        "one was, null when no slot in that window is known. This is the "
        "onset-relevance flag: the gate for this track is evaluated "
        "separately on the rows where it is 1.",
    ),
    (
        "g_known",
        "1 when the gauge reported at least one slot in the visible "
        "60-minute window, 0 otherwise — including at every point that "
        "is not a gauge at all (home, a subscriber). Never null: it is "
        "the indicator that says whether the rest of the ``g_*`` block "
        "is a measurement or an absence.",
    ),
    # -- F2: radar history at the point ----------------------------------
    (
        "obs_prev10_mm_h",
        "The point's observed rain rate (mm/h) one frame earlier — the "
        "second-newest frame of the same history triple the flow and the "
        "cascade ate, converted with the anchor frame's own Z-R "
        "parameters. NaN when the cycle has no previous frame or the "
        "point is off that frame's grid.",
    ),
    (
        "obs_prev20_mm_h",
        "The same, two frames earlier. With ``observed_mm_h`` the two "
        "give the model the trend at the point rather than a snapshot.",
    ),
    (
        "obs_max_5km_prev10_mm_h",
        "``obs_max_5km_mm_h`` one frame earlier: the 5 km disc maximum on "
        "the previous frame, so a shower that is growing and one that is "
        "dying are not the same row.",
    ),
    (
        "wet_frac_5km",
        "Share of the in-composite pixels within 5 km of the point at or "
        "above 0.5 mm/h, NOW. Separates 'a cell clips the disc' from 'the "
        "disc is inside a band' at the same maximum.",
    ),
    (
        "wet_frac_10km",
        "The same share over a 10 km disc.",
    ),
    # -- F3: the upwind corridor, resolved -------------------------------
    (
        "up_mean_40km_mm_h",
        "Mean observed rain rate (mm/h) over the in-composite pixels of "
        "the 40 km upwind corridor — the same corridor "
        "``up_max_40km_mm_h`` is taken over, so a solid band and a single "
        "core no longer read alike. NaN when the corridor is undefined.",
    ),
) + tuple(
    (
        f"up_max_b{index}",
        f"Maximum observed rain rate (mm/h) in the {index * 5}-"
        f"{index * 5 + 5} km bin of the upwind corridor. The eight bins "
        "resolve WHERE along the corridor the rain is, which the two "
        "cumulative maxima cannot say. NaN for a bin with no "
        "in-composite pixel, and for every bin when the corridor is "
        "undefined.",
    )
    for index in range(8)
) + (
    # -- F4: the shape of the ensemble -----------------------------------
    (
        "ens_eta_spread_min",
        "Inter-quartile range, in minutes, of the member arrival times at "
        "the point's product pixel — per member, the first timestep whose "
        "rate crosses the detection threshold, on the same minutes-from-"
        "now clock ``eta_min`` uses. Null when fewer than four members "
        "ever arrive, because a quartile over three numbers is not a "
        "spread. A tight spread is agreement; a wide one is four members "
        "carrying the whole probability.",
    ),
)

# ---------------------------------------------------------------------------
# The neighbour gauges: what the stations AROUND a point are measuring
# ---------------------------------------------------------------------------
#
# The ``g_*`` block above is the biggest thing v2 added and the one thing a
# subscriber cannot have: it is what the gauge AT the point measured, and a
# point is a place someone lives, not a DMI station. The model is fitted at
# gauges and served at addresses, so a feature that exists only at gauges
# flatters every offline number by exactly the amount it will not deliver.
#
# The approximation is the gauges AROUND the point, placed in the frame of
# the cycle's own motion. A gauge 20 km upstream of a point, in a band
# moving at 40 km/h, is telling that point what is going to happen in half
# an hour; the same gauge 20 km DOWNSTREAM is telling it what already
# happened somewhere else. Distance alone cannot tell the two apart, which
# is why everything here is measured along the flow rather than as a radius
# — with one deliberate exception, the vicinity block, which is
# direction-free on purpose so that something survives when the motion
# field does not (:data:`NG_MIN_SPEED_KMH`).
#
# Leave-self-out is the whole point and is enforced by construction: the
# gauge standing at the point, if there is one, is removed before anything
# is computed (``exclude_self``, plus the :data:`NG_SELF_KM` rule for a
# point whose station id the caller does not know). A column here can
# therefore be filled at a gauge station in the archive and mean exactly
# what it will mean at an address.

#: Radius of the neighbour search, km. Wide enough that a point anywhere in
#: Denmark has several gauges inside it (the gauge-to-nearest-other-gauge
#: distance runs ~10-35 km) and narrow enough that "upstream" still means
#: the same weather system rather than a different front.
NG_RADIUS_KM = 60.0

#: Half-width of the upstream corridor, km. Wider than the radar corridor
#: (:data:`UPSTREAM_HALF_WIDTH_KM`, 3 km) by design: that one is laid over a
#: 500 m grid and can afford to be thin, this one has ~100 gauges for the
#: whole country and a 3 km corridor would be empty almost everywhere.
NG_CROSS_KM = 15.0

#: Upper edges of the travel-time bins, minutes. The bins are DISJOINT —
#: (0, 30], (30, 60], (60, 120] — not cumulative: "there is rain half an
#: hour upstream" and "there is rain two hours upstream" are different
#: statements about the next hour, and a cumulative column would let the
#: second hide inside the first.
NG_TAU_EDGES_MIN: tuple[float, ...] = (30.0, 60.0, 120.0)

#: Radius of the direction-free vicinity block, km.
NG_VICINITY_KM = 20.0

#: Cross-track scale of the corridor weight, km: a gauge ``|c|`` km off the
#: track counts ``1 / (1 + |c| / this)``. A gauge on the track counts
#: fully, one 15 km off counts a quarter — a soft version of the corridor's
#: own hard edge, so the weighted mean does not jump when a gauge drifts
#: across it.
NG_CROSS_WEIGHT_KM = 5.0

#: A gauge closer than this to the point IS the point. The belt to
#: ``exclude_self``'s braces: a caller that resolves a point to a station id
#: excludes it by name, and a caller that does not still cannot read its own
#: gauge back as a neighbour.
NG_SELF_KM = 0.5

#: Below this speed there is no motion frame to place anything in: the
#: direction is noise and ``tau = a / v`` is an hour per kilometre. The
#: upstream and nearest-upstream-wet blocks go null and ``ng_frame_ok`` says
#: so; the vicinity block, which never needed a direction, is still filled.
NG_MIN_SPEED_KMH = 5.0

#: The window a neighbour's "is it raining there" is read over, minutes —
#: back from the visibility horizon, exactly like ``g_mm_30``.
NG_WET_WINDOW_MIN = 30

#: The window ``ng_near_mm_60`` sums, minutes.
NG_NEAR_WINDOW_MIN = 60

#: Latitude the equirectangular kilometre grid is linearised at. Denmark
#: spans 54.6-57.7 N, so one grid at 56 N costs at most ~1.4 % in the
#: east-west scale at the far ends of the country — a few hundred metres on
#: a 20 km distance, against features binned in 10 km steps and a corridor
#: 30 km wide. A haversine per (point, gauge) pair per cycle would be
#: exact and would buy nothing.
NG_REF_LAT_DEG = 56.0

#: Kilometres per degree of latitude on the sphere the rest of the project
#: measures on (``lightning.EARTH_RADIUS_KM``).
KM_PER_DEG_LAT = math.pi * 6371.0088 / 180.0

#: Kilometres per degree of longitude at :data:`NG_REF_LAT_DEG`.
KM_PER_DEG_LON = KM_PER_DEG_LAT * math.cos(math.radians(NG_REF_LAT_DEG))


def ng_tau_suffix(edge: float) -> str:
    """``30.0`` -> ``"t30"`` — the travel-time bin's name in a column."""
    return f"t{int(edge)}"


def _ng_bin_range(index: int) -> tuple[float, float]:
    lower = 0.0 if index == 0 else float(NG_TAU_EDGES_MIN[index - 1])
    return lower, float(NG_TAU_EDGES_MIN[index])


_NG_UPSTREAM_STATS: tuple[tuple[str, str], ...] = (
    (
        "ng_up_mm_max",
        "the LARGEST of their last-30-visible-minute rainfall totals (mm) "
        "— one gauge in a core is enough to say a core is coming",
    ),
    (
        "ng_up_mm_wmean",
        "the same totals averaged, weighted "
        f"1 / (1 + |cross-track| / {NG_CROSS_WEIGHT_KM:.0f} km) so a gauge "
        "on the track counts for more than one at the corridor's edge — "
        "the maximum's counterpart: how WIDE the rain is, not how hard",
    ),
    (
        "ng_up_wet_share",
        "the share of them that were wet in their last 30 visible minutes "
        "(``warning_score``'s wet rule). Separates one shower crossing the "
        "corridor from a front filling it",
    ),
    (
        "ng_up_count",
        "how many of them there are. Zero is a real answer and not a null: "
        "'no gauge is due to reach this point in that window' is "
        "information, and the columns above are null exactly then",
    ),
)


def _ng_upstream_columns() -> tuple[tuple[str, str], ...]:
    out: list[tuple[str, str]] = []
    for stem, meaning in _NG_UPSTREAM_STATS:
        for index, edge in enumerate(NG_TAU_EDGES_MIN):
            lower, upper = _ng_bin_range(index)
            out.append((
                f"{stem}_{ng_tau_suffix(edge)}",
                "Over the OTHER gauges whose rain is due to reach this "
                f"point in ({lower:.0f}, {upper:.0f}] minutes — within "
                f"{NG_RADIUS_KM:.0f} km, upstream along the cycle's bulk "
                f"motion, at most {NG_CROSS_KM:.0f} km off the track, and "
                "with a known last half hour — " + meaning + ". Null when "
                "the motion frame is unusable (``ng_frame_ok`` = 0) and, "
                "except for the count, when no such gauge has an amount.",
            ))
    return tuple(out)


#: The neighbour-gauge block (post-processing v2, 2026-09-17). A family of
#: its own for the ablation (``neighbour``), appended after the v2 columns
#: and never mixed into them, for the same append-only reason.
SCALAR_FEATURE_COLUMNS_NG: tuple[tuple[str, str], ...] = (
    _ng_upstream_columns()
    + (
        (
            "ng_upwet_tau_min",
            "Travel time in minutes from the NEAREST-IN-TIME upstream gauge "
            "that is actually wet — the same candidate set as the bins "
            f"above (within {NG_RADIUS_KM:.0f} km, upstream, at most "
            f"{NG_CROSS_KM:.0f} km off the track, tau in (0, "
            f"{NG_TAU_EDGES_MIN[-1]:.0f}]), restricted to gauges wet in "
            "their last 30 visible minutes, and then the smallest tau. "
            "This is the one column that answers 'when', rather than 'how "
            "much, somewhere in a window'. Null when there is no such "
            "gauge, and whenever ``ng_frame_ok`` is 0.",
        ),
        (
            "ng_upwet_cross_km",
            "How far that gauge sits off the track, km, unsigned. A wet "
            "gauge dead ahead and one at the corridor's edge carry very "
            "different weight at the same tau, and the model cannot see "
            "the difference from the tau alone.",
        ),
        (
            "ng_upwet_mm_30",
            "That gauge's last-30-visible-minute total, mm. Null when it "
            "reported wet slots but no amount.",
        ),
        (
            "ng_near_km",
            "Distance in km to the nearest OTHER gauge — pure geometry, "
            "no weather in it, and the only column here that is (almost) "
            "constant per point. It is what says how much the rest of this "
            "block is worth: at a point 6 km from a gauge the neighbour "
            "block is nearly the gauge block, at 35 km it is a rumour. The "
            "random-point validation bins its rows on exactly this.",
        ),
        (
            "ng_near_mm_60",
            "That nearest gauge's rainfall total (mm) over its last 60 "
            "visible minutes. Null when it said nothing.",
        ),
        (
            "ng_near_min_since_wet",
            "Minutes from the DECISION INSTANT back to the end of that "
            "gauge's most recent visible wet slot, capped at 360 "
            "(``GAUGE_SINCE_CAP_MIN``) — ``g_min_since_wet``'s rule, read "
            "at the neighbour instead of at the point. The cap means "
            "'known, and dry for at least six hours'; null means the gauge "
            "said nothing at all.",
        ),
        (
            "ng_wet_share_20km",
            f"Share of the other gauges within {NG_VICINITY_KM:.0f} km with "
            "a known last half hour that were wet in it. Direction-free on "
            "purpose: it is the one upstream-ish signal that survives a "
            "stalled or missing motion field.",
        ),
        (
            "ng_count_20km",
            f"How many other gauges within {NG_VICINITY_KM:.0f} km had a "
            "known last half hour — the denominator of the share above, so "
            "a 1.0 over one gauge and a 1.0 over five are not read alike. "
            "Zero, never null.",
        ),
        (
            "ng_frame_ok",
            "1 when the cycle's bulk motion gave a usable frame to place "
            f"the neighbours in — speed at least {NG_MIN_SPEED_KMH:.0f} "
            "km/h and a finite direction — and 0 when it did not, in which "
            "case every upstream and upwet column above is null and the "
            "vicinity block is still filled. Never null: it is the "
            "indicator that says which half of this block is a measurement.",
        ),
    )
)


#: Number of 5 km bins the upwind corridor is resolved into (F3).
UPSTREAM_BINS = 8

#: Prefix of the per-lead ensemble-shape columns (F4). Both are cumulative
#: over the lead, exactly like ``raw_frac_<lead>``: the member's running
#: maximum rain rate up to that lead, reduced across members.
ENS_MEAN_PREFIX = "ens_mean_"
ENS_P90_PREFIX = "ens_p90_"


def ens_mean_column(lead: int) -> str:
    """Column holding the member-mean cumulative max at ``lead`` minutes."""
    return f"{ENS_MEAN_PREFIX}{int(lead)}"


def ens_p90_column(lead: int) -> str:
    """Column holding the member-P90 cumulative max at ``lead`` minutes."""
    return f"{ENS_P90_PREFIX}{int(lead)}"


def upstream_bin_column(index: int) -> str:
    """Column holding the corridor maximum in the ``index``-th 5 km bin."""
    return f"up_max_b{int(index)}"


def feature_columns(leads: Sequence[int]) -> tuple[tuple[str, str], ...]:
    """``((name, definition), ...)`` for every feature column, in write order.

    Append-only by construction: the v1 block (``raw_frac_<lead>`` then
    :data:`SCALAR_FEATURE_COLUMNS`) keeps the exact order it has always
    had, and every column added since follows it. A parquet file written
    before a column existed reads it as null, and every reader aligns by
    name, so the order matters only for the writers — which all come
    through here.
    """
    sorted_leads = sorted({int(lead) for lead in leads})
    raw = tuple(
        (
            raw_fraction_column(lead),
            f"UNcalibrated ensemble fraction at {int(lead)} min — the "
            "share of members whose cumulative max crosses the detection "
            "threshold by then, before the served isotonic curve. NaN off "
            "coverage.",
        )
        for lead in sorted_leads
    )
    ens = tuple(
        entry
        for lead in sorted_leads
        for entry in (
            (
                ens_mean_column(lead),
                f"Mean across ensemble members of the member's cumulative "
                f"maximum rain rate (mm/h) by {lead} min. The fraction says "
                "how many members cross the threshold; this says how hard "
                "it rains in them. NaN off coverage.",
            ),
            (
                ens_p90_column(lead),
                f"90th percentile across members of the same cumulative "
                f"maximum by {lead} min — the wet tail, which a mean over "
                "16 members hides. NaN off coverage.",
            ),
        )
    )
    return (
        raw + SCALAR_FEATURE_COLUMNS + ens + SCALAR_FEATURE_COLUMNS_V2
        + SCALAR_FEATURE_COLUMNS_NG
    )


#: name → definition, flattened, for the leads the products publish today.
#: The per-lead columns are not here (they depend on the served leads);
#: :func:`feature_documentation` is the one that answers for a run.
FEATURE_DOC: dict[str, str] = dict(
    SCALAR_FEATURE_COLUMNS + SCALAR_FEATURE_COLUMNS_V2
    + SCALAR_FEATURE_COLUMNS_NG
)


#: Prefix of the POST-PROCESSED probability column — what the push engine
#: decides on once ``push.probability_source`` is ``postprocess``. Written
#: BESIDE the served ``p_rain_<lead>``, never over it: both arms of every
#: comparison have to survive in the same row, for the same reason
#: ``raw_frac_<lead>`` does.
POST_PREFIX = "p_post_"


def post_column(lead: int) -> str:
    """Column holding the post-processed probability at ``lead`` minutes."""
    return f"{POST_PREFIX}{int(lead)}"


#: The same name as a ``str.format`` template, for the readers that take
#: one (``benchmark_report --probability-column``,
#: ``threshold_sweep.SweepOptions.probability_column``). Derived from the
#: prefix so the writer and every reader cannot name three columns.
POST_COLUMN_TEMPLATE = POST_PREFIX + "{lead}"


#: What a model was fitted to predict — the outcome, not the design.
#:
#: ``wet`` is the historical target and Layer B's outcome: the gauge was
#: wet at some point inside ``(t, t + L]``. ``onset`` is the event the
#: push is GRADED on: a gauge onset (rain after ``dry_min`` dry minutes
#: that delivers ``onset_min_mm``, ``threshold_sweep.gauge_truth``) inside
#: the scorer's window ``(t, t + L + tolerance]``. The two differ most on
#: exactly the rows a push matters for — a point already under rain is
#: "wet" and is never an onset — so a model of one is not a model of the
#: other. Additive with the historical default: a document without
#: ``target`` is a ``wet`` model, which is what every one written before
#: the field existed was.
TARGET_WET = "wet"
TARGET_ONSET = "onset"
TARGETS: tuple[str, ...] = (TARGET_WET, TARGET_ONSET)

#: Prefix of the out-of-fold ONSET probability a ``--target onset`` fit
#: writes back — beside ``p_post_<lead>`` rather than over it, so a copy
#: can never be mistaken for the wet model's probability.
ONSET_PREFIX = "p_onset_"

#: ``str.format`` template of that column, for
#: ``benchmark_report --probability-column``.
ONSET_COLUMN_TEMPLATE = ONSET_PREFIX + "{lead}"


def target_column_template(target: str) -> str:
    """The write-back column template for a fit ``target``."""
    if str(target) == TARGET_ONSET:
        return ONSET_COLUMN_TEMPLATE
    if str(target) == TARGET_WET:
        return POST_COLUMN_TEMPLATE
    raise ValueError(
        f"unknown target {target!r}; expected one of {', '.join(TARGETS)}"
    )


#: Columns :func:`build_design` reads that the decision schema ALREADY
#: carries, so their presence in a file says nothing about whether the
#: writer computed features. Named once: the nightly fit and the threshold
#: sweep both have to answer "does this row have features?", and a second
#: opinion would silently change which rows are trained and scored on.
SHARED_SOURCE_COLUMNS: frozenset[str] = frozenset(
    {"observed_mm_h", "eta_min", "intensity_mm_h"}
)


def feature_source_columns(design_leads: Sequence[int]) -> list[str]:
    """Every stored column the design reads, in a stable order.

    ``hour_utc`` is absent on purpose: it is a function of the decision
    instant, which every loader already has, so deriving it keeps the read
    numeric-only. ``season`` comes from the same stamp. The writers still
    put both in the parquet for a human or a DuckDB query, and
    ``tests/test_postprocess.py`` pins the derivation against the column.
    """
    leads = sorted({int(x) for x in design_leads})
    names = [raw_fraction_column(lead) for lead in leads]
    # The per-lead ensemble-shape columns ride with the fraction they
    # describe: one lead set, read the same way, so a model fitted on
    # five design leads never scores on three filled columns and two nulls.
    names += [
        name
        for lead in leads
        for name in (ens_mean_column(lead), ens_p90_column(lead))
    ]
    names += [name for name in DESIGN_SOURCE_COLUMNS if name != "hour_utc"]
    return list(dict.fromkeys(names))


def feature_only_columns(design_leads: Sequence[int]) -> list[str]:
    """The design's source columns that exist ONLY when features were written.

    A row carrying none of these cannot be scored by the model at all: the
    nightly fit skips it and the threshold sweep excludes it, both
    counting what they dropped rather than imputing a whole row.
    """
    return [
        name for name in feature_source_columns(design_leads)
        if name not in SHARED_SOURCE_COLUMNS
    ]


def feature_schema(leads: Sequence[int]):
    """Arrow fields for the feature columns, in write order.

    Additive to ``warning_score.decision_schema`` and deliberately NOT
    part of it: ``align_decision_table`` conforms any file to the shared
    schema, so every existing consumer (the threshold sweep, both
    benchmark layers, the nightly fit) reads a run with features exactly
    as it reads one without and simply never sees these columns.

    Types mirror the decision schema's rules. Every numeric feature is
    nullable float32 and a null means "not computable at this point this
    cycle", never zero — a station off the composite has no upstream
    corridor, and 0 mm/h would be a claim that it is dry there.

    pyarrow is imported lazily for the same reason ``decision_schema``
    does it: the scoring half of this module must stay importable in an
    environment with no Arrow.
    """
    import pyarrow as pa

    fields = []
    for name, _definition in feature_columns(leads):
        if name == "season":
            fields.append((name, pa.string()))
        elif name == "hour_utc":
            fields.append((name, pa.int8()))
        else:
            fields.append((name, pa.float32()))
    return pa.schema(fields)


def post_schema(leads: Sequence[int]):
    """Arrow fields for the ``p_post_<lead>`` columns, in lead order.

    Nullable float32, like every other probability in a decision row: null
    means "the model could not speak for this row" — no fitted model, a
    lead it does not carry, or a point off coverage — and never 0 %.
    """
    import pyarrow as pa

    return pa.schema([
        (post_column(lead), pa.float32())
        for lead in sorted({int(x) for x in leads})
    ])


def feature_documentation(leads: Sequence[int]) -> dict[str, str]:
    """``{column: definition}`` — what goes in a run's ``summary.json``."""
    return dict(feature_columns(leads))


def finite_or_none(value: Any) -> float | None:
    """A feature value for parquet: non-finite becomes a null, never a 0."""
    if value is None:
        return None
    out = float(value)
    return out if math.isfinite(out) else None


@lru_cache(maxsize=16)
def _row_column_names(leads: tuple[int, ...]) -> tuple[str, ...]:
    """Feature column names in write order, memoised per lead set.

    :func:`feature_row` runs once per point per cycle — a few hundred
    thousand times in a month's replay — and the catalogue it walks is a
    constant for a given lead set. Cached on the leads tuple so the
    definitions are not rebuilt for every row.
    """
    return tuple(name for name, _definition in feature_columns(leads))


def feature_row(
    grid_features: Mapping[str, Any],
    index: int,
    *,
    raw_fractions: Mapping[int, Any],
    leads: Sequence[int],
    season: str,
    hour_utc: int,
    frame_age_min: float,
    station_radar_km: float,
) -> dict[str, Any]:
    """One point's feature columns for one cycle, in write order.

    The single assembler for both writers — the offline replay and the
    live cycle. Parity is the whole point: a model fitted on replay rows
    is applied to live rows, so identical inputs must produce an identical
    row, down to the key order. ``sidecar/tests/test_push_postprocess.py``
    pins the two against each other.

    ``grid_features`` is the per-point column table the caller assembled —
    :func:`station_features`' output, plus :func:`station_gauge_features`'
    and :func:`ensemble_point_features`' where the caller has them — and
    ``index`` selects this point's entry. A column the caller could not
    compute may simply be absent: the row still carries the key, with a
    ``None``, so every row this function writes has exactly the schema's
    columns in exactly the schema's order whatever the caller had.
    ``raw_fractions`` maps lead → the UNcalibrated ensemble fraction
    ALREADY read at this point's product pixel (``None`` off coverage):
    only the caller knows which pixel the calibrated probability came off,
    and the two have to be the same one.
    """
    caller_owned = {
        "season": season,
        "hour_utc": int(hour_utc),
        "frame_age_min": float(frame_age_min),
        "station_radar_km": float(station_radar_km),
    }
    row: dict[str, Any] = {}
    for name in _row_column_names(tuple(int(lead) for lead in leads)):
        if name in caller_owned:
            row[name] = caller_owned[name]
        elif name.startswith(RAW_FRACTION_PREFIX):
            row[name] = finite_or_none(
                raw_fractions.get(int(name[len(RAW_FRACTION_PREFIX):])),
            )
        else:
            values = grid_features.get(name)
            row[name] = None if values is None else finite_or_none(values[index])
    return row


# ---------------------------------------------------------------------------
# Season / hour, from the decision instant
# ---------------------------------------------------------------------------

#: The project-wide seasonal split. Restated rather than imported from
#: ``dmi_nowcast_sidecar.threshold_sweep`` so this module's import graph
#: stays inside the core package; ``tests/test_postprocess.py`` pins the
#: two against each other.
SEASON_MONTHS: dict[str, tuple[int, ...]] = {
    "summer": (5, 6, 7, 8, 9),
    "winter": (12, 1, 2, 3),
    "shoulder": (4, 10, 11),
}

#: Reporting order, matching ``benchmark_report.STRATA``'s tail.
SEASONS: tuple[str, ...] = ("summer", "winter", "shoulder")


def season_of_month(month: int) -> str:
    """``"summer"`` / ``"winter"`` / ``"shoulder"`` for a calendar month."""
    for name in SEASONS:
        if int(month) in SEASON_MONTHS[name]:
            return name
    raise ValueError(f"month {month} belongs to no season")


def _epoch_seconds(t: np.ndarray | Sequence[int]) -> np.ndarray:
    return np.asarray(t, dtype=np.int64)


def months_from_epoch(t: np.ndarray | Sequence[int]) -> np.ndarray:
    """Calendar month (1-12) of each UTC epoch-second stamp."""
    stamps = _epoch_seconds(t).astype("datetime64[s]")
    return (stamps.astype("datetime64[M]").astype(np.int64) % 12) + 1


def year_months_from_epoch(t: np.ndarray | Sequence[int]) -> np.ndarray:
    """``year * 12 + month - 1`` per row — the leave-one-month-out fold key.

    Folds are ``(year, month)`` pairs, not month numbers: once the archive
    passes a year there are two Decembers, and pooling them would train on
    the month being held out.
    """
    stamps = _epoch_seconds(t).astype("datetime64[s]")
    return stamps.astype("datetime64[M]").astype(np.int64) + 12 * 1970


def hours_from_epoch(t: np.ndarray | Sequence[int]) -> np.ndarray:
    """UTC hour (0-23) of each epoch-second stamp."""
    return (_epoch_seconds(t) // 3600) % 24


def seasons_from_epoch(t: np.ndarray | Sequence[int]) -> np.ndarray:
    """Season label per row, as a ``<U8`` array."""
    month = months_from_epoch(t)
    out = np.empty(month.shape, dtype="<U8")
    for name in SEASONS:
        out[np.isin(month, np.array(SEASON_MONTHS[name], dtype=np.int64))] = name
    return out


# ---------------------------------------------------------------------------
# Grid features: the disc and the upwind corridor
# ---------------------------------------------------------------------------

#: What counts as a wet pixel, mm/h. The production detection threshold
#: (``ForecastConfig.rain_threshold_mm_h``), so "upstream rain" here means
#: the same thing "rain" means everywhere else in the pipeline.
WET_MM_H = 0.5

#: Radius of the observed-maximum disc, km. Wider than the 1 km
#: ``raining_now`` disc on purpose: this feature is meant to see the
#: shower that is *about* to be overhead, not to decide whether it already
#: is.
OBS_DISC_KM = 5.0

#: The upwind corridor: 6 km wide (± 3 km across the flow), sampled to
#: 20 km and to 40 km along it. 6 km is about the width a shower has to
#: have to survive the trip; 40 km is 60-90 minutes of travel at Danish
#: storm speeds, so it covers every served lead.
UPSTREAM_HALF_WIDTH_KM = 3.0
UPSTREAM_NEAR_KM = 20.0
UPSTREAM_FAR_KM = 40.0

#: Corridor sampling step, km. One native pixel — finer would resample the
#: same pixels, coarser would step over a narrow band.
UPSTREAM_STEP_KM = 0.5

#: Width of one corridor bin, km (post-processing v2, F3). Eight of them
#: tile the 40 km corridor, so a bin is 7-12 minutes of travel at Danish
#: storm speeds — about the resolution the served leads can use.
UPSTREAM_BIN_KM = 5.0

#: Radii of the observed wet-fraction discs, km (v2, F2). The 5 km one
#: shares its radius with ``obs_max_5km_mm_h`` on purpose: the maximum and
#: the wet share of the SAME disc are what separate a clipping cell from a
#: band overhead.
WET_FRAC_DISC_KM: tuple[float, ...] = (5.0, 10.0)

#: Below this the flow carries no usable direction (0.1 px per frame is
#: 0.3 km/h on the 500 m grid at a 10-minute spacing). The station's own
#: flow is tried first, then the cycle's bulk motion; if both are under the
#: floor the corridor is undefined and every upstream feature is NaN.
MIN_FLOW_PX_PER_FRAME = 0.1


def _gather(
    field: np.ndarray, rows: np.ndarray, cols: np.ndarray,
) -> np.ndarray:
    """Nearest-pixel read of ``field`` at fractional indices; off-grid → NaN.

    ``rows``/``cols`` may have any shape as long as they broadcast
    together; the result has that shape. One fancy-index gather for every
    station and every offset at once — the whole point of doing the
    geometry this way rather than slicing a window per station.
    """
    height, width = field.shape
    r = np.rint(rows).astype(np.int64)
    c = np.rint(cols).astype(np.int64)
    inside = (r >= 0) & (r < height) & (c >= 0) & (c < width)
    out = np.full(np.broadcast(r, c).shape, np.nan, dtype=np.float32)
    safe_r = np.where(inside, r, 0)
    safe_c = np.where(inside, c, 0)
    values = field[safe_r, safe_c]
    np.copyto(out, values, where=inside)
    return out


def _nanmax(values: np.ndarray, axis) -> np.ndarray:
    """``np.nanmax`` that returns NaN for an all-NaN slice, silently."""
    with np.errstate(invalid="ignore"):
        valid = np.any(np.isfinite(values), axis=axis)
        filled = np.where(np.isfinite(values), values, -np.inf)
        peak = np.max(filled, axis=axis)
    return np.where(valid, peak, np.nan).astype(np.float32)


def _wet_fraction(values: np.ndarray, axis) -> np.ndarray:
    """Share of the FINITE samples at or above :data:`WET_MM_H`.

    NaN where nothing along ``axis`` is finite — an all-off-composite
    neighbourhood is unknown, never 0 % wet.
    """
    finite = np.isfinite(values)
    n_finite = finite.sum(axis=axis)
    with np.errstate(invalid="ignore"):
        wet = (finite & (values >= WET_MM_H)).sum(axis=axis)
        out = np.where(
            n_finite > 0, wet / np.maximum(n_finite, 1), np.nan,
        )
    return out.astype(np.float32)


def _disc_values(
    rain_mm_h: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    radius_km: float,
    pixel_km: float,
) -> np.ndarray:
    """Every pixel within ``radius_km`` of each point: ``(n_points, n_px)``.

    One gather for the whole point list and the whole disc, which is what
    keeps the per-cycle cost of a few hundred points in the milliseconds.
    Off-grid samples come back NaN (see :func:`_gather`), so a disc that
    hangs over the composite's edge is partly unknown rather than partly
    dry.
    """
    reach = int(math.ceil(float(radius_km) / float(pixel_km)))
    span = np.arange(-reach, reach + 1, dtype=np.float64)
    dy, dx = np.meshgrid(span, span, indexing="ij")
    keep = (dy ** 2 + dx ** 2) * (pixel_km ** 2) <= radius_km ** 2 + 1e-9
    dy, dx = dy[keep], dx[keep]
    return _gather(
        rain_mm_h,
        np.asarray(rows, dtype=np.float64)[:, None] + dy[None, :],
        np.asarray(cols, dtype=np.float64)[:, None] + dx[None, :],
    )


def disc_max(
    rain_mm_h: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    radius_km: float,
    pixel_km: float,
) -> np.ndarray:
    """Max rain rate within ``radius_km`` of each ``(row, col)``.

    Radius-based, not nearest-pixel: ~500 m pixels and fuzzy rain edges
    make a single pixel brittle, which is the same reason the Home
    Assistant detection rule reads a disc.
    """
    return _nanmax(
        _disc_values(
            rain_mm_h, rows, cols, radius_km=radius_km, pixel_km=pixel_km,
        ),
        axis=1,
    )


def _flow_direction(
    vy_local: np.ndarray,
    vx_local: np.ndarray,
    bulk_vy: float,
    bulk_vx: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(uy, ux, usable)`` — the unit vector the rain is travelling along.

    The station's own completed flow decides the corridor, because a
    corridor drawn along the national bulk would point the wrong way
    wherever the local flow does something else. Where the local vector is
    too short to carry a direction the bulk stands in; where the bulk is
    too short as well, the corridor has no meaning and the caller returns
    NaN rather than picking an arbitrary compass point.
    """
    speed = np.hypot(vy_local, vx_local)
    bulk_speed = float(math.hypot(bulk_vy, bulk_vx))
    use_bulk = speed < MIN_FLOW_PX_PER_FRAME
    vy = np.where(use_bulk, bulk_vy, vy_local)
    vx = np.where(use_bulk, bulk_vx, vx_local)
    length = np.hypot(vy, vx)
    usable = length >= MIN_FLOW_PX_PER_FRAME
    safe = np.where(usable, length, 1.0)
    return (vy / safe), (vx / safe), usable


def _upstream(
    rain_mm_h: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    uy: np.ndarray,
    ux: np.ndarray,
    usable: np.ndarray,
    *,
    pixel_km: float,
) -> dict[str, np.ndarray]:
    """The four corridor features, for every station in one gather.

    Geometry, stated once. ``(uy, ux)`` is the unit vector the rain moves
    along, so **upwind** is ``(-uy, -ux)`` and the corridor's cross-track
    unit is its perpendicular ``(-ux, uy)``. A sample at along-track ``s``
    and cross-track ``c`` kilometres therefore sits at

        dy = (-uy · s) + (-ux · c),    dx = (-ux · s) + (uy · c)

    kilometres from the station, which divided by the pixel size gives the
    row/column offsets gathered above.
    """
    s = np.arange(
        0.0, UPSTREAM_FAR_KM + UPSTREAM_STEP_KM / 2.0, UPSTREAM_STEP_KM,
    )
    c = np.arange(
        -UPSTREAM_HALF_WIDTH_KM,
        UPSTREAM_HALF_WIDTH_KM + UPSTREAM_STEP_KM / 2.0,
        UPSTREAM_STEP_KM,
    )
    # (n_stations, n_along, n_across)
    along_y = (-uy)[:, None, None] * s[None, :, None]
    along_x = (-ux)[:, None, None] * s[None, :, None]
    across_y = (-ux)[:, None, None] * c[None, None, :]
    across_x = (uy)[:, None, None] * c[None, None, :]
    dy = (along_y + across_y) / pixel_km
    dx = (along_x + across_x) / pixel_km
    values = _gather(
        rain_mm_h,
        np.asarray(rows, dtype=np.float64)[:, None, None] + dy,
        np.asarray(cols, dtype=np.float64)[:, None, None] + dx,
    )
    # A station with no usable direction has no corridor at all.
    values = np.where(usable[:, None, None], values, np.nan)

    near = s <= UPSTREAM_NEAR_KM + 1e-9
    finite = np.isfinite(values)
    wet = finite & (values >= WET_MM_H)
    n_finite = finite.sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        wet_frac = np.where(
            n_finite > 0, wet.sum(axis=(1, 2)) / np.maximum(n_finite, 1), np.nan,
        )
    wet_any = wet.any(axis=2)
    has_echo = wet_any.any(axis=1)
    first = np.argmax(wet_any, axis=1)
    with warnings.catch_warnings():
        # An all-NaN corridor legitimately has no mean.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        corridor_mean = np.nanmean(values, axis=(1, 2)).astype(np.float32)
    out = {
        "up_max_20km_mm_h": _nanmax(values[:, near, :], axis=(1, 2)),
        "up_max_40km_mm_h": _nanmax(values, axis=(1, 2)),
        "up_dist_km": np.where(has_echo, s[first], np.nan).astype(np.float32),
        "up_wet_frac_40km": wet_frac.astype(np.float32),
        "up_mean_40km_mm_h": corridor_mean,
    }
    # The same samples, resolved into 5 km bins: which bin holds the rain
    # is what separates "it is 5 km away" from "it is 35 km away" at the
    # same corridor maximum. ``s`` runs 0 … 40 inclusive, so the last
    # sample is clipped into the final bin rather than opening a ninth.
    bin_of = np.minimum(
        (s / UPSTREAM_BIN_KM).astype(np.int64), UPSTREAM_BINS - 1,
    )
    for index in range(UPSTREAM_BINS):
        out[upstream_bin_column(index)] = _nanmax(
            values[:, bin_of == index, :], axis=(1, 2),
        )
    return out


def station_features(
    rain_mm_h: np.ndarray,
    vy: np.ndarray,
    vx: np.ndarray,
    rows: Sequence[float],
    cols: Sequence[float],
    *,
    pixel_km: float,
    dt_min: float,
    bulk_vy: float,
    bulk_vx: float,
    stalled_share: float,
    rain_prev10_mm_h: np.ndarray | None = None,
    rain_prev20_mm_h: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Every grid-derived feature, for a whole station list at once.

    Parameters
    ----------
    rain_mm_h:
        Native-grid rain rate of the cycle's ANCHOR field — the very grid
        the flow and the ensemble were built from, NaN outside coverage.
    vy, vx:
        The COMPLETED flow in pixels per frame, i.e. what the forecast is
        advected with (``dense_flow.MotionEstimate.vy/vx``).
    rows, cols:
        Fractional native-grid indices of the stations
        (``CompositeGeo.lonlat_to_grid``). Off-grid stations come back all
        NaN rather than reading some other pixel.
    pixel_km, dt_min:
        The frame's own pixel size in km and the inter-frame spacing in
        minutes — never assumed, so a 5-minute product would convert
        correctly without a change here.
    bulk_vy, bulk_vx, stalled_share:
        The cycle's ``MotionEstimate`` diagnostics. ``stalled_share`` is
        one number per cycle; it is broadcast so a row carries the state
        of the flow it was decided on.
    rain_prev10_mm_h, rain_prev20_mm_h:
        The SAME field one and two frames earlier — the other two frames
        of the history triple the flow and the cascade ate, on the same
        grid and converted with the same Z-R parameters. Optional: a
        caller without them (the on-demand ``/forecast`` lookup) leaves
        the history columns out of the result entirely, and
        :func:`feature_row` writes them as nulls.

    Returns
    -------
    ``{column: float32 array of shape (n_stations,)}`` for every
    grid-derived feature name in the catalogue. The columns that do not
    come off the grid (``season``, ``hour_utc``, ``frame_age_min``,
    ``station_radar_km``, the ``g_*`` gauge block and the ``ens_*``
    ensemble block) belong to the caller, which is the only place that
    knows the decision instant, the station's coordinates, its gauge and
    the ensemble.
    """
    field = np.asarray(rain_mm_h, dtype=np.float32)
    if field.ndim != 2:
        raise ValueError(f"rain_mm_h must be 2-D, got {field.ndim}-D")
    if pixel_km <= 0:
        raise ValueError(f"pixel_km must be > 0, got {pixel_km}")
    if dt_min <= 0:
        raise ValueError(f"dt_min must be > 0, got {dt_min}")
    row = np.asarray(rows, dtype=np.float64)
    col = np.asarray(cols, dtype=np.float64)
    if row.shape != col.shape:
        raise ValueError("rows and cols must have the same shape")

    # px per frame → km/h: one pixel is pixel_km, one frame is dt_min.
    to_kmh = float(pixel_km) * 60.0 / float(dt_min)

    vy_local = _gather(np.asarray(vy, dtype=np.float32), row, col)
    vx_local = _gather(np.asarray(vx, dtype=np.float32), row, col)
    # An off-grid station has no local flow; the corridor then falls back
    # to the bulk, which is the honest answer for a point just outside the
    # composite's edge.
    vy_local = np.nan_to_num(vy_local, nan=0.0)
    vx_local = np.nan_to_num(vx_local, nan=0.0)
    uy, ux, usable = _flow_direction(vy_local, vx_local, bulk_vy, bulk_vx)

    bulk_speed = float(math.hypot(bulk_vy, bulk_vx))
    bulk_dir = (
        float(math.degrees(math.atan2(bulk_vx, -bulk_vy)) % 360.0)
        if bulk_speed >= MIN_FLOW_PX_PER_FRAME else float("nan")
    )
    n = int(row.size)
    # The 5 km disc is gathered ONCE and reduced twice: the maximum and
    # the wet share are the same pixels asked two questions.
    disc5 = _disc_values(
        field, row, col, radius_km=OBS_DISC_KM, pixel_km=pixel_km,
    )
    out: dict[str, np.ndarray] = {
        "obs_max_5km_mm_h": _nanmax(disc5, axis=1),
        "bulk_kmh": np.full(n, bulk_speed * to_kmh, dtype=np.float32),
        "bulk_dir_deg": np.full(n, bulk_dir, dtype=np.float32),
        "local_speed_kmh": (
            np.hypot(vy_local, vx_local) * to_kmh
        ).astype(np.float32),
        "stalled_share": np.full(n, float(stalled_share), dtype=np.float32),
        "wet_frac_5km": _wet_fraction(disc5, axis=1),
        "wet_frac_10km": _wet_fraction(
            _disc_values(
                field, row, col,
                radius_km=WET_FRAC_DISC_KM[1], pixel_km=pixel_km,
            ),
            axis=1,
        ),
    }
    del disc5
    out.update(_upstream(field, row, col, uy, ux, usable, pixel_km=pixel_km))

    # v2/F2 — the same reads on the previous frames. The trend at the
    # point is what separates a shower building over the station from one
    # that has already passed, and both read 'wet now'.
    if rain_prev10_mm_h is not None:
        previous = np.asarray(rain_prev10_mm_h, dtype=np.float32)
        if previous.shape != field.shape:
            raise ValueError(
                "rain_prev10_mm_h must have the anchor field's shape, got "
                f"{previous.shape} against {field.shape}",
            )
        out["obs_prev10_mm_h"] = _gather(previous, row, col)
        out["obs_max_5km_prev10_mm_h"] = disc_max(
            previous, row, col, radius_km=OBS_DISC_KM, pixel_km=pixel_km,
        )
    if rain_prev20_mm_h is not None:
        previous = np.asarray(rain_prev20_mm_h, dtype=np.float32)
        if previous.shape != field.shape:
            raise ValueError(
                "rain_prev20_mm_h must have the anchor field's shape, got "
                f"{previous.shape} against {field.shape}",
            )
        out["obs_prev20_mm_h"] = _gather(previous, row, col)
    return out


# ---------------------------------------------------------------------------
# The gauge block: what the station itself measured, as of what was visible
# ---------------------------------------------------------------------------

#: Default availability lag of a gauge slot, minutes (DECIDE-2, measured
#: 2026-09-16). DMI publishes a 10-minute slot ~1.5 min after it ends
#: (p50 1.5, p90 1.6, p99 21.6 over 15,184 slots at 105 DK stations), and
#: the live store polls every 10 minutes — so one poll interval is what a
#: cycle can actually count on having. The replay must apply the same lag
#: or it trains on rain the service will not have seen.
DEFAULT_GAUGE_LAG_MIN = 10.0

#: The ``g_mm_*`` windows, minutes back from the visibility horizon.
GAUGE_WINDOWS_MIN: tuple[int, ...] = (10, 30, 60)

#: Window of the dryness flag and of ``g_known``, minutes.
GAUGE_DRY_WINDOW_MIN = 60

#: Cap on ``g_min_since_wet``, minutes. Six hours: beyond that "it rained
#: a while ago" carries nothing the season and the hour do not already.
GAUGE_SINCE_CAP_MIN = 360.0


def _slot_utc(value: Any) -> datetime:
    """A slot end as an aware UTC datetime; naive input is a bug, not a zone."""
    if not isinstance(value, datetime):
        raise TypeError(f"slot end must be a datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("slot ends must be timezone-aware (UTC)")
    return value.astimezone(timezone.utc)


def station_gauge_features(
    slots: Sequence[Any],
    *,
    now_utc: datetime,
    lag_min: float = DEFAULT_GAUGE_LAG_MIN,
) -> dict[str, np.ndarray]:
    """The ``g_*`` block for a whole point list, from the gauge archive.

    ``slots[i]`` is what
    :func:`~dmi_nowcast_core.warning_score.gauge_slot_amounts` returns for
    point *i* — ``(slot_end, wet, mm)`` on a contiguous 10-minute grid,
    with the project's one wet rule already applied — or ``None`` for a
    point that is not a gauge at all (home, a subscriber). The single
    producer for both writers, exactly as :func:`station_features` is: the
    replay reads the corpus store, the cycle reads the store the service
    keeps, and from there the two run the same code.

    **Availability.** Only slots whose END is at or before
    ``now_utc - lag_min`` exist here — the *visibility horizon*. A model
    trained on the archive's full knowledge would learn to read rain the
    service has not been told about yet, and would be worth less in
    production than the number it replaced. The ``g_mm_*`` windows are
    measured back from that horizon rather than from the decision
    instant, because with the default 10-minute lag no slot at all ends
    inside the last 10 minutes and the column would be forever null;
    ``g_min_since_wet`` is measured from the decision instant, because
    "how long since it last rained here" is a physical age and the lag is
    part of it.

    **How far back the caller read does not change the answer**, as long
    as it reaches at least :data:`GAUGE_SINCE_CAP_MIN` behind the horizon:
    a wet slot older than the cap is capped to it, which is the same
    number a caller that never read it would produce. That is what lets
    the replay hand in a whole day of slots and the cycle hand in six
    hours, and still write the same row.

    Returns ``{column: float32 array of shape (n_points,)}`` with NaN for
    every unknown — except ``g_known``, which is 0 or 1 and never NaN: it
    is the indicator that tells the design whether the rest of the block
    is a measurement or an absence.
    """
    now = _slot_utc(now_utc)
    horizon = now - timedelta(minutes=float(lag_min))
    n = len(slots)
    out = {
        name: np.full(n, np.nan, dtype=np.float32)
        for name in (
            [f"g_mm_{window}" for window in GAUGE_WINDOWS_MIN]
            + ["g_min_since_wet", "g_dry_60"]
        )
    }
    out["g_known"] = np.zeros(n, dtype=np.float32)
    for index, series in enumerate(slots):
        if not series:
            continue
        # One coercion per slot: this runs for every point of every cycle
        # of every replayed day, and the series is a whole day long.
        visible = []
        for end, wet, mm in series:
            end = _slot_utc(end)
            if end <= horizon:
                visible.append((end, wet, mm))
        if not visible:
            continue
        known_any = False
        wet_in_dry_window = False
        known_in_dry_window = False
        since_min: float | None = None
        sums: dict[int, float | None] = {w: None for w in GAUGE_WINDOWS_MIN}
        for end, wet, mm in visible:
            back = (horizon - end).total_seconds() / 60.0
            if wet is not None:
                known_any = True
                if back < GAUGE_DRY_WINDOW_MIN:
                    known_in_dry_window = True
                    wet_in_dry_window = wet_in_dry_window or bool(wet)
                if wet:
                    age = (now - end).total_seconds() / 60.0
                    if since_min is None or age < since_min:
                        since_min = age
            if mm is not None:
                for window in GAUGE_WINDOWS_MIN:
                    if back < window:
                        sums[window] = (sums[window] or 0.0) + float(mm)
        if not known_any:
            continue
        for window in GAUGE_WINDOWS_MIN:
            if sums[window] is not None:
                out[f"g_mm_{window}"][index] = sums[window]
        out["g_min_since_wet"][index] = (
            GAUGE_SINCE_CAP_MIN if since_min is None
            else min(float(since_min), GAUGE_SINCE_CAP_MIN)
        )
        if known_in_dry_window:
            out["g_dry_60"][index] = 0.0 if wet_in_dry_window else 1.0
            out["g_known"][index] = 1.0
    return out



@dataclass(frozen=True)
class GaugeSlotTable:
    """``slots_by_station`` digested once into parallel numpy arrays.

    The live cycle calls :func:`neighbour_gauge_features` once per radar
    frame and can hand it the mapping directly. The offline builder calls
    it once per decision instant — ~145 instants for a replayed day, each
    over the same ~100 stations and the same ~180 slots — and re-walking
    the Python lists that many times costs more than the features do. Same
    producer, same numbers; this is only the caller's choice about when the
    coercion happens.

    ``ends`` is the union of every station's slot ends as epoch seconds,
    ascending. ``wet`` is 1 / 0 / NaN for wet / dry / not reported, and
    ``mm`` is NaN where the station reported no amount — the two distinct
    absences :func:`~dmi_nowcast_core.warning_score.gauge_slot_amounts`
    is careful to keep apart.
    """

    stations: tuple[str, ...]
    ends: np.ndarray
    wet: np.ndarray
    mm: np.ndarray

    @classmethod
    def from_slots(
        cls, slots_by_station: Mapping[str, Sequence[Any]],
    ) -> "GaugeSlotTable":
        stations = tuple(str(sid) for sid in slots_by_station)
        moments: set[int] = set()
        per_station: list[list[tuple[int, Any, Any]]] = []
        for sid in stations:
            series = slots_by_station[sid] or ()
            rows = [
                (int(_slot_utc(end).timestamp()), wet, mm)
                for end, wet, mm in series
            ]
            per_station.append(rows)
            moments.update(row[0] for row in rows)
        ends = np.array(sorted(moments), dtype=np.int64)
        index = {value: position for position, value in enumerate(ends.tolist())}
        shape = (len(stations), ends.size)
        wet = np.full(shape, np.nan, dtype=np.float32)
        mm = np.full(shape, np.nan, dtype=np.float32)
        for row, series in enumerate(per_station):
            for end, is_wet, amount in series:
                column = index[end]
                if is_wet is not None:
                    wet[row, column] = 1.0 if is_wet else 0.0
                if amount is not None:
                    mm[row, column] = float(amount)
        return cls(stations=stations, ends=ends, wet=wet, mm=mm)

    def stats(self, now_utc: datetime, lag_min: float) -> dict[str, np.ndarray]:
        """Per-station scalars as of one decision instant.

        The same availability rule :func:`station_gauge_features` applies,
        applied to everyone at once: only slots that ENDED at or before
        ``now - lag`` exist, the millimetre windows are measured back from
        that horizon, and ``min_since_wet`` is measured from the decision
        instant because "how long since it last rained there" is a physical
        age that the lag is part of.
        """
        now = _slot_utc(now_utc)
        now_s = now.timestamp()
        horizon_s = now_s - float(lag_min) * 60.0
        visible = self.ends <= horizon_s
        known = np.isfinite(self.wet) & visible[None, :]
        is_wet = known & (self.wet > 0.5)
        recent = visible & (self.ends > horizon_s - NG_WET_WINDOW_MIN * 60.0)
        near = visible & (self.ends > horizon_s - NG_NEAR_WINDOW_MIN * 60.0)

        def _sum(window: np.ndarray) -> np.ndarray:
            block = np.isfinite(self.mm) & window[None, :]
            total = np.where(block, np.nan_to_num(self.mm, nan=0.0), 0.0).sum(1)
            return np.where(block.any(1), total, np.nan)

        age = (now_s - self.ends.astype(np.float64)) / 60.0
        since = np.where(
            is_wet, age[None, :], np.inf,
        ).min(1) if self.ends.size else np.full(len(self.stations), np.inf)
        known_any = known.any(1)
        return {
            "known_30": (known & recent[None, :]).any(1),
            "wet_30": (is_wet & recent[None, :]).any(1),
            "mm_30": _sum(recent),
            "mm_60": _sum(near),
            "min_since_wet": np.where(
                known_any,
                np.minimum(np.where(np.isfinite(since), since, GAUGE_SINCE_CAP_MIN),
                           GAUGE_SINCE_CAP_MIN),
                np.nan,
            ),
        }


def _ng_columns() -> tuple[str, ...]:
    return tuple(name for name, _definition in SCALAR_FEATURE_COLUMNS_NG)


def neighbour_gauge_features(
    points: Sequence[tuple[float, float]],
    slots_by_station: Mapping[str, Sequence[Any]] | GaugeSlotTable,
    station_coords: Mapping[str, tuple[float, float]],
    *,
    now_utc: datetime,
    bulk_kmh: float | None,
    bulk_dir_deg: float | None,
    lag_min: float = DEFAULT_GAUGE_LAG_MIN,
    exclude_self: Sequence[str | None] | None = None,
) -> dict[str, np.ndarray]:
    """The ``ng_*`` block for a whole point list, in the motion's frame.

    ONE producer, for the offline builder now
    (``scripts/add_neighbour_gauge_features.py``) and the live cycle later,
    exactly as :func:`station_gauge_features` is: the inputs are what
    :class:`~dmi_nowcast_sidecar.gauge_history.GaugeHistory` already has in
    hand at the end of its one read per cycle — the ``{station_id:
    [(slot_end, wet, mm), ...]}`` dict its own
    :func:`~dmi_nowcast_sidecar.gauge_history.slots_by_station` builds, and
    the coordinates from the station points file it already parses.

    ``points`` is ``(lat, lon)`` per point, in degrees — a place, which may
    or may not have a gauge standing on it. ``exclude_self[i]`` is point
    *i*'s own station id when the caller knows it (``None`` otherwise); a
    gauge within :data:`NG_SELF_KM` of the point is dropped regardless, so
    a caller that cannot name the point still cannot read the point's own
    gauge back as its neighbour.

    ``bulk_kmh`` / ``bulk_dir_deg`` are the cycle's bulk storm motion — the
    ``bulk_kmh`` and ``bulk_dir_deg`` feature columns, one pair per
    decision instant. The bearing is the one this module uses everywhere:
    the compass direction the rain is heading TOWARD, 0 = north, 90 = east.
    So a gauge is UPSTREAM of a point when it lies in the direction the
    rain is coming FROM — opposite the bearing — and ``a``, the along-track
    distance, is positive there. Denmark under westerlies means the
    upstream gauges are usually the western ones; the sign is pinned by a
    test rather than by this sentence.

    Returns ``{column: float32 array of shape (n_points,)}`` with NaN for
    every unknown, except ``ng_frame_ok`` and ``ng_count_20km``, which are
    counts and never null.
    """
    table = (
        slots_by_station if isinstance(slots_by_station, GaugeSlotTable)
        else GaugeSlotTable.from_slots(slots_by_station)
    )
    names = _ng_columns()
    n = len(points)
    out = {name: np.full(n, np.nan, dtype=np.float32) for name in names}

    speed = float(bulk_kmh) if bulk_kmh is not None else float("nan")
    bearing = float(bulk_dir_deg) if bulk_dir_deg is not None else float("nan")
    frame_ok = bool(
        math.isfinite(speed) and math.isfinite(bearing)
        and speed >= NG_MIN_SPEED_KMH
    )
    out["ng_frame_ok"] = np.full(n, 1.0 if frame_ok else 0.0, dtype=np.float32)
    # The counts start at a real zero rather than a null: "no gauge is due
    # to reach this point in that window" is an answer. The upstream ones
    # only when there is a frame to count in.
    out["ng_count_20km"] = np.zeros(n, dtype=np.float32)
    if frame_ok:
        for edge in NG_TAU_EDGES_MIN:
            out[f"ng_up_count_{ng_tau_suffix(edge)}"] = np.zeros(
                n, dtype=np.float32,
            )
    if not n:
        return out

    # Only the gauges this caller gave a coordinate for: a station in the
    # slot table with no place to put it is not a neighbour of anywhere.
    positions = {sid: index for index, sid in enumerate(table.stations)}
    known = [sid for sid in table.stations if sid in station_coords]
    if not known:
        return out
    stats = table.stats(now_utc, lag_min)
    order = [positions[sid] for sid in known]
    gauge_lat = np.array(
        [float(station_coords[sid][0]) for sid in known], dtype=np.float64,
    )
    gauge_lon = np.array(
        [float(station_coords[sid][1]) for sid in known], dtype=np.float64,
    )
    known_30 = np.asarray(stats["known_30"])[order]
    wet_30 = np.asarray(stats["wet_30"])[order]
    mm_30 = np.asarray(stats["mm_30"], dtype=np.float64)[order]
    mm_60 = np.asarray(stats["mm_60"], dtype=np.float64)[order]
    since = np.asarray(stats["min_since_wet"], dtype=np.float64)[order]

    coordinates = np.asarray(points, dtype=np.float64).reshape(n, 2)
    # Equirectangular kilometres at NG_REF_LAT_DEG — see the constant.
    north = (gauge_lat[None, :] - coordinates[:, :1]) * KM_PER_DEG_LAT
    east = (gauge_lon[None, :] - coordinates[:, 1:2]) * KM_PER_DEG_LON
    distance = np.hypot(north, east)

    other = distance > NG_SELF_KM
    if exclude_self is not None:
        if len(exclude_self) != n:
            raise ValueError("exclude_self must have one entry per point")
        by_id = {sid: index for index, sid in enumerate(known)}
        for row, sid in enumerate(exclude_self):
            column = by_id.get(str(sid)) if sid is not None else None
            if column is not None:
                other[row, column] = False

    # -- the vicinity block: no direction, so it survives a dead flow ------
    vicinity = other & (distance <= NG_VICINITY_KM) & known_30[None, :]
    count_20 = vicinity.sum(1)
    out["ng_count_20km"] = count_20.astype(np.float32)
    with np.errstate(invalid="ignore"):
        share = np.where(
            count_20 > 0,
            (vicinity & wet_30[None, :]).sum(1) / np.maximum(count_20, 1),
            np.nan,
        )
    out["ng_wet_share_20km"] = share.astype(np.float32)

    nearest = np.where(other, distance, np.inf)
    pick = nearest.argmin(1)
    has_neighbour = np.isfinite(nearest[np.arange(n), pick])
    out["ng_near_km"] = np.where(
        has_neighbour, distance[np.arange(n), pick], np.nan,
    ).astype(np.float32)
    out["ng_near_mm_60"] = np.where(
        has_neighbour, mm_60[pick], np.nan,
    ).astype(np.float32)
    out["ng_near_min_since_wet"] = np.where(
        has_neighbour, since[pick], np.nan,
    ).astype(np.float32)

    if not frame_ok:
        return out

    # -- the motion frame --------------------------------------------------
    heading = math.radians(bearing)
    d_east, d_north = math.sin(heading), math.cos(heading)
    # Positive = UPSTREAM: the gauge lies opposite the heading, so the rain
    # over it is travelling toward this point.
    along = -(east * d_east + north * d_north)
    # Positive to the RIGHT of the heading. Only the magnitude is used; the
    # sign is kept so a reader of a debug dump can tell the sides apart.
    cross = east * d_north - north * d_east
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = 60.0 * along / speed

    candidate = (
        other
        & (distance <= NG_RADIUS_KM)
        & (along > 0.0)
        & (np.abs(cross) <= NG_CROSS_KM)
        & (tau <= float(NG_TAU_EDGES_MIN[-1]))
        & known_30[None, :]
    )
    weight = 1.0 / (1.0 + np.abs(cross) / NG_CROSS_WEIGHT_KM)
    has_mm = np.isfinite(mm_30)[None, :]
    amounts = np.where(has_mm, np.nan_to_num(mm_30, nan=0.0)[None, :], 0.0)

    for index, edge in enumerate(NG_TAU_EDGES_MIN):
        lower, upper = _ng_bin_range(index)
        selected = candidate & (tau > lower) & (tau <= upper)
        suffix = ng_tau_suffix(edge)
        count = selected.sum(1)
        out[f"ng_up_count_{suffix}"] = count.astype(np.float32)
        with_mm = selected & has_mm
        any_mm = with_mm.any(1)
        out[f"ng_up_mm_max_{suffix}"] = np.where(
            any_mm, np.where(with_mm, amounts, -np.inf).max(1), np.nan,
        ).astype(np.float32)
        weights = np.where(with_mm, weight, 0.0)
        total = weights.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"ng_up_mm_wmean_{suffix}"] = np.where(
                total > 0.0, (weights * amounts).sum(1) / np.maximum(total, 1e-12),
                np.nan,
            ).astype(np.float32)
        out[f"ng_up_wet_share_{suffix}"] = np.where(
            count > 0,
            (selected & wet_30[None, :]).sum(1) / np.maximum(count, 1),
            np.nan,
        ).astype(np.float32)

    # -- the nearest upstream gauge that is actually wet --------------------
    wet_candidate = candidate & wet_30[None, :]
    ranked = np.where(wet_candidate, tau, np.inf)
    chosen = ranked.argmin(1)
    rows = np.arange(n)
    found = np.isfinite(ranked[rows, chosen])
    out["ng_upwet_tau_min"] = np.where(
        found, tau[rows, chosen], np.nan,
    ).astype(np.float32)
    out["ng_upwet_cross_km"] = np.where(
        found, np.abs(cross[rows, chosen]), np.nan,
    ).astype(np.float32)
    out["ng_upwet_mm_30"] = np.where(found, mm_30[chosen], np.nan).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# The ensemble block: more than one number per lead
# ---------------------------------------------------------------------------


def ensemble_point_features(
    ensemble: np.ndarray,
    pixels: Sequence[Any],
    *,
    leads_min: Sequence[int],
    threshold_mm_h: float,
    timestep_min: float,
    frame_age_min: float,
) -> dict[str, np.ndarray]:
    """``ens_mean_<lead>`` / ``ens_p90_<lead>`` / ``ens_eta_spread_min``.

    ``ensemble`` is the ``(n_members, n_timesteps, h, w)`` mm/h array
    :func:`~dmi_nowcast_core.probabilistic.run_ensemble` returns, and
    ``pixels[i]`` is point *i*'s ``(row, col)`` on that grid (``None`` off
    it) — the SAME pixel ``raw_frac_<lead>`` was read at, so the fraction
    and its shape describe one place.

    The ensemble is touched exactly once, by one fancy-index gather of the
    points' own columns: ``(n_members, n_timesteps, n_points)`` is a few
    thousand floats where the array itself is ~150 MB, and every reduction
    below runs on the small one. That is the whole reason this lives here
    rather than beside ``national_products``, which would have to build a
    cumulative-maximum grid the size of the ensemble to answer the same
    question.

    Conventions, all shared with ``national_products`` so a feature and
    the grid beside it cannot disagree: NaN is outside coverage and never
    compares True against the threshold; a lead picks its timestep through
    ``_steps_in_lead`` (effective lead = nominal + frame age); and an
    arrival time is ``(step + 1) * timestep_min - frame_age_min``, clamped
    at zero — minutes from NOW, exactly like ``eta_min``.
    """
    forecast = np.asarray(ensemble)
    if forecast.ndim != 4:
        raise ValueError(
            "ensemble must be (n_members, n_timesteps, h, w); got shape "
            f"{forecast.shape}",
        )
    if timestep_min <= 0:
        raise ValueError(f"timestep_min must be > 0, got {timestep_min}")
    leads = sorted({int(lead) for lead in leads_min})
    n_timesteps = int(forecast.shape[1])
    n = len(pixels)
    out: dict[str, np.ndarray] = {}
    for lead in leads:
        out[ens_mean_column(lead)] = np.full(n, np.nan, dtype=np.float32)
        out[ens_p90_column(lead)] = np.full(n, np.nan, dtype=np.float32)
    out["ens_eta_spread_min"] = np.full(n, np.nan, dtype=np.float32)
    inside = np.array([pixel is not None for pixel in pixels], dtype=bool)
    if not inside.any():
        return out
    rows = np.array(
        [0 if pixel is None else int(pixel[0]) for pixel in pixels],
        dtype=np.int64,
    )
    cols = np.array(
        [0 if pixel is None else int(pixel[1]) for pixel in pixels],
        dtype=np.int64,
    )
    # (n_members, n_timesteps, n_points) — the only read of the ensemble.
    series = np.asarray(
        forecast[:, :, rows, cols], dtype=np.float32,
    )
    series[:, :, ~inside] = np.nan
    # Cumulative maximum along time, NaN-skipping: ``np.maximum`` would
    # poison a member's whole tail on one missing timestep, ``np.fmax``
    # carries the largest value seen so far and only stays NaN where
    # nothing has been seen at all.
    cumulative = np.fmax.accumulate(series, axis=1)
    with warnings.catch_warnings():
        # All-NaN member slices legitimately yield NaN.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for lead in leads:
            k = _steps_in_lead(
                lead, float(frame_age_min), float(timestep_min), n_timesteps,
            )
            at_lead = cumulative[:, k - 1, :]
            out[ens_mean_column(lead)] = np.nanmean(
                at_lead, axis=0,
            ).astype(np.float32)
            out[ens_p90_column(lead)] = np.nanpercentile(
                at_lead, 90.0, axis=0,
            ).astype(np.float32)
        # Arrival time per member: the first timestep whose RAW rate
        # crosses the threshold. NaN >= threshold is False, so a member
        # that never arrives (or never had data) simply has none.
        with np.errstate(invalid="ignore"):
            exceed = series >= float(threshold_mm_h)
        arrives = exceed.any(axis=1)
        first = np.argmax(exceed, axis=1)
        arrival = (first + 1).astype(np.float32) * np.float32(timestep_min)
        arrival = np.maximum(
            arrival - np.float32(frame_age_min), np.float32(0.0),
        )
        arrival = np.where(arrives, arrival, np.nan)
        enough = arrives.sum(axis=0) >= 4
        if enough.any():
            spread = np.nanpercentile(
                arrival, 75.0, axis=0,
            ) - np.nanpercentile(arrival, 25.0, axis=0)
            out["ens_eta_spread_min"] = np.where(
                enough & inside, spread, np.nan,
            ).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# The design matrix
# ---------------------------------------------------------------------------

#: ETA is NaN when no member brings rain inside the forecast horizon. The
#: model gets an indicator plus this capped value rather than a sentinel
#: that a linear term would read as "very late but still coming".
ETA_CAP_MIN = 90.0

#: Cap on the upwind distance, km: the corridor's own length, so "no echo
#: in 40 km" and "echo at exactly 40 km" differ only by the indicator.
UP_DIST_CAP_KM = UPSTREAM_FAR_KM


#: The design the shipped model was fitted on: the 27 columns of
#: :func:`_v1_design_columns` and nothing else. Named so an ablation can
#: always get back to it, and so "v1 is unchanged" is a testable claim.
DESIGN_V1 = "v1"

#: v1, plus every scalar feature column the catalogue has grown since,
#: plus a natural-cubic-spline basis on the columns whose effect is
#: visibly not linear, plus the hand-picked interactions. Resolved at fit
#: time — see :func:`design_spec`.
DESIGN_V2 = "v2"

DESIGN_VERSIONS: tuple[str, ...] = (DESIGN_V1, DESIGN_V2)

#: Interior knots in each spline basis. Three: enough to bend twice, few
#: enough that 450 000 rows pin every coefficient.
SPLINE_INTERIOR_KNOTS = 3

#: Quantiles the five knots (two boundary, three interior) sit at —
#: Harrell's placement for a five-knot natural spline. The boundary knots
#: are inside the range on purpose: a knot at the extreme sample leaves
#: the outermost basis function standing on one row.
SPLINE_QUANTILES: tuple[float, ...] = (0.05, 0.275, 0.5, 0.725, 0.95)

#: Design columns v2 gives a spline basis to, on top of their linear term.
#: ``raw_frac_<lead>`` is added per design lead by :func:`design_spec`.
#:
#: * ``log1p_up_dist_km`` — distance bites hard near the station and is
#:   flat past 25 km, which one slope cannot be both.
#: * ``bulk_kmh`` — slow motion means a stalling or decaying field, fast
#:   motion means the cell has come and gone before the horizon; both ends
#:   lower the probability and a linear term has to pick one.
#: * ``raw_frac_<L>`` — the fraction saturates, which is the defect this
#:   whole module exists for; a spline lets the fit spend resolution
#:   inside the top bin.
SPLINE_BASE_COLUMNS: tuple[str, ...] = ("log1p_up_dist_km", "bulk_kmh")

#: Names of the columns carrying the lead being predicted, in a model
#: whose one fit answers for every lead (``--model logistic-shared`` /
#: ``trees-shared``). Both, because they serve different arms: a tree
#: splits on the raw minutes and the split is readable, while a logistic
#: wants the log — skill falls off roughly with the log of the horizon,
#: not with the horizon. A tree given both simply ignores one.
LEAD_COLUMN = "lead_min"
LOG_LEAD_COLUMN = "log_lead_min"
LEAD_COLUMNS: tuple[str, ...] = (LEAD_COLUMN, LOG_LEAD_COLUMN)

#: Suffix of the "this row's own lead" columns in a shared design.
OWN_SUFFIX = "own"

#: Prefixes of the per-lead stored columns a shared design collapses into
#: one column apiece. The row for lead L carries ``raw_frac_L`` in
#: ``raw_frac_own``, ``ens_mean_L`` in ``ens_mean_own``, and so on.
#:
#: This is what makes a shared fit worth doing at all. Without it the one
#: coefficient vector would have to read every lead's ensemble fraction
#: for every lead's row and work out from ``lead_min`` which of the five
#: is the one that matters; with it, "the ensemble fraction at the
#: horizon I am being asked about" is a single predictor with a single
#: coefficient, estimated on four times the rows.
#:
#: Discovered by prefix from the design's own column names, not listed by
#: hand: the ensemble block is the feature side's to grow.
OWN_COLUMN_PREFIXES: tuple[str, ...] = (RAW_FRACTION_PREFIX, "ens_")


def own_column(prefix: str) -> str:
    """``"raw_frac_"`` → ``"raw_frac_own"``."""
    return f"{prefix}{OWN_SUFFIX}" if prefix.endswith("_") else f"{prefix}_{OWN_SUFFIX}"

#: Stored feature columns the v1 design already reads explicitly, so
#: :func:`design_spec` can tell which catalogue entries are *new*.
#: ``season`` and ``hour_utc`` are derived from the decision instant
#: rather than read as features; ``raw_frac_<lead>`` is handled per lead.
_V1_HANDLED: frozenset[str] = frozenset({
    "observed_mm_h", "obs_max_5km_mm_h", "up_max_20km_mm_h",
    "up_max_40km_mm_h", "up_dist_km", "up_wet_frac_40km", "eta_min",
    "intensity_mm_h", "bulk_kmh", "bulk_dir_deg", "local_speed_kmh",
    "stalled_share", "frame_age_min", "station_radar_km", "hour_utc",
    "season",
})

#: Suffixes that say outright that a catalogue column is a rate or a
#: distance. A hint, not the rule: the catalogue also holds rain rates
#: whose names do not say so (``ens_mean_20``, ``up_max_b0``), and v2
#: discovers its columns at run time, so the decision below is made from
#: the TRAINING VALUES and only then stored.
_LOG1P_SUFFIXES: tuple[str, ...] = ("_mm_h", "_km")

#: A non-negative column whose 99.9th percentile is more than this many
#: times its 90th is heavy-tailed enough that a linear term would let its
#: top percentile write the coefficient — a 60 mm/h hail core outvoting a
#: thousand drizzle rows. Ten is a wide margin: a fraction in [0, 1] is
#: nowhere near it, and every rain rate in the catalogue clears it easily.
LOG1P_TAIL_RATIO = 10.0


@dataclass(frozen=True)
class DesignSpec:
    """Everything beyond ``design_leads`` that decides the design matrix.

    Serialised into ``postprocess.json`` and replayed verbatim when the
    service scores a row, which is the whole reason it exists: the v2
    design reads the feature catalogue **at fit time**, so a column the
    replay grew last week is in this week's model without anyone editing a
    list here — and the model must then keep scoring the columns it was
    fitted on even after the catalogue grows again. Fit time resolves,
    serving time replays.

    The default spec IS the v1 design, so ``--design v1`` is a real
    control arm rather than a re-implementation of one.
    """

    version: str = DESIGN_V1
    #: Stored scalar feature columns beyond the ones v1 names, in
    #: catalogue order, resolved when the model was fitted.
    extra_columns: tuple[str, ...] = ()
    #: Which of ``extra_columns`` go through ``log1p`` — decided from the
    #: TRAINING values (:func:`wants_log1p`) and stored, because a
    #: transform re-derived at serving time would move under a fitted
    #: coefficient. These are the columns whose design name is
    #: ``log1p_<column>``.
    log_columns: tuple[str, ...] = ()
    #: ``(design column, knots)`` per spline basis. The knots are stored
    #: because they are training quantiles: re-deriving them at serving
    #: time from whatever rows are in hand would silently change the basis
    #: under a fitted coefficient.
    splines: tuple[tuple[str, tuple[float, ...]], ...] = ()
    #: ``(a, b)`` pairs of base design columns multiplied together.
    interactions: tuple[tuple[str, str], ...] = ()
    #: Column carrying the lead being predicted, for one fit shared
    #: across leads. None everywhere else, and its presence is what makes
    #: the design a shared one.
    lead_column: str | None = None
    #: ``(own column, {lead: source column})`` per per-lead family that a
    #: shared design collapses. Resolved at fit time from the design's own
    #: columns, stored so serving replays the same selection.
    own_columns: tuple[tuple[str, tuple[tuple[int, str], ...]], ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "extra_columns": list(self.extra_columns),
            "log_columns": list(self.log_columns),
            "splines": [
                {"column": name, "knots": list(knots)}
                for name, knots in self.splines
            ],
            "interactions": [list(pair) for pair in self.interactions],
            "lead_column": self.lead_column,
            "own_columns": [
                {"column": name, "sources": {str(lead): source
                                             for lead, source in sources}}
                for name, sources in self.own_columns
            ],
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any] | None) -> "DesignSpec":
        if not raw:
            return cls()
        return cls(
            version=str(raw.get("version", DESIGN_V1)),
            extra_columns=tuple(str(v) for v in raw.get("extra_columns") or ()),
            log_columns=tuple(str(v) for v in raw.get("log_columns") or ()),
            splines=tuple(
                (str(entry["column"]), tuple(float(k) for k in entry["knots"]))
                for entry in raw.get("splines") or ()
            ),
            interactions=tuple(
                (str(pair[0]), str(pair[1]))
                for pair in raw.get("interactions") or ()
            ),
            lead_column=(
                None if raw.get("lead_column") in (None, "")
                else str(raw["lead_column"])
            ),
            own_columns=tuple(
                (
                    str(entry["column"]),
                    tuple(
                        (int(lead), str(source))
                        for lead, source in sorted(
                            (entry.get("sources") or {}).items(),
                            key=lambda item: int(item[0]),
                        )
                    ),
                )
                for entry in raw.get("own_columns") or ()
            ),
        )


def _extra_design_name(column: str, logged: bool) -> str:
    """The design column name a stored catalogue column contributes."""
    return f"log1p_{column}" if logged else column


def wants_log1p(column: str, values: np.ndarray | None = None) -> bool:
    """Should this catalogue column go through ``log1p`` in the design?

    Yes when the name says it is a rate or a distance, and yes when the
    training values behave like one: non-negative, and with a tail more
    than :data:`LOG1P_TAIL_RATIO` times the 90th percentile. The second
    test is what keeps this honest as the catalogue grows — the ensemble
    and upwind-bin columns are rain rates in mm/h whose names never say
    so, and a v2 design that treated them linearly would be undoing the
    one transform the v1 design was careful to apply.

    Decided once, at fit time, and stored in :class:`DesignSpec`. Deciding
    it again at serving time from whatever rows are in hand would change
    the transform under a fitted coefficient.
    """
    if column.endswith(_LOG1P_SUFFIXES):
        return True
    if values is None:
        return False
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size < 100 or finite.min() < 0.0:
        return False
    p90, tail = np.quantile(finite, [0.90, 0.999])
    if tail <= 0.0:
        return False
    return bool(tail > LOG1P_TAIL_RATIO * p90) if p90 > 0.0 else True


def spline_knots(
    values: np.ndarray, *, n_interior: int = SPLINE_INTERIOR_KNOTS,
) -> tuple[float, ...]:
    """Knot positions for one column, at the training quantiles.

    ``()`` when the column cannot carry a spline — too few rows, all
    missing, or too few distinct values to place the knots apart. The
    caller then drops the basis rather than fitting one on a column that
    is effectively constant.
    """
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size < 100:
        return ()
    quantiles = np.asarray(SPLINE_QUANTILES, dtype=np.float64)
    if int(n_interior) != SPLINE_INTERIOR_KNOTS:
        inner = np.linspace(0.0, 1.0, int(n_interior) + 2)[1:-1]
        quantiles = np.r_[0.05, 0.05 + 0.90 * inner, 0.95]
    knots = np.unique(np.quantile(finite, quantiles))
    if knots.size < 3:
        return ()
    return tuple(float(v) for v in knots)


def natural_spline_basis(
    values: np.ndarray, knots: Sequence[float],
) -> np.ndarray:
    """``(n, len(knots) - 2)`` natural cubic spline basis, minus its linear term.

    Hastie, Tibshirani & Friedman, *ESL* §5.2.1: with knots
    ``ξ₁ < … < ξ_K`` the natural cubic spline basis is ``{1, X, N₃, …,
    N_K}``, where ``dₖ(X) = ((X − ξₖ)₊³ − (X − ξ_K)₊³) / (ξ_K − ξₖ)`` and
    ``N_{k+2} = dₖ(X) − d_{K−1}(X)``.

    The constant is the intercept and ``X`` is already a design column, so
    only the ``K − 2`` nonlinear columns come back: v2 **adds** to v1
    rather than replacing anything, which is what makes the two designs
    nest and keeps the v1 coefficients readable in the report.

    A NaN in, a NaN out — :meth:`Standardiser.transform` imputes it with
    the training mean like any other missing value, and a tree reads the
    missing directly.
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    xi = np.asarray(knots, dtype=np.float64)
    if xi.size < 3:
        return np.empty((x.size, 0), dtype=np.float64)
    last, before_last = float(xi[-1]), float(xi[-2])

    def d(k: float) -> np.ndarray:
        return (
            np.clip(x - k, 0.0, None) ** 3 - np.clip(x - last, 0.0, None) ** 3
        ) / (last - k)

    tail = d(before_last)
    out = np.empty((x.size, xi.size - 2), dtype=np.float64)
    for index in range(xi.size - 2):
        out[:, index] = d(float(xi[index])) - tail
    out[~np.isfinite(x)] = np.nan
    return out


def _spline_name(column: str, index: int) -> str:
    return f"{column}_ns{index + 1}"


def _interaction_name(a: str, b: str) -> str:
    return f"{a}_x_{b}"


def design_spec(
    features: Mapping[str, Any] | None,
    leads: Sequence[int],
    *,
    version: str = DESIGN_V1,
    shared_lead: bool = False,
) -> DesignSpec:
    """Resolve the design for one fit, reading the catalogue at run time.

    ``features`` is the training rows, used only to place the spline knots
    at their quantiles; ``None`` gives a spec with no splines, which is
    what a caller that only wants the column names needs.

    The v2 extra columns come from :func:`feature_columns` — the same list
    :func:`feature_schema` builds its Arrow fields from — so a column the
    replay started writing is in the next fit without its name being
    repeated here. A catalogue column this run's rows do not carry is
    still in the design and imputed, exactly as a missing value is; that
    is what lets one model score a tree of old and new partitions.
    """
    if version not in DESIGN_VERSIONS:
        raise ValueError(
            f"unknown design version {version!r}; expected one of "
            f"{', '.join(DESIGN_VERSIONS)}"
        )
    in_design = tuple(sorted({int(x) for x in leads}))
    lead_column = LEAD_COLUMN if shared_lead else None
    if version == DESIGN_V1:
        return DesignSpec(
            version=version, lead_column=lead_column,
            own_columns=_own_columns(
                _v1_design_columns(in_design), in_design,
            ) if shared_lead else (),
        )

    catalogue = [name for name, _definition in feature_columns(in_design)]
    raw_names = {raw_fraction_column(lead) for lead in in_design}
    extras = tuple(
        name for name in catalogue
        if name not in _V1_HANDLED and name not in raw_names
    )
    logged = tuple(
        name for name in extras
        if wants_log1p(
            name, None if features is None else features.get(name),
        )
    )

    splines: list[tuple[str, tuple[float, ...]]] = []
    if features is not None:
        targets = tuple(SPLINE_BASE_COLUMNS) + tuple(
            raw_fraction_column(lead) for lead in in_design
        )
        base = build_design(
            features, in_design,
            spec=DesignSpec(
                version=version, extra_columns=extras, log_columns=logged,
            ),
        )
        lookup = {
            name: index
            for index, name in enumerate(
                _base_design_columns(in_design, extras, logged)
            )
        }
        for column in targets:
            index = lookup.get(column)
            if index is None:
                continue
            knots = spline_knots(base[:, index])
            if knots:
                splines.append((column, knots))
        del base

    seasons = tuple(f"season_{name}" for name in SEASONS)
    interactions: list[tuple[str, str]] = [("up_no_echo", "bulk_kmh")]
    # The fraction's meaning is seasonal: the same 0.8 is a summer shower
    # that may still miss and a winter band that will not. Built for every
    # design lead rather than only the lead being predicted, because ONE
    # design matrix serves every lead (:meth:`PostprocessModel.predict`)
    # and each lead's ridge is free to ignore the other leads' terms.
    for lead in in_design:
        interactions += [
            (raw_fraction_column(lead), season) for season in seasons
        ]
    # The diurnal cycle is a summer cycle: convection peaks in the
    # afternoon and winter stratiform does not care what time it is.
    for angle in ("hour_sin", "hour_cos"):
        interactions += [(angle, season) for season in seasons]
    return DesignSpec(
        version=version,
        extra_columns=extras,
        log_columns=logged,
        splines=tuple(splines),
        interactions=tuple(interactions),
        lead_column=lead_column,
        own_columns=_own_columns(
            _base_design_columns(in_design, extras, logged), in_design,
        ) if shared_lead else (),
    )


def _own_columns(
    base_names: Sequence[str], leads: Sequence[int],
) -> tuple[tuple[str, tuple[tuple[int, str], ...]], ...]:
    """Which per-lead base columns a shared design collapses, and from where.

    A family qualifies when the base design carries one column per design
    lead named ``<prefix><lead>``. Found by pattern rather than by a list
    of names: the ensemble block belongs to the feature side of this
    module and is expected to grow.
    """
    available = set(base_names)
    wanted = tuple(sorted({int(x) for x in leads}))
    out: list[tuple[str, tuple[tuple[int, str], ...]]] = []
    seen: set[str] = set()
    for name in base_names:
        for prefix in OWN_COLUMN_PREFIXES:
            if not name.startswith(prefix) or prefix in seen:
                continue
            stem = name[len(prefix):]
            # ``ens_mean_20`` under the ``ens_`` prefix: the family is
            # everything up to the trailing lead.
            head, _, tail = stem.rpartition("_")
            for family in ({prefix + head + "_"} if head else set()) | {prefix}:
                if family in seen:
                    continue
                sources = tuple(
                    (lead, f"{family}{lead}") for lead in wanted
                )
                if all(source in available for _lead, source in sources):
                    seen.add(family)
                    out.append((own_column(family), sources))
            del tail
    return tuple(out)


def _base_design_columns(
    leads: Sequence[int],
    extra_columns: Sequence[str] = (),
    log_columns: Sequence[str] = (),
) -> tuple[str, ...]:
    """The v1 columns plus the spec's extras — what the rest is built from."""
    logged = set(log_columns)
    return _v1_design_columns(leads) + tuple(
        _extra_design_name(name, name in logged) for name in extra_columns
    )


def design_columns(
    leads: Sequence[int], spec: DesignSpec | None = None,
) -> tuple[str, ...]:
    """Names of the design matrix's columns, in order.

    ``leads`` is the set of leads whose raw ensemble fraction goes into the
    design — every lead the products publish, not just the one being
    predicted. The *shape* of the fraction against lead is what separates
    "a big cell 40 km away" from "a drizzle edge overhead", and a model
    given only its own lead cannot see it.

    ``spec=None`` is the v1 design, in the order it always had, so every
    caller written before the spec keeps getting exactly what it got.

    The curve-calibrated ``p_rain_<lead>`` is deliberately absent: it is a
    monotone map of ``raw_frac_<lead>`` and adds nothing the design does
    not already have, and keeping it out means the baseline the report
    compares against is never also an input.
    """
    if spec is None:
        return _v1_design_columns(leads)
    names = list(
        _base_design_columns(leads, spec.extra_columns, spec.log_columns)
    )
    for column, knots in spec.splines:
        names += [
            _spline_name(column, index) for index in range(max(len(knots) - 2, 0))
        ]
    names += [_interaction_name(a, b) for a, b in spec.interactions]
    if spec.lead_column:
        names += [name for name, _sources in spec.own_columns]
        names += list(LEAD_COLUMNS)
    return tuple(names)


def _v1_design_columns(leads: Sequence[int]) -> tuple[str, ...]:
    """The 27 shipped columns, frozen. Never grows: v2 appends instead."""
    raw = tuple(raw_fraction_column(lead) for lead in sorted({int(x) for x in leads}))
    return raw + (
        "log1p_observed_mm_h",
        "log1p_obs_max_5km_mm_h",
        "log1p_up_max_20km_mm_h",
        "log1p_up_max_40km_mm_h",
        "log1p_up_dist_km",
        "up_no_echo",
        "up_wet_frac_40km",
        "eta_min_capped",
        "eta_missing",
        "log1p_intensity_mm_h",
        "bulk_kmh",
        "bulk_dir_sin",
        "bulk_dir_cos",
        "local_speed_kmh",
        "stalled_share",
        "frame_age_min",
        "log1p_station_radar_km",
        "hour_sin",
        "hour_cos",
        "season_summer",
        "season_winter",
        "season_shoulder",
    )


#: Stored columns :func:`build_design` reads. A caller assembling the
#: feature dict from parquet needs exactly these plus ``season`` (or a
#: timestamp to derive it from).
DESIGN_SOURCE_COLUMNS: tuple[str, ...] = (
    "observed_mm_h",
    "obs_max_5km_mm_h",
    "up_max_20km_mm_h",
    "up_max_40km_mm_h",
    "up_dist_km",
    "up_wet_frac_40km",
    "eta_min",
    "intensity_mm_h",
    "bulk_kmh",
    "bulk_dir_deg",
    "local_speed_kmh",
    "stalled_share",
    "frame_age_min",
    "station_radar_km",
    "hour_utc",
    # -- the v2 block (2026-09-16). Appended, never inserted: a reader
    # that loads these by name off a file written before they existed
    # gets a column of nulls, which is what "the writer did not have it"
    # has always meant here.
    "g_mm_10",
    "g_mm_30",
    "g_mm_60",
    "g_min_since_wet",
    "g_dry_60",
    "g_known",
    "obs_prev10_mm_h",
    "obs_prev20_mm_h",
    "obs_max_5km_prev10_mm_h",
    "wet_frac_5km",
    "wet_frac_10km",
    "up_mean_40km_mm_h",
) + tuple(
    upstream_bin_column(index) for index in range(UPSTREAM_BINS)
) + (
    "ens_eta_spread_min",
    # -- the neighbour-gauge block (2026-09-17). Appended for the same
    # reason, and generated from the catalogue rather than retyped: this
    # family is 21 columns and a typo in one of them would be a silently
    # imputed column rather than an error.
) + tuple(
    name for name, _definition in SCALAR_FEATURE_COLUMNS_NG
)


def _column(features: Mapping[str, Any], name: str, n: int) -> np.ndarray:
    """One source column as float64, or an all-NaN column when absent.

    A missing column is treated exactly like a column of missing values —
    imputed to the training mean, with the fit's coefficient then doing
    nothing. That is what lets a model fitted on a run with features score
    a run written before they existed, instead of raising in the middle of
    a batch.
    """
    values = features.get(name)
    if values is None:
        return np.full(n, np.nan, dtype=np.float64)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _rows_in(features: Mapping[str, Any]) -> int:
    for value in features.values():
        arr = np.asarray(value)
        if arr.ndim >= 1:
            return int(arr.shape[0])
    raise ValueError("features carries no array column")


def build_design(
    features: Mapping[str, Any],
    leads: Sequence[int],
    spec: DesignSpec | None = None,
) -> np.ndarray:
    """Stored feature columns → the ``(n, k)`` float64 design matrix.

    ``spec=None`` builds the v1 design and nothing else — the same 27
    columns, in the same order, from the same transforms as before this
    parameter existed. A spec appends, in this order: its extra catalogue
    columns, the spline bases, the interactions, and the shared-lead
    column (left NaN here, filled per lead by the model). Appending rather
    than interleaving is what lets the report line a v2 coefficient up
    with its v1 twin.

    Every transform in one place, so the fit and the serving path cannot
    drift:

    * **log1p** on rain rates and on distances. Rain rate is roughly
      log-normal and a 60 mm/h core must not be sixty times the weight of
      a 1 mm/h drizzle; ``log1p`` keeps 0 at 0 so "dry" stays exactly dry.
    * **Capped value + indicator** where missing is informative:
      ``up_dist_km`` NaN means "no echo in the 40 km corridor", and
      ``eta_min`` NaN means "no member brings rain inside the horizon".
      Both get the cap and an explicit 0/1 column; ``intensity_mm_h``,
      which is NaN exactly when the ETA is, folds to 0 mm/h under the same
      indicator.
    * **sin/cos** on the two angles — the hour of day and the bulk
      bearing — so 23:00 is next to 00:00 and 359° next to 1°.
    * **One-hot** on the season. All three levels are kept: with an L2
      penalty on the slopes the collinearity with the intercept is
      harmless, and dropping one would make the coefficients unreadable
      as "this season's effect".

    Any value still non-finite here is left as NaN;
    :meth:`Standardiser.transform` imputes it with the training mean.
    """
    n = _rows_in(features)
    columns: list[np.ndarray] = []
    for lead in sorted({int(x) for x in leads}):
        columns.append(_column(features, raw_fraction_column(lead), n))

    def log1p(name: str) -> np.ndarray:
        values = _column(features, name, n)
        return np.log1p(np.clip(values, 0.0, None))

    up_dist = _column(features, "up_dist_km", n)
    up_missing = ~np.isfinite(up_dist)
    eta = _column(features, "eta_min", n)
    eta_missing = ~np.isfinite(eta)
    intensity = _column(features, "intensity_mm_h", n)
    bearing = np.radians(_column(features, "bulk_dir_deg", n))
    hour = _column(features, "hour_utc", n) * (2.0 * math.pi / 24.0)
    season = np.asarray(features.get("season", np.full(n, "", dtype="<U8")))
    season = season.astype("<U8").reshape(-1)

    columns += [
        log1p("observed_mm_h"),
        log1p("obs_max_5km_mm_h"),
        log1p("up_max_20km_mm_h"),
        log1p("up_max_40km_mm_h"),
        np.log1p(np.where(up_missing, UP_DIST_CAP_KM, np.clip(up_dist, 0.0, UP_DIST_CAP_KM))),
        up_missing.astype(np.float64),
        _column(features, "up_wet_frac_40km", n),
        np.where(eta_missing, ETA_CAP_MIN, np.clip(eta, 0.0, ETA_CAP_MIN)),
        eta_missing.astype(np.float64),
        np.log1p(np.clip(np.where(eta_missing, 0.0, intensity), 0.0, None)),
        _column(features, "bulk_kmh", n),
        np.where(np.isfinite(bearing), np.sin(bearing), 0.0),
        np.where(np.isfinite(bearing), np.cos(bearing), 0.0),
        _column(features, "local_speed_kmh", n),
        _column(features, "stalled_share", n),
        _column(features, "frame_age_min", n),
        log1p("station_radar_km"),
        np.sin(hour),
        np.cos(hour),
    ]
    columns += [(season == name).astype(np.float64) for name in SEASONS]

    if spec is not None:
        # The catalogue columns v1 never named. Raw, except the ones the
        # fit decided are rates or distances (:func:`wants_log1p`), which
        # go through the same log1p every v1 column of that kind does.
        # The decision is the SPEC's, never re-derived here.
        logged = set(spec.log_columns)
        for name in spec.extra_columns:
            values = _column(features, name, n)
            columns.append(
                np.log1p(np.clip(values, 0.0, None))
                if name in logged else values
            )
        by_name = {
            name: index for index, name in enumerate(
                _base_design_columns(
                    leads, spec.extra_columns, spec.log_columns,
                )
            )
        }
        # A snapshot of the base block: the loops below append to
        # ``columns``, and a spline of a spline is not a thing.
        base = list(columns)
        for column_name, knots in spec.splines:
            index = by_name.get(column_name)
            if index is None:
                # A spline on a column this design does not have. Keep the
                # width the names promise — the coefficients are stored
                # positionally — and let the imputation neutralise it.
                columns += [
                    np.full(n, np.nan) for _ in range(max(len(knots) - 2, 0))
                ]
                continue
            basis = natural_spline_basis(base[index], knots)
            columns += [basis[:, k] for k in range(basis.shape[1])]
        for a, b in spec.interactions:
            left, right = by_name.get(a), by_name.get(b)
            if left is None or right is None:
                columns.append(np.full(n, np.nan))
                continue
            columns.append(base[left] * base[right])
        if spec.lead_column:
            # The shared design's tail: one column per collapsed per-lead
            # family, then the lead itself twice. All left NaN here and
            # written per lead by
            # :meth:`PostprocessModel._fill_lead_columns` — NaN is what
            # makes a caller that forgot to fill them produce a missing
            # value rather than a confident answer about lead 0.
            for _name, _sources in spec.own_columns:
                columns.append(np.full(n, np.nan))
            for _name in LEAD_COLUMNS:
                columns.append(np.full(n, np.nan))

    design = np.empty((n, len(columns)), dtype=np.float64)
    for index, column in enumerate(columns):
        design[:, index] = column
    return design


# ---------------------------------------------------------------------------
# Feature families, for the ablation
# ---------------------------------------------------------------------------

#: Which family a design column belongs to, as an ordered list of
#: ``(family, predicate)``. First match wins, so the order is the rule.
#:
#: The ablation drops one family at a time and reports what the model
#: loses, which is the only honest way to answer "is the gauge history
#: doing anything?" — a coefficient's size answers a different question,
#: and answers it badly when the predictors are correlated.
#:
#: A spline column inherits its base column's family by name
#: (``bulk_kmh_ns1`` starts with ``bulk_kmh``); an interaction is dropped
#: when EITHER parent's family is dropped, which is handled in
#: :func:`family_columns` rather than here.
FEATURE_FAMILIES: tuple[tuple[str, Callable[[str], bool]], ...] = (
    ("raw_frac", lambda name: name.startswith(RAW_FRACTION_PREFIX)),
    ("gauge", lambda name: name.startswith(("g_", "log1p_g_"))),
    # Tested BEFORE "obs" and "up": every column here starts with the
    # same three characters, and the ablation's whole job is that
    # "drop the neighbour gauges" drops all of them and nothing else.
    ("neighbour", lambda name: name.startswith(("ng_", "log1p_ng_"))),
    ("ensemble", lambda name: name.startswith(("ens_", "log1p_ens_"))),
    (
        "obs",
        lambda name: name.startswith(
            # ``wet_frac_<r>km`` is the observation disc's wet share; the
            # upwind corridor's is ``up_wet_frac_40km`` and belongs to
            # ``up``, which is why the ``up_`` rule below is not enough
            # and this one has to name it.
            ("log1p_observed", "log1p_obs_", "obs_", "observed", "wet_frac_"),
        ),
    ),
    ("up", lambda name: name.startswith(("up_", "log1p_up_"))),
    (
        "motion",
        lambda name: name.startswith(
            ("bulk_", "local_speed", "stalled_share"),
        ),
    ),
    (
        "eta",
        lambda name: name.startswith(
            ("eta_", "log1p_intensity"),
        ),
    ),
    (
        "station",
        lambda name: name.startswith(("log1p_station_radar", "station_")),
    ),
    (
        "time",
        lambda name: name.startswith(("hour_", "season_", "frame_age")),
    ),
)

#: Every family name, in report order.
FAMILY_NAMES: tuple[str, ...] = tuple(name for name, _test in FEATURE_FAMILIES)


def feature_family(column: str) -> str:
    """The family a design column belongs to, or ``"other"``."""
    for name, test in FEATURE_FAMILIES:
        if test(column):
            return name
    return "other"


def family_columns(
    names: Sequence[str], family: str,
) -> np.ndarray:
    """Boolean mask over ``names``: which columns belong to ``family``.

    An interaction ``a_x_b`` belongs to a family when EITHER parent does,
    because dropping ``up`` while keeping ``up_no_echo × bulk_kmh`` would
    leave the family in the design under another name — which is exactly
    the mistake an ablation exists to avoid.
    """
    wanted = str(family)
    out = np.zeros(len(names), dtype=bool)
    for index, name in enumerate(names):
        parts = name.split("_x_") if "_x_" in name else [name]
        out[index] = any(feature_family(part) == wanted for part in parts)
    return out


# ---------------------------------------------------------------------------
# "Was the gauge dry before the decision?" — the onset-relevant subset
# ---------------------------------------------------------------------------

#: The window the onset-relevance flag looks back over, minutes. Matches
#: ``GAUGE_DRY_WINDOW_MIN``, which is the same question asked at the point
#: rather than at the row; one definition, two shapes.
DRY_BEFORE_MIN = GAUGE_DRY_WINDOW_MIN

#: The stored column carrying the answer, when the writer computed it.
DRY_COLUMN = "g_dry_60"

#: Stratum label for the onset-relevant rows, beside :data:`POOLED` and
#: the seasons.
DRY = "dry"


def dry_before(
    grid: Any,
    t: np.ndarray,
    station: np.ndarray,
    *,
    minutes: int = DRY_BEFORE_MIN,
    lag_min: float = DEFAULT_GAUGE_LAG_MIN,
) -> np.ndarray:
    """Was the gauge dry for the ``minutes`` before each decision instant?

    ``1.0`` when no gauge slot in the window was wet and every slot in it
    was known, ``0.0`` when one was wet, ``NaN`` when the window ran off
    the archive or a slot in it said nothing — a gauge that was silent
    cannot certify a dry hour, and calling it dry would put exactly the
    rows with no evidence into the subset the gate is read off.

    THE definition, in one place. ``station_gauge_features`` computes the
    same thing per point from the live gauge slots and writes it as
    ``g_dry_60``; this computes it for a whole table of decision rows from
    the archive grid the benchmark already built. The evaluation prefers
    the stored column whenever the rows carry it (see
    :func:`dry_subset`), so the two can only ever be consulted about rows
    the other never saw.

    ``grid`` is anything with :class:`decision_rows.GaugeGrid`'s
    ``outcome(t, station, lead) -> (wet, usable)`` — the window is
    expressed through it rather than through its internals, so the slot
    grid, the wet rule and the off-the-edge rule are the benchmark's and
    not a second opinion.

    ``lag_min`` is the gauge availability lag: the window ends that many
    minutes BEFORE the decision instant, because that is the freshest slot
    the running service could have read. It is the same default
    ``station_gauge_features`` uses.
    """
    stamps = np.asarray(t, dtype=np.int64).reshape(-1)
    codes = np.asarray(station, dtype=np.int64).reshape(-1)
    span = int(minutes)
    # ``outcome`` grades the slots ending in (u, u + span]; the window
    # wanted here ends at the visibility horizon and starts ``span``
    # minutes before it.
    horizon = stamps - int(round(float(lag_min) * 60.0))
    wet, usable = grid.outcome(horizon - span * 60, codes, span)
    out = np.full(stamps.size, np.nan, dtype=np.float64)
    graded = np.asarray(usable, dtype=bool)
    out[graded] = np.where(np.asarray(wet, dtype=np.float64)[graded] > 0, 0.0, 1.0)
    return out


def dry_subset(
    features: Mapping[str, Any],
    *,
    grid: Any = None,
    t: np.ndarray | None = None,
    station: np.ndarray | None = None,
    lag_min: float = DEFAULT_GAUGE_LAG_MIN,
) -> np.ndarray:
    """The onset-relevant mask: rows whose gauge was dry for the last hour.

    Prefers the stored :data:`DRY_COLUMN` — the writer had the slots in
    hand and the archive may since have been trimmed — and falls back to
    :func:`dry_before` over ``grid`` for rows written before the column
    existed. A row that neither can answer for is **not** in the subset:
    the dry gate is a claim about rows known to have been dry.
    """
    n = _rows_in(features)
    stored = features.get(DRY_COLUMN)
    values = (
        np.full(n, np.nan, dtype=np.float64) if stored is None
        else np.asarray(stored, dtype=np.float64).reshape(-1)
    )
    if grid is not None and t is not None and station is not None:
        missing = ~np.isfinite(values)
        if missing.any():
            derived = dry_before(grid, t, station, lag_min=lag_min)
            values = np.where(missing, derived, values)
    return np.isfinite(values) & (values > 0.5)


# ---------------------------------------------------------------------------
# Validating where the subscriber lives, not where the gauge stands
# ---------------------------------------------------------------------------
#
# Every row in the archive is a DMI gauge. Every row in service is somebody's
# address. The two differ in three ways that all flatter the offline number:
#
# 1. the ``g_*`` block is a measurement at a gauge and an absence at an
#    address, and it is the single biggest thing v2 added;
# 2. a per-station intercept is learnable at a gauge and meaningless at an
#    address;
# 3. leave-one-MONTH-out lets a fold train on the very station it is scored
#    at, so a model that has quietly learned "station 06180" is never caught.
#
# :data:`PROTOCOL_RANDOM_POINT` closes all three: the own-gauge columns are
# masked to "unknown" in training AND in scoring, station offsets are
# refused, and the folds hold out a month and a GROUP OF STATIONS together,
# so every prediction comes from a model that saw neither that month nor
# that place. What is left is the model a subscriber would actually get —
# the radar features, and the neighbour gauges, which are leave-self-out by
# construction and therefore mean the same thing at both kinds of point.
#
# The remaining gap is geographic and is reported rather than closed: a
# gauge's nearest other gauge is not as far away as a random Dane's nearest
# gauge. :func:`random_point_distance_weights` measures how much not, and
# the report re-weights the distance-binned skill by it.

#: The evaluation as it has always run: rows are gauges, and they are
#: allowed to be.
PROTOCOL_AT_GAUGE = "at-gauge"

#: Rows are treated as the addresses they stand in for. See above.
PROTOCOL_RANDOM_POINT = "random-point"

PROTOCOLS: tuple[str, ...] = (PROTOCOL_AT_GAUGE, PROTOCOL_RANDOM_POINT)

#: The ``g_*`` columns that are a measurement AT the point — the ones an
#: address does not have. ``g_known`` is not among them: it is the
#: indicator, and the masking sets it to 0 rather than removing it.
OWN_GAUGE_COLUMNS: tuple[str, ...] = (
    "g_mm_10", "g_mm_30", "g_mm_60", "g_min_since_wet", DRY_COLUMN,
)

#: The indicator that says whether the rest of the block is a measurement.
GAUGE_KNOWN_COLUMN = "g_known"


def mask_own_gauge(features: Mapping[str, Any]) -> dict[str, Any]:
    """``features`` with the point's OWN gauge masked to "unknown".

    A shallow copy with :data:`OWN_GAUGE_COLUMNS` replaced by NaN and
    :data:`GAUGE_KNOWN_COLUMN` by 0 — exactly the block
    :func:`station_gauge_features` writes for a point that is not a gauge,
    so the masked row is not an invented shape but a shape the live cycle
    produces for most of its points already.

    Applied to the WHOLE table, before any fold is cut, so the model is
    trained on masked rows as well as scored on them. Masking only at
    scoring time would be worse than not masking at all: a model fitted to
    lean on ``g_min_since_wet`` and then handed a NaN for it is a model
    being asked a question in a language it was not taught.

    The ``ng_*`` block is deliberately NOT masked. It never contained the
    point's own gauge — :func:`neighbour_gauge_features` excludes it by
    construction — so at a gauge station it already says what it would say
    at an address a few metres away.

    The truth-side use of the same information is untouched: the
    onset-relevant :data:`DRY` subset is a statement about what actually
    happened at the point, not a predictor, and the caller derives it
    BEFORE masking (or from the gauge grid) for exactly that reason.
    """
    out = dict(features)
    n = _rows_in(features)
    for name in OWN_GAUGE_COLUMNS:
        if name in out:
            out[name] = np.full(n, np.nan, dtype=np.float64)
    if GAUGE_KNOWN_COLUMN in out:
        out[GAUGE_KNOWN_COLUMN] = np.zeros(n, dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# Distance to the nearest gauge: the axis the two kinds of point differ on
# ---------------------------------------------------------------------------

#: Bin edges on ``ng_near_km``, km. Ten-kilometre steps up to 30 and one
#: open bin above it: the gauge network's own nearest-neighbour distances
#: mostly land in the first two bins, and a random point in Denmark mostly
#: does not, which is the whole point of reporting the two side by side.
DISTANCE_BIN_EDGES: tuple[float, ...] = (10.0, 20.0, 30.0)

#: The column the bins are cut on.
DISTANCE_COLUMN = "ng_near_km"


def distance_bin_labels(
    edges: Sequence[float] = DISTANCE_BIN_EDGES,
) -> tuple[str, ...]:
    """``("0-10 km", "10-20 km", "20-30 km", "30+ km")`` for the default edges."""
    bounds = [0.0] + [float(x) for x in edges]
    labels = [
        f"{bounds[i]:.0f}-{bounds[i + 1]:.0f} km" for i in range(len(bounds) - 1)
    ]
    return tuple(labels + [f"{bounds[-1]:.0f}+ km"])


def distance_bins(
    values: Any, edges: Sequence[float] = DISTANCE_BIN_EDGES,
) -> dict[str, np.ndarray]:
    """``{label: boolean mask}`` over rows, by distance to the nearest gauge.

    A row with no distance (a run whose rows predate the ``ng_*`` columns)
    is in no bin, rather than in the first one.
    """
    km = np.asarray(values, dtype=np.float64).reshape(-1)
    bounds = [0.0] + [float(x) for x in edges] + [float("inf")]
    labels = distance_bin_labels(edges)
    finite = np.isfinite(km)
    return {
        label: finite & (km >= bounds[index]) & (km < bounds[index + 1])
        for index, label in enumerate(labels)
    }


def _nearest_other_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Per station, the distance to the nearest OTHER station, km."""
    north = (lat[:, None] - lat[None, :]) * KM_PER_DEG_LAT
    east = (lon[:, None] - lon[None, :]) * KM_PER_DEG_LON
    distance = np.hypot(north, east)
    np.fill_diagonal(distance, np.inf)
    return distance.min(1)


def _coordinate_arrays(
    station_coords: Mapping[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    lat = np.array(
        [float(v[0]) for v in station_coords.values()], dtype=np.float64,
    )
    lon = np.array(
        [float(v[1]) for v in station_coords.values()], dtype=np.float64,
    )
    return lat, lon


def _distance_summary(
    km: np.ndarray, edges: Sequence[float] = DISTANCE_BIN_EDGES,
) -> dict[str, Any]:
    """Weights over the distance bins, plus the quantiles behind them."""
    finite = km[np.isfinite(km)]
    labels = distance_bin_labels(edges)
    bounds = [0.0] + [float(x) for x in edges] + [float("inf")]
    counts = [
        int(((finite >= bounds[i]) & (finite < bounds[i + 1])).sum())
        for i in range(len(labels))
    ]
    total = max(sum(counts), 1)
    quantiles = (
        np.quantile(finite, [0.1, 0.25, 0.5, 0.75, 0.9]) if finite.size
        else np.full(5, np.nan)
    )
    return {
        "n": int(finite.size),
        "labels": list(labels),
        "counts": counts,
        "weights": [count / total for count in counts],
        "km": {
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "p10": float(quantiles[0]), "p25": float(quantiles[1]),
            "p50": float(quantiles[2]), "p75": float(quantiles[3]),
            "p90": float(quantiles[4]),
            "max": float(finite.max()) if finite.size else float("nan"),
        },
    }


def gauge_distance_weights(
    station_coords: Mapping[str, tuple[float, float]],
    edges: Sequence[float] = DISTANCE_BIN_EDGES,
) -> dict[str, Any]:
    """The distribution the ARCHIVE's rows are drawn from.

    Per gauge, the distance to its nearest other gauge — which is what
    ``ng_near_km`` holds at a training row, because the point's own gauge
    is excluded. Reported beside
    :func:`random_point_distance_weights` so a reader can see the size of
    the extrapolation the re-weighting performs rather than take it on
    trust.
    """
    lat, lon = _coordinate_arrays(station_coords)
    if lat.size < 2:
        return _distance_summary(np.full(lat.size, np.nan), edges)
    return _distance_summary(_nearest_other_km(lat, lon), edges)


def _inside_ring(lon: np.ndarray, lat: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Ray casting: is each ``(lon, lat)`` inside this closed ring?

    The standard crossing-number test, vectorised over points. A ray is
    cast toward -lon and the crossings of the ring's edges are counted; an
    odd count is inside. Points exactly on an edge are undefined and there
    is no attempt to define them: this is a sampling domain at ~1 km
    tolerance, and a point on the coastline is a rounding decision either
    way.
    """
    x1, y1 = ring[:, 0][None, :], ring[:, 1][None, :]
    x2 = np.roll(ring[:, 0], -1)[None, :]
    y2 = np.roll(ring[:, 1], -1)[None, :]
    px, py = lon[:, None], lat[:, None]
    straddles = (y1 > py) != (y2 > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        crossing = (x2 - x1) * (py - y1) / (y2 - y1) + x1
    return ((straddles & (px < crossing)).sum(1) % 2) == 1


def points_inside(
    lon: np.ndarray, lat: np.ndarray, outline: Sequence[Any],
) -> np.ndarray:
    """Which of these points fall inside any ring of ``outline``.

    Every ring of :data:`~dmi_nowcast_core.denmark_outline.DENMARK_OUTLINE`
    is the outer ring of a separate landmass — there are no holes — so
    "inside Denmark" is "inside any one of them".
    """
    inside = np.zeros(lon.size, dtype=bool)
    for raw in outline:
        ring = np.asarray(raw, dtype=np.float64)
        if ring.shape[0] < 3:
            continue
        inside |= _inside_ring(lon, lat, ring)
    return inside


#: Rejection-sampling batch. Denmark fills ~30 % of its bounding box, so a
#: batch this size yields ~15 000 accepted points and the whole draw is a
#: handful of passes.
_SAMPLE_BATCH = 50_000


def random_points_in(
    outline: Sequence[Any], *, n: int = 200_000, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """``(lat, lon)`` of ``n`` points drawn uniformly by AREA inside ``outline``.

    Uniform in longitude and in the SINE of latitude, not in latitude
    itself: a degree of latitude covers the same area everywhere but a
    degree of longitude does not, and sampling latitude flat would over-
    represent Jutland's north by ~9 % across Denmark's 3.2 degrees. The
    correction costs one ``arcsin`` and removes an argument.

    Rejection sampling against the bounding box, with a cap on the number
    of batches so a caller that hands in a degenerate outline gets fewer
    points rather than an infinite loop.
    """
    from .denmark_outline import DENMARK_BBOX

    rings = [np.asarray(ring, dtype=np.float64) for ring in outline]
    if rings and all(ring.size for ring in rings):
        stacked = np.concatenate(rings)
        lon_min, lon_max = float(stacked[:, 0].min()), float(stacked[:, 0].max())
        lat_min, lat_max = float(stacked[:, 1].min()), float(stacked[:, 1].max())
    else:
        lon_min, lat_min, lon_max, lat_max = DENMARK_BBOX
    rng = np.random.default_rng(int(seed))
    sin_lo, sin_hi = math.sin(math.radians(lat_min)), math.sin(math.radians(lat_max))
    lats: list[np.ndarray] = []
    lons: list[np.ndarray] = []
    taken = 0
    for _attempt in range(max(4 * (int(n) // _SAMPLE_BATCH + 1), 64)):
        if taken >= int(n):
            break
        lon = rng.uniform(lon_min, lon_max, _SAMPLE_BATCH)
        lat = np.degrees(np.arcsin(rng.uniform(sin_lo, sin_hi, _SAMPLE_BATCH)))
        keep = points_inside(lon, lat, rings)
        if not keep.any():
            continue
        lats.append(lat[keep])
        lons.append(lon[keep])
        taken += int(keep.sum())
    if not lats:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    return (
        np.concatenate(lats)[:int(n)], np.concatenate(lons)[:int(n)],
    )


def random_point_distance_weights(
    station_coords: Mapping[str, tuple[float, float]],
    outline: Sequence[Any] | None = None,
    *,
    edges: Sequence[float] = DISTANCE_BIN_EDGES,
    n: int = 200_000,
    seed: int = 0,
) -> dict[str, Any]:
    """How far a random place in Denmark is from the nearest rain gauge.

    Draws ``n`` points uniformly by area inside ``outline``
    (:data:`~dmi_nowcast_core.denmark_outline.DENMARK_OUTLINE` by default),
    measures each one's distance to the nearest station in
    ``station_coords`` on the same equirectangular kilometre grid the
    ``ng_*`` features use, and returns the share falling in each distance
    bin.

    Those shares are the weights the random-point report re-weights its
    distance-binned skill by. The re-weighting is what turns "the model is
    worth this much at a gauge 12 km from its neighbour" into "the model is
    worth this much to a subscriber", and it is an extrapolation only in
    the bins the archive is thin in — which is why
    :func:`gauge_distance_weights` is printed beside it.
    """
    if outline is None:
        from .denmark_outline import DENMARK_OUTLINE

        outline = DENMARK_OUTLINE
    lat, lon = _coordinate_arrays(station_coords)
    point_lat, point_lon = random_points_in(outline, n=n, seed=seed)
    if not point_lat.size or not lat.size:
        summary = _distance_summary(np.full(0, np.nan), edges)
        summary["seed"] = int(seed)
        summary["stations"] = int(lat.size)
        return summary
    best = np.full(point_lat.size, np.inf, dtype=np.float64)
    # One station at a time: 200 000 x 100 in one matrix is 160 MB for no
    # reason, and the running minimum costs nothing.
    for index in range(lat.size):
        north = (lat[index] - point_lat) * KM_PER_DEG_LAT
        east = (lon[index] - point_lon) * KM_PER_DEG_LON
        np.minimum(best, np.hypot(north, east), out=best)
    summary = _distance_summary(best, edges)
    summary["seed"] = int(seed)
    summary["stations"] = int(lat.size)
    return summary


def expected_at_random_point(
    by_bin: Mapping[str, float | None], weights: Mapping[str, float],
) -> float | None:
    """One number from a distance-binned one, re-weighted.

    ``None`` when no bin that carries weight has a value — an honest
    refusal, rather than an average over the bins that happen to be filled.
    Bins with a value are renormalised among themselves and the share of
    the weight they cover is the caller's to report (``covered`` in
    :func:`random_point_expectation`).
    """
    total = 0.0
    mass = 0.0
    for name, weight in weights.items():
        value = by_bin.get(name)
        if value is None or not math.isfinite(float(value)):
            continue
        total += float(weight) * float(value)
        mass += float(weight)
    return None if mass <= 0.0 else total / mass


def random_point_expectation(
    by_bin: Mapping[str, float | None], weights: Mapping[str, float],
) -> dict[str, Any]:
    """``{"value": ..., "covered": ...}`` — the re-weighted number and its reach.

    ``covered`` is the share of the random-point weight the filled bins
    account for. A value standing on 60 % of the weight is a different
    claim from one standing on 99 %, and the report prints both.
    """
    mass = sum(
        float(weight) for name, weight in weights.items()
        if by_bin.get(name) is not None
        and math.isfinite(float(by_bin.get(name)))
    )
    return {
        "value": expected_at_random_point(by_bin, weights),
        "covered": mass / max(sum(float(w) for w in weights.values()), 1e-12),
    }


# ---------------------------------------------------------------------------
# Station groups and (month, group) folds
# ---------------------------------------------------------------------------

#: How many station groups the spatial hold-out cuts the country into.
DEFAULT_STATION_GROUPS = 5


def station_groups(
    station_coords: Mapping[str, tuple[float, float]],
    *,
    groups: int = DEFAULT_STATION_GROUPS,
) -> dict[str, int]:
    """``{station_id: group}`` — a deterministic spatial hold-out split.

    Stations are sorted by LONGITUDE and dealt out round-robin, so each
    group is a west-to-east comb through the country rather than a region.
    That is deliberate. A split into contiguous regions would hold out
    Bornholm or west Jutland as a block and measure "does a model trained
    on the rest of Denmark work there", which is a different and much
    harder question than the one being asked — "does this model work at a
    station it has never seen". Interleaving keeps every group spanning the
    country, so the held-out stations are unfamiliar places in familiar
    weather, which is exactly a new subscriber.

    Deterministic by construction: no seed, no shuffle. Ties in longitude
    break on the station id, so the same catalogue always gives the same
    groups and two runs are comparable.
    """
    count = max(int(groups), 1)
    order = sorted(
        station_coords, key=lambda sid: (float(station_coords[sid][1]), str(sid)),
    )
    return {sid: index % count for index, sid in enumerate(order)}


def group_codes(
    stations: Any, groups: Mapping[str, int], *, unknown: int = -1,
) -> np.ndarray:
    """Per row, its station's group; ``unknown`` for a station not in the map.

    An unmatched row gets its own single-member group rather than being
    folded into group 0, so a station the split never saw cannot end up
    training on itself.
    """
    labels = np.asarray(stations).astype(str).reshape(-1)
    distinct, inverse = np.unique(labels, return_inverse=True)
    table = np.array(
        [int(groups.get(str(name), unknown)) for name in distinct], dtype=np.int64,
    )
    return table[inverse]


@dataclass(frozen=True)
class FoldPlan:
    """What each fold holds out, on one axis or on several.

    ``axes`` is one integer array per row per axis, and a fold is one
    combination of their values. A row is in the fold's TEST set when it
    matches on every axis, and in its TRAINING set when it differs on
    every axis — which is what makes a two-axis plan a genuinely spatial
    hold-out and not a month hold-out with extra steps: a row sharing the
    month but not the group, or the group but not the month, is used by
    neither side.

    One axis reproduces leave-one-(year, month)-out exactly, which is how
    the default protocol keeps its numbers.
    """

    axes: tuple[np.ndarray, ...]
    names: tuple[str, ...] = ()
    #: Per axis, ``{value: label}`` for the fold names in the report.
    labels: tuple[Mapping[int, str], ...] = ()

    @classmethod
    def by_month(cls, month: Any) -> "FoldPlan":
        """The historical plan: one fold per ``(year, month)``."""
        keys = np.asarray(month, dtype=np.int64).reshape(-1)
        return cls(
            axes=(keys,), names=("month",),
            labels=({int(k): _month_label(int(k)) for k in np.unique(keys)},),
        )

    @classmethod
    def by_month_and_group(cls, month: Any, group: Any) -> "FoldPlan":
        """Month crossed with station group — the random-point hold-out."""
        months = np.asarray(month, dtype=np.int64).reshape(-1)
        groups = np.asarray(group, dtype=np.int64).reshape(-1)
        if months.size != groups.size:
            raise ValueError("month and group must have one entry per row")
        return cls(
            axes=(months, groups), names=("month", "group"),
            labels=(
                {int(k): _month_label(int(k)) for k in np.unique(months)},
                {int(k): f"g{int(k)}" for k in np.unique(groups)},
            ),
        )

    @property
    def rows(self) -> int:
        return int(self.axes[0].size) if self.axes else 0

    def folds(self) -> list[tuple[int, ...]]:
        """Every combination present in the rows, in a stable order.

        A NEGATIVE value on any axis is the project's "not one of ours"
        code (:func:`group_codes`, :func:`_station_codes`) and never forms
        a fold of its own: those rows are a station the gauge grid
        dropped, they carry no gradable outcome, and holding them out
        would fit a model per month to predict nothing. They still TRAIN
        the other folds, where they are simply not gradable — which is
        the same thing that happens to them under a one-axis plan.
        """
        stacked = np.stack([np.asarray(a, dtype=np.int64) for a in self.axes], 1)
        return [
            tuple(int(v) for v in row) for row in np.unique(stacked, axis=0)
            if (row >= 0).all()
        ]

    def test_mask(self, fold: Sequence[int]) -> np.ndarray:
        mask = np.ones(self.rows, dtype=bool)
        for axis, value in zip(self.axes, fold):
            mask &= axis == int(value)
        return mask

    def train_mask(self, fold: Sequence[int]) -> np.ndarray:
        mask = np.ones(self.rows, dtype=bool)
        for axis, value in zip(self.axes, fold):
            mask &= axis != int(value)
        return mask

    def label(self, fold: Sequence[int]) -> str:
        parts = []
        for index, value in enumerate(fold):
            table = self.labels[index] if index < len(self.labels) else {}
            parts.append(str(table.get(int(value), int(value))))
        return " x ".join(parts)

    def to_json(self) -> dict[str, Any]:
        return {
            "axes": list(self.names),
            "n_folds": len(self.folds()),
            "sizes": [
                len({int(v) for v in np.unique(axis)}) for axis in self.axes
            ],
        }


@dataclass(frozen=True)
class Standardiser:
    """Per-column mean and scale, plus the NaN → mean imputation."""

    mean: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def fit(cls, design: np.ndarray) -> "Standardiser":
        clean = np.where(np.isfinite(design), design, np.nan)
        # An all-NaN column (a feature the run never wrote) makes nanmean
        # shout; it is a legitimate input here — the column simply carries
        # no information and ends up constant-zero after standardising.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(clean, axis=0)
            std = np.nanstd(clean, axis=0)
        del clean
        mean = np.nan_to_num(mean, nan=0.0)
        # A constant column (a season absent from the training window, a
        # frame age that never moved) would divide by zero; it carries no
        # information, so a scale of 1 makes it a column of zeros.
        std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
        return cls(mean=tuple(float(x) for x in mean),
                   scale=tuple(float(x) for x in std))

    def transform(self, design: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        if design.shape[1] != mean.size:
            raise ValueError(
                f"design has {design.shape[1]} columns, standardiser has "
                f"{mean.size}"
            )
        filled = np.where(np.isfinite(design), design, mean[None, :])
        return (filled - mean[None, :]) / scale[None, :]


# ---------------------------------------------------------------------------
# L2-regularised logistic regression, by L-BFGS
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


#: How much harder the per-station offsets are shrunk than the slopes.
#: They are 100-odd parameters each standing on one station's rows, and
#: an unshrunk one would happily memorise a gauge's quirks; at 10× the
#: base ridge an offset has to be worth something across a season before
#: it moves. Exposed as ``--station-l2-multiple``.
STATION_L2_MULTIPLE = 10.0


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-8,
    groups: np.ndarray | None = None,
    n_groups: int = 0,
    group_l2: float | None = None,
) -> dict[str, Any]:
    """Binary logistic regression with an L2 penalty on the slopes.

    ``groups`` adds a **per-group intercept offset**: an integer code per
    row, ``-1`` (or anything outside ``range(n_groups)``) meaning "no
    group", and one free parameter per group added to the linear
    predictor. It is a one-hot with its own, stronger ridge
    (``group_l2``, default ``STATION_L2_MULTIPLE × l2``) and it is never
    materialised as columns: at a hundred stations and half a million rows
    the dense one-hot would be 400 MB and the gradient is a ``bincount``.

    A row with no group gets an offset of 0, which is what a subscriber's
    own coordinate gets at serving time — the offsets are a statement
    about the gauges the model was trained at, and a house down the road
    is not one of them.

    Minimised, with ``n`` the row count::

        (1/n) · [ Σ logistic-loss(y, Xw + b)  +  ½ · l2 · ‖w‖² ]

    which is the usual "sum of losses plus ridge" objective — ``l2`` is
    scikit-learn's ``1/C`` — divided by ``n`` so the gradient stays
    O(1) at half a million rows and L-BFGS's own tolerances mean the same
    thing whatever the sample size. The intercept is **not** penalised: it
    carries the base rate, and shrinking it toward zero would push every
    prediction toward 50 %.

    ``x`` is expected standardised (:class:`Standardiser`) — a ridge on
    unstandardised columns penalises "km" and "mm/h" at different rates,
    which is not a modelling choice anyone would make on purpose.

    No class weighting. The event is not rare enough to need it (the base
    rate is around 10 % at these leads), a weighted fit is no longer
    calibrated, and the isotonic step afterwards would have to undo it.
    The base rate is reported instead so a reader can see what the
    intercept is standing on.

    Returns ``{"intercept", "coefficients", "group_offsets", "n",
    "base_rate", "loss", "converged", "iterations", "message"}``.
    ``group_offsets`` is an empty list when no groups were given, and the
    result is then bit-for-bit what it was before groups existed.
    """
    from scipy.optimize import minimize

    design = np.asarray(x, dtype=np.float64)
    outcome = np.asarray(y, dtype=np.float64).reshape(-1)
    if design.ndim != 2:
        raise ValueError(f"x must be 2-D, got {design.ndim}-D")
    if outcome.size != design.shape[0]:
        raise ValueError(
            f"x has {design.shape[0]} rows and y has {outcome.size}"
        )
    if design.shape[0] == 0:
        raise ValueError("no rows to fit")
    if not np.all(np.isfinite(design)):
        raise ValueError("x carries non-finite values; standardise first")
    if not np.all((outcome == 0.0) | (outcome == 1.0)):
        raise ValueError("y must be 0/1")
    if l2 < 0:
        raise ValueError(f"l2 must be >= 0, got {l2}")

    n, k = design.shape
    n_groups = max(int(n_groups), 0)
    if groups is None:
        n_groups = 0
    if n_groups:
        # One extra slot at the end is the "no group" sink: a row coded
        # there picks up a zero offset and contributes no gradient to any
        # real group, with no branch in the inner loop.
        codes = np.asarray(groups, dtype=np.int64).reshape(-1)
        if codes.size != n:
            raise ValueError(
                f"x has {n} rows and groups has {codes.size}"
            )
        codes = np.where((codes >= 0) & (codes < n_groups), codes, n_groups)
    else:
        codes = None
    base_rate = float(outcome.mean())
    # A degenerate fold (one class only) has no slope to estimate; the
    # honest answer is the constant that class implies, clipped away from
    # an infinite logit.
    if base_rate <= 0.0 or base_rate >= 1.0:
        clipped = min(max(base_rate, 1.0 / (n + 2.0)), 1.0 - 1.0 / (n + 2.0))
        return {
            "intercept": float(math.log(clipped / (1.0 - clipped))),
            "coefficients": [0.0] * k,
            "group_offsets": [0.0] * n_groups,
            "n": n,
            "base_rate": base_rate,
            "loss": float("nan"),
            "converged": False,
            "iterations": 0,
            "message": "single-class training fold; intercept only",
        }

    penalty = float(l2) / n
    offset_penalty = (
        float(STATION_L2_MULTIPLE * l2 if group_l2 is None else group_l2) / n
    )
    if offset_penalty < 0:
        raise ValueError(f"group_l2 must be >= 0, got {group_l2}")
    start = np.zeros(k + 1 + n_groups, dtype=np.float64)
    start[0] = math.log(base_rate / (1.0 - base_rate))

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = theta[0]
        weights = theta[1:k + 1]
        z = design @ weights + intercept
        if n_groups:
            offsets = np.r_[theta[k + 1:], 0.0]
            z = z + offsets[codes]
        # log(1 + exp(z)) computed as the stable max(z,0) + log1p(exp(-|z|)).
        loss = float(
            np.mean(np.maximum(z, 0.0) - z * outcome + np.log1p(np.exp(-np.abs(z))))
        )
        residual = (_sigmoid(z) - outcome) / n
        grad = np.empty_like(theta)
        grad[0] = float(residual.sum())
        grad[1:k + 1] = design.T @ residual + penalty * weights
        value = loss + 0.5 * penalty * float(weights @ weights)
        if n_groups:
            group_weights = theta[k + 1:]
            grad[k + 1:] = (
                np.bincount(codes, weights=residual, minlength=n_groups + 1)
                [:n_groups] + offset_penalty * group_weights
            )
            value += 0.5 * offset_penalty * float(group_weights @ group_weights)
        return value, grad

    result = minimize(
        objective, start, jac=True, method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": tol, "gtol": tol},
    )
    theta = np.asarray(result.x, dtype=np.float64)
    return {
        "intercept": float(theta[0]),
        "coefficients": [float(v) for v in theta[1:k + 1]],
        "group_offsets": [float(v) for v in theta[k + 1:]],
        "n": n,
        "base_rate": base_rate,
        "loss": float(result.fun),
        "converged": bool(result.success),
        "iterations": int(getattr(result, "nit", 0)),
        "message": str(getattr(result, "message", "")),
    }


# ---------------------------------------------------------------------------
# Isotonic recalibration of the logistic output
# ---------------------------------------------------------------------------

#: Knots in the stored isotonic curve. The logistic output is continuous,
#: so a knot per distinct value would put half a million numbers in the
#: artefact; binning by quantile first keeps the file readable and the
#: curve is interpolated between knots exactly as the served national
#: curves are.
DEFAULT_ISOTONIC_BINS = 200


def fit_isotonic_binned(
    p: np.ndarray, y: np.ndarray, *, n_bins: int = DEFAULT_ISOTONIC_BINS,
) -> IsotonicCalibrator:
    """Isotonic recalibration of a continuous score, on quantile bins.

    Same machinery as the served curves (``calibrate.pava_weighted``), fed
    bin means weighted by bin counts instead of one point per row. Ties
    and empty bins are folded away so the breakpoints are strictly
    increasing, which ``IsotonicCalibrator.predict``'s ``np.interp``
    requires.
    """
    prob = np.asarray(p, dtype=np.float64).reshape(-1)
    outcome = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(prob) & np.isfinite(outcome)
    prob, outcome = prob[keep], outcome[keep]
    if prob.size == 0:
        raise ValueError("no finite samples to calibrate on")
    if prob.size <= n_bins:
        edges = np.unique(prob)
    else:
        edges = np.unique(
            np.quantile(prob, np.linspace(0.0, 1.0, int(n_bins) + 1))
        )
    if edges.size < 2:
        # A constant score: the calibrated value is the base rate, and a
        # two-point flat curve says exactly that at every input.
        rate = float(outcome.mean())
        return IsotonicCalibrator(
            raw_breakpoints=(0.0, 1.0), calibrated_values=(rate, rate),
        )
    index = np.clip(
        np.searchsorted(edges, prob, side="right") - 1, 0, edges.size - 2,
    )
    counts = np.bincount(index, minlength=edges.size - 1).astype(np.float64)
    sum_p = np.bincount(index, weights=prob, minlength=edges.size - 1)
    sum_y = np.bincount(index, weights=outcome, minlength=edges.size - 1)
    occupied = counts > 0
    x = sum_p[occupied] / counts[occupied]
    observed = sum_y[occupied] / counts[occupied]
    weight = counts[occupied]
    # Bin means are ascending by construction, but float ties are possible;
    # merge them so np.interp gets a strictly increasing x.
    unique_x, inverse = np.unique(x, return_inverse=True)
    if unique_x.size != x.size:
        merged_w = np.bincount(inverse, weights=weight, minlength=unique_x.size)
        merged_y = np.bincount(
            inverse, weights=weight * observed, minlength=unique_x.size,
        ) / merged_w
        x, observed, weight = unique_x, merged_y, merged_w
    if x.size == 1:
        rate = float(observed[0])
        return IsotonicCalibrator(
            raw_breakpoints=(0.0, 1.0), calibrated_values=(rate, rate),
        )
    fitted = pava_weighted(observed, weight)
    return IsotonicCalibrator(
        raw_breakpoints=tuple(float(v) for v in x),
        calibrated_values=tuple(float(v) for v in fitted),
    )


#: Rows a season needs before it gets a calibration curve of its own. A
#: season below this keeps the pooled one: an isotonic fit on a few
#: hundred rows is a step function through the noise, and it would be
#: served to everyone in that season for three months.
MIN_SEASON_ISOTONIC_ROWS = 5000


def fit_isotonic_by_season(
    p: np.ndarray,
    y: np.ndarray,
    season: np.ndarray,
    *,
    n_bins: int = DEFAULT_ISOTONIC_BINS,
    min_rows: int = MIN_SEASON_ISOTONIC_ROWS,
) -> tuple[IsotonicCalibrator, dict[str, IsotonicCalibrator]]:
    """``(pooled curve, {season: curve})`` — recalibration, per season.

    The reliability of the *model's* score is not seasonal in the same way
    the model is: a summer 0.6 and a winter 0.6 come out of different
    regions of the design and land on different observed frequencies, and
    one curve over both splits the difference. One curve per season fixes
    that where there is enough data to fit one.

    A season with fewer than ``min_rows`` rows is simply absent from the
    returned mapping, and :class:`LeadModel` falls back to the pooled
    curve for it. The pooled curve is always fitted, so there is always
    something to fall back to.
    """
    prob = np.asarray(p, dtype=np.float64).reshape(-1)
    outcome = np.asarray(y, dtype=np.float64).reshape(-1)
    labels = np.asarray(season).astype("<U8").reshape(-1)
    pooled = fit_isotonic_binned(prob, outcome, n_bins=n_bins)
    curves: dict[str, IsotonicCalibrator] = {}
    for name in SEASONS:
        keep = labels == name
        if int(keep.sum()) < int(min_rows):
            continue
        try:
            curves[name] = fit_isotonic_binned(
                prob[keep], outcome[keep], n_bins=n_bins,
            )
        except ValueError:
            # No finite sample in this season: keep the pooled curve
            # rather than an empty one.
            continue
    return pooled, curves


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeadModel:
    """One lead's fit plus the isotonic map on top of it.

    The fit is a logistic (``coefficients`` over the standardised design,
    plus the optional per-station offsets) or a boosted ensemble
    (``trees`` over the RAW design) — :attr:`kind` on the owning
    :class:`PostprocessModel` says which, and everything below branches on
    whether ``trees`` is there. The isotonic map is the same either way,
    because a tree ensemble's output needs recalibrating for exactly the
    reason a logistic's does: it was fitted to minimise loss, not to be
    reliable in the top bin.
    """

    lead_min: int
    intercept: float
    coefficients: tuple[float, ...]
    isotonic: IsotonicCalibrator
    n: int
    base_rate: float
    converged: bool
    iterations: int
    message: str = ""
    #: The boosted ensemble, when this lead is a tree model. A
    #: :class:`~dmi_nowcast_core.postprocess_trees.TreeEnsemble`, typed
    #: loosely so importing this module never imports that one.
    trees: Any | None = None
    #: ``{station_id: offset}`` added to the linear predictor for rows at
    #: that station. Empty unless the fit was asked for station effects; a
    #: station absent from the map — every subscriber's own coordinate —
    #: gets 0.
    station_offsets: Mapping[str, float] = field(default_factory=dict)
    #: One recalibration curve per season, where a season had the rows to
    #: earn it. A season absent here uses :attr:`isotonic`.
    isotonic_by_season: Mapping[str, IsotonicCalibrator] = field(
        default_factory=dict,
    )

    def offsets_for(self, stations: Any | None) -> np.ndarray | float:
        """Per-row station offset, or the scalar 0.0 when there is none."""
        if stations is None or not self.station_offsets:
            return 0.0
        labels = np.asarray(stations).astype(str).reshape(-1)
        # np.unique once, a dict lookup per DISTINCT station: at half a
        # million rows and a hundred stations the per-row dict lookup
        # would be most of the scoring time.
        distinct, inverse = np.unique(labels, return_inverse=True)
        table = np.array(
            [float(self.station_offsets.get(str(s), 0.0)) for s in distinct],
            dtype=np.float64,
        )
        return table[inverse]

    def score(
        self, design: np.ndarray, stations: Any | None = None,
    ) -> np.ndarray:
        """The model's probability, before recalibration.

        ``design`` is standardised for a logistic and raw for a tree
        ensemble — :meth:`PostprocessModel.design` produces the right one.
        """
        if self.trees is not None:
            return np.asarray(self.trees.predict_proba(design), dtype=np.float64)
        weights = np.asarray(self.coefficients, dtype=np.float64)
        return _sigmoid(
            design @ weights + self.intercept + self.offsets_for(stations)
        )

    def calibrate(
        self, raw: np.ndarray, season: Any | None = None,
    ) -> np.ndarray:
        """Apply the isotonic map — the per-season one where there is one."""
        out = np.asarray(self.isotonic.predict(raw), dtype=np.float64)
        if season is not None and self.isotonic_by_season:
            labels = np.asarray(season).astype("<U8").reshape(-1)
            for name, curve in self.isotonic_by_season.items():
                keep = labels == name
                if keep.any():
                    out[keep] = np.asarray(
                        curve.predict(raw[keep]), dtype=np.float64,
                    )
        return np.clip(out, 0.0, 1.0)

    def predict(
        self,
        design: np.ndarray,
        stations: Any | None = None,
        season: Any | None = None,
    ) -> np.ndarray:
        """The calibrated probability — what would be served."""
        return self.calibrate(self.score(design, stations), season)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
            "isotonic": {
                "raw_breakpoints": list(self.isotonic.raw_breakpoints),
                "calibrated_values": list(self.isotonic.calibrated_values),
            },
            "n": self.n,
            "base_rate": self.base_rate,
            "converged": self.converged,
            "iterations": self.iterations,
            "message": self.message,
        }
        # Every block below is additive and written only when it carries
        # something, so a default fit's document is the document this
        # module has always written.
        if self.station_offsets:
            out["station_offsets"] = {
                str(k): float(v) for k, v in sorted(self.station_offsets.items())
            }
        if self.isotonic_by_season:
            out["isotonic_by_season"] = {
                str(name): {
                    "raw_breakpoints": list(curve.raw_breakpoints),
                    "calibrated_values": list(curve.calibrated_values),
                }
                for name, curve in sorted(self.isotonic_by_season.items())
            }
        if self.trees is not None:
            out["trees"] = self.trees.to_json()
        return out

    @classmethod
    def from_json(
        cls, lead: int, raw: Mapping[str, Any], *, trees: Any | None = None,
    ) -> "LeadModel":
        """``trees`` is the ensemble shared across leads, when there is one."""
        own = raw.get("trees")
        if own is not None:
            from .postprocess_trees import TreeEnsemble

            trees = TreeEnsemble.from_json(own)
        return cls(
            lead_min=int(lead),
            intercept=float(raw["intercept"]),
            coefficients=tuple(float(v) for v in raw["coefficients"]),
            isotonic=_curve_from_json(raw["isotonic"]),
            n=int(raw.get("n", 0)),
            base_rate=float(raw.get("base_rate", float("nan"))),
            converged=bool(raw.get("converged", False)),
            iterations=int(raw.get("iterations", 0)),
            message=str(raw.get("message", "")),
            trees=trees,
            station_offsets={
                str(k): float(v)
                for k, v in (raw.get("station_offsets") or {}).items()
            },
            isotonic_by_season={
                str(name): _curve_from_json(entry)
                for name, entry in (raw.get("isotonic_by_season") or {}).items()
            },
        )


def _curve_from_json(raw: Mapping[str, Any]) -> IsotonicCalibrator:
    return IsotonicCalibrator(
        raw_breakpoints=tuple(float(v) for v in raw["raw_breakpoints"]),
        calibrated_values=tuple(float(v) for v in raw["calibrated_values"]),
    )


@dataclass(frozen=True)
class PostprocessModel:
    """One fitted post-processor: the transform, and a model per lead.

    Serialised through :meth:`to_json` into ``postprocess.json``, the
    stable artefact the serving path would read. The document carries what
    it takes to reproduce a prediction — the design's column names in
    order, the standardiser, every coefficient, the isotonic knots — and
    what it takes to judge one: the training window, the row and station
    counts, and the base rate behind each lead.
    """

    leads: tuple[int, ...]
    design_leads: tuple[int, ...]
    feature_names: tuple[str, ...]
    standardiser: Standardiser
    models: dict[int, LeadModel]
    l2: float
    fitted_at_utc: str
    training: dict[str, Any]
    #: ``"logistic"`` or ``"trees"``. Decides whether :meth:`design`
    #: standardises (and imputes) or hands the raw matrix over. Additive
    #: with a default, so a document written before it existed loads as
    #: the logistic it is.
    kind: str = KIND_LOGISTIC
    #: How the design was built. The default spec IS the v1 design.
    spec: DesignSpec = field(default_factory=DesignSpec)
    #: One ensemble answering for every lead (``--model trees-shared``),
    #: stored once. Per-lead ensembles live on their :class:`LeadModel`.
    shared_trees: Any | None = None
    #: Which validation protocol fitted this document —
    #: :data:`PROTOCOL_AT_GAUGE` or :data:`PROTOCOL_RANDOM_POINT`. Not
    #: provenance: under ``random-point`` the fit never saw the point's
    #: own gauge, so serving has to take it away too. See
    #: :meth:`masked_features`. Additive with the historical default, so
    #: a document written before this field existed loads as the at-gauge
    #: model it is.
    protocol: str = PROTOCOL_AT_GAUGE
    #: What the model was fitted to predict — :data:`TARGET_WET` or
    #: :data:`TARGET_ONSET`. Provenance for serving (nothing branches on
    #: it), but part of the document's contract, like ``protocol``: an
    #: onset model's number is P(onset in the scorer window), a different
    #: quantity from the wet model's, and a threshold fitted on one means
    #: nothing on the other. Additive with the historical default.
    target: str = TARGET_WET

    # -- prediction ---------------------------------------------------------

    @property
    def is_trees(self) -> bool:
        return is_tree_kind(self.kind)

    @property
    def is_shared(self) -> bool:
        """One fit over every lead, with the lead in the design."""
        return bool(self.spec.lead_column) or is_shared_kind(self.kind)

    @property
    def masks_own_gauge(self) -> bool:
        """Does serving this model hide the point's own gauge from it?

        True exactly for :data:`PROTOCOL_RANDOM_POINT`. An unrecognised
        protocol string reads as False — the at-gauge behaviour, which is
        what every document written before the field existed meant.
        """
        return str(self.protocol) == PROTOCOL_RANDOM_POINT

    def masked_features(self, features: Mapping[str, Any]) -> Mapping[str, Any]:
        """``features`` as this model is entitled to see them.

        Under ``random-point`` that is :func:`mask_own_gauge` — the very
        transform ``scripts/fit_postprocess.py`` applied to the training
        table — and under ``at-gauge`` it is ``features`` itself, the same
        object, so nothing about an at-gauge document changes.

        Serving a random-point model on unmasked rows would be the exact
        train/serve skew the protocol exists to remove: at a gauge it
        would hand the model a live ``g_min_since_wet`` that every row it
        was fitted on said was unknown, and the number it returned there
        would stop being the number a subscriber's address gets.

        For the artefacts fitted so far this is a guard rather than a
        correction, and deliberately so: masking the whole table before
        the fit leaves the own-gauge columns constant, so a logistic gives
        them a coefficient of exactly zero and the shipped tree ensembles
        never split on them (94-96 of 104 columns used, none of the six).
        The property that matters is that a random-point document is blind
        to the own gauge by construction, not by the luck of how this fit
        happened to go — a fold-local mask, or a protocol that masks after
        some feature is derived, would leave weights that CAN read it.
        """
        if not self.masks_own_gauge:
            return features
        return mask_own_gauge(features)

    def design(self, features: Mapping[str, Any]) -> np.ndarray:
        """The design matrix for stored feature columns.

        Standardised and mean-imputed for a logistic; RAW for a tree
        model, because a tree is invariant to a monotone rescaling and
        learns its own direction for a missing value — imputing first
        would throw that away and tell it a mean where the cycle knew
        nothing.

        The own-gauge mask is applied HERE, before the columns become a
        matrix, so every path that scores through this model — the cycle's
        table, the single row, the on-demand lookup — is masked by
        construction and none of them has to remember to be.
        """
        design = build_design(
            self.masked_features(features), self.design_leads, self.spec,
        )
        if self.is_trees:
            return design
        return self.standardiser.transform(design)

    def predict(
        self,
        features_by_column: Mapping[str, Any],
        lead: int | None = None,
        *,
        stations: Any | None = None,
    ) -> Any:
        """Post-processed probability from stored feature columns.

        ``lead=None`` returns ``{lead: array}`` for every fitted lead;
        a lead returns that lead's array. One design matrix is built and
        shared, because it does not depend on the lead being predicted —
        except for the shared-lead column, which is written in place per
        lead rather than copying the matrix.

        ``stations`` (falling back to a ``station_id`` column in the
        features) selects the learned per-station offsets; a point the
        model never saw simply gets none.
        """
        if lead is not None and int(lead) not in self.models:
            raise KeyError(f"no model for lead {lead}")
        design = self.design(features_by_column)
        if stations is None:
            stations = features_by_column.get("station_id")
        season = features_by_column.get("season")
        # EVERY lead, even when one was asked for: the running max below
        # is a statement about the set, and a single-lead answer that
        # skipped it would differ from the same lead inside a full call.
        out: dict[int, np.ndarray] = {}
        for key in sorted(self.models):
            self._fill_lead_columns(design, key)
            out[int(key)] = self.models[int(key)].predict(
                design, stations, season,
            )
        out = enforce_lead_monotonic(out)
        if lead is not None:
            return out[int(lead)]
        return out

    def _fill_lead_columns(self, design: np.ndarray, lead: int) -> None:
        """Write this lead into the shared design's per-lead columns.

        ``lead_min`` and ``log_lead_min`` become the horizon, and every
        ``<family>_own`` column is copied from that family's column for
        THIS lead — ``raw_frac_own`` from ``raw_frac_45`` when predicting
        45 minutes. In place, because the alternative is a full copy of
        the design per lead and the design is the largest array here.
        """
        if not self.spec.lead_column:
            return
        index = self._column_index()
        for name, sources in self.spec.own_columns:
            target = index.get(name)
            source = dict(sources).get(int(lead))
            if target is None:
                continue
            origin = index.get(str(source))
            design[:, target] = (
                np.nan if origin is None else design[:, origin]
            )
        if LEAD_COLUMN in index:
            design[:, index[LEAD_COLUMN]] = float(lead)
        if LOG_LEAD_COLUMN in index:
            design[:, index[LOG_LEAD_COLUMN]] = math.log(float(lead))

    def _column_index(self) -> dict[str, int]:
        cached = self.__dict__.get("_index")
        if cached is None:
            cached = {name: i for i, name in enumerate(self.feature_names)}
            object.__setattr__(self, "_index", cached)
        return cached

    # -- persistence --------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "fitted_at_utc": self.fitted_at_utc,
            "leads": list(self.leads),
            "l2": self.l2,
            "kind": self.kind,
            # Top level, beside ``kind``, and not only inside the
            # free-form ``training`` block: the serving path branches on
            # it, so it is part of the contract the document states, not
            # a note about how the document came to be.
            "protocol": self.protocol,
            # Beside ``protocol`` for the same reason: which outcome the
            # numbers are probabilities OF is part of what the document
            # states, not a note in the free-form provenance.
            "target": self.target,
            "design": self.spec.to_json(),
            "features": {
                "design_leads": list(self.design_leads),
                "names": list(self.feature_names),
                "source_columns": list(DESIGN_SOURCE_COLUMNS),
                "definitions": {
                    name: text for name, text in feature_columns(self.design_leads)
                },
            },
            "scaling": {
                "mean": list(self.standardiser.mean),
                "scale": list(self.standardiser.scale),
            },
            "training": dict(self.training),
            "models": {
                str(lead): self.models[lead].to_json()
                for lead in sorted(self.models)
            },
        }
        if self.shared_trees is not None:
            out["shared_trees"] = self.shared_trees.to_json()
        return out

    def dumps(self, *, indent: int = 1) -> str:
        return json.dumps(self.to_json(), indent=indent, default=str) + "\n"

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "PostprocessModel":
        version = int(raw.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"postprocess document schema_version {version}, "
                f"expected {SCHEMA_VERSION}"
            )
        features = raw["features"]
        scaling = raw["scaling"]
        shared = raw.get("shared_trees")
        shared_trees = None
        if shared is not None:
            from .postprocess_trees import TreeEnsemble

            shared_trees = TreeEnsemble.from_json(shared)
        models = {
            int(lead): LeadModel.from_json(int(lead), entry, trees=shared_trees)
            for lead, entry in raw["models"].items()
        }
        training = dict(raw.get("training") or {})
        # Top-level key first; then the provenance block, which is where
        # the protocol lived before the key existed and is therefore the
        # only place the already-fitted random-point artefacts say it;
        # then the historical default.
        protocol = str(
            raw.get("protocol")
            or training.get("protocol")
            or PROTOCOL_AT_GAUGE
        )
        # Same order as ``protocol``: the top-level key, then the
        # provenance block, then the historical default.
        target = str(
            raw.get("target")
            or training.get("target")
            or TARGET_WET
        )
        return cls(
            leads=tuple(int(v) for v in raw["leads"]),
            design_leads=tuple(int(v) for v in features["design_leads"]),
            feature_names=tuple(str(v) for v in features["names"]),
            standardiser=Standardiser(
                mean=tuple(float(v) for v in scaling["mean"]),
                scale=tuple(float(v) for v in scaling["scale"]),
            ),
            models=models,
            l2=float(raw.get("l2", 1.0)),
            fitted_at_utc=str(raw.get("fitted_at_utc", "")),
            training=training,
            kind=str(raw.get("kind", KIND_LOGISTIC)),
            spec=DesignSpec.from_json(raw.get("design")),
            shared_trees=shared_trees,
            protocol=protocol,
            target=target,
        )

    @classmethod
    def loads(cls, text: str) -> "PostprocessModel":
        return cls.from_json(json.loads(text))


def _usable_outcome(
    truth: Mapping[int, Any], lead: int, n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``(y, usable)`` for one lead, from either supported truth shape.

    ``truth[lead]`` is the ``(outcome, usable)`` pair
    ``benchmark_report.GaugeGrid.outcome`` returns, or a bare outcome array
    with NaN where the window could not be graded.
    """
    entry = truth[int(lead)]
    if isinstance(entry, tuple):
        y = np.asarray(entry[0], dtype=np.float64).reshape(-1)
        usable = np.asarray(entry[1], dtype=bool).reshape(-1)
    else:
        y = np.asarray(entry, dtype=np.float64).reshape(-1)
        usable = np.isfinite(y)
    if y.size != n or usable.size != n:
        raise ValueError(
            f"truth for lead {lead} has {y.size} rows, features have {n}"
        )
    return np.nan_to_num(y, nan=0.0), usable


@dataclass(frozen=True)
class FitSettings:
    """Everything about a fit that is a CHOICE rather than an input.

    One value passed down through :func:`fit_postprocess`,
    :func:`leave_one_month_out`, :func:`learning_curve` and
    :func:`ablation`, so a configuration that was evaluated out-of-fold is
    the configuration that gets fitted and shipped. The defaults are the
    shipped model exactly — logistic, v1 design, no station offsets, one
    pooled isotonic curve per lead — and ``tests/test_postprocess_model.py``
    pins that.
    """

    kind: str = KIND_LOGISTIC
    design: str = DESIGN_V1
    l2: float = 1.0
    isotonic: str = ISOTONIC_POOLED
    isotonic_bins: int = DEFAULT_ISOTONIC_BINS
    #: Learn a per-station intercept offset (ridge ``station_l2_multiple ×
    #: l2``) alongside the slopes. Logistic only: a tree ensemble can read
    #: the static station features directly.
    station_offsets: bool = False
    station_l2_multiple: float = STATION_L2_MULTIPLE
    #: LightGBM overrides, on top of
    #: :data:`postprocess_trees.DEFAULT_TREE_PARAMS`.
    tree_params: Mapping[str, Any] | None = None
    seed: int = 0
    #: Feature families to leave out — the ablation's one knob.
    drop_families: tuple[str, ...] = ()

    @property
    def shared_lead(self) -> bool:
        """One model over every lead, with the lead in the design.

        Derived from :attr:`kind` rather than stored beside it: two
        sources of truth for "is this shared?" is exactly the sort of
        thing that ships a model whose design and whose fit disagree.
        """
        return is_shared_kind(self.kind)

    @property
    def is_trees(self) -> bool:
        return is_tree_kind(self.kind)

    def validate(self) -> "FitSettings":
        if self.kind not in MODEL_KINDS:
            raise ValueError(
                f"unknown model kind {self.kind!r}; expected one of "
                f"{', '.join(MODEL_KINDS)}"
            )
        if self.design not in DESIGN_VERSIONS:
            raise ValueError(f"unknown design version {self.design!r}")
        if self.isotonic not in ISOTONIC_MODES:
            raise ValueError(f"unknown isotonic mode {self.isotonic!r}")
        if self.l2 < 0:
            raise ValueError(f"l2 must be >= 0, got {self.l2}")
        unknown = [
            name for name in self.drop_families
            if name not in FAMILY_NAMES and name != "other"
        ]
        if unknown:
            raise ValueError(f"unknown feature famil(ies) {unknown}")
        return self

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "design": self.design,
            "l2": float(self.l2),
            "isotonic": self.isotonic,
            "isotonic_bins": int(self.isotonic_bins),
            "station_offsets": bool(self.station_offsets),
            "station_l2_multiple": float(self.station_l2_multiple),
            # Derived, but written out: a reader of the report should not
            # have to know the naming rule to see what was fitted.
            "shared_lead": bool(self.shared_lead),
            "tree_params": dict(self.tree_params or {}),
            "seed": int(self.seed),
            "drop_families": list(self.drop_families),
        }


def _station_codes(
    stations: Any | None, known: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """``(code per row, station id per code)``; ``-1`` for "not one of ours"."""
    if stations is None:
        return np.empty(0, dtype=np.int64), []
    labels = np.asarray(stations).astype(str).reshape(-1)
    order = list(known) if known is not None else sorted(set(labels.tolist()))
    lookup = {name: index for index, name in enumerate(order)}
    distinct, inverse = np.unique(labels, return_inverse=True)
    table = np.array(
        [lookup.get(str(name), -1) for name in distinct], dtype=np.int64,
    )
    return table[inverse], order


def _column_mask(names: Sequence[str], drop_families: Sequence[str]) -> np.ndarray:
    """Which design columns survive an ablation. All of them, by default."""
    keep = np.ones(len(names), dtype=bool)
    for family in drop_families:
        keep &= ~family_columns(names, family)
    return keep


def _fit_shared(
    x_all: np.ndarray,
    truth: Mapping[int, Any],
    leads: Sequence[int],
    n: int,
    names: Sequence[str],
    settings: FitSettings,
    spec: DesignSpec,
    *,
    stations: Any | None = None,
) -> tuple[Any | None, dict[str, Any] | None]:
    """``(shared ensemble, shared logistic fit)`` — at most one is not None."""
    if not settings.shared_lead:
        return None, None
    if settings.is_trees:
        return _fit_shared_trees(
            x_all, truth, leads, n, names, settings, spec,
        ), None
    return None, _fit_shared_logistic(
        x_all, truth, leads, n, names, settings, spec, stations=stations,
    )


def fit_lead(
    design: np.ndarray,
    y: np.ndarray,
    lead: int,
    *,
    settings: FitSettings,
    standardiser: Standardiser | None,
    stations: Any | None = None,
    season: Any | None = None,
    feature_names: Sequence[str] = (),
    shared_trees: Any | None = None,
    shared_logistic: Mapping[str, Any] | None = None,
) -> LeadModel:
    """One lead's whole two-stage fit, on rows already reduced to gradable.

    THE fit, in one place: :func:`fit_postprocess` (all rows) and every
    fold of :func:`leave_one_month_out` (in-fold) call this, so an
    out-of-fold number is about the model that would ship and not about a
    second implementation of it.

    ``design`` is standardised for a logistic and raw for a tree model;
    the caller does the transform once for every lead because it does not
    depend on the lead.
    """
    if settings.is_trees:
        from .postprocess_trees import fit_trees

        if shared_trees is None:
            fit = fit_trees(
                design, y,
                params=settings.tree_params,
                feature_names=feature_names,
                seed=int(settings.seed),
            )
            ensemble = fit["ensemble"]
            counts = {
                "n": int(fit["n"]), "base_rate": float(fit["base_rate"]),
                "iterations": int(fit["n_trees"]),
                "message": str(fit.get("message", "")),
            }
        else:
            ensemble = shared_trees
            counts = {
                "n": int(np.size(y)), "base_rate": float(np.mean(y)),
                "iterations": shared_trees.n_trees,
                "message": "shared ensemble",
            }
        model_only = LeadModel(
            lead_min=int(lead), intercept=0.0, coefficients=(),
            isotonic=IsotonicCalibrator((0.0, 1.0), (0.0, 1.0)),
            converged=True, trees=ensemble, **counts,
        )
        raw = model_only.score(design)
        pooled, by_season = _calibrators(raw, y, season, settings)
        return LeadModel(
            lead_min=int(lead), intercept=0.0, coefficients=(),
            isotonic=pooled, isotonic_by_season=by_season,
            converged=True, trees=ensemble, **counts,
        )

    if shared_logistic is not None:
        # One coefficient vector, fitted once over every lead's rows. Only
        # the recalibration below is this lead's own.
        fit = dict(shared_logistic)
        order = list(fit.get("station_order") or ())
        fit["n"] = int(np.size(y))
        fit["base_rate"] = float(np.mean(y))
        fit["message"] = "shared across leads"
    else:
        codes, order = (
            _station_codes(stations) if settings.station_offsets
            else (np.empty(0, dtype=np.int64), [])
        )
        fit = fit_logistic(
            design, y, l2=float(settings.l2),
            groups=codes if order else None,
            n_groups=len(order),
            group_l2=float(settings.station_l2_multiple) * float(settings.l2),
        )
    offsets = {
        name: float(value)
        for name, value in zip(order, fit.get("group_offsets") or ())
    }
    common = {
        "lead_min": int(lead),
        "intercept": fit["intercept"],
        "coefficients": tuple(fit["coefficients"]),
        "n": fit["n"],
        "base_rate": fit["base_rate"],
        "converged": fit["converged"],
        "iterations": fit["iterations"],
        "message": fit["message"],
        "station_offsets": offsets,
    }
    model_only = LeadModel(
        isotonic=IsotonicCalibrator((0.0, 1.0), (0.0, 1.0)), **common,
    )
    raw = model_only.score(design, stations if offsets else None)
    pooled, by_season = _calibrators(raw, y, season, settings)
    return LeadModel(isotonic=pooled, isotonic_by_season=by_season, **common)


def _calibrators(
    raw: np.ndarray,
    y: np.ndarray,
    season: Any | None,
    settings: FitSettings,
) -> tuple[IsotonicCalibrator, dict[str, IsotonicCalibrator]]:
    if settings.isotonic == ISOTONIC_PER_SEASON and season is not None:
        return fit_isotonic_by_season(
            raw, y, season, n_bins=int(settings.isotonic_bins),
        )
    return (
        fit_isotonic_binned(raw, y, n_bins=int(settings.isotonic_bins)),
        {},
    )


def fit_postprocess(
    rows: Mapping[str, Any],
    truth: Mapping[int, Any],
    leads: Sequence[int],
    *,
    l2: float = 1.0,
    design_leads: Sequence[int] | None = None,
    training: Mapping[str, Any] | None = None,
    isotonic_bins: int = DEFAULT_ISOTONIC_BINS,
    fitted_at: datetime | None = None,
    settings: FitSettings | None = None,
    stations: Any | None = None,
    protocol: str = PROTOCOL_AT_GAUGE,
    target: str = TARGET_WET,
) -> PostprocessModel:
    """Fit one model + isotonic map per lead on ``rows``.

    ``rows`` maps stored column names to equal-length arrays (the replay's
    feature columns, plus ``season``/``hour_utc`` derived from the decision
    instant). ``truth`` maps a lead to the Layer B outcome for the same
    rows — the ``(outcome, usable)`` pair
    ``benchmark_report.GaugeGrid.outcome`` returns, so the definition and
    the exclusions are the report's, not a second opinion.

    ``settings`` picks the family, the design, the recalibration and the
    station effects; leaving it out (with ``l2`` / ``isotonic_bins`` as
    before) is the shipped logistic on the v1 design, unchanged.

    ``protocol`` is the validation protocol the CALLER fitted under, and
    it is recorded on the model because serving has to honour it: it is
    the caller that masked the rows (``mask_own_gauge`` over the whole
    table, before any fold is cut), and nothing about a table of numbers
    says afterwards that it was masked. Passing ``random-point`` here
    without having masked ``rows`` would produce a model that is served
    blind to a gauge it was taught to read — so the flag and the mask
    belong to the same call site, which is
    ``scripts/fit_postprocess.py``.

    ``target`` names the outcome ``truth`` encodes — :data:`TARGET_WET`
    or :data:`TARGET_ONSET` — and, like ``protocol``, is recorded rather
    than acted on: the caller built ``truth``, and nothing about a pair
    of arrays says afterwards which event they were labels for.

    Both stages of each lead are fitted on the SAME rows: the model first,
    then the isotonic map of its output. That is in-sample for the
    isotonic, which is why nothing here is a result — the number that
    counts comes from :func:`leave_one_month_out`, where the whole
    two-stage fit sits inside the fold.
    """
    chosen = (
        FitSettings(l2=float(l2), isotonic_bins=int(isotonic_bins))
        if settings is None else settings
    ).validate()
    wanted = tuple(sorted({int(lead) for lead in leads}))
    if not wanted:
        raise ValueError("no leads to fit")
    chosen_protocol = str(protocol)
    if chosen_protocol not in PROTOCOLS:
        raise ValueError(
            f"unknown protocol {chosen_protocol!r}; expected one of "
            f"{', '.join(PROTOCOLS)}"
        )
    chosen_target = str(target)
    if chosen_target not in TARGETS:
        raise ValueError(
            f"unknown target {chosen_target!r}; expected one of "
            f"{', '.join(TARGETS)}"
        )
    missing = [lead for lead in wanted if lead not in truth]
    if missing:
        raise ValueError(f"no truth for lead(s) {missing}")
    in_design = tuple(sorted({int(x) for x in (design_leads or wanted)}))
    n = _rows_in(rows)
    if stations is None:
        stations = rows.get("station_id")
    season = rows.get("season")

    spec = design_spec(
        rows, in_design, version=chosen.design, shared_lead=chosen.shared_lead,
    )
    names = design_columns(in_design, spec)
    design = build_design(rows, in_design, spec)
    keep = _column_mask(names, chosen.drop_families)
    if not keep.all():
        design = design[:, keep]
        names = tuple(name for name, ok in zip(names, keep) if ok)
    standardiser = Standardiser.fit(design)
    x_all = design if chosen.is_trees else standardiser.transform(design)
    del design

    shared_trees, shared_logistic = _fit_shared(
        x_all, truth, wanted, n, names, chosen, spec, stations=stations,
    )

    models: dict[int, LeadModel] = {}
    for lead in wanted:
        y, usable = _usable_outcome(truth, lead, n)
        if not usable.any():
            raise ValueError(f"lead {lead} has no gradable rows")
        x_train = x_all[usable]
        if chosen.shared_lead:
            x_train = x_train.copy()
            fill_shared_columns(x_train, lead, names, spec)
        models[lead] = fit_lead(
            x_train, y[usable], lead,
            settings=chosen, standardiser=standardiser,
            stations=None if stations is None else np.asarray(stations)[usable],
            season=None if season is None else np.asarray(season)[usable],
            feature_names=names,
            shared_trees=shared_trees,
            shared_logistic=shared_logistic,
        )
    stamp = (fitted_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    provenance = dict(training or {"rows": n})
    provenance.setdefault("settings", chosen.to_json())
    # The provenance block has carried the protocol since the random-point
    # track opened; keep writing it, so a reader of either half of the
    # document sees the same answer.
    provenance.setdefault("protocol", chosen_protocol)
    provenance.setdefault("target", chosen_target)
    return PostprocessModel(
        leads=wanted,
        design_leads=in_design,
        feature_names=tuple(names),
        standardiser=standardiser,
        models=models,
        l2=float(chosen.l2),
        fitted_at_utc=stamp.isoformat(timespec="seconds"),
        training=provenance,
        kind=chosen.kind,
        spec=spec,
        shared_trees=shared_trees,
        protocol=chosen_protocol,
        target=chosen_target,
    )


def fill_shared_columns(
    block: np.ndarray, lead: int, names: Sequence[str], spec: DesignSpec,
) -> None:
    """Write one lead into a shared design block, in place.

    The fit-time twin of :meth:`PostprocessModel._fill_lead_columns`, and
    deliberately the same three lines: if the fit filled ``raw_frac_own``
    from a different column than serving does, every coefficient in the
    model is about a predictor the service never supplies.
    """
    if not spec.lead_column:
        return
    index = {name: i for i, name in enumerate(names)}
    for name, sources in spec.own_columns:
        target = index.get(name)
        origin = index.get(str(dict(sources).get(int(lead))))
        if target is None:
            continue
        block[:, target] = np.nan if origin is None else block[:, origin]
    if LEAD_COLUMN in index:
        block[:, index[LEAD_COLUMN]] = float(lead)
    if LOG_LEAD_COLUMN in index:
        block[:, index[LOG_LEAD_COLUMN]] = math.log(float(lead))


def _stack_leads(
    x_all: np.ndarray,
    truth: Mapping[int, Any],
    leads: Sequence[int],
    n: int,
    names: Sequence[str],
    spec: DesignSpec,
    *,
    stations: Any | None = None,
    season: Any | None = None,
) -> dict[str, Any]:
    """Every lead's gradable rows, stacked, with the lead columns written.

    The stack is the point of a shared fit: a model that has seen lead 20
    and lead 60 of the same cycle learns how the answer bends with the
    horizon, instead of learning it four times from a quarter of the rows
    each. The price is one matrix four times as tall, which is why the
    lead columns are written per block rather than the whole design being
    copied per lead.
    """
    if not spec.lead_column:
        raise ValueError("a shared fit needs a lead column in the design")
    blocks: list[np.ndarray] = []
    outcomes: list[np.ndarray] = []
    station_blocks: list[np.ndarray] = []
    season_blocks: list[np.ndarray] = []
    for lead in leads:
        y, usable = _usable_outcome(truth, lead, n)
        if not usable.any():
            continue
        block = x_all[usable].copy()
        fill_shared_columns(block, int(lead), names, spec)
        blocks.append(block)
        outcomes.append(y[usable])
        if stations is not None:
            station_blocks.append(np.asarray(stations)[usable])
        if season is not None:
            season_blocks.append(np.asarray(season)[usable])
    if not blocks:
        raise ValueError("no gradable rows at any lead")
    return {
        "x": np.concatenate(blocks),
        "y": np.concatenate(outcomes),
        "stations": np.concatenate(station_blocks) if station_blocks else None,
        "season": np.concatenate(season_blocks) if season_blocks else None,
    }


def _lead_monotone_constraints(names: Sequence[str]) -> list[int]:
    """``+1`` on the lead columns, ``0`` everywhere else.

    P(rain within L) is non-decreasing in L by definition — the 60-minute
    window contains the 45-minute one — and a shared tree model is the one
    arm that could learn otherwise from noise, because the lead is a
    feature it splits on. LightGBM enforces the constraint during
    training, which is better than the running max at scoring time: the
    running max repairs the answer, the constraint stops the ensemble
    spending capacity on a shape that cannot be real. Both are applied.
    """
    return [1 if name in LEAD_COLUMNS else 0 for name in names]


def _fit_shared_trees(
    x_all: np.ndarray,
    truth: Mapping[int, Any],
    leads: Sequence[int],
    n: int,
    names: Sequence[str],
    settings: FitSettings,
    spec: DesignSpec,
) -> Any:
    """One ensemble over every lead's rows stacked, with the lead as a column."""
    from .postprocess_trees import fit_trees

    stacked = _stack_leads(x_all, truth, leads, n, names, spec)
    fit = fit_trees(
        stacked["x"], stacked["y"],
        params=settings.tree_params, feature_names=names,
        seed=int(settings.seed),
        monotone_constraints=_lead_monotone_constraints(names),
    )
    return fit["ensemble"]


def _fit_shared_logistic(
    x_all: np.ndarray,
    truth: Mapping[int, Any],
    leads: Sequence[int],
    n: int,
    names: Sequence[str],
    settings: FitSettings,
    spec: DesignSpec,
    *,
    stations: Any | None = None,
) -> dict[str, Any]:
    """One coefficient vector over every lead's rows stacked.

    Every slope is shared; what the lead changes is the value in
    ``lead_min`` / ``log_lead_min`` and which per-lead column lands in
    ``raw_frac_own`` and its siblings. The per-station offsets are shared
    too — a gauge that reads wet is wet at every horizon.

    Only the recalibration stays per lead, and it has to: the isotonic map
    is where the model's score becomes a probability, and the base rate it
    is mapping onto is four times higher at 60 minutes than at 20.
    """
    stacked = _stack_leads(
        x_all, truth, leads, n, names, spec, stations=stations,
    )
    codes, order = (
        _station_codes(stacked["stations"]) if settings.station_offsets
        else (np.empty(0, dtype=np.int64), [])
    )
    return fit_logistic(
        stacked["x"], stacked["y"], l2=float(settings.l2),
        groups=codes if order else None,
        n_groups=len(order),
        group_l2=float(settings.station_l2_multiple) * float(settings.l2),
    ) | {"station_order": order}


# ---------------------------------------------------------------------------
# Leave-one-(year, month)-out evaluation
# ---------------------------------------------------------------------------

#: The pooled stratum's label, matching the sweep's CSV and the Layer B
#: report so one name means one thing across the project.
POOLED = "all"

#: Reliability bins, ten, ``[k/K, (k+1)/K)`` with p == 1 in the last —
#: ``benchmark.reliability_bins``' convention and DuckDB's.
N_BINS = 10


def _scores(p: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = dict(brier_decomposition(p, y, n_bins=N_BINS))
    out["roc_auc"] = roc_auc(p, y)
    out["pr_auc"] = pr_auc(p, y)
    out["reliability_table"] = reliability_bins(p, y, n_bins=N_BINS)
    return out


def _blocks(
    day: np.ndarray, p: np.ndarray, y: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """``{epoch day: (p, y)}`` as slices of one sorted copy.

    Sorted once and sliced rather than masked per day: the bootstrap draws
    hundreds of samples of every day and a mask per draw would copy the
    whole array every time.
    """
    order = np.argsort(day, kind="stable")
    days = day[order]
    ps, ys = p[order], y[order]
    edges = np.flatnonzero(np.diff(days)) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, days.size]
    return {
        int(days[start]): (ps[start:end], ys[start:end])
        for start, end in zip(starts, ends)
    }


def _bss_of(blocks: Sequence[tuple[np.ndarray, np.ndarray]]) -> float:
    if not blocks:
        return float("nan")
    p = np.concatenate([b[0] for b in blocks])
    y = np.concatenate([b[1] for b in blocks])
    return float(brier_decomposition(p, y, n_bins=N_BINS)["bss"])


def _pr_auc_of(blocks: Sequence[tuple[np.ndarray, np.ndarray]]) -> float:
    if not blocks:
        return float("nan")
    p = np.concatenate([b[0] for b in blocks])
    y = np.concatenate([b[1] for b in blocks])
    return float(pr_auc(p, y))


def _paired_difference(
    day: np.ndarray,
    post: np.ndarray,
    base: np.ndarray,
    y: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    ci: float,
) -> dict[str, Any] | None:
    """Paired day-block CIs on the BSS and PR-AUC differences.

    Paired because the two forecasts are scored on the SAME rows, so the
    day-to-day variance that dominates either score cancels in the
    difference; the interval on the difference is what the shipping rule
    is written in terms of.
    """
    if n_resamples <= 0 or day.size == 0:
        return None
    post_blocks = _blocks(day, post, y)
    base_blocks = _blocks(day, base, y)
    shared = sorted(set(post_blocks) & set(base_blocks))
    if not shared:
        return None
    a = [post_blocks[d] for d in shared]
    b = [base_blocks[d] for d in shared]
    bss = paired_block_bootstrap(
        a, b, _bss_of, n_resamples=n_resamples, seed=seed, ci=ci,
    )
    area = paired_block_bootstrap(
        a, b, _pr_auc_of, n_resamples=n_resamples, seed=seed + 1, ci=ci,
    )
    return {
        "days": len(shared),
        "bss": [float(v) for v in bss],
        "bss_excludes_zero": _excludes_zero(bss),
        "pr_auc": [float(v) for v in area],
        "pr_auc_excludes_zero": _excludes_zero(area),
    }


def _excludes_zero(triple: tuple[float, float, float]) -> bool | None:
    _point, lo, hi = triple
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return lo > 0.0 or hi < 0.0


def out_of_fold_predictions(
    rows: Mapping[str, Any],
    truth: Mapping[int, Any],
    leads: Sequence[int],
    *,
    month: np.ndarray,
    settings: FitSettings,
    design_leads: Sequence[int] | None = None,
    stations: Any | None = None,
    folds: "FoldPlan | None" = None,
    train_mask: Callable[[np.ndarray, int], np.ndarray] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Held-out predictions for one configuration, one fold per (year, month).

    The whole two-stage fit — the model AND the isotonic map on top of it
    — is redone on the training rows of each fold and applied to the
    held-out month, so nothing is ever scored on data it saw. Factored out
    of :func:`leave_one_month_out` because the baseline of record is now
    itself a fitted model (``--baseline refit-v1``), and "the same folds"
    has to mean the same code and not two copies of it.

    ``folds`` replaces that plan with any other :class:`FoldPlan` — the
    random-point protocol hands in a (month, station group) plan, where a
    row trains a fold only when it differs from it on BOTH axes, so no
    prediction ever comes from a model that saw that month or that place.
    ``None`` is ``FoldPlan.by_month(month)``, which is the historical
    behaviour to the row.

    ``train_mask(month, key)`` narrows the training set beyond "not this
    fold" — what :func:`learning_curve` uses to train one fold on a
    subset of its days. It is handed the plan's FIRST axis and the fold's
    value on it, which is the month in both plans that exist.

    Returns ``{"out_of_fold": {lead: array}, "folds": [...]}``; a row a
    fold could not be fitted for stays NaN.

    One approximation, stated rather than hidden: the design SPEC — the
    spline knots — is resolved once over all rows, not per fold. Knots are
    quantiles of a predictor and carry no outcome, so this cannot leak
    skill; everything that touches ``y``, including the standardiser's
    means, is refitted inside each fold.
    """
    chosen = settings.validate()
    wanted = tuple(sorted({int(lead) for lead in leads}))
    in_design = tuple(sorted({int(x) for x in (design_leads or wanted)}))
    n = _rows_in(rows)
    month = np.asarray(month, dtype=np.int64).reshape(-1)
    if month.size != n:
        raise ValueError("month must have one entry per row")
    if stations is None:
        stations = rows.get("station_id")
    station_ids = None if stations is None else np.asarray(stations)
    season_all = rows.get("season")
    season_ids = None if season_all is None else np.asarray(season_all)

    spec, names, design = _design_context(rows, in_design, chosen)
    outcomes = {lead: _usable_outcome(truth, lead, n) for lead in wanted}
    plan = FoldPlan.by_month(month) if folds is None else folds
    if plan.rows != n:
        raise ValueError("the fold plan must have one entry per row")
    out_of_fold = {lead: np.full(n, np.nan, dtype=np.float64) for lead in wanted}
    fold_reports: list[dict[str, Any]] = []

    for fold in plan.folds():
        test = plan.test_mask(fold)
        train = plan.train_mask(fold)
        if train_mask is not None:
            train = train & np.asarray(
                train_mask(plan.axes[0], fold[0]), dtype=bool,
            )
        entry: dict[str, Any] = {
            "fold": plan.label(fold),
            "n_test": int(test.sum()),
            "n_train": int(train.sum()),
            "leads": {},
        }
        if not train.any():
            entry["skipped"] = "no training rows outside the fold"
            fold_reports.append(entry)
            continue
        train_design = design[train]
        standardiser = Standardiser.fit(train_design)
        if chosen.is_trees:
            x_train_all, x_test = train_design, design[test]
        else:
            x_train_all = standardiser.transform(train_design)
            x_test = standardiser.transform(design[test])
            del train_design
        shared_trees, shared_logistic = _fit_shared(
            x_train_all,
            {lead: (y[train], u[train]) for lead, (y, u) in outcomes.items()},
            wanted, int(train.sum()), names, chosen, spec,
            stations=None if station_ids is None else station_ids[train],
        )
        held: dict[int, np.ndarray] = {}
        for lead in wanted:
            y, usable = outcomes[lead]
            fit_mask = usable[train]
            if not fit_mask.any():
                entry["leads"][str(lead)] = {"n_train": 0, "skipped": True}
                continue
            x_fit = x_train_all[fit_mask]
            y_fit = y[train][fit_mask]
            x_score = x_test
            if chosen.shared_lead:
                x_fit = x_fit.copy()
                fill_shared_columns(x_fit, lead, names, spec)
                x_score = x_test.copy()
                fill_shared_columns(x_score, lead, names, spec)
            model = fit_lead(
                x_fit, y_fit, lead,
                settings=chosen, standardiser=standardiser,
                stations=(
                    None if station_ids is None
                    else station_ids[train][fit_mask]
                ),
                season=(
                    None if season_ids is None else season_ids[train][fit_mask]
                ),
                feature_names=names,
                shared_trees=shared_trees,
                shared_logistic=shared_logistic,
            )
            held[lead] = model.predict(
                x_score,
                None if station_ids is None else station_ids[test],
                None if season_ids is None else season_ids[test],
            )
            entry["leads"][str(lead)] = {
                "n_train": int(model.n),
                "base_rate": float(model.base_rate),
                "converged": bool(model.converged),
                "iterations": int(model.iterations),
            }
        # The same running max the serving path applies, applied here:
        # the out-of-fold numbers this returns are written back as
        # ``p_post_<lead>`` and scored, and scoring a number the service
        # would never serve is a measurement of nothing.
        for lead, values in enforce_lead_monotonic(held).items():
            out_of_fold[lead][test] = values
        fold_reports.append(entry)
        if log:
            log(
                f"fold {entry['fold']}: {entry['n_train']} train / "
                f"{entry['n_test']} test row(s)"
            )
    del design
    return {"out_of_fold": out_of_fold, "folds": fold_reports, "names": names}


def _design_context(
    rows: Mapping[str, Any], in_design: Sequence[int], settings: FitSettings,
) -> tuple[DesignSpec, tuple[str, ...], np.ndarray]:
    """``(spec, column names, raw design)`` with any ablated family removed."""
    spec = design_spec(
        rows, in_design,
        version=settings.design, shared_lead=settings.shared_lead,
    )
    names = design_columns(in_design, spec)
    design = build_design(rows, in_design, spec)
    keep = _column_mask(names, settings.drop_families)
    if not keep.all():
        design = design[:, keep]
        names = tuple(name for name, ok in zip(names, keep) if ok)
    return spec, tuple(names), design


def leave_one_month_out(
    rows: Mapping[str, Any],
    truth: Mapping[int, Any],
    leads: Sequence[int],
    *,
    month: np.ndarray,
    day: np.ndarray,
    baseline: Mapping[int, Any],
    folds: "FoldPlan | None" = None,
    l2: float = 1.0,
    design_leads: Sequence[int] | None = None,
    strata: Mapping[str, np.ndarray] | None = None,
    n_resamples: int = 500,
    seed: int = 0,
    ci: float = 0.95,
    isotonic_bins: int = DEFAULT_ISOTONIC_BINS,
    log: Callable[[str], None] | None = None,
    settings: FitSettings | None = None,
    baseline_settings: FitSettings | None = None,
    baseline_label: str = "curve",
    stations: Any | None = None,
) -> dict[str, Any]:
    """Out-of-fold scores for the post-processor and its baseline.

    One fold per ``(year, month)``: the whole two-stage fit is redone on
    the other months and applied to the held-out one, so nothing is ever
    scored on data it saw. The baseline is scored on **exactly the same
    rows**, which is what makes the difference attributable to the model
    rather than to a different sample.

    Parameters
    ----------
    month:
        ``year * 12 + month - 1`` per row (:func:`year_months_from_epoch`).
    day:
        Epoch day per row — the bootstrap's block.
    baseline:
        ``{lead: p_rain_<lead> array}``, the curve-calibrated probability.
        Ignored when ``baseline_settings`` is given.
    baseline_settings:
        Fit a MODEL as the baseline instead, on the same folds — the v1
        design and the shipped logistic (``--baseline refit-v1``) is the
        baseline of record for post-processing v2, because "beats the
        curve" was already settled and the open question is whether the
        new design beats the old one on the same rows.
    strata:
        ``{name: boolean mask}`` beside the pooled stratum. Defaults to
        the three seasons plus the onset-relevant ``dry`` subset, taken
        from ``rows``.

    Returns a dict with ``folds``, ``leads`` (per lead, per stratum:
    ``baseline`` / ``postprocess`` scores and the paired ``difference``)
    and ``out_of_fold`` — ``{lead: array}`` of the held-out predictions,
    NaN where a fold could not be fitted. ``out_of_fold`` is the thing to
    write back as ``p_post_<lead>``; nothing else in here is safe to.
    """
    chosen = (
        FitSettings(l2=float(l2), isotonic_bins=int(isotonic_bins), seed=int(seed))
        if settings is None else settings
    ).validate()
    wanted = tuple(sorted({int(lead) for lead in leads}))
    in_design = tuple(sorted({int(x) for x in (design_leads or wanted)}))
    n = _rows_in(rows)
    month = np.asarray(month, dtype=np.int64).reshape(-1)
    day = np.asarray(day, dtype=np.int64).reshape(-1)
    if month.size != n or day.size != n:
        raise ValueError("month/day must have one entry per row")

    held = out_of_fold_predictions(
        rows, truth, wanted, month=month, settings=chosen,
        design_leads=in_design, stations=stations, folds=folds, log=log,
    )
    out_of_fold = held["out_of_fold"]
    fold_reports = held["folds"]
    outcomes = {lead: _usable_outcome(truth, lead, n) for lead in wanted}

    if baseline_settings is not None:
        if log:
            log("refitting the baseline model on the same folds")
        baseline = out_of_fold_predictions(
            rows, truth, wanted, month=month,
            settings=baseline_settings.validate(),
            design_leads=in_design, stations=stations, folds=folds, log=None,
        )["out_of_fold"]

    if strata is None:
        season = np.asarray(rows.get("season", np.full(n, "", dtype="<U8")))
        season = season.astype("<U8").reshape(-1)
        strata = {name: season == name for name in SEASONS}
        # The onset-relevant subset, beside the seasons: the gate for this
        # track is read on ALL rows and on these. A row whose gauge was
        # already wet is one the service says nothing new about, and
        # pooling it in flatters every arm equally but hides which one is
        # better at the thing a subscriber actually notices.
        dry = dry_subset(rows)
        if dry.any():
            strata[DRY] = dry
    all_strata: list[tuple[str, np.ndarray]] = [
        (POOLED, np.ones(n, dtype=bool))
    ] + [(name, np.asarray(mask, dtype=bool)) for name, mask in strata.items()]

    strata_map = dict(all_strata)
    per_lead: dict[str, Any] = {}
    for lead in wanted:
        y, _usable = outcomes[lead]
        base = np.asarray(baseline[lead], dtype=np.float64).reshape(-1)
        post = out_of_fold[lead]
        # :func:`subset_scores` is THE scorer, called once per arm. Each
        # call is told about the other arm's predictions, so both are
        # graded on exactly the rows both can grade — and the learning
        # curve, which calls the same function with the same argument,
        # lands on the same rows rather than on a wider set of its own.
        scored = subset_scores(
            {lead: post}, outcomes, strata_map,
            also_finite={lead: base}, day=day, detail=True,
        )[str(lead)]
        against = subset_scores(
            {lead: base}, outcomes, strata_map,
            also_finite={lead: post}, day=day, detail=True,
        )[str(lead)]
        by_stratum: dict[str, Any] = {}
        for name in strata_map:
            got = scored[name]
            if not got["n"]:
                by_stratum[name] = None
                continue
            keep = got["keep"]
            block = {
                "n": got["n"],
                "days": got["days"],
                "baseline": against[name]["scores"],
                "postprocess": got["scores"],
                "difference": _paired_difference(
                    day[keep], post[keep], base[keep], y[keep],
                    n_resamples=n_resamples, seed=seed, ci=ci,
                ),
            }
            by_stratum[name] = block
            if log:
                log(
                    f"lead {lead} {name}: n={block['n']}, BSS "
                    f"{block['baseline']['bss']:.4f} → "
                    f"{block['postprocess']['bss']:.4f}"
                )
        per_lead[str(lead)] = by_stratum

    return {
        "folds": fold_reports,
        "leads": per_lead,
        "out_of_fold": out_of_fold,
        "baseline_out_of_fold": (
            dict(baseline) if baseline_settings is not None else None
        ),
        "settings": {
            "l2": float(chosen.l2),
            "design_leads": list(in_design),
            "resamples": int(n_resamples),
            "ci": float(ci),
            "seed": int(seed),
            "n_bins": N_BINS,
            "isotonic_bins": int(chosen.isotonic_bins),
            "folds": (
                FoldPlan.by_month(month) if folds is None else folds
            ).to_json(),
            "baseline": baseline_label,
            "fit": chosen.to_json(),
            "baseline_fit": (
                None if baseline_settings is None
                else baseline_settings.to_json()
            ),
        },
    }


def _month_label(key: int) -> str:
    year, month = divmod(int(key), 12)
    return f"{year:04d}-{month + 1:02d}"


def month_labels(month: Iterable[int]) -> list[str]:
    """``year * 12 + month - 1`` keys → ``"YYYY-MM"`` labels."""
    return [_month_label(int(key)) for key in month]


# ---------------------------------------------------------------------------
# How much archive does this model need, and which features earn their keep
# ---------------------------------------------------------------------------


def subset_scores(
    predictions: Mapping[int, np.ndarray],
    outcomes: Mapping[int, tuple[np.ndarray, np.ndarray]],
    subsets: Mapping[str, np.ndarray],
    *,
    also_finite: Mapping[int, Any] | None = None,
    day: np.ndarray | None = None,
    detail: bool = False,
) -> dict[str, dict[str, Any]]:
    """THE out-of-fold scorer: ``{lead: {subset: block}}``.

    One function for the main table (:func:`leave_one_month_out`), for
    :func:`learning_curve` and for :func:`ablation`, so that a BSS in one
    of those tables and a BSS in another are the same measurement — the
    same truth, the same usable mask, the same subsets, and the same
    :func:`~dmi_nowcast_core.benchmark.brier_decomposition` with the same
    bins and the same climatological reference.

    A row a lead can grade is one whose outcome is usable and whose
    prediction is finite. ``also_finite`` names the OTHER arm's
    predictions per lead and narrows that further: the main table is a
    *paired* comparison and can only score rows both arms answer for, so
    anything meant to be read beside it has to drop those rows too.
    Passing the baseline there is what makes "the curve at N = every day"
    and "the table" one number instead of two numbers about two samples.

    ``day`` adds the distinct scored days per block. ``detail`` adds the
    full score block and the boolean ``keep`` mask — arrays, so a caller
    that serialises the result asks for neither.
    """
    out: dict[str, dict[str, Any]] = {}
    for lead, values in sorted(predictions.items()):
        y, usable = outcomes[int(lead)]
        p = np.asarray(values, dtype=np.float64).reshape(-1)
        gradable = usable & np.isfinite(p)
        other = None if also_finite is None else also_finite.get(int(lead))
        if other is not None:
            gradable = gradable & np.isfinite(
                np.asarray(other, dtype=np.float64).reshape(-1)
            )
        block: dict[str, Any] = {}
        for name, mask in subsets.items():
            keep = gradable & np.asarray(mask, dtype=bool).reshape(-1)
            entry: dict[str, Any] = {"n": int(keep.sum())}
            if day is not None:
                entry["days"] = int(
                    np.unique(np.asarray(day).reshape(-1)[keep]).size
                )
            scores = _scores(p[keep], y[keep]) if keep.any() else None
            entry["bss"] = (
                float("nan") if scores is None else float(scores["bss"])
            )
            if detail:
                entry["scores"] = scores
                entry["keep"] = keep
            block[name] = entry
        out[str(lead)] = block
    return out


#: How :func:`learning_curve` chooses the N training days of a subset.
#:
#: ``random`` draws them at random, stratified by month. ``date`` takes
#: the first N calendar days, which is what the curve did until
#: 2026-09-17 and which confounds "less data" with "winter only": on the
#: 90-day archive its 30-day point was a December-to-February fit scored
#: on every month, and the curve read non-monotone for that reason.
CURVE_ORDER_RANDOM = "random"
CURVE_ORDER_DATE = "date"
CURVE_ORDERS: tuple[str, ...] = (CURVE_ORDER_RANDOM, CURVE_ORDER_DATE)

#: Seed for the stratified draw, so that a curve is reproducible and two
#: runs of the same configuration can be compared row for row.
DEFAULT_CURVE_SEED = 0


def _day_pools(
    day: np.ndarray, month: np.ndarray, *, seed: int,
) -> dict[int, np.ndarray]:
    """``{month key: that month's distinct days, shuffled once}``.

    Shuffled once per call — not per fold and not per budget — which buys
    two properties the curve needs. A month contributes the same days to
    every fold that is allowed to use it, so the folds differ in which
    month is held out and not in which draw they got; and a larger
    budget's draw CONTAINS a smaller one's, so a step along the curve is
    more data rather than other data.
    """
    rng = np.random.default_rng(int(seed))
    pools: dict[int, np.ndarray] = {}
    for key in sorted({int(m) for m in np.unique(month)}):
        days = np.unique(day[month == key])
        rng.shuffle(days)
        pools[int(key)] = days
    return pools


def _drawn_days(
    pools: Mapping[int, np.ndarray], budget: int,
) -> dict[int, np.ndarray]:
    """``budget`` days over ``pools``, each month's share proportional.

    Largest-remainder allotment: a month with a fifth of the available
    days supplies a fifth of the draw, and the seats left over by the
    rounding go round-robin to the months with the largest unserved
    fraction. So the subset's month mix is the full training set's month
    mix, which is the whole point — the N-day point has to differ from
    the all-days point in volume and not in season.

    A budget at or above what is available takes everything, which is
    what makes the curve's last point the main table's fit.
    """
    sizes = {int(key): int(values.size) for key, values in pools.items()}
    keys = sorted(sizes)
    total = sum(sizes.values())
    if budget >= total:
        quota = dict(sizes)
    else:
        exact = {key: budget * sizes[key] / total for key in keys}
        quota = {key: int(math.floor(exact[key])) for key in keys}
        left = budget - sum(quota.values())
        order = sorted(keys, key=lambda key: (-(exact[key] - quota[key]), key))
        index = 0
        # Terminates: ``sum(quota) + left == budget <= total``, so while
        # ``left > 0`` at least one month still has a day to give.
        while left > 0:
            key = order[index % len(order)]
            if quota[key] < sizes[key]:
                quota[key] += 1
                left -= 1
            index += 1
    return {
        key: np.sort(pools[key][:quota[key]]) for key in keys if quota[key]
    }


def _month_histogram(
    day: np.ndarray, month: np.ndarray, mask: np.ndarray,
) -> dict[str, int]:
    """``{"YYYY-MM": distinct training days}`` for one fold's draw."""
    out: dict[str, int] = {}
    for key in sorted({int(m) for m in np.unique(month[mask])}):
        out[_month_label(key)] = int(
            np.unique(day[mask & (month == key)]).size
        )
    return out


def learning_curve(
    rows: Mapping[str, Any],
    truth: Mapping[int, Any],
    leads: Sequence[int],
    *,
    month: np.ndarray,
    day: np.ndarray,
    settings: FitSettings,
    days: Sequence[int],
    design_leads: Sequence[int] | None = None,
    subsets: Mapping[str, np.ndarray] | None = None,
    stations: Any | None = None,
    also_finite: Mapping[int, Any] | None = None,
    order: str = CURVE_ORDER_RANDOM,
    seed: int = DEFAULT_CURVE_SEED,
    log: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Out-of-fold skill against how many days of archive the fit was given.

    For each ``N`` in ``days``: keep the same leave-one-month-out folds,
    but train each fold on only ``N`` of its training DAYS and score the
    held-out month as usual. One row per N, pooled over the folds.

    ``order="random"`` (the default) draws those N days **stratified by
    month**: every training month contributes a share of the draw
    proportional to the days it has (:func:`_drawn_days`), from one
    seeded shuffle reused across folds and budgets
    (:func:`_day_pools`). The N-day point then differs from the all-days
    point in volume and not in season, which is the question a learning
    curve is asked.

    ``order="date"`` keeps the older behaviour — the first N calendar
    days of each fold's training rows. That is a seasonal experiment, not
    a data-volume one: on a December-to-February-plus archive its 30-day
    point trains on winter alone and is then scored on every month, and
    the curve comes back non-monotone for that reason rather than because
    more archive stopped helping. It is kept because the experiment is
    worth running on purpose; the report says which order produced it.

    ``also_finite`` is the baseline the main table scored against — pass
    it, and the curve grades exactly the table's rows, so the last point
    of the curve IS the table's out-of-fold BSS. See
    :func:`subset_scores`, which does the scoring for both.

    Each row carries the budget, the draw's ``order`` and ``seed``, the
    distinct training days and rows actually used, the drawn days' month
    histogram (pooled over the folds, so a day drawn for three folds is
    counted three times), a per-fold breakdown, and the BSS per lead and
    subset.
    """
    if order not in CURVE_ORDERS:
        raise ValueError(
            f"unknown learning-curve order {order!r}; expected one of "
            + ", ".join(CURVE_ORDERS)
        )
    n = _rows_in(rows)
    day = np.asarray(day, dtype=np.int64).reshape(-1)
    month = np.asarray(month, dtype=np.int64).reshape(-1)
    outcomes = {int(lead): _usable_outcome(truth, int(lead), n) for lead in leads}
    if subsets is None:
        dry = dry_subset(rows)
        subsets = {POOLED: np.ones(n, dtype=bool)}
        if dry.any():
            subsets[DRY] = dry

    folds = sorted({int(m) for m in np.unique(month)})
    pools = _day_pools(day, month, seed=seed)

    out: list[dict[str, Any]] = []
    for budget in sorted({int(v) for v in days}):
        masks: dict[int, np.ndarray] = {}
        drawn: dict[str, dict[str, Any]] = {}
        for key in folds:
            train = month != key
            if not train.any():
                masks[key] = train
                drawn[_month_label(key)] = {"train_days": 0, "months": {}}
                continue
            if order == CURVE_ORDER_DATE:
                start = int(day[train].min())
                mask = train & (day <= start + budget - 1)
            else:
                chosen = _drawn_days(
                    {k: v for k, v in pools.items() if k != key}, budget,
                )
                mask = train & np.isin(
                    day,
                    np.concatenate(list(chosen.values())) if chosen
                    else np.empty(0, dtype=np.int64),
                )
            masks[key] = mask
            histogram = _month_histogram(day, month, mask)
            drawn[_month_label(key)] = {
                "train_days": int(sum(histogram.values())),
                "months": histogram,
            }

        def window(
            months: np.ndarray, key: int, masks: dict = masks,
        ) -> np.ndarray:
            return masks[int(key)]

        if log:
            log(
                f"learning curve: {budget} training day(s), {order} draw"
                + (f" (seed {int(seed)})" if order == CURVE_ORDER_RANDOM else "")
            )
        held = out_of_fold_predictions(
            rows, truth, leads, month=month, settings=settings,
            design_leads=design_leads, stations=stations,
            train_mask=window, log=None,
        )
        by_fold = [
            {
                "fold": entry["fold"],
                "train_days": drawn.get(entry["fold"], {}).get("train_days", 0),
                "train_rows": int(entry.get("n_train", 0)),
                "months": drawn.get(entry["fold"], {}).get("months", {}),
            }
            for entry in held["folds"]
        ]
        months_used: dict[str, int] = {}
        for entry in by_fold:
            for label, count in entry["months"].items():
                months_used[label] = months_used.get(label, 0) + count
        out.append({
            "days": budget,
            "order": order,
            "seed": int(seed),
            # What the budget actually bought: a fold with fewer days
            # available than the budget asks for gives what it has.
            "train_days": max((entry["train_days"] for entry in by_fold), default=0),
            "train_rows": sum(entry["train_rows"] for entry in by_fold),
            "months": dict(sorted(months_used.items())),
            "folds": by_fold,
            "leads": subset_scores(
                held["out_of_fold"], outcomes, subsets,
                also_finite=also_finite,
            ),
        })
    return out


def ablation(
    rows: Mapping[str, Any],
    truth: Mapping[int, Any],
    leads: Sequence[int],
    *,
    month: np.ndarray,
    settings: FitSettings,
    families: Sequence[str] | None = None,
    design_leads: Sequence[int] | None = None,
    subsets: Mapping[str, np.ndarray] | None = None,
    stations: Any | None = None,
    folds: "FoldPlan | None" = None,
    also_finite: Mapping[int, Any] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """ΔBSS from dropping each feature family, out of fold.

    The full configuration is scored first, then once per family with that
    family's columns — and every interaction either parent appears in —
    removed from the design entirely. ``delta`` is *dropped minus full*,
    so a negative number means the family was carrying something.

    Dropping the columns rather than zeroing them is the honest version:
    a zeroed column is still standardised, still penalised and still
    occupies a coefficient, and a fit can route round it.

    ``also_finite`` is the main table's baseline, as in
    :func:`learning_curve`: pass it and ``full`` is the table's
    out-of-fold BSS on the table's rows, so the deltas hang off a number
    a reader can find above them.
    """
    n = _rows_in(rows)
    outcomes = {int(lead): _usable_outcome(truth, int(lead), n) for lead in leads}
    if subsets is None:
        dry = dry_subset(rows)
        subsets = {POOLED: np.ones(n, dtype=bool)}
        if dry.any():
            subsets[DRY] = dry

    full = out_of_fold_predictions(
        rows, truth, leads, month=month, settings=settings,
        design_leads=design_leads, stations=stations, folds=folds, log=None,
    )
    present = {feature_family(name) for name in full["names"]}
    wanted = [
        name for name in (families or FAMILY_NAMES)
        if name in present or name == "station"
    ]
    reference = subset_scores(
        full["out_of_fold"], outcomes, subsets, also_finite=also_finite,
    )

    dropped: list[dict[str, Any]] = []
    for family in wanted:
        if log:
            log(f"ablation: dropping the {family} family")
        without = dataclasses_replace(
            settings, drop_families=tuple(settings.drop_families) + (family,),
        )
        # The station family covers the learned offsets too: dropping the
        # static station columns while keeping a per-station intercept
        # would answer a question nobody asked.
        if family == "station":
            without = dataclasses_replace(without, station_offsets=False)
        held = out_of_fold_predictions(
            rows, truth, leads, month=month, settings=without,
            design_leads=design_leads, stations=stations, folds=folds,
            log=None,
        )
        scores = subset_scores(
            held["out_of_fold"], outcomes, subsets, also_finite=also_finite,
        )
        dropped.append({
            "family": family,
            "columns": int(
                len(full["names"]) - len(held["names"])
            ),
            "leads": {
                lead: {
                    subset: {
                        **scores[lead][subset],
                        "delta": (
                            scores[lead][subset]["bss"]
                            - reference[lead][subset]["bss"]
                        ),
                    }
                    for subset in subsets
                }
                for lead in scores
            },
        })
    return {"full": reference, "dropped": dropped}


def dataclasses_replace(settings: FitSettings, **changes: Any) -> FitSettings:
    """``dataclasses.replace`` for :class:`FitSettings`, named for grep."""
    import dataclasses

    return dataclasses.replace(settings, **changes)
