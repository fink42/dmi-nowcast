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
from dataclasses import dataclass
from datetime import datetime, timezone
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

__all__ = [
    "SCHEMA_VERSION",
    "FEATURE_DOC",
    "SCALAR_FEATURE_COLUMNS",
    "DESIGN_SOURCE_COLUMNS",
    "RAW_FRACTION_PREFIX",
    "raw_fraction_column",
    "POST_PREFIX",
    "POST_COLUMN_TEMPLATE",
    "post_column",
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
    "MIN_FLOW_PX_PER_FRAME",
    "ETA_CAP_MIN",
    "UP_DIST_CAP_KM",
    "disc_max",
    "station_features",
    "design_columns",
    "build_design",
    "Standardiser",
    "fit_logistic",
    "DEFAULT_ISOTONIC_BINS",
    "fit_isotonic_binned",
    "LeadModel",
    "PostprocessModel",
    "fit_postprocess",
    "leave_one_month_out",
    "POOLED",
    "N_BINS",
]

#: Version of the ``postprocess.json`` document this module writes and reads.
SCHEMA_VERSION = 1


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


def feature_columns(leads: Sequence[int]) -> tuple[tuple[str, str], ...]:
    """``((name, definition), ...)`` for every feature column, in write order."""
    raw = tuple(
        (
            raw_fraction_column(lead),
            f"UNcalibrated ensemble fraction at {int(lead)} min — the "
            "share of members whose cumulative max crosses the detection "
            "threshold by then, before the served isotonic curve. NaN off "
            "coverage.",
        )
        for lead in sorted({int(lead) for lead in leads})
    )
    return raw + SCALAR_FEATURE_COLUMNS


#: name → definition, flattened, for the leads the products publish today.
FEATURE_DOC: dict[str, str] = dict(SCALAR_FEATURE_COLUMNS)


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
    names = [
        raw_fraction_column(lead) for lead in sorted({int(x) for x in design_leads})
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

    ``grid_features`` is :func:`station_features`' output for the whole
    point list and ``index`` selects this point's entry. ``raw_fractions``
    maps lead → the UNcalibrated ensemble fraction ALREADY read at this
    point's product pixel (``None`` off coverage): only the caller knows
    which pixel the calibrated probability came off, and the two have to
    be the same one.
    """
    row: dict[str, Any] = {}
    for lead in leads:
        row[raw_fraction_column(lead)] = finite_or_none(
            raw_fractions.get(int(lead)),
        )
    for name, _definition in SCALAR_FEATURE_COLUMNS:
        values = grid_features.get(name)
        if values is not None:
            row[name] = finite_or_none(values[index])
    row["season"] = season
    row["hour_utc"] = int(hour_utc)
    row["frame_age_min"] = float(frame_age_min)
    row["station_radar_km"] = float(station_radar_km)
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
    reach = int(math.ceil(float(radius_km) / float(pixel_km)))
    span = np.arange(-reach, reach + 1, dtype=np.float64)
    dy, dx = np.meshgrid(span, span, indexing="ij")
    keep = (dy ** 2 + dx ** 2) * (pixel_km ** 2) <= radius_km ** 2 + 1e-9
    dy, dx = dy[keep], dx[keep]
    values = _gather(
        rain_mm_h,
        np.asarray(rows, dtype=np.float64)[:, None] + dy[None, :],
        np.asarray(cols, dtype=np.float64)[:, None] + dx[None, :],
    )
    return _nanmax(values, axis=1)


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
    return {
        "up_max_20km_mm_h": _nanmax(values[:, near, :], axis=(1, 2)),
        "up_max_40km_mm_h": _nanmax(values, axis=(1, 2)),
        "up_dist_km": np.where(has_echo, s[first], np.nan).astype(np.float32),
        "up_wet_frac_40km": wet_frac.astype(np.float32),
    }


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

    Returns
    -------
    ``{column: float32 array of shape (n_stations,)}`` for every
    grid-derived name in :data:`SCALAR_FEATURE_COLUMNS`. The columns that
    do not come off the grid (``season``, ``hour_utc``, ``frame_age_min``,
    ``station_radar_km``) belong to the caller, which is the only place
    that knows the decision instant and the station's coordinates.
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
    out: dict[str, np.ndarray] = {
        "obs_max_5km_mm_h": disc_max(
            field, row, col, radius_km=OBS_DISC_KM, pixel_km=pixel_km,
        ),
        "bulk_kmh": np.full(n, bulk_speed * to_kmh, dtype=np.float32),
        "bulk_dir_deg": np.full(n, bulk_dir, dtype=np.float32),
        "local_speed_kmh": (
            np.hypot(vy_local, vx_local) * to_kmh
        ).astype(np.float32),
        "stalled_share": np.full(n, float(stalled_share), dtype=np.float32),
    }
    out.update(_upstream(field, row, col, uy, ux, usable, pixel_km=pixel_km))
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


def design_columns(leads: Sequence[int]) -> tuple[str, ...]:
    """Names of the design matrix's columns, in order.

    ``leads`` is the set of leads whose raw ensemble fraction goes into the
    design — every lead the products publish, not just the one being
    predicted. The *shape* of the fraction against lead is what separates
    "a big cell 40 km away" from "a drizzle edge overhead", and a model
    given only its own lead cannot see it.

    The curve-calibrated ``p_rain_<lead>`` is deliberately absent: it is a
    monotone map of ``raw_frac_<lead>`` and adds nothing the design does
    not already have, and keeping it out means the baseline the report
    compares against is never also an input.
    """
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
    features: Mapping[str, Any], leads: Sequence[int],
) -> np.ndarray:
    """Stored feature columns → the ``(n, k)`` float64 design matrix.

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
    design = np.empty((n, len(columns)), dtype=np.float64)
    for index, column in enumerate(columns):
        design[:, index] = column
    return design


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


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> dict[str, Any]:
    """Binary logistic regression with an L2 penalty on the slopes.

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

    Returns ``{"intercept", "coefficients", "n", "base_rate", "loss",
    "converged", "iterations", "message"}``.
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
    base_rate = float(outcome.mean())
    # A degenerate fold (one class only) has no slope to estimate; the
    # honest answer is the constant that class implies, clipped away from
    # an infinite logit.
    if base_rate <= 0.0 or base_rate >= 1.0:
        clipped = min(max(base_rate, 1.0 / (n + 2.0)), 1.0 - 1.0 / (n + 2.0))
        return {
            "intercept": float(math.log(clipped / (1.0 - clipped))),
            "coefficients": [0.0] * k,
            "n": n,
            "base_rate": base_rate,
            "loss": float("nan"),
            "converged": False,
            "iterations": 0,
            "message": "single-class training fold; intercept only",
        }

    penalty = float(l2) / n
    start = np.zeros(k + 1, dtype=np.float64)
    start[0] = math.log(base_rate / (1.0 - base_rate))

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = theta[0]
        weights = theta[1:]
        z = design @ weights + intercept
        # log(1 + exp(z)) computed as the stable max(z,0) + log1p(exp(-|z|)).
        loss = float(
            np.mean(np.maximum(z, 0.0) - z * outcome + np.log1p(np.exp(-np.abs(z))))
        )
        residual = (_sigmoid(z) - outcome) / n
        grad = np.empty_like(theta)
        grad[0] = float(residual.sum())
        grad[1:] = design.T @ residual + penalty * weights
        return loss + 0.5 * penalty * float(weights @ weights), grad

    result = minimize(
        objective, start, jac=True, method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": tol, "gtol": tol},
    )
    theta = np.asarray(result.x, dtype=np.float64)
    return {
        "intercept": float(theta[0]),
        "coefficients": [float(v) for v in theta[1:]],
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


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeadModel:
    """One lead's logistic fit plus the isotonic map on top of it."""

    lead_min: int
    intercept: float
    coefficients: tuple[float, ...]
    isotonic: IsotonicCalibrator
    n: int
    base_rate: float
    converged: bool
    iterations: int
    message: str = ""

    def score(self, standardised: np.ndarray) -> np.ndarray:
        """The logistic probability, before recalibration."""
        weights = np.asarray(self.coefficients, dtype=np.float64)
        return _sigmoid(standardised @ weights + self.intercept)

    def predict(self, standardised: np.ndarray) -> np.ndarray:
        """The calibrated probability — what would be served."""
        raw = self.score(standardised)
        out = np.asarray(self.isotonic.predict(raw), dtype=np.float64)
        return np.clip(out, 0.0, 1.0)

    def to_json(self) -> dict[str, Any]:
        return {
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

    @classmethod
    def from_json(cls, lead: int, raw: Mapping[str, Any]) -> "LeadModel":
        curve = raw["isotonic"]
        return cls(
            lead_min=int(lead),
            intercept=float(raw["intercept"]),
            coefficients=tuple(float(v) for v in raw["coefficients"]),
            isotonic=IsotonicCalibrator(
                raw_breakpoints=tuple(float(v) for v in curve["raw_breakpoints"]),
                calibrated_values=tuple(
                    float(v) for v in curve["calibrated_values"]
                ),
            ),
            n=int(raw.get("n", 0)),
            base_rate=float(raw.get("base_rate", float("nan"))),
            converged=bool(raw.get("converged", False)),
            iterations=int(raw.get("iterations", 0)),
            message=str(raw.get("message", "")),
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

    # -- prediction ---------------------------------------------------------

    def design(self, features: Mapping[str, Any]) -> np.ndarray:
        """Standardised design matrix for stored feature columns."""
        return self.standardiser.transform(
            build_design(features, self.design_leads)
        )

    def predict(
        self, features_by_column: Mapping[str, Any], lead: int | None = None,
    ) -> Any:
        """Post-processed probability from stored feature columns.

        ``lead=None`` returns ``{lead: array}`` for every fitted lead;
        a lead returns that lead's array. One design matrix is built and
        shared, because it does not depend on the lead being predicted.
        """
        standardised = self.design(features_by_column)
        if lead is not None:
            model = self.models.get(int(lead))
            if model is None:
                raise KeyError(f"no model for lead {lead}")
            return model.predict(standardised)
        return {
            int(key): model.predict(standardised)
            for key, model in sorted(self.models.items())
        }

    # -- persistence --------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "fitted_at_utc": self.fitted_at_utc,
            "leads": list(self.leads),
            "l2": self.l2,
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
        models = {
            int(lead): LeadModel.from_json(int(lead), entry)
            for lead, entry in raw["models"].items()
        }
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
            training=dict(raw.get("training") or {}),
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
) -> PostprocessModel:
    """Fit one logistic + isotonic model per lead on ``rows``.

    ``rows`` maps stored column names to equal-length arrays (the replay's
    feature columns, plus ``season``/``hour_utc`` derived from the decision
    instant). ``truth`` maps a lead to the Layer B outcome for the same
    rows — the ``(outcome, usable)`` pair
    ``benchmark_report.GaugeGrid.outcome`` returns, so the definition and
    the exclusions are the report's, not a second opinion.

    Both stages of each lead are fitted on the SAME rows: the logistic
    first, then the isotonic map of its output. That is in-sample for the
    isotonic, which is why nothing here is a result — the number that
    counts comes from :func:`leave_one_month_out`, where the whole
    two-stage fit sits inside the fold.
    """
    wanted = tuple(sorted({int(lead) for lead in leads}))
    if not wanted:
        raise ValueError("no leads to fit")
    missing = [lead for lead in wanted if lead not in truth]
    if missing:
        raise ValueError(f"no truth for lead(s) {missing}")
    in_design = tuple(sorted({int(x) for x in (design_leads or wanted)}))
    n = _rows_in(rows)
    design = build_design(rows, in_design)
    standardiser = Standardiser.fit(design)
    standardised = standardiser.transform(design)
    del design

    models: dict[int, LeadModel] = {}
    for lead in wanted:
        y, usable = _usable_outcome(truth, lead, n)
        x_train = standardised[usable]
        y_train = y[usable]
        if x_train.shape[0] == 0:
            raise ValueError(f"lead {lead} has no gradable rows")
        fit = fit_logistic(x_train, y_train, l2=l2)
        model_only = LeadModel(
            lead_min=lead,
            intercept=fit["intercept"],
            coefficients=tuple(fit["coefficients"]),
            isotonic=IsotonicCalibrator((0.0, 1.0), (0.0, 1.0)),
            n=fit["n"], base_rate=fit["base_rate"],
            converged=fit["converged"], iterations=fit["iterations"],
            message=fit["message"],
        )
        curve = fit_isotonic_binned(
            model_only.score(x_train), y_train, n_bins=isotonic_bins,
        )
        models[lead] = LeadModel(
            lead_min=lead,
            intercept=fit["intercept"],
            coefficients=tuple(fit["coefficients"]),
            isotonic=curve,
            n=fit["n"], base_rate=fit["base_rate"],
            converged=fit["converged"], iterations=fit["iterations"],
            message=fit["message"],
        )
    stamp = (fitted_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return PostprocessModel(
        leads=wanted,
        design_leads=in_design,
        feature_names=design_columns(in_design),
        standardiser=standardiser,
        models=models,
        l2=float(l2),
        fitted_at_utc=stamp.isoformat(timespec="seconds"),
        training=dict(training or {"rows": n}),
    )


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


def leave_one_month_out(
    rows: Mapping[str, Any],
    truth: Mapping[int, Any],
    leads: Sequence[int],
    *,
    month: np.ndarray,
    day: np.ndarray,
    baseline: Mapping[int, Any],
    l2: float = 1.0,
    design_leads: Sequence[int] | None = None,
    strata: Mapping[str, np.ndarray] | None = None,
    n_resamples: int = 500,
    seed: int = 0,
    ci: float = 0.95,
    isotonic_bins: int = DEFAULT_ISOTONIC_BINS,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Out-of-fold scores for the post-processor and the curve baseline.

    One fold per ``(year, month)``: the whole two-stage fit — logistic and
    isotonic — is redone on the other months and applied to the held-out
    one, so nothing is ever scored on data it saw. The baseline is the
    served ``p_rain_<lead>`` on **exactly the same rows**, which is what
    makes the difference attributable to the model rather than to a
    different sample.

    Parameters
    ----------
    month:
        ``year * 12 + month - 1`` per row (:func:`year_months_from_epoch`).
    day:
        Epoch day per row — the bootstrap's block.
    baseline:
        ``{lead: p_rain_<lead> array}``, the curve-calibrated probability.
    strata:
        ``{name: boolean mask}`` beside the pooled stratum. Defaults to the
        three seasons taken from ``rows["season"]``.

    Returns a dict with ``folds``, ``leads`` (per lead, per stratum:
    ``baseline`` / ``postprocess`` scores and the paired ``difference``)
    and ``out_of_fold`` — ``{lead: array}`` of the held-out predictions,
    NaN where a fold could not be fitted. ``out_of_fold`` is the thing to
    write back as ``p_post_<lead>``; nothing else in here is safe to.
    """
    wanted = tuple(sorted({int(lead) for lead in leads}))
    in_design = tuple(sorted({int(x) for x in (design_leads or wanted)}))
    n = _rows_in(rows)
    month = np.asarray(month, dtype=np.int64).reshape(-1)
    day = np.asarray(day, dtype=np.int64).reshape(-1)
    if month.size != n or day.size != n:
        raise ValueError("month/day must have one entry per row")

    design = build_design(rows, in_design)
    outcomes = {lead: _usable_outcome(truth, lead, n) for lead in wanted}
    folds = sorted(set(int(m) for m in np.unique(month)))
    out_of_fold = {lead: np.full(n, np.nan, dtype=np.float64) for lead in wanted}
    fold_reports: list[dict[str, Any]] = []

    for key in folds:
        test = month == key
        train = ~test
        entry: dict[str, Any] = {
            "fold": _month_label(key),
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
        x_train_all = standardiser.transform(train_design)
        del train_design
        x_test = standardiser.transform(design[test])
        for lead in wanted:
            y, usable = outcomes[lead]
            fit_mask = usable[train]
            x_fit = x_train_all[fit_mask]
            y_fit = y[train][fit_mask]
            if x_fit.shape[0] == 0:
                entry["leads"][str(lead)] = {"n_train": 0, "skipped": True}
                continue
            fit = fit_logistic(x_fit, y_fit, l2=l2)
            weights = np.asarray(fit["coefficients"], dtype=np.float64)
            train_score = _sigmoid(x_fit @ weights + fit["intercept"])
            curve = fit_isotonic_binned(
                train_score, y_fit, n_bins=isotonic_bins,
            )
            test_score = _sigmoid(x_test @ weights + fit["intercept"])
            out_of_fold[lead][test] = np.clip(
                np.asarray(curve.predict(test_score), dtype=np.float64), 0.0, 1.0,
            )
            entry["leads"][str(lead)] = {
                "n_train": int(fit["n"]),
                "base_rate": float(fit["base_rate"]),
                "converged": bool(fit["converged"]),
                "iterations": int(fit["iterations"]),
            }
        fold_reports.append(entry)
        if log:
            log(
                f"fold {entry['fold']}: {entry['n_train']} train / "
                f"{entry['n_test']} test row(s)"
            )
    del design

    if strata is None:
        season = np.asarray(rows.get("season", np.full(n, "", dtype="<U8")))
        season = season.astype("<U8").reshape(-1)
        strata = {name: season == name for name in SEASONS}
    all_strata: list[tuple[str, np.ndarray]] = [
        (POOLED, np.ones(n, dtype=bool))
    ] + [(name, np.asarray(mask, dtype=bool)) for name, mask in strata.items()]

    per_lead: dict[str, Any] = {}
    for lead in wanted:
        y, usable = outcomes[lead]
        base = np.asarray(baseline[lead], dtype=np.float64).reshape(-1)
        post = out_of_fold[lead]
        gradable = usable & np.isfinite(base) & np.isfinite(post)
        by_stratum: dict[str, Any] = {}
        for name, mask in all_strata:
            keep = gradable & mask
            if not keep.any():
                by_stratum[name] = None
                continue
            block = {
                "n": int(keep.sum()),
                "days": int(np.unique(day[keep]).size),
                "baseline": _scores(base[keep], y[keep]),
                "postprocess": _scores(post[keep], y[keep]),
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
        "settings": {
            "l2": float(l2),
            "design_leads": list(in_design),
            "resamples": int(n_resamples),
            "ci": float(ci),
            "seed": int(seed),
            "n_bins": N_BINS,
            "isotonic_bins": int(isotonic_bins),
        },
    }


def _month_label(key: int) -> str:
    year, month = divmod(int(key), 12)
    return f"{year:04d}-{month + 1:02d}"


def month_labels(month: Iterable[int]) -> list[str]:
    """``year * 12 + month - 1`` keys → ``"YYYY-MM"`` labels."""
    return [_month_label(int(key)) for key in month]
