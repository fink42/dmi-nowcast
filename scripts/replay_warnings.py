"""Replay the live push decision rule at every rain gauge, over history.

Phase F ("how good are we?"). The site sends a browser notification when
the calibrated probability of rain at a subscriber's point crosses their
threshold at their lead. Nobody has ever checked those notifications
against a measurement. This script puts a **virtual subscriber at every
DMI rain gauge**, replays the exact production decision rule over the
archived radar frames, and scores every warning it would have sent
against what that gauge actually recorded.

Production parity, step by step
-------------------------------
Each frame runs the live cycle
(``sidecar/dmi_nowcast_sidecar/compute.py::_compute_sync``) minus the
parts that only serve pixels to a browser:

1. frames at ``T-20`` / ``T-10`` / ``T`` off the archive (fullRange, the
   10-min product — the runtime is fullRange-only since Phase B);
2. ``dense_flow`` (Farnebäck) on the last two dBZ frames, then
   ``complete_flow`` at the production support threshold, ``nan_to_num``,
   clip to ±30 px/frame;
3. ``run_ensemble`` — the vendored STEPS, 24 members, 6 cascade levels,
   ×4 downsample, ``ceil(horizon_min / dt)`` timesteps from radar time;
4. ``national_products`` at ``--frame-age-min``, then the same isotonic
   national curves the service serves (``--national-curves``) — the
   decision reads the CALIBRATED probability or it is not the live rule;
5. ``observed_rain_grid`` for the observed arm of "already raining", and
   the lead-0 deterministic field (the newest composite advected to
   ``radar_ts + frame_age``) for ``forecast_now_mm_h``;
6. ``sample_point`` per station — the same sampler ``/forecast`` and the
   push service use, so a replayed warning reads the same pixel a real
   one would have;
7. ``push.engine.evaluate`` per station, carrying ``SubState`` from frame
   to frame.

The simulated frame age is the one number that cannot be recovered from
the archive: live, it is ``now - radar_ts`` at compute time (median 14
min). It moves the products' lead bookkeeping AND the wall clock the
decision runs on, so ``generated_at = radar_ts + frame_age_min``.

The anchor policy (Phase H, H-L / L3)
-------------------------------------
``--anchor`` decides WHICH frame a cycle stands on;
:mod:`dmi_nowcast_core.anchor` holds the policy itself and the reasoning.

``fullRange`` (default)
    Today's product, ``:x0``, 240 km range, published
    ``--lag-fullrange-min`` (13.1) minutes after its nominal time.
``freshest``
    The per-pixel freshest covering frame: the ``:x5`` doppler composite,
    harmonised onto fullRange's distribution through the L2 table
    (``--harmonisation``, required) and filled from the newest fullRange
    frame outside its 120 km range. The flow and the STEPS cascade still
    run on a **same-type** triple 10 minutes apart (DECIDE-3, DECIDE-4).

Two consequences worth knowing before reading a run:

1. **The decision cadence does not change.** The measured lags are
   ``:x0 + 13.1`` and ``:x5 + 8.1`` — the same wall instant: DMI
   publishes both products in one event every ten minutes. So
   ``freshest`` does not add decisions; it makes each of the same
   decisions five minutes fresher (anchor age 10 min instead of 15).
   That also keeps ``radar_ts`` unique per decision, which every
   downstream consumer depends on: decision rows are deduplicated on
   ``(radar_ts, station_id)``, so two instants sharing one anchor would
   silently collapse into one row.
2. **The frame age is no longer flat.** ``--frame-age-min`` was one
   simulated latency for every frame; it is now derived per instant from
   the product's publication lag plus the wait for the next 5-minute
   poll, so a default ``fullRange`` run sits at 15 min where the old runs
   sat at 14. Pass ``--frame-age-min 14`` to reproduce a run made before
   this existed — it restores the flat model exactly, for both products
   and with no poll grid. Scoring a new candidate against an OLD baseline
   folds that one-minute step into the difference: re-run the baseline.

Parallelism and the state simplification
----------------------------------------
The per-frame pipeline is state-free; only ``evaluate`` carries state,
and it carries it along one station's sequence of frames. So the work is
parallelised **by day, one process per day**, and — the simplification —
**every day starts armed, with an empty streak**. A subscription
disarmed at 23:50 is armed again at 00:00 the next day, which can only
ever add warnings near a midnight boundary (at most one per station per
day, and only when rain was already firing at midnight). Chaining state
across days would serialise the whole run into one process; at ~22 s per
frame and 144 frames a day that is the difference between hours and
weeks. Frames inside a day are strictly ordered, so the state machine
sees exactly the observation sequence the live service would have seen.

Resume granularity is one day: the progress file records each finished
day (its row count and the per-station end state), and the day's parquet
is written once, atomically, when the day completes. A run interrupted
mid-day redoes that day; a run interrupted between days picks up at the
next one.

Usage
-----
Validate on a couple of hours, one worker, four stations::

    python scripts/replay_warnings.py \\
        --archive-dir /var/lib/dmi-nowcast-corpus/composites \\
        --corpus-dir /var/lib/dmi-nowcast-corpus \\
        --points station_points.json --days 2026-09-05 \\
        --start-utc 06:00 --end-utc 08:00 --workers 1 \\
        --out-dir /tmp/replay

Then the real thing::

    python scripts/replay_warnings.py \\
        --archive-dir /var/lib/dmi-nowcast-corpus/composites \\
        --corpus-dir /var/lib/dmi-nowcast-corpus \\
        --points /var/lib/dmi-nowcast-corpus/stations/station_points.json \\
        --days-file days.txt --workers 6 --frame-age-min 14 \\
        --national-curves /var/lib/dmi-nowcast/national_curves.json \\
        --out-dir /var/lib/dmi-nowcast-corpus/stations/replay \\
        --progress /var/lib/dmi-nowcast-corpus/stations/replay/progress.json

The L3 candidate — same days, same everything, freshest anchor
(``CORPUS=/var/lib/dmi-nowcast-corpus``)::

    python scripts/replay_warnings.py \\
        ... as above ... \\
        --anchor freshest \\
        --harmonisation "$CORPUS"/stations/product_study/doppler_harmonisation.json \\
        --out-dir /var/lib/dmi-nowcast-corpus/stations/replay_freshest

Post-processing features (Phase H, H-P)
--------------------------------------
``--features`` (ON by default) writes, beside each decision, the
predictors the gauge-trained post-processor is fitted on
(``scripts/fit_postprocess.py``). They come off the SAME anchor field, the
SAME completed flow and the SAME national grids the cycle already
computed — no second STEPS run, no second motion estimate — and they are
computed for the whole station list in one vectorised pass rather than a
Python loop per pixel.

The columns are **additive**. They are not part of
``warning_score.decision_schema``, and every existing consumer
(``threshold_sweep.load_decisions``, both benchmark layers, the nightly
fit) conforms a file to that schema before reading it, so a run with
features scores identically to one without. ``--no-features`` reproduces
the pre-2026-09-09 file byte for byte.

Definitions, also written to ``summary.json`` under ``run.features``:

``raw_frac_<lead>``
    The UNcalibrated ensemble fraction at that lead, read at the same
    product pixel the calibrated ``p_rain_<lead>`` was. The calibrated
    column is untouched — it is the baseline the post-processor has to
    beat.
``obs_max_5km_mm_h``
    Max observed rain rate within 5 km of the station on the native grid.
    (The block-p90 observation at the station's own product pixel is
    already the schema's ``observed_mm_h``, and is not duplicated.)
``up_max_20km_mm_h`` / ``up_max_40km_mm_h`` / ``up_dist_km`` /
``up_wet_frac_40km``
    The upwind corridor: 6 km wide, laid along the LOCAL completed flow at
    the station, out to 20 and 40 km. Maximum rain rate in it, distance to
    the nearest pixel at or above 0.5 mm/h (NaN — "no echo upwind" — when
    there is none), and the share of its in-composite pixels that are wet.
``bulk_kmh`` / ``bulk_dir_deg`` / ``local_speed_kmh`` / ``stalled_share``
    The cycle's motion: bulk speed and the compass bearing it heads
    toward, the completed flow's speed at the station, and the share of
    wet pixels whose raw estimate is stalled (the H-F health signal).
``season`` / ``hour_utc`` / ``frame_age_min`` / ``station_radar_km``
    The decision instant's season (the project-wide May-Sep / Dec-Mar /
    shoulder split) and UTC hour, the cycle's own simulated latency, and
    the station's great-circle distance to the nearest DMI radar.

``eta_min`` and ``intensity_mm_h`` are features too, and are already
decision columns; they are read from there rather than written twice.

Outputs under ``--out-dir``:

``decisions/YYYY-MM-DD.parquet``
    One row per (frame, station): the sampled forecast and the action the
    engine took. :data:`dmi_nowcast_core.warning_score.DECISION_COLUMNS`,
    plus the H-P feature columns above when ``--features`` is on.
``events.parquet``
    Every warning the replay sent, with its outcome, matched onset and
    lead error.
``onsets.parquet``
    Every gauge onset, and the warning that claimed it (or "miss").
``summary.json``
    Pooled and per-station scores, plus the run's provenance.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace as dc_replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# The repo layout puts the algorithm library under src/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
_SIDECAR = _REPO_ROOT / "sidecar"
if str(_SIDECAR) not in sys.path:
    sys.path.insert(0, str(_SIDECAR))

from dmi_nowcast_core import anchor as anchor_policy  # noqa: E402
from dmi_nowcast_core.advect import advect_field_series  # noqa: E402
from dmi_nowcast_core.calibrate import load_calibration_curves  # noqa: E402
from dmi_nowcast_core.corpus import (  # noqa: E402
    SCAN_TYPE_DOPPLER,
    SCAN_TYPE_FULL_RANGE,
)
from dmi_nowcast_core.dense_flow import (  # noqa: E402
    DEFAULT_CONFIDENCE_PERCENTILE,
    DEFAULT_CONFIDENCE_WINDOW_PX,
    DEFAULT_TEXTURE_PERCENTILE,
)
from dmi_nowcast_core.geo import CompositeGeo  # noqa: E402
from dmi_nowcast_core.national import (  # noqa: E402
    national_products,
    observed_rain_grid,
)
from dmi_nowcast_core.parse import RadarComposite, parse_composite  # noqa: E402
from dmi_nowcast_core import postprocess  # noqa: E402
from dmi_nowcast_core.probabilistic import run_ensemble  # noqa: E402
from dmi_nowcast_core.product_pairs import nearest_radar_km  # noqa: E402
from dmi_nowcast_core.transform import dbz_to_rain_rate  # noqa: E402
from dmi_nowcast_core.warning_score import (  # noqa: E402
    DEFAULT_DRY_MIN,
    DEFAULT_ONSET_MIN_MM,
    DEFAULT_TOLERANCE_MIN,
    PRECIP_DUR_PARAM,
    PRECIP_PARAM,
    ScoreResult,
    align_decision_table,
    decision_schema,  # noqa: F401 — re-exported for the replay tests
    decision_table,
    coverage_runs,
    gauge_slot_amounts,
    onsets,
    per_lead_columns,
    pooled_summary,
    raining_now_agreement,
    score_warnings,
)

# ---------------------------------------------------------------------------
# Production constants, copied (not imported) so the study does not depend
# on a sidecar Config object. Each cites its source of truth.
# ---------------------------------------------------------------------------
MAX_PX_PER_FRAME = 30.0        # compute.py::_MAX_PX_PER_FRAME
RAIN_THRESHOLD_MM_H = 0.5      # config.py ForecastConfig.rain_threshold_mm_h
DOWNSAMPLE_FACTOR = 4          # config.py StepsConfig.downsample_factor
ENSEMBLE_SIZE = 16             # config.py StepsConfig.ensemble_size — what the VM runs (verified 2026-09-08)
N_CASCADE_LEVELS = 6           # config.py StepsConfig.n_cascade_levels
HORIZON_MIN = 90               # config.py StepsConfig.horizon_min
NATIONAL_LEADS = (10, 20, 30, 45, 60)   # config.py NationalConfig.leads_min
# H-F motion completion (2026-09-08). config.py ForecastConfig.
# flow_completion / flow_confidence_window_px / flow_confidence_percentile
# / flow_texture_percentile. The three numbers come from dense_flow so the
# replay cannot drift from the library's own defaults.
DEFAULT_FLOW_COMPLETION = "confidence"
FRAME_INTERVAL_MIN = 10        # fullRange cadence (Phase B addendum)
FRAME_TOLERANCE_S = 60         # a frame is "on the grid" within a minute
#: The live subscriber row this replay reproduces.
#:
#: ``threshold_pct`` here is only the FALLBACK — the percent a run warns
#: at when it is given no fitted threshold table. The service has not
#: warned at a fixed 40 % since Phase G: it warns at whatever
#: ``push_thresholds.json`` picks for the subscriber's horizon, refitted
#: nightly. Pass ``--thresholds`` (and leave ``--rules threshold_pct``
#: alone) to generate a tree under the rule the service is actually on;
#: the 40 here is what the run falls back to, and what every tree written
#: before ``--thresholds`` existed was generated with.
DEFAULT_RULES: dict[str, float] = {
    "threshold_pct": 40,
    "lead_min": 30,
    "rearm_after_min": 60,
    "persistence_obs": 1,
    "raining_now_eta_min": 1.5,
    "raining_now_mm_h": RAIN_THRESHOLD_MM_H,
}

#: Where a run's ``threshold_pct`` came from, recorded in
#: ``summary.json``'s ``run.rules``. ``table`` is a fitted pick from a
#: ``--thresholds`` document, ``fallback`` that document's own default for
#: a lead it cannot speak for, ``rules`` the ``--rules`` / built-in value
#: because no document was given.
THRESHOLD_SOURCE_TABLE = "table"
THRESHOLD_SOURCE_FALLBACK = "fallback"
THRESHOLD_SOURCE_RULES = "rules"
#: Live median compute latency — see the module docstring.
DEFAULT_FRAME_AGE_MIN = 14.0
#: Pad each scored day by this much so a 23:5x warning can still find its
#: onset, and so the first slots of a day have their dry evidence.
GAUGE_PAD_MIN = 120


# ---------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StationPoint:
    id: str
    lat: float
    lon: float
    region: str | None = None


def load_points(path: Path) -> tuple[StationPoint, ...]:
    """Read the v2 station points file the Phase F builder writes."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict) or "points" not in raw:
        raise ValueError(f"{path}: expected an object with a 'points' array")
    version = int(raw.get("version", 0))
    if version != 2:
        raise ValueError(f"{path}: unsupported points version {version!r} (want 2)")
    points: list[StationPoint] = []
    seen: set[str] = set()
    for entry in raw["points"]:
        pid = str(entry["id"])
        if pid in seen:
            raise ValueError(f"{path}: duplicate station id {pid!r}")
        seen.add(pid)
        points.append(StationPoint(
            id=pid,
            lat=float(entry["lat"]),
            lon=float(entry["lon"]),
            region=entry.get("region"),
        ))
    if not points:
        raise ValueError(f"{path}: no points")
    return tuple(points)


def apply_thresholds(
    rules: dict[str, float], path: Path | str | None,
) -> tuple[dict[str, float], str]:
    """Override ``threshold_pct`` from a served threshold document.

    ``(rules, source)``. Without a path nothing moves and the source is
    ``"rules"`` — the ``--rules`` value, or the built-in fallback. With
    one, the percent is the document's own answer for this run's lead,
    read through :mod:`dmi_nowcast_core.push_thresholds` so a replayed
    tree is generated under the rule the running service is on rather
    than under a constant nobody is subscribed to.

    A missing or unusable document is an ERROR here, not a silent
    fallback: a replay is hours of CPU, and one that quietly warned at
    40 % because a path was mistyped would be discovered a day later.
    """
    if path is None:
        return rules, THRESHOLD_SOURCE_RULES
    from dmi_nowcast_core.push_thresholds import (
        effective_threshold,
        lead_pick,
        load_thresholds,
    )

    doc = load_thresholds(path)
    if doc is None:
        raise ValueError(
            f"{path}: not a usable push-threshold document "
            "(drop --thresholds to replay at the --rules percent)",
        )
    lead = str(int(rules["lead_min"]))
    rules = dict(rules)
    rules["threshold_pct"] = float(effective_threshold(doc, lead))
    source = (
        THRESHOLD_SOURCE_TABLE if lead_pick(doc, lead) is not None
        else THRESHOLD_SOURCE_FALLBACK
    )
    return rules, source


def parse_rules(spec: str | None) -> dict[str, float]:
    """``k=v,k=v`` over :data:`DEFAULT_RULES`; unknown keys are an error.

    ``threshold_pct`` set here is the FALLBACK for a run with no
    ``--thresholds`` document — see :data:`DEFAULT_RULES`.
    """
    rules = dict(DEFAULT_RULES)
    if not spec:
        return rules
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, _, value = chunk.partition("=")
        key = key.strip()
        if key not in rules:
            raise ValueError(
                f"unknown rule {key!r}; known: {', '.join(sorted(rules))}"
            )
        rules[key] = float(value)
    if not 0 < rules["threshold_pct"] < 100:
        raise ValueError("threshold_pct must be in (0, 100)")
    if int(rules["lead_min"]) not in NATIONAL_LEADS:
        raise ValueError(
            f"lead_min must be one of the served leads {NATIONAL_LEADS}"
        )
    if rules["persistence_obs"] < 1:
        raise ValueError("persistence_obs must be >= 1")
    return rules


# ---------------------------------------------------------------------------
# Archive access
# ---------------------------------------------------------------------------


def frame_path(archive_dir: Path, ts: datetime) -> Path:
    return (
        Path(archive_dir)
        / f"{ts.year:04d}"
        / f"{ts.month:02d}"
        / f"dk.com.{ts:%Y%m%d%H%M}.500_max.h5"
    )


def full_range_frames(archive_dir: Path, day: date) -> list[datetime]:
    """Every fullRange (minute :x0) frame of ``day`` present on disk."""
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    out: list[datetime] = []
    for i in range(24 * 60 // FRAME_INTERVAL_MIN):
        ts = start + timedelta(minutes=FRAME_INTERVAL_MIN * i)
        if frame_path(archive_dir, ts).exists():
            out.append(ts)
    return out


def archive_frames(
    archive_dir: Path, start: datetime, end: datetime,
) -> dict[str, dict[datetime, Path]]:
    """``{scan_type: {timestamp: path}}`` for ``[start, end]``, by probing.

    Deliberately a directory probe on the 5-minute grid rather than an
    :class:`dmi_nowcast_core.corpus.ArchiveIndex`: the index scans all
    ~78,000 archived filenames, and this runs once per day worker. A day
    plus its lookback is ~300 ``exists()`` calls, which is nothing beside
    one STEPS run.

    The minute IS the product marker (``corpus.scan_type_from_filename``),
    so only ``:x0`` and ``:x5`` slots are probed and anything else in the
    tree is ignored — an off-grid frame must never be mistaken for either
    product.
    """
    out: dict[str, dict[datetime, Path]] = {
        SCAN_TYPE_FULL_RANGE: {}, SCAN_TYPE_DOPPLER: {},
    }
    step = timedelta(minutes=FRAME_INTERVAL_MIN / 2)
    ts = start.replace(second=0, microsecond=0)
    ts -= timedelta(minutes=ts.minute % 5)
    while ts <= end:
        path = frame_path(archive_dir, ts)
        if path.exists():
            scan = (
                SCAN_TYPE_FULL_RANGE if ts.minute % FRAME_INTERVAL_MIN == 0
                else SCAN_TYPE_DOPPLER
            )
            out[scan][ts] = path
        ts += step
    return out


class CompositeCache:
    """FIFO cache so a day's frames are parsed once, not three times."""

    def __init__(self, archive_dir: Path, maxsize: int = 4) -> None:
        self.archive_dir = Path(archive_dir)
        self.maxsize = maxsize
        self._store: dict[datetime, RadarComposite] = {}
        self._order: list[datetime] = []

    def get(self, ts: datetime) -> RadarComposite:
        hit = self._store.get(ts)
        if hit is not None:
            return hit
        comp = parse_composite(frame_path(self.archive_dir, ts))
        self._store[ts] = comp
        self._order.append(ts)
        while len(self._order) > self.maxsize:
            self._store.pop(self._order.pop(0), None)
        return comp


# ---------------------------------------------------------------------------
# One frame: the live pipeline, reduced to what a point needs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorSettings:
    """Which frame each cycle stands on — Phase H, L3.

    The default is the pre-L3 model, so a ``FrameSettings()`` built in a
    test or another script keeps behaving exactly as it did: the
    ``fullRange`` policy with a flat simulated frame age. The CLI's
    default is the *new* one (``frame_age_override_min=None``, so the age
    comes from the publication lag and the poll grid); ``--frame-age-min``
    puts the flat model back.
    """

    policy: str = anchor_policy.POLICY_FULLRANGE
    lag_fullrange_min: float = anchor_policy.DEFAULT_LAG_FULLRANGE_MIN
    lag_doppler_min: float = anchor_policy.DEFAULT_LAG_DOPPLER_MIN
    poll_interval_min: float = anchor_policy.DEFAULT_POLL_INTERVAL_MIN
    max_age_min: float = anchor_policy.DEFAULT_MAX_ANCHOR_AGE_MIN
    frame_age_override_min: float | None = DEFAULT_FRAME_AGE_MIN
    harmonisation_path: str | None = None
    history_mode: str = anchor_policy.HISTORY_SAME_TYPE

    @property
    def lag(self) -> anchor_policy.ProductLag:
        return anchor_policy.ProductLag(
            fullrange_min=self.lag_fullrange_min,
            doppler_min=self.lag_doppler_min,
            flat=self.frame_age_override_min,
        )

    @property
    def poll_min(self) -> float:
        """0 with a flat override: instants are the publication instants.

        The flat model has no poll grid — a frame is seen exactly
        ``frame_age_override_min`` after its nominal time — and snapping
        it to a 5-minute grid would move every old run by a minute.
        """
        if self.frame_age_override_min is not None:
            return 0.0
        return self.poll_interval_min

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "lag_min": self.lag.as_dict(),
            "poll_interval_min": self.poll_min,
            "frame_age_override_min": self.frame_age_override_min,
            "max_anchor_age_min": self.max_age_min,
            "history_mode": self.history_mode,
            "harmonisation": anchor_policy.harmonisation_stamp(
                self.harmonisation_path,
            ),
        }


@dataclass(frozen=True)
class FrameSettings:
    """STEPS / product settings for a replay. Defaults mirror production."""

    frame_age_min: float = DEFAULT_FRAME_AGE_MIN
    ensemble_size: int = ENSEMBLE_SIZE
    n_cascade_levels: int = N_CASCADE_LEVELS
    downsample_factor: int = DOWNSAMPLE_FACTOR
    horizon_min: int = HORIZON_MIN
    leads_min: tuple[int, ...] = NATIONAL_LEADS
    threshold_mm_h: float = RAIN_THRESHOLD_MM_H
    national_curves_path: str | None = None
    # H-F motion-completion policy and its three numbers (2026-09-08).
    # Mirror ``ForecastConfig.flow_completion`` and friends; recorded in
    # the run summary so a replay's decisions can be traced to the flow
    # they were made on.
    flow_completion: str = DEFAULT_FLOW_COMPLETION
    flow_confidence_window_px: int = DEFAULT_CONFIDENCE_WINDOW_PX
    flow_confidence_percentile: float = DEFAULT_CONFIDENCE_PERCENTILE
    flow_texture_percentile: float = DEFAULT_TEXTURE_PERCENTILE
    # L3 (2026-09-09): which frame each cycle stands on. See AnchorSettings.
    anchor: AnchorSettings = field(default_factory=AnchorSettings)
    # H-P (2026-09-09): write the per-station post-processing features
    # beside the decision. Additive columns; every consumer of the decision
    # schema ignores them (``align_decision_table`` conforms a file to the
    # schema and drops what is not in it), so a run WITH features scores
    # identically to one without.
    features: bool = True


def production_motion(
    prev: RadarComposite,
    now: RadarComposite,
    rain_now: np.ndarray,
    settings: "FrameSettings | None" = None,
    *,
    dt_min: float = FRAME_INTERVAL_MIN,
) -> Any:
    """The cycle's :class:`~dmi_nowcast_core.dense_flow.MotionEstimate`.

    One call into ``dense_flow.estimate_motion``, the same entry point the
    runtime and the corpus builder use, so the replay cannot drift from
    what production serves.

    The whole estimate is returned, not just ``(vy, vx)``: the H-P feature
    block needs the bulk motion and the stall share beside the completed
    field. The caller drops it before STEPS runs — it carries two more
    native-grid float32 grids (~14 MB each on the DMI composite) that
    nothing downstream of the features reads.

    Falls back to a uniform phase-correlation shift when OpenCV is absent,
    exactly as ``build_calibration_corpus._process_event`` does, so the
    replay still runs (with a cruder motion field) on a box without cv2.
    """
    from dmi_nowcast_core.dense_flow import (
        DenseFlowUnavailable,
        dense_flow,
        estimate_motion,
    )

    cfg = settings if settings is not None else FrameSettings()
    raw_flow: tuple[np.ndarray, np.ndarray]
    try:
        raw_flow = dense_flow(prev.reflectivity_dbz, now.reflectivity_dbz)
    except DenseFlowUnavailable:
        from dmi_nowcast_core.motion import phase_correlation_shift

        rain_prev = dbz_to_rain_rate(
            prev.reflectivity_dbz, zr_a=prev.zr_a, zr_b=prev.zr_b,
        )
        dy, dx = phase_correlation_shift(rain_prev, rain_now)
        raw_flow = (
            np.full(rain_now.shape, dy, dtype=np.float32),
            np.full(rain_now.shape, dx, dtype=np.float32),
        )
    motion = estimate_motion(
        prev.reflectivity_dbz, now.reflectivity_dbz, rain_now,
        pixel_km=float(now.xscale_m) / 1000.0,
        dt_min=dt_min,
        support_threshold_mm_h=cfg.threshold_mm_h,
        completion=cfg.flow_completion,
        confidence_window_px=cfg.flow_confidence_window_px,
        confidence_percentile=cfg.flow_confidence_percentile,
        texture_percentile=cfg.flow_texture_percentile,
        max_px_per_frame=MAX_PX_PER_FRAME,
        flow=raw_flow,
    )
    return motion


#: One isotonic curve set per process, keyed on the file path — the curves
#: are read once per worker rather than once per frame.
_CURVE_CACHE: dict[str, dict[int, Any]] = {}


def _curves(path: str | None) -> dict[int, Any]:
    if not path:
        return {}
    cached = _CURVE_CACHE.get(path)
    if cached is None:
        cached = load_calibration_curves(Path(path))
        _CURVE_CACHE[path] = cached
    return cached


def legacy_history(
    archive_dir: Path, radar_ts: datetime, settings: FrameSettings,
) -> anchor_policy.HistorySelection:
    """The pre-L3 input triple: fullRange at ``T-20`` / ``T-10`` / ``T``.

    Built from paths without touching the disk, so a missing predecessor
    still surfaces as the ``OSError`` from the parser rather than as a
    silently shorter history — the day worker records it as a frame error
    either way, and the difference matters when debugging one frame.
    """
    step = timedelta(minutes=FRAME_INTERVAL_MIN)
    stamps = tuple(radar_ts - i * step for i in (2, 1, 0))
    age = settings.frame_age_min
    selection = anchor_policy.AnchorSelection(
        now=radar_ts + timedelta(minutes=age),
        policy=anchor_policy.POLICY_FULLRANGE,
        scan_type=SCAN_TYPE_FULL_RANGE,
        timestamp=radar_ts,
        path=frame_path(archive_dir, radar_ts),
        frame_age_min=age,
        fullrange_ts=radar_ts,
        fullrange_path=frame_path(archive_dir, radar_ts),
    )
    return anchor_policy.HistorySelection(
        scan_type=SCAN_TYPE_FULL_RANGE,
        timestamps=stamps,
        paths=tuple(frame_path(archive_dir, ts) for ts in stamps),
        fill_timestamps=(None, None, None),
        fill_paths=(None, None, None),
        selection=selection,
    )


def anchor_inputs(
    cache: CompositeCache,
    history: anchor_policy.HistorySelection,
    settings: FrameSettings,
) -> tuple[list[RadarComposite], np.ndarray, np.ndarray]:
    """``(history composites, anchor dBZ, flow support dBZ)`` for one cycle.

    Under the ``fullRange`` policy all three are the archive's own frames
    and nothing is computed — the anchor IS the newest history frame.

    Under a doppler anchor the history triple is harmonised frame by frame
    (never raw: plan §0.3) and the anchor field additionally takes its
    outside-coverage pixels from the newest fullRange frame. The flow's
    echo support then has to be the *same-type* field, not the filled one:
    the flow was estimated from a doppler pair, so beyond 120 km it has no
    observation behind it and the completion step must be allowed to fill
    it from the bulk motion instead of being told there is echo there.

    ``history_mode="filled"`` fills every history frame the same way, so
    each pixel's series is still one product at 10-minute spacing while
    the cascade — and therefore ``p_rain`` — keeps fullRange's coverage.
    """
    composites = [cache.get(ts) for ts in history.timestamps]
    if history.selection.scan_type != SCAN_TYPE_DOPPLER:
        newest = composites[-1].reflectivity_dbz
        return composites, newest, newest

    table = anchor_policy.load_harmonisation(settings.anchor.harmonisation_path)
    distance = anchor_policy.distance_km_grid(composites[-1])
    fields = [
        anchor_policy.anchor_field(
            comp.reflectivity_dbz,
            scan_type=SCAN_TYPE_DOPPLER,
            anchor_ts=comp.timestamp_utc,
            harmonisation=table,
            distance_km=distance,
        )
        for comp in composites
    ]

    def _filled(index: int) -> Any:
        fill_ts = history.fill_timestamps[index]
        if fill_ts is None:
            return fields[index]
        return anchor_policy.fill_uncovered(
            fields[index], cache.get(fill_ts).reflectivity_dbz, fill_ts,
        )

    anchor = _filled(len(fields) - 1)
    if settings.anchor.history_mode == anchor_policy.HISTORY_FILLED:
        used = [_filled(i) for i in range(len(fields) - 1)] + [anchor]
    else:
        used = fields
    out = [
        dc_replace(comp, reflectivity_dbz=field.dbz)
        for comp, field in zip(composites, used)
    ]
    # Under "filled" the unfilled harmonised grids are now unreferenced;
    # they are ~14 MB each on the national grid and this runs two to a
    # 5 GB cgroup cap.
    del fields
    return out, anchor.dbz, out[-1].reflectivity_dbz


def sample_frame(
    cache: CompositeCache,
    radar_ts: datetime,
    points: Sequence[StationPoint],
    settings: FrameSettings,
    *,
    history: anchor_policy.HistorySelection | None = None,
) -> list[dict[str, Any]]:
    """Run one cycle end-to-end; return one sample dict per station.

    ``history`` is the planned cycle (:func:`plan_day`). Without one the
    pre-L3 fullRange triple at ``radar_ts`` is used, which is what a
    direct caller and the older tests expect.

    Raises on a missing input frame or a STEPS failure — the day worker
    catches it and records the frame as an error rather than pretending
    the stations were dry.
    """
    from dmi_nowcast_sidecar.national_sample import sample_point

    if history is None:
        history = legacy_history(cache.archive_dir, radar_ts, settings)
    composites, anchor_dbz, support_dbz = anchor_inputs(cache, history, settings)
    spacing = [
        (b.timestamp_utc - a.timestamp_utc).total_seconds() / 60.0
        for a, b in zip(composites, composites[1:])
    ]
    if any(abs(s - FRAME_INTERVAL_MIN) > 1.0 for s in spacing):
        raise RuntimeError(f"input frames are not on the {FRAME_INTERVAL_MIN}-min grid")
    now = composites[-1]
    dt_min = spacing[-1]
    frame_age_min = history.selection.frame_age_min
    geo = CompositeGeo(now)
    rain_now = dbz_to_rain_rate(anchor_dbz, zr_a=now.zr_a, zr_b=now.zr_b)
    rain_support = (
        rain_now if support_dbz is anchor_dbz
        else dbz_to_rain_rate(support_dbz, zr_a=now.zr_a, zr_b=now.zr_b)
    )
    motion = production_motion(
        composites[-2], now, rain_support, settings, dt_min=dt_min,
    )
    vy, vx = motion.vy, motion.vx
    # The H-P features come off the SAME anchor field and the SAME
    # completed flow the ensemble is about to run on — no second estimate,
    # no second STEPS. Computed here, before the cascade, so the
    # MotionEstimate (which carries two more native-grid grids) can be
    # dropped before the memory-hungry part of the cycle.
    grid_features = None
    if settings.features:
        native = [geo.lonlat_to_grid(point.lon, point.lat) for point in points]
        grid_features = postprocess.station_features(
            rain_now, vy, vx,
            np.array([idx.row for idx in native], dtype=np.float64),
            np.array([idx.col for idx in native], dtype=np.float64),
            pixel_km=float(now.xscale_m) / 1000.0,
            dt_min=dt_min,
            bulk_vy=motion.bulk_vy,
            bulk_vx=motion.bulk_vx,
            stalled_share=motion.stalled_share,
        )
    del motion

    n_timesteps = max(1, math.ceil(settings.horizon_min / dt_min - 1e-9))
    forecast = run_ensemble(
        [c.reflectivity_dbz for c in composites],
        vy, vx,
        zr_a=now.zr_a, zr_b=now.zr_b,
        n_ens_members=settings.ensemble_size,
        n_timesteps=n_timesteps,
        timestep_min=dt_min,
        n_cascade_levels=settings.n_cascade_levels,
        threshold_mm_h=settings.threshold_mm_h,
        downsample_factor=settings.downsample_factor,
        pixel_scale_m=float(now.xscale_m),
    )
    products = national_products(
        forecast,
        leads_min=settings.leads_min,
        threshold_mm_h=settings.threshold_mm_h,
        timestep_min=dt_min,
        frame_age_min=frame_age_min,
        downsample_factor=settings.downsample_factor,
    )
    del forecast

    # The UNcalibrated fractions, kept by reference before the curves
    # replace them: ``raw_frac_<lead>`` is the H-P model's main predictor
    # and the calibrated ``p_rain_<lead>`` is the baseline it has to beat,
    # so both have to survive the next block.
    raw_p_rain = dict(products.p_rain)

    # §B4: the calibrated grid REPLACES the raw one, exactly as the cycle
    # does it — a decision taken on a raw ensemble fraction is not the
    # decision the service takes.
    curves = _curves(settings.national_curves_path)
    if curves:
        from dataclasses import replace as _replace

        products = _replace(products, p_rain={
            int(lead): (
                products.p_rain[int(lead)] if curves.get(int(lead)) is None
                else curves[int(lead)].predict(products.p_rain[int(lead)])
            )
            for lead in products.leads_min
        })

    observed_grid = observed_rain_grid(
        rain_now, downsample_factor=settings.downsample_factor,
    )
    # Lead 0 of the deterministic series: the newest composite advected to
    # generated_at. The live cycle computes the whole series for its
    # overlays; a point decision only ever reads lead 0.
    forecast_now_field = next(iter(advect_field_series(
        rain_now, vy, vx,
        horizons_minutes=[frame_age_min],
        dt_minutes=dt_min,
    )))
    forecast_grids = {
        0: observed_rain_grid(
            forecast_now_field, downsample_factor=settings.downsample_factor,
        )
    }

    # The composite's OWN timestamp, not the one in its filename — the
    # live cycle stamps rows with ``composite_now.timestamp_utc`` and the
    # two must agree for a replay row and a live row to be comparable.
    stamped_ts = now.timestamp_utc
    if stamped_ts.tzinfo is None:
        stamped_ts = stamped_ts.replace(tzinfo=timezone.utc)
    generated_at = stamped_ts + timedelta(minutes=frame_age_min)
    season = postprocess.season_of_month(generated_at.month)
    out: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        sample = sample_point(
            products, geo, point.lat, point.lon,
            observed_mm_h=observed_grid,
            forecast_mm_h=forecast_grids,
        )
        series = sample.forecast_mm_h if sample else None
        row: dict[str, Any] = {
            "radar_ts": stamped_ts,
            "generated_at": generated_at,
            "station_id": point.id,
            "p_rain": sample.p_rain if sample else {},
            "eta_min": sample.eta_min if sample else None,
            "intensity_mm_h": sample.intensity_mm_h if sample else None,
            "observed_mm_h": sample.observed_mm_h if sample else None,
            "forecast_now_mm_h": series.get(0) if series else None,
        }
        if grid_features is not None:
            row["features"] = _feature_row(
                grid_features, index, point,
                raw_p_rain=raw_p_rain,
                pixel=None if sample is None else (sample.row, sample.col),
                leads_min=products.leads_min,
                season=season,
                hour_utc=generated_at.hour,
                frame_age_min=frame_age_min,
            )
        out.append(row)
    return out


#: ``(lat, lon)`` → km to the nearest radar, memoised per process. The
#: distance is a property of the station, not of the cycle, and the replay
#: asks for it once per station per frame.
_RADAR_KM_CACHE: dict[tuple[float, float], float] = {}


def station_radar_km(lat: float, lon: float) -> float:
    key = (round(float(lat), 6), round(float(lon), 6))
    hit = _RADAR_KM_CACHE.get(key)
    if hit is None:
        hit = float(nearest_radar_km(float(lat), float(lon)))
        _RADAR_KM_CACHE[key] = hit
    return hit


#: Non-finite → null, never 0. Kept as a module name because the replay's
#: own tests reach for it; the definition is the shared one.
_finite = postprocess.finite_or_none


def _feature_row(
    grid_features: dict[str, np.ndarray],
    index: int,
    point: StationPoint,
    *,
    raw_p_rain: dict[int, np.ndarray],
    pixel: tuple[int, int] | None,
    leads_min: Sequence[int],
    season: str,
    hour_utc: int,
    frame_age_min: float,
) -> dict[str, Any]:
    """One station's H-P feature columns for one cycle.

    ``pixel`` is the PRODUCT-grid pixel ``sample_point`` read, so the raw
    ensemble fraction comes off exactly the pixel the calibrated one did;
    ``None`` means the station is off coverage and every ensemble feature
    is unknown.

    The row itself is assembled by ``postprocess.feature_row`` — the same
    function the live cycle calls (``dmi_nowcast_sidecar.push.postprocess``),
    so a replay row and a live row for identical inputs are identical. All
    this adds is the two things only the replay knows: which product pixel
    this station read, and the memoised distance to the nearest radar.
    """
    return postprocess.feature_row(
        grid_features,
        index,
        raw_fractions={
            int(lead): (
                None if pixel is None or raw_p_rain.get(int(lead)) is None
                else raw_p_rain[int(lead)][pixel[0], pixel[1]]
            )
            for lead in leads_min
        },
        leads=leads_min,
        season=season,
        hour_utc=hour_utc,
        frame_age_min=frame_age_min,
        station_radar_km=station_radar_km(point.lat, point.lon),
    )


# ---------------------------------------------------------------------------
# Planning a day: which cycles run, on which frame
# ---------------------------------------------------------------------------


def plan_day(
    archive_dir: Path,
    day: date,
    settings: FrameSettings,
    *,
    start_min: int = 0,
    end_min: int = 24 * 60,
) -> list[tuple[Any, Any]]:
    """``[(selection, history | None)]`` — the cycles to replay for ``day``.

    A cycle belongs to the day of its **anchor frame**, not of the instant
    it runs at, so a 23:50 anchor decided at 00:05 still lands in the
    23:50 day's parquet — the same convention the flat model had, where a
    23:50 frame produced a 00:04 ``generated_at``. ``--start-utc`` /
    ``--end-utc`` likewise clip on the anchor frame.

    ``history is None`` means the anchor has no input triple behind it
    (the first frames of a day whose predecessors are outside the
    archive, or a gap). The caller records those as frame errors; that is
    what the pre-L3 loop did when the parser raised on a missing file.
    """
    cfg = settings.anchor
    day_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1) - timedelta(seconds=1)
    # Back far enough for the oldest frame an anchor at 00:00 could need
    # (its triple) and for a stale-but-eligible frame; forward far enough
    # that the last frame of the day is still seen by a poll.
    lookback = timedelta(minutes=cfg.max_age_min + 2 * FRAME_INTERVAL_MIN)
    lags = [cfg.lag.fullrange_min, cfg.lag.doppler_min]
    if cfg.frame_age_override_min is not None:
        lags.append(cfg.frame_age_override_min)
    tail = timedelta(minutes=max(lags) + max(cfg.poll_min, 0.0) + 1.0)

    frames = archive_frames(archive_dir, day_start - lookback, day_end)
    selections = anchor_policy.decision_instants(
        day_start, day_end + tail, frames,
        policy=cfg.policy,
        lag=cfg.lag,
        poll_interval_min=cfg.poll_min,
        max_age_min=cfg.max_age_min,
    )
    out: list[tuple[Any, Any]] = []
    for selection in selections:
        stamp = selection.timestamp
        if stamp.date() != day:
            continue
        if not start_min <= stamp.hour * 60 + stamp.minute <= end_min:
            continue
        out.append((selection, anchor_policy.anchor_history(
            selection, frames,
            step_min=FRAME_INTERVAL_MIN,
            tolerance_s=FRAME_TOLERANCE_S,
        )))
    return out


# ---------------------------------------------------------------------------
# One day: frames in order, the state machine carried along
# ---------------------------------------------------------------------------


def _engine():
    from dmi_nowcast_sidecar.push import engine as decision_engine

    return decision_engine


def state_to_json(state: Any) -> dict:
    return {
        "armed": bool(state.armed),
        "streak": int(state.streak),
        "below_since_utc": (
            state.below_since_utc.isoformat() if state.below_since_utc else None
        ),
        "last_eval_radar_ts": (
            state.last_eval_radar_ts.isoformat()
            if state.last_eval_radar_ts else None
        ),
    }


def state_from_json(raw: dict | None):
    eng = _engine()
    if not raw:
        return eng.INITIAL_STATE
    return eng.SubState(
        armed=bool(raw.get("armed", True)),
        streak=int(raw.get("streak", 0)),
        below_since_utc=_parse_iso(raw.get("below_since_utc")),
        last_eval_radar_ts=_parse_iso(raw.get("last_eval_radar_ts")),
    )


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def run_day(args: tuple) -> dict:
    """Worker entry point: one whole day, frames in order. Never raises.

    Every station starts ARMED with an empty streak — the day-parallel
    simplification stated in the module docstring.
    """
    (
        archive_dir_s, day_s, points, settings, rules, out_dir_s,
        start_min, end_min,
    ) = args
    started = time.time()
    day = date.fromisoformat(day_s)
    eng = _engine()
    engine_rules = eng.Rules(
        persistence_obs=int(rules["persistence_obs"]),
        rearm_after_min=int(rules["rearm_after_min"]),
        raining_now_eta_min=float(rules["raining_now_eta_min"]),
        raining_now_mm_h=float(rules["raining_now_mm_h"]),
    )
    threshold_pct = int(rules["threshold_pct"])
    lead = int(rules["lead_min"])

    result: dict[str, Any] = {
        "day": day_s, "rows": 0, "frames": 0, "errors": [],
        "state": {}, "elapsed_s": 0.0, "frame_ms": [], "failed": False,
        "anchor": anchor_policy.counts_template(), "frame_age_min": [],
    }
    try:
        archive_dir = Path(archive_dir_s)
        plan = plan_day(
            archive_dir, day, settings, start_min=start_min, end_min=end_min,
        )
        cache = CompositeCache(archive_dir)
        states = {p.id: eng.INITIAL_STATE for p in points}
        rows: list[dict[str, Any]] = []
        for selection, history in plan:
            t0 = time.time()
            if history is None:
                result["anchor"]["no_history"] += 1
                result["errors"].append(
                    f"{selection.timestamp:%Y-%m-%dT%H:%MZ}: no "
                    f"{selection.scan_type} input triple behind the anchor"
                )
                continue
            radar_ts = history.selection.timestamp
            try:
                samples = sample_frame(
                    cache, radar_ts, points, settings, history=history,
                )
            except Exception as exc:  # noqa: BLE001 — one frame, not the day
                result["errors"].append(
                    f"{radar_ts:%Y-%m-%dT%H:%MZ}: {type(exc).__name__}: {exc}"
                )
                continue
            anchor_policy.count_selection(result["anchor"], history)
            result["frame_age_min"].append(
                round(history.selection.frame_age_min, 2)
            )
            for sample in samples:
                station = sample["station_id"]
                obs = eng.Observation(
                    radar_ts_utc=sample["radar_ts"],
                    p_rain=sample["p_rain"].get(lead),
                    eta_min=sample["eta_min"],
                    intensity_mm_h=sample["intensity_mm_h"],
                    observed_mm_h=sample["observed_mm_h"],
                    forecast_now_mm_h=sample["forecast_now_mm_h"],
                )
                decision = eng.evaluate(
                    states[station], obs,
                    threshold_pct=threshold_pct,
                    quiet=None,
                    tz="UTC",
                    now_utc=sample["generated_at"],
                    rules=engine_rules,
                )
                states[station] = decision.state
                rows.append({
                    "radar_ts": sample["radar_ts"],
                    "generated_at": sample["generated_at"],
                    "station_id": station,
                    # The rule's lead — the number the decision was taken
                    # on — plus every served lead beside it, so a
                    # threshold/horizon sweep needs no second STEPS run.
                    "p_rain": obs.p_rain,
                    **per_lead_columns(sample["p_rain"]),
                    "eta_min": obs.eta_min,
                    "intensity_mm_h": obs.intensity_mm_h,
                    "observed_mm_h": obs.observed_mm_h,
                    "forecast_now_mm_h": obs.forecast_now_mm_h,
                    "action": decision.action,
                    "armed_after": decision.state.armed,
                    "streak_after": decision.state.streak,
                    # The percent this row was decided at — the served
                    # table's pick for the run's lead, or the fallback.
                    # It moves with every refit, so a row is only
                    # interpretable beside the rule it was taken under.
                    "threshold_pct": threshold_pct,
                    # H-P: additive, and only when --features is on.
                    **(sample.get("features") or {}),
                })
            result["frames"] += 1
            result["frame_ms"].append(round((time.time() - t0) * 1000.0, 1))
        out_path = Path(out_dir_s) / "decisions" / f"{day_s}.parquet"
        write_decisions(
            out_path, rows, settings.leads_min, features=settings.features,
        )
        result["rows"] = len(rows)
        result["state"] = {sid: state_to_json(s) for sid, s in states.items()}
    except Exception as exc:  # noqa: BLE001 — a dead day must not kill the run
        result["errors"].append(f"{day_s}: {type(exc).__name__}: {exc}")
        # Only a day-level failure is unresumable; a frame that threw is
        # recorded and the rest of the day still counts as replayed.
        result["failed"] = True
    result["elapsed_s"] = round(time.time() - started, 2)
    return result


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------


def _write_table_atomic(table, path: Path) -> None:
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def feature_schema(leads_min=None):
    """Arrow fields for the H-P feature columns, in write order.

    The schema itself is ``postprocess.feature_schema`` — shared with the
    live cycle's writer (``station_eval``), so the replay's parquet and
    the live parquet concatenate. All this adds is the replay's default
    lead set.
    """
    return postprocess.feature_schema(
        NATIONAL_LEADS if leads_min is None else leads_min,
    )


def feature_documentation(leads_min=None) -> dict[str, str]:
    """``{column: definition}`` — what goes in ``summary.json``."""
    return postprocess.feature_documentation(
        NATIONAL_LEADS if leads_min is None else leads_min,
    )


def write_decisions(
    path: Path, rows: Sequence[dict], leads_min=None, *, features: bool = False,
) -> None:
    """Write one day's decision rows, atomically, in the shared schema.

    ``features`` appends the H-P columns after the shared ones. A row
    missing one contributes a null, exactly as in the base schema.
    """
    table = decision_table(rows, leads_min)
    if features:
        import pyarrow as pa

        for field_ in feature_schema(leads_min):
            table = table.append_column(field_, pa.array(
                [row.get(field_.name) for row in rows], type=field_.type,
            ))
    _write_table_atomic(table, path)


def read_decisions(path: Path, leads_min=None) -> list[dict]:
    """Read one day's rows, tolerating a file written under other leads.

    Deliberately NOT ``read_table(..., schema=...)``: a parquet written
    before the per-lead columns existed has only the base columns, and
    pinning the schema on read would refuse it. The file is read as it is
    and then aligned to the union schema, so an old day and a new day are
    the same dict shape by the time anything scores them.
    """
    import pyarrow.parquet as pq

    return align_decision_table(
        pq.read_table(path), leads_min,
    ).to_pylist()


def write_events(path: Path, rows: Sequence[dict]) -> None:
    import pyarrow as pa

    schema = pa.schema([
        ("station_id", pa.string()),
        ("sent_utc", pa.timestamp("us", tz="UTC")),
        ("eta_min", pa.float32()),
        ("outcome", pa.string()),
        ("onset_utc", pa.timestamp("us", tz="UTC")),
        ("lead_error_min", pa.float32()),
    ])
    table = pa.table(
        {
            name: pa.array([r.get(name) for r in rows], type=schema.field(name).type)
            for name in schema.names
        },
        schema=schema,
    )
    _write_table_atomic(table, path)


def write_onsets(path: Path, rows: Sequence[dict]) -> None:
    import pyarrow as pa

    schema = pa.schema([
        ("station_id", pa.string()),
        ("onset_utc", pa.timestamp("us", tz="UTC")),
        ("outcome", pa.string()),
        ("sent_utc", pa.timestamp("us", tz="UTC")),
        ("lead_error_min", pa.float32()),
    ])
    table = pa.table(
        {
            name: pa.array([r.get(name) for r in rows], type=schema.field(name).type)
            for name in schema.names
        },
        schema=schema,
    )
    _write_table_atomic(table, path)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False, default=str)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Gauge truth + scoring
# ---------------------------------------------------------------------------


def day_slots(
    store: Any,
    day: date,
    station_ids: Sequence[str],
    *,
    pad_min: int = GAUGE_PAD_MIN,
) -> dict[str, list[tuple[datetime, bool | None, float | None]]]:
    """Gauge slots for one day ± ``pad_min``, per station.

    ``(slot_end, wet, mm)``: the depth rides along because the onset rule
    asks the onset slot and the one after it for millimetres, and a grid
    of bare wet flags cannot answer that.

    The pad is what lets a 23:5x warning find its onset after midnight,
    and what gives the first slots of the day the dry slots the onset rule
    needs behind them.
    """
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc) - timedelta(
        minutes=pad_min
    )
    end = start + timedelta(days=1, minutes=2 * pad_min)
    table = store.read(
        start, end, [PRECIP_PARAM, PRECIP_DUR_PARAM], list(station_ids),
    )
    return {
        sid: gauge_slot_amounts(table, sid, start_utc=start, end_utc=end)
        for sid in station_ids
    }


def merge_slots(
    into: dict[str, dict[datetime, tuple[bool | None, float | None]]],
    add: dict[str, list[tuple[datetime, bool | None, float | None]]],
) -> None:
    """Union day windows, preferring a determined value over ``None``."""
    for sid, slots in add.items():
        target = into.setdefault(sid, {})
        for ts, wet, mm in slots:
            if wet is not None or ts not in target:
                target[ts] = (wet, mm)


def score(
    decisions: Sequence[dict],
    slots_by_day: Sequence[
        dict[str, list[tuple[datetime, bool | None, float | None]]]
    ],
    points: Sequence[StationPoint],
    *,
    lead_min: int,
    tolerance_min: int,
    dry_min: int,
    onset_min_mm: float = DEFAULT_ONSET_MIN_MM,
    threshold_mm_h: float,
) -> tuple[
    dict[str, ScoreResult],
    dict[str, list[tuple[datetime, bool | None, float | None]]],
    dict,
]:
    """Per-station scores, the merged slot grid, and the pooled agreement.

    Onsets are computed per day window and then **deduplicated by instant**
    across windows: an onset near a window edge is detected in whichever
    window holds its dry evidence, and counted once.
    """
    onset_by_station: dict[str, set[datetime]] = {p.id: set() for p in points}
    for window in slots_by_day:
        for sid, slots in window.items():
            onset_by_station.setdefault(sid, set()).update(
                onsets(slots, dry_min, onset_min_mm=onset_min_mm),
            )

    merged: dict[str, dict[datetime, tuple[bool | None, float | None]]] = {}
    for window in slots_by_day:
        merge_slots(merged, window)
    slot_lists = {
        sid: [(ts, wet, mm) for ts, (wet, mm) in sorted(grid.items())]
        for sid, grid in merged.items()
    }

    warnings_by_station: dict[str, list[tuple[datetime, float | None]]] = {
        p.id: [] for p in points
    }
    frames_by_station: dict[str, list[datetime]] = {p.id: [] for p in points}
    for row in decisions:
        if row.get("radar_ts") is not None:
            frames_by_station.setdefault(row["station_id"], []).append(
                row["radar_ts"],
            )
        if row.get("action") != "notify":
            continue
        warnings_by_station.setdefault(row["station_id"], []).append(
            (row["generated_at"], row.get("eta_min"))
        )

    # The intervals each station was actually being watched over. A replay
    # runs contiguous days, so this is normally one run per day-block plus
    # the lead window at its end — but a resumed run with a missing day
    # must not count that day's rain as misses, and the gauge archive
    # reaches back years further than any replay does.
    coverage_by_station = {
        sid: coverage_runs(stamps, extend_min=lead_min + tolerance_min)
        for sid, stamps in frames_by_station.items()
    }

    # The last slot each station actually reported. A warning whose window
    # reaches past it has not come due yet, and neither has an onset within
    # tolerance of it — both come back "pending" rather than being graded
    # on evidence that does not exist. Matters on the trailing edge of a
    # replay run as much as it does live.
    known_until = {
        sid: max(
            (ts for ts, wet, _mm in slots if wet is not None),
            default=None,
        )
        for sid, slots in slot_lists.items()
    }
    results = {
        sid: score_warnings(
            warnings_by_station.get(sid, []),
            sorted(onset_by_station.get(sid, ())),
            lead_min=lead_min,
            tolerance_min=tolerance_min,
            dry_min=dry_min,
            onset_min_mm=onset_min_mm,
            known_until=known_until.get(sid),
            coverage=coverage_by_station.get(sid),
        )
        for sid in (p.id for p in points)
    }
    agreement = raining_now_agreement(
        decisions, slot_lists, threshold_mm_h=threshold_mm_h,
    )
    return results, slot_lists, agreement


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _split(arg: str | None) -> list[str]:
    return [x.strip() for x in arg.split(",") if x.strip()] if arg else []


def _hhmm_to_min(value: str | None, default: int) -> int:
    if not value:
        return default
    hh, _, mm = value.partition(":")
    return int(hh) * 60 + int(mm or 0)


def _load_progress(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {"version": 1, "days": {}}
    try:
        raw = json.loads(path.read_text())
    except Exception:  # noqa: BLE001 — a corrupt progress file restarts the run
        return {"version": 1, "days": {}}
    if not isinstance(raw, dict) or "days" not in raw:
        return {"version": 1, "days": {}}
    return raw


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--archive-dir", required=True, type=Path,
                   help="root holding YYYY/MM/dk.com.*.500_max.h5")
    p.add_argument("--corpus-dir", required=True, type=Path,
                   help="station observation store root (holds stations/obs/)")
    p.add_argument("--points", required=True, type=Path,
                   help="station points JSON (version 2)")
    p.add_argument("--days", help="comma-separated YYYY-MM-DD")
    p.add_argument("--days-file", type=Path, help="one YYYY-MM-DD per line")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument(
        "--frame-age-min", type=float, default=None,
        help="Flat simulated frame age for EVERY frame, the pre-L3 model. "
             "Given, it overrides the per-product publication lag and the "
             "poll grid, reproducing an older run exactly (the runs before "
             "2026-09-09 used 14). Omitted, the age comes from "
             "--lag-fullrange-min / --lag-doppler-min plus the wait for the "
             "next poll, so a fullRange cycle sits at 15 min.",
    )
    p.add_argument(
        "--anchor", choices=anchor_policy.POLICIES,
        default=anchor_policy.POLICY_FULLRANGE,
        help="Which frame each cycle stands on (Phase H, L3). 'fullRange' "
             "is today's product; 'freshest' anchors on the :x5 doppler "
             "composite, harmonised through --harmonisation and filled from "
             "the newest fullRange frame outside its 120 km range.",
    )
    p.add_argument(
        "--harmonisation", type=Path, default=None,
        help="L2 doppler->fullRange quantile map (doppler_harmonisation.json). "
             "Required by --anchor freshest: raw doppler carries 20-30 %% "
             "less echo at 20 dBZ and must never enter the chain unmapped.",
    )
    p.add_argument(
        "--lag-fullrange-min", type=float,
        default=anchor_policy.DEFAULT_LAG_FULLRANGE_MIN,
        help="Publication lag of the :x0 composite (measured 2026-09-06/07).",
    )
    p.add_argument(
        "--lag-doppler-min", type=float,
        default=anchor_policy.DEFAULT_LAG_DOPPLER_MIN,
        help="Publication lag of the :x5 composite (measured 2026-09-06/07).",
    )
    p.add_argument(
        "--poll-interval-min", type=float,
        default=anchor_policy.DEFAULT_POLL_INTERVAL_MIN,
        help="The cycle's poll cadence. A frame is first seen at the first "
             "poll at or after it is published; a poll that finds no new "
             "frame is not a decision instant (the runtime's no-new-frame "
             "fast path), which also keeps radar_ts unique per decision.",
    )
    p.add_argument(
        "--anchor-history", choices=anchor_policy.HISTORY_MODES,
        default=anchor_policy.HISTORY_SAME_TYPE,
        help="What the flow and the cascade eat under a doppler anchor. "
             "'same-type' is DECIDE-3 as written: three doppler frames, so "
             "p_rain exists only inside doppler's 120 km range and six "
             "gauges go dark. 'filled' makes every history frame a "
             "per-pixel freshest composite instead — each pixel's series "
             "is still one product at 10-min spacing, and the national "
             "grid keeps fullRange's coverage.",
    )
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--rules",
        help="k=v,... over the live subscriber row. threshold_pct here is "
             "the FALLBACK percent, used only when --thresholds is not "
             "given: since Phase G the service warns at the fitted table's "
             "pick for the subscriber's horizon, not at a constant.",
    )
    p.add_argument(
        "--thresholds", type=Path,
        help="a served push_thresholds.json; its pick for --rules lead_min "
             "overrides threshold_pct, so the tree is generated under the "
             "rule the service is actually on. Recorded in the run summary "
             "as rules.threshold_source (table|fallback|rules).",
    )
    p.add_argument("--progress", type=Path,
                   help="JSON progress file; a finished day is not redone")
    p.add_argument("--national-curves", type=Path,
                   help="isotonic curve file the service serves (§B4)")
    p.add_argument("--start-utc", help="clip each day at HH:MM (inclusive)")
    p.add_argument("--end-utc", help="clip each day at HH:MM (inclusive)")
    p.add_argument("--tolerance-min", type=int, default=DEFAULT_TOLERANCE_MIN)
    p.add_argument("--dry-min", type=int, default=DEFAULT_DRY_MIN)
    p.add_argument("--onset-min-mm", type=float, default=DEFAULT_ONSET_MIN_MM,
                   help="millimetres an onset must deliver over the onset "
                        "slot and the one after it; 0 counts every wet slot "
                        "after a dry spell")
    p.add_argument("--ensemble-size", type=int, default=ENSEMBLE_SIZE)
    p.add_argument("--cascade-levels", type=int, default=N_CASCADE_LEVELS)
    p.add_argument("--downsample-factor", type=int, default=DOWNSAMPLE_FACTOR)
    p.add_argument("--horizon-min", type=int, default=HORIZON_MIN)
    p.add_argument(
        "--flow-completion", choices=("bulk", "confidence"),
        default=DEFAULT_FLOW_COMPLETION,
        help="Motion-completion policy (config.py "
             "forecast.flow_completion). 'confidence' is the H-F hotfix "
             "(2026-09-08), 'bulk' the behaviour before it — the two sides "
             "of the A/B. Recorded in the run summary.",
    )
    p.add_argument(
        "--features", action=argparse.BooleanOptionalAction, default=True,
        help="write the H-P post-processing features beside each decision "
             "(scripts/fit_postprocess.py trains on them). Additive "
             "columns: every existing consumer conforms the file to the "
             "shared decision schema and ignores them, so a run with "
             "features scores identically to one without. --no-features "
             "reproduces the pre-2026-09-09 file exactly.",
    )
    p.add_argument("--no-score", action="store_true",
                   help="replay only; skip the gauge scoring pass")
    args = p.parse_args(argv)

    rules, threshold_source = apply_thresholds(
        parse_rules(args.rules), args.thresholds,
    )
    points = load_points(args.points)
    days = _split(args.days)
    if args.days_file:
        days += [
            ln.strip() for ln in args.days_file.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    if not days:
        p.error("need --days or --days-file")
    days = sorted(set(days))
    start_min = _hhmm_to_min(args.start_utc, 0)
    end_min = _hhmm_to_min(args.end_utc, 24 * 60)

    anchor_settings = AnchorSettings(
        policy=args.anchor,
        lag_fullrange_min=float(args.lag_fullrange_min),
        lag_doppler_min=float(args.lag_doppler_min),
        poll_interval_min=float(args.poll_interval_min),
        frame_age_override_min=(
            None if args.frame_age_min is None else float(args.frame_age_min)
        ),
        harmonisation_path=(
            str(args.harmonisation) if args.harmonisation else None
        ),
        history_mode=args.anchor_history,
    )
    if anchor_settings.policy == anchor_policy.POLICY_FRESHEST:
        if not anchor_settings.harmonisation_path:
            p.error("--anchor freshest requires --harmonisation")
        try:
            # Fail here, not 4,000 frames in: an unreadable or
            # wrong-schema table cannot be applied to a single frame.
            anchor_policy.load_harmonisation(anchor_settings.harmonisation_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            p.error(f"--harmonisation: {exc}")

    settings = FrameSettings(
        frame_age_min=(
            DEFAULT_FRAME_AGE_MIN if args.frame_age_min is None
            else float(args.frame_age_min)
        ),
        anchor=anchor_settings,
        ensemble_size=int(args.ensemble_size),
        n_cascade_levels=int(args.cascade_levels),
        downsample_factor=int(args.downsample_factor),
        horizon_min=int(args.horizon_min),
        leads_min=NATIONAL_LEADS,
        threshold_mm_h=RAIN_THRESHOLD_MM_H,
        national_curves_path=(
            str(args.national_curves) if args.national_curves else None
        ),
        flow_completion=args.flow_completion,
        features=bool(args.features),
    )
    out_dir = Path(args.out_dir)
    (out_dir / "decisions").mkdir(parents=True, exist_ok=True)

    progress = _load_progress(args.progress)
    done = {
        d for d, entry in progress["days"].items()
        if entry.get("status") == "done"
        and (out_dir / "decisions" / f"{d}.parquet").is_file()
    }
    todo = [d for d in days if d not in done]
    if done:
        print(f"resuming: {len(done)} day(s) already done, {len(todo)} to go",
              file=sys.stderr)

    tasks = [
        (str(args.archive_dir), d, points, settings, rules, str(out_dir),
         start_min, end_min)
        for d in todo
    ]
    errors: list[str] = []
    frame_ms: list[float] = []
    anchor_counts = anchor_policy.counts_template()
    frame_ages: list[float] = []
    if args.workers <= 1:
        results = (run_day(t) for t in tasks)
        for i, res in enumerate(results, 1):
            _absorb(res, progress, args.progress, errors, frame_ms, i,
                    len(tasks), anchor_counts, frame_ages)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_day, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                _absorb(
                    fut.result(), progress, args.progress, errors, frame_ms,
                    i, len(tasks), anchor_counts, frame_ages,
                )

    decisions: list[dict] = []
    n_frames = 0
    for d in days:
        path = out_dir / "decisions" / f"{d}.parquet"
        if not path.is_file():
            continue
        rows = read_decisions(path, settings.leads_min)
        decisions += rows
        n_frames += len({r["radar_ts"] for r in rows})

    # Anchor totals over every requested day, not only the ones this
    # invocation replayed: a resumed run would otherwise report the
    # candidate as having anchored on almost nothing.
    all_anchor_counts = anchor_policy.counts_template()
    all_frame_ages: list[float] = []
    for d in days:
        entry = progress["days"].get(d) or {}
        if d in done:
            anchor_policy.sum_counts(all_anchor_counts, entry.get("anchor"))
            all_frame_ages.extend(entry.get("frame_age_min") or ())
    anchor_policy.sum_counts(all_anchor_counts, anchor_counts)
    all_frame_ages.extend(frame_ages)

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": {
            "days": days,
            "n_days": len(days),
            "n_frames": n_frames,
            "n_stations": len(points),
            "n_decision_rows": len(decisions),
            # The flat simulated age, or null when the age came from the
            # publication lags. A parity key in benchmark_report: two runs
            # that disagree here are not measuring the same thing.
            "frame_age_min": settings.anchor.frame_age_override_min,
            # L3: which frame each cycle stood on, what it cost in
            # freshness, and how often the candidate fell back to the
            # baseline. Everything needed to read the run's numbers.
            "anchor": {
                **settings.anchor.as_dict(),
                "counts": all_anchor_counts,
                "frame_age_min": anchor_policy.frame_age_stats(all_frame_ages),
            },
            # The effective rule, and where its percent came from: a
            # tree generated at the fitted table's pick reads very
            # differently from one generated at the 40 % fallback, and
            # nothing else in the run records which happened.
            "rules": {**rules, "threshold_source": threshold_source},
            "steps": {
                "ensemble_size": settings.ensemble_size,
                "n_cascade_levels": settings.n_cascade_levels,
                "downsample_factor": settings.downsample_factor,
                "horizon_min": settings.horizon_min,
                "leads_min": list(settings.leads_min),
                "threshold_mm_h": settings.threshold_mm_h,
            },
            # H-F: which motion field these decisions were made on. The
            # completion changes the STEPS velocity and therefore every
            # probability in the table, so a replay's numbers mean nothing
            # without it.
            "flow": {
                "completion": settings.flow_completion,
                "confidence_window_px": settings.flow_confidence_window_px,
                "confidence_percentile": settings.flow_confidence_percentile,
                "texture_percentile": settings.flow_texture_percentile,
            },
            "national_curves": settings.national_curves_path,
            # H-P: what the extra columns in decisions/*.parquet mean.
            # Written whether or not this run produced them, so a reader
            # of an old run can see what it is missing.
            "features": {
                "enabled": settings.features,
                "columns": feature_documentation(settings.leads_min),
                "wet_mm_h": postprocess.WET_MM_H,
                "obs_disc_km": postprocess.OBS_DISC_KM,
                "upstream_corridor": {
                    "half_width_km": postprocess.UPSTREAM_HALF_WIDTH_KM,
                    "near_km": postprocess.UPSTREAM_NEAR_KM,
                    "far_km": postprocess.UPSTREAM_FAR_KM,
                    "step_km": postprocess.UPSTREAM_STEP_KM,
                },
            },
            "archive_dir": str(args.archive_dir),
            "corpus_dir": str(args.corpus_dir),
            "points_file": str(args.points),
            "frame_ms": {
                "n": len(frame_ms),
                "mean": round(sum(frame_ms) / len(frame_ms), 1) if frame_ms else None,
                "min": round(min(frame_ms), 1) if frame_ms else None,
                "max": round(max(frame_ms), 1) if frame_ms else None,
            },
            "errors": errors[:100],
            "n_errors": len(errors),
        },
    }

    if args.no_score:
        summary["gauge"] = {"available": False, "reason": "--no-score"}
    else:
        summary.update(_score_and_write(
            out_dir, decisions, days, points, args, rules,
        ))
    _write_json_atomic(out_dir / "summary.json", summary)
    print(json.dumps(summary.get("pooled", summary["run"]), indent=2, default=str))
    return 0


def _absorb(
    res: dict,
    progress: dict,
    progress_path: Path | None,
    errors: list[str],
    frame_ms: list[float],
    index: int,
    total: int,
    anchor_counts: dict[str, int] | None = None,
    frame_ages: list[float] | None = None,
) -> None:
    """Fold one finished day into the run state and persist progress."""
    errors.extend(res["errors"])
    frame_ms.extend(res["frame_ms"])
    if anchor_counts is not None:
        anchor_policy.sum_counts(anchor_counts, res.get("anchor"))
    if frame_ages is not None:
        frame_ages.extend(res.get("frame_age_min") or ())
    progress["days"][res["day"]] = {
        "status": "failed" if res.get("failed") else "done",
        "rows": res["rows"],
        "frames": res["frames"],
        "elapsed_s": res["elapsed_s"],
        "state": res["state"],
        "n_errors": len(res["errors"]),
        # Which product each of the day's cycles stood on, so a resumed
        # run can still report the anchor totals for days it skipped.
        "anchor": res.get("anchor"),
        "frame_age_min": res.get("frame_age_min"),
    }
    if progress_path is not None:
        _write_json_atomic(progress_path, progress)
    print(
        f"[{index}/{total}] {res['day']}: {res['frames']} frames, "
        f"{res['rows']} rows, {res['elapsed_s']}s"
        + (f", {len(res['errors'])} error(s)" if res["errors"] else ""),
        file=sys.stderr, flush=True,
    )


def _score_and_write(
    out_dir: Path,
    decisions: list[dict],
    days: Sequence[str],
    points: Sequence[StationPoint],
    args: Any,
    rules: dict,
) -> dict:
    """The gauge pass: onsets, scores, events.parquet, onsets.parquet."""
    try:
        from dmi_nowcast_core.station_store import StationObsStore
    except Exception as exc:  # noqa: BLE001 — the store may not be built yet
        return {"gauge": {
            "available": False,
            "reason": f"station store unavailable: {type(exc).__name__}: {exc}",
        }}
    station_ids = [p.id for p in points]
    store = StationObsStore(args.corpus_dir)
    windows = []
    for d in days:
        try:
            windows.append(day_slots(store, date.fromisoformat(d), station_ids))
        except Exception as exc:  # noqa: BLE001 — one unreadable month
            windows.append({})
            print(f"gauge read failed for {d}: {exc}", file=sys.stderr)
    n_known = sum(
        1 for w in windows for slots in w.values() for _ts, wet, _mm in slots
        if wet is not None
    )
    if n_known == 0:
        return {"gauge": {
            "available": False,
            "reason": "no gauge observations in the store for these days",
        }}

    results, slot_lists, agreement = score(
        decisions, windows, points,
        lead_min=int(rules["lead_min"]),
        tolerance_min=int(args.tolerance_min),
        dry_min=int(args.dry_min),
        onset_min_mm=float(args.onset_min_mm),
        threshold_mm_h=float(rules["raining_now_mm_h"]),
    )
    events = [
        {
            "station_id": sid,
            "sent_utc": w.sent_utc,
            "eta_min": w.eta_min,
            "outcome": w.outcome,
            "onset_utc": w.onset_utc,
            "lead_error_min": w.lead_error_min,
        }
        for sid, res in results.items() for w in res.warnings
    ]
    events.sort(key=lambda r: (r["sent_utc"], r["station_id"]))
    write_events(out_dir / "events.parquet", events)
    onset_rows = [
        {
            "station_id": sid,
            "onset_utc": o.onset_utc,
            "outcome": o.outcome,
            "sent_utc": o.sent_utc,
            "lead_error_min": o.lead_error_min,
        }
        for sid, res in results.items() for o in res.onsets
    ]
    onset_rows.sort(key=lambda r: (r["onset_utc"], r["station_id"]))
    write_onsets(out_dir / "onsets.parquet", onset_rows)

    by_station: dict[str, dict] = {}
    for point in points:
        res = results[point.id]
        rows = [r for r in decisions if r["station_id"] == point.id]
        by_station[point.id] = {
            "lat": point.lat,
            "lon": point.lon,
            "region": point.region,
            "n_rows": len(rows),
            "n_slots_known": sum(
                1 for _ts, wet, _mm in slot_lists.get(point.id, ())
                if wet is not None
            ),
            "warnings": res.summary,
            "raining_now": raining_now_agreement(
                rows, {point.id: slot_lists.get(point.id, [])},
                threshold_mm_h=float(rules["raining_now_mm_h"]),
            ),
        }
    return {
        "gauge": {
            "available": True,
            "dry_min": int(args.dry_min),
            "onset_min_mm": float(args.onset_min_mm),
            "tolerance_min": int(args.tolerance_min),
            "n_known_slots": n_known,
            "n_stations_with_obs": sum(
                1 for sid in slot_lists
                if any(row[1] is not None for row in slot_lists[sid])
            ),
        },
        "pooled": {
            "warnings": pooled_summary(
                results.values(),
                lead_min=int(rules["lead_min"]),
                tolerance_min=int(args.tolerance_min),
                dry_min=int(args.dry_min),
                onset_min_mm=float(args.onset_min_mm),
            ),
            "raining_now": agreement,
        },
        "stations": by_station,
    }


if __name__ == "__main__":
    raise SystemExit(main())
