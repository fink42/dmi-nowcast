"""Build a multi-point national calibration corpus from the local radar archive.

Every calibration point is sampled from the same ensemble run, so the
corpus calibrates exactly the quantity the service serves. The old
single-point mode is retired: any one point's curve is just that point's
row subset of this corpus.

For each sampled event time ``T`` over the past N days:

- **Input frames** at ``T-20``, ``T-10``, ``T`` (the three frames STEPS
  expects, spaced at the single-scan-type frame cadence of 10 min).
- **Verification frames** at ``T + lead`` snapped to the same 10-min
  frame grid (snapping rule below).

**Scan type (fullRange-only — Phase B addendum 2026-08-29)** — DMI's
composite collection interleaves two products: **fullRange** at minutes
:x0 and **doppler** at :x5. Mixing them alternates between materially
different views (doppler covers only ~40% of fullRange's area), so the
runtime went fullRange-only and the corpus must match. ``--scan-type``
(default ``fullRange``) filters EVERY frame listing and download —
input frames, verification frames, and the wet/dry index construction.
API listings pass DMI's ``scanType`` parameter; and because composite
filenames (``dk.com.YYYYMMDDhhmm.500_max.h5``) carry no scan-type
marker — the filename minute IS the marker, encoding the interleave —
any frame resolved locally by filename/timestamp is additionally
filtered by minute-of-hour: ``minute % 10 == 0`` is fullRange,
``minute % 10 == 5`` is doppler (:func:`feature_matches_scan_type`).
Event anchors are sampled on the chosen type's 10-min grid (whole hours
for fullRange; hh:05 for doppler).

**Archive-first listing (2026-09-02)** — *why*: DMI's items API lists
only the last **180 days**, so an API-driven build can never reach an
event older than that no matter what ``--days-back`` says: the listing
comes back empty, the hour classifies as "unknown", and the event is
dropped even when its frames are sitting on disk. The persistent corpus
archive (``--corpus-dir``, ``composites/YYYY/MM/``) has no such bound —
it only grows — so it, not the API, is the long-term record and is
consulted FIRST.

Every window listing in this builder — the wet/dry hour classification,
the event's input frames, and its verification frames — goes through
:func:`_list_frames_in_window`, whose order is:

1. :class:`dmi_nowcast_core.corpus.ArchiveIndex` — an in-memory
   ``(time, scan type, path)`` listing built once per process from
   ``os.scandir`` alone (no stat, no open, no HDF5 parse). A hit resolves
   straight to the archived path: :func:`_resolve_frame` short-circuits
   on :class:`~dmi_nowcast_core.corpus.ArchivedFrame`, so an archive hit
   NEVER touches the network.
2. DMI's ``list_in_window`` — only when the archive holds no frame at all
   in that window. Resolution then behaves exactly as before: reuse the
   canonical slot if present, otherwise download into it (so the archive
   keeps growing).

A window the archive covers only *partially* is therefore not topped up
from the API. That is deliberate: past 180 days the API cannot help
anyway, and inside 180 days a hole in the archive is a gap to repair with
``sidecar/deploy/backfill_corpus.sh``, not to paper over on every
calibration run. The cost of a hole is bounded and visible — a missing
verification frame leaves that lead's outcome null, a missing input frame
errors the event with "(archive listing)" in the message.

*Scan type from the filename*: an archived frame carries no ``scanType``
label, so :func:`~dmi_nowcast_core.corpus.scan_type_from_filename` reads
it off the minute — ``:x0`` is fullRange, ``:x5`` is doppler, anything
else is ``unknown`` and matches NO product filter. Verified against DMI's
own ``scanType``-labelled listing for 12 consecutive frames on
2026-09-02. The index filters by equality against a concrete product, so
an ``unknown`` frame can never be mistaken for fullRange.

*Window*: ``--days-back 0`` means "as far back as the archive goes" —
:func:`_candidate_hours` starts the window at the first whole hour at or
after :meth:`ArchiveIndex.earliest` for the build's scan type (and
requires ``--corpus-dir``). Any positive value keeps the old meaning.
Like the wet reference set, the window is a *sampling* choice: it decides
which hours are candidates, not what quantity a row measures, so it is
deliberately NOT part of ``settings_hash`` and a corpus can be extended
backwards across runs. The actual window is logged and recorded in the
progress JSON (``event_window``) instead.

**Verification snapping rule** — a nominal lead ``L`` reads ensemble
timestep ``ceil((L + frame_age_min) / timestep_min)`` (frame-age
convention below), so the verification frame is resolved at that
timestep's radar time:
``T + ceil((L + frame_age_min) / timestep_min) * timestep_min`` —
``T + lead + age`` snapped UP to the 10-min frame grid. At the default
12–18 min simulated age, lead 30 therefore always verifies against the
fullRange frame at ``T+50`` and lead 60 against ``T+80`` — exactly the
instants the *served* probabilities for those leads describe. Leads
whose ``L + age`` straddles a grid line (lead 5 at ``T+20`` or ``T+30``,
depending on the draw) land on either side of it, again exactly as the
live service does. The usual ±4 min match tolerance applies around the
*snapped* target; when no frame lands inside tolerance the lead's
outcome stays null.

**Wet/dry index cache re-key** — the index cache filename incorporates
both the scan type and the wet-reference set
(``wet_dry_index_fullRange_<refs-hash>.json``): wet/dry counts change
under type filtering and under a different reference set, so a stale
``wet_dry_index.json`` / single-reference index is deliberately never
reused.

**Multi-point wet reference** — an hour counts as wet when rain fell
within ``search_km`` (60 km) of ANY reference in the set, and the set
defaults to five spread references (:data:`DEFAULT_WET_REFS`: NW
Jutland, NE Jutland, SW Jutland, Zealand, Bornholm — Bornholm because it
is meteorologically isolated). The sample weights fix the base rate
whatever the strata are, but WHICH storms get oversampled still tilts
toward the reference's neighbourhood, and a national product should
sample national weather. The reference set is deliberately NOT part of
``settings_hash``: it changes the sampling strata only, and the
per-row weights stay self-consistent, so a corpus can be resumed or
appended across a reference-set change without mixing incompatible
quantities.

One STEPS ensemble runs per event (``run_ensemble``), is reduced to the
national grids via :func:`dmi_nowcast_core.national.national_products`,
and ALL calibration points are sampled from those grids — so calibration
calibrates exactly the ``p_rain`` quantity the website serves. Outcomes
come from the verification frames sampled at each point's disc with the
runtime detection statistic.

**Frame-age convention — the corpus simulates the live latency**
(``--frame-age-range``, default ``12,18``). The service does not compute
at radar-frame time: a cycle finishes ~12–18 min after the timestamp of
its newest frame, and the runtime corrects for that gap. "P(rain within
``L`` minutes **from now**)" is read from ensemble timestep
``ceil((L + frame_age) / timestep_min)``, which is valid at radar time
``T + ceil((L + frame_age) / timestep_min) * timestep_min``.

A corpus built at zero age fits lead ``L`` against timestep
``ceil(L / timestep_min)`` while the service serves lead ``L`` from a
LATER timestep: on the 10-min grid, live "lead 30" reads +50 min from
the image, but a zero-age curve for lead 30 was fitted on +30 min. The
curve and the quantity it corrects would describe different horizons —
so the builder simulates the latency instead. Each event draws

    frame_age_min ~ Uniform(LO, HI)

from an RNG seeded on ``(--seed, event time)``, and that age is used
consistently everywhere for the event: ``national_products(...,
frame_age_min=age)`` for the forecast, ``T + ceil((L + age) / timestep)
* timestep`` for the verification instant (snapping rule above), and a
per-row ``frame_age_min`` column for the record. Fitted and served leads
then coincide. Because the age is a pure function of the seed and the
event time, a resumed or appended build redraws exactly the same age for
the same event; ``n_timesteps`` is grown to ``ceil((max lead + HI) /
timestep_min)`` so the longest served lead is inside the ensemble
horizon (8 timesteps = 80 min at the defaults).

``--frame-age-range 0,0`` reproduces the old zero-age convention
exactly — but under a DIFFERENT ``settings_hash``, because
``frame_age_range`` is a new key in the hashed settings dict. No
pre-frame-age corpus can therefore be resumed or appended to; that is
deliberate, since its rows verify at different instants. (The ±4 min
frame-matching tolerance jitters inputs and verification alike and has
no systematic direction.)

**Pixel convention** — identical to the sidecar's ``/forecast`` endpoint:
``idx = CompositeGeo.lonlat_to_grid(lon, lat)`` on the native grid, then
``row = int(round(idx.row / downsample_factor))`` (same for col), then a
bounds check against the product-grid shape. Out-of-grid points record
``raw_prob = NaN``.

**Sample weights (Finding 3)** — the sampler draws ``n_wet`` events from
the ``N_wet`` classified wet hours and ``n_dry`` from the ``N_dry`` dry
hours (``n_wet = round(n_events * wet_bias)``, remainder dry, both
clamped to availability). Simple random sampling within each stratum
gives every hour in a stratum the same inclusion probability, so::

    inclusion_prob(wet hour) = n_wet / N_wet
    inclusion_prob(dry hour) = n_dry / N_dry

    sample_weight = 1 / inclusion_prob
                  = N_wet / n_wet   (event drawn from the wet stratum)
                  = N_dry / n_dry   (event drawn from the dry stratum)

With ``--wet-bias 0`` (uniform sampling) every event's inclusion
probability is identical and ``sample_weight = 1.0``. Weights are stored
UN-normalised — a weighted isotonic fit is invariant to a global weight
scale, and unnormalised weights stay consistent if runs are resumed or
appended. The fit (package B3) consumes them as relative weights.

**Parquet schema v2** — one row per ``(event_time, point, lead)``, zstd:

    event_time     str, ISO UTC
    point_id       str
    lat, lon       float64
    region         str
    lead_min       int32
    raw_prob       float32 (NaN when the forecast failed / out of grid)
    outcome        int8, nullable (null = verification frame missing)
    sample_weight  float64
    frame_age_min  float32 (this event's simulated frame age, minutes —
                   constant across the event's rows)
    error          str ("" when clean; per-event diagnostics)
    -- settings columns (B0 parity; identical on every row of a corpus) --
    ensemble_size, n_cascade_levels, downsample_factor, n_timesteps  int32
    threshold_mm_h, disc_radius_m, timestep_min                      float64
    detection_stat, scan_type, motion_method                         str
    leads_min_csv, frame_age_range_csv, settings_hash                str
    schema_version                                                   int32 (2)

``settings_hash`` is a stable hash over the sorted settings dict; the
builder itself refuses to append to an output whose hash differs, and
the fit refuses a mixed corpus the same way. ``scan_type``,
``timestep_min``, ``motion_method`` and ``frame_age_range`` are part of
the hashed dict, so any pre-fix corpus (mixed-type frames, 5-min
timestep, uncompleted motion field, zero frame age) hashes differently
and the fitter refuses it automatically — no manual audit needed.

Resumable: skips events already present in the output Parquet. Progress
JSON is written to ``--progress`` after every event for the companion
dashboard.

Example::

    python scripts/build_calibration_corpus.py \\
        --points src/dmi_nowcast_core/calibration_points_v2.json \\
        --days-back 125 --n-events 500 \\
        --wet-bias 0.15 --frame-age-range 12,18 \\
        --output reports/calibration_corpus.parquet \\
        --progress /tmp/calib_progress.json --workers 3

With ``--wet-bias > 0`` and no ``--wet-ref``, the five national
references above are used. To override::

    --wet-ref 57.2,9.6 --wet-ref 55.5,11.8      # repeatable, or
    --wet-ref "57.2,9.6;55.5,11.8"              # one flag, ';'-separated
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import multiprocessing as mp
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Union

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dmi_nowcast_core.fetch import (  # noqa: E402
    RadarFeature,
    download,
    list_in_window,
)
from dmi_nowcast_core.corpus import (  # noqa: E402
    SCAN_TYPE_DOPPLER,
    SCAN_TYPE_FULL_RANGE,
    ArchivedFrame,
    ArchiveIndex,
    archive_path_for,
)
from dmi_nowcast_core.geo import CompositeGeo  # noqa: E402
from dmi_nowcast_core.national import NationalProducts, national_products  # noqa: E402
from dmi_nowcast_core.parse import parse_composite  # noqa: E402
from dmi_nowcast_core.probabilistic import run_ensemble  # noqa: E402
from dmi_nowcast_core.sample import DiscStats, sample_disc  # noqa: E402
from dmi_nowcast_core.transform import dbz_to_rain_rate  # noqa: E402

_LOGGER = logging.getLogger("calib")

SCHEMA_VERSION = 2

# DMI publishes a composite every 5 min but alternates product types
# (fullRange at :x0, doppler at :x5) — a single scan type arrives every
# 10 min. Match-tolerance for finding a frame at a requested time.
FRAME_TOLERANCE_MIN = 4
# Single-scan-type frame spacing — and therefore the STEPS timestep,
# which follows the frame spacing. Not a tunable: it is a property of
# DMI's interleaved product (each type is a 10-min product). It joins
# the settings hash so pre-fix 5-min corpora are refused.
FRAME_SPACING_MIN = 10.0
# STEPS noise seed. Fixed (as at runtime, run_ensemble's default) so
# reruns of an event reproduce the same ensemble.
STEPS_SEED = 42
# Identifier of the motion field STEPS is driven with. Part of the settings
# hash, exactly like ``scan_type`` / ``timestep_min``: R5 completes the
# Farnebäck field away from the echo (dmi_nowcast_core.dense_flow.
# complete_flow) before it reaches STEPS, which changes every advected
# probability. Corpora built before that fix hash differently and — since
# they also lack the column entirely — the fitter refuses them structurally.
# Bump the suffix whenever the motion pipeline changes materially.
MOTION_METHOD = "farneback_complete_v1"
# Simulated frame age, minutes: [LO, HI] of the uniform draw per event.
# The live cycle finishes ~12-18 min after its newest frame's radar
# timestamp (fetch + STEPS + render), and the runtime shifts every lead
# by that age before picking an ensemble timestep — so the corpus must
# too, or the fitted curve corrects a different horizon than the one
# served (module docstring, "Frame-age convention").
DEFAULT_FRAME_AGE_RANGE = (12.0, 18.0)
# Sanity bound on --frame-age-range. A cycle older than an hour is a
# broken service, not a latency to calibrate for.
MAX_FRAME_AGE_MIN = 60.0
# Flush cadence: append (full rewrite, atomic rename) every N events.
FLUSH_EVERY_EVENTS = 10

#: Default wet/dry classification references — five points spread across
#: Denmark, so the wet stratum samples national weather rather than one
#: neighbourhood's. Bornholm is in because it is meteorologically isolated
#: (a Baltic shower day can leave the mainland dry). An hour is wet when
#: rain fell within ``search_km`` of ANY of them.
DEFAULT_WET_REFS: tuple[tuple[float, float], ...] = (
    (57.2, 9.6),    # NW Jutland (Thy / Hanstholm)
    (57.0, 10.3),   # NE Jutland (Vendsyssel)
    (55.5, 8.7),    # SW Jutland (Esbjerg)
    (55.5, 11.8),   # Zealand (Ringsted)
    (55.1, 14.9),   # Bornholm
)


# ---------------------------------------------------------------------------
# B0 — settings + hash
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusSettings:
    """Every STEPS / sampling setting that defines a corpus (B0 parity).

    Defaults are set at the CLI layer, not here — this object always
    represents an explicit, fully-resolved configuration.
    """

    ensemble_size: int
    n_cascade_levels: int
    downsample_factor: int
    threshold_mm_h: float
    disc_radius_m: float
    detection_stat: str
    leads_min: tuple[int, ...]
    scan_type: str = "fullRange"
    timestep_min: float = FRAME_SPACING_MIN
    motion_method: str = MOTION_METHOD
    #: [LO, HI] minutes of the per-event uniform frame-age draw. Part of
    #: the hashed settings, so a zero-age corpus can never be mixed with
    #: (or resumed as) a latency-simulating one.
    frame_age_range: tuple[float, float] = DEFAULT_FRAME_AGE_RANGE

    @property
    def n_timesteps(self) -> int:
        """Ensemble horizon: enough whole timesteps to span the longest
        EFFECTIVE lead — the longest nominal lead plus the largest frame
        age the sampler can draw, since the runtime reads lead ``L`` at
        timestep ``ceil((L + age) / timestep_min)``. 8 timesteps (80 min)
        at the defaults, against 6 for the old zero-age convention."""
        longest = max(self.leads_min) + max(self.frame_age_range)
        return max(1, math.ceil(longest / self.timestep_min))

    def to_dict(self) -> dict:
        """Canonical settings dict — the hash input and the Parquet columns."""
        return {
            "ensemble_size": int(self.ensemble_size),
            "n_cascade_levels": int(self.n_cascade_levels),
            "downsample_factor": int(self.downsample_factor),
            "threshold_mm_h": float(self.threshold_mm_h),
            "disc_radius_m": float(self.disc_radius_m),
            "detection_stat": str(self.detection_stat),
            "scan_type": str(self.scan_type),
            "motion_method": str(self.motion_method),
            "leads_min": [int(x) for x in self.leads_min],
            "timestep_min": float(self.timestep_min),
            "frame_age_range": [float(x) for x in self.frame_age_range],
            "n_timesteps": int(self.n_timesteps),
        }

    @property
    def settings_hash(self) -> str:
        """Stable hash over the sorted settings dict (16 hex chars)."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def settings_columns(self) -> dict:
        """The per-row Parquet settings columns.

        Parquet columns are scalars, so the two list-valued settings are
        flattened to CSV strings (``leads_min`` → ``leads_min_csv``,
        ``frame_age_range`` → ``frame_age_range_csv``). The hash is
        computed over the LIST form, so the flattening cannot affect it.
        """
        d = self.to_dict()
        leads = d.pop("leads_min")
        d["leads_min_csv"] = ",".join(str(x) for x in leads)
        age_range = d.pop("frame_age_range")
        d["frame_age_range_csv"] = ",".join(f"{float(x):g}" for x in age_range)
        d["settings_hash"] = self.settings_hash
        d["schema_version"] = SCHEMA_VERSION
        return d


def parse_leads(spec: str) -> tuple[int, ...]:
    """Parse ``"5,10,15"`` → ``(5, 10, 15)``; must be ascending positive ints."""
    try:
        leads = tuple(int(part.strip()) for part in spec.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"--leads must be comma-separated integers, got {spec!r}") from exc
    if not leads:
        raise ValueError("--leads must not be empty")
    if any(x <= 0 for x in leads) or list(leads) != sorted(set(leads)):
        raise ValueError(f"--leads must be strictly ascending positive minutes, got {spec!r}")
    return leads


def parse_frame_age_range(spec: str) -> tuple[float, float]:
    """Parse ``"12,18"`` → ``(12.0, 18.0)``: the uniform frame-age draw's
    bounds in minutes. Requires ``0 <= LO <= HI <= 60``; ``"0,0"`` is the
    legal degenerate case (the old zero-age convention)."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(
            f"--frame-age-range must be 'LO,HI' in minutes, got {spec!r}"
        )
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"--frame-age-range {spec!r} is not numeric") from exc
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"--frame-age-range {spec!r} must be finite")
    if not (0.0 <= lo <= hi <= MAX_FRAME_AGE_MIN):
        raise ValueError(
            f"--frame-age-range must satisfy 0 <= LO <= HI <= "
            f"{MAX_FRAME_AGE_MIN:g}, got {spec!r}"
        )
    return (lo, hi)


def event_frame_age_min(
    event_time: datetime, seed: int, frame_age_range: tuple[float, float]
) -> float:
    """The simulated frame age (minutes) for one event: ``Uniform(LO, HI)``.

    Deterministic in ``(seed, event_time)`` and NOT in the order events
    are processed, so a resumed, appended or reordered build redraws the
    identical age for an event that is already in the corpus — the same
    reproducibility guarantee ``STEPS_SEED`` gives the ensemble. A
    degenerate range returns its single value without consuming the RNG.
    """
    lo, hi = float(frame_age_range[0]), float(frame_age_range[1])
    if hi <= lo:
        return lo
    key = f"{int(seed)}|{_ts_str(event_time)}".encode("utf-8")
    stream = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return lo + (hi - lo) * random.Random(stream).random()


# ---------------------------------------------------------------------------
# C1 — scan-type interleave (fullRange :x0 / doppler :x5)
# ---------------------------------------------------------------------------

#: Minute-of-hour offset of each product's 10-min frame grid. DMI
#: interleaves the two products in one collection; composite filenames
#: carry no scan-type marker, so the filename minute IS the marker.
_SCAN_GRID_OFFSET_MIN = {SCAN_TYPE_FULL_RANGE: 0, SCAN_TYPE_DOPPLER: 5}


def scan_grid_offset_min(scan_type: str) -> int:
    """Minute-of-hour offset of ``scan_type``'s 10-min frame grid."""
    try:
        return _SCAN_GRID_OFFSET_MIN[scan_type]
    except KeyError:
        raise ValueError(
            f"unknown scan type {scan_type!r} "
            f"(expected one of {sorted(_SCAN_GRID_OFFSET_MIN)})"
        ) from None


#: Either kind of listed frame. An ``ArchivedFrame`` is already on disk
#: (``path``, no ``download_url``); a ``RadarFeature`` came from the DMI
#: items API. Everything downstream of a listing reads only
#: ``datetime_utc`` / ``scan_type`` / ``filename``, which both provide.
Frame = Union[RadarFeature, ArchivedFrame]


def feature_matches_scan_type(feature: Frame, scan_type: str) -> bool:
    """Does ``feature`` belong to ``scan_type``?

    API features carry ``scanType`` and are compared directly, as do
    :class:`~dmi_nowcast_core.corpus.ArchivedFrame` s (the archive index
    derives the type from the filename minute up front). A frame with an
    EMPTY scan-type field falls back to the minute-of-hour that encodes
    the interleave: ``minute % 10 == 0`` is fullRange, ``minute % 10 == 5``
    is doppler. An empty ``scan_type`` argument disables filtering (legacy
    mixed behaviour — never reachable from the CLI, which requires a
    concrete type).

    Note the equality test: an archived frame on neither product's grid
    is labelled ``"unknown"`` and matches nothing, so it can never be
    mistaken for fullRange.
    """
    if not scan_type:
        return True
    if feature.scan_type:
        return feature.scan_type == scan_type
    return feature.datetime_utc.minute % 10 == scan_grid_offset_min(scan_type)


def filter_scan_type(feats: list[Frame], scan_type: str) -> list[Frame]:
    """Keep only features of ``scan_type``.

    Belt-and-braces on top of the API-side ``scanType`` parameter and the
    archive index's own filter — and the ONLY filter for an unlabeled
    frame, where the filename minute is all we have.
    """
    return [f for f in feats if feature_matches_scan_type(f, scan_type)]


def _list_frames_in_window(
    start: datetime,
    end: datetime,
    *,
    limit: int,
    scan_type: str,
    archive: Optional[ArchiveIndex] = None,
) -> list[Frame]:
    """Frames in ``[start, end]`` of ``scan_type`` — archive first, API second.

    The single listing entry point for the whole builder (module
    docstring, "Archive-first listing"). The archive is unbounded and
    local; DMI's items API stops at 180 days and costs a request. So:
    return the archive's frames whenever it has ANY in the window, and
    only otherwise ask DMI (raising whatever the client raises, exactly as
    before, so callers keep their existing failure handling).

    Both branches end in :func:`filter_scan_type`; for the archive branch
    that is a no-op (the index already filtered by equality) kept for the
    symmetry of one filter in one place.
    """
    if archive is not None:
        frames = archive.list_in_window(start, end, scan_type=scan_type or None)
        if frames:
            return filter_scan_type(frames, scan_type)
    return filter_scan_type(
        list_in_window(start, end, limit=limit, scan_type=scan_type or None),
        scan_type,
    )


def snap_lead_min(lead_min: float, timestep_min: float) -> int:
    """Snap a lead onto the frame grid: UP to the next whole timestep,
    ``ceil(lead / timestep) * timestep``.

    Callers pass the EFFECTIVE lead — nominal lead + the event's frame
    age — because that is what the runtime resolves a timestep from
    (``national_products`` maps lead L → timestep
    ``ceil((L + frame_age) / timestep_min)``, valid at that many minutes
    after radar-frame time). Verification truth and forecast quantity
    therefore describe the same instant. On zero age and the 10-min grid
    this reduces to the old rule — nearest-grid rounding with ties toward
    the later frame for leads that are multiples of 5 (5 → 10, 15 → 20,
    20 → 20, …) — and to the identity on a 5-min grid.

    The epsilon mirrors ``national._steps_in_lead`` (and
    ``probabilistic.frame_age_corrected_leads``) so float fuzz at an
    exact timestep multiple can never split verification and forecast
    into different buckets.
    """
    return int(math.ceil(lead_min / timestep_min - 1e-9) * timestep_min)


# ---------------------------------------------------------------------------
# B2 — calibration points
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationPoint:
    id: str
    lat: float
    lon: float
    region: str


#: Points-file schema versions this builder understands. The point schema
#: is identical across them; v2 differs only in which fixed reference point
#: the set carries (see scripts/build_calibration_points.py).
SUPPORTED_POINTS_VERSIONS = (1, 2)


def load_points(path: Path) -> tuple[CalibrationPoint, ...]:
    """Load a ``calibration_points`` JSON file (schema version 1 or 2).

    Schema: ``{"version": 2, "points": [{"id", "lat", "lon", "region", ...}]}``.
    Extra keys per point (strata tags etc.) are tolerated and ignored.
    """
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("version") not in SUPPORTED_POINTS_VERSIONS:
        raise ValueError(
            f"{path}: expected a points file with version in "
            f"{list(SUPPORTED_POINTS_VERSIONS)}, "
            f"got version={data.get('version') if isinstance(data, dict) else type(data).__name__!r}"
        )
    raw_points = data.get("points")
    if not isinstance(raw_points, list) or not raw_points:
        raise ValueError(f"{path}: 'points' must be a non-empty list")
    points: list[CalibrationPoint] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw_points):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: points[{i}] is not an object")
        try:
            point = CalibrationPoint(
                id=str(entry["id"]),
                lat=float(entry["lat"]),
                lon=float(entry["lon"]),
                region=str(entry["region"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: points[{i}] needs id/lat/lon/region ({exc})"
            ) from exc
        if point.id in seen:
            raise ValueError(f"{path}: duplicate point id {point.id!r}")
        seen.add(point.id)
        points.append(point)
    return tuple(points)


def sample_points_from_products(
    products: NationalProducts,
    geo: CompositeGeo,
    points: tuple[CalibrationPoint, ...],
) -> dict[str, dict[int, float]]:
    """Sample every point from the national ``p_rain`` grids.

    Returns ``{point_id: {lead_min: raw_prob}}``. Pixel mapping is the
    sidecar ``/forecast`` convention exactly: native row/col from
    ``geo.lonlat_to_grid``, divided by the downsample factor, rounded to
    the nearest product pixel. Out-of-grid points (and NaN product
    pixels) yield NaN.
    """
    f = max(1, int(products.downsample_factor))
    h, w = products.eta_min.shape
    out: dict[str, dict[int, float]] = {}
    for point in points:
        idx = geo.lonlat_to_grid(point.lon, point.lat)
        row = int(round(idx.row / f))
        col = int(round(idx.col / f))
        if 0 <= row < h and 0 <= col < w:
            per_lead = {
                int(lead): float(products.p_rain[lead][row, col])
                for lead in products.leads_min
            }
        else:
            per_lead = {int(lead): float("nan") for lead in products.leads_min}
        out[point.id] = per_lead
    return out


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


def detection_stat_value(stats: DiscStats, detection_stat: str) -> float:
    """The configured disc statistic — mirrors the runtime raining-now check."""
    if detection_stat == "max":
        return stats.max_mm_h
    if detection_stat == "mean":
        return stats.mean_mm_h
    if detection_stat == "p90":
        return stats.p90_mm_h
    raise ValueError(f"unknown detection_stat {detection_stat!r}")


def outcome_from_stats(
    stats: DiscStats, detection_stat: str, threshold_mm_h: float
) -> Optional[int]:
    """0/1 rain outcome from disc stats; None when the disc had no valid data."""
    value = detection_stat_value(stats, detection_stat)
    if not np.isfinite(value):
        return None
    return int(value >= threshold_mm_h)


def build_event_rows(
    event_time_iso: str,
    points: tuple[CalibrationPoint, ...],
    leads_min: tuple[int, ...],
    raw_by_point: dict[str, dict[int, float]],
    outcomes_by_point: dict[str, dict[int, Optional[int]]],
    error: str = "",
    frame_age_min: float = 0.0,
) -> list[dict]:
    """Assemble the per-(point, lead) base rows for one event.

    ``frame_age_min`` is the event's simulated frame age and is stamped
    onto every row of the event (it is a per-event draw, and the forecast
    and verification instants both derive from it). ``sample_weight`` and
    the settings columns are attached by the parent process (they are
    per-run, not per-worker, concerns).
    """
    rows: list[dict] = []
    for point in points:
        raws = raw_by_point.get(point.id, {})
        outs = outcomes_by_point.get(point.id, {})
        for lead in leads_min:
            rows.append({
                "event_time": event_time_iso,
                "point_id": point.id,
                "lat": point.lat,
                "lon": point.lon,
                "region": point.region,
                "lead_min": int(lead),
                "raw_prob": float(raws.get(lead, float("nan"))),
                "outcome": outs.get(lead),
                "frame_age_min": float(frame_age_min),
                "error": error,
            })
    return rows


def error_event_rows(
    event_time_iso: str,
    points: tuple[CalibrationPoint, ...],
    leads_min: tuple[int, ...],
    error: str,
    frame_age_min: float = 0.0,
) -> list[dict]:
    """All-NaN / null rows for a failed event (kept so resume skips it)."""
    return build_event_rows(
        event_time_iso, points, leads_min, {}, {},
        error=error, frame_age_min=frame_age_min,
    )


# ---------------------------------------------------------------------------
# Event sampling + wet-bias weights (Finding 3)
# ---------------------------------------------------------------------------


def stratum_weights(
    *,
    n_wet_available: int,
    n_dry_available: int,
    n_wet_drawn: int,
    n_dry_drawn: int,
) -> tuple[float, float]:
    """Per-event inverse-inclusion-probability weights ``(wet_w, dry_w)``.

    Simple random sampling of ``n`` from ``N`` within a stratum makes each
    hour's inclusion probability ``n / N``, so the importance weight is
    ``N / n``. A stratum nothing was drawn from gets weight NaN (unused).
    """
    wet_w = (n_wet_available / n_wet_drawn) if n_wet_drawn > 0 else float("nan")
    dry_w = (n_dry_available / n_dry_drawn) if n_dry_drawn > 0 else float("nan")
    return wet_w, dry_w


def _candidate_hours(
    *,
    days_back: int,
    archive: Optional[ArchiveIndex] = None,
    scan_type: str = "",
) -> list[datetime]:
    """Every whole hour in the event window, ending 2 hours ago.

    ``days_back > 0`` — the last ``days_back`` days, as always.

    ``days_back == 0`` — "as far back as the archive goes": the window
    starts at the first whole hour at or after the oldest archived frame
    of ``scan_type``, so extending the window past DMI's 180-day listing
    horizon needs nothing but a deeper archive. Requires an
    :class:`ArchiveIndex` (i.e. ``--corpus-dir``), since there is no other
    way to know how far back "back" goes.

    The window is a sampling choice, not a measured quantity — it is NOT
    part of ``settings_hash`` (module docstring), so it is logged here
    instead.
    """
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    end = now - timedelta(hours=2)
    if days_back > 0:
        start = now - timedelta(days=days_back)
        source = f"--days-back {days_back}"
    else:
        if archive is None:
            raise ValueError(
                "--days-back 0 means 'the whole corpus archive' and needs "
                "--corpus-dir to say how far back that is"
            )
        oldest = archive.earliest(scan_type or None)
        if oldest is None:
            raise ValueError(
                f"--days-back 0: the archive at {archive.corpus_dir} holds no "
                f"{scan_type or 'radar'} frame to start the window at"
            )
        # Round UP to a whole hour: candidates are whole hours, and the
        # hour containing the oldest frame may have nothing at its :00.
        start = oldest.replace(minute=0, second=0, microsecond=0)
        if start < oldest:
            start += timedelta(hours=1)
        source = (
            f"--days-back 0 → oldest archived {scan_type or 'radar'} frame "
            f"{oldest.isoformat()}"
        )
    span = max(0, int((end - start).total_seconds() / 3600))
    _LOGGER.info(
        "event window: %s → %s (%d candidate hours; %s)",
        start.isoformat(), end.isoformat(), span, source,
    )
    return [start + timedelta(hours=h) for h in range(span)]


def _resolve_frame(
    feature: Frame, cache_dir: Path, corpus_dir: Optional[Path]
) -> Path:
    """Local path to ``feature``'s HDF5, preferring the persistent corpus.

    A frame that came out of the :class:`ArchiveIndex` already IS a local
    path: return it and never touch the network. That is the point of the
    archive-first listing — for an event outside DMI's 180-day window
    there is nothing to download.

    Otherwise, with ``corpus_dir`` set, look up the canonical
    ``composites/YYYY/MM`` slot first and reuse it on a hit; on a miss,
    download from DMI straight into that slot so the corpus grows and
    subsequent runs (and the live sidecar) hit locally — i.e. only
    genuinely new frames touch the network. Without a ``corpus_dir``, fall
    back to a flat download into ``cache_dir`` (standalone use, preserves
    the original behaviour).
    """
    if isinstance(feature, ArchivedFrame):
        return feature.path
    if corpus_dir is not None:
        dest = archive_path_for(corpus_dir, feature.filename)
        if dest.exists():
            return dest
        return download(feature, dest.parent)
    return download(feature, cache_dir)


def parse_wet_refs(spec: str) -> tuple[tuple[float, float], ...]:
    """Parse ``"lat,lon"`` or ``"lat,lon;lat,lon;..."`` → ``((lat, lon), ...)``.

    Whitespace is tolerated; empty segments are ignored. Raises
    ``ValueError`` on anything that isn't a pair of finite degrees inside
    the WGS84 range.
    """
    refs: list[tuple[float, float]] = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        bits = [b.strip() for b in part.split(",")]
        if len(bits) != 2:
            raise ValueError(
                f"wet reference must be 'lat,lon' (or 'lat,lon;lat,lon'), got {part!r}"
            )
        try:
            lat, lon = float(bits[0]), float(bits[1])
        except ValueError as exc:
            raise ValueError(f"wet reference {part!r} is not numeric") from exc
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise ValueError(f"wet reference {part!r} is outside lat/lon range")
        refs.append((lat, lon))
    if not refs:
        raise ValueError(f"no wet reference parsed from {spec!r}")
    return tuple(refs)


def resolve_wet_refs(
    wet_ref: Optional[list[str]],
    wet_ref_lat: Optional[float] = None,
    wet_ref_lon: Optional[float] = None,
) -> tuple[tuple[float, float], ...]:
    """CLI → reference set, in precedence order.

    ``--wet-ref`` (repeatable, each value possibly ';'-separated) wins;
    the deprecated ``--wet-ref-lat/--wet-ref-lon`` pair is accepted as a
    single reference; otherwise :data:`DEFAULT_WET_REFS`. Duplicates are
    collapsed while keeping first-seen order (the cache key sorts anyway).
    """
    refs: list[tuple[float, float]] = []
    for spec in wet_ref or []:
        refs.extend(parse_wet_refs(spec))
    if (wet_ref_lat is None) != (wet_ref_lon is None):
        raise ValueError("--wet-ref-lat and --wet-ref-lon must be given together")
    if wet_ref_lat is not None and wet_ref_lon is not None:
        refs.extend(parse_wet_refs(f"{wet_ref_lat},{wet_ref_lon}"))
    if not refs:
        return DEFAULT_WET_REFS
    seen: set[tuple[float, float]] = set()
    unique: list[tuple[float, float]] = []
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            unique.append(ref)
    return tuple(unique)


def wet_refs_key(refs: tuple[tuple[float, float], ...]) -> str:
    """Stable 8-hex key for a reference SET (order-independent).

    The wet/dry index cache is keyed on this: classifying against a
    different set of references produces different wet/dry counts, so a
    cache built for another set must never be reused (same reasoning as
    the scan-type re-key).
    """
    canonical = ";".join(
        f"{lat:.4f},{lon:.4f}" for lat, lon in sorted(set(refs))
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def wet_dry_index_name(
    scan_type: str, refs: tuple[tuple[float, float], ...]
) -> str:
    """Cache filename for the wet/dry index of ``(scan_type, refs)``."""
    stem = f"wet_dry_index_{scan_type}" if scan_type else "wet_dry_index"
    return f"{stem}_{wet_refs_key(refs)}.json"


def _classify_hour_wet(
    hour: datetime,
    refs: tuple[tuple[float, float], ...],
    cache_dir: Path,
    corpus_dir: Optional[Path] = None,
    threshold_mm_h: float = 0.5, search_km: float = 60.0,
    scan_type: str = "",
    archive: Optional[ArchiveIndex] = None,
) -> Optional[bool]:
    """Wet/dry classification for a single hour: any rain pixel ≥
    ``threshold_mm_h`` within ``search_km`` of ANY reference in ``refs``.

    Fetches the nearest ``scan_type`` composite to ``hour:00``, parses
    it ONCE, and tests each reference's ±``search_km`` window against the
    threshold — short-circuiting on the first wet reference. Returns
    ``True``/``False``/``None`` (network or parse failure). ``None`` is
    treated as "unknown" by the caller and contributes to neither bucket.

    The per-reference thresholds are UNCHANGED from the single-point
    builder (0.5 mm/h within 60 km); what changed is that the union of
    several spread references decides, so the wet stratum samples
    national weather instead of one neighbourhood's. Frames are filtered
    to ``scan_type``; both the type and the reference set re-key the
    cache (:func:`wet_dry_index_name`), so an index built under other
    settings is never reused. The index defines *strata only* — the
    weights above make the fit correct for whatever stratification was
    actually used.

    ``archive`` is what makes an hour classifiable at all beyond DMI's
    180-day listing horizon: with an :class:`ArchiveIndex` the ±10 min
    window is listed from disk and resolved without a request, so an hour
    whose frames are archived is never returned as ``None`` merely
    because DMI has forgotten it.
    """
    win_start = hour - timedelta(minutes=10)
    win_end = hour + timedelta(minutes=10)
    try:
        feats = _list_frames_in_window(
            win_start, win_end, limit=6, scan_type=scan_type, archive=archive,
        )
    except Exception:  # noqa: BLE001
        return None
    if not feats:
        return None
    # Closest to the top-of-the-hour.
    feature = min(feats, key=lambda f: abs(f.datetime_utc - hour))
    try:
        path = _resolve_frame(feature, cache_dir, corpus_dir)
        composite = parse_composite(path)
        geo = CompositeGeo(composite)
        rain = dbz_to_rain_rate(
            composite.reflectivity_dbz, zr_a=composite.zr_a, zr_b=composite.zr_b,
        )
        search_px = int(round(search_km * 1000.0 / composite.xscale_m))
        for lat, lon in refs:
            # Crop a ±search_km square around this reference; wet if ANY
            # reference sees rain, so the first hit ends the search.
            idx = geo.lonlat_to_grid(lon, lat)
            r_, c_ = int(round(idx.row)), int(round(idx.col))
            rs = slice(max(0, r_ - search_px), min(rain.shape[0], r_ + search_px + 1))
            cs = slice(max(0, c_ - search_px), min(rain.shape[1], c_ + search_px + 1))
            crop = rain[rs, cs]
            finite = crop[np.isfinite(crop)]
            if finite.size == 0:
                continue  # this reference has no usable data — try the next
            if bool(finite.max() >= threshold_mm_h):
                return True
        return False
    except Exception:  # noqa: BLE001
        return None


def _build_or_load_wet_dry_index(
    candidates: list[datetime],
    refs: tuple[tuple[float, float], ...],
    cache_dir: Path,
    workers: int = 6, corpus_dir: Optional[Path] = None,
    scan_type: str = "",
    archive: Optional[ArchiveIndex] = None,
) -> dict[str, bool]:
    """Build a {ISO timestamp: is_wet} index for every candidate hour.

    Cached at ``cache_dir/wet_dry_index_<scanType>_<refsHash>.json`` so
    repeat runs reuse the classification work. The cache key incorporates
    the scan type AND the reference set deliberately: wet/dry counts
    change under type filtering and under a different set of references,
    so a stale mixed-type or single-reference index must not — and cannot
    — be reused. Hours newly added since the last run are fetched +
    classified, all parallelised across ``workers`` threads. Uses threads
    (not processes) because the cost is network-bound; httpx already
    releases the GIL on socket I/O. With an ``archive`` most hours cost
    no request at all — the frame is listed and read from disk — which is
    what lets the index cover hours older than DMI's 180-day listing.
    """
    import concurrent.futures

    index_path = cache_dir / wet_dry_index_name(scan_type, refs)
    existing: dict[str, bool] = {}
    if index_path.exists():
        try:
            existing = {k: bool(v) for k, v in json.loads(index_path.read_text()).items()}
        except Exception:  # noqa: BLE001
            existing = {}

    missing = [h for h in candidates if _ts_str(h) not in existing]
    if not missing:
        _LOGGER.info("wet/dry index: all %d candidates already classified", len(candidates))
        return existing

    _LOGGER.info(
        "wet/dry index: classifying %d new hours (%d cached) with %d workers",
        len(missing), len(existing), workers,
    )

    def _one(h: datetime) -> tuple[str, Optional[bool]]:
        return _ts_str(h), _classify_hour_wet(
            h, refs, cache_dir, corpus_dir, scan_type=scan_type, archive=archive,
        )

    new: dict[str, bool] = {}
    n_done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for key, val in ex.map(_one, missing):
            n_done += 1
            if val is not None:
                new[key] = val
            if n_done % 100 == 0:
                _LOGGER.info("  classified %d/%d", n_done, len(missing))

    existing.update(new)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(existing, indent=0))
    _LOGGER.info(
        "wet/dry index: %d total (%d wet, %d dry)",
        len(existing), sum(existing.values()), sum(1 for v in existing.values() if not v),
    )
    return existing


def _sample_event_times(
    *,
    days_back: int,
    n_events: int,
    seed: int,
    wet_bias: float = 0.0,
    wet_refs: tuple[tuple[float, float], ...] = DEFAULT_WET_REFS,
    cache_dir: Optional[Path] = None,
    classify_workers: int = 6,
    corpus_dir: Optional[Path] = None,
    scan_type: str = "",
    archive: Optional[ArchiveIndex] = None,
) -> list[tuple[datetime, float]]:
    """Pick ``n_events`` UTC datetimes with their sample weights.

    Returns ``[(event_time, sample_weight), ...]`` sorted by time. The
    weight is the inverse inclusion probability of the event under the
    sampler's own design (module docstring; :func:`stratum_weights`).

    Event anchors land on ``scan_type``'s frame grid: candidate hours
    are whole hours — minute :00, the fullRange grid by construction —
    and a doppler corpus shifts each anchor to hh:05. The wet/dry index
    stays keyed on the whole hour either way (the shift is applied after
    stratification).

    ``wet_bias = 0`` (default): uniform random — climatologically
    representative, ~95% of events will be dry hours; every weight is 1.

    ``wet_bias > 0``: target this fraction of events from "wet" hours
    (rain ≥ 0.5 mm/h within 60 km of ANY point in ``wet_refs``, judged on
    ``scan_type`` frames). E.g. ``wet_bias=0.5`` samples 50% wet / 50%
    dry hours. Necessary for calibration corpora — dry-only hours all map
    raw=0 → cal=0 and would starve the fit — and corrected for at fit
    time via the weights. ``wet_refs`` defaults to the five spread
    national references (:data:`DEFAULT_WET_REFS`).

    ``archive`` is threaded down to the hour classification (so hours
    older than DMI's 180-day listing can be classified from disk) and to
    :func:`_candidate_hours`, which needs it to resolve ``days_back = 0``
    into "the whole archive".
    """
    rng = random.Random(seed)
    candidates = _candidate_hours(
        days_back=days_back, archive=archive, scan_type=scan_type,
    )
    offset = (
        timedelta(minutes=scan_grid_offset_min(scan_type))
        if scan_type else timedelta(0)
    )
    if wet_bias <= 0.0:
        return sorted((t + offset, 1.0) for t in rng.sample(candidates, n_events))

    if cache_dir is None:
        raise ValueError("wet_bias > 0 requires cache_dir to store the wet/dry index")
    if not wet_refs:
        raise ValueError("wet_bias > 0 requires at least one wet reference point")
    _LOGGER.info(
        "wet/dry references (%d): %s  [cache key %s]",
        len(wet_refs),
        ", ".join(f"{lat:.4g},{lon:.4g}" for lat, lon in wet_refs),
        wet_refs_key(wet_refs),
    )
    index = _build_or_load_wet_dry_index(
        candidates, wet_refs, cache_dir=cache_dir, workers=classify_workers,
        corpus_dir=corpus_dir, scan_type=scan_type, archive=archive,
    )
    wet = [h for h in candidates if index.get(_ts_str(h)) is True]
    dry = [h for h in candidates if index.get(_ts_str(h)) is False]
    n_wet = int(round(n_events * wet_bias))
    n_dry = n_events - n_wet
    # Clamp if we don't have enough wet or dry hours.
    n_wet = min(n_wet, len(wet))
    n_dry = min(n_dry, len(dry))
    _LOGGER.info(
        "stratified sample: %d wet (of %d available) + %d dry (of %d available)",
        n_wet, len(wet), n_dry, len(dry),
    )
    wet_w, dry_w = stratum_weights(
        n_wet_available=len(wet), n_dry_available=len(dry),
        n_wet_drawn=n_wet, n_dry_drawn=n_dry,
    )
    chosen = [(t + offset, wet_w) for t in rng.sample(wet, n_wet)]
    chosen += [(t + offset, dry_w) for t in rng.sample(dry, n_dry)]
    return sorted(chosen)


def _ts_str(t: datetime) -> str:
    return t.replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Per-event worker (spawned process; one event end-to-end, all points)
# ---------------------------------------------------------------------------


@dataclass
class EventResult:
    """One sampled event's per-(point, lead) base rows."""

    event_time: str  # ISO UTC
    rows: list[dict] = field(default_factory=list)
    error: Optional[str] = None


def _find_nearest_feature(
    feats: list[Frame], target: datetime, tol_min: int = FRAME_TOLERANCE_MIN
) -> Optional[Frame]:
    """Closest feature to ``target`` within ``tol_min`` minutes."""
    best = None
    best_dt = timedelta(minutes=tol_min + 1)
    for f in feats:
        d = abs(f.datetime_utc - target)
        if d < best_dt:
            best, best_dt = f, d
    return best


def _gather_event_frames(
    event_time: datetime,
    settings: CorpusSettings,
    frame_age_min: float = 0.0,
    archive: Optional[ArchiveIndex] = None,
) -> tuple[list[Frame], dict[int, Frame]]:
    """Return (input_features[3], {lead: verification_feature}).

    Lists ONE window covering every frame the event needs, archive first
    and DMI second (:func:`_list_frames_in_window`), filtered to
    ``settings.scan_type`` — the archive index by the filename minute, the
    API server-side via the ``scanType`` parameter and client-side via
    :func:`filter_scan_type` — so a mixed listing can never leak a doppler
    frame into a fullRange corpus (or vice versa).

    Inputs are spaced at the frame cadence (``timestep_min``): T-20,
    T-10, T on the 10-min grid. Verification targets are the snapped
    EFFECTIVE leads — ``snap_lead_min(lead + frame_age_min, timestep)``,
    the module-docstring snapping rule — so the fetch window stretches
    with the simulated age (out to T+80 at the defaults, from T+60 at
    zero age); a lead whose snapped target has no frame within
    ``FRAME_TOLERANCE_MIN`` is simply absent from the truth dict, so its
    outcome stays null.

    Every frame this returns is resolved by :func:`_resolve_frame`, which
    short-circuits on an archived frame — so an event whose window the
    archive covers runs with zero network I/O, whatever its age.
    """
    step = int(round(settings.timestep_min))
    snapped = {
        lead: snap_lead_min(lead + frame_age_min, settings.timestep_min)
        for lead in settings.leads_min
    }
    win_start = event_time - timedelta(minutes=2 * step + FRAME_TOLERANCE_MIN + 1)
    win_end = event_time + timedelta(
        minutes=max(snapped.values()) + FRAME_TOLERANCE_MIN + 1
    )
    feats = _list_frames_in_window(
        win_start, win_end, limit=50,
        scan_type=settings.scan_type, archive=archive,
    )
    source = "archive listing" if feats and isinstance(feats[0], ArchivedFrame) else "DMI listing"

    inputs: list[Frame] = []
    for off in (-2 * step, -step, 0):
        f = _find_nearest_feature(feats, event_time + timedelta(minutes=off))
        if f is None:
            raise RuntimeError(
                f"no {settings.scan_type or 'radar'} input frame within "
                f"{FRAME_TOLERANCE_MIN} min of "
                f"{(event_time + timedelta(minutes=off)).isoformat()} "
                f"({source}, {len(feats)} frame(s) in window)"
            )
        inputs.append(f)

    truth: dict[int, Frame] = {}
    for lead in settings.leads_min:
        f = _find_nearest_feature(
            feats, event_time + timedelta(minutes=snapped[lead])
        )
        if f is not None:  # missing verification just leaves the lead's outcome null
            truth[lead] = f
    return inputs, truth


def _score_outcomes(
    truth: dict[int, Frame],
    points: tuple[CalibrationPoint, ...],
    settings: CorpusSettings,
    cache_dir: Path,
    corpus_dir: Optional[Path],
) -> dict[str, dict[int, Optional[int]]]:
    """Outcomes per (point, lead): parse each verification frame ONCE, then
    sample every point's disc from it. Missing frame / parse failure / empty
    disc → None (null in Parquet, filtered at fit time)."""
    outcomes: dict[str, dict[int, Optional[int]]] = {
        p.id: {int(lead): None for lead in settings.leads_min} for p in points
    }
    for lead in settings.leads_min:
        feature = truth.get(lead)
        if feature is None:
            continue
        try:
            path = _resolve_frame(feature, cache_dir, corpus_dir)
            composite = parse_composite(path)
            geo = CompositeGeo(composite)
            rain = dbz_to_rain_rate(
                composite.reflectivity_dbz, zr_a=composite.zr_a, zr_b=composite.zr_b
            )
        except Exception:  # noqa: BLE001 — frame-level failure nulls this lead
            continue
        for point in points:
            try:
                stats = sample_disc(
                    rain, geo, point.lon, point.lat, radius_m=settings.disc_radius_m
                )
                outcomes[point.id][int(lead)] = outcome_from_stats(
                    stats, settings.detection_stat, settings.threshold_mm_h
                )
            except Exception:  # noqa: BLE001 — point-level failure stays None
                continue
    return outcomes


#: Per-process :class:`ArchiveIndex` cache, keyed on the corpus dir.
#: See :func:`_worker_archive` for why the index is rebuilt in each worker
#: rather than shipped from the parent.
_WORKER_ARCHIVES: dict[str, ArchiveIndex] = {}


def _worker_archive(corpus_dir: Optional[Path]) -> Optional[ArchiveIndex]:
    """The archive index for this process, built at most once.

    Deliberately NOT passed in the worker args. ``imap_unordered(...,
    chunksize=1)`` pickles the args tuple once PER EVENT, so shipping the
    index would move its whole size per event: measured at 80k frames
    (~9 months of both products) it pickles to 8.1 MB, i.e. ~32 GB through
    the pipe over a 4000-event run. A spawned worker instead handles
    hundreds of events, so building the index once per process — 0.5 s of
    pure directory reads for those same 80k frames — is a rounding error
    against a single STEPS run.

    The cache is keyed on the corpus dir so a process is correct even if
    it were ever handed two different archives.
    """
    if corpus_dir is None:
        return None
    key = str(corpus_dir)
    index = _WORKER_ARCHIVES.get(key)
    if index is None:
        started = time.time()
        index = ArchiveIndex(corpus_dir)
        _WORKER_ARCHIVES[key] = index
        _LOGGER.info(
            "worker archive index: %d frames (%d %s) from %s in %.2fs",
            len(index), index.count(SCAN_TYPE_FULL_RANGE), SCAN_TYPE_FULL_RANGE,
            key, time.time() - started,
        )
    return index


def _process_event(
    args: tuple[
        datetime, Path, Optional[Path], tuple[CalibrationPoint, ...],
        CorpusSettings, float,
    ]
) -> EventResult:
    """One worker iteration: fetch + STEPS + national products + all points.

    The last argument is the event's simulated frame age in minutes,
    drawn by the parent (:func:`event_frame_age_min`) so the whole run
    shares one source of truth for it. It reaches the forecast
    (``national_products``), the verification instants
    (:func:`_gather_event_frames`) and the rows themselves.

    The archive index is NOT an argument — it is rebuilt once per worker
    process from ``corpus_dir`` (:func:`_worker_archive`).

    Returns an EventResult; never raises (the caller stamps any error
    onto the result for diagnostics in the dashboard).
    """
    event_time, cache_dir, corpus_dir, points, settings, frame_age_min = args
    event_iso = _ts_str(event_time)
    try:
        archive = _worker_archive(corpus_dir)
        inputs, truth = _gather_event_frames(
            event_time, settings, frame_age_min, archive=archive,
        )
        # Inputs: download + parse + dBZ.
        composites = []
        for f in inputs:
            p = _resolve_frame(f, cache_dir, corpus_dir)
            composites.append(parse_composite(p))
        composite_now = composites[-1]
        geo = CompositeGeo(composite_now)
        dbz_frames = [c.reflectivity_dbz for c in composites]

        # Best-effort motion: skimage / opencv from the integration's own
        # dense_flow module. (If unavailable, fall back to FFT phase
        # correlation.)
        from dmi_nowcast_core.dense_flow import (
            complete_flow,
            dense_flow,
            DenseFlowUnavailable,
        )
        from dmi_nowcast_core.motion import phase_correlation_shift

        rain_prev = dbz_to_rain_rate(composites[-2].reflectivity_dbz)
        rain_now = dbz_to_rain_rate(composite_now.reflectivity_dbz)
        try:
            vy, vx = dense_flow(
                composites[-2].reflectivity_dbz, composite_now.reflectivity_dbz
            )
        except DenseFlowUnavailable:
            dy, dx = phase_correlation_shift(rain_prev, rain_now)
            vy = np.full(rain_now.shape, dy, dtype=np.float32)
            vx = np.full(rain_now.shape, dx, dtype=np.float32)
        # Motion completion, in the same place the sidecar does it (before
        # the clip, so STEPS sees the completed field) — corpus/runtime
        # parity is the whole point of the settings hash, and
        # ``motion_method`` in that hash asserts exactly this call.
        vy, vx = complete_flow(
            vy, vx, rain_now,
            pixel_km=float(composite_now.xscale_m) / 1000.0,
            support_threshold_mm_h=settings.threshold_mm_h,
        )
        # Per-pixel clip (NOT zero-everything) — the grid-wide mean-abs
        # check would trip on every event because skimage's dense flow
        # fills dry pixels with noisy 50-100 px/frame extrapolations,
        # making the whole field get zeroed and reducing STEPS to a
        # persistence forecaster. Mirrors the fix in coordinator.py.
        MAX_PX = 30.0
        vy = np.nan_to_num(vy, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        vx = np.nan_to_num(vx, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        vy = np.clip(vy, -MAX_PX, MAX_PX).astype(np.float32)
        vx = np.clip(vx, -MAX_PX, MAX_PX).astype(np.float32)

        forecast = run_ensemble(
            dbz_frames, vy, vx,
            zr_a=composite_now.zr_a, zr_b=composite_now.zr_b,
            n_ens_members=settings.ensemble_size,
            n_timesteps=settings.n_timesteps,
            timestep_min=settings.timestep_min,
            n_cascade_levels=settings.n_cascade_levels,
            threshold_mm_h=settings.threshold_mm_h,
            downsample_factor=settings.downsample_factor,
            pixel_scale_m=composite_now.xscale_m,
            seed=STEPS_SEED,
        )
        # The event's simulated frame age — the SAME value that moved the
        # verification instants above, so the probability read out here and
        # the truth it is scored against describe one instant (module
        # docstring, "Frame-age convention").
        products = national_products(
            forecast,
            leads_min=settings.leads_min,
            threshold_mm_h=settings.threshold_mm_h,
            timestep_min=settings.timestep_min,
            frame_age_min=frame_age_min,
            downsample_factor=settings.downsample_factor,
        )
        raw_by_point = sample_points_from_products(products, geo, points)
        outcomes_by_point = _score_outcomes(
            truth, points, settings, cache_dir, corpus_dir
        )
        rows = build_event_rows(
            event_iso, points, settings.leads_min, raw_by_point, outcomes_by_point,
            frame_age_min=frame_age_min,
        )
        return EventResult(event_time=event_iso, rows=rows)
    except Exception as exc:  # noqa: BLE001
        return EventResult(
            event_time=event_iso,
            rows=error_event_rows(
                event_iso, points, settings.leads_min, f"{type(exc).__name__}: {exc}",
                frame_age_min=frame_age_min,
            ),
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------


def _parquet_schema():
    """Explicit Arrow schema — makes ``outcome`` a NULLABLE int8."""
    import pyarrow as pa

    return pa.schema([
        ("event_time", pa.string()),
        ("point_id", pa.string()),
        ("lat", pa.float64()),
        ("lon", pa.float64()),
        ("region", pa.string()),
        ("lead_min", pa.int32()),
        ("raw_prob", pa.float32()),
        ("outcome", pa.int8()),
        ("sample_weight", pa.float64()),
        # Per-EVENT (constant across the event's rows): the simulated
        # frame age the forecast and its verification instant both used.
        ("frame_age_min", pa.float32()),
        ("error", pa.string()),
        ("ensemble_size", pa.int32()),
        ("n_cascade_levels", pa.int32()),
        ("downsample_factor", pa.int32()),
        ("threshold_mm_h", pa.float64()),
        ("disc_radius_m", pa.float64()),
        ("detection_stat", pa.string()),
        ("scan_type", pa.string()),
        ("motion_method", pa.string()),
        ("timestep_min", pa.float64()),
        ("n_timesteps", pa.int32()),
        ("leads_min_csv", pa.string()),
        ("frame_age_range_csv", pa.string()),
        ("settings_hash", pa.string()),
        ("schema_version", pa.int32()),
    ])


def check_existing_corpus(parquet_path: Path, settings_hash: str) -> set[str]:
    """Validate a pre-existing output against the current settings.

    Returns the set of event_times already present (for resume). Raises
    ``ValueError`` when the file is a legacy v1 corpus (no settings
    columns) or was built with different settings — appending would
    create exactly the mixed corpus the fit must refuse.
    """
    if not parquet_path.exists():
        return set()
    import pyarrow.parquet as pq

    schema = pq.read_schema(parquet_path)
    if "settings_hash" not in schema.names:
        raise ValueError(
            f"{parquet_path} is a legacy (v1) corpus without settings columns — "
            "pick a fresh --output path instead of appending to it"
        )
    tbl = pq.read_table(parquet_path, columns=["event_time", "settings_hash"])
    hashes = set(tbl.column("settings_hash").to_pylist())
    if hashes and hashes != {settings_hash}:
        raise ValueError(
            f"{parquet_path} was built with settings hash(es) "
            f"{sorted(hashes)} but the current settings hash is "
            f"{settings_hash} — refusing to build a mixed corpus. Use a "
            "fresh --output path, or rerun with the original settings."
        )
    return set(tbl.column("event_time").to_pylist())


def _append_rows(parquet_path: Path, rows: list[dict]) -> None:
    """Append rows to Parquet, atomic on rename."""
    if not rows:
        return
    import pyarrow as pa
    import pyarrow.parquet as pq

    new_table = pa.Table.from_pylist(rows, schema=_parquet_schema())
    if parquet_path.exists():
        existing = pq.read_table(parquet_path)
        combined = pa.concat_tables([existing, new_table])
    else:
        combined = new_table
    tmp = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(combined, tmp, compression="zstd")
    tmp.replace(parquet_path)


def _write_progress(progress_path: Path, payload: dict) -> None:
    tmp = progress_path.with_suffix(progress_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, default=str))
    tmp.replace(progress_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--points", type=Path, required=True,
        help="Calibration points JSON (version 1 or 2: {'version': 2, "
             "'points': [{'id', 'lat', 'lon', 'region', ...}]}). One STEPS "
             "run per event serves every point.",
    )
    ap.add_argument(
        "--days-back", type=int, default=125, metavar="DAYS",
        help="How far back events are sampled from, ending 2 h ago. "
             "0 means 'as far back as the corpus archive goes' — the "
             "window starts at the oldest archived frame of --scan-type "
             "and needs --corpus-dir. Use it to grow the calibration "
             "window past DMI's 180-day listing horizon, which no "
             "positive value can reach. The window is a sampling choice "
             "(like --wet-ref) and is deliberately NOT in the settings "
             "hash, so a corpus can be extended backwards across runs; "
             "the actual window is logged and written to --progress.",
    )
    ap.add_argument("--n-events", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    # --- STEPS / sampling settings (B0: mirror the VM runtime config;
    # the ops script feeds them from the live sidecar config). ---
    ap.add_argument(
        "--scan-type", choices=("fullRange", "doppler"), default="fullRange",
        help="DMI composite product to build the corpus from. The "
             "collection interleaves fullRange (:x0 minutes) and doppler "
             "(:x5); the runtime is fullRange-only, so the corpus defaults "
             "to match (Phase B addendum 2026-08-29). Filters every frame "
             "listing/download: inputs, verification, and the wet/dry "
             "index. Joins the settings hash, so corpora of different "
             "types never mix.",
    )
    ap.add_argument(
        "--ensemble-size", type=int, default=16,
        help="STEPS ensemble members (runtime: forecast.steps.ensemble_size).",
    )
    ap.add_argument(
        "--threshold-mm-h", type=float, default=0.5,
        help="Detection threshold for both the ensemble exceedance and the "
             "verification outcome (runtime: forecast.rain_threshold_mm_h).",
    )
    ap.add_argument(
        "--leads", type=str, default="5,10,15,20,25,30,45,60",
        help="Comma-separated forecast leads in minutes "
             "(runtime: forecast.leads_min).",
    )
    ap.add_argument(
        "--downsample-factor", type=int, default=4,
        help="Grid downsample before STEPS (runtime: "
             "forecast.steps.downsample_factor).",
    )
    ap.add_argument(
        "--n-cascade-levels", type=int, default=6,
        help="STEPS cascade levels (runtime: forecast.steps.n_cascade_levels).",
    )
    ap.add_argument(
        "--disc-radius-m", type=float, default=1000.0,
        help="Verification disc radius around each point (runtime: "
             "home.radius_km).",
    )
    ap.add_argument(
        "--detection-stat", choices=("max", "p90", "mean"), default="p90",
        help="Disc statistic tested against the threshold for outcomes "
             "(runtime: forecast.detection_stat).",
    )
    ap.add_argument(
        "--frame-age-range", type=str,
        default="{:g},{:g}".format(*DEFAULT_FRAME_AGE_RANGE),
        metavar="LO,HI",
        help="Simulated live frame age in minutes: each event draws "
             "Uniform(LO, HI) from an RNG seeded on (--seed, event time). "
             "The runtime computes 12-18 min after its newest frame's "
             "radar timestamp and shifts every lead by that age before "
             "picking an ensemble timestep, so the corpus must too — "
             "otherwise the fitted curve corrects a different horizon "
             "than the one served. '0,0' reproduces the old zero-age "
             "convention (under a different settings hash). Joins the "
             "settings hash and grows n_timesteps to cover max(lead) + HI.",
    )
    # --- Wet-bias sampling ---
    ap.add_argument(
        "--wet-bias", type=float, default=0.0,
        help="Stratified sampling: fraction of events drawn from 'wet' hours "
             "(rain ≥ 0.5 mm/h within 60 km of ANY --wet-ref point). 0.0 = "
             "uniform random. Corrected at fit time via the sample_weight "
             "column (inverse inclusion probability).",
    )
    ap.add_argument(
        "--wet-ref", action="append", default=None, metavar="LAT,LON",
        help="Wet/dry classification reference point. REPEATABLE, and one "
             "flag may carry several ';'-separated pairs "
             "(--wet-ref '57.2,9.6;55.5,11.8'). An hour is wet when rain "
             "fell within 60 km of ANY reference. Default: the five spread "
             "national references "
             + "; ".join(f"{la},{lo}" for la, lo in DEFAULT_WET_REFS)
             + ". The set re-keys the wet/dry index cache, so changing it "
               "reclassifies rather than reusing another set's index.",
    )
    ap.add_argument(
        "--wet-ref-lat", type=float, default=None,
        help="Deprecated single-reference form: latitude. Use --wet-ref.",
    )
    ap.add_argument(
        "--wet-ref-lon", type=float, default=None,
        help="Deprecated single-reference form: longitude. Use --wet-ref.",
    )
    ap.add_argument(
        "--cache-dir", type=Path, default=Path("radar_archive"),
        help="Flat fallback store for downloaded radar HDF5 files when "
             "--corpus-dir is not given. Also holds the "
             "wet_dry_index_<scanType>_<refsHash>.json cache. Do NOT delete "
             "— frames older than 180 days have aged out of the DMI archive "
             "and cannot be re-fetched.",
    )
    ap.add_argument(
        "--corpus-dir", type=Path, default=None,
        help="Persistent corpus archive (composites/YYYY/MM layout). When "
             "set, frames are resolved here first and only genuine gaps are "
             "downloaded from DMI — straight into the corpus, so it keeps "
             "growing. Use this to reuse the backfilled archive instead of "
             "re-fetching every frame from DMI.",
    )
    ap.add_argument("--output", type=Path, default=Path("reports/calibration_corpus.parquet"))
    ap.add_argument("--progress", type=Path, default=Path("/tmp/calib_progress.json"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        leads = parse_leads(args.leads)
    except ValueError as exc:
        ap.error(str(exc))
    try:
        wet_refs = resolve_wet_refs(
            args.wet_ref, args.wet_ref_lat, args.wet_ref_lon
        )
    except ValueError as exc:
        ap.error(str(exc))
    try:
        frame_age_range = parse_frame_age_range(args.frame_age_range)
    except ValueError as exc:
        ap.error(str(exc))

    settings = CorpusSettings(
        ensemble_size=args.ensemble_size,
        n_cascade_levels=args.n_cascade_levels,
        downsample_factor=args.downsample_factor,
        threshold_mm_h=args.threshold_mm_h,
        disc_radius_m=args.disc_radius_m,
        detection_stat=args.detection_stat,
        leads_min=leads,
        scan_type=args.scan_type,
        # The STEPS timestep follows the single-type frame spacing (10
        # min) — derived, not a CLI tunable: corpus/runtime parity by
        # construction (n_timesteps = ceil((max lead + max frame age) /
        # timestep) → 8 for a 60-min horizon at the default 12-18 min age).
        timestep_min=FRAME_SPACING_MIN,
        frame_age_range=frame_age_range,
    )
    settings_cols = settings.settings_columns()
    _LOGGER.info("settings hash: %s  %s", settings.settings_hash, settings.to_dict())

    points = load_points(args.points)
    _LOGGER.info("loaded %d calibration points from %s", len(points), args.points)

    # Archive-first listing (module docstring): one index for the parent,
    # built before anything lists a window. Workers build their own
    # (:func:`_worker_archive`) — it is far too big to ship per task.
    archive: Optional[ArchiveIndex] = None
    if args.corpus_dir is not None:
        started = time.time()
        archive = ArchiveIndex(args.corpus_dir)
        _LOGGER.info(
            "archive index: %d frames (%d %s) in %.2fs from %s; "
            "%s span %s → %s%s",
            len(archive), archive.count(args.scan_type), args.scan_type,
            time.time() - started, args.corpus_dir, args.scan_type,
            archive.earliest(args.scan_type), archive.latest(args.scan_type),
            f"; {archive.n_malformed} unparseable name(s) skipped"
            if archive.n_malformed else "",
        )
    if args.days_back < 0:
        ap.error("--days-back must be >= 0 (0 = the whole corpus archive)")
    if args.days_back == 0 and archive is None:
        ap.error(
            "--days-back 0 means 'the whole corpus archive' — pass "
            "--corpus-dir so the builder knows how far back that is"
        )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        existing = check_existing_corpus(args.output, settings.settings_hash)
    except ValueError as exc:
        _LOGGER.error("%s", exc)
        return 2
    _LOGGER.info("Found %d existing events in %s", len(existing), args.output)

    try:
        sampled = _sample_event_times(
            days_back=args.days_back, n_events=args.n_events, seed=args.seed,
            wet_bias=args.wet_bias, wet_refs=wet_refs,
            cache_dir=args.cache_dir, corpus_dir=args.corpus_dir,
            scan_type=args.scan_type, archive=archive,
        )
    except ValueError as exc:
        _LOGGER.error("%s", exc)
        return 2
    # The window is not in the settings hash, so it has to be on the
    # record some other way: the log line and the progress JSON.
    event_window = {
        "days_back": args.days_back,
        "first_event": _ts_str(sampled[0][0]) if sampled else None,
        "last_event": _ts_str(sampled[-1][0]) if sampled else None,
        "n_events": len(sampled),
    }
    _LOGGER.info(
        "sampled event window: %s → %s (%d events, --days-back %d)",
        event_window["first_event"], event_window["last_event"],
        len(sampled), args.days_back,
    )
    weight_by_ts = {_ts_str(t): w for t, w in sampled}
    # Simulated frame age per event: a pure function of (--seed, event
    # time), so a resumed run redraws the same age for an event already
    # in the corpus (module docstring, "Frame-age convention").
    age_by_ts = {
        _ts_str(t): event_frame_age_min(t, args.seed, frame_age_range)
        for t, _w in sampled
    }
    todo = [t for t, _w in sampled if _ts_str(t) not in existing]
    _LOGGER.info("Will process %d new events (of %d sampled)", len(todo), len(sampled))
    _LOGGER.info(
        "simulated frame age: %.4g-%.4g min (mean of draws %.2f min over %d events)",
        frame_age_range[0], frame_age_range[1],
        float(np.mean(list(age_by_ts.values()))) if age_by_ts else 0.0,
        len(age_by_ts),
    )

    rows_per_event = len(points) * len(leads)
    start_ts = time.time()
    progress_payload = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "total": len(sampled),
        "skipped_existing": len(existing),
        "todo": len(todo),
        "completed": 0,
        "errored": 0,
        "elapsed_s": 0.0,
        "events_per_min": 0.0,
        "eta_min": None,
        "n_points": len(points),
        "rows_per_event": rows_per_event,
        "settings": settings.to_dict(),
        "settings_hash": settings.settings_hash,
        # Sampling window — NOT part of settings_hash (module docstring),
        # so it is recorded here for the report and the dashboard.
        "event_window": event_window,
        "recent": [],  # most recent ~20 events (summaries)
        "lead_summary": {},  # rolling stats per lead, pooled over points
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    _write_progress(args.progress, progress_payload)

    if not todo:
        _LOGGER.info("Nothing to do.")
        return 0

    worker_args = [
        (t, args.cache_dir, args.corpus_dir, points, settings, age_by_ts[_ts_str(t)])
        for t in todo
    ]

    new_rows: list[dict] = []
    completed = 0
    errored = 0
    # Rolling per-lead counters (pooled over points): n, sum_raw, n_out, sum_out.
    lead_counters: dict[int, list[float]] = {lead: [0, 0.0, 0, 0] for lead in leads}
    recent_events: list[dict] = []

    with mp.get_context("spawn").Pool(args.workers) as pool:
        for result in pool.imap_unordered(_process_event, worker_args, chunksize=1):
            completed += 1
            if result.error:
                errored += 1
                _LOGGER.warning("event %s errored: %s", result.event_time, result.error)

            weight = weight_by_ts.get(result.event_time, 1.0)
            event_raws: list[float] = []
            event_outs: list[int] = []
            for row in result.rows:
                row["sample_weight"] = weight
                row.update(settings_cols)
                raw = row["raw_prob"]
                out = row["outcome"]
                if np.isfinite(raw):
                    c = lead_counters[row["lead_min"]]
                    c[0] += 1
                    c[1] += raw
                    event_raws.append(raw)
                    if out is not None:
                        c[2] += 1
                        c[3] += out
                        event_outs.append(out)
            new_rows.extend(result.rows)

            recent_events.append({
                "event_time": result.event_time,
                "n_rows": len(result.rows),
                "mean_raw": round(float(np.mean(event_raws)), 4) if event_raws else None,
                "base_rate": round(float(np.mean(event_outs)), 4) if event_outs else None,
                "error": result.error,
            })
            recent_events = recent_events[-20:]

            # Flush periodically (append = full rewrite, atomic rename); with
            # n_points × n_leads rows per event, every-event flushes would
            # spend real time re-writing the file.
            if len(new_rows) >= FLUSH_EVERY_EVENTS * rows_per_event:
                _append_rows(args.output, new_rows)
                new_rows = []

            elapsed = time.time() - start_ts
            rate = completed / max(elapsed / 60.0, 1e-6)
            remaining = len(todo) - completed
            eta_min = remaining / rate if rate > 0 else None
            progress_payload.update({
                "completed": completed,
                "errored": errored,
                "elapsed_s": round(elapsed, 1),
                "events_per_min": round(rate, 2),
                "eta_min": round(eta_min, 1) if eta_min else None,
                "recent": recent_events,
                "lead_summary": {
                    str(lead): {
                        "n_finite": int(c[0]),
                        "mean_raw": round(c[1] / c[0], 4) if c[0] else None,
                        "n_outcomes": int(c[2]),
                        "base_rate": round(c[3] / c[2], 4) if c[2] else None,
                    }
                    for lead, c in lead_counters.items()
                },
            })
            _write_progress(args.progress, progress_payload)

    # Final flush.
    _append_rows(args.output, new_rows)
    _LOGGER.info("Done. %d completed, %d errored. Output → %s", completed, errored, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
