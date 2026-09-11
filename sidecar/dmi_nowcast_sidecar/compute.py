"""One nowcast cycle — fetch → compute → emit ``state.json``.

The cycle reuses ``dmi_nowcast_core`` for the actual algorithm; the
sidecar's job is orchestration:

- Drive the DMI API via :class:`AsyncDMIClient`
- Persist per-cycle state (raining_now hysteresis, rain_incoming streak)
  across firings of the scheduler
- Apply the calibration curves
- Marshal results into the ``state.json`` schema

The STEPS ensemble (vendored pysteps subset) runs inside the cycle when
``forecast.steps.enabled`` and at least 3 frames are available — see
``_run_steps_ensemble`` and website Phase A plan §A0. Ensemble output is
additive on ``state.json`` (``p_ensemble``, ``probabilistic`` block); any
ensemble failure falls back to the deterministic-only state unchanged.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import structlog

from dmi_nowcast_core import postprocess as core_postprocess
from dmi_nowcast_core.advect import advect_field_series
from dmi_nowcast_core.basemap import build_basemap
from dmi_nowcast_core.cache import CacheConfig, DiskCache
from dmi_nowcast_core.calibrate import IsotonicCalibrator, load_calibration_curves
from dmi_nowcast_core.confidence import (
    compute_confidence,
    intensity_volatility_from_disc,
    motion_divergence,
)
from dmi_nowcast_core.corpus import CorpusArchiver
from dmi_nowcast_core.dense_flow import (
    DenseFlowUnavailable,
    dense_flow,
    estimate_motion,
)
from dmi_nowcast_core.fetch import AsyncDMIClient, RadarFeature
from dmi_nowcast_core.geo import CompositeGeo
from dmi_nowcast_core.motion import phase_correlation_shift
from dmi_nowcast_core.national import (
    NationalProducts,
    enforce_lead_monotonic,
    motion_grids_kmh,
    national_products,
    observed_rain_grid,
)
from dmi_nowcast_core.parse import RadarComposite, parse_composite
from dmi_nowcast_core.probabilistic import (
    aggregate_at_home,
    frame_age_corrected_leads,
    run_ensemble,
)
from dmi_nowcast_core.product_pairs import nearest_radar_km
from dmi_nowcast_core.raining_now import RainingNow, RainingNowConfig
from dmi_nowcast_core.sample import sample_disc
from dmi_nowcast_core.transform import dbz_to_rain_rate

from .config import Config
from .eta_smoother import EtaSmoother
from .lightning_tracker import LightningTracker
from .national_artifacts import write_national_artifacts
from .national_sample import finite_or_none, product_pixel_of
from .push.paths import resolved_postprocess_path
from .push.postprocess import (
    CyclePostprocess,
    PostprocessTable,
    build_cycle_postprocess,
    point_key,
)
from .render import render_frames
from .strike_archive import StrikeArchive
from .state_schema import (
    CalibrationBlock,
    DiagnosticsBlock,
    ForecastBlock,
    HomeBlock,
    MotionBlock,
    NowBlock,
    PerLeadEntry,
    ProbabilisticBlock,
    RadarBlock,
    State,
)
from .storage import StateStore
from .workers import run_in_pool

_log = structlog.get_logger(__name__)

# Per-pixel motion clip (in px per frame). Same value the integration uses;
# guards against optical-flow noise spikes that would otherwise blow up
# ``advect_field``. 30 px at 500 m/pixel = 15 km per inter-frame interval,
# i.e. ~180 km/h cell motion — generous upper bound for real weather.
_MAX_PX_PER_FRAME = 30.0

# 8-point compass for the motion arrow caption.
_COMPASS_LABELS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

# The STEPS ensemble horizon is ``forecast.steps.horizon_min`` (config),
# measured from RADAR-FRAME time — the ensemble's timesteps count from the
# frame while every served lead counts from now, so the horizon must cover
# the longest served lead plus the frame age at compute (14–18 min live).
# The timestep is DERIVED per cycle from the measured inter-frame spacing
# (Phase B addendum 2026-08-29: fullRange-only frames arrive every ~10 min),
# with n_timesteps = ceil(horizon_min / timestep) keeping the horizon fixed.

# Expected spacing between consecutive fullRange composites (DMI publishes
# fullRange at minutes :x0). Used only as the fallback when the measured
# inter-frame dt is degenerate (<= 0, i.e. duplicate timestamps).
_EXPECTED_FRAME_INTERVAL_MIN = 10.0


def _bearing_compass_label(dy_per_min: float, dx_per_min: float) -> str:
    """8-point compass direction the rain is coming FROM (NE, S, …).

    Mirrors :func:`custom_components.dmi_rain_incoming.coordinator._bearing_from_label`
    so the sidecar renders the same arrow labels the HA integration used to
    show. Returns "" if motion is exactly zero.
    """
    if dy_per_min == 0.0 and dx_per_min == 0.0:
        return ""
    rev_dy, rev_dx = -dy_per_min, -dx_per_min
    bearing = math.degrees(math.atan2(rev_dx, -rev_dy)) % 360.0
    idx = int(round(bearing / 45.0)) % 8
    return _COMPASS_LABELS[idx]


#: Hard cap on the points one cycle computes post-processing features
#: for. ``push.max_subscriptions`` is 200 and the gauge scoreboard adds
#: ~103 stations, so the real number is a few hundred; this only exists so
#: a misconfigured point source cannot turn a cheap step into a cycle-long
#: one. Anything past it is dropped, loudly.
_MAX_POSTPROCESS_POINTS = 2000


@dataclass(frozen=True)
class PointProducts:
    """Per-point reads of one cycle's national grids.

    Taken at the ONE moment both halves exist together: inside
    ``_run_steps_ensemble``, after ``national_products`` and *before*
    ``_calibrate_national`` replaces each ``p_rain`` grid with its
    calibrated twin. ``raw_frac_<lead>`` is the post-processing model's
    main predictor and the calibrated ``p_rain_<lead>`` is the baseline it
    has to beat, so both have to survive that swap — exactly as the replay
    keeps ``raw_p_rain`` by reference across the same line.

    ``pixels`` is the product-grid pixel each point read, carried so the
    rest of the cycle reads the same one without a second projection pass.
    """

    pixels: tuple[tuple[int, int] | None, ...]
    raw_fractions: dict[int, tuple[float | None, ...]]


@dataclass(frozen=True)
class EnsembleOutcome:
    """Home-reduced STEPS ensemble result for one cycle (plan §A0).

    Only derived scalars — the ensemble array itself is dropped inside
    ``_run_steps_ensemble`` as soon as the home reduction is done (the
    national-products consumer arrives in package A1).
    """

    # Nominal lead (minutes from now) → raw ensemble exceedance fraction.
    p_by_lead: dict[int, float]
    # (P25, P75) first-exceedance window, minutes from now; None when no
    # member predicts rain within the horizon.
    eta_window_min: tuple[float, float] | None
    n_members: int
    ensemble_ms: float
    # National ×4 product grids (plan §A1); None when ``national.enabled``
    # is off or their reduction failed (the home forecast is unaffected).
    # When national curves are loaded the ``p_rain`` grids are already
    # calibrated per lead (§B4) — the calibrated grid IS the served grid.
    national: NationalProducts | None = None
    # Wall time of the ``national_products`` reduction; the artifact-write
    # share of ``diagnostics.national_ms`` is added later in the cycle.
    national_ms: float = 0.0
    # Home leads whose ``p_by_lead`` fraction went through a national curve
    # (§B4). None when no national curves are loaded (raw pre-B4 path);
    # possibly-empty tuple when curves are loaded but don't cover a lead.
    calibrated_leads: tuple[int, ...] | None = None
    # Per-point reads of the national grids for the post-processing
    # features (H-P). None when the cycle serves no points, or when the
    # national reduction produced nothing to read.
    points: PointProducts | None = None


class NationalSnapshot(tuple):
    """What ``CycleEngine.national_latest`` publishes: the cycle's national
    products, their radar timestamp, the observed-rain grid, and the
    deterministic forecast series with the instant it was generated.

    Deliberately a **2-tuple subclass** rather than a many-field
    dataclass. ``national_latest`` has pinned readers — ``/forecast``, the
    push service, the push subscribe route and three test modules all do
    ``products, radar_ts = engine.national_latest`` — and every field past
    the pair is additive here in the same sense a manifest key is additive:
    the unpacking and indexing that existed keep working untouched, while
    readers that want a new field reach it by name (or, defensively,
    ``getattr(latest, "observed_mm_h", None)``, which also tolerates a
    plain tuple).

    ``forecast_mm_h`` maps lead minutes → the deterministic advected rain
    rate on the product grid, lead 0 being the field advected to
    ``generated_at_utc`` — the point series a "raining here now" readout
    must come from, since the observation is 14-24 min old.

    Swapped as one object so a reader on another thread can never see a
    products grid from one cycle paired with an observation from the next.
    """

    def __new__(
        cls,
        products: NationalProducts,
        radar_ts_utc: datetime,
        observed_mm_h: np.ndarray | None = None,
        forecast_mm_h: dict[int, np.ndarray] | None = None,
        generated_at_utc: datetime | None = None,
    ) -> "NationalSnapshot":
        self = super().__new__(cls, (products, radar_ts_utc))
        # tuple subclasses can't carry __slots__, so these land in __dict__.
        self.observed_mm_h = observed_mm_h
        self.forecast_mm_h = forecast_mm_h
        self.generated_at_utc = generated_at_utc
        return self

    @property
    def products(self) -> NationalProducts:
        return self[0]

    @property
    def radar_ts_utc(self) -> datetime:
        return self[1]


@dataclass
class CycleResult:
    """Outcome of one cycle. ``state`` is None when the cycle failed
    before it could produce a meaningful state object."""
    state: State | None
    error: str | None = None
    diagnostics: dict = field(default_factory=dict)


class CycleEngine:
    """Long-lived object that owns cross-cycle state.

    One instance per sidecar process. Methods are mostly async-safe (or
    are explicitly meant to run in a worker thread).

    The cycle's compute is the heaviest transient in the process — parse,
    dense flow, a STEPS ensemble, the national reduction — so it runs on
    the dedicated ``"cycle"`` worker rather than the shared default
    executor. :mod:`dmi_nowcast_sidecar.workers` has the measurements;
    the short version is that a multi-gigabyte transient costs that much
    RSS for every thread it has ever run on, and pinning it to one keeps
    the bill to one.
    """

    def __init__(
        self,
        config: Config,
        *,
        client: AsyncDMIClient | None = None,
        store: StateStore | None = None,
    ) -> None:
        self.config = config
        self._client = client or AsyncDMIClient(
            base_url=config.dmi.base_url,
            timeout_s=30.0,
        )
        self._store = store or StateStore(config.storage.data_dir)
        # Cross-cycle state. The detection threshold + statistic are configurable
        # (default p90 @ 0.5 mm/h) so a single faint column-max pixel can't trip a
        # false "raining"/"rain incoming"; off-threshold is 60 % of on.
        self._rain_threshold = config.forecast.rain_threshold_mm_h
        self._raining_now = RainingNow(
            RainingNowConfig(
                detection_threshold_mm_h=self._rain_threshold,
                hysteresis_offset_mm_h=round(self._rain_threshold * 0.4, 3),
            ),
        )
        # rain_incoming requires two consecutive cycles with predicted rain.
        # Plan §6.4 / §14.
        self._rain_incoming_streak: int = 0
        # No-new-frame fast path (website Phase B plan, addendum
        # 2026-08-29): fullRange frames land every ~10 min while the cycle
        # polls every 5, so about half the cycles re-fetch the exact frame
        # set they already computed. ``_last_frame_ts`` is the newest
        # frame's radar timestamp from the last full compute;
        # ``_last_state`` the state it produced. When the newest fetched
        # frame matches, the cycle re-emits that state with refreshed
        # clock fields instead of recomputing — critically WITHOUT
        # advancing the rain_incoming streak or stepping the raining_now
        # hysteresis (both count radar observations, not poll firings).
        self._last_frame_ts: datetime | None = None
        self._last_state: State | None = None
        # Motion diagnostics of the last full compute, folded into the
        # ``cycle_ok`` log line (H-F, 2026-09-08). Scalars only — never a
        # grid — so a re-emitted state carries no stale array. Empty until
        # the first full cycle, and deliberately NOT cleared by the
        # no-new-frame fast path: that path re-emits the same state, so the
        # same motion numbers still describe it.
        self._last_motion_diag: dict[str, Any] = {}
        # Geo cached after first composite (projection rarely changes).
        self._geo: CompositeGeo | None = None
        # Calibration curves loaded once. ``_curves`` are the legacy
        # home-point curves feeding the binary ``p_calibrated`` field (the
        # HA contract, untouched); ``_national_curves`` are the pooled
        # national curves (§B4) applied to the ensemble ``p_ensemble``
        # fractions and the national ``p_rain`` grids.
        self._curves: dict[int, IsotonicCalibrator] = {}
        self._calibration_metadata: dict | None = None
        self._national_curves: dict[int, IsotonicCalibrator] = {}
        self._national_calibration_metadata: dict | None = None
        self._national_fitted_at: datetime | None = None
        # (mtime_ns, size) of the national curve file as loaded. The
        # sync task (F4) can replace that file under a running process;
        # comparing the stamp at the start of each cycle is how a new
        # fit takes effect without a restart.
        self._national_curves_stamp: tuple[int, int] | None = None
        self._national_curves_dirty = False
        self._reload_calibration()
        # Working cache for downloaded HDF5 files — short-lived, LRU-evicted
        # after each cycle to keep disk under ``working_cache_max_bytes``.
        self._cache_dir = config.storage.data_dir / "composites"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._working_cache = DiskCache(
            CacheConfig(
                root=self._cache_dir,
                max_bytes=config.storage.working_cache_max_bytes,
            ),
        )
        # Persistent corpus archive (Phase A). When ``corpus_dir`` is set,
        # every successful download is also copied into the corpus's
        # ``composites/YYYY/MM/`` tree. The archive is bind-mounted from the
        # host so it survives ``docker compose down -v``.
        self._corpus: CorpusArchiver | None = None
        if config.storage.corpus_dir is not None:
            try:
                self._corpus = CorpusArchiver(config.storage.corpus_dir)
            except OSError as exc:
                _log.warning(
                    "corpus_init_failed",
                    corpus_dir=str(config.storage.corpus_dir),
                    error=str(exc),
                )
        # Frames directory served by /frames/*.png.
        self._frames_dir = config.storage.data_dir / "frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        # National artifacts directory served by /nowcast/* (plan §A2/§A3).
        self._national_dir = config.storage.data_dir / "nowcast"
        self._national_dir.mkdir(parents=True, exist_ok=True)
        # Latest national products + their radar timestamp + the observed
        # rain grid, held in memory for the /forecast point lookup (plan
        # §A3) and the push decision engine. Swapped as one object so
        # readers on other threads never see a torn set.
        self._national_latest: NationalSnapshot | None = None
        # Gauge-trained post-processing (Phase H, H-P). The model is read
        # from the served file and hot-reloaded on an mtime change, like
        # the national curves and the push threshold table; the per-cycle
        # answer is published as one immutable object beside the national
        # products so a reader on another thread can never pair one
        # frame's features with another frame's probabilities.
        #
        # ``_point_sources`` are the things that ask to be scored: the
        # push store's subscriptions and the gauge scoreboard's stations,
        # registered by ``app.create_app``. Callables rather than a list,
        # because both move between cycles. Home is always in, so the
        # cycle has at least one point to answer for.
        self._postprocess = PostprocessTable(resolved_postprocess_path(config))
        self._point_sources: list[tuple[str, Any]] = []
        self._postprocess_latest: CyclePostprocess | None = None
        #: ``(lat, lon)`` → km to the nearest radar. A property of the
        #: point, not of the cycle, and the cycle asks for it once per
        #: point per frame.
        self._radar_km: dict[tuple[float, float], float] = {}
        # Basemap dir + cached image; lazy-loaded on first cycle.
        self._basemap_dir = config.storage.data_dir / "basemap"
        self._basemap_dir.mkdir(parents=True, exist_ok=True)
        self._basemap: Any | None = None
        self._basemap_attempted: bool = False
        # Previous-cycle disc max for intensity volatility.
        self._prev_disc_max: float | None = None
        # Rolling buffer of Blitzortung strikes pushed by HA (lightning ETA),
        # with optional append-only persistence for backtesting/calibration.
        self._strike_archive: StrikeArchive | None = None
        if config.lightning.archive_enabled:
            try:
                self._strike_archive = StrikeArchive(config.lightning.archive_dir)
            except OSError as exc:
                _log.warning(
                    "strike_archive_init_failed",
                    archive_dir=str(config.lightning.archive_dir),
                    error=str(exc),
                )
        self._lightning = LightningTracker(config.lightning, archive=self._strike_archive)
        # Cross-cycle EMA smoothing state for the lightning ETA, per target.
        self._eta_smoother = EtaSmoother(config.lightning)

    @property
    def store(self) -> StateStore:
        return self._store

    @property
    def lightning(self) -> LightningTracker:
        return self._lightning

    @property
    def eta_smoother(self) -> EtaSmoother:
        return self._eta_smoother

    @property
    def strike_archive(self) -> StrikeArchive | None:
        return self._strike_archive

    @property
    def geo(self) -> CompositeGeo | None:
        """Cached projection from the latest composite (None before first cycle)."""
        return self._geo

    @property
    def postprocess(self) -> PostprocessTable:
        """The fitted post-processing model this process reads (H-P).

        Shared with ``/api/push/options`` and with the sync task's reload
        nudge, so what the panel reports and what the cycle scored with
        can never drift apart.
        """
        return self._postprocess

    @property
    def postprocess_latest(self) -> CyclePostprocess | None:
        """The last cycle's post-processing answer, or None.

        Swapped as one object; a reader checks ``radar_ts_utc`` before
        trusting it, exactly as it does for ``national_latest``.
        """
        return self._postprocess_latest

    def add_point_source(self, name: str, provider: Any) -> None:
        """Register a supplier of points the cycle should score (H-P).

        ``provider()`` returns an iterable of ``(lat, lon)`` and is called
        once per full cycle, inside the cycle worker — so it may block,
        and it must not raise anything the cycle cannot survive (it is
        called under a guard that logs and skips the source).

        Coordinates only, deliberately: the cycle computes features for a
        set of *places* and has no business learning whose they are. A
        push endpoint is a bearer capability that never leaves the store.
        """
        self._point_sources.append((str(name), provider))

    @property
    def national_latest(self) -> NationalSnapshot | None:
        """Latest national products + radar timestamp + observed grid (§A3).

        None until the first successful ensemble cycle with national
        products enabled. The /forecast endpoint and the push decision
        engine sample these grids. Unpacks as the ``(products, radar_ts)``
        pair it has always been; the observed grid is the named
        ``observed_mm_h`` attribute (see :class:`NationalSnapshot`)."""
        return self._national_latest

    @property
    def national_curve_leads(self) -> frozenset[int]:
        """Leads covered by the loaded national calibration curves (§B4).

        Curves load at init and are re-read only at the START of a cycle,
        when the file on disk has changed (Phase F, F4: the public
        instance's ``sync`` task drops a freshly-fitted file in). That is
        the one instant a swap is safe, so the products held in
        ``national_latest`` were still calibrated with exactly the curves
        this set names — /forecast derives its
        truthful ``calibrated`` flag from this set."""
        return frozenset(self._national_curves)

    @property
    def national_calibration_fitted_at(self) -> datetime | None:
        """``fitted_at`` of the loaded national curve file; None without one."""
        return self._national_fitted_at

    @property
    def basemap(self) -> Any | None:
        """Cached OSM basemap PIL image for the home crop (None if unavailable)."""
        return self._basemap

    async def aclose(self) -> None:
        await self._client.close()

    def _reload_calibration(self) -> None:
        """Load both curve files (legacy home + national pooled, §B4).

        Called once at init; calibrate.sh's restart-to-pick-up flow covers
        both files. Each file degrades independently: missing/corrupt →
        empty dict → that path serves raw values, exactly as before.
        """
        self._reload_legacy_curves()
        self._reload_national_curves()

    def _reload_legacy_curves(self) -> None:
        path = self.config.calibration.curves_path
        if not path.exists():
            self._curves = {}
            self._calibration_metadata = None
            _log.info("calibration_curves_missing", path=str(path))
            return
        try:
            self._curves = load_calibration_curves(path)
            raw = json.loads(path.read_text())
            self._calibration_metadata = raw.get("metadata") or None
            _log.info(
                "calibration_curves_loaded",
                n_leads=len(self._curves),
                fitted_at=(self._calibration_metadata or {}).get("fitted_at"),
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("calibration_load_failed", error=str(exc))
            self._curves = {}
            self._calibration_metadata = None

    def _reload_national_curves(self) -> None:
        """National pooled curves (§B4) — same file format as the legacy
        curves, loaded via the same :func:`load_calibration_curves`.
        Missing/corrupt file → empty dict → everything behaves exactly as
        today (``calibrated: false``, raw fractions)."""
        path = self.config.calibration.national_curves_path
        self._national_curves_stamp = _file_stamp(path)
        self._national_curves_dirty = False
        if not path.exists():
            self._national_curves = {}
            self._national_calibration_metadata = None
            self._national_fitted_at = None
            _log.info("national_curves_missing", path=str(path))
            return
        try:
            self._national_curves = load_calibration_curves(path)
            raw = json.loads(path.read_text())
            self._national_calibration_metadata = raw.get("metadata") or None
            self._national_fitted_at = _parse_iso(
                (self._national_calibration_metadata or {}).get("fitted_at"),
            )
            _log.info(
                "national_curves_loaded",
                n_leads=len(self._national_curves),
                fitted_at=(self._national_calibration_metadata or {}).get("fitted_at"),
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("national_curves_load_failed", error=str(exc))
            self._national_curves = {}
            self._national_calibration_metadata = None
            self._national_fitted_at = None

    def note_curves_changed(self) -> None:
        """Ask for a curve re-read at the start of the next cycle.

        Called by the ``sync`` task after it writes a freshly-fitted
        national curve file (Phase F, F4). It only sets a flag: swapping
        the curves mid-cycle would leave ``national_latest`` calibrated
        with one set of curves and ``national_curve_leads`` describing
        another, which is exactly the kind of quiet inconsistency
        /forecast's ``calibrated`` flag exists to prevent.
        """
        self._national_curves_dirty = True

    def _reload_national_curves_if_changed(self) -> None:
        """Re-read the national curves when the file on disk moved.

        Cheap: a ``stat`` on every cycle, a JSON parse only when the
        (mtime, size) pair changed or ``note_curves_changed`` asked for
        one. Without this the private instance's monthly fit reaches the
        public instance only on the next restart.
        """
        path = self.config.calibration.national_curves_path
        stamp = _file_stamp(path)
        if not self._national_curves_dirty and stamp == self._national_curves_stamp:
            return
        _log.info(
            "national_curves_changed_on_disk",
            path=str(path),
            was=self._national_curves_stamp,
            now=stamp,
        )
        self._reload_national_curves()

    async def run_cycle(self) -> CycleResult:
        """Execute one cycle. Network in this coroutine; compute in a thread."""
        t0 = time.perf_counter()
        fetch_ms = compute_ms = 0.0
        # The start of a cycle is the one safe moment to swap the national
        # curves: everything this cycle publishes is calibrated with what
        # is loaded now, so the pair (products, curve leads) stays honest.
        self._reload_national_curves_if_changed()
        try:
            t_fetch = time.perf_counter()
            paths = await self._fetch_latest_frames()
            fetch_ms = (time.perf_counter() - t_fetch) * 1000
            if len(paths) < 2:
                raise RuntimeError(f"not enough frames available (got {len(paths)})")

            t_compute = time.perf_counter()
            state = await run_in_pool("cycle", self._compute_sync, paths, fetch_ms)
            compute_ms = (time.perf_counter() - t_compute) * 1000

            cycle_ms = (time.perf_counter() - t0) * 1000
            # Patch in the cycle_ms now that we know it (the compute fn only
            # had fetch_ms + its own compute_ms; cycle_ms is the total).
            state = state.model_copy(
                update={
                    "diagnostics": state.diagnostics.model_copy(
                        update={"cycle_ms": cycle_ms},
                    )
                }
            )
            self._store.write(state)
            _log.info(
                "cycle_ok",
                cycle_ms=round(cycle_ms, 1),
                fetch_ms=round(fetch_ms, 1),
                compute_ms=round(compute_ms, 1),
                raining=state.now.raining,
                rain_incoming=state.forecast.rain_incoming,
                eta_minutes=state.forecast.eta_minutes,
                # H-F: which completion ran, how much of the echo the
                # estimator stalled on, how much of that survived into the
                # advected field, and the bulk vector it relaxed toward.
                **self._last_motion_diag,
            )
            return CycleResult(
                state=state,
                diagnostics={
                    "cycle_ms": cycle_ms,
                    "fetch_ms": fetch_ms,
                    "compute_ms": compute_ms,
                },
            )
        except Exception as exc:  # noqa: BLE001
            cycle_ms = (time.perf_counter() - t0) * 1000
            _log.exception("cycle_failed", error=str(exc), cycle_ms=round(cycle_ms, 1))
            return CycleResult(
                state=None,
                error=str(exc),
                diagnostics={
                    "cycle_ms": cycle_ms,
                    "fetch_ms": fetch_ms,
                    "compute_ms": compute_ms,
                },
            )

    async def _fetch_latest_frames(self) -> list[Path]:
        """Download the last 4 fullRange composites into the working cache,
        archive each into the persistent corpus, then LRU-evict the cache.

        The ``scan_type`` filter (``config.dmi.scan_type``, decided
        fullRange-only — Phase B addendum) is applied server-side, so the
        4 frames kept are consecutive fullRange composites spanning ~30
        min at the 10-min fullRange cadence — never mixed with the
        interleaved doppler product.
        """
        features = await self._client.list_latest(
            limit=6, scan_type=self.config.dmi.scan_type,
        )
        if not features:
            return []
        # Keep the most recent 4 frames; need at least 2 for motion, more
        # gives STEPS room to estimate ARI parameters in Phase B+.
        features_sorted: list[RadarFeature] = sorted(features, key=lambda f: f.datetime_utc)
        latest = features_sorted[-4:]
        paths: list[Path] = []
        for feat in latest:
            try:
                path = await self._client.download(feat, self._cache_dir)
            except Exception as exc:  # noqa: BLE001
                _log.warning("download_failed", filename=feat.filename, error=str(exc))
                continue
            paths.append(path)
            if self._corpus is not None:
                try:
                    res = await asyncio.to_thread(self._corpus.archive, path)
                    if res.archived:
                        _log.debug("corpus_archived", filename=path.name)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("corpus_archive_failed", filename=path.name, error=str(exc))
        # Evict the LRU tail of the working cache so disk stays bounded.
        # The corpus owns the long-term record; the cache only needs the
        # last few hours of frames.
        try:
            evicted_files, evicted_bytes = await asyncio.to_thread(self._working_cache.evict)
            if evicted_files:
                _log.info(
                    "cache_evicted",
                    files=evicted_files,
                    bytes=evicted_bytes,
                )
        except Exception as exc:  # noqa: BLE001
            _log.warning("cache_evict_failed", error=str(exc))
        return paths

    def _compute_sync(self, paths: list[Path], fetch_ms: float) -> State:
        """Blocking compute path — call from inside the ``"cycle"`` worker."""
        t_compute = time.perf_counter()

        composites: list[RadarComposite] = [parse_composite(p) for p in paths]
        composites.sort(key=lambda c: c.timestamp_utc)
        composite_now = composites[-1]
        composite_prev = composites[-2]

        # No-new-frame fast path (Phase B addendum): with fullRange frames
        # every ~10 min and the cycle polling every 5, roughly every other
        # cycle sees the exact frame set it already computed. Recomputing
        # would double-count one radar observation in the rain_incoming
        # two-cycle persistence and the raining_now hysteresis (and burn a
        # STEPS run for nothing), so re-emit the previous good state with
        # refreshed clock fields instead. Nothing below this line runs.
        if (
            self._last_state is not None
            and self._last_frame_ts == composite_now.timestamp_utc
        ):
            return self._emit_unchanged_state(self._last_state, fetch_ms)

        if self._geo is None or self._geo.composite.projection != composite_now.projection:
            self._geo = CompositeGeo(composite_now)
        geo = self._geo

        # Rain fields.
        rain_now = dbz_to_rain_rate(
            composite_now.reflectivity_dbz,
            zr_a=composite_now.zr_a, zr_b=composite_now.zr_b,
        )
        rain_prev = dbz_to_rain_rate(
            composite_prev.reflectivity_dbz,
            zr_a=composite_prev.zr_a, zr_b=composite_prev.zr_b,
        )

        lon = self.config.home.lon
        lat = self.config.home.lat
        radius_m = self.config.home.radius_km * 1000.0

        stats_now = sample_disc(rain_now, geo, lon, lat, radius_m=radius_m)
        stats_prev = sample_disc(rain_prev, geo, lon, lat, radius_m=radius_m)

        # raining_now state machine (cross-cycle), keyed on the configured disc
        # statistic (default p90) so one hot clutter/virga pixel can't trip it.
        stat = self.config.forecast.detection_stat
        rn_result = self._raining_now.update(getattr(stats_now, f"{stat}_mm_h"))

        # Motion field. ``dt_min`` is the measured inter-frame spacing —
        # ~10 min on the fullRange-only feed — and doubles as the STEPS
        # timestep downstream (Phase B addendum: timestep follows frame
        # spacing). Degenerate spacing falls back to the nominal cadence.
        dt_min = (composite_now.timestamp_utc - composite_prev.timestamp_utc).total_seconds() / 60.0
        if dt_min <= 0:
            dt_min = _EXPECTED_FRAME_INTERVAL_MIN
        method_used: str = self.config.forecast.method
        raw_flow: tuple[np.ndarray, np.ndarray]
        try:
            if self.config.forecast.method == "mean-motion":
                raise DenseFlowUnavailable("forced via config")
            raw_flow = dense_flow(
                composite_prev.reflectivity_dbz,
                composite_now.reflectivity_dbz,
            )
        except DenseFlowUnavailable:
            method_used = "mean-motion"
            dy, dx = phase_correlation_shift(rain_prev, rain_now)
            shape = rain_now.shape
            raw_flow = (
                np.full(shape, dy, dtype=np.float32),
                np.full(shape, dx, dtype=np.float32),
            )

        # Estimate → complete → sanitise, through the ONE entry point the
        # corpus builder and the warning replay also call. Completion runs
        # ahead of the clip so BOTH consumers get the completed field: the
        # deterministic overlay advection here, and the STEPS velocity
        # (``_run_steps_ensemble`` downsamples this same array).
        #
        # ``flow_completion`` (config, default ``confidence``) is the H-F
        # hotfix: without it the Farnebäck stall inside broad echo survives
        # into the advection and into STEPS. See
        # ``dense_flow.complete_flow`` and
        # ``archive/flow_stall_20260908/README.md``.
        pixel_km = float(composite_now.xscale_m) / 1000.0
        motion = estimate_motion(
            composite_prev.reflectivity_dbz,
            composite_now.reflectivity_dbz,
            rain_now,
            pixel_km=pixel_km,
            dt_min=dt_min,
            support_threshold_mm_h=self._rain_threshold,
            completion=self.config.forecast.flow_completion,
            confidence_window_px=self.config.forecast.flow_confidence_window_px,
            confidence_percentile=self.config.forecast.flow_confidence_percentile,
            texture_percentile=self.config.forecast.flow_texture_percentile,
            max_px_per_frame=_MAX_PX_PER_FRAME,
            flow=raw_flow,
        )
        del raw_flow
        vy, vx = motion.vy, motion.vx
        # Cycle diagnostics, logged with ``cycle_ok`` and (the raw stalled
        # share) served in ``state.motion``.
        self._last_motion_diag = {
            "flow_completion": motion.completion,
            "stalled_share": round(motion.stalled_share, 3),
            "stalled_share_completed": round(motion.stalled_share_completed, 3),
            "bulk_kmh": round(
                math.hypot(motion.bulk_vy, motion.bulk_vx) * pixel_km * 60.0 / dt_min,
                1,
            ),
        }
        motion_stalled_share = motion.stalled_share

        # H-P: the post-processing features, for every point this cycle
        # serves, off the SAME anchor field and the SAME completed flow
        # the ensemble is about to run on — no second motion estimate, no
        # second STEPS. Computed here, before the cascade, because the
        # MotionEstimate carries the bulk motion and the stall share the
        # features need and is dropped on the next line.
        #
        # The table itself is a few hundred rows of float32 and keeps no
        # reference to any grid; the transient inside ``station_features``
        # is the 40 km corridor gather, a few megabytes at the point
        # counts this service can reach.
        self._postprocess.maybe_reload()
        pp_keys = self._serving_points()
        pp_native: list[Any] = []
        pp_grid: dict[str, np.ndarray] | None = None
        if pp_keys:
            try:
                pp_native = [geo.lonlat_to_grid(lon, lat) for lat, lon in pp_keys]
                pp_grid = core_postprocess.station_features(
                    rain_now, vy, vx,
                    np.array([idx.row for idx in pp_native], dtype=np.float64),
                    np.array([idx.col for idx in pp_native], dtype=np.float64),
                    pixel_km=pixel_km,
                    dt_min=dt_min,
                    bulk_vy=motion.bulk_vy,
                    bulk_vx=motion.bulk_vx,
                    stalled_share=motion.stalled_share,
                )
            except Exception as exc:  # noqa: BLE001 — a feature failure costs
                # the post-processed probability for one cycle, never the
                # cycle: the engine falls back to the served curve.
                _log.warning("postprocess_features_failed", error=str(exc))
                pp_grid = None

        # ``motion`` also holds the two RAW native grids (~14 MB each) that
        # only the stall diagnostic needed; the served arrows now show the
        # completed field. Drop them before STEPS, the cycle's memory
        # high-water mark.
        del motion

        # Disc-area mean motion (rain-weighted in 120 km window around home).
        disc_dy_per_min, disc_dx_per_min = _disc_motion(
            rain_now, vy, vx, geo, composite_now, lon, lat, dt_min,
        )
        pixel_scale_m = float(composite_now.xscale_m)
        disc_speed_kmh = math.hypot(disc_dy_per_min, disc_dx_per_min) * pixel_scale_m * 60.0 / 1000.0
        bearing_from = _bearing_from_deg(disc_dy_per_min, disc_dx_per_min)

        # Per-lead forecasts.
        per_lead: list[PerLeadEntry] = []
        peak_rate = 0.0
        peak_lead = 0
        eta_minutes: float | None = None
        frame_age_s = max(
            0.0,
            (datetime.now(timezone.utc) - composite_now.timestamp_utc).total_seconds(),
        )
        frame_age_min = frame_age_s / 60.0

        # STEPS ensemble on the sanitised native-resolution flow (plan §A0).
        # Any failure inside returns None and the cycle emits exactly the
        # deterministic-only state below.
        ensemble = self._run_steps_ensemble(
            composites, vy, vx, geo, frame_age_min=frame_age_min, dt_min=dt_min,
            points=pp_native if pp_grid is not None else None,
        )

        # Native-500 m advected fields double as the national overlay frames
        # (plan §A2); key 0 is the "now" frame. Collected only when there are
        # national products to write them alongside.
        collect_overlays = ensemble is not None and ensemble.national is not None
        overlay_fields: dict[int, np.ndarray] = {0: rain_now} if collect_overlays else {}

        # Project rain forward from radar-frame time, so "lead minutes from
        # now" must add back the frame age. ``leads_min`` is validated
        # ascending, so one integration pass serves every lead: the
        # trajectory is carried forward between leads rather than being
        # re-integrated from zero for each. The sub-stepped semi-Lagrangian
        # scheme costs far more than the old one-shot Euler back-step
        # (~7 s for 8 leads on the native 1728×1984 grid), and chaining
        # takes ~20 % off that.
        #
        # The series LEADS with the frame age alone: the field advected to
        # NOW, which is the clock a viewer is on. The newest composite is
        # 14-24 min old at any moment (10-min cadence + ~12 min DMI delay +
        # a cycle serving up to 10 min), so the observation cannot answer
        # "is it raining here now" and the first forecast lead is already
        # ten minutes past it. Horizons stay non-decreasing, chaining makes
        # the extra one near-free, and the per-lead pairing below is
        # unchanged: the lead-0 field is consumed off an explicit iterator
        # first, so lead i still pairs with the field for lead i.
        advected = iter(advect_field_series(
            rain_now, vy, vx,
            horizons_minutes=(
                [frame_age_min]
                + [lead + frame_age_min for lead in self.config.forecast.leads_min]
            ),
            dt_minutes=dt_min,
        ))
        forecast_now_field = next(advected)
        for lead, field in zip(self.config.forecast.leads_min, advected):
            if collect_overlays:
                overlay_fields[int(lead)] = field
            disc = sample_disc(field, geo, lon, lat, radius_m=radius_m)
            disc_val = getattr(disc, f"{stat}_mm_h")
            mm_h = float(disc_val) if np.isfinite(disc_val) else 0.0
            # Yes/No raw probability — kept deterministic on purpose: the
            # isotonic curves were fitted on this binary forecast, so the
            # ensemble fraction goes into the separate ``p_ensemble`` field
            # instead of replacing ``p_rain`` (plan §A0, calibration honesty).
            p_raw = 1.0 if mm_h >= self._rain_threshold else 0.0
            cal = self._curves.get(int(lead))
            p_cal = float(cal.predict(p_raw)) if cal is not None else p_raw
            per_lead.append(PerLeadEntry(
                lead_min=int(lead),
                rain_rate_mm_h=mm_h,
                p_rain=p_raw,
                p_calibrated=p_cal,
                p_ensemble=(
                    ensemble.p_by_lead.get(int(lead)) if ensemble is not None else None
                ),
            ))
            if mm_h > peak_rate:
                peak_rate = mm_h
                peak_lead = int(lead)
            if eta_minutes is None and mm_h >= self._rain_threshold:
                eta_minutes = float(lead)

        # rain_incoming with persistence (two consecutive cycles).
        wet_predicted = eta_minutes is not None
        self._rain_incoming_streak = (
            self._rain_incoming_streak + 1 if wet_predicted else 0
        )
        rain_incoming = self._rain_incoming_streak >= 2

        # Confidence.
        volatility = intensity_volatility_from_disc(stats_prev.max_mm_h, stats_now.max_mm_h)
        divergence = motion_divergence(vy, vx)
        conf = compute_confidence(
            horizon_minutes=30.0,
            frame_age_seconds=frame_age_s,
            intensity_volatility=volatility,
            motion_divergence=divergence,
            n_frames=len(composites),
        )

        compute_ms = (time.perf_counter() - t_compute) * 1000

        # Frame rendering — kicked off here so the manifest references the
        # same per-cycle state we're about to emit. Render time is tracked
        # separately so users can see it on /state.json.
        #
        # Public mode (Phase C §P1) skips this block entirely: the home crop
        # and the OSM basemap it draws on exist only to serve /frames/*,
        # which the public instance hides — ~3.5 s of CPU per cycle plus one
        # network fetch, for nobody. Everything the public site consumes
        # (national artifacts, state writing) is below and untouched.
        # ``render_ms`` stays 0.0, exactly as it does when a render fails.
        render_ms = 0.0
        bearing_compass = _bearing_compass_label(disc_dy_per_min, disc_dx_per_min)
        if self.config.server.public_mode:
            _log.debug("render_skipped_public_mode")
        else:
            try:
                self._ensure_basemap()
                now_subline = _build_now_subline(
                    stats_now=stats_now,
                    eta_minutes=eta_minutes,
                    peak_rate=peak_rate,
                    peak_lead=peak_lead,
                )
                apng_bytes, render_ms = render_frames(
                    composites=composites,
                    rain_now=rain_now,
                    vy=vy, vx=vx,
                    dt_min=dt_min,
                    frame_age_min=frame_age_min,
                    geo=geo,
                    home_lat=lat, home_lon=lon,
                    radius_km=self.config.home.radius_km,
                    out_dir=self._frames_dir,
                    now_stats_subline=now_subline,
                    disc_motion_dy_per_min=disc_dy_per_min,
                    disc_motion_dx_per_min=disc_dx_per_min,
                    disc_motion_speed_kmh=disc_speed_kmh,
                    disc_motion_bearing_from=bearing_compass,
                    basemap=self._basemap,
                )
                # APNG to disk too — served at /frames/loop.png so the HA
                # image entity can fetch a single self-animating artifact.
                _atomic_write_bytes(self._frames_dir / "loop.png", apng_bytes)
            except Exception as exc:  # noqa: BLE001
                # A render failure shouldn't kill the cycle — state.json still
                # gets written, the Lovelace card just won't have a fresh loop.
                _log.warning("render_failed", error=str(exc))

        # National artifacts (plan §A2). The in-memory products are published
        # for the /forecast lookup (plan §A3) even when the disk write fails —
        # memory and disk are independent consumers of the same reduction.
        # Artifact failure follows the render policy: warn, never kill the cycle.
        national_ms = ensemble.national_ms if ensemble is not None else 0.0
        artifact_bytes = 0
        if ensemble is not None and ensemble.national is not None:
            t_art = time.perf_counter()
            # OBSERVED rain on the product grid, from the same ``rain_now``
            # the "now" overlay is rendered from and the same downsample
            # factor the ensemble reduced with — so it aligns pixel-for-
            # pixel with p_rain / eta / intensity / motion. This is the
            # only served grid that answers "is it raining at this point
            # RIGHT NOW": the ensemble's first timestep is already ~10 min
            # out, so a point under a shower that clears within 10 min
            # reads as a fresh arrival on the ETA grid alone. ~17 ms on
            # the full 1728×1984 composite. A failure costs the observed
            # grid, not the cycle.
            observed_grid: np.ndarray | None = None
            try:
                observed_grid = observed_rain_grid(
                    rain_now,
                    downsample_factor=ensemble.national.downsample_factor,
                )
            except Exception as exc:  # noqa: BLE001
                _log.warning("observed_grid_failed", error=str(exc))
            # DETERMINISTIC forecast series on the same product grid, from
            # the same fields the overlays are drawn from and through the
            # same reduction — so the panel's headline, the loop frame it
            # sits next to and the ETA grid all read one pixel. Lead 0 is
            # the field advected to ``generated_at``: the answer to "is it
            # raining here now" that the 14-24 min old composite cannot
            # give. ~16 ms per lead; a failure costs the series, not the
            # cycle.
            #
            # ONE generation instant for the series, the snapshot and the
            # artifacts: a client must be able to add a lead to it and land
            # on the frame the manifest says is valid then.
            generated_at_utc = datetime.now(timezone.utc)
            forecast_grids: dict[int, np.ndarray] | None = None
            try:
                factor = ensemble.national.downsample_factor
                fields: dict[int, np.ndarray] = {0: forecast_now_field}
                for lead in self.config.forecast.leads_min:
                    if int(lead) != 0 and int(lead) in overlay_fields:
                        fields[int(lead)] = overlay_fields[int(lead)]
                forecast_grids = {
                    lead: observed_rain_grid(fld, downsample_factor=factor)
                    for lead, fld in sorted(fields.items())
                }
            except Exception as exc:  # noqa: BLE001
                _log.warning("forecast_grids_failed", error=str(exc))
            self._national_latest = NationalSnapshot(
                ensemble.national,
                composite_now.timestamp_utc,
                observed_grid,
                forecast_grids,
                generated_at_utc,
            )
            # H-P: the served points' feature rows and post-processed
            # probabilities, published right beside the snapshot they were
            # read off. Same instant, same grids, same frame stamp — the
            # push fan-out and the gauge scoreboard both check
            # ``radar_ts_utc`` before trusting either object.
            self._publish_postprocess(
                keys=pp_keys,
                grid_features=pp_grid,
                points=ensemble.points,
                products=ensemble.national,
                observed_grid=observed_grid,
                radar_ts_utc=composite_now.timestamp_utc,
                generated_at_utc=generated_at_utc,
                frame_age_min=frame_age_min,
            )
            # R2 cell-motion grids: the display product, on the product
            # grid, in km/h. Fed the COMPLETED flow — the same array the
            # overlays and STEPS ran on — so the arrow the user clicks and
            # the motion the loop shows are one number. It used to be the
            # raw estimate, on the argument that "on the echo the vector is
            # the measured optical flow"; since the H-F hotfix that is no
            # longer what the forecast advects with, and drawing the
            # measured-but-stalled vector next to a loop that moves would
            # be a straight contradiction (2026-09-08). Off the echo,
            # ``motion_grids_kmh`` still runs its own nearest-cells
            # completion rather than the national bulk (issue #6). A
            # failure here costs the click-anywhere arrow, not the cycle.
            motion_east = motion_north = None
            try:
                motion_east, motion_north = motion_grids_kmh(
                    vy, vx, rain_now,
                    pixel_km=float(composite_now.xscale_m) / 1000.0,
                    # ``vy``/``vx`` are pixels per inter-frame interval.
                    timestep_min=dt_min,
                    downsample_factor=ensemble.national.downsample_factor,
                    support_threshold_mm_h=self._rain_threshold,
                )
            except Exception as exc:  # noqa: BLE001
                _log.warning("motion_grids_failed", error=str(exc))
            try:
                art = write_national_artifacts(
                    ensemble.national,
                    geo=geo,
                    radar_ts_utc=composite_now.timestamp_utc,
                    generated_at_utc=generated_at_utc,
                    overlay_fields_mm_h=overlay_fields,
                    out_dir=self._national_dir,
                    keep_cycles=self.config.forecast.national.keep_cycles,
                    motion_east_kmh=motion_east,
                    motion_north_kmh=motion_north,
                    observed_mm_h=observed_grid,
                    forecast_mm_h=forecast_grids,
                    # The lead-0 forecast as an overlay frame too: the loop
                    # frame nearest wall-clock now, beside the lead-0
                    # observation it must not be confused with.
                    overlay_now_forecast_mm_h=(
                        forecast_now_field if collect_overlays else None
                    ),
                    # §B4: null when the served grids are raw; otherwise
                    # fitted_at + curve-file echo + calibrated_leads.
                    calibration=self._national_calibration_manifest(ensemble.national),
                    # The honest horizon from now is this minus the
                    # manifest's ``frame_age_min`` — the site needs both
                    # numbers to know which leads the grids can answer.
                    ensemble_horizon_min=float(
                        self.config.forecast.steps.horizon_min
                    ),
                )
                artifact_bytes = art.bytes_written
            except Exception as exc:  # noqa: BLE001
                _log.warning("national_artifacts_failed", error=str(exc))
            national_ms += (time.perf_counter() - t_art) * 1000

        # State payload.
        now_utc = datetime.now(timezone.utc)

        state = State(
            schema_version=1,
            generated_at=now_utc,
            radar=RadarBlock(
                latest_ts=composite_now.timestamp_utc,
                data_age_minutes=round(frame_age_min, 2),
            ),
            home=HomeBlock(
                lat=lat, lon=lon, radius_km=self.config.home.radius_km,
            ),
            now=NowBlock(
                rain_rate_mm_h=float(stats_now.max_mm_h)
                    if np.isfinite(stats_now.max_mm_h) else 0.0,
                rain_rate_p90_mm_h=float(stats_now.p90_mm_h)
                    if np.isfinite(stats_now.p90_mm_h) else 0.0,
                raining=bool(rn_result.state),
                raining_hysteresis_state="wet" if rn_result.state else "dry",
            ),
            forecast=ForecastBlock(
                method=method_used,  # type: ignore[arg-type]
                rain_incoming=rain_incoming,
                eta_minutes=eta_minutes,
                eta_p50_window_min=(
                    ensemble.eta_window_min if ensemble is not None else None
                ),
                peak_intensity_mm_h=peak_rate,
                peak_lead_min=peak_lead,
                per_lead=per_lead,
            ),
            probabilistic=(
                ProbabilisticBlock(
                    n_members=ensemble.n_members,
                    # §B4: true only when EVERY served home lead went
                    # through a national curve; partial coverage stays
                    # false, with calibrated_leads naming the subset.
                    calibrated=(
                        ensemble.calibrated_leads is not None
                        and set(ensemble.calibrated_leads)
                        >= {int(lead) for lead in self.config.forecast.leads_min}
                    ),
                    eta_p50_window_min=ensemble.eta_window_min,
                    calibration_fitted_at=(
                        self._national_fitted_at
                        if ensemble.calibrated_leads else None
                    ),
                    calibrated_leads=(
                        list(ensemble.calibrated_leads)
                        if ensemble.calibrated_leads is not None else None
                    ),
                )
                if ensemble is not None
                else None
            ),
            motion=MotionBlock(
                dy_px_per_min=disc_dy_per_min,
                dx_px_per_min=disc_dx_per_min,
                speed_km_per_h=disc_speed_kmh,
                bearing_deg_from=bearing_from,
                stalled_share=round(motion_stalled_share, 3),
            ),
            confidence=float(conf.score),
            calibration=CalibrationBlock(
                fitted_at=_parse_iso((self._calibration_metadata or {}).get("fitted_at")),
                n_events=(self._calibration_metadata or {}).get("n_samples"),
                brier_before=(self._calibration_metadata or {}).get("brier_before"),
                brier_after=(self._calibration_metadata or {}).get("brier_after"),
            ),
            diagnostics=DiagnosticsBlock(
                cycle_ms=0.0,  # filled in by caller (it knows the wall-clock total)
                fetch_ms=fetch_ms,
                compute_ms=compute_ms,
                render_ms=render_ms,
                ensemble_ms=ensemble.ensemble_ms if ensemble is not None else 0.0,
                national_ms=national_ms,
                artifact_bytes=artifact_bytes,
            ),
        )
        # Cross-cycle memory for the no-new-frame fast path: remember which
        # radar frame this state was computed from, and the state itself.
        # Set only on a fully successful compute, so a failed cycle can
        # never park a half-built state behind the fast path.
        self._last_frame_ts = composite_now.timestamp_utc
        self._last_state = state
        return state

    def _emit_unchanged_state(self, prev: State, fetch_ms: float) -> State:
        """The no-new-frame fast path's output (Phase B addendum).

        The previous good state verbatim, with only the clock-derived
        fields refreshed: ``generated_at`` (now) and
        ``radar.data_age_minutes`` (the unchanged frame has aged). Every
        cross-cycle state machine is left untouched — the rain_incoming
        streak, the raining_now hysteresis, ``_national_latest``, the
        rendered frames and national artifacts on disk all still describe
        the same radar observation, so the cycle behaves exactly as if it
        hadn't fired. ``diagnostics.compute_ms == 0.0`` (with
        ensemble/render/national all zeroed) is the distinguishable
        skipped-cycle marker, alongside the ``cycle_skipped_no_new_frame``
        log event.
        """
        now_utc = datetime.now(timezone.utc)
        frame_age_min = max(
            0.0,
            (now_utc - prev.radar.latest_ts).total_seconds() / 60.0,
        )
        _log.info(
            "cycle_skipped_no_new_frame",
            radar_ts=prev.radar.latest_ts.isoformat(),
            data_age_minutes=round(frame_age_min, 2),
            fetch_ms=round(fetch_ms, 1),
        )
        return prev.model_copy(
            update={
                "generated_at": now_utc,
                "radar": prev.radar.model_copy(
                    update={"data_age_minutes": round(frame_age_min, 2)},
                ),
                "diagnostics": DiagnosticsBlock(
                    cycle_ms=0.0,  # filled in by caller, as on the full path
                    fetch_ms=fetch_ms,
                    compute_ms=0.0,  # the skipped-cycle marker
                    render_ms=0.0,
                ),
            },
        )

    def _run_steps_ensemble(
        self,
        composites: list[RadarComposite],
        vy: np.ndarray,
        vx: np.ndarray,
        geo: CompositeGeo,
        *,
        frame_age_min: float,
        dt_min: float,
        points: Sequence[Any] | None = None,
    ) -> EnsembleOutcome | None:
        """Run STEPS and reduce it at home; None means "fall back" (plan §A0).

        Called from ``_compute_sync``, i.e. already inside the cycle's
        worker thread — the async-discipline contract holds. Every
        failure mode (disabled, < 3 frames, ``EnsembleUnavailable``, any
        exception from the vendored pysteps) logs a warning and returns
        None so the cycle emits exactly the deterministic-only state, with
        the additive ensemble fields left at their None/0 defaults.

        Lead-time bookkeeping: the ensemble timesteps count from
        radar-frame time while ``state.json`` leads are minutes from now,
        so timestep selection uses ``frame_age_corrected_leads`` and the
        results are reported under the nominal lead labels — mirroring the
        deterministic loop's ``lead + frame_age_min``.
        """
        steps_cfg = self.config.forecast.steps
        if not steps_cfg.enabled:
            return None
        if len(composites) < 3:
            _log.info(
                "ensemble_skipped",
                reason="insufficient_frames",
                n_frames=len(composites),
            )
            return None
        composite_now = composites[-1]

        # The STEPS timestep follows the measured frame spacing (Phase B
        # addendum: fullRange-only frames arrive every ~10 min, and STEPS'
        # AR(2) model assumes the forecast timestep equals the input frame
        # spacing). The horizon is fixed by config and measured from
        # radar-frame time, so the step count adapts:
        # ceil(horizon_min / timestep) — 9 steps at the 10-min cadence for
        # the default 90 min horizon. That 90 is 60 min of served lead plus
        # the frame age at compute; at a 60 min horizon the honest horizon
        # from now was only ~43 min and the last timesteps collapsed
        # P(<=45)/P(<=60) onto one another.
        # ``dt_min`` arrives sanitised (> 0) from ``_compute_sync``.
        timestep_min = float(dt_min)
        horizon_min = float(steps_cfg.horizon_min)
        n_timesteps = max(
            1, math.ceil(horizon_min / timestep_min - 1e-9),
        )

        # ``run_ensemble`` wants velocity in pixels per STEPS timestep; the
        # Farnebäck flow is pixels per inter-frame interval (``dt_min``).
        # With the timestep derived from ``dt_min`` this rescale is the
        # identity; it is kept generalised so the unit contract survives
        # any future decoupling of timestep from frame spacing.
        if abs(dt_min - timestep_min) > 1e-6:
            scale = np.float32(timestep_min / dt_min)
            vy = vy * scale
            vx = vx * scale

        t_ens = time.perf_counter()
        try:
            forecast = run_ensemble(
                [c.reflectivity_dbz for c in composites[-3:]],
                vy, vx,
                zr_a=composite_now.zr_a,
                zr_b=composite_now.zr_b,
                n_timesteps=n_timesteps,
                timestep_min=timestep_min,
                n_ens_members=steps_cfg.ensemble_size,
                n_cascade_levels=steps_cfg.n_cascade_levels,
                threshold_mm_h=self._rain_threshold,
                downsample_factor=steps_cfg.downsample_factor,
                pixel_scale_m=float(composite_now.xscale_m),
            )
        except Exception as exc:  # noqa: BLE001 — includes EnsembleUnavailable
            _log.warning(
                "ensemble_failed",
                error=str(exc),
                elapsed_ms=round((time.perf_counter() - t_ens) * 1000, 1),
            )
            return None
        ensemble_ms = (time.perf_counter() - t_ens) * 1000

        nominal_leads = [int(lead) for lead in self.config.forecast.leads_min]
        corrected_leads = frame_age_corrected_leads(
            nominal_leads,
            frame_age_min,
            n_timesteps=n_timesteps,
            timestep_min=timestep_min,
        )
        national: NationalProducts | None = None
        national_ms = 0.0
        point_products: PointProducts | None = None
        try:
            home = aggregate_at_home(
                forecast,
                geo,
                self.config.home.lon,
                self.config.home.lat,
                radius_m=self.config.home.radius_km * 1000.0,
                threshold_mm_h=self._rain_threshold,
                timestep_min=timestep_min,
                leads_min=corrected_leads,
                downsample_factor=steps_cfg.downsample_factor,
            )
            # National ×4 product grids (plan §A1) — same ensemble, same
            # threshold, same frame-age convention as the home reduction, so
            # the §A4 agreement at the home pixel holds by construction. A
            # failure here degrades the website products only; the home
            # forecast above is already safe.
            nat_cfg = self.config.forecast.national
            if nat_cfg.enabled:
                t_nat = time.perf_counter()
                try:
                    # §B4: the calibrated grid REPLACES the raw one — one
                    # grid set served (raw is recoverable by inverting the
                    # published breakpoints). ``national`` is still
                    # assigned by exactly one expression, so a calibration
                    # failure leaves it None (a products failure) rather
                    # than half-calibrated grids the manifest's metadata
                    # would then misdescribe. One np.interp per lead grid —
                    # O(grid), inside the existing national timing.
                    raw_national = national_products(
                        forecast,
                        leads_min=nat_cfg.leads_min,
                        threshold_mm_h=self._rain_threshold,
                        timestep_min=timestep_min,
                        frame_age_min=frame_age_min,
                        downsample_factor=steps_cfg.downsample_factor,
                    )
                    # H-P: the UNcalibrated fractions at the served points,
                    # read before the line below replaces the grids. The
                    # model's main predictor is the raw fraction and its
                    # baseline is the calibrated one, so both have to
                    # survive the swap — the same reason the replay keeps
                    # ``raw_p_rain`` across it.
                    point_products = _read_points(raw_national, points)
                    national = self._calibrate_national(raw_national)
                    del raw_national
                except Exception as exc:  # noqa: BLE001
                    _log.warning("national_products_failed", error=str(exc))
                national_ms = (time.perf_counter() - t_nat) * 1000
        except Exception as exc:  # noqa: BLE001
            _log.warning("ensemble_aggregate_failed", error=str(exc))
            return None
        finally:
            # Memory hygiene: the array is
            # ``n_ens_members × n_timesteps × 432 × 496`` float32, and
            # n_timesteps now follows ``horizon_min`` — ~154 MB at 24
            # members × 9 steps (~103 MB at the old 6). Drop it as soon as
            # the home + national reductions are done: the retained
            # products are ~7 MB of derived grids, never the raw ensemble.
            del forecast

        # ETA quantiles are minutes from radar-frame time → shift to minutes
        # from now (same frame-age convention as the leads above).
        if math.isfinite(home.eta_p25_min) and math.isfinite(home.eta_p75_min):
            eta_window = (
                max(0.0, home.eta_p25_min - frame_age_min),
                max(0.0, home.eta_p75_min - frame_age_min),
            )
        else:
            eta_window = None

        # Report under the nominal labels (order preserved by aggregate_at_home).
        p_by_lead = {
            lead: float(p)
            for lead, p in zip(nominal_leads, home.probability_by_lead)
        }
        # §B4: the pooled national curves also calibrate the home
        # ``p_ensemble`` (one curve set, one truth — the home point's rows
        # are in the pool). Leads without a curve stay raw — never
        # interpolated between neighbouring leads' curves — and the exact
        # calibrated subset is reported so the flags can't lie. Same
        # float32 arithmetic as the grid path, so the §A4 home-pixel
        # agreement survives calibration by construction.
        calibrated_leads: tuple[int, ...] | None = None
        if self._national_curves:
            done: list[int] = []
            for lead in nominal_leads:
                curve = self._national_curves.get(lead)
                if curve is not None:
                    p_by_lead[lead] = float(curve.predict(p_by_lead[lead]))
                    done.append(lead)
            # Same guard as the grids: "rain within L" must never be less
            # likely than "rain within a shorter L" (see _calibrate_national).
            p_by_lead.update(
                enforce_lead_monotonic({lead: p_by_lead[lead] for lead in done})
            )
            calibrated_leads = tuple(done)
        _log.info(
            "ensemble_ok",
            n_members=home.n_members,
            ensemble_ms=round(ensemble_ms, 1),
            eta_window_min=eta_window,
        )
        return EnsembleOutcome(
            p_by_lead=p_by_lead,
            eta_window_min=eta_window,
            n_members=home.n_members,
            ensemble_ms=ensemble_ms,
            national=national,
            national_ms=national_ms,
            calibrated_leads=calibrated_leads,
            points=point_products if national is not None else None,
        )

    # -- post-processing (Phase H, H-P) -------------------------------------

    def _serving_points(self) -> list[tuple[float, float]]:
        """Every point this cycle should score, deduplicated, home first.

        Runs inside the cycle worker, so a source may block on SQLite or
        on a points file. A source that raises is skipped with one log
        line: a broken subscription store must cost the post-processed
        probability, never the nowcast.

        Two points that round to the same coordinate are one point — the
        rounding is far finer than the 500 m pixel, so this only ever
        merges rows that would have read the same pixel anyway, and it is
        what keeps a gauge station and a subscriber at the same address
        from being computed twice.
        """
        home = point_key(self.config.home.lat, self.config.home.lon)
        keys: list[tuple[float, float]] = [home]
        seen = {home}
        for name, provider in self._point_sources:
            try:
                points = list(provider())
            except Exception as exc:  # noqa: BLE001 — one bad source only
                _log.warning(
                    "postprocess_points_failed", source=name, error=str(exc),
                )
                continue
            for lat, lon in points:
                try:
                    key = point_key(lat, lon)
                except (TypeError, ValueError):
                    continue
                if key in seen:
                    continue
                seen.add(key)
                keys.append(key)
                if len(keys) >= _MAX_POSTPROCESS_POINTS:
                    _log.warning(
                        "postprocess_points_truncated",
                        limit=_MAX_POSTPROCESS_POINTS,
                    )
                    return keys
        return keys

    def _station_radar_km(self, lat: float, lon: float) -> float:
        """Great-circle km to the nearest DMI radar, memoised per point."""
        key = point_key(lat, lon)
        hit = self._radar_km.get(key)
        if hit is None:
            hit = float(nearest_radar_km(float(lat), float(lon)))
            self._radar_km[key] = hit
        return hit

    def _publish_postprocess(
        self,
        *,
        keys: Sequence[tuple[float, float]],
        grid_features: dict[str, np.ndarray] | None,
        points: PointProducts | None,
        products: NationalProducts,
        observed_grid: np.ndarray | None,
        radar_ts_utc: datetime,
        generated_at_utc: datetime,
        frame_age_min: float,
    ) -> None:
        """Assemble and score this cycle's feature rows; publish the result.

        Best-effort by construction: every failure mode leaves
        ``postprocess_latest`` at the previous cycle's object, whose
        ``radar_ts_utc`` no longer matches the frame — which is exactly
        how every consumer already decides not to use it.

        The three decision columns the design also reads
        (``observed_mm_h``, ``eta_min``, ``intensity_mm_h``) are sampled
        here and handed to the model, but are NOT written into the feature
        row: the decision schema already carries them, and one column has
        one writer.
        """
        if not keys or grid_features is None or points is None:
            return
        try:
            shared: list[dict[str, Any]] = []
            for pixel in points.pixels:
                if pixel is None:
                    shared.append(
                        {"observed_mm_h": None, "eta_min": None,
                         "intensity_mm_h": None},
                    )
                    continue
                row, col = pixel
                observed = None
                if (
                    observed_grid is not None
                    and observed_grid.shape == products.eta_min.shape
                ):
                    observed = finite_or_none(observed_grid[row, col])
                shared.append({
                    "observed_mm_h": observed,
                    "eta_min": finite_or_none(products.eta_min[row, col]),
                    "intensity_mm_h": finite_or_none(
                        products.intensity_mm_h[row, col],
                    ),
                })
            self._postprocess_latest = build_cycle_postprocess(
                self._postprocess,
                radar_ts_utc=radar_ts_utc,
                generated_at_utc=generated_at_utc,
                keys=keys,
                grid_features=grid_features,
                raw_fractions=points.raw_fractions,
                shared=shared,
                station_radar_km=[
                    self._station_radar_km(lat, lon) for lat, lon in keys
                ],
                leads=products.leads_min,
                season=core_postprocess.season_of_month(generated_at_utc.month),
                hour_utc=generated_at_utc.hour,
                frame_age_min=frame_age_min,
            )
            _log.info(
                "postprocess_cycle",
                points=len(keys),
                active=self._postprocess_latest.active,
                leads=list(self._postprocess_latest.leads),
                fitted_at=self._postprocess_latest.fitted_at_utc,
            )
        except Exception as exc:  # noqa: BLE001 — see the docstring
            _log.warning("postprocess_cycle_failed", error=str(exc))

    def _calibrate_national(self, products: NationalProducts | None) -> NationalProducts | None:
        """Map each lead's ``p_rain`` grid through that lead's national curve (§B4).

        Vectorised piecewise-linear interpolation over the breakpoints
        (``IsotonicCalibrator.predict`` → one ``np.interp`` per grid; NaN
        passes through). Leads with no curve keep their RAW grid — never
        interpolated between leads' curves. No-op (same object back) when no
        national curves are loaded or ``products`` is None.

        The calibrated grids then go through
        :func:`~dmi_nowcast_core.national.enforce_lead_monotonic`, because
        the curves are fitted per lead with nothing tying them together and
        the served claim "rain within L" must never be less likely than the
        window it contains. Uncalibrated leads are deliberately left out of
        that pass: their raw grids are not on the calibrated scale, so
        lifting one against the other would compare two different things.
        """
        if products is None or not self._national_curves:
            return products
        p_rain: dict[int, np.ndarray] = {}
        calibrated: dict[int, np.ndarray] = {}
        for lead in products.leads_min:
            grid = products.p_rain[int(lead)]
            curve = self._national_curves.get(int(lead))
            if curve is None:
                p_rain[int(lead)] = grid
            else:
                p_rain[int(lead)] = calibrated[int(lead)] = curve.predict(grid)
        # ``p_rain[L]`` is P(rain WITHIN L), so longer leads contain shorter
        # ones; per-lead curves are fitted independently and can invert that.
        p_rain.update(enforce_lead_monotonic(calibrated))
        return replace(products, p_rain=p_rain)

    def _national_calibration_manifest(self, products: NationalProducts) -> dict | None:
        """The manifest's ``calibration`` block for one cycle's grids (§B4).

        None when the served grids are raw (no curves loaded, or none
        covering a served lead) — the manifest then carries
        ``"calibration": null``. Otherwise: the curve file's ``fitted_at``,
        its metadata echo (n_samples / brier if present), and the exact
        leads whose grids were calibrated.
        """
        if not self._national_curves:
            return None
        cal_leads = [
            int(lead) for lead in products.leads_min
            if int(lead) in self._national_curves
        ]
        if not cal_leads:
            return None
        meta = self._national_calibration_metadata or {}
        block: dict[str, Any] = {
            "fitted_at": meta.get("fitted_at"),
            "calibrated_leads": cal_leads,
        }
        for key in ("n_samples", "n_events", "n_points", "brier_before", "brier_after"):
            if key in meta:
                block[key] = meta[key]
        return block

    def _ensure_basemap(self) -> None:
        """Lazy-load the OSM basemap on first cycle; cached on disk after."""
        if self._basemap is not None or self._basemap_attempted:
            return
        self._basemap_attempted = True
        try:
            self._basemap = build_basemap(
                home_lat=self.config.home.lat,
                home_lon=self.config.home.lon,
                zoom_km=100.0,
                output_px=(500, 500),
                cache_dir=self._basemap_dir,
            )
            _log.info("basemap_loaded", cached=(self._basemap is not None))
        except Exception as exc:  # noqa: BLE001
            _log.warning("basemap_load_failed", error=str(exc))


def _build_now_subline(
    *,
    stats_now,
    eta_minutes: float | None,
    peak_rate: float,
    peak_lead: int,
) -> str:
    """Human-readable subline on the 'now' frame.

    Mirrors the integration's overlay caption: current intensity, ETA, peak
    forecast. Kept short so it fits the overlay band.
    """
    parts: list[str] = []
    if np.isfinite(stats_now.max_mm_h) and stats_now.max_mm_h > 0.0:
        parts.append(f"now {stats_now.max_mm_h:.1f} mm/h")
    else:
        parts.append("now dry")
    if eta_minutes is not None:
        parts.append(f"ETA +{int(eta_minutes)} min")
    if peak_rate > 0.0:
        parts.append(f"peak {peak_rate:.1f} mm/h at +{peak_lead}")
    return "  ·  ".join(parts)


def _disc_motion(
    rain_now: np.ndarray,
    vy: np.ndarray,
    vx: np.ndarray,
    geo: CompositeGeo,
    composite: RadarComposite,
    lon: float,
    lat: float,
    dt_min: float,
) -> tuple[float, float]:
    """Rain-weighted mean motion within 120 km of home, normalised to /min."""
    home = geo.lonlat_to_grid(lon, lat)
    hr, hc = int(round(home.row)), int(round(home.col))
    search_px = int(round(120_000 / composite.xscale_m))
    rs = slice(max(0, hr - search_px), min(rain_now.shape[0], hr + search_px + 1))
    cs = slice(max(0, hc - search_px), min(rain_now.shape[1], hc + search_px + 1))
    lr, lvy, lvx = rain_now[rs, cs], vy[rs, cs], vx[rs, cs]
    weights = np.where(
        np.isfinite(lr) & (lr > 0.1) & np.isfinite(lvy) & np.isfinite(lvx),
        lr, 0.0,
    )
    w_sum = float(weights.sum())
    if w_sum > 0:
        dvy = float((lvy * weights).sum() / w_sum) / max(dt_min, 1.0)
        dvx = float((lvx * weights).sum() / w_sum) / max(dt_min, 1.0)
    else:
        dvy = dvx = 0.0
    return dvy, dvx


def _read_points(
    products: NationalProducts, points: Sequence[Any] | None,
) -> PointProducts | None:
    """Per-point reads of the RAW national grids (H-P).

    ``points`` are the fractional NATIVE indices the cycle already
    projected for the feature extraction; the product pixel comes off them
    through ``national_sample.product_pixel_of``, which is the same
    arithmetic ``/forecast``, the push fan-out and the browser sampler use.
    A point off the product grid contributes ``None`` at every lead —
    unknown, never 0 %.
    """
    if not points:
        return None
    pixels = tuple(
        product_pixel_of(products, idx.row, idx.col) for idx in points
    )
    raw: dict[int, tuple[float | None, ...]] = {}
    for lead in products.leads_min:
        grid = products.p_rain[int(lead)]
        raw[int(lead)] = tuple(
            None if pixel is None else finite_or_none(grid[pixel[0], pixel[1]])
            for pixel in pixels
        )
    return PointProducts(pixels=pixels, raw_fractions=raw)


def _bearing_from_deg(dy_per_min: float, dx_per_min: float) -> float:
    """Compass bearing the rain is coming FROM (0° = from north)."""
    # Motion vector (dy,dx) is image-space (dy positive = south). The
    # "from" direction is the opposite of motion.
    if dy_per_min == 0 and dx_per_min == 0:
        return 0.0
    angle = math.degrees(math.atan2(-dx_per_min, dy_per_min))
    # atan2(-dx, dy) returns 0=south-bound, 90=west-bound, etc. Convert
    # to compass-from convention: 0=from north.
    return (angle + 180.0) % 360.0


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    """Write ``data`` to ``target`` atomically (tempfile in same dir → replace)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(target)


def _file_stamp(path: Path) -> tuple[int, int] | None:
    """``(mtime_ns, size)`` of a file, or None when it does not exist.

    Both halves, not just mtime: a filesystem with coarse timestamps and
    a same-second rewrite would otherwise look unchanged.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None
