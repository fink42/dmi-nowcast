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
from typing import Any, Iterable

import numpy as np
import structlog

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
    complete_flow,
    dense_flow,
)
from dmi_nowcast_core.fetch import AsyncDMIClient, RadarFeature
from dmi_nowcast_core.geo import CompositeGeo
from dmi_nowcast_core.motion import phase_correlation_shift
from dmi_nowcast_core.national import (
    NationalProducts,
    motion_grids_kmh,
    national_products,
)
from dmi_nowcast_core.parse import RadarComposite, parse_composite
from dmi_nowcast_core.probabilistic import (
    aggregate_at_home,
    frame_age_corrected_leads,
    run_ensemble,
)
from dmi_nowcast_core.raining_now import RainingNow, RainingNowConfig
from dmi_nowcast_core.sample import sample_disc
from dmi_nowcast_core.transform import dbz_to_rain_rate

from .config import Config
from .eta_smoother import EtaSmoother
from .lightning_tracker import LightningTracker
from .national_artifacts import write_national_artifacts
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

_log = structlog.get_logger(__name__)

# Per-pixel motion clip (in px per frame). Same value the integration uses;
# guards against optical-flow noise spikes that would otherwise blow up
# ``advect_field``. 30 px at 500 m/pixel = 15 km per inter-frame interval,
# i.e. ~180 km/h cell motion — generous upper bound for real weather.
_MAX_PX_PER_FRAME = 30.0

# 8-point compass for the motion arrow caption.
_COMPASS_LABELS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

# STEPS ensemble horizon in minutes from radar-frame time, matching the
# default deterministic leads_min ceiling (website Phase A plan §A0). The
# timestep itself is DERIVED per cycle from the measured inter-frame spacing
# (Phase B addendum 2026-08-29: fullRange-only frames arrive every ~10 min),
# with n_timesteps = ceil(horizon / timestep) keeping the horizon fixed.
_ENSEMBLE_HORIZON_MIN = 60.0

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
    are explicitly meant to run in a thread via ``asyncio.to_thread``).
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
        # Latest national products + their radar timestamp, held in memory
        # for the /forecast point lookup (plan §A3). Swapped as one tuple so
        # readers on other threads never see a torn pair.
        self._national_latest: tuple[NationalProducts, datetime] | None = None
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
    def national_latest(self) -> tuple[NationalProducts, datetime] | None:
        """Latest national products + their radar timestamp (plan §A3).

        None until the first successful ensemble cycle with national
        products enabled. The /forecast endpoint samples these grids."""
        return self._national_latest

    @property
    def national_curve_leads(self) -> frozenset[int]:
        """Leads covered by the loaded national calibration curves (§B4).

        Static for the process lifetime (curves load once at init, before
        any cycle), so the products held in ``national_latest`` were
        calibrated with exactly these curves — /forecast derives its
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

    async def run_cycle(self) -> CycleResult:
        """Execute one cycle. Network in this coroutine; compute in a thread."""
        t0 = time.perf_counter()
        fetch_ms = compute_ms = 0.0
        try:
            t_fetch = time.perf_counter()
            paths = await self._fetch_latest_frames()
            fetch_ms = (time.perf_counter() - t_fetch) * 1000
            if len(paths) < 2:
                raise RuntimeError(f"not enough frames available (got {len(paths)})")

            t_compute = time.perf_counter()
            state = await asyncio.to_thread(self._compute_sync, paths, fetch_ms)
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
        """Blocking compute path — call from inside ``asyncio.to_thread``."""
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
        try:
            if self.config.forecast.method == "mean-motion":
                raise DenseFlowUnavailable("forced via config")
            vy, vx = dense_flow(
                composite_prev.reflectivity_dbz,
                composite_now.reflectivity_dbz,
            )
        except DenseFlowUnavailable:
            method_used = "mean-motion"
            dy, dx = phase_correlation_shift(rain_prev, rain_now)
            shape = rain_now.shape
            vy = np.full(shape, dy, dtype=np.float32)
            vx = np.full(shape, dx, dtype=np.float32)

        # Motion-field completion (R5). Farnebäck returns exactly zero away
        # from the echo, which stalls advected rain along a stationary line
        # ~20-30 km ahead of it; relax the far field toward bulk storm
        # motion before anything consumes the flow. Deliberately ahead of
        # the sanitise/clip below so BOTH consumers get the completed field:
        # the deterministic overlay advection here, and the STEPS velocity
        # (``_run_steps_ensemble`` downsamples this same array).
        vy, vx = complete_flow(
            vy, vx, rain_now,
            pixel_km=float(composite_now.xscale_m) / 1000.0,
            support_threshold_mm_h=self._rain_threshold,
        )

        # Sanitize motion.
        vy = np.nan_to_num(vy, nan=0.0).astype(np.float32)
        vx = np.nan_to_num(vx, nan=0.0).astype(np.float32)
        np.clip(vy, -_MAX_PX_PER_FRAME, _MAX_PX_PER_FRAME, out=vy)
        np.clip(vx, -_MAX_PX_PER_FRAME, _MAX_PX_PER_FRAME, out=vx)

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
        advected = advect_field_series(
            rain_now, vy, vx,
            horizons_minutes=[lead + frame_age_min for lead in self.config.forecast.leads_min],
            dt_minutes=dt_min,
        )
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
            self._national_latest = (ensemble.national, composite_now.timestamp_utc)
            t_art = time.perf_counter()
            # R2 cell-motion grids: the same completed+sanitised flow the
            # overlays and STEPS ran on, on the product grid, in km/h. A
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
                    generated_at_utc=datetime.now(timezone.utc),
                    overlay_fields_mm_h=overlay_fields,
                    out_dir=self._national_dir,
                    keep_cycles=self.config.forecast.national.keep_cycles,
                    motion_east_kmh=motion_east,
                    motion_north_kmh=motion_north,
                    # §B4: null when the served grids are raw; otherwise
                    # fitted_at + curve-file echo + calibrated_leads.
                    calibration=self._national_calibration_manifest(ensemble.national),
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
    ) -> EnsembleOutcome | None:
        """Run STEPS and reduce it at home; None means "fall back" (plan §A0).

        Called from ``_compute_sync``, i.e. already inside
        ``asyncio.to_thread`` — the async-discipline contract holds. Every
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
        # spacing). The horizon stays fixed at 60 min, so the step count
        # adapts: ceil(60 / timestep) — 6 steps at the 10-min cadence.
        # ``dt_min`` arrives sanitised (> 0) from ``_compute_sync``.
        timestep_min = float(dt_min)
        n_timesteps = max(
            1, math.ceil(_ENSEMBLE_HORIZON_MIN / timestep_min - 1e-9),
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
                    # published breakpoints). Composed in one assignment so
                    # a calibration failure leaves ``national`` None (a
                    # products failure), never half-calibrated grids that
                    # the manifest's metadata would then misdescribe. One
                    # np.interp per lead grid — O(grid), inside the
                    # existing national timing.
                    national = self._calibrate_national(national_products(
                        forecast,
                        leads_min=nat_cfg.leads_min,
                        threshold_mm_h=self._rain_threshold,
                        timestep_min=timestep_min,
                        frame_age_min=frame_age_min,
                        downsample_factor=steps_cfg.downsample_factor,
                    ))
                except Exception as exc:  # noqa: BLE001
                    _log.warning("national_products_failed", error=str(exc))
                national_ms = (time.perf_counter() - t_nat) * 1000
        except Exception as exc:  # noqa: BLE001
            _log.warning("ensemble_aggregate_failed", error=str(exc))
            return None
        finally:
            # Memory hygiene: ~205 MB float32 at 24 × 12 × 432 × 496. Drop it
            # as soon as the home + national reductions are done — the
            # retained products are ~7 MB of derived grids, never the raw
            # ensemble.
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
        )

    def _calibrate_national(self, products: NationalProducts | None) -> NationalProducts | None:
        """Map each lead's ``p_rain`` grid through that lead's national curve (§B4).

        Vectorised piecewise-linear interpolation over the breakpoints
        (``IsotonicCalibrator.predict`` → one ``np.interp`` per grid; NaN
        passes through). Leads with no curve keep their RAW grid — never
        interpolated between leads' curves. No-op (same object back) when no
        national curves are loaded or ``products`` is None.
        """
        if products is None or not self._national_curves:
            return products
        p_rain: dict[int, np.ndarray] = {}
        for lead in products.leads_min:
            grid = products.p_rain[int(lead)]
            curve = self._national_curves.get(int(lead))
            p_rain[int(lead)] = grid if curve is None else curve.predict(grid)
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


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None
