"""Pulling published artifacts from the private instance (Phase F, F4).

Files the public instance serves, or reads, but cannot produce:

``nowcast/quality.json``
    Built nightly from the corpora, the warning replay and the live gauge
    scoreboard — all of which live on the corpus volume the private
    instance owns.
``calibration/national_curves.json``
    Fitted monthly from the same archive. Copying it here is what makes
    the public instance's probabilities calibrated instead of raw
    ensemble fractions.
``calibration/push_thresholds.json``
    Fitted nightly from the decision rows and the gauge store (Phase G).
    Copying it here is what makes the public instance's notifications
    warn at the measured threshold for each horizon instead of the
    shipped fallback.
``calibration/postprocess.json``
    The gauge-trained post-processing model, refit nightly beside the
    thresholds (Phase H, H-P). The public instance runs its own cycle and
    its own push engine but can never fit this, so it only ever serves it.
    Without the copy its notifications fall back to the curve-calibrated
    probability — a working service, with the worse number.
``calibration/postprocess_push.json``
    The push-only onset-target model (S11), installed by hand on the
    private instance. Without it this instance's push rule falls back to
    the single threshold on every lead.
``stations/station_points.json``
    The version-2 catalogue of DMI's rain gauges, built on the private
    instance (``scripts/build_station_points.py``), and the one input the
    public instance's ``ng_*`` features cannot derive: a station id, and
    the coordinate that turns it into a place. NOT in the default
    ``sync.files`` — it is only wanted where ``postprocess`` reads
    neighbour gauges out of a local store (v2, S5), which is the public
    stack's example config and nothing else. Hourly is ample: the file
    names DMI's gauge network and changes about never, and the READINGS
    are not synced at all — that instance polls metObs itself, because the
    features read them at a ten-minute horizon.

These are small, static-per-cycle documents on a network only these two
containers share, so the transport is deliberately dull: one conditional
GET per file per ``interval_min``, ``If-None-Match`` against the ETag from
last time, and a content hash as the fallback when the source sends no
ETag.

The failure policy is the whole design. **Last good wins**: a refused
connection, a 500, a body that is not the JSON it claims to be, a body
past ``max_bytes`` — every one of them leaves the file already on disk
untouched and logs one line. A public instance whose private peer is down
keeps serving yesterday's report; the document carries its own
``generated_at_utc`` and the page shows how old it is. The alternative,
truncating or blanking a served file because a fetch failed, turns one
instance's outage into the other's.

**Validated before it is swapped in.** "Valid JSON" is not "a file the
reader can use": a curve file with no curves, a thresholds document of the
wrong schema, a post-processing model fitted to the other target, all
parse. So each body is written to a temporary file (fsynced), checked with
the loader the consumer itself uses — ``load_calibration_curves`` for the
curves, ``validate_thresholds`` for the thresholds, the
``PostprocessTable`` checks (structure + the slot's target: ``wet`` for
``postprocess.json``, ``onset`` for ``postprocess_push.json``) for the two
models, a minimal shape for the catalogue and the quality report — and
only then renamed over the target. The replaced file is kept beside it as
``<name>.prev`` (hard link, so there is never an instant with no file).
A rejected body leaves the file in service untouched; its ETag is
remembered and sent back, so the same bad body is not re-downloaded every
interval. Hashing, parsing and writing all run in ``asyncio.to_thread`` —
the model is ~7 MB and takes seconds to parse.

The private routes answer ``If-None-Match`` with ``304`` (``app.py``), so
an unchanged file costs one round trip, not a download.

Async discipline: httpx async for the fetch, ``asyncio.to_thread`` for
every disk write, its own ``AsyncIOScheduler`` so a slow private instance
can never delay a radar cycle.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import httpx
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from dmi_nowcast_core.calibrate import load_calibration_curves
from dmi_nowcast_core.postprocess import TARGET_ONSET, TARGET_WET
from dmi_nowcast_core.push_thresholds import validate_thresholds

from .config import Config
from .gauge_history import resolved_gauge_points_path
from .push.paths import (
    resolved_onset_model_path,
    resolved_postprocess_path,
    resolved_thresholds_path,
)

_log = structlog.get_logger(__name__)

#: Spread the poll off the exact minute boundary, as every other task does.
JITTER_SEC = 30

#: The files that do NOT land under ``data_dir`` by their relative path:
#: the engine reads its curves from ``calibration.national_curves_path``,
#: the push service reads its thresholds from ``push.thresholds_path``, the
#: cycle reads its post-processing model from ``push.postprocess_path`` and
#: its gauge catalogue from ``postprocess.gauge_points_file``, wherever the
#: operator put them.
CURVES_FILE = "calibration/national_curves.json"
THRESHOLDS_FILE = "calibration/push_thresholds.json"
POSTPROCESS_FILE = "calibration/postprocess.json"
ONSET_MODEL_FILE = "calibration/postprocess_push.json"
STATION_POINTS_FILE = "stations/station_points.json"
QUALITY_FILE = "nowcast/quality.json"


def target_path(config: Config, name: str) -> Path:
    """Where a synced file lands on this instance.

    Everything is ``<storage.data_dir>/<name>`` except the fitted files and
    the gauge catalogue, which go to the paths the engine, the push service
    and the cycle actually read. Getting this wrong is silent — the file
    appears, nothing loads it — so it lives in one function with one test.

    The catalogue is the one entry whose special case is conditional: with
    no ``postprocess.gauge_points_file`` configured there is nothing
    reading it, and the generic rule puts it at
    ``<data_dir>/stations/station_points.json``, which is where the public
    example points that key anyway.
    """
    if name == CURVES_FILE:
        return Path(config.calibration.national_curves_path)
    if name == THRESHOLDS_FILE:
        return resolved_thresholds_path(config)
    if name == POSTPROCESS_FILE:
        return resolved_postprocess_path(config)
    if name == ONSET_MODEL_FILE:
        return resolved_onset_model_path(config)
    if name == STATION_POINTS_FILE:
        configured = resolved_gauge_points_path(config)
        if configured is not None:
            return configured
    return Path(config.storage.data_dir).joinpath(*PurePosixPath(name).parts)


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def validation_problem(name: str, path: Path) -> str | None:
    """Why the file at ``path`` must not be installed as ``name``, or None.

    Uses the consumer's own loader wherever there is one. ``path`` is the
    candidate (a temporary file), never the file in service.
    """
    if not name.endswith(".json"):
        return None
    try:
        text = path.read_text(encoding="utf-8")
        doc = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return f"not valid JSON: {exc}"
    if name == CURVES_FILE:
        curves_block = doc.get("curves") if isinstance(doc, dict) else None
        if not isinstance(curves_block, dict) or not curves_block:
            return "rejected: not a calibration-curve document (no curves)"
        try:
            curves = load_calibration_curves(path)
        except Exception as exc:  # noqa: BLE001 — every way a curve is junk
            return f"rejected: curves do not load: {type(exc).__name__}: {exc}"
        if len(curves) != len(curves_block):
            return "rejected: curves do not load"
        return None
    if name == THRESHOLDS_FILE:
        problems = validate_thresholds(doc)
        if problems:
            return "rejected: " + "; ".join(problems[:3])
        return None
    if name in (POSTPROCESS_FILE, ONSET_MODEL_FILE):
        # Imported here: the push package pulls numpy-heavy model code that
        # an instance syncing only the quality report never needs.
        from .push.postprocess import document_problem

        target = TARGET_WET if name == POSTPROCESS_FILE else TARGET_ONSET
        problem = document_problem(text, target)
        return None if problem is None else f"rejected: {problem}"
    if name == STATION_POINTS_FILE:
        points = doc.get("points") if isinstance(doc, dict) else None
        if not isinstance(points, list) or not points:
            return "rejected: no points"
        try:
            for entry in points:
                str(entry["id"])
                float(entry["lat"])
                float(entry["lon"])
        except (KeyError, TypeError, ValueError) as exc:
            return f"rejected: malformed point: {type(exc).__name__}"
        return None
    if name == QUALITY_FILE:
        if not isinstance(doc, dict) or "schema_version" not in doc:
            return "rejected: not a quality report (no schema_version)"
        return None
    return None


def _keep_previous(path: Path) -> None:
    """``path`` → ``path.prev``, leaving ``path`` itself in place."""
    prev = path.with_name(path.name + ".prev")
    staging = path.with_name(f".{path.name}.prev.tmp")
    try:
        if staging.exists():
            staging.unlink()
        os.link(path, staging)
        os.replace(staging, prev)
    except OSError:
        shutil.copy2(path, prev)


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def install_file(
    path: Path, name: str, body: bytes, known_digest: str | None,
) -> tuple[str, str | None, str]:
    """Validate ``body`` and swap it in. ``(status, error, sha256)``. Blocking.

    ``status`` is ``"unchanged"`` (same bytes as the file in service),
    ``"updated"`` or ``"failed"`` (``error`` says why; nothing touched).
    Order: hash → tmp write + fsync → validate the tmp with the real
    loader → hard-link the old file to ``.prev`` → rename → fsync the dir.
    """
    digest = hashlib.sha256(body).hexdigest()
    current = known_digest if known_digest is not None else _sha256_file(path)
    if digest == current and path.is_file():
        return "unchanged", None, digest
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        problem = validation_problem(name, Path(tmp))
        if problem is not None:
            return "failed", problem, digest
        if path.is_file():
            _keep_previous(path)
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return "updated", None, digest


@dataclass
class SyncFileResult:
    """One file's outcome this cycle."""

    name: str
    status: str  # "updated" | "unchanged" | "failed"
    http_status: int | None = None
    bytes_written: int = 0
    error: str | None = None


@dataclass
class SyncResult:
    files: list[SyncFileResult] = field(default_factory=list)

    @property
    def updated(self) -> int:
        return sum(1 for f in self.files if f.status == "updated")

    @property
    def failed(self) -> int:
        return sum(1 for f in self.files if f.status == "failed")

    @property
    def ok(self) -> bool:
        return self.failed == 0


class ArtifactSync:
    """Mirrors the private instance's published artifacts onto this one.

    ``client`` is injectable so the tests drive the whole task — 200, 304,
    500, oversize, garbage — against a transport stub with no network.
    """

    def __init__(
        self,
        config: Config,
        *,
        client: httpx.AsyncClient | None = None,
        on_file_updated=None,
    ) -> None:
        self.config = config
        self.settings = config.sync
        self._client = client
        self._owns_client = client is None
        #: file name → (etag, sha256) of the copy currently on disk.
        self._seen: dict[str, tuple[str | None, str | None]] = {}
        #: file name → ETag of the last body REJECTED by validation, sent
        #: back so the same bad body is answered 304, not re-downloaded.
        self._rejected: dict[str, str] = {}
        self._on_file_updated = on_file_updated
        self._scheduler = AsyncIOScheduler(timezone=timezone.utc)
        self._started = False

    # -- plumbing ---------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"User-Agent": "dmi-nowcast-sidecar-sync"}
            if self.settings.api_key:
                headers["Authorization"] = f"Bearer {self.settings.api_key}"
            self._client = httpx.AsyncClient(
                timeout=self.settings.timeout_s, headers=headers,
            )
        return self._client

    def url_for(self, name: str) -> str:
        base = (self.settings.source_url or "").rstrip("/")
        return f"{base}/{name.lstrip('/')}"

    # -- one file ---------------------------------------------------------

    async def sync_file(self, name: str) -> SyncFileResult:
        """Fetch one file if it changed; never raise, never clobber on failure."""
        etag, digest = self._seen.get(name, (None, None))
        tags = [tag for tag in (etag, self._rejected.get(name)) if tag]
        headers = {"If-None-Match": ", ".join(tags)} if tags else {}
        try:
            response = await self._get_client().get(
                self.url_for(name), headers=headers,
            )
        except Exception as exc:  # noqa: BLE001 — the peer is allowed to be down
            return SyncFileResult(
                name, "failed", error=f"{type(exc).__name__}: {exc}",
            )
        if response.status_code == 304:
            return SyncFileResult(name, "unchanged", 304)
        if response.status_code != 200:
            return SyncFileResult(
                name, "failed", response.status_code,
                error=f"HTTP {response.status_code}",
            )
        body = response.content
        if len(body) > self.settings.max_bytes:
            return SyncFileResult(
                name, "failed", 200,
                error=f"body of {len(body)} bytes exceeds max_bytes "
                      f"{self.settings.max_bytes}",
            )
        # A body that is not the document it claims to be — a proxy's HTML
        # error page, a truncated write, a model of the wrong target — must
        # not replace a good file: ``install_file`` validates it with the
        # consumer's own loader first, off the loop.
        path = target_path(self.config, name)
        try:
            status, problem, new_digest = await asyncio.to_thread(
                install_file, path, name, body, digest,
            )
        except Exception as exc:  # noqa: BLE001
            return SyncFileResult(
                name, "failed", 200, error=f"{type(exc).__name__}: {exc}",
            )
        new_etag = response.headers.get("ETag")
        if status == "failed":
            if new_etag:
                self._rejected[name] = new_etag
            return SyncFileResult(name, "failed", 200, error=problem)
        self._rejected.pop(name, None)
        self._seen[name] = (new_etag, new_digest)
        if status == "unchanged":
            # No ETag from the source, but the bytes are the ones we have.
            return SyncFileResult(name, "unchanged", 200)
        return SyncFileResult(name, "updated", 200, bytes_written=len(body))

    # -- one cycle --------------------------------------------------------

    async def sync_once(self) -> SyncResult:
        """One pass over every configured file. One log line per file."""
        result = SyncResult()
        for name in self.settings.files:
            outcome = await self.sync_file(name)
            result.files.append(outcome)
            if outcome.status == "failed":
                _log.warning(
                    "sync_file_failed", file=name, url=self.url_for(name),
                    http_status=outcome.http_status, error=outcome.error,
                    note="keeping the last good copy",
                )
            else:
                _log.info(
                    "sync_file", file=name, status=outcome.status,
                    http_status=outcome.http_status,
                    bytes=outcome.bytes_written,
                    target=str(target_path(self.config, name)),
                )
            if outcome.status == "updated" and self._on_file_updated is not None:
                try:
                    self._on_file_updated(name, target_path(self.config, name))
                except Exception as exc:  # noqa: BLE001
                    _log.warning(
                        "sync_after_update_hook_failed", file=name, error=str(exc),
                    )
        return result

    async def _run_once(self) -> None:
        try:
            await self.sync_once()
        except Exception as exc:  # noqa: BLE001
            _log.warning("sync_cycle_failed", error=str(exc))

    # -- lifecycle --------------------------------------------------------

    async def start(
        self, *, run_immediately: bool = True, wait: bool = True,
    ) -> None:
        """Start the interval job; the first pass awaited, or (``wait=False``)
        scheduled right now in the background so app start-up is not held
        by a slow or absent peer."""
        first_run: dict = {}
        if run_immediately and wait:
            await self._run_once()
        elif run_immediately:
            first_run["next_run_time"] = datetime.now(timezone.utc)
        self._scheduler.add_job(
            self._run_once,
            trigger=IntervalTrigger(
                minutes=self.settings.interval_min, jitter=JITTER_SEC,
            ),
            id="artifact_sync",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            **first_run,
        )
        self._scheduler.start()
        self._started = True
        _log.info(
            "artifact_sync_running",
            source_url=self.settings.source_url,
            interval_min=self.settings.interval_min,
            files=list(self.settings.files),
        )

    async def shutdown(self) -> None:
        if self._started:
            try:
                self._scheduler.shutdown(wait=False)
            except Exception as exc:  # noqa: BLE001
                _log.warning("artifact_sync_shutdown_error", error=str(exc))
            self._started = False
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None


def build_artifact_sync(
    config: Config, engine=None, *, push_thresholds=None,
) -> ArtifactSync | None:
    """The sync task for this config, or ``None`` when it must not run.

    When ``engine`` is given, a freshly-synced curve file nudges it to
    re-read the curves at the start of its next cycle, and a freshly-synced
    post-processing model nudges its ``PostprocessTable`` the same way;
    ``push_thresholds`` is the same arrangement for the fitted threshold
    table, nudged at the start of the next fan-out. Without them, the file
    still lands — it just takes a restart to take effect.

    ``stations/station_points.json`` needs no hook of either kind: the
    cycle's ``GaugeHistory`` re-reads that file every cycle until it parses
    (see :meth:`~dmi_nowcast_sidecar.gauge_history.GaugeHistory._load_points`),
    so the first sync is picked up by the next cycle on its own.
    """
    if not config.sync.enabled:
        return None
    if not config.sync.source_url:
        _log.warning("sync_disabled_no_source_url")
        return None

    def _updated(name: str, path: Path) -> None:
        target = None
        if name == CURVES_FILE:
            target = (engine, "note_curves_changed")
        elif name == THRESHOLDS_FILE:
            target = (push_thresholds, "note_changed")
        elif name == POSTPROCESS_FILE:
            target = (getattr(engine, "postprocess", None), "note_changed")
        elif name == ONSET_MODEL_FILE:
            target = (getattr(engine, "onset_postprocess", None), "note_changed")
        if target is None or target[0] is None:
            return
        owner, hook = target
        note = getattr(owner, hook, None)
        if callable(note):
            note()
        else:  # pragma: no cover — a stub without the hook
            _log.info(
                "synced_file_needs_restart", file=name, path=str(path),
                note="restart to apply",
            )

    return ArtifactSync(config, on_file_updated=_updated)


__all__ = [
    "CURVES_FILE",
    "ONSET_MODEL_FILE",
    "POSTPROCESS_FILE",
    "QUALITY_FILE",
    "STATION_POINTS_FILE",
    "THRESHOLDS_FILE",
    "ArtifactSync",
    "SyncFileResult",
    "SyncResult",
    "build_artifact_sync",
    "install_file",
    "target_path",
    "validation_problem",
]
