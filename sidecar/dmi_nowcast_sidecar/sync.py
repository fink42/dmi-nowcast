"""Pulling published artifacts from the private instance (Phase F, F4).

Two files the public instance serves but cannot produce:

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

Both are small, static-per-cycle JSON documents on a network only these
two containers share, so the transport is deliberately dull: one
conditional GET per file per ``interval_min``, ``If-None-Match`` against
the ETag from last time, and a content hash as the fallback when the
source sends no ETag.

The failure policy is the whole design. **Last good wins**: a refused
connection, a 500, a body that is not the JSON it claims to be, a body
past ``max_bytes`` — every one of them leaves the file already on disk
untouched and logs one line. A public instance whose private peer is down
keeps serving yesterday's report; the document carries its own
``generated_at_utc`` and the page shows how old it is. The alternative,
truncating or blanking a served file because a fetch failed, turns one
instance's outage into the other's.

Writes are tmp + rename in the target directory, so a route reading the
file concurrently sees either the old document or the new one.

Async discipline: httpx async for the fetch, ``asyncio.to_thread`` for
every disk write, its own ``AsyncIOScheduler`` so a slow private instance
can never delay a radar cycle.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import timezone
from pathlib import Path, PurePosixPath

import httpx
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from .config import Config
from .push.paths import resolved_thresholds_path

_log = structlog.get_logger(__name__)

#: Spread the poll off the exact minute boundary, as every other task does.
JITTER_SEC = 30

#: The two files that do NOT land under ``data_dir`` by their relative
#: path: the engine reads its curves from
#: ``calibration.national_curves_path`` and the push service reads its
#: thresholds from ``push.thresholds_path``, wherever the operator put
#: them.
CURVES_FILE = "calibration/national_curves.json"
THRESHOLDS_FILE = "calibration/push_thresholds.json"


def target_path(config: Config, name: str) -> Path:
    """Where a synced file lands on this instance.

    Everything is ``<storage.data_dir>/<name>`` except the two fitted
    files, which go to the paths the engine and the push service actually
    read. Getting this wrong is silent — the file appears, nothing loads
    it — so it lives in one function with one test.
    """
    if name == CURVES_FILE:
        return Path(config.calibration.national_curves_path)
    if name == THRESHOLDS_FILE:
        return resolved_thresholds_path(config)
    return Path(config.storage.data_dir).joinpath(*PurePosixPath(name).parts)


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


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
        headers = {"If-None-Match": etag} if etag else {}
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
        if name.endswith(".json"):
            # A body that is not the document it claims to be — a proxy's
            # HTML error page, a truncated write at the source — must not
            # replace a good file.
            try:
                json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                return SyncFileResult(
                    name, "failed", 200, error=f"not valid JSON: {exc}",
                )
        new_digest = hashlib.sha256(body).hexdigest()
        path = target_path(self.config, name)
        if new_digest == digest and path.is_file():
            # No ETag from the source, but the bytes are the ones we have.
            self._seen[name] = (response.headers.get("ETag"), new_digest)
            return SyncFileResult(name, "unchanged", 200)
        try:
            await asyncio.to_thread(_write_atomic, path, body)
        except Exception as exc:  # noqa: BLE001
            return SyncFileResult(
                name, "failed", 200, error=f"{type(exc).__name__}: {exc}",
            )
        self._seen[name] = (response.headers.get("ETag"), new_digest)
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

    async def start(self, *, run_immediately: bool = True) -> None:
        if run_immediately:
            await self._run_once()
        self._scheduler.add_job(
            self._run_once,
            trigger=IntervalTrigger(
                minutes=self.settings.interval_min, jitter=JITTER_SEC,
            ),
            id="artifact_sync",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
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
    re-read the curves at the start of its next cycle; ``push_thresholds``
    is the same arrangement for the fitted threshold table, nudged at the
    start of the next fan-out. Without either, the file still lands — it
    just takes a restart to take effect.
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
    "THRESHOLDS_FILE",
    "ArtifactSync",
    "SyncFileResult",
    "SyncResult",
    "build_artifact_sync",
    "target_path",
]
