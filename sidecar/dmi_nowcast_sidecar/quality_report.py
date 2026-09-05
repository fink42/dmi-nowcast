"""The nightly ``quality.json`` build (Phase F, F4).

Once a day, on its own scheduler, this task turns the corpora, the warning
replay and the live gauge scoreboard into the document the website's
/quality page renders, and writes it to
``<storage.data_dir>/nowcast/quality.json`` — the directory the
``/nowcast/*`` routes serve from, so publishing is a file write and
nothing else.

Why its own ``AsyncIOScheduler`` rather than the radar cycle's, exactly as
``station_obs`` does: the cadences are unrelated (daily against 5 min),
the report is not an input to a nowcast, and a corpus read that takes two
minutes must not be able to delay a cycle.

Async discipline: the build is entirely numpy/pyarrow/Parquet work, so it
goes to a thread via ``asyncio.to_thread`` and never touches the event
loop. The write is tmp + rename in the target directory, so the HTTP route
can never serve a half-written document — and, on failure, the previous
report stays exactly where it was. A quality page is allowed to be a day
stale; it is not allowed to be truncated.

Private-instance only. ``Config`` refuses ``enabled`` under
``server.public_mode`` at load; :func:`build_quality_report_task` checks
again, because a config object assembled in code never went through that
validator.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .config import Config

_log = structlog.get_logger(__name__)

#: The served file name, under ``<data_dir>/nowcast/``. Fixed: the
#: frontend's ``qualityUrl()`` builds ``/nowcast/quality.json`` and the
#: public gate allows exactly that path.
QUALITY_FILENAME = "quality.json"


def quality_path(config: Config) -> Path:
    """``<storage.data_dir>/nowcast/quality.json`` — what the route serves."""
    return Path(config.storage.data_dir) / "nowcast" / QUALITY_FILENAME


def _write_atomic(path: Path, text: str) -> None:
    """tmp + rename in the target directory; a reader never sees a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@dataclass
class QualityBuildResult:
    """What one build did — the shape of its log line, and of its tests."""

    ok: bool = False
    path: Path | None = None
    bytes_written: int = 0
    sections: list[str] = field(default_factory=list)
    schema_problems: list[str] = field(default_factory=list)
    error: str | None = None


class QualityReportTask:
    """Builds the quality report once a day and writes it where it is served.

    ``builder`` is injectable so tests exercise the whole task — the
    schedule, the executor hop, the atomic write, the failure policy —
    without a corpus on disk.
    """

    def __init__(
        self,
        config: Config,
        *,
        builder: Callable[[Any], dict] | None = None,
        renderer: Callable[[dict], str] | None = None,
    ) -> None:
        self.config = config
        self.settings = config.quality_report
        self._builder = builder
        self._renderer = renderer
        self._scheduler = AsyncIOScheduler(timezone=timezone.utc)
        self._started = False
        self._last: QualityBuildResult | None = None

    @property
    def last_result(self) -> QualityBuildResult | None:
        return self._last

    # -- inputs -----------------------------------------------------------

    def inputs(self) -> Any:
        """The :class:`QualityInputs` this config describes.

        ``national_curves`` defaults to the file the running engine reads,
        so the reliability diagrams describe the probability the site
        actually served rather than the raw ensemble fraction.
        """
        from dmi_nowcast_core.quality_report import QualityInputs

        settings = self.settings
        curves = settings.national_curves
        if curves is None:
            curves = self.config.calibration.national_curves_path
        return QualityInputs(
            radar_corpus=settings.radar_corpus,
            station_corpus=settings.station_corpus,
            replay_dir=settings.replay_dir,
            corpus_dir=self.config.storage.corpus_dir,
            persistence_json=settings.persistence_json,
            national_curves=Path(curves) if curves is not None else None,
            live_days=settings.live_days,
            live_days_secondary=settings.live_days_secondary,
        )

    # -- the job ----------------------------------------------------------

    def _build_sync(self) -> QualityBuildResult:
        """The blocking half: read every corpus, produce, validate, write."""
        from dmi_nowcast_core.quality_report import (
            build_quality_report,
            render_markdown,
            validate_report,
        )

        build = self._builder or build_quality_report
        render = self._renderer or render_markdown
        report = build(self.inputs())
        problems = validate_report(report)
        payload = json.dumps(report, indent=1, sort_keys=False)
        path = quality_path(self.config)
        _write_atomic(path, payload)

        markdown_dir = self.settings.markdown_dir
        if markdown_dir is not None:
            stamp = str(report.get("generated_at_utc") or "")[:10] or (
                datetime.now(timezone.utc).date().isoformat()
            )
            try:
                _write_atomic(Path(markdown_dir) / f"{stamp}.md", render(report))
            except Exception as exc:  # noqa: BLE001 — the archive twin is a nicety
                _log.warning("quality_report_markdown_failed", error=str(exc))

        return QualityBuildResult(
            ok=True,
            path=path,
            bytes_written=len(payload.encode("utf-8")),
            sections=[
                key for key in (
                    "windows", "headline", "reliability", "raining_now",
                    "stations", "events", "methods",
                )
                if isinstance(report, dict) and report.get(key) is not None
            ],
            schema_problems=problems,
        )

    async def build_once(self) -> QualityBuildResult:
        """One build. Never raises: a failure leaves the previous report.

        The whole point of the atomic write plus this swallow is that the
        page's worst case is a stale document with an honest
        ``generated_at_utc``, never a 500 and never a truncated one.
        """
        started = datetime.now(timezone.utc)
        try:
            result = await asyncio.to_thread(self._build_sync)
        except Exception as exc:  # noqa: BLE001
            result = QualityBuildResult(
                ok=False, error=f"{type(exc).__name__}: {exc}",
            )
            _log.warning("quality_report_build_failed", error=str(exc))
        else:
            _log.info(
                "quality_report_built",
                path=str(result.path),
                bytes=result.bytes_written,
                sections=result.sections,
                schema_problems=len(result.schema_problems),
                elapsed_s=round(
                    (datetime.now(timezone.utc) - started).total_seconds(), 1,
                ),
            )
            for problem in result.schema_problems:
                _log.warning("quality_report_schema_problem", problem=problem)
        self._last = result
        return result

    async def _run_once(self) -> None:
        """apscheduler job target — swallows everything by contract."""
        await self.build_once()

    # -- lifecycle --------------------------------------------------------

    async def start(self, *, run_immediately: bool = False) -> None:
        """Schedule the daily build.

        ``run_immediately`` is off by default: a restart at 09:00 should
        not spend two minutes of CPU rebuilding a report that is already
        on disk and at most a day old. The deploy script's one-off build
        (``sidecar/deploy/quality_report.sh``) is how the first one is
        made.
        """
        if run_immediately:
            await self._run_once()
        hour, _, minute = self.settings.at_utc.partition(":")
        self._scheduler.add_job(
            self._run_once,
            trigger=CronTrigger(
                hour=int(hour), minute=int(minute), timezone=timezone.utc,
            ),
            id="quality_report_build",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self._scheduler.start()
        self._started = True
        _log.info(
            "quality_report_task_running",
            at_utc=self.settings.at_utc,
            out=str(quality_path(self.config)),
        )

    async def shutdown(self) -> None:
        if not self._started:
            return
        try:
            self._scheduler.shutdown(wait=False)
        except Exception as exc:  # noqa: BLE001
            _log.warning("quality_report_scheduler_shutdown_error", error=str(exc))
        self._started = False


def build_quality_report_task(config: Config) -> QualityReportTask | None:
    """The task for this config, or ``None`` when it must not run.

    Refuses in public mode as a second line of defence: ``Config`` already
    rejects that combination at load, so reaching this branch means a
    config object was assembled in code rather than loaded, and the safe
    answer is still "no builder".
    """
    if not config.quality_report.enabled:
        return None
    if config.server.public_mode:
        _log.warning("quality_report_disabled_public_mode")
        return None
    if config.storage.corpus_dir is None:
        _log.warning("quality_report_disabled_no_corpus_dir")
        return None
    return QualityReportTask(config)


__all__ = [
    "QUALITY_FILENAME",
    "QualityBuildResult",
    "QualityReportTask",
    "build_quality_report_task",
    "quality_path",
]
