"""The nightly ``quality.json`` build (Phase F, F4).

Once a day, on its own scheduler, this task turns the corpora, the warning
replay and the live gauge scoreboard into the document the website's
/quality page renders, and writes it to
``<storage.data_dir>/nowcast/quality.json`` — the directory the
``/nowcast/*`` routes serve from, so publishing is a file write and
nothing else.

Phase G adds one step in front of the build: the nightly **push-threshold
fit** (``quality_report.fit_thresholds``). It replays the same decision
rows against the same gauge store to answer "which threshold should each
horizon warn at?", damps the answer against the table already in service
(``push_thresholds.apply_stability_guard``), writes it where the running
service reads it, and nudges the service to re-read. It runs *before* the
report is built so ``quality.json``'s ``thresholds`` section describes
tonight's table rather than last night's, and it fails the way everything
else here fails — one log line, the previous table left exactly where it
was.

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
from .push.paths import resolved_thresholds_path

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
    #: The push-threshold fit, when it ran: where the table was written
    #: and what the stability guard did per lead. ``None`` when the step
    #: is off; ``thresholds_error`` when it ran and failed — the report is
    #: still built and still written either way.
    thresholds_path: Path | None = None
    thresholds_guard: dict[str, str] = field(default_factory=dict)
    thresholds_error: str | None = None


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
        fitter: Callable[[Any], dict] | None = None,
        thresholds: Any = None,
    ) -> None:
        self.config = config
        self.settings = config.quality_report
        self._builder = builder
        self._renderer = renderer
        #: The sweep. Injectable so a test can exercise the whole step —
        #: the guard, the atomic write, the reload nudge, the report's
        #: embedded section — without a season of parquet on disk.
        self._fitter = fitter
        #: The running service's ``push.thresholds.ThresholdTable``, told
        #: to re-read after a successful fit. ``None`` means the file
        #: still lands; it just takes a restart to take effect.
        self._thresholds = thresholds
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
            # The fitted push thresholds the service is serving right
            # now — after the fit step below has run, tonight's. A missing
            # or unusable file nulls the section rather than faking it.
            thresholds_path=self.thresholds_out(),
            live_days=settings.live_days,
            live_days_secondary=settings.live_days_secondary,
        )

    # -- the push-threshold fit (Phase G, G4) -----------------------------

    def thresholds_out(self) -> Path:
        """Where the fitted table is written, and read back from.

        ``fit_thresholds.thresholds_out`` when set, else the file the
        running push service reads (``push.thresholds_path``). The default
        is the point: fitting into a file nothing loads would produce a
        very well-documented no-op.
        """
        configured = self.settings.fit_thresholds.thresholds_out
        if configured is not None:
            return Path(configured)
        return resolved_thresholds_path(self.config)

    def _fit_options(self) -> Any:
        """The :class:`SweepOptions` this config describes."""
        from .push.routes import lead_options
        from .threshold_sweep import SweepOptions, parse_thresholds

        settings = self.settings.fit_thresholds
        radar = settings.radar_decisions_dir
        return SweepOptions(
            decisions_dirs=[Path(d) for d in settings.decisions_dirs],
            corpus_dir=Path(self.config.storage.corpus_dir),  # type: ignore[arg-type]
            radar_decisions_dirs=None if radar is None else [Path(radar)],
            # The horizons that can actually be subscribed to. Fitting a
            # lead nobody can choose spends minutes of CPU on a column of
            # the report nobody reads.
            leads=tuple(settings.leads or lead_options(self.config)),
            thresholds=parse_thresholds(settings.thresholds),
            # The live rule's constants, so the replay measures the
            # service rather than a hypothetical one.
            rearm_after_min=self.config.push.rearm_after_min,
            persistence_obs=self.config.push.persistence_obs,
            min_useful_lead_min=float(settings.min_useful_lead_min),
            plateau_frac=float(settings.plateau_frac),
            min_warnings=int(settings.min_warnings),
            workers=int(settings.workers),
        )

    def fit_thresholds(self) -> QualityBuildResult:
        """Refit the push thresholds and publish the guarded table.

        Blocking (the sweep is minutes of CPU over a season of decision
        rows) — called from :meth:`_build_sync`, which is already in a
        worker thread. Returns a partial result carrying only the fields
        this step owns; :meth:`_build_sync` merges them into the build's.

        Never raises. Every failure — no corpus, no rows, a sweep that
        blew up — leaves the table already in service exactly where it
        was, which is the same failure policy as the report itself.
        """
        from dmi_nowcast_core.push_thresholds import (
            apply_stability_guard,
            load_thresholds,
        )

        from .threshold_sweep import SweepError, run_fit, write_atomic

        settings = self.settings.fit_thresholds
        out = self.thresholds_out()
        try:
            fit = self._fitter or run_fit
            payload = fit(self._fit_options())
            new_doc = payload["thresholds"]
            # Guard against the table in service, which is the file we are
            # about to overwrite — read it BEFORE the write, obviously,
            # and treat an unusable one as a first fit.
            previous = load_thresholds(out)
            doc = apply_stability_guard(
                new_doc, previous,
                min_delta_pct=int(settings.min_delta_pct),
                min_warnings=int(settings.min_warnings),
            )
            write_atomic(out, json.dumps(doc, indent=1) + "\n")
        except SweepError as exc:
            # Nothing to fit on: not a bug, and not worth a warning every
            # night while the corpus is still filling up.
            _log.info("quality_report_fit_skipped", reason=str(exc))
            return QualityBuildResult(thresholds_error=str(exc))
        except Exception as exc:  # noqa: BLE001 - the report still builds
            _log.warning(
                "quality_report_fit_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            return QualityBuildResult(
                thresholds_error=f"{type(exc).__name__}: {exc}",
            )
        guard = {
            key: str(entry.get("guard"))
            for key, entry in (doc.get("leads") or {}).items()
            if isinstance(entry, dict)
        }
        # The service re-reads at the start of its next fan-out; without
        # this hook the file still lands and takes effect on restart.
        note = getattr(self._thresholds, "note_changed", None)
        if callable(note):
            note()
        _log.info(
            "quality_report_fit_done",
            path=str(out),
            fitted_at=doc.get("fitted_at_utc"),
            guard=guard,
            reloaded=callable(note),
        )
        return QualityBuildResult(thresholds_path=out, thresholds_guard=guard)

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
        # The fit first, so ``inputs()`` hands the builder tonight's table
        # and ``quality.json``'s thresholds section describes the rule the
        # service is on as of now.
        fit = (
            self.fit_thresholds()
            if self.settings.fit_thresholds.enabled
            else QualityBuildResult()
        )
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
            thresholds_path=fit.thresholds_path,
            thresholds_guard=fit.thresholds_guard,
            thresholds_error=fit.thresholds_error,
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
                thresholds=(
                    None if result.thresholds_path is None
                    else str(result.thresholds_path)
                ),
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


def build_quality_report_task(
    config: Config, *, thresholds: Any = None,
) -> QualityReportTask | None:
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
    return QualityReportTask(config, thresholds=thresholds)


__all__ = [
    "QUALITY_FILENAME",
    "QualityBuildResult",
    "QualityReportTask",
    "build_quality_report_task",
    "quality_path",
]
