"""The ``quality.json`` build (Phase F, F4).

Once a day, on its own scheduler, this task turns the corpora, the warning
replay and the live gauge scoreboard into the document the website's
/quality page renders, and writes it to
``<storage.data_dir>/nowcast/quality.json`` — the directory the
``/nowcast/*`` routes serve from, so publishing is a file write and
nothing else.

**Two schedules, one document.** The nightly cron job above is the FULL
build. Beside it runs a second, ``IntervalTrigger`` job every
``quality_report.live_refresh_min`` minutes (hourly by default, ``0``
turns it off) that rebuilds only the document's live half — the warning
scoreboard, the station map, the recent warnings, the "is it raining
now?" check — and carries the corpus-based reliability sections over
unchanged from the document already on disk. The live inputs move every
10 minutes; the corpus fits take minutes and change once a day, so
refreshing them hourly would be minutes of CPU spent reproducing
yesterday's numbers.

The two jobs share one ``asyncio.Lock``. They read the same decision
rows and write the same file, and two builders racing on a tmp+rename
would leave whichever finished second in place regardless of which had
the better evidence. The nightly build waits for the lock; a live tick
that finds it held is SKIPPED with a log line rather than queued —
another one is due in an hour, and a queued refresh would only pile up
behind a long build.

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

**The build runs in a child process**, not in a thread of this one
(:mod:`dmi_nowcast_sidecar.quality_job`). It reads the whole evidence base
into Arrow and numpy — two multi-million-row corpora, every decision row,
the gauge store behind them — and neither Arrow's memory pool nor
CPython's allocator returns that memory afterwards. Built in-process it
took the live service from ~0.9 GB to 5.5 GB anon RSS and got it
OOM-killed beside a batch replay. A process that exits gives the memory
back, and a build that crashes, hangs or is itself OOM-killed costs a log
line rather than the service: the parent sees a non-zero exit or a
``quality_report.timeout_s`` timeout and keeps serving yesterday's
document.

What stays in the parent is the in-process hook: after a successful fit
the running ``push.thresholds.ThresholdTable`` is told to re-read. The
child cannot reach the parent's objects, and does not need to — the file
is on disk by the time it exits.

Async discipline: the parent only awaits a subprocess, so nothing blocks
the event loop. When a test injects a builder (or a renderer, or a
fitter) the same job runs in-process via ``asyncio.to_thread``, because a
lambda cannot cross a process boundary. Either way the writes are tmp +
rename in the target directory, so the HTTP route can never serve a
half-written document — and, on failure, the previous report stays
exactly where it was. A quality page is allowed to be a day stale; it is
not allowed to be truncated.

Private-instance only. ``Config`` refuses ``enabled`` under
``server.public_mode`` at load; :func:`build_quality_report_task` checks
again, because a config object assembled in code never went through that
validator.
"""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .config import Config
from .push.paths import resolved_postprocess_path, resolved_thresholds_path
from .quality_job import JOB_MODULE, inputs_to_json, run_job, sweep_options_to_json

_log = structlog.get_logger(__name__)

#: The served file name, under ``<data_dir>/nowcast/``. Fixed: the
#: frontend's ``qualityUrl()`` builds ``/nowcast/quality.json`` and the
#: public gate allows exactly that path.
QUALITY_FILENAME = "quality.json"


#: How long to wait for a killed child to be reaped before giving up on it.
_REAP_TIMEOUT_S = 10.0

#: Longest excerpt of the child's stderr carried into a log line. Enough
#: to see the exception; short enough not to flood the log with a
#: traceback the child already printed in full.
_STDERR_TAIL = 500


def post_column_template() -> str:
    """``"p_post_{lead}"`` — the engine's probability column, as a template.

    Read from the core module rather than typed out, so the sweep, the
    live writer and the offline study cannot end up naming three
    different columns.
    """
    from dmi_nowcast_core.postprocess import POST_COLUMN_TEMPLATE

    return POST_COLUMN_TEMPLATE


def _postprocess_options_to_json(options: Any) -> dict:
    """Lazy shim around ``postprocess_fit.options_to_json``.

    Imported inside the function for the same reason the sweep's options
    are: ``postprocess_fit`` pulls in pyarrow and numpy, which an instance
    with the nightly build switched off should not have to load to start.
    """
    from .postprocess_fit import options_to_json

    return options_to_json(options)


def quality_path(config: Config) -> Path:
    """``<storage.data_dir>/nowcast/quality.json`` — what the route serves."""
    return Path(config.storage.data_dir) / "nowcast" / QUALITY_FILENAME


def _tail(stream: bytes | None) -> str:
    """The last useful line(s) of a child's stderr, for one log field."""
    text = (stream or b"").decode("utf-8", "replace").strip()
    return text[-_STDERR_TAIL:] if text else ""


@dataclass
class QualityBuildResult:
    """What one build did — the shape of its log line, and of its tests."""

    ok: bool = False
    #: Which half of the document this run rebuilt: ``"full"`` for the
    #: nightly build, ``"live"`` for a live refresh. Recorded so
    #: ``last_result`` — which keeps the last run of EITHER kind — can
    #: still be read for what it was.
    mode: str = "full"
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
    #: The post-processing refit (Phase H), on the same terms: where the
    #: model landed, its stamp, its counts, and the error when it did not.
    postprocess_path: Path | None = None
    postprocess_fitted_at: str | None = None
    postprocess: dict = field(default_factory=dict)
    postprocess_error: str | None = None


class QualityReportTask:
    """Builds the quality report once a day and writes it where it is served.

    The build itself runs in a child process
    (:mod:`dmi_nowcast_sidecar.quality_job`) so its gigabytes leave with
    it; this class owns the schedule, the timeout, the summary and the
    one hook that has to stay in the parent.

    ``builder`` / ``renderer`` / ``fitter`` are injectable so tests
    exercise the whole job — the guard, the atomic writes, the failure
    policy — without a corpus on disk; injecting any of them runs the job
    in a worker thread instead of a child, because a Python callable
    cannot be handed to one.
    """

    def __init__(
        self,
        config: Config,
        *,
        builder: Callable[[Any], dict] | None = None,
        renderer: Callable[[dict], str] | None = None,
        fitter: Callable[[Any], dict] | None = None,
        postprocess_fitter: Callable[[Any], dict] | None = None,
        thresholds: Any = None,
        postprocess: Any = None,
        executable: str | None = None,
    ) -> None:
        self.config = config
        self.settings = config.quality_report
        self._builder = builder
        self._renderer = renderer
        #: The sweep. Injectable so a test can exercise the whole step —
        #: the guard, the atomic write, the reload nudge, the report's
        #: embedded section — without a season of parquet on disk.
        self._fitter = fitter
        #: The post-processing refit, injectable on the same terms.
        self._postprocess_fitter = postprocess_fitter
        #: The running service's ``push.thresholds.ThresholdTable``, told
        #: to re-read after a successful fit. ``None`` means the file
        #: still lands; it just takes a restart to take effect.
        self._thresholds = thresholds
        #: The engine's ``push.postprocess.PostprocessTable``, on the same
        #: terms — nudged after a successful refit so the next cycle
        #: scores with tonight's model.
        self._postprocess = postprocess
        #: The interpreter the child is spawned with. Overridable so a
        #: test can point it at a stub that records its argv.
        self._executable = executable or sys.executable
        self._scheduler = AsyncIOScheduler(timezone=timezone.utc)
        #: Serialises the nightly build against the live refresh. Both
        #: read the same decision rows and write the same file, so they
        #: must never overlap; see the module docstring for who waits and
        #: who is skipped.
        self._lock = asyncio.Lock()
        self._started = False
        self._last: QualityBuildResult | None = None

    @property
    def last_result(self) -> QualityBuildResult | None:
        return self._last

    @property
    def in_process(self) -> bool:
        """True when something injectable was supplied — a test.

        Production injects nothing and the job is spawned; a builder, a
        renderer or a fitter is a Python object that cannot cross a
        process boundary, so it runs in a worker thread instead.
        """
        return any(
            hook is not None
            for hook in (
                self._builder, self._renderer, self._fitter,
                self._postprocess_fitter,
            )
        )

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
            # Same evidence floor the nightly fit uses, so the page and the
            # table exclude the same broken buckets.
            min_known_slots=int(self.settings.fit_thresholds.min_known_slots),
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

    def postprocess_out(self) -> Path:
        """Where the refit model is written, and read back from.

        ``fit_postprocess.out`` when set, else the file the running cycle
        reads (``push.postprocess_path``). The default is the point:
        fitting into a file nothing loads would produce a very
        well-documented no-op.
        """
        configured = self.settings.fit_postprocess.out
        if configured is not None:
            return Path(configured)
        return resolved_postprocess_path(self.config)

    def _postprocess_options(self) -> Any:
        """The :class:`PostprocessFitOptions` this config describes.

        Deliberately shares the threshold fit's rows, gauge rule and lead
        set: the model and the thresholds fitted on top of it have to
        stand on the same evidence, or the percent is a threshold on a
        probability that was never measured there.
        """
        from .postprocess_fit import PostprocessFitOptions
        from .push.routes import lead_options

        settings = self.settings.fit_postprocess
        thresholds = self.settings.fit_thresholds
        dirs = settings.decisions_dirs or thresholds.decisions_dirs
        return PostprocessFitOptions(
            decisions_dirs=[Path(d) for d in dirs],
            corpus_dir=Path(self.config.storage.corpus_dir),  # type: ignore[arg-type]
            out=self.postprocess_out(),
            # The horizons that can actually be subscribed to — the same
            # set the threshold fit uses, for the same reason.
            leads=tuple(thresholds.leads or lead_options(self.config)),
            # Every lead the national products publish: the SHAPE of the
            # ensemble fraction against lead is itself a predictor, and
            # the cycle fills exactly these columns.
            design_leads=tuple(
                int(lead) for lead in self.config.forecast.national.leads_min
            ),
            l2=float(settings.l2),
            dry_min=int(thresholds.dry_min),
            onset_min_mm=float(thresholds.onset_min_mm),
            min_known_slots=int(thresholds.min_known_slots),
        )

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
            # The onset definition the fit is scored under, carried into
            # the thresholds document's ``objective``.
            dry_min=int(settings.dry_min),
            onset_min_mm=float(settings.onset_min_mm),
            min_known_slots=int(settings.min_known_slots),
            workers=int(settings.workers),
            # Phase H: the sweep replays the rule on the SAME probability
            # the engine decides with. A table fitted on the curve and
            # served against the post-processed number would warn at the
            # wrong percent on every horizon — so this follows
            # ``push.probability_source`` and nothing else, and the
            # thresholds document records which one it was.
            probability_column=(
                post_column_template()
                if self.config.push.probability_source == "postprocess"
                else None
            ),
            postprocess_model=(
                self.postprocess_out()
                if self.config.push.probability_source == "postprocess"
                else None
            ),
            design_leads=tuple(
                int(lead) for lead in self.config.forecast.national.leads_min
            ),
        )

    def job_config(self, *, live_only: bool = False) -> dict:
        """Everything the job needs, resolved, as one JSON-able document.

        This is the whole contract with the child: paths already resolved
        against the config, the sweep's options already reduced to the
        live rule's constants. The child reads no YAML and no
        environment, so what the parent decided is what runs.

        A live refresh never carries a fit. The job refuses to run one
        under ``--live-only`` anyway; switching it off here as well means
        the command line a reader sees says what actually happens.
        """
        settings = self.settings
        fit = settings.fit_thresholds
        payload: dict = {
            "quality": {
                "out_json": str(quality_path(self.config)),
                "markdown_dir": (
                    None if settings.markdown_dir is None
                    else str(settings.markdown_dir)
                ),
                "inputs": inputs_to_json(self.inputs()),
            },
            "fit": {"enabled": False},
            "fit_postprocess": {"enabled": False},
        }
        post = settings.fit_postprocess
        # The refit only runs where it can: it needs rows and a gauge
        # store, and ``decisions_dirs`` empty means there is nothing to
        # fit on — the same "empty disables it" rule the threshold fit has.
        post_dirs = post.decisions_dirs or fit.decisions_dirs
        if post.enabled and post_dirs and not live_only:
            payload["fit_postprocess"] = {
                "enabled": True,
                "out": str(self.postprocess_out()),
                "options": _postprocess_options_to_json(
                self._postprocess_options(),
            ),
            }
        if fit.enabled and not live_only:
            payload["fit"] = {
                "enabled": True,
                "thresholds_out": str(self.thresholds_out()),
                "min_delta_pct": int(fit.min_delta_pct),
                "min_warnings": int(fit.min_warnings),
                "options": sweep_options_to_json(self._fit_options()),
            }
        return payload

    def child_argv(
        self, payload: dict | None = None, *, live_only: bool = False,
    ) -> list[str]:
        """The command line one build is spawned as."""
        config = (
            self.job_config(live_only=live_only) if payload is None else payload
        )
        argv = [
            self._executable, "-m", JOB_MODULE,
            "--config-json", json.dumps(config, sort_keys=True),
        ]
        if live_only:
            argv.append("--live-only")
        return argv

    # -- the job ----------------------------------------------------------

    @staticmethod
    def _summary_of(stdout: bytes) -> dict | None:
        """The child's last stdout line, parsed. ``None`` if it is not one."""
        for line in reversed(stdout.decode("utf-8", "replace").splitlines()):
            text = line.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except ValueError:
                return None
            return parsed if isinstance(parsed, dict) else None
        return None

    @staticmethod
    def _result_of(summary: dict) -> QualityBuildResult:
        """One job summary as the result this task reports and logs."""
        thresholds = summary.get("thresholds_path")
        postprocess = summary.get("postprocess_path")
        path = summary.get("path")
        return QualityBuildResult(
            ok=bool(summary.get("ok")),
            # An older job (or a stub) that does not say gets "full",
            # which is what every summary meant before live refreshes.
            mode=str(summary.get("mode") or "full"),
            path=Path(path) if path else None,
            bytes_written=int(summary.get("bytes") or 0),
            sections=[str(s) for s in (summary.get("sections") or [])],
            schema_problems=[
                str(p) for p in (summary.get("schema_problems") or [])
            ],
            error=summary.get("error"),
            thresholds_path=Path(thresholds) if thresholds else None,
            thresholds_guard={
                str(k): str(v)
                for k, v in (summary.get("thresholds_guard") or {}).items()
            },
            thresholds_error=summary.get("thresholds_error"),
            postprocess_path=Path(postprocess) if postprocess else None,
            postprocess_fitted_at=summary.get("postprocess_fitted_at"),
            postprocess=dict(summary.get("postprocess") or {}),
            postprocess_error=summary.get("postprocess_error"),
        )

    @staticmethod
    async def _kill(proc: "asyncio.subprocess.Process") -> None:
        """SIGKILL and reap, so a hung build cannot outlive its slot."""
        try:
            proc.kill()
        except ProcessLookupError:  # pragma: no cover — it just exited
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=_REAP_TIMEOUT_S)
        except (asyncio.TimeoutError, TimeoutError):  # pragma: no cover
            _log.warning("quality_report_child_unreaped", pid=proc.pid)

    async def _run_child(
        self, payload: dict, *, live_only: bool = False,
    ) -> QualityBuildResult:
        """Spawn the job, await it under a timeout, read its summary.

        Every failure mode ends the same way: a result with ``ok=False``
        and an error string. The service is untouched by all of them —
        the report on disk is last night's, and the process that held the
        gigabytes is gone.
        """
        argv = self.child_argv(payload, live_only=live_only)
        mode = "live" if live_only else "full"
        timeout = float(self.settings.timeout_s)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            return QualityBuildResult(
                ok=False, mode=mode,
                error=f"spawn failed: {type(exc).__name__}: {exc}",
            )
        try:
            # communicate(), not wait(): it drains both pipes, so a chatty
            # build cannot fill a pipe buffer and deadlock against us.
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except (asyncio.TimeoutError, TimeoutError):
            await self._kill(proc)
            return QualityBuildResult(
                ok=False, mode=mode, error=f"timed out after {timeout:g}s",
            )
        summary = self._summary_of(stdout)
        if summary is None:
            tail = _tail(stderr)
            return QualityBuildResult(
                ok=False,
                mode=mode,
                error=(
                    f"no summary from the job (exit {proc.returncode})"
                    + (f": {tail}" if tail else "")
                ),
            )
        result = self._result_of(summary)
        if proc.returncode != 0 and result.ok:
            # A summary that claims success under a non-zero exit is a bug
            # in the job, not a report worth trusting.
            return QualityBuildResult(
                ok=False,
                mode=mode,
                error=f"job exited {proc.returncode} after claiming success",
            )
        if not result.ok and result.error is None:
            result.error = _tail(stderr) or f"exit {proc.returncode}"
        return result

    async def _run_in_process(
        self, payload: dict, *, live_only: bool = False,
    ) -> QualityBuildResult:
        """The injected-hook path: the same job, in a worker thread.

        Only tests reach it (see :attr:`in_process`), and it exists so
        they exercise the real job — the guard, the atomic writes, the
        summary — rather than a second implementation of it.
        """
        try:
            summary = await asyncio.to_thread(
                run_job,
                payload,
                live_only=live_only,
                builder=self._builder,
                renderer=self._renderer,
                fitter=self._fitter,
                postprocess_fitter=self._postprocess_fitter,
            )
        except Exception as exc:  # noqa: BLE001
            return QualityBuildResult(
                ok=False, mode="live" if live_only else "full",
                error=f"{type(exc).__name__}: {exc}",
            )
        return self._result_of(summary)

    async def build_once(self, *, live_only: bool = False) -> QualityBuildResult:
        """One build. Never raises: a failure leaves the previous report.

        The whole point of the atomic write plus this swallow is that the
        page's worst case is a stale document with an honest
        ``generated_at_utc``, never a 500 and never a truncated one.

        ``live_only`` runs the hourly refresh instead of the full build:
        no threshold fit, no markdown twin, and the corpus-based sections
        carried over from the document on disk. A live refresh with no
        usable document to carry from fails like any other build — one
        log line, and the good document untouched.
        """
        mode = "live" if live_only else "full"
        started = datetime.now(timezone.utc)
        try:
            # Resolving the config can fail on its own (a fit configured
            # without a corpus dir, say), and this method promises never
            # to raise — so it is inside the guard with everything else.
            payload = self.job_config(live_only=live_only)
        except Exception as exc:  # noqa: BLE001
            result = QualityBuildResult(
                ok=False, mode=mode,
                error=f"job config failed: {type(exc).__name__}: {exc}",
            )
        else:
            result = await (
                self._run_in_process(payload, live_only=live_only)
                if self.in_process
                else self._run_child(payload, live_only=live_only)
            )
        elapsed = round(
            (datetime.now(timezone.utc) - started).total_seconds(), 1,
        )
        if result.ok:
            _log.info(
                "quality_report_built",
                mode=mode,
                path=str(result.path),
                bytes=result.bytes_written,
                sections=result.sections,
                thresholds=(
                    None if result.thresholds_path is None
                    else str(result.thresholds_path)
                ),
                schema_problems=len(result.schema_problems),
                in_process=self.in_process,
                elapsed_s=elapsed,
            )
            for problem in result.schema_problems:
                _log.warning("quality_report_schema_problem", problem=problem)
        else:
            _log.warning(
                "quality_report_build_failed",
                mode=mode,
                error=result.error,
                elapsed_s=elapsed,
            )
        if result.thresholds_error:
            _log.info("quality_report_fit_not_applied", reason=result.thresholds_error)
        if result.postprocess_error:
            _log.info(
                "quality_report_postprocess_not_applied",
                reason=result.postprocess_error,
            )
        # The same hook, for the model: the child wrote the file, the
        # parent tells the running cycle to re-read it. Without this the
        # refit takes effect on the next restart instead of the next
        # cycle — and the thresholds fitted beside it would then be
        # thresholds on a probability the engine is not yet computing.
        if result.postprocess_path is not None:
            note = getattr(self._postprocess, "note_changed", None)
            if callable(note):
                note()
            _log.info(
                "quality_report_postprocess_done",
                path=str(result.postprocess_path),
                fitted_at=result.postprocess_fitted_at,
                rows=result.postprocess.get("rows"),
                leads=result.postprocess.get("leads"),
                reloaded=callable(note),
            )
        # The hook the child cannot call: the service re-reads the table at
        # the start of its next fan-out. Without it the file still landed
        # and takes effect on restart.
        if result.thresholds_path is not None:
            note = getattr(self._thresholds, "note_changed", None)
            if callable(note):
                note()
            _log.info(
                "quality_report_fit_done",
                path=str(result.thresholds_path),
                guard=result.thresholds_guard,
                reloaded=callable(note),
            )
        self._last = result
        return result

    async def _run_once(self, live_only: bool = False) -> None:
        """apscheduler job target — swallows everything by contract.

        The lock is what keeps the nightly build and the hourly refresh
        off each other: they read the same decision rows and rename onto
        the same file. The nightly build waits for it. A live tick that
        finds it held gives up instead, because another is due within the
        interval and a queue of refreshes behind a long build would all
        write the same thing in a row.

        ``Lock.locked()`` followed by ``async with`` is safe here without
        a second guard: acquiring a free ``asyncio.Lock`` never suspends,
        so nothing else on this loop can take it in between.
        """
        if live_only and self._lock.locked():
            _log.info("quality_report_live_refresh_skipped", reason="build in progress")
            return
        async with self._lock:
            await self.build_once(live_only=live_only)

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
            kwargs={"live_only": False},
            id="quality_report_build",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        live_min = int(self.settings.live_refresh_min)
        if live_min:
            # ``IntervalTrigger`` with no ``start_date`` first fires one
            # interval from now, which is what this wants: a restart must
            # not spend CPU refreshing a document that was just written,
            # and the nightly build is the one that makes the first one.
            self._scheduler.add_job(
                self._run_once,
                trigger=IntervalTrigger(minutes=live_min, timezone=timezone.utc),
                kwargs={"live_only": True},
                id="quality_report_live",
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
        self._scheduler.start()
        self._started = True
        _log.info(
            "quality_report_task_running",
            at_utc=self.settings.at_utc,
            live_refresh_min=live_min,
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
    config: Config, *, thresholds: Any = None, postprocess: Any = None,
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
    return QualityReportTask(
        config, thresholds=thresholds, postprocess=postprocess,
    )


__all__ = [
    "QUALITY_FILENAME",
    "QualityBuildResult",
    "QualityReportTask",
    "build_quality_report_task",
    "quality_path",
]
