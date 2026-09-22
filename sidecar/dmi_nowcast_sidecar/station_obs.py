"""Rain-gauge polling task (Phase F, F1; bounded store v2, S5).

Keeps a gauge archive current so the benchmark has ground truth that the
radar did not produce. Every ``station_obs.interval_min`` the poller asks
DMI's metObs API for the last ``lookback_min`` of each configured
parameter and merges the result into
``<root>/stations/obs/YYYY/MM.parquet``.

Two roots, and the difference is retention, not shape:

``storage.corpus_dir`` (``store_dir`` null — the private instance)
    The archive. Every month kept for ever, because the backtest, the
    replay and the quality report all read history out of it.
``station_obs.store_dir`` (the public instance, S5)
    A working set on the data volume, pruned to
    ``station_obs.retention_days`` after every poll. The public stack owns
    no corpus and cannot archive anything, but its cycle computes the
    ``ng_*`` neighbour-gauge features on every frame at a ten-minute
    visibility horizon — so it needs the last few hours of readings
    locally, which is a different thing from needing the record.

Why a lookback rather than a since-cursor: DMI backfills late station
reports into slots that already passed, and the store dedupes on
``(station_id, observed_utc, parameter_id)``, so re-reading the same
forty minutes every ten is both cheap and self-healing. A missed cycle
needs no recovery logic at all.

Request budget, both instances together: one GET per parameter per poll
(no ``stationId`` filter, and 40 minutes of every Danish gauge is a few
hundred rows — one page under the client's 300k limit), so 12 an hour
each against DMI's 500 per 5 seconds. No API key: DMI stopped requiring
one on ``opendataapi.dmi.dk`` on 2025-12-02, and
``AsyncMetObsClient`` only sends ``X-Gravitee-Api-Key`` when
``station_obs.api_key`` is set.

This task owns its own ``AsyncIOScheduler`` rather than riding the radar
cycle's. The two cadences are unrelated (10 min against 5 min ± jitter),
gauge data is not an input to a nowcast, and a metObs outage must not be
able to delay a radar cycle.

Async discipline, as everywhere in this service: the HTTP call is async,
and every Parquet read/rewrite goes to a worker thread. Nothing touches
the filesystem on the event loop.

Memory: the rewrite reads the whole current month, concatenates, dedupes,
sorts and writes it back, and it does that every ten minutes — by the end
of a month that is well over a million rows of Arrow buffers, six times a
day, in the process that also serves the API. Nothing holds on to them
(``StationObsStore`` keeps a root path and no cached table, and this
poller keeps neither), but two things have to be asked for explicitly.
Arrow's pool does not return the high-water mark to the kernel on its
own, so :func:`release_arrow_pool` asks it to after each append; and the
rewrite runs on the dedicated ``"station_obs"`` worker rather than the
shared executor, so its high-water lives in one allocator arena instead
of one per thread the shared pool grows (see
:mod:`dmi_nowcast_sidecar.workers` — this poller being a *second*
independent ``to_thread`` caller beside the radar cycle is exactly what
made the shared pool grow in the first place).

Data licence: CC BY 4.0 (DMI Open Data).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from dmi_nowcast_core.metobs import AsyncMetObsClient, is_trace
from dmi_nowcast_core.station_store import StationObsStore

from .config import Config
from .workers import release_arrow_pool, run_in_pool

_log = structlog.get_logger(__name__)

#: Spread the poll off the exact minute boundary, as the radar cycle does.
JITTER_SEC = 30


def month_partition_end(year: int, month: int) -> datetime:
    """The first instant AFTER everything a ``YYYY/MM`` partition can hold.

    Retention compares this rather than the month's start, so a partition
    is only ever deleted once every row it could contain is older than the
    cutoff. Raises ``ValueError`` on a year/month that is not one, which is
    how :meth:`StationObsPoller.prune_once` refuses to date a file whose
    name it does not recognise.
    """
    return (
        datetime(year + 1, 1, 1, tzinfo=timezone.utc) if month == 12
        else datetime(year, month + 1, 1, tzinfo=timezone.utc)
    )


@dataclass
class StationObsPollResult:
    """What one poll did — the shape of its log line, and of its tests."""

    fetched: int = 0
    new_rows: int = 0
    traces: int = 0
    skipped: int = 0
    #: Month partitions retention deleted after this poll. Always 0 on the
    #: private instance, which prunes nothing.
    pruned: int = 0
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


class StationObsPoller:
    """Periodically mirror DMI gauge observations into a store on this host.

    ``client`` and ``store`` are injectable so tests exercise the whole
    task with no network and no real corpus directory.

    ``prunes`` is decided here, once, from ``station_obs.store_dir``: only
    the bounded store is ever pruned, and an injected store inherits that
    decision from the config rather than from its own path. Retention is
    the one operation in this module that DELETES data, so which root it
    may run against is a property of the configuration and not of a
    per-call argument.
    """

    def __init__(
        self,
        config: Config,
        *,
        client: AsyncMetObsClient | None = None,
        store: StationObsStore | None = None,
    ) -> None:
        self.config = config
        self.settings = config.station_obs
        root = self.settings.store_dir or config.storage.corpus_dir
        if store is None and root is None:
            raise ValueError(
                "StationObsPoller needs station_obs.store_dir or "
                "storage.corpus_dir (or an injected store)",
            )
        self.store = store or StationObsStore(Path(root))  # type: ignore[arg-type]
        #: Retention applies to the bounded store and nothing else.
        self.prunes = self.settings.store_dir is not None
        self._client = client
        self._owns_client = client is None
        self._scheduler = AsyncIOScheduler(timezone=timezone.utc)
        self._started = False

    # -- plumbing ---------------------------------------------------------

    def _get_client(self) -> AsyncMetObsClient:
        if self._client is None:
            self._client = AsyncMetObsClient(
                base_url=self.settings.base_url,
                api_key=self.settings.api_key,
            )
        return self._client

    def _append_and_release(self, observations) -> dict[str, int]:
        """Merge one parameter's rows, then give the buffers back."""
        try:
            return self.store.append(observations)
        finally:
            release_arrow_pool()

    def window(self, now: datetime | None = None) -> tuple[datetime, datetime]:
        """The ``[start, end]`` this poll asks DMI for.

        ``end`` is *now*, not a rounded slot: DMI publishes a slot within
        a minute of its stamp, and asking past the present costs nothing.
        """
        end = now or datetime.now(timezone.utc)
        return end - timedelta(minutes=self.settings.lookback_min), end

    # -- retention (bounded store only) -----------------------------------

    def prune_once(self, now: datetime | None = None) -> list[Path]:
        """Delete month partitions that ended before the retention cutoff.

        Blocking (``unlink``), so callers put it on the ``"station_obs"``
        worker with the append. Returns the files it removed.

        Month granularity, deliberately: a partition is the unit the store
        writes atomically, and trimming rows out of the *current* month
        would mean rewriting the file the next append is about to rewrite.
        So retention keeps whole months, and the window it actually holds
        is ``retention_days`` rounded up to the containing months — a week
        holds between 7 and 38 days. At ~0.3 MiB per month of real gauge
        data that is the cheapest safe rule there is.

        Two refusals, both about deleting the wrong thing. WHICH store may
        be pruned is :attr:`prunes`, decided from the config once and
        checked by the caller, so a poller over the corpus never reaches
        this method at all. And a file whose path does not parse as
        ``YYYY/MM`` is left where it is: an unrecognised name is not
        evidence of age.
        """
        cutoff = (now or datetime.now(timezone.utc)) - timedelta(
            days=self.settings.retention_days,
        )
        removed: list[Path] = []
        for path in self.store.partitions():
            try:
                end = month_partition_end(int(path.parent.name), int(path.stem))
            except ValueError:
                continue
            if end > cutoff:
                continue
            try:
                path.unlink()
            except OSError as exc:
                _log.warning(
                    "station_obs_prune_failed", path=str(path), error=str(exc),
                )
                continue
            removed.append(path)
        for year_dir in {path.parent for path in removed}:
            # An emptied year directory is litter, not data. Anything still
            # in it (a partition retention kept) makes rmdir fail, which is
            # the check.
            try:
                year_dir.rmdir()
            except OSError:
                pass
        return removed

    # -- the job ----------------------------------------------------------

    async def poll_once(self, now: datetime | None = None) -> StationObsPollResult:
        """One fetch-and-merge pass over every configured parameter.

        Never raises: a parameter that fails is recorded in
        ``result.errors`` and the others still land. The next interval is
        the retry, and the overlapping lookback means nothing is lost by
        having skipped one.

        Retention runs last, and only on the bounded store. After the
        append, so a poll never deletes a month it is about to write into;
        and inside the same failure policy, so a volume that refuses an
        unlink costs one warning line rather than the readings.
        """
        start, end = self.window(now)
        client = self._get_client()
        result = StationObsPollResult()
        for parameter in self.settings.parameters:
            try:
                observations = await client.fetch_observations(parameter, start, end)
            except Exception as exc:  # noqa: BLE001 — one parameter must not sink the rest
                result.errors[parameter] = f"{type(exc).__name__}: {exc}"
                _log.warning(
                    "station_obs_fetch_failed", parameter=parameter, error=str(exc),
                )
                continue
            result.fetched += len(observations)
            result.traces += sum(1 for o in observations if is_trace(o.value))
            result.skipped += client.last_stats.skipped
            if not observations:
                continue
            try:
                # Parquet rewrite — blocking, so off the loop it goes,
                # onto this task's own worker (see .workers). The pool
                # release rides in the same thread: it is a C call of
                # microseconds, but it belongs where the buffers died.
                written = await run_in_pool(
                    "station_obs", self._append_and_release, observations,
                )
            except Exception as exc:  # noqa: BLE001
                result.errors[parameter] = f"{type(exc).__name__}: {exc}"
                _log.warning(
                    "station_obs_append_failed", parameter=parameter, error=str(exc),
                )
                continue
            result.new_rows += int(written.get("new", 0))
        if self.prunes:
            try:
                # Same worker as the append, for the same reason: it is
                # filesystem work and it must not ride the event loop.
                removed = await run_in_pool("station_obs", self.prune_once, end)
            except Exception as exc:  # noqa: BLE001 — housekeeping only
                _log.warning("station_obs_prune_failed", error=str(exc))
            else:
                result.pruned = len(removed)
                if removed:
                    _log.info(
                        "station_obs_pruned",
                        partitions=[str(path) for path in removed],
                        retention_days=self.settings.retention_days,
                    )
        _log.info(
            "station_obs_poll",
            start=start.isoformat(timespec="seconds"),
            end=end.isoformat(timespec="seconds"),
            fetched=result.fetched,
            new_rows=result.new_rows,
            traces=result.traces,
            skipped=result.skipped,
            pruned=result.pruned,
            errors=len(result.errors),
        )
        return result

    async def _run_once(self) -> None:
        """apscheduler job target — swallows everything by contract."""
        try:
            await self.poll_once()
        except Exception as exc:  # noqa: BLE001
            _log.warning("station_obs_poll_failed", error=str(exc))

    # -- lifecycle --------------------------------------------------------

    async def start(self, *, run_immediately: bool = True) -> None:
        if run_immediately:
            await self._run_once()
        self._scheduler.add_job(
            self._run_once,
            trigger=IntervalTrigger(
                minutes=self.settings.interval_min,
                jitter=JITTER_SEC,
            ),
            id="station_obs_poll",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self._scheduler.start()
        self._started = True
        _log.info(
            "station_obs_poller_running",
            interval_min=self.settings.interval_min,
            lookback_min=self.settings.lookback_min,
            parameters=list(self.settings.parameters),
            store=str(self.store.obs_dir),
            # Which of the two roots this is, in one field, because the
            # answer decides whether months get deleted.
            retention_days=(
                self.settings.retention_days if self.prunes else None
            ),
        )

    async def shutdown(self) -> None:
        if self._started:
            try:
                self._scheduler.shutdown(wait=False)
            except Exception as exc:  # noqa: BLE001
                _log.warning("station_obs_scheduler_shutdown_error", error=str(exc))
            self._started = False
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None


def build_station_obs_poller(config: Config) -> StationObsPoller | None:
    """The poller for this config, or ``None`` when it must not run.

    A ``store_dir`` is the whole permission: it names a bounded store on
    this instance's own volume, which is the one arrangement a public
    instance may poll under (S5) and which the private instance never
    sets. Without it the two old refusals stand, as a second line of
    defence — ``Config`` already rejects both at load, so reaching either
    branch means a config object was assembled in code rather than loaded,
    and the safe answer is still "no poller".
    """
    if not config.station_obs.enabled:
        return None
    if config.station_obs.store_dir is not None:
        return StationObsPoller(config)
    if config.server.public_mode:
        _log.warning("station_obs_disabled_public_mode")
        return None
    if config.storage.corpus_dir is None:
        _log.warning("station_obs_disabled_no_corpus_dir")
        return None
    return StationObsPoller(config)


__all__ = [
    "StationObsPoller",
    "StationObsPollResult",
    "build_station_obs_poller",
    "month_partition_end",
    "release_arrow_pool",
]
