"""Rain-gauge polling task (Phase F, F1).

Keeps the corpus's gauge archive current so the benchmark has ground
truth that the radar did not produce. Every ``station_obs.interval_min``
the poller asks DMI's metObs API for the last ``lookback_min`` of each
configured parameter and merges the result into
``<storage.corpus_dir>/stations/obs/YYYY/MM.parquet``.

Why a lookback rather than a since-cursor: DMI backfills late station
reports into slots that already passed, and the store dedupes on
``(station_id, observed_utc, parameter_id)``, so re-reading the same
forty minutes every ten is both cheap and self-healing. A missed cycle
needs no recovery logic at all.

This task owns its own ``AsyncIOScheduler`` rather than riding the radar
cycle's. The two cadences are unrelated (10 min against 5 min ± jitter),
gauge data is not an input to a nowcast, and a metObs outage must not be
able to delay a radar cycle.

Async discipline, as everywhere in this service: the HTTP call is async,
and every Parquet read/rewrite goes to a thread via ``asyncio.to_thread``.
Nothing touches the filesystem on the event loop.

Data licence: CC BY 4.0 (DMI Open Data).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from dmi_nowcast_core.metobs import AsyncMetObsClient, is_trace
from dmi_nowcast_core.station_store import StationObsStore

from .config import Config

_log = structlog.get_logger(__name__)

#: Spread the poll off the exact minute boundary, as the radar cycle does.
JITTER_SEC = 30


@dataclass
class StationObsPollResult:
    """What one poll did — the shape of its log line, and of its tests."""

    fetched: int = 0
    new_rows: int = 0
    traces: int = 0
    skipped: int = 0
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


class StationObsPoller:
    """Periodically mirror DMI gauge observations into the corpus.

    ``client`` and ``store`` are injectable so tests exercise the whole
    task with no network and no real corpus directory.
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
        corpus_dir = config.storage.corpus_dir
        if store is None and corpus_dir is None:
            raise ValueError(
                "StationObsPoller needs storage.corpus_dir (or an injected store)",
            )
        self.store = store or StationObsStore(Path(corpus_dir))  # type: ignore[arg-type]
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

    def window(self, now: datetime | None = None) -> tuple[datetime, datetime]:
        """The ``[start, end]`` this poll asks DMI for.

        ``end`` is *now*, not a rounded slot: DMI publishes a slot within
        a minute of its stamp, and asking past the present costs nothing.
        """
        end = now or datetime.now(timezone.utc)
        return end - timedelta(minutes=self.settings.lookback_min), end

    # -- the job ----------------------------------------------------------

    async def poll_once(self, now: datetime | None = None) -> StationObsPollResult:
        """One fetch-and-merge pass over every configured parameter.

        Never raises: a parameter that fails is recorded in
        ``result.errors`` and the others still land. The next interval is
        the retry, and the overlapping lookback means nothing is lost by
        having skipped one.
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
                # Parquet rewrite — blocking, so off the loop it goes.
                written = await asyncio.to_thread(self.store.append, observations)
            except Exception as exc:  # noqa: BLE001
                result.errors[parameter] = f"{type(exc).__name__}: {exc}"
                _log.warning(
                    "station_obs_append_failed", parameter=parameter, error=str(exc),
                )
                continue
            result.new_rows += int(written.get("new", 0))
        _log.info(
            "station_obs_poll",
            start=start.isoformat(timespec="seconds"),
            end=end.isoformat(timespec="seconds"),
            fetched=result.fetched,
            new_rows=result.new_rows,
            traces=result.traces,
            skipped=result.skipped,
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

    Refuses in public mode as a second line of defence: ``Config``
    already rejects that combination at load, so reaching this branch
    means a config object was assembled in code rather than loaded, and
    the safe answer is still "no poller".
    """
    if not config.station_obs.enabled:
        return None
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
]
