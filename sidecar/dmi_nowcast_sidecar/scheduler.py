"""apscheduler wiring for the 5 min ± jitter cycle.

The scheduler owns no domain logic — it just fires :meth:`CycleEngine.run_cycle`
on the configured cadence and updates ``app.state.last_cycle_at`` so the
``/healthz`` endpoint can report freshness. Failures inside the cycle are
logged but never propagate up; the next firing is the retry.

Two post-cycle hooks, both optional and both isolated from the cycle:
``on_cycle_complete`` is synchronous and trivial (it stamps app state),
``after_cycle`` is awaited and may do real work off the loop (Web Push
evaluation and fan-out, website Phase D). An exception in either is logged
and swallowed — a notification that fails must never cost the next radar
cycle.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Awaitable, Callable

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from .compute import CycleEngine, CycleResult

_log = structlog.get_logger(__name__)


class CycleScheduler:
    """Wraps an AsyncIOScheduler around a single ``CycleEngine`` job."""

    def __init__(
        self,
        engine: CycleEngine,
        *,
        interval_min: int = 5,
        jitter_sec: int = 30,
        on_cycle_complete: Callable[[CycleResult], None] | None = None,
        after_cycle: Callable[[CycleResult], Awaitable[None]] | None = None,
    ) -> None:
        self.engine = engine
        self.interval_min = interval_min
        self.jitter_sec = jitter_sec
        self._on_complete = on_cycle_complete
        self._after_cycle = after_cycle
        self._scheduler = AsyncIOScheduler(timezone=timezone.utc)

    async def _run_once(self) -> None:
        """Single-cycle wrapper used as the apscheduler job target."""
        result = await self.engine.run_cycle()
        if self._on_complete is not None:
            try:
                self._on_complete(result)
            except Exception as exc:  # noqa: BLE001
                _log.warning("on_cycle_complete_hook_failed", error=str(exc))
        if self._after_cycle is not None:
            try:
                await self._after_cycle(result)
            except Exception as exc:  # noqa: BLE001
                _log.warning("after_cycle_hook_failed", error=str(exc))

    async def start(self, *, run_immediately: bool = True) -> None:
        """Run an initial cycle (optional), then start the recurring trigger."""
        if run_immediately:
            _log.info("scheduler_starting", initial_cycle=True)
            await self._run_once()
        self._scheduler.add_job(
            self._run_once,
            trigger=IntervalTrigger(
                minutes=self.interval_min,
                jitter=self.jitter_sec,
            ),
            id="dmi_cycle",
            replace_existing=True,
            max_instances=1,  # don't overlap if a cycle runs long
            coalesce=True,    # drop missed firings instead of running them all
        )
        self._scheduler.start()
        _log.info(
            "scheduler_running",
            interval_min=self.interval_min,
            jitter_sec=self.jitter_sec,
        )

    async def shutdown(self) -> None:
        _log.info("scheduler_stopping")
        try:
            self._scheduler.shutdown(wait=False)
        except Exception as exc:  # noqa: BLE001
            _log.warning("scheduler_shutdown_error", error=str(exc))
        await self.engine.aclose()
