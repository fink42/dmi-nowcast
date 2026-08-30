"""CycleScheduler — basic wiring + lifecycle.

Doesn't exercise the actual 5-minute trigger (apscheduler's interval
trigger is well-tested upstream); we just confirm that ``start`` calls
``run_cycle``, ``shutdown`` is clean, and the on-complete callback
fires.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from dmi_nowcast_sidecar.compute import CycleEngine, CycleResult
from dmi_nowcast_sidecar.config import Config
from dmi_nowcast_sidecar.scheduler import CycleScheduler


@pytest.mark.asyncio
async def test_start_runs_initial_cycle(minimal_config: Config) -> None:
    engine = CycleEngine(minimal_config)
    engine.run_cycle = AsyncMock(  # type: ignore[method-assign]
        return_value=CycleResult(state=None, error="mocked"),
    )
    on_complete = MagicMock()
    scheduler = CycleScheduler(
        engine, interval_min=5, jitter_sec=30, on_cycle_complete=on_complete,
    )
    try:
        await scheduler.start(run_immediately=True)
        engine.run_cycle.assert_awaited_once()
        # On-complete hook fires once for the initial cycle.
        on_complete.assert_called_once()
        args = on_complete.call_args[0]
        assert isinstance(args[0], CycleResult)
    finally:
        await scheduler.shutdown()


@pytest.mark.asyncio
async def test_start_can_skip_initial_cycle(minimal_config: Config) -> None:
    engine = CycleEngine(minimal_config)
    engine.run_cycle = AsyncMock(  # type: ignore[method-assign]
        return_value=CycleResult(state=None, error=None),
    )
    scheduler = CycleScheduler(engine)
    try:
        await scheduler.start(run_immediately=False)
        engine.run_cycle.assert_not_awaited()
    finally:
        await scheduler.shutdown()


@pytest.mark.asyncio
async def test_on_complete_hook_failure_does_not_crash_cycle(
    minimal_config: Config,
) -> None:
    """If the on_cycle_complete callback raises, the cycle itself must
    not propagate that exception — we only log it."""
    engine = CycleEngine(minimal_config)
    engine.run_cycle = AsyncMock(  # type: ignore[method-assign]
        return_value=CycleResult(state=None),
    )

    def _boom(_r: CycleResult) -> None:
        raise RuntimeError("kaboom")

    scheduler = CycleScheduler(engine, on_cycle_complete=_boom)
    try:
        # Must not raise.
        await scheduler.start(run_immediately=True)
    finally:
        await scheduler.shutdown()


@pytest.mark.asyncio
async def test_shutdown_closes_engine(minimal_config: Config) -> None:
    engine = CycleEngine(minimal_config)
    engine.run_cycle = AsyncMock(  # type: ignore[method-assign]
        return_value=CycleResult(state=None),
    )
    engine.aclose = AsyncMock()  # type: ignore[method-assign]
    scheduler = CycleScheduler(engine)
    await scheduler.start(run_immediately=False)
    await scheduler.shutdown()
    engine.aclose.assert_awaited_once()
