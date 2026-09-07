"""The dedicated worker pools — where the sidecar's heavy blocking work runs.

The bug these exist for is not a Python-object leak and no object count
would have caught it: ``asyncio.to_thread`` grows the shared executor
whenever two independent callers overlap, glibc gives every thread its own
malloc arena and never returns an arena's high-water mark, so a
multi-gigabyte transient costs that much RSS *per thread it has ever run
on*. Phase F added the second and third ``to_thread`` callers and the
private instance started climbing ~1 GB an hour.

So the invariant under test is a thread invariant: a named pool is exactly
one thread, that thread is stable for the life of the process, and it stays
stable while other blocking work is in flight — which is the case that used
to grow the shared pool. Everything else here is the plumbing that keeps
that true (context propagation, isolation between pools, shutdown).
"""
from __future__ import annotations

import asyncio
import contextvars
import threading

import pytest

from dmi_nowcast_sidecar import workers


@pytest.fixture(autouse=True)
def _fresh_pools():
    """Every test starts and ends with no pools of its own."""
    workers.shutdown_pools(wait=True)
    yield
    workers.shutdown_pools(wait=True)


async def test_a_pool_is_one_stable_thread_across_many_cycles() -> None:
    """N cycles must not mean N threads — that is the whole mechanism."""
    seen: set[int] = set()

    for _ in range(25):
        await workers.run_in_pool("cycle", lambda: seen.add(threading.get_ident()))

    assert len(seen) == 1
    assert seen != {threading.get_ident()}, "the work ran on the event loop thread"


async def test_the_pool_thread_is_stable_while_other_blocking_work_overlaps() -> None:
    """The regression case: a second, independent ``to_thread`` caller.

    The gauge poller runs on its own scheduler and lands mid-cycle often
    enough to make the *shared* executor grow a worker. The dedicated pool
    must not notice.
    """
    seen: set[int] = set()
    release = threading.Event()

    def occupy() -> None:
        release.wait(timeout=5.0)

    for _ in range(10):
        release.clear()
        # Three shared-executor jobs held open across the pooled call, so
        # asyncio's default pool is provably growing underneath us.
        side = [asyncio.create_task(asyncio.to_thread(occupy)) for _ in range(3)]
        await asyncio.sleep(0)
        await workers.run_in_pool("cycle", lambda: seen.add(threading.get_ident()))
        release.set()
        await asyncio.gather(*side)

    assert len(seen) == 1


async def test_pools_are_isolated_from_each_other() -> None:
    """Each named job gets its own thread, so one cannot inherit another's arena."""
    idents: dict[str, int] = {}
    for name in workers.POOL_NAMES:
        idents[name] = await workers.run_in_pool(name, threading.get_ident)

    assert len(set(idents.values())) == len(workers.POOL_NAMES)


async def test_worker_threads_are_named_after_their_job() -> None:
    """``dmi-cycle_0`` in a stack trace beats ``asyncio_3``."""
    name = await workers.run_in_pool("cycle", lambda: threading.current_thread().name)
    assert name.startswith("dmi-cycle")


async def test_an_unknown_pool_name_is_refused() -> None:
    """The set is fixed: an invented name would be an unbounded thread source."""
    with pytest.raises(ValueError):
        workers.pool("whatever")
    with pytest.raises(ValueError):
        await workers.run_in_pool("whatever", lambda: None)


async def test_the_calling_context_is_visible_inside_the_worker() -> None:
    """Same contract as ``asyncio.to_thread``: structlog contextvars survive."""
    var: contextvars.ContextVar[str] = contextvars.ContextVar("marker")
    var.set("set-on-the-loop")
    assert await workers.run_in_pool("cycle", var.get) == "set-on-the-loop"


async def test_arguments_and_return_value_pass_through() -> None:
    def add(a: int, b: int, *, c: int = 0) -> int:
        return a + b + c

    assert await workers.run_in_pool("cycle", add, 1, 2, c=3) == 6


async def test_an_exception_propagates_to_the_awaiting_coroutine() -> None:
    def boom() -> None:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        await workers.run_in_pool("cycle", boom)


async def test_shutdown_is_idempotent_and_a_pool_can_be_reacquired() -> None:
    """Teardown drops the registry, so the next call builds a fresh worker.

    Compared by executor identity rather than thread id: the OS is free to
    hand the new thread the id the old one just released, and has.
    """
    first = workers.pool("cycle")
    await workers.run_in_pool("cycle", lambda: None)
    workers.shutdown_pools(wait=True)
    workers.shutdown_pools(wait=True)  # twice is not an error
    second = workers.pool("cycle")
    assert second is not first, "shutdown left the old executor in the registry"
    assert await workers.run_in_pool("cycle", lambda: 7) == 7


def test_release_arrow_pool_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pyarrow that cannot release is not a reason to fail a finished job."""
    import builtins

    real_import = builtins.__import__

    def exploding(name, *args, **kwargs):
        if name == "pyarrow":
            raise ImportError("no pyarrow here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", exploding)
    workers.release_arrow_pool()  # must not raise


def test_process_rss_is_a_number_or_an_honest_none() -> None:
    """Linux gives a size; anywhere else the debug line omits it rather than lying."""
    rss = workers.process_rss_bytes()
    assert rss is None or rss > 0


async def test_the_debug_line_is_off_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(workers.DEBUG_ENV, raising=False)
    lines: list[tuple] = []
    monkeypatch.setattr(
        workers._log, "info", lambda *a, **k: lines.append((a, k)),
    )
    await workers.run_in_pool("cycle", lambda: None)
    assert lines == []

    monkeypatch.setenv(workers.DEBUG_ENV, "1")
    await workers.run_in_pool("cycle", lambda: None)
    assert [a[0] for a, _ in lines] == ["worker_job_done"]
