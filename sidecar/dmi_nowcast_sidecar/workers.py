"""Where the sidecar's blocking work runs, and why it gets its own thread.

Every heavy job in this service is already off the event loop. What this
module adds is *which* thread it goes to, and that turns out to matter as
much as the offloading did.

The problem
-----------

``asyncio.to_thread`` submits to the loop's **default** executor, a
``ThreadPoolExecutor`` with ``min(32, cpu_count + 4)`` workers that grows
a new thread whenever a job arrives and no worker is idle. One caller
therefore means one thread for the life of the process; two independent
callers whose cadences overlap mean two, then three, and work is handed
to whichever worker reaches the queue first.

glibc gives each thread that allocates its own malloc arena, and it does
not return an arena's high-water mark to the kernel — only the top of the
main arena is ever trimmed. A job with a multi-hundred-megabyte transient
working set therefore costs that much RSS **per thread it has ever run
on**, permanently.

That is what happened here. Until Phase F the radar cycle was the only
``to_thread`` caller in the process, so STEPS always ran on thread #1 and
RSS sat flat at ~0.9 GB for days. Phase F added two more callers on
unrelated cadences — the gauge poller (10 min, own scheduler) and the
station scoreboard (after each cycle) — which overlap the 5-min cycle
often enough that the default pool grows, and each new worker that takes
a turn at the cycle inflates another arena to the STEPS high-water. The
public instance never got the second caller and never grew.

Measured on glibc (python:3.12-slim), a ~1.5 GB transient repeated 14
times:

===========================================  ==========================
one ``to_thread`` caller                     plateaus at ~0.97 GB
a second, independent ``to_thread`` caller   2.1 GB at cycle 13, rising
the heavy job pinned to its own worker       plateaus at ~0.97 GB
===========================================  ==========================

The rule
--------

A job whose transient working set is measured in hundreds of megabytes
gets a **named single-worker pool** and always runs there, so its
high-water mark is one arena for the life of the process rather than one
per thread the shared pool happens to grow. Small jobs — a file copy, a
SQLite read, a PNG, a JSON write — stay on ``asyncio.to_thread``, where
the pool's elasticity is worth having and its cost is nothing.

The second half of the same discipline is :func:`release_arrow_pool`:
Arrow's memory pool keeps the largest partition it ever built, and RSS is
what the OOM killer scores, so a job that rewrote a month of parquet asks
for those pages back before it returns.
"""
from __future__ import annotations

import asyncio
import contextvars
import functools
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

import structlog

_log = structlog.get_logger(__name__)

T = TypeVar("T")

#: The jobs that get a thread of their own, and the only names
#: :func:`run_in_pool` accepts. Fixed on purpose: the whole point is a
#: small, bounded set of long-lived threads, so a pool name is not
#: something a caller may invent at run time.
#:
#: - ``cycle`` — ``CycleEngine._compute_sync``: parse, dense flow, STEPS,
#:   the national reduction, the artifacts. Gigabytes of transient numpy.
#: - ``station_eval`` — the per-cycle gauge scoreboard: sampling plus a
#:   month-partition rewrite.
#: - ``station_obs`` — the gauge poller's month-partition rewrite.
POOL_NAMES: tuple[str, ...] = ("cycle", "station_eval", "station_obs")

#: Set ``DMI_NOWCAST_WORKER_DEBUG=1`` to log the pool, the worker thread
#: and the process RSS after every pooled job. Off by default and read
#: once at import: this is the instrument for "which thread is holding
#: the memory", and it is worth exactly one /proc read per heavy job when
#: someone is asking that question.
DEBUG_ENV = "DMI_NOWCAST_WORKER_DEBUG"

_pools: dict[str, ThreadPoolExecutor] = {}
_lock = threading.Lock()


def _debug_enabled() -> bool:
    return os.environ.get(DEBUG_ENV, "").strip().lower() in {"1", "true", "yes"}


def process_rss_bytes() -> int | None:
    """Resident set size of this process, or ``None`` where unavailable.

    Linux only (``/proc/self/statm``), which is where the service runs.
    Anywhere else this returns ``None`` and the debug line simply omits
    the number rather than guessing at one.
    """
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as fh:
            resident_pages = int(fh.read().split()[1])
    except (OSError, IndexError, ValueError):
        return None
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def pool(name: str) -> ThreadPoolExecutor:
    """The single-worker executor for ``name``, created on first use."""
    if name not in POOL_NAMES:
        raise ValueError(f"unknown worker pool {name!r}; expected one of {POOL_NAMES}")
    with _lock:
        executor = _pools.get(name)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"dmi-{name}",
            )
            _pools[name] = executor
        return executor


async def run_in_pool(
    name: str, func: Callable[..., T], /, *args: Any, **kwargs: Any,
) -> T:
    """``asyncio.to_thread`` against this job's own dedicated worker.

    Same contract as ``asyncio.to_thread`` — the current context is
    copied, so ``structlog`` contextvars and anything else the caller set
    are visible inside — with the one difference that matters: the work
    always lands on the same thread, and therefore in the same allocator
    arena, however many other blocking jobs the process is running.

    Calls to one pool serialise behind each other, which is what the
    callers already assumed: a cycle never overlaps a cycle
    (``max_instances=1``), a poll never overlaps a poll (``coalesce``),
    and the scoreboard runs once per radar frame.
    """
    loop = asyncio.get_running_loop()
    context = contextvars.copy_context()
    call = functools.partial(context.run, func, *args, **kwargs)
    result = await loop.run_in_executor(pool(name), call)
    if _debug_enabled():
        rss = process_rss_bytes()
        _log.info(
            "worker_job_done",
            pool=name,
            rss_mb=None if rss is None else round(rss / 1e6, 1),
            threads=threading.active_count(),
        )
    return result


def release_arrow_pool() -> None:
    """Hand a finished table job's buffers back to the kernel.

    ``pa.default_memory_pool().release_unused()`` frees pool pages nothing
    is using any more. Without it the pool keeps the largest partition it
    ever built, which on the live instance is a permanent step up in RSS
    every month — and RSS is what the OOM killer scores.

    Never raises: a pyarrow that cannot do this is not a reason to fail a
    job that already succeeded.
    """
    try:
        import pyarrow as pa

        pa.default_memory_pool().release_unused()
    except Exception:  # noqa: BLE001 — best effort, by design
        pass


def shutdown_pools(*, wait: bool = False) -> None:
    """Stop every pool this process created. Safe to call twice.

    ``wait=False`` by default for the same reason the schedulers shut down
    that way: a service being torn down should not block on a parquet
    rewrite it no longer cares about. The next :func:`pool` call after
    this rebuilds what it needs, which is what lets a test suite create
    and drop pools freely.
    """
    with _lock:
        executors = list(_pools.items())
        _pools.clear()
    for name, executor in executors:
        try:
            executor.shutdown(wait=wait)
        except Exception as exc:  # noqa: BLE001
            _log.warning("worker_pool_shutdown_error", pool=name, error=str(exc))


__all__ = [
    "DEBUG_ENV",
    "POOL_NAMES",
    "pool",
    "process_rss_bytes",
    "release_arrow_pool",
    "run_in_pool",
    "shutdown_pools",
]
