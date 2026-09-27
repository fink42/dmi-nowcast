"""One persistent thread pool for the nowcast's data-parallel loops.

VENDORING MODIFICATION 8 (performance, bit-identical output). Not an
upstream pysteps module. Upstream parallelises the STEPS member loop with
``dask.delayed`` when dask is importable; the service does not ship dask,
so every member ran serially. This module replaces those dask branches with
a small work-sharing helper over ONE module-level
:class:`~concurrent.futures.ThreadPoolExecutor`, created lazily on first use
and kept for the life of the process — never a pool per cycle (the VM runs
with ``MALLOC_ARENA_MAX=2`` after a glibc-arena leak; thread churn is what
that setting cannot absorb).

Why the output stays bit-identical: every task handed to :func:`run_each`
computes one independent item (one ensemble member with its own random
generator, or one block of rows of a ``map_coordinates`` call) and writes
only its own slot, so which thread runs it — and in what order — cannot
change a single value.

Nested use runs serially. A task already running under :func:`run_each`
(e.g. a STEPS member whose extrapolation would chunk its
``map_coordinates`` rows) does not submit to the pool again; that is what
keeps a fixed-size pool from dead-locking on itself.

``workers`` is the TOTAL parallelism, the calling thread included: the pool
holds ``workers - 1`` threads and the caller does a share of the work while
it waits. Default 4 (``DMI_NOWCAST_POOL_WORKERS`` overrides; 1 disables all
threading and restores the exact serial code path).
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Callable, TypeVar

T = TypeVar("T")

_DEFAULT_WORKERS = 4


def _env_workers() -> int:
    raw = os.environ.get("DMI_NOWCAST_POOL_WORKERS", "")
    try:
        n = int(raw) if raw.strip() else _DEFAULT_WORKERS
    except ValueError:
        n = _DEFAULT_WORKERS
    return max(1, min(n, 32))


_workers = _env_workers()
_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()
_local = threading.local()


def workers() -> int:
    """Total parallelism (caller included) :func:`run_each` will use."""
    return _workers


def set_workers(n: int) -> None:
    """Change the parallelism. Takes effect for the NEXT pool; call before
    the first parallel loop (e.g. at service start). ``1`` = serial."""
    global _workers, _pool
    n = max(1, int(n))
    with _pool_lock:
        if n == _workers:
            return
        _workers = n
        old, _pool = _pool, None
    if old is not None:
        old.shutdown(wait=True)


def in_worker() -> bool:
    """True inside a task that :func:`run_each` is running."""
    return getattr(_local, "active", False)


def _get_pool() -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(
                max_workers=max(1, _workers - 1),
                thread_name_prefix="nowcast-pool",
            )
        return _pool


def run_each(
    fn: Callable[[int], T], n_items: int, max_workers: int | None = None,
) -> list[T]:
    """``[fn(0), fn(1), …, fn(n_items - 1)]``, computed in parallel.

    Results come back in index order. Items are handed out one at a time to
    up to :func:`workers` runners (the caller is one of them), so uneven
    items balance themselves. The first exception raised by any item is
    re-raised here after every runner has stopped; items not yet started
    when it happened are skipped.

    Runs serially — the plain loop — when parallelism is 1, when there is
    at most one item, or when called from inside another ``run_each`` task.
    ``max_workers`` caps the parallelism of this one call (never above the
    pool's).
    """
    n = int(n_items)
    k = min(_workers, n)
    if max_workers is not None:
        k = min(k, int(max_workers))
    if k <= 1 or in_worker():
        return [fn(i) for i in range(n)]

    results: list = [None] * n
    errors: list[BaseException] = []
    lock = threading.Lock()
    next_index = [0]

    def runner() -> None:
        prev = getattr(_local, "active", False)
        _local.active = True
        try:
            while True:
                with lock:
                    if errors or next_index[0] >= n:
                        return
                    i = next_index[0]
                    next_index[0] += 1
                try:
                    results[i] = fn(i)
                except BaseException as exc:  # noqa: BLE001 — re-raised below
                    with lock:
                        errors.append(exc)
                    return
        finally:
            _local.active = prev

    pool = _get_pool()
    futures = [pool.submit(runner) for _ in range(k - 1)]
    try:
        runner()
    finally:
        wait(futures)
    if errors:
        raise errors[0]
    return results
