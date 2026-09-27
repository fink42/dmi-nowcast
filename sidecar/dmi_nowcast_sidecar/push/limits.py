"""Abuse limits in front of ``/api/push/*``: a body cap and a rate limit.

The subscribe route is anonymous on the public site and every accepted
call writes a row, so two cheap limits sit in front of it, in one pure
ASGI middleware that runs BEFORE FastAPI reads or parses the body:

- **Body cap.** A request under ``/api/push/`` whose body is larger than
  ``push.max_request_bytes`` is answered ``413`` — on its
  ``Content-Length`` when it declares one, else while it is being read,
  so a chunked body cannot stream past the cap either. A real subscribe
  body is well under 2 KB.
- **Rate limit.** An in-process token bucket per client over the routes
  that write or send (subscribe, unsubscribe, test): ``rate_per_min``
  requests of burst, refilled at ``rate_per_min`` per minute; ``429`` with
  ``Retry-After`` when the bucket is empty. The client is the
  ``CF-Connecting-IP`` header when present (the public site sits behind
  Cloudflare, where the peer is always a Cloudflare edge) and the peer
  address otherwise. Memory is bounded: at most ``max_clients`` buckets,
  least-recently-used evicted first — an evicted client simply starts
  again with a full bucket.

Per process, not shared, and not persisted: this is a speed bump against
a script, not an accounting system. Nothing here logs a client address.
"""
from __future__ import annotations

import math
import time
from collections import OrderedDict
from typing import Callable, Iterable

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

PREFIX = "/api/push/"
#: The routes that write a row or trigger a send.
RATE_LIMITED_PATHS = frozenset({
    "/api/push/subscribe",
    "/api/push/unsubscribe",
    "/api/push/test",
})


class TokenBucket:
    """Per-key token buckets with LRU eviction. Event-loop only (no lock)."""

    def __init__(
        self,
        rate_per_min: float,
        *,
        max_clients: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.capacity = float(rate_per_min)
        self.refill_per_s = float(rate_per_min) / 60.0
        self.max_clients = int(max_clients)
        self._clock = clock
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._buckets)

    def take(self, key: str) -> float:
        """Take one token: 0.0 when allowed, else seconds until one is free."""
        now = self._clock()
        tokens, last = self._buckets.pop(key, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - last) * self.refill_per_s)
        wait = 0.0
        if tokens >= 1.0:
            tokens -= 1.0
        else:
            wait = (1.0 - tokens) / self.refill_per_s
        self._buckets[key] = (tokens, now)
        while len(self._buckets) > self.max_clients:
            self._buckets.popitem(last=False)
        return wait


def client_key(scope: Scope) -> str:
    """``CF-Connecting-IP`` when present, else the peer host."""
    for name, value in scope.get("headers") or ():
        if name == b"cf-connecting-ip":
            text = value.decode("latin-1").strip()
            if text:
                return text
    client = scope.get("client")
    return str(client[0]) if client else "unknown"


def _content_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers") or ():
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


class PushLimitsMiddleware:
    """Body cap + rate limit for ``/api/push/*`` (see module docstring)."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_bytes: int,
        rate_per_min: int,
        limited_paths: Iterable[str] = RATE_LIMITED_PATHS,
        max_clients: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.app = app
        self.max_bytes = int(max_bytes)
        self.limited_paths = frozenset(limited_paths)
        self.bucket = (
            TokenBucket(rate_per_min, max_clients=max_clients, clock=clock)
            if rate_per_min > 0 else None
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith(PREFIX):
            await self.app(scope, receive, send)
            return

        if self.bucket is not None and scope["path"] in self.limited_paths:
            wait = self.bucket.take(client_key(scope))
            if wait > 0:
                response = JSONResponse(
                    {"detail": "too many requests; slow down"},
                    status_code=429,
                    headers={"Retry-After": str(max(1, math.ceil(wait)))},
                )
                await response(scope, receive, send)
                return

        declared = _content_length(scope)
        if declared is not None and declared > self.max_bytes:
            await self._too_large(scope, receive, send)
            return

        # Read the whole body here, bounded, then hand it on. Routes under
        # this prefix take small JSON bodies; nothing streams.
        chunks: list[bytes] = []
        size = 0
        more = True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.max_bytes:
                await self._too_large(scope, receive, send)
                return
            chunks.append(chunk)
            more = bool(message.get("more_body", False))
        body = b"".join(chunks)
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

    async def _too_large(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            {"detail": f"request body exceeds {self.max_bytes} bytes"},
            status_code=413,
        )
        await response(scope, receive, send)


__all__ = [
    "PREFIX",
    "RATE_LIMITED_PATHS",
    "PushLimitsMiddleware",
    "TokenBucket",
    "client_key",
]
