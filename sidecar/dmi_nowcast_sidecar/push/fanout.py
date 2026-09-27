"""One encrypted Web Push delivery.

``send`` is deliberately narrow: it takes a subscription's endpoint and
keys, an already-localised payload and the VAPID material, and returns a
verdict. It never raises — a fan-out over hundreds of subscriptions must
not die on one dead device — and it never lets an endpoint or a key into
a log line or an error string. Endpoints are the subscriber's identity
and the tokens in them are bearer credentials for pushing to that
device; a subscription is identified in logs by
``sha256(endpoint)[:10]``, and a failure is logged as the exception
TYPE and the HTTP status only — exception text from ``requests`` carries
the URL, and push services echo tokens in error bodies.

**No redirects.** Every send goes through a :func:`new_session` session,
which refuses to follow a 3xx: the endpoint was checked against the
allow-list (``endpoint_policy``), the ``Location`` it answers with was
not. A 3xx is therefore a failed send, never a second request.

``gone`` is the only verdict the caller must act on beyond retry
bookkeeping: a 404 or 410 from the push service means the subscription
is dead and the row should be deleted.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache

import pywebpush
import requests
import structlog
from requests.adapters import HTTPAdapter

_log = structlog.get_logger(__name__)

__all__ = ["SendResult", "new_session", "send", "is_transient"]

#: Errors are truncated to this before being stored or logged.
_MAX_ERROR_CHARS = 160


@dataclass(frozen=True)
class SendResult:
    ok: bool
    #: 404/410 from the push service — the caller deletes the subscription.
    gone: bool
    status: int | None
    #: Short and sanitised; never contains the endpoint or the keys.
    error: str | None


class _NoRedirectSession(requests.Session):
    """A ``requests.Session`` that never follows a redirect.

    ``pywebpush`` calls ``session.post(endpoint, ...)``, and ``requests``
    follows redirects on POST by default; forcing ``allow_redirects`` off
    here is the one place that cannot be forgotten by a caller.
    """

    def request(self, *args, **kwargs):  # type: ignore[override]
        kwargs["allow_redirects"] = False
        return super().request(*args, **kwargs)


def new_session(pool_size: int = 8) -> requests.Session:
    """A keep-alive session for one push host; never follows redirects."""
    session = _NoRedirectSession()
    adapter = HTTPAdapter(pool_connections=1, pool_maxsize=max(1, int(pool_size)))
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def is_transient(status: int | None) -> bool:
    """Is a failed send worth retrying on this subscription later?

    No status (network error, timeout), 408, 429 and 5xx are the push
    service's or the network's problem; any other 4xx is about THIS
    subscription and counts toward its garbage collection (``store``).
    404/410 are not asked about — they delete the row at once.
    """
    if status is None:
        return True
    return status in (408, 429) or status >= 500


def _sub_id(endpoint: str) -> str:
    """Stable, non-reversible log handle for a subscription.

    Duplicated from ``store`` on purpose: this module stays importable
    without pulling the persistence layer in.
    """
    return hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:10]


@lru_cache(maxsize=4)
def _vapid_from_pem(pem: bytes):
    """A ``py_vapid`` signer built from PEM bytes, never from a file.

    ``pywebpush`` would otherwise treat the string as a path (or re-parse
    it per send); handing it a ``Vapid`` instance keeps the private key
    off the filesystem and out of every code path that logs arguments.
    Cached because a fan-out re-uses the same key for every subscription.
    """
    return pywebpush.Vapid.from_pem(pem)


def _redact(text: str, endpoint: str, p256dh: str, auth: str) -> str:
    """Strip anything subscription-identifying out of a message.

    ``requests`` puts the full URL into connection errors, so the raw
    exception text is not safe to keep. The endpoint's last path segment
    is the device token and is scrubbed on its own too, in case a library
    echoed only that part back.
    """
    secrets = [endpoint, endpoint.rsplit("/", 1)[-1], p256dh, auth]
    for secret in secrets:
        if secret and len(secret) > 6:
            text = text.replace(secret, "<redacted>")
    text = " ".join(text.split())
    if len(text) > _MAX_ERROR_CHARS:
        text = text[:_MAX_ERROR_CHARS] + "…"
    return text


def _status_of(exc: Exception) -> int | None:
    """HTTP status from a ``WebPushException``, whichever response it carries."""
    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("status_code", "status"):
            value = getattr(response, attr, None)
            if isinstance(value, int):
                return value
    value = getattr(exc, "status_code", None)
    return value if isinstance(value, int) else None


def send(
    *,
    endpoint: str,
    p256dh: str,
    auth: str,
    payload: dict,
    vapid_private_pem: bytes,
    vapid_subject: str,
    ttl_s: int,
    timeout_s: float = 10.0,
    session: requests.Session | None = None,
) -> SendResult:
    """Deliver one payload. Blocking (``requests``); run it off the loop.

    ``session`` is the shared per-host session (connection reuse); without
    one a fresh no-redirect session is used for this one send.
    """
    sub_id = _sub_id(endpoint)
    try:
        response = pywebpush.webpush(
            subscription_info={
                "endpoint": endpoint,
                "keys": {"p256dh": p256dh, "auth": auth},
            },
            data=json.dumps(payload),
            vapid_private_key=_vapid_from_pem(vapid_private_pem),
            # A fresh dict every call: pywebpush writes `aud` and `exp`
            # into whatever it is given, and a stale `exp` would be reused.
            vapid_claims={"sub": vapid_subject},
            ttl=ttl_s,
            timeout=timeout_s,
            headers={"Urgency": "high"},
            requests_session=session if session is not None else new_session(1),
        )
    except pywebpush.WebPushException as exc:
        status = _status_of(exc)
        gone = status in (404, 410)
        error = f"WebPushException: HTTP {status}" if status else "WebPushException"
        # Neither the response body nor the reason phrase is kept: push
        # services echo the registration token back in some error
        # documents, and only the status is needed to act.
        _log.info(
            "push_send_rejected", sub=sub_id, status=status, gone=gone
        )
        return SendResult(ok=False, gone=gone, status=status, error=error)
    except Exception as exc:  # noqa: BLE001 - one dead sub must not stop a fan-out
        # The type only: ``requests`` puts the full URL (the device token)
        # into connection-error text.
        error = _redact(type(exc).__name__, endpoint, p256dh, auth)
        _log.warning("push_send_failed", sub=sub_id, error=error)
        return SendResult(ok=False, gone=False, status=None, error=error)

    status = getattr(response, "status_code", None)
    _log.info("push_sent", sub=sub_id, status=status)
    return SendResult(ok=True, gone=False, status=status, error=None)
