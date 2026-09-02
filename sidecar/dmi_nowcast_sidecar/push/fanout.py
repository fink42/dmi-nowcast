"""One encrypted Web Push delivery.

``send`` is deliberately narrow: it takes a subscription's endpoint and
keys, an already-localised payload and the VAPID material, and returns a
verdict. It never raises — a fan-out over hundreds of subscriptions must
not die on one dead device — and it never lets an endpoint or a key into
a log line or an error string. Endpoints are the subscriber's identity
and the tokens in them are bearer credentials for pushing to that
device; a subscription is identified in logs by
``sha256(endpoint)[:10]``.

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
import structlog

_log = structlog.get_logger(__name__)

__all__ = ["SendResult", "send"]

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
) -> SendResult:
    """Deliver one payload. Blocking (``requests``); run it off the loop."""
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
        )
    except pywebpush.WebPushException as exc:
        status = _status_of(exc)
        gone = status in (404, 410)
        reason = getattr(getattr(exc, "response", None), "reason", None)
        error = f"WebPushException: HTTP {status}" if status else "WebPushException"
        if reason:
            error = f"{error} {reason}"
        # The response body is not appended: push services echo the
        # registration token back in some error documents.
        _log.info(
            "push_send_rejected", sub=sub_id, status=status, gone=gone
        )
        return SendResult(ok=False, gone=gone, status=status, error=error)
    except Exception as exc:  # noqa: BLE001 - one dead sub must not stop a fan-out
        error = _redact(
            f"{type(exc).__name__}: {exc}", endpoint, p256dh, auth
        )
        _log.warning("push_send_failed", sub=sub_id, error=error)
        return SendResult(ok=False, gone=False, status=None, error=error)

    status = getattr(response, "status_code", None)
    _log.info("push_sent", sub=sub_id, status=status)
    return SendResult(ok=True, gone=False, status=status, error=None)
