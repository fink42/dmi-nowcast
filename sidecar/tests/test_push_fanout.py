"""One Web Push delivery: verdicts, error isolation, and no leaks.

``pywebpush.webpush`` is monkeypatched throughout — no test ever touches
a real push service — but the VAPID key is a real EC key, so the signing
path is exercised for real.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
import pywebpush
import structlog
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from dmi_nowcast_sidecar.push import fanout
from dmi_nowcast_sidecar.push.fanout import SendResult, send

TOKEN = "dGVzdC10b2tlbi1BQkNERUZHSElKS0xNTk9Q"
ENDPOINT = f"https://fcm.googleapis.com/fcm/send/{TOKEN}"
P256DH = "BJ3xk9pQ2sVQm0m9Ex4mS3aWQ1r2zZ9r5W6y8Uu7Tt6Ss5Rr4Qq3Pp2Oo1Nn0Mm"
AUTH = "kZm9vYmFyYmF6cXV1eA"
SUBJECT = "mailto:ops@example.invalid"
PAYLOAD = {"type": "rain_incoming", "title": "Regn på vej — ca. 18 min"}

_KEY = ec.generate_private_key(ec.SECP256R1())
PEM = _KEY.private_bytes(
    serialization.Encoding.PEM,
    serialization.PrivateFormat.PKCS8,
    serialization.NoEncryption(),
)


@dataclass
class FakeResponse:
    status_code: int
    reason: str = ""
    text: str = ""


class Recorder:
    """Stands in for ``pywebpush.webpush``; records how it was called."""

    def __init__(self, result=None, raises: Exception | None = None) -> None:
        self.result = result
        self.raises = raises
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        # pywebpush mutates vapid_claims in place, so snapshot it now.
        snapshot = dict(kwargs)
        snapshot["vapid_claims"] = dict(kwargs.get("vapid_claims") or {})
        self.calls.append(snapshot)
        if self.raises is not None:
            raise self.raises
        return self.result


def _install(monkeypatch: pytest.MonkeyPatch, recorder: Recorder) -> Recorder:
    monkeypatch.setattr(pywebpush, "webpush", recorder)
    return recorder


def _send(**overrides) -> SendResult:
    kwargs = dict(
        endpoint=ENDPOINT,
        p256dh=P256DH,
        auth=AUTH,
        payload=PAYLOAD,
        vapid_private_pem=PEM,
        vapid_subject=SUBJECT,
        ttl_s=900,
    )
    kwargs.update(overrides)
    return send(**kwargs)


def _webpush_error(status: int, text: str = "") -> pywebpush.WebPushException:
    return pywebpush.WebPushException(
        f"Push failed: {status}",
        response=FakeResponse(status_code=status, reason="", text=text),
    )


# --------------------------------------------------------------------------
# Happy path and call shape
# --------------------------------------------------------------------------


def test_successful_send(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, Recorder(result=FakeResponse(status_code=201)))
    result = _send()
    assert result == SendResult(ok=True, gone=False, status=201, error=None)


@pytest.mark.parametrize("status", [200, 201, 202])
def test_any_accepted_status_is_ok(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    _install(monkeypatch, Recorder(result=FakeResponse(status_code=status)))
    assert _send().ok is True


def test_call_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _install(
        monkeypatch, Recorder(result=FakeResponse(status_code=201))
    )
    _send(ttl_s=900, timeout_s=7.5)

    (call,) = recorder.calls
    assert call["subscription_info"] == {
        "endpoint": ENDPOINT,
        "keys": {"p256dh": P256DH, "auth": AUTH},
    }
    assert json.loads(call["data"]) == PAYLOAD
    assert call["ttl"] == 900
    assert call["timeout"] == 7.5
    assert call["headers"] == {"Urgency": "high"}
    assert call["vapid_claims"] == {"sub": SUBJECT}


def test_default_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _install(
        monkeypatch, Recorder(result=FakeResponse(status_code=201))
    )
    _send()
    assert recorder.calls[0]["timeout"] == 10.0


def test_private_key_is_a_signer_not_a_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Handing pywebpush a py_vapid instance keeps the PEM off the
    # filesystem — a string would be probed with os.path.isfile.
    recorder = _install(
        monkeypatch, Recorder(result=FakeResponse(status_code=201))
    )
    _send()
    signer = recorder.calls[0]["vapid_private_key"]
    assert isinstance(signer, pywebpush.Vapid01)
    header = signer.sign({"sub": SUBJECT, "aud": "https://fcm.googleapis.com"})
    assert header["Authorization"].startswith("vapid ")


def test_vapid_claims_are_a_fresh_dict_each_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # pywebpush writes `aud` and `exp` into whatever dict it is handed; a
    # shared one would carry a stale expiry into the next fan-out.
    recorder = _install(
        monkeypatch, Recorder(result=FakeResponse(status_code=201))
    )

    def mutating(**kwargs):
        # Record what arrived, *then* scribble on it the way pywebpush does.
        result = recorder(**kwargs)
        kwargs["vapid_claims"]["aud"] = "https://fcm.googleapis.com"
        kwargs["vapid_claims"]["exp"] = 1
        return result

    monkeypatch.setattr(pywebpush, "webpush", mutating)
    _send()
    _send()
    assert [c["vapid_claims"] for c in recorder.calls] == [
        {"sub": SUBJECT},
        {"sub": SUBJECT},
    ]


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", [404, 410])
def test_dead_subscription_is_gone(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    _install(monkeypatch, Recorder(raises=_webpush_error(status)))
    result = _send()
    assert result.gone is True
    assert result.ok is False
    assert result.status == status


@pytest.mark.parametrize("status", [400, 401, 403, 413, 429, 500, 503])
def test_other_http_errors_are_not_gone(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    _install(monkeypatch, Recorder(raises=_webpush_error(status)))
    result = _send()
    assert result.ok is False
    assert result.gone is False
    assert result.status == status
    assert result.error is not None
    assert str(status) in result.error


def test_status_read_from_an_aiohttp_style_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass
    class Aiohttpish:
        status: int

    _install(
        monkeypatch,
        Recorder(
            raises=pywebpush.WebPushException(
                "Push failed", response=Aiohttpish(status=410)
            )
        ),
    )
    result = _send()
    assert (result.status, result.gone) == (410, True)


def test_webpush_error_without_a_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(
        monkeypatch,
        Recorder(raises=pywebpush.WebPushException("VAPID dict missing key")),
    )
    result = _send()
    assert result == SendResult(
        ok=False, gone=False, status=None, error="WebPushException"
    )


def test_timeout_is_a_soft_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, Recorder(raises=TimeoutError("timed out")))
    result = _send()
    assert result.ok is False
    assert result.gone is False
    assert result.status is None
    assert result.error is not None
    assert result.error.startswith("TimeoutError")


@pytest.mark.parametrize(
    "exc",
    [
        TimeoutError("timed out"),
        ConnectionError("connection reset"),
        ValueError("bad key"),
        RuntimeError("boom"),
        OSError(101, "network unreachable"),
    ],
)
def test_send_never_raises(
    monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    _install(monkeypatch, Recorder(raises=exc))
    result = _send()
    assert result.ok is False
    assert type(exc).__name__ in (result.error or "")


def test_a_broken_pem_is_a_failure_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _install(
        monkeypatch, Recorder(result=FakeResponse(status_code=201))
    )
    result = _send(vapid_private_pem=b"-----BEGIN PRIVATE KEY-----\nnope\n")
    assert result.ok is False
    assert result.status is None
    assert recorder.calls == []


# --------------------------------------------------------------------------
# Nothing identifying escapes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        ConnectionError(
            f"HTTPSConnectionPool: Max retries exceeded with url: {ENDPOINT}"
        ),
        RuntimeError(f"failed to POST {ENDPOINT} with auth {AUTH}"),
        ValueError(f"bad key {P256DH}"),
    ],
)
def test_error_strings_never_contain_the_endpoint_or_keys(
    monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    _install(monkeypatch, Recorder(raises=exc))
    error = _send().error or ""
    assert ENDPOINT not in error
    assert TOKEN not in error
    assert P256DH not in error
    assert AUTH not in error
    assert type(exc).__name__ in error


def test_webpush_error_body_is_not_echoed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Some push services quote the registration token back in the error
    # document; the response body never reaches our error string.
    _install(
        monkeypatch,
        Recorder(
            raises=_webpush_error(
                400, text=f'{{"error": "InvalidRegistration: {TOKEN}"}}'
            )
        ),
    )
    error = _send().error or ""
    assert TOKEN not in error
    assert "400" in error


def test_error_strings_stay_short(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, Recorder(raises=RuntimeError("x" * 5000)))
    assert len(_send().error or "") <= 200


def test_logs_identify_a_subscription_only_by_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, Recorder(result=FakeResponse(status_code=201)))
    with structlog.testing.capture_logs() as logs:
        _send()
    _install(monkeypatch, Recorder(raises=_webpush_error(410)))
    with structlog.testing.capture_logs() as more:
        _send()
    _install(monkeypatch, Recorder(raises=ConnectionError(ENDPOINT)))
    with structlog.testing.capture_logs() as failures:
        _send()

    every = logs + more + failures
    assert every, "the fan-out should say something about every delivery"
    for entry in every:
        rendered = json.dumps(entry, default=str)
        assert ENDPOINT not in rendered
        assert TOKEN not in rendered
        assert P256DH not in rendered
        assert AUTH not in rendered
        assert entry["sub"] == fanout._sub_id(ENDPOINT)


def test_sub_id_is_a_short_stable_hash() -> None:
    first = fanout._sub_id(ENDPOINT)
    assert first == fanout._sub_id(ENDPOINT)
    assert len(first) == 10
    assert all(c in "0123456789abcdef" for c in first)
    assert first != fanout._sub_id(ENDPOINT + "x")
    assert TOKEN[:6] not in first
