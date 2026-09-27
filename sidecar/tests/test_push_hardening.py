"""R2 hardening of the push surface, the artefact sync and the app lifecycle.

One module for the review's work package R2 (2026-09-27), in the order of
the brief:

1. the endpoint policy's parser-differential bypasses, and the sender
   never following a redirect;
2. subscribe abuse: key validation, the body cap, the rate limit, the
   garbage collection of dead rows;
3. the fan-out: shared per-host sessions, a thread pool, per-row error
   isolation;
5. ``upsert`` keeps the machine for a preferences-only edit, and the
   cycle's state write never clobbers an upsert that landed mid-cycle;
6. the post-processing table keeps its model when a bad file lands;
7. the sync validates with the real loaders, keeps ``.prev``, and the
   private routes answer ``If-None-Match`` with ``304``;
8. the app serves before its first cycle has finished.

The engine's None semantics (item 4) are in ``test_push_engine.py``, next
to the rest of the state machine.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
import requests
import structlog
from fastapi.testclient import TestClient
from requests.adapters import HTTPAdapter

from dmi_nowcast_core.postprocess import TARGET_ONSET, TARGET_WET
from dmi_nowcast_sidecar.app import create_app
from dmi_nowcast_sidecar.compute import CycleEngine, CycleResult
from dmi_nowcast_sidecar.config import Config
from dmi_nowcast_sidecar.push import fanout as fanout_mod
from dmi_nowcast_sidecar.push import service as service_mod
from dmi_nowcast_sidecar.push import vapid
from dmi_nowcast_sidecar.push.fanout import SendResult
from dmi_nowcast_sidecar.push.limits import TokenBucket
from dmi_nowcast_sidecar.push.paths import (
    resolved_db_path,
    resolved_onset_model_path,
    resolved_postprocess_path,
)
from dmi_nowcast_sidecar.push.postprocess import PostprocessTable
from dmi_nowcast_sidecar.push.routes import key_problem
from dmi_nowcast_sidecar.push.service import PushService
from dmi_nowcast_sidecar.push.store import PushStore
from dmi_nowcast_sidecar.scheduler import CycleScheduler
from dmi_nowcast_sidecar.sync import (
    CURVES_FILE,
    ONSET_MODEL_FILE,
    POSTPROCESS_FILE,
    QUALITY_FILE,
    STATION_POINTS_FILE,
    THRESHOLDS_FILE,
    ArtifactSync,
    target_path,
)
from tests import test_push_postprocess as tpp
from tests import test_quality_publication as tqp
from tests.test_push_backend import (  # noqa: F401 - fixtures
    API_KEY,
    AUTH,
    ENDPOINT_A,
    ENDPOINT_B,
    P256DH,
    RADAR_TS,
    RADAR_TS2,
    _new_sub,
    _state_with,
    _sub_body,
    client,
    geo,
    products,
    push_config,
    seeded_engine,
    sends,
    service,
)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# 1. The sender never follows a redirect
# ---------------------------------------------------------------------------


class _Redirecting(HTTPAdapter):
    """An adapter whose every answer is a 302 to the metadata address."""

    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []

    def send(self, request, **kwargs):  # type: ignore[override]
        self.urls.append(request.url)
        response = requests.Response()
        response.status_code = 302
        response.headers["Location"] = "https://169.254.169.254/latest/"
        response.url = request.url
        response.request = request
        response._content = b""
        return response


def test_the_push_session_does_not_follow_a_redirect() -> None:
    session = fanout_mod.new_session()
    adapter = _Redirecting()
    session.mount("https://", adapter)
    response = session.post("https://fcm.googleapis.com/fcm/send/x", data=b"x")
    assert response.status_code == 302
    assert adapter.urls == ["https://fcm.googleapis.com/fcm/send/x"]


def test_a_redirecting_push_service_is_a_failed_send_not_a_second_request() -> None:
    """End to end through pywebpush: real encryption, one request, a 302."""
    session = fanout_mod.new_session()
    adapter = _Redirecting()
    session.mount("https://", adapter)
    result = fanout_mod.send(
        endpoint=ENDPOINT_A, p256dh=P256DH, auth=AUTH,
        payload={"type": "test"},
        vapid_private_pem=vapid.generate_private_key_pem(),
        vapid_subject="mailto:ops@example.com", ttl_s=60, session=session,
    )
    assert result.ok is False and result.status == 302
    assert len(adapter.urls) == 1
    assert fanout_mod.is_transient(302) is False


def test_a_send_failure_logs_only_the_type_status_and_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(**kwargs):
        raise requests.ConnectionError(
            f"HTTPSConnectionPool: Max retries exceeded with url {ENDPOINT_A} "
            f"key {P256DH}",
        )

    monkeypatch.setattr(fanout_mod.pywebpush, "webpush", _boom)
    with structlog.testing.capture_logs() as logs:
        result = fanout_mod.send(
            endpoint=ENDPOINT_A, p256dh=P256DH, auth=AUTH, payload={},
            vapid_private_pem=vapid.generate_private_key_pem(),
            vapid_subject="mailto:ops@example.com", ttl_s=60,
        )
    assert result.error == "ConnectionError"
    text = json.dumps(logs, default=str)
    assert "fcm.googleapis.com" not in text and P256DH not in text
    assert "Max retries" not in text


# ---------------------------------------------------------------------------
# 2. Subscribe abuse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p256dh,auth", [
    ("B" + "x" * 86, AUTH),                  # right length, not a point
    (P256DH[:-4], AUTH),                     # truncated
    ("Ag" + P256DH[2:44], AUTH),             # compressed-looking prefix
    (P256DH + "+/", AUTH),                   # not base64url
    (P256DH, "AAECAwQFBgcICQoL"),            # 12-byte auth
    (P256DH, "AAECAwQFBgcICQoLDA0ODxA"),     # 17-byte auth
    (P256DH, "!!!!"),
])
def test_bad_subscription_keys_are_refused(p256dh: str, auth: str) -> None:
    assert isinstance(key_problem(p256dh, auth), str)


def test_real_keys_pass_with_or_without_padding() -> None:
    assert key_problem(P256DH, AUTH) is None
    assert key_problem(P256DH + "=", AUTH + "==") is None


def test_subscribe_with_an_invalid_point_is_400_and_writes_nothing(
    client: TestClient, push_config: Config,
) -> None:
    body = _sub_body()
    body["subscription"]["keys"]["p256dh"] = "B" + "x" * 86
    r = client.post("/api/push/subscribe", json=body)
    assert r.status_code == 400
    assert "P-256" in r.json()["detail"]
    store = PushStore(resolved_db_path(push_config))
    assert store.count() == 0
    store.close()


@pytest.mark.parametrize("bypass", [
    "https://127.0.0.1\\.fcm.googleapis.com/x",
    "https://127.0.0.1%5C.fcm.googleapis.com/x",
    "https://127.0.0.1。fcm.googleapis.com/x",
    "https://fcm.googleapis.com./x",
    "https://fcm.googleapis.com@127.0.0.1/x",
])
def test_subscribe_refuses_the_ssrf_bypasses(
    client: TestClient, bypass: str,
) -> None:
    r = client.post("/api/push/subscribe", json=_sub_body(bypass))
    assert r.status_code == 400, r.text


def test_an_oversize_body_is_413_before_parsing(
    client: TestClient, push_config: Config,
) -> None:
    big = json.dumps({**_sub_body(), "pad": "x" * 20_000})
    r = client.post(
        "/api/push/subscribe", content=big,
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 413


def test_an_oversize_chunked_body_is_413_too(client: TestClient) -> None:
    def chunks():
        for _ in range(40):
            yield b"x" * 1024

    r = client.post(
        "/api/push/subscribe", content=chunks(),
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 413


def test_a_hidden_operator_route_stays_404_not_413_in_public_mode(
    client: TestClient,
) -> None:
    """The public-mode gate is outside the limits: an anonymous probe of
    ``/api/push/test`` must not learn the route exists from a 413/429."""
    r = client.post("/api/push/test", content=b"x" * 40_000)
    assert r.status_code == 404
    for _ in range(15):
        assert client.post("/api/push/test", json={}).status_code == 404


def test_the_rate_limit_is_per_client_and_says_when_to_retry(
    client: TestClient,
) -> None:
    rate = 10
    codes = [
        client.post(
            "/api/push/unsubscribe", json={"endpoint": ENDPOINT_A},
            headers={"CF-Connecting-IP": "203.0.113.7"},
        ).status_code
        for _ in range(rate + 1)
    ]
    assert codes[:rate] == [200] * rate
    assert codes[rate] == 429
    last = client.post(
        "/api/push/unsubscribe", json={"endpoint": ENDPOINT_A},
        headers={"CF-Connecting-IP": "203.0.113.7"},
    )
    assert last.status_code == 429 and int(last.headers["Retry-After"]) >= 1
    # Another client is unaffected; reads are not limited at all.
    other = client.post(
        "/api/push/unsubscribe", json={"endpoint": ENDPOINT_A},
        headers={"CF-Connecting-IP": "198.51.100.9"},
    )
    assert other.status_code == 200
    assert client.get(
        "/api/push/config", headers={"CF-Connecting-IP": "203.0.113.7"},
    ).status_code == 200


def test_the_rate_limit_can_be_turned_off(
    push_config: Config, seeded_engine: CycleEngine,
) -> None:
    push_config.push.rate_limit_per_min = 0
    app = create_app(push_config, engine=seeded_engine, auto_start_scheduler=False)
    with TestClient(app) as c:
        for _ in range(30):
            r = c.post("/api/push/unsubscribe", json={"endpoint": ENDPOINT_A})
            assert r.status_code == 200


def test_the_token_bucket_refills_and_is_memory_bounded() -> None:
    now = [0.0]
    bucket = TokenBucket(6, max_clients=3, clock=lambda: now[0])
    assert all(bucket.take("a") == 0 for _ in range(6))
    assert bucket.take("a") == pytest.approx(10.0)   # 6/min = one per 10 s
    now[0] += 10.0
    assert bucket.take("a") == 0
    for key in ("b", "c", "d", "e"):
        bucket.take(key)
    assert len(bucket) == 3                           # "a" evicted first


class TestGarbageCollection:
    def _store(self, tmp_path: Path) -> PushStore:
        store = PushStore(tmp_path / "subs.sqlite")
        store.upsert(_new_sub(ENDPOINT_A))
        return store

    def _fail(self, store, *, transient, when, n=1) -> list[bool]:
        return [
            store.record_send(
                ENDPOINT_A, ok=False, transient=transient, now_utc=when,
                max_failures=5, stale_days=14,
            )
            for _ in range(n)
        ]

    def test_five_non_transient_failures_collect_the_row(self, tmp_path):
        store = self._store(tmp_path)
        t0 = datetime(2026, 9, 1, tzinfo=UTC)
        assert self._fail(store, transient=False, when=t0, n=4) == [False] * 4
        assert store.get(ENDPOINT_A).fail_streak == 4  # type: ignore[union-attr]
        assert self._fail(store, transient=False, when=t0) == [True]
        assert store.get(ENDPOINT_A) is None

    def test_a_success_clears_the_record(self, tmp_path):
        store = self._store(tmp_path)
        t0 = datetime(2026, 9, 1, tzinfo=UTC)
        self._fail(store, transient=False, when=t0, n=4)
        store.record_send(
            ENDPOINT_A, ok=True, transient=False, now_utc=t0,
            max_failures=5, stale_days=14,
        )
        row = store.get(ENDPOINT_A)
        assert row is not None and row.fail_streak == 0
        assert row.first_fail_utc is None

    def test_transient_failures_count_only_toward_the_14_day_rule(self, tmp_path):
        store = self._store(tmp_path)
        t0 = datetime(2026, 9, 1, tzinfo=UTC)
        assert self._fail(store, transient=True, when=t0, n=20) == [False] * 20
        assert store.get(ENDPOINT_A).fail_streak == 0  # type: ignore[union-attr]
        later = t0 + timedelta(days=13, hours=23)
        assert self._fail(store, transient=True, when=later) == [False]
        assert self._fail(
            store, transient=True, when=t0 + timedelta(days=14),
        ) == [True]
        assert store.get(ENDPOINT_A) is None

    def test_a_forbidden_push_service_answer_collects_through_the_service(
        self, service: PushService, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(
            fanout_mod, "send",
            lambda **kw: SendResult(
                ok=False, gone=False, status=403, error="WebPushException",
            ),
        )
        import asyncio

        for _ in range(4):
            asyncio.run(service.send_test(ENDPOINT_A))
        assert service.store.get(ENDPOINT_A) is not None
        counts = asyncio.run(service.send_test(ENDPOINT_A))
        assert counts["removed"] == 1
        assert service.store.get(ENDPOINT_A) is None


# ---------------------------------------------------------------------------
# 3. The fan-out
# ---------------------------------------------------------------------------


async def test_sends_share_one_session_per_host_across_worker_threads(
    service: PushService, seeded_engine: CycleEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service.config.push.persistence_obs = 1
    fcm = [f"https://fcm.googleapis.com/fcm/send/t{i}" for i in range(6)]
    moz = [f"https://updates.push.services.mozilla.com/wpush/v2/m{i}" for i in range(3)]
    for endpoint in fcm + moz:
        service.store.upsert(_new_sub(endpoint))
    seen: list[tuple[str, int, str]] = []
    lock = threading.Lock()

    def _send(**kwargs) -> SendResult:
        time.sleep(0.02)
        with lock:
            seen.append((
                kwargs["endpoint"], id(kwargs["session"]),
                threading.current_thread().name,
            ))
        return SendResult(ok=True, gone=False, status=201, error=None)

    monkeypatch.setattr(fanout_mod, "send", _send)
    await service.after_cycle(CycleResult(state=_state_with(RADAR_TS)))
    by_host: dict[str, set[int]] = {}
    for endpoint, session_id, _thread in seen:
        by_host.setdefault(endpoint.split("/")[2], set()).add(session_id)
    assert {len(v) for v in by_host.values()} == {1}
    assert len({sid for v in by_host.values() for sid in v}) == 2
    assert len({t for *_, t in seen}) > 1               # really concurrent
    assert service.last_fanout["sent"] == len(seen) >= 9  # type: ignore[index]
    # The sessions live across cycles and close with the service.
    service.close()
    assert service._sessions == {}


async def test_one_row_that_raises_costs_only_that_row(
    service: PushService, seeded_engine: CycleEngine, sends: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service.config.push.persistence_obs = 1
    service.store.upsert(_new_sub(ENDPOINT_B))    # both on the wet pixel
    real = service_mod.sample_point

    def _sample(products, geo, lat, lon, **kw):
        if _sample.calls == 0:
            _sample.calls += 1
            raise RuntimeError("corrupt grid for this row")
        _sample.calls += 1
        return real(products, geo, lat, lon, **kw)

    _sample.calls = 0
    monkeypatch.setattr(service_mod, "sample_point", _sample)
    with structlog.testing.capture_logs() as logs:
        await service.after_cycle(CycleResult(state=_state_with(RADAR_TS)))
    assert service.last_fanout["eval_errors"] == 1  # type: ignore[index]
    assert len(sends) == 1
    errors = [e for e in logs if e["event"] == "push_eval_error"]
    assert errors and errors[0]["error"] == "RuntimeError"


async def test_a_payload_failure_after_a_persisted_decision_still_fans_out(
    service: PushService, seeded_engine: CycleEngine, sends: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The queue built before a failure is always sent (``finally``)."""
    service.config.push.persistence_obs = 1
    service.store.upsert(_new_sub(ENDPOINT_B))
    real = service_mod.rain_incoming_payload
    calls = {"n": 0}

    def _payload(**kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("bad template")
        return real(**kwargs)

    monkeypatch.setattr(service_mod, "rain_incoming_payload", _payload)
    await service.after_cycle(CycleResult(state=_state_with(RADAR_TS)))
    assert len(sends) == 1
    assert service.last_fanout["eval_errors"] == 1  # type: ignore[index]


# ---------------------------------------------------------------------------
# 5. Upsert keeps the machine; the cycle never clobbers a mid-cycle upsert
# ---------------------------------------------------------------------------


def _disarm(store: PushStore, endpoint: str = ENDPOINT_A) -> None:
    assert store.update_state(
        endpoint, armed=False, streak=1,
        below_since_utc=RADAR_TS, last_eval_radar_ts=RADAR_TS,
        notified=True, below_streak=1, all_clear_sent=False,
    )


@pytest.mark.parametrize("edit", [
    {"quiet_enabled": True, "quiet_start": "23:00", "quiet_end": "06:00"},
    {"tz": "UTC"},
    {"lang": "en"},
    {"p256dh": P256DH + "x"},
    {},                                   # the browser re-sent the same thing
])
def test_a_preferences_only_edit_keeps_the_state(tmp_path: Path, edit) -> None:
    store = PushStore(tmp_path / "s.sqlite")
    store.upsert(_new_sub())
    _disarm(store)
    before = store.get(ENDPOINT_A)
    assert store.upsert(_new_sub(**edit)) is False
    row = store.get(ENDPOINT_A)
    assert row is not None and before is not None
    assert (row.armed, row.streak, row.notified, row.below_streak) == (
        False, 1, True, 1,
    )
    assert row.last_eval_radar_ts == RADAR_TS
    assert row.version == before.version + 1
    for key, value in edit.items():
        assert getattr(row, key) == value


@pytest.mark.parametrize("edit", [
    {"lat": 55.40}, {"lon": 10.40}, {"lead_min": 45},
    {"threshold_pct": 80}, {"threshold_pct": None},
])
def test_an_edit_of_the_rule_restarts_the_machine(tmp_path: Path, edit) -> None:
    store = PushStore(tmp_path / "s.sqlite")
    store.upsert(_new_sub())
    _disarm(store)
    store.upsert(_new_sub(**edit))
    row = store.get(ENDPOINT_A)
    assert row is not None
    assert (row.armed, row.streak, row.notified) == (True, 0, False)
    assert row.below_since_utc is None and row.last_eval_radar_ts is None


def test_a_state_write_with_a_stale_version_does_not_land(tmp_path: Path) -> None:
    store = PushStore(tmp_path / "s.sqlite")
    store.upsert(_new_sub())
    seen = store.get(ENDPOINT_A)
    store.upsert(_new_sub(lang="en"))            # lands "mid-cycle"
    assert store.update_state(
        ENDPOINT_A, armed=False, streak=1, below_since_utc=None,
        last_eval_radar_ts=RADAR_TS, expected_version=seen.version,  # type: ignore[union-attr]
    ) is False
    row = store.get(ENDPOINT_A)
    assert row is not None and row.armed is True and row.lang == "en"


async def test_the_cycle_does_not_act_on_a_row_edited_mid_cycle(
    service: PushService, seeded_engine: CycleEngine, sends: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service.config.push.persistence_obs = 1
    real = service.store.update_state
    edited = {"done": False}

    def _update_state(endpoint, **kwargs):
        if endpoint == ENDPOINT_A and not edited["done"]:
            edited["done"] = True
            service.store.upsert(_new_sub(ENDPOINT_A, tz="UTC"))
        return real(endpoint, **kwargs)

    monkeypatch.setattr(service.store, "update_state", _update_state)
    await service.after_cycle(CycleResult(state=_state_with(RADAR_TS)))
    assert sends == []                          # not persisted -> not sent
    assert service.last_fanout["stale_writes"] == 1  # type: ignore[index]
    row = service.store.get(ENDPOINT_A)
    assert row is not None and row.tz == "UTC" and row.armed is True


def test_the_store_migrates_in_the_new_columns(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "old.sqlite"
    store = PushStore(path)
    store.upsert(_new_sub())
    store.close()
    conn = sqlite3.connect(path)
    for column in ("version", "fail_streak", "first_fail_utc"):
        conn.execute(f"ALTER TABLE subscriptions DROP COLUMN {column}")
    conn.commit()
    conn.close()
    store = PushStore(path)
    row = store.get(ENDPOINT_A)
    assert row is not None
    assert (row.version, row.fail_streak, row.first_fail_utc) == (0, 0, None)
    store.close()


# ---------------------------------------------------------------------------
# 6. The model swap keeps the old model on a bad file
# ---------------------------------------------------------------------------


def test_a_bad_file_keeps_the_model_in_service(tmp_path: Path) -> None:
    path = tpp._write_model(tmp_path / "postprocess.json")
    table = PostprocessTable(path)
    table.load()
    model = table.model
    assert table.active

    path.write_text("{not json")
    with structlog.testing.capture_logs() as logs:
        assert table.maybe_reload() is True
    assert table.model is model and table.active
    assert any(e["event"] == "push_postprocess_kept_previous" for e in logs)
    # No re-parse every cycle while the bad file sits there.
    assert table.maybe_reload() is False

    path.write_text(json.dumps({**json.loads(model.dumps()), "target": "onset"}))
    table.maybe_reload()
    assert table.model is model                 # wrong target: kept too

    path.unlink()
    table.maybe_reload()
    assert table.model is None and not table.active   # gone: cleared


def test_the_model_is_never_none_while_a_new_one_is_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tpp._write_model(tmp_path / "postprocess.json")
    table = PostprocessTable(path)
    table.load()
    old = table.model
    seen_during_parse: list = []
    real = PostprocessTable._parse

    def _slow_parse(self, text):
        seen_during_parse.append(self.model)
        return real(self, text)

    monkeypatch.setattr(PostprocessTable, "_parse", _slow_parse)
    table.note_changed()
    table.maybe_reload()
    assert seen_during_parse == [old]


def test_the_routes_never_reload_the_engines_tables(
    client: TestClient, seeded_engine: CycleEngine, push_config: Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    for name in ("postprocess", "onset_postprocess"):
        table = getattr(seeded_engine, name)
        monkeypatch.setattr(
            table, "maybe_reload", lambda n=name: calls.append(n) or False,
        )
    client.get("/api/push/options")
    body = _sub_body()
    body.pop("threshold_pct")
    assert client.post("/api/push/subscribe", json=body).status_code == 200
    assert calls == []


# ---------------------------------------------------------------------------
# 7. Sync validates with the real loaders; routes answer 304
# ---------------------------------------------------------------------------


def _wet_model_doc() -> dict:
    return json.loads(tpp._model().dumps())


def _sync(tmp_path: Path, name: str, *responses):
    config = tqp._sync_config(tmp_path, files=[name])
    peer = tqp._Peer({name: list(responses)})
    return config, peer, ArtifactSync(config, client=peer.client())


@pytest.mark.parametrize("name,bad", [
    (POSTPROCESS_FILE, {"schema_version": 1, "models": {}}),
    (ONSET_MODEL_FILE, "wet-model"),           # a wet model in the onset slot
    (POSTPROCESS_FILE, "onset-model"),         # and the other way round
    (CURVES_FILE, {"metadata": {}, "curves": {}}),
    (CURVES_FILE, {"curves": {"30": {"raw_breakpoints": [0.0]}}}),
    (THRESHOLDS_FILE, {"schema_version": 99}),
    (STATION_POINTS_FILE, {"version": 2, "points": []}),
    (STATION_POINTS_FILE, {"version": 2, "points": [{"id": "x"}]}),
    (QUALITY_FILE, ["not", "a", "report"]),
])
def test_sync_refuses_a_document_its_consumer_cannot_use(
    tmp_path: Path, name: str, bad,
) -> None:
    if bad == "wet-model":
        bad = _wet_model_doc()
    elif bad == "onset-model":
        bad = {**_wet_model_doc(), "target": TARGET_ONSET}
    config, _peer, sync = _sync(tmp_path, name, tqp._ok(bad, etag='"bad"'))
    result = tqp.anyio_run(sync.sync_once())
    assert result.failed == 1
    assert "rejected" in (result.files[0].error or "")
    assert not target_path(config, name).exists()


def test_sync_keeps_prev_and_never_replaces_a_good_model_with_a_bad_one(
    tmp_path: Path,
) -> None:
    good = _wet_model_doc()
    newer = {**good, "fitted_at_utc": "2026-09-20T03:40:00+00:00"}
    bad = {**good, "target": TARGET_ONSET}
    config, peer, sync = _sync(
        tmp_path, POSTPROCESS_FILE,
        tqp._ok(good, etag='"v1"'), tqp._ok(newer, etag='"v2"'),
        tqp._ok(bad, etag='"v3"'), httpx.Response(304),
    )
    path = target_path(config, POSTPROCESS_FILE)
    assert path == resolved_postprocess_path(config)
    assert tqp.anyio_run(sync.sync_once()).updated == 1
    assert tqp.anyio_run(sync.sync_once()).updated == 1
    assert json.loads(path.read_text()) == newer
    prev = path.with_name(path.name + ".prev")
    assert json.loads(prev.read_text()) == good

    third = tqp.anyio_run(sync.sync_once())
    assert third.failed == 1
    assert json.loads(path.read_text()) == newer         # untouched
    assert json.loads(prev.read_text()) == good
    # The rejected body is not downloaded again: its ETag goes back too.
    fourth = tqp.anyio_run(sync.sync_once())
    assert fourth.files[0].status == "unchanged"
    sent = peer.requests[-1].headers["If-None-Match"]
    assert '"v2"' in sent and '"v3"' in sent
    assert not list(path.parent.glob("*.tmp"))


def test_sync_accepts_the_real_documents(tmp_path: Path) -> None:
    onset = {**_wet_model_doc(), "target": TARGET_ONSET}
    wet = _wet_model_doc()
    for name, doc in (
        (POSTPROCESS_FILE, wet),
        (ONSET_MODEL_FILE, onset),
        (CURVES_FILE, tqp.CURVES),
        (QUALITY_FILE, tqp.DOC),
        (STATION_POINTS_FILE, {"version": 2, "points": [
            {"id": "06180", "lat": 55.6, "lon": 12.6},
        ]}),
    ):
        config, _peer, sync = _sync(tmp_path / name.replace("/", "_"), name, tqp._ok(doc))
        result = tqp.anyio_run(sync.sync_once())
        assert result.ok and result.updated == 1, (name, result)
    assert TARGET_WET != TARGET_ONSET
    assert resolved_onset_model_path(config) is not None


def test_sync_after_a_restart_does_not_rewrite_identical_bytes(
    tmp_path: Path,
) -> None:
    config, _peer, sync = _sync(tmp_path, QUALITY_FILE, tqp._ok(tqp.DOC))
    assert tqp.anyio_run(sync.sync_once()).updated == 1
    mtime = target_path(config, QUALITY_FILE).stat().st_mtime_ns
    fresh = ArtifactSync(config, client=tqp._Peer({QUALITY_FILE: [tqp._ok(tqp.DOC)]}).client())
    assert tqp.anyio_run(fresh.sync_once()).files[0].status == "unchanged"
    assert target_path(config, QUALITY_FILE).stat().st_mtime_ns == mtime


def test_private_artefact_routes_answer_304_on_a_matching_etag(
    minimal_config: Config,
) -> None:
    path = resolved_postprocess_path(minimal_config)
    tpp._write_model(path)
    app = create_app(minimal_config, auto_start_scheduler=False)
    with TestClient(app) as c:
        first = c.get("/calibration/postprocess.json")
        assert first.status_code == 200
        etag = first.headers["ETag"]
        assert json.loads(first.content)["target"] == TARGET_WET
        again = c.get(
            "/calibration/postprocess.json", headers={"If-None-Match": etag},
        )
        assert again.status_code == 304 and again.content == b""
        assert again.headers["ETag"] == etag
        # A new file is a new ETag, so the peer downloads it.
        tpp._write_model(path, tpp._model())
        import os

        os.utime(path, ns=(time.time_ns(), time.time_ns() + 1_000_000))
        changed = c.get(
            "/calibration/postprocess.json", headers={"If-None-Match": etag},
        )
        assert changed.status_code == 200 and changed.headers["ETag"] != etag


def test_sync_and_the_private_route_roundtrip_a_304(
    minimal_config: Config, tmp_path: Path,
) -> None:
    """The public sync against the real private app: the second pass of an
    unchanged 7 MB model costs a 304, not a download."""
    tpp._write_model(resolved_postprocess_path(minimal_config))
    private = create_app(minimal_config, auto_start_scheduler=False)
    with TestClient(private) as c:
        def handler(request: httpx.Request) -> httpx.Response:
            r = c.get(
                request.url.path,
                headers={
                    k: v for k, v in request.headers.items()
                    if k.lower() == "if-none-match"
                },
            )
            return httpx.Response(
                r.status_code, content=r.content, headers=dict(r.headers),
            )

        config = tqp._sync_config(tmp_path / "public", files=[POSTPROCESS_FILE])
        sync = ArtifactSync(
            config, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        assert tqp.anyio_run(sync.sync_once()).updated == 1
        second = tqp.anyio_run(sync.sync_once())
        assert second.files[0].status == "unchanged"
        assert second.files[0].http_status == 304


# ---------------------------------------------------------------------------
# 8. The app serves before the first cycle has finished
# ---------------------------------------------------------------------------


def test_the_app_serves_while_the_first_cycle_is_still_running(
    minimal_config: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    tpp_state = _state_with(RADAR_TS)
    from dmi_nowcast_sidecar.storage import StateStore

    StateStore(minimal_config.storage.data_dir).write(tpp_state)
    engine = CycleEngine(minimal_config)
    release = threading.Event()
    started = threading.Event()

    async def _slow_cycle():
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.01)
        return CycleResult(state=_state_with(RADAR_TS2))

    monkeypatch.setattr(engine, "run_cycle", _slow_cycle)
    app = create_app(minimal_config, engine=engine, auto_start_scheduler=True)
    try:
        with TestClient(app) as c:
            assert started.wait(5.0)
            health = c.get("/healthz")
            assert health.status_code == 200
            assert health.json()["last_cycle"].startswith("2026-08-28T12:00")
            assert c.get("/state.json").status_code == 200
            release.set()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if c.get("/healthz").json()["last_cycle"].startswith(
                    "2026-08-28T12:10",
                ):
                    break
                time.sleep(0.02)
            else:
                pytest.fail("the background first cycle never completed")
    finally:
        release.set()


async def test_scheduler_start_without_wait_returns_before_the_cycle(
    minimal_config: Config,
) -> None:
    import asyncio

    engine = CycleEngine(minimal_config)
    gate = asyncio.Event()
    done: list[bool] = []

    async def _cycle():
        await gate.wait()
        done.append(True)
        return CycleResult(state=None, error="x")

    engine.run_cycle = _cycle  # type: ignore[method-assign]
    scheduler = CycleScheduler(engine, interval_min=5, jitter_sec=0)
    await asyncio.wait_for(scheduler.start(run_immediately=True, wait=False), 1.0)
    assert done == []
    gate.set()
    for _ in range(200):
        if done:
            break
        await asyncio.sleep(0.01)
    assert done == [True]
    await scheduler.shutdown()
