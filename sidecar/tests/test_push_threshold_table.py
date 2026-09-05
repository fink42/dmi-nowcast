"""G2 — the fitted threshold table, from the file to the notification.

Phase G takes the threshold away from the subscriber: they pick a horizon,
and the percent that horizon warns at comes from a table fitted nightly
against the rain gauges. That is a rule the whole service reads from one
small file which is replaced under a running process, so the things worth
testing are the seams:

1. **The reader is total.** No file, a truncated file, a document of the
   wrong shape — every one of them means "not fitted yet" and the shipped
   fallback, never an exception on a request path or in a fan-out.
2. **One table, three readers.** ``/api/push/options`` promises a percent,
   ``/api/push/subscribe`` echoes the one the row will be evaluated at,
   and the fan-out evaluates it. They share an instance so they cannot
   drift apart, and a swap only happens between fan-outs.
3. **The override survives.** The API still takes an explicit
   ``threshold_pct``; a row that has one is pinned to it and no refit
   moves it.
4. **Old rows migrate to null.** Existing subscriptions carry a percent
   picked from a menu that no longer exists; the migration clears it so
   they follow the table, and they must keep evaluating across it.

Offline and synthetic throughout: no radar, no ODIM, no network. The
sampler is stubbed where a real one would need a composite.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import structlog
from fastapi.testclient import TestClient

from dmi_nowcast_core.push_thresholds import SCHEMA_VERSION
from dmi_nowcast_sidecar.app import create_app
from dmi_nowcast_sidecar.config import Config, PushConfig
from dmi_nowcast_sidecar.push import service as service_mod
from dmi_nowcast_sidecar.push.paths import (
    THRESHOLDS_FILENAME,
    resolved_db_path,
    resolved_thresholds_path,
)
from dmi_nowcast_sidecar.push.service import PushService
from dmi_nowcast_sidecar.push.store import NewSubscription, PushStore
from dmi_nowcast_sidecar.push.thresholds import ThresholdTable
from dmi_nowcast_sidecar.sync import THRESHOLDS_FILE, target_path

API_KEY = "operator-key"
SUBJECT = "mailto:ops@example.com"
HOME_LAT, HOME_LON = 55.33, 10.32
ENDPOINT_A = "https://fcm.googleapis.com/fcm/send/AAAAAAAAAAA-token-a"
ENDPOINT_B = "https://updates.push.services.mozilla.com/wpush/v2/token-b"
P256DH = "B" + "x" * 86
AUTH = "y" * 22
RADAR_TS = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------

def _lead_row(threshold: int | None, **over) -> dict:
    row = {
        "threshold_pct": threshold,
        "insufficient": threshold is None,
        "f1": 0.42, "precision": 0.48, "recall": 0.37,
        "far": 0.55, "csi": 0.27,
        "warnings": 210, "hits": 94, "false_alarms": 101,
        "misses": 150, "late": 15,
        "plateau": None if threshold is None else [threshold - 5, threshold + 5],
        "radar_plateau": None,
        "agrees_with_radar": None,
    }
    row.update(over)
    return row


def _table_doc(leads: dict[str, int | None] | None = None, **over) -> dict:
    doc = {
        "schema_version": SCHEMA_VERSION,
        "fitted_at_utc": "2026-09-05T02:11:07+00:00",
        "objective": {
            "metric": "f1", "min_useful_lead_min": 5.0, "plateau_frac": 0.95,
            "min_warnings": 30, "rearm_after_min": 60, "persistence_obs": 1,
            "tolerance_min": 10, "dry_min": 30,
        },
        "window": {
            "from": "2026-07-01T00:00:00+00:00",
            "to": "2026-09-01T00:00:00+00:00",
            "days": 62, "stations": 97, "rows": 1841203,
        },
        "fallback_threshold_pct": 40,
        "leads": {
            key: _lead_row(value)
            for key, value in (leads or {"20": 65, "30": 45, "45": None}).items()
        },
    }
    doc.update(over)
    return doc


def _write_table(path: Path, doc: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc if doc is not None else _table_doc()))
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def push_config(minimal_config: Config) -> Config:
    """Push on, public mode with an operator key — the deployed shape."""
    minimal_config.push = PushConfig(
        enabled=True, vapid_subject=SUBJECT, lead_options=[20, 30, 45, 60],
    )
    minimal_config.server.public_mode = True
    minimal_config.server.api_key = API_KEY
    return minimal_config


@pytest.fixture
def client(push_config: Config):
    """No cycle has run: no products, so /subscribe skips the grid check."""
    app = create_app(push_config, auto_start_scheduler=False)
    with TestClient(app) as c:
        yield c


def _sub_body(endpoint: str = ENDPOINT_A, **over) -> dict:
    body: dict = {
        "subscription": {
            "endpoint": endpoint,
            "keys": {"p256dh": P256DH, "auth": AUTH},
            "expirationTime": None,
        },
        "lat": HOME_LAT,
        "lon": HOME_LON,
        "lead_min": 30,
        "quiet_hours": {"enabled": False, "start": "22:00", "end": "07:00"},
        "tz": "Europe/Copenhagen",
        "lang": "da",
    }
    body.update(over)
    return body


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

class TestThresholdTable:
    def test_a_fitted_lead_reads_from_the_table(self, tmp_path: Path) -> None:
        table = ThresholdTable(_write_table(tmp_path / "t.json"))
        table.load()
        assert table.effective(30) == (45, "table")
        assert table.effective(20) == (65, "table")
        assert table.fitted_at_utc == "2026-09-05T02:11:07+00:00"
        assert table.leads == [20, 30]

    @pytest.mark.parametrize("lead", [45, 60, 0, "soon", None])
    def test_everything_the_table_cannot_speak_for_falls_back(
        self, tmp_path: Path, lead,
    ) -> None:
        """45 is fitted-but-insufficient, 60 is absent, the rest are junk."""
        table = ThresholdTable(_write_table(tmp_path / "t.json"))
        table.load()
        assert table.effective(lead) == (40, "fallback")

    def test_a_missing_file_is_the_fallback_and_one_log_line(
        self, tmp_path: Path,
    ) -> None:
        table = ThresholdTable(tmp_path / "nothing.json")
        with structlog.testing.capture_logs() as logs:
            assert table.load() is None
        assert [e["event"] for e in logs] == ["push_thresholds_missing"]
        assert table.effective(30) == (40, "fallback")
        assert table.fitted_at_utc is None
        assert table.fallback_threshold_pct == 40

    def test_a_broken_file_is_the_fallback_not_half_a_rule(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "t.json"
        path.write_text("{not json")
        table = ThresholdTable(path)
        table.load()
        assert table.effective(30) == (40, "fallback")

        path.write_text(json.dumps(_table_doc(schema_version=99)))
        table.load()
        assert table.effective(30) == (40, "fallback")

    def test_the_fallback_comes_from_the_document_when_it_has_one(
        self, tmp_path: Path,
    ) -> None:
        doc = _table_doc(fallback_threshold_pct=55)
        table = ThresholdTable(_write_table(tmp_path / "t.json", doc))
        table.load()
        assert table.effective(45) == (55, "fallback")

    def test_a_replaced_file_is_re_read_at_the_next_reload(
        self, tmp_path: Path,
    ) -> None:
        path = _write_table(tmp_path / "t.json")
        table = ThresholdTable(path)
        assert table.maybe_reload() is True          # the first load
        assert table.effective(30) == (45, "table")
        # Unchanged on disk: no parse, no swap.
        assert table.maybe_reload() is False

        _write_table(path, _table_doc({"30": 70}))
        assert table.maybe_reload() is True
        assert table.effective(30) == (70, "table")

    def test_note_changed_forces_a_re_read(self, tmp_path: Path) -> None:
        """A same-size rewrite inside one filesystem timestamp tick.

        The sync task and the nightly fit both call ``note_changed`` after
        writing precisely so a stamp that did not appear to move cannot
        strand the process on the old rule.
        """
        path = _write_table(tmp_path / "t.json")
        table = ThresholdTable(path)
        table.maybe_reload()
        stat = path.stat()
        _write_table(path, _table_doc({"30": 75}))
        import os
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        table.note_changed()
        assert table.maybe_reload() is True
        assert table.effective(30) == (75, "table")

    def test_a_table_with_no_path_answers_the_fallback(self) -> None:
        table = ThresholdTable(None)
        table.maybe_reload()
        assert table.effective(30) == (40, "fallback")

    def test_the_snapshot_is_what_the_options_route_serves(
        self, tmp_path: Path,
    ) -> None:
        table = ThresholdTable(_write_table(tmp_path / "t.json"))
        table.load()
        assert table.snapshot([20, 30, 45]) == {
            "20": {"threshold_pct": 65, "source": "table"},
            "30": {"threshold_pct": 45, "source": "table"},
            "45": {"threshold_pct": 40, "source": "fallback"},
        }


def test_the_default_path_is_beside_the_other_synced_files(
    minimal_config: Config,
) -> None:
    assert resolved_thresholds_path(minimal_config) == (
        minimal_config.storage.data_dir / THRESHOLDS_FILENAME
    )
    minimal_config.push.thresholds_path = Path("/somewhere/else.json")
    assert resolved_thresholds_path(minimal_config) == Path("/somewhere/else.json")


# ---------------------------------------------------------------------------
# GET /api/push/options
# ---------------------------------------------------------------------------

class TestOptionsRoute:
    def test_it_is_anonymous_cacheable_and_resolved(
        self, push_config: Config, client: TestClient,
    ) -> None:
        _write_table(resolved_thresholds_path(push_config))
        r = client.get("/api/push/options")     # no bearer, public mode
        assert r.status_code == 200
        assert r.headers["cache-control"] == "public, max-age=300"
        assert r.json() == {
            "lead_options": [20, 30, 45, 60],
            "fallback_threshold_pct": 40,
            "fitted_at_utc": "2026-09-05T02:11:07+00:00",
            "thresholds": {
                "20": {"threshold_pct": 65, "source": "table"},
                "30": {"threshold_pct": 45, "source": "table"},
                # Fitted but insufficient, and absent from the table:
                # both are the fallback, and both say so.
                "45": {"threshold_pct": 40, "source": "fallback"},
                "60": {"threshold_pct": 40, "source": "fallback"},
            },
        }

    def test_without_a_table_every_horizon_is_the_fallback(
        self, client: TestClient,
    ) -> None:
        body = client.get("/api/push/options").json()
        assert body["fitted_at_utc"] is None
        assert all(
            entry == {"threshold_pct": 40, "source": "fallback"}
            for entry in body["thresholds"].values()
        )

    def test_a_refitted_table_is_served_without_a_restart(
        self, push_config: Config, client: TestClient,
    ) -> None:
        path = resolved_thresholds_path(push_config)
        _write_table(path)
        assert client.get("/api/push/options").json()["thresholds"]["30"][
            "threshold_pct"
        ] == 45
        _write_table(path, _table_doc({"30": 80}))
        assert client.get("/api/push/options").json()["thresholds"]["30"][
            "threshold_pct"
        ] == 80

    def test_it_is_503_when_push_is_off(self, minimal_config: Config) -> None:
        """The route exists, the feature does not — never a 404."""
        with TestClient(
            create_app(minimal_config, auto_start_scheduler=False),
        ) as c:
            assert c.get("/api/push/options").status_code == 503


# ---------------------------------------------------------------------------
# GET /calibration/push_thresholds.json (private)
# ---------------------------------------------------------------------------

class TestPrivateFileRoute:
    def test_the_private_instance_serves_the_file_it_fitted(
        self, minimal_config: Config,
    ) -> None:
        minimal_config.server.api_key = API_KEY
        with TestClient(
            create_app(minimal_config, auto_start_scheduler=False),
        ) as c:
            auth = {"Authorization": f"Bearer {API_KEY}"}
            assert c.get(
                "/calibration/push_thresholds.json", headers=auth,
            ).status_code == 503
            doc = _table_doc()
            _write_table(resolved_thresholds_path(minimal_config), doc)
            r = c.get("/calibration/push_thresholds.json", headers=auth)
            assert r.status_code == 200
            assert r.json() == doc
            assert r.headers["cache-control"] == "public, max-age=300"

    def test_public_mode_hides_it_entirely(
        self, push_config: Config, client: TestClient,
    ) -> None:
        """The public instance PULLS this file; it does not republish it."""
        _write_table(resolved_thresholds_path(push_config))
        anonymous = client.get("/calibration/push_thresholds.json")
        assert anonymous.status_code == 404
        assert anonymous.json() == {"detail": "Not Found"}
        assert client.get(
            "/calibration/push_thresholds.json",
            headers={"Authorization": f"Bearer {API_KEY}"},
        ).status_code == 200


# ---------------------------------------------------------------------------
# POST /api/push/subscribe
# ---------------------------------------------------------------------------

class TestSubscribe:
    def test_a_horizon_alone_stores_null_and_echoes_the_table(
        self, push_config: Config, client: TestClient,
    ) -> None:
        _write_table(resolved_thresholds_path(push_config))
        r = client.post("/api/push/subscribe", json=_sub_body())
        assert r.status_code == 200, r.text
        assert r.json() == {
            "ok": True, "created": True,
            "effective_threshold_pct": 45,
            "threshold_source": "table",
            "fitted_at_utc": "2026-09-05T02:11:07+00:00",
        }
        store = PushStore(resolved_db_path(push_config))
        assert store.get(ENDPOINT_A).threshold_pct is None
        store.close()

    def test_an_unfitted_horizon_says_fallback(
        self, push_config: Config, client: TestClient,
    ) -> None:
        _write_table(resolved_thresholds_path(push_config))
        body = client.post(
            "/api/push/subscribe", json=_sub_body(lead_min=60),
        ).json()
        assert body["effective_threshold_pct"] == 40
        assert body["threshold_source"] == "fallback"

    def test_no_table_at_all_is_the_fallback_too(
        self, client: TestClient,
    ) -> None:
        body = client.post("/api/push/subscribe", json=_sub_body()).json()
        assert body["effective_threshold_pct"] == 40
        assert body["threshold_source"] == "fallback"
        assert body["fitted_at_utc"] is None

    def test_an_explicit_threshold_is_stored_as_an_override(
        self, push_config: Config, client: TestClient,
    ) -> None:
        _write_table(resolved_thresholds_path(push_config))
        body = client.post(
            "/api/push/subscribe", json=_sub_body(threshold_pct=80),
        ).json()
        assert body["effective_threshold_pct"] == 80
        assert body["threshold_source"] == "override"
        store = PushStore(resolved_db_path(push_config))
        assert store.get(ENDPOINT_A).threshold_pct == 80
        store.close()

    def test_re_subscribing_without_one_clears_the_override(
        self, push_config: Config, client: TestClient,
    ) -> None:
        client.post("/api/push/subscribe", json=_sub_body(threshold_pct=80))
        client.post("/api/push/subscribe", json=_sub_body())
        store = PushStore(resolved_db_path(push_config))
        assert store.get(ENDPOINT_A).threshold_pct is None
        store.close()

    @pytest.mark.parametrize("body,expected", [
        (_sub_body(lead_min=10), 400),            # a national lead, not offered
        (_sub_body(lead_min=25), 400),            # not a lead at all
        (_sub_body(threshold_pct=50), 400),       # an override out of range
        (_sub_body(threshold_pct=0), 422),        # below the model's bound
    ])
    def test_what_is_refused_writes_nothing(
        self, push_config: Config, client: TestClient, body: dict, expected: int,
    ) -> None:
        assert client.post(
            "/api/push/subscribe", json=body,
        ).status_code == expected
        store = PushStore(resolved_db_path(push_config))
        assert store.count() == 0
        store.close()

    def test_the_offered_leads_are_the_configured_ones(
        self, client: TestClient,
    ) -> None:
        assert client.get("/api/push/options").json()["lead_options"] == [
            20, 30, 45, 60,
        ]
        assert client.get("/api/push/config").json()["lead_options_min"] == [
            20, 30, 45, 60,
        ]


# ---------------------------------------------------------------------------
# The store migration
# ---------------------------------------------------------------------------

_OLD_SCHEMA = """
CREATE TABLE subscriptions (
    endpoint            TEXT PRIMARY KEY,
    p256dh              TEXT NOT NULL,
    auth                TEXT NOT NULL,
    lat                 REAL NOT NULL,
    lon                 REAL NOT NULL,
    threshold_pct       INTEGER NOT NULL,
    lead_min            INTEGER NOT NULL,
    quiet_enabled       INTEGER NOT NULL,
    quiet_start         TEXT NOT NULL,
    quiet_end           TEXT NOT NULL,
    tz                  TEXT NOT NULL,
    lang                TEXT NOT NULL,
    created_utc         TEXT NOT NULL,
    last_notified_utc   TEXT,
    armed               INTEGER NOT NULL,
    streak              INTEGER NOT NULL,
    below_since_utc     TEXT,
    last_eval_radar_ts  TEXT
)
"""


def _pre_phase_g_db(path: Path) -> None:
    """A subscription database as Phase D left it: NOT NULL, one row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(_OLD_SCHEMA)
    conn.execute(
        "INSERT INTO subscriptions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            ENDPOINT_A, P256DH, AUTH, HOME_LAT, HOME_LON, 60, 30,
            0, "22:00", "07:00", "Europe/Copenhagen", "da",
            "2026-08-01T00:00:00+00:00", "2026-08-02T00:00:00+00:00",
            0, 2, "2026-08-02T00:00:00+00:00", "2026-08-02T00:10:00+00:00",
        ),
    )
    conn.commit()
    conn.close()


class TestMigration:
    def test_existing_thresholds_become_null_and_the_row_survives(
        self, tmp_path: Path,
    ) -> None:
        db = tmp_path / "subscriptions.sqlite"
        _pre_phase_g_db(db)
        store = PushStore(db)
        row = store.get(ENDPOINT_A)
        assert row is not None
        # The choice is gone…
        assert row.threshold_pct is None
        # …and everything else — the horizon, the history, the state
        # machine mid-flight — is exactly where it was.
        assert row.lead_min == 30
        assert row.created_utc == datetime(2026, 8, 1, tzinfo=timezone.utc)
        assert row.last_notified_utc == datetime(2026, 8, 2, tzinfo=timezone.utc)
        assert row.armed is False and row.streak == 2
        assert row.below_since_utc == datetime(2026, 8, 2, tzinfo=timezone.utc)
        store.close()

    def test_it_runs_once_and_a_null_can_then_be_written(
        self, tmp_path: Path,
    ) -> None:
        db = tmp_path / "subscriptions.sqlite"
        _pre_phase_g_db(db)
        PushStore(db).close()
        store = PushStore(db)                    # second open: nothing to do
        store.upsert(NewSubscription(
            endpoint=ENDPOINT_B, p256dh=P256DH, auth=AUTH,
            lat=HOME_LAT, lon=HOME_LON, threshold_pct=None, lead_min=20,
            quiet_enabled=False, quiet_start="22:00", quiet_end="07:00",
            tz="UTC", lang="en",
        ))
        assert store.count() == 2
        assert store.get(ENDPOINT_B).threshold_pct is None
        store.close()

    def test_a_fresh_database_needs_no_migration(self, tmp_path: Path) -> None:
        with structlog.testing.capture_logs() as logs:
            store = PushStore(tmp_path / "new.sqlite")
        assert not [e for e in logs if e["event"] == "push_store_migrated"]
        store.close()


# ---------------------------------------------------------------------------
# The fan-out reads the table
# ---------------------------------------------------------------------------

class _Sample:
    """What ``national_sample.sample_point`` returns, reduced to fields."""

    def __init__(self, p_rain: float, lead: int) -> None:
        self.p_rain = {lead: p_rain}
        self.eta_min = 25.0
        self.intensity_mm_h = 1.2
        self.observed_mm_h = 0.0
        self.forecast_mm_h = {0: 0.0}


@pytest.fixture
def service(push_config: Config, monkeypatch: pytest.MonkeyPatch) -> PushService:
    """A service whose sampler is a constant, so only the rule is under test."""
    monkeypatch.setattr(
        service_mod, "sample_point",
        lambda products, geo, lat, lon, **kw: _Sample(0.5, 30),
    )
    monkeypatch.setattr(
        service_mod.fanout, "send",
        lambda **kw: type("R", (), {"ok": True, "gone": False, "status": 201,
                                    "error": None})(),
    )
    store = PushStore(resolved_db_path(push_config))
    return PushService(
        push_config, engine=None, store=store, vapid_private_pem=b"pem",
        public_key="key",
        thresholds=ThresholdTable(resolved_thresholds_path(push_config)),
    )


def _evaluate(service: PushService, ts: datetime = RADAR_TS) -> list[dict]:
    with structlog.testing.capture_logs() as logs:
        service._evaluate_and_send(None, None, ts, ts)
    return [e for e in logs if e["event"] == "push_eval"]


class TestServiceUsesTheTable:
    def _subscribe(self, service: PushService, **over) -> None:
        kwargs = {
            "endpoint": ENDPOINT_A, "p256dh": P256DH, "auth": AUTH,
            "lat": HOME_LAT, "lon": HOME_LON,
            "threshold_pct": None, "lead_min": 30,
            "quiet_enabled": False, "quiet_start": "22:00",
            "quiet_end": "07:00", "tz": "Europe/Copenhagen", "lang": "da",
        }
        kwargs.update(over)
        service.store.upsert(NewSubscription(**kwargs))  # type: ignore[arg-type]

    def test_the_fitted_threshold_decides(
        self, push_config: Config, service: PushService,
    ) -> None:
        """p_rain 0.5: over a fitted 45 %, under the shipped 40 %… no —

        under 45 % it would NOT be over. So: the table says 45, the sample
        is 0.50, the row is over threshold; with the fallback of 40 it
        would also be over. The discriminating case is the table at 70.
        """
        _write_table(resolved_thresholds_path(push_config), _table_doc({"30": 70}))
        self._subscribe(service)
        entry = _evaluate(service)[0]
        assert entry["threshold_pct"] == 70
        assert entry["threshold_source"] == "table"
        # 0.50 < 0.70 — no streak, nothing sent.
        assert entry["action"] == "none"
        assert service.store.get(ENDPOINT_A).streak == 0

    def test_the_fallback_decides_when_the_lead_is_unfitted(
        self, push_config: Config, service: PushService,
    ) -> None:
        _write_table(resolved_thresholds_path(push_config))
        self._subscribe(service, lead_min=45)     # insufficient in the table
        entry = _evaluate(service)[0]
        assert (entry["threshold_pct"], entry["threshold_source"]) == (
            40, "fallback",
        )

    def test_an_override_beats_the_table(
        self, push_config: Config, service: PushService,
    ) -> None:
        _write_table(resolved_thresholds_path(push_config), _table_doc({"30": 70}))
        self._subscribe(service, threshold_pct=40)
        entry = _evaluate(service)[0]
        assert (entry["threshold_pct"], entry["threshold_source"]) == (
            40, "override",
        )
        # 0.50 >= 0.40, and persistence_obs defaults to 2 → a streak, no push.
        assert service.store.get(ENDPOINT_A).streak == 1

    def test_a_migrated_row_still_evaluates(
        self, push_config: Config, service: PushService,
    ) -> None:
        """The Phase D row, threshold cleared, is evaluated by the table."""
        _write_table(resolved_thresholds_path(push_config), _table_doc({"30": 45}))
        self._subscribe(service)
        service.store.update_state(
            ENDPOINT_A, armed=True, streak=1,
            below_since_utc=None, last_eval_radar_ts=None,
        )
        entry = _evaluate(service)[0]
        assert entry["threshold_pct"] == 45
        assert entry["action"] == "notify"       # streak 1 + this one = 2

    def test_a_refit_between_cycles_is_picked_up(
        self, push_config: Config, service: PushService,
    ) -> None:
        path = resolved_thresholds_path(push_config)
        _write_table(path, _table_doc({"30": 45}))
        self._subscribe(service)
        assert _evaluate(service)[0]["threshold_pct"] == 45
        _write_table(path, _table_doc({"30": 90}))
        later = _evaluate(service, RADAR_TS + timedelta(minutes=10))[0]
        assert later["threshold_pct"] == 90

    def test_the_eval_line_carries_the_rule_and_no_secrets(
        self, push_config: Config, service: PushService,
    ) -> None:
        _write_table(resolved_thresholds_path(push_config))
        self._subscribe(service)
        entry = _evaluate(service)[0]
        assert set(entry) >= {
            "sub", "radar_ts", "action", "lead_min",
            "threshold_pct", "threshold_source", "p_rain",
        }
        assert ENDPOINT_A not in json.dumps(entry)
        assert str(HOME_LAT) not in json.dumps(entry)


# ---------------------------------------------------------------------------
# The sync task
# ---------------------------------------------------------------------------

class TestSync:
    def test_the_table_lands_where_the_service_reads_it(
        self, push_config: Config,
    ) -> None:
        assert target_path(push_config, THRESHOLDS_FILE) == (
            resolved_thresholds_path(push_config)
        )
        push_config.push.thresholds_path = Path("/data/elsewhere.json")
        assert target_path(push_config, THRESHOLDS_FILE) == Path(
            "/data/elsewhere.json",
        )

    def test_a_synced_table_nudges_the_running_service(
        self, push_config: Config, tmp_path: Path,
    ) -> None:
        """The public instance pulls the fit; the fan-out must see it.

        Without the ``note_changed`` hook the file lands and takes effect
        on the next restart, which for a 5-minute service is the wrong
        kind of eventually.
        """
        import asyncio

        import httpx

        from dmi_nowcast_sidecar.sync import build_artifact_sync

        push_config.sync.enabled = True
        push_config.sync.source_url = "http://private:8081"
        push_config.sync.files = [THRESHOLDS_FILE]
        table = ThresholdTable(resolved_thresholds_path(push_config))
        table.maybe_reload()
        assert table.effective(30) == (40, "fallback")

        doc = _table_doc({"30": 55})
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json=doc),
        )
        task = build_artifact_sync(push_config, engine=None, push_thresholds=table)
        assert task is not None
        task._client = httpx.AsyncClient(transport=transport)
        result = asyncio.run(task.sync_once())
        assert [f.status for f in result.files] == ["updated"]

        assert resolved_thresholds_path(push_config).is_file()
        # Nudged, not yet swapped: the swap happens at the next fan-out.
        assert table.maybe_reload() is True
        assert table.effective(30) == (55, "table")
        asyncio.run(task._client.aclose())
