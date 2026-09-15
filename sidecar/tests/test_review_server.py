"""WP4 — the review server (``scripts/review_server.py``).

The script lives at the repo root but is tested from this suite for the
same reason ``test_replay_warnings.py`` is: the sidecar environment is the
one that has both the core package and ``pyarrow``, and the Parquet export
is half the deliverable.

What is worth testing here is not "does HTTP work" but the four places a
year of human judgement could quietly be lost:

- the schema migrates instead of erroring on a database written by an
  older shape, and says how many rows it moved;
- an edit bumps the revision, appends history, and does **not** renumber
  ``created_utc`` or ``review_seq`` — the drift check reads the second and
  a re-judgement must not move an event's place in the review order;
- a stale ``If-Match`` loses rather than overwriting the judgement that
  won (two browser tabs on one event is a routine accident);
- an unknown tag code is refused, because a vocabulary that accepts
  anything is not controlled and 300 events produce 300 unique tags.

Everything is offline and synthetic: a fixture bundle of a few JSON
documents and one PNG-shaped file, no VM, no radar archive.
"""
from __future__ import annotations

import http.client
import json
import logging
import sqlite3
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import review_server as rs  # noqa: E402  (after the sys.path edit)

from dmi_nowcast_core import review_schema  # noqa: E402

BUNDLE_ID = "review-test-20260915"

#: A one-pixel PNG header is enough: nothing in the server decodes it, and
#: a real composite render belongs to WP3's tests.
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _index_row(
    event_id: str,
    *,
    event_class: str,
    station_id: str = "06074",
    anchor_utc: str = "2026-06-12T13:40:00+00:00",
    dual_truth: str = "both_dry",
    season: str = "summer",
    region: str = "fyn",
) -> dict[str, Any]:
    """One ``events.json`` row, with the fields the server reads.

    Note ``class``, not ``event_class``: the bundle's index uses the
    keyword-shaped name and the database uses the safe one, and the server
    is the only place that maps between them.
    """
    return {
        "event_id": event_id,
        "class": event_class,
        "control": event_class == "hit",
        "station_id": station_id,
        "station_name": "Test station",
        "anchor_utc": anchor_utc,
        "dual_truth": dual_truth,
        "season": season,
        "region": region,
    }


_EVENTS = [
    _index_row("fa-06074-20260612T1340Z", event_class="false_alarm"),
    _index_row(
        "miss-06080-20260612T1500Z", event_class="miss",
        station_id="06080", anchor_utc="2026-06-12T15:00:00+00:00",
        dual_truth="gauge_wet_radar_dry",
    ),
    _index_row(
        "hit-06074-20260613T0900Z", event_class="hit",
        anchor_utc="2026-06-13T09:00:00+00:00", dual_truth="both_wet",
    ),
]


@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    """A minimal but layout-correct bundle: manifest, index, tags, a frame."""
    root = tmp_path / "bundle"
    (root / "events").mkdir(parents=True)
    (root / "frames").mkdir()
    (root / "manifest.json").write_text(
        json.dumps({
            "schema_version": review_schema.REVIEW_SCHEMA_VERSION,
            "bundle_id": BUNDLE_ID,
            "built_at_utc": "2026-09-15T08:00:00+00:00",
        }),
        encoding="utf-8",
    )
    (root / "events.json").write_text(
        json.dumps({
            "schema_version": review_schema.REVIEW_SCHEMA_VERSION,
            "bundle_id": BUNDLE_ID,
            "events": _EVENTS,
        }),
        encoding="utf-8",
    )
    (root / "tags.json").write_text(
        json.dumps(review_schema.tags_document()), encoding="utf-8",
    )
    for row in _EVENTS:
        (root / "events" / f"{row['event_id']}.json").write_text(
            json.dumps({"index": row}), encoding="utf-8",
        )
    (root / "frames" / "20260612T1340Z.overlay.png").write_bytes(_PNG_BYTES)
    return root


@pytest.fixture
def store(tmp_path: Path):
    handle = rs.ReviewStore(tmp_path / "annotations.sqlite")
    try:
        yield handle
    finally:
        handle.close()


@pytest.fixture
def server(bundle_dir: Path, tmp_path: Path):
    """A real ``ThreadingHTTPServer`` on a free port, in a background thread."""
    srv = rs.build_server(bundle_dir, tmp_path / "annotations.sqlite", port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        srv.store.close()
        thread.join(timeout=5)


def _identity(event_id: str) -> rs.EventIdentity:
    row = next(r for r in _EVENTS if r["event_id"] == event_id)
    return rs._identity_of(row)


def _values(**overrides: Any) -> dict[str, Any]:
    body = {
        "verdict": "metric_artefact",
        "tags": ["fa_drizzle_below_gauge_floor"],
        "confidence": 2,
        "needs_second_look": False,
        "note": "",
        "cursor_utc": None,
    }
    body.update(overrides)
    return rs.validate_annotation(body)


def _request(
    port: int,
    method: str,
    path: str,
    body: Any = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, Any, dict[str, str]]:
    """One request, decoded as JSON. Errors come back, they do not raise."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, method=method,
    )
    request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None, dict(
                response.headers,
            )
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        return exc.code, json.loads(raw) if raw else None, dict(exc.headers)


def _raw_get(port: int, path: str) -> tuple[int, bytes, dict[str, str]]:
    """A GET whose path is sent verbatim — ``urllib`` would tidy ``..`` away."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.putrequest("GET", path, skip_host=False, skip_accept_encoding=True)
        conn.endheaders()
        response = conn.getresponse()
        return response.status, response.read(), dict(response.getheaders())
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 1. Schema and migrations
# ---------------------------------------------------------------------------


def _columns(db_path: Path, table: str = "annotations") -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    finally:
        conn.close()


def _primary_key(db_path: Path) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = [r for r in conn.execute("PRAGMA table_info(annotations)") if r[5]]
    finally:
        conn.close()
    return [r[1] for r in sorted(rows, key=lambda r: r[5])]


def test_schema_creates_from_empty_and_reopening_is_a_no_op(tmp_path: Path) -> None:
    db = tmp_path / "annotations.sqlite"
    first = rs.ReviewStore(db)
    first.close()

    assert _columns(db) == list(rs._COLUMNS)
    assert _primary_key(db) == ["bundle_id", "event_id"]

    conn = sqlite3.connect(db)
    try:
        names = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')",
            )
        }
    finally:
        conn.close()
    assert {"annotations", "annotation_history", "schema_meta"} <= names
    assert {"annotations_class", "annotations_verdict", "history_event"} <= names

    # Re-opening an already-current database must not migrate anything.
    second = rs.ReviewStore(db)
    try:
        assert second.meta()["db_version"] == str(rs.DB_VERSION)
        assert _columns(db) == list(rs._COLUMNS)
    finally:
        second.close()


def test_older_schema_migrates_and_logs_a_row_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The first cut: one bundle assumed, ``event_id`` the whole primary key."""
    db = tmp_path / "annotations.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE annotations (
            event_id    TEXT PRIMARY KEY,
            verdict     TEXT,
            tags        TEXT NOT NULL DEFAULT '[]',
            note        TEXT NOT NULL DEFAULT '',
            created_utc TEXT NOT NULL,
            updated_utc TEXT NOT NULL,
            revision    INTEGER NOT NULL DEFAULT 1
        );
        """,
    )
    conn.executemany(
        "INSERT INTO annotations "
        "(event_id, verdict, tags, note, created_utc, updated_utc, revision) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("hit-06074-20260613T0900Z", "real_failure", '["fa_cell_died"]',
             "second", "2026-09-02T10:00:00+00:00", "2026-09-02T10:00:00+00:00", 3),
            ("fa-06074-20260612T1340Z", "metric_artefact", '["fa_gauge_missed_it"]',
             "first", "2026-09-01T09:00:00+00:00", "2026-09-01T09:00:00+00:00", 1),
        ],
    )
    conn.commit()
    conn.close()

    with caplog.at_level(logging.INFO, logger="review_server"):
        store = rs.ReviewStore(db, legacy_bundle_id=BUNDLE_ID)
    try:
        assert _columns(db) == list(rs._COLUMNS)
        assert _primary_key(db) == ["bundle_id", "event_id"]

        rows = store.list(BUNDLE_ID)
        assert [r["event_id"] for r in rows] == [
            "fa-06074-20260612T1340Z", "hit-06074-20260613T0900Z",
        ]
        # Legacy rows keep their verdict, tags and revision, gain the bundle
        # they could only have belonged to, and are numbered by created_utc.
        assert rows[0]["verdict"] == "metric_artefact"
        assert rows[0]["tags"] == ["fa_gauge_missed_it"]
        assert rows[0]["revision"] == 1
        assert [r["review_seq"] for r in rows] == [1, 2]
        assert all(r["bundle_id"] == BUNDLE_ID for r in rows)
        # Columns the old shape never had come back as their documented
        # "unset", not as a value a later aggregation would believe.
        assert rows[0]["confidence"] is None
        assert rows[0]["needs_second_look"] is False
    finally:
        store.close()

    text = caplog.text
    assert "review_store migrated" in text
    assert "2 row(s) copied" in text           # the structural rebuild
    assert "2 existing row(s) carried forward" in text   # the additive step
    assert "review_seq backfilled by created_utc for 2 row(s)" in text

    # Idempotent: a second open of the migrated file migrates nothing.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="review_server"):
        again = rs.ReviewStore(db, legacy_bundle_id=BUNDLE_ID)
    try:
        assert "migrated" not in caplog.text
        assert len(again.list(BUNDLE_ID)) == 2
        # A new annotation continues the numbering rather than colliding
        # with a backfilled row.
        row, created = again.upsert(
            BUNDLE_ID, _identity("miss-06080-20260612T1500Z"), _values(
                verdict="real_failure", tags=["miss_below_threshold"],
            ),
        )
        assert created and row["review_seq"] == 3
    finally:
        again.close()


# ---------------------------------------------------------------------------
# 2. Upsert semantics
# ---------------------------------------------------------------------------


def test_upsert_bumps_revision_and_preserves_created_and_seq(
    store: rs.ReviewStore,
) -> None:
    identity = _identity("fa-06074-20260612T1340Z")
    first, created = store.upsert(BUNDLE_ID, identity, _values())
    assert created
    assert first["revision"] == 1
    assert first["review_seq"] == 1

    # Another event in between, so a renumbering bug would be visible.
    store.upsert(
        BUNDLE_ID, _identity("miss-06080-20260612T1500Z"),
        _values(verdict="real_failure", tags=["miss_disarmed_rearm"]),
    )

    second, created = store.upsert(
        BUNDLE_ID, identity,
        _values(verdict="real_failure", tags=["fa_cell_died"], confidence=3),
    )
    assert not created
    assert second["revision"] == 2
    assert second["created_utc"] == first["created_utc"]
    assert second["updated_utc"] >= first["updated_utc"]
    assert second["review_seq"] == first["review_seq"] == 1
    assert second["verdict"] == "real_failure"
    assert second["tags"] == ["fa_cell_died"]

    history = store.history(BUNDLE_ID, identity.event_id)
    assert [h["revision"] for h in history] == [1, 2]
    assert history[0]["payload"]["verdict"] == "metric_artefact"
    assert history[0]["payload"]["tags"] == ["fa_drizzle_below_gauge_floor"]
    assert history[1]["payload"]["verdict"] == "real_failure"

    # The denormalised identity is written from the bundle index, so an
    # export stands alone.
    assert second["station_id"] == "06074"
    assert second["event_class"] == "false_alarm"
    assert second["anchor_utc"] == "2026-06-12T13:40:00+00:00"
    assert second["season"] == "summer"
    assert second["region"] == "fyn"


def test_review_seq_never_reuses_a_deleted_number(store: rs.ReviewStore) -> None:
    identity = _identity("fa-06074-20260612T1340Z")
    first, _ = store.upsert(BUNDLE_ID, identity, _values())
    assert first["review_seq"] == 1
    store.delete(BUNDLE_ID, identity.event_id)
    again, created = store.upsert(BUNDLE_ID, identity, _values())
    assert created
    assert again["review_seq"] == 2, "a re-judgement is a later moment in the review"
    # The deletion is kept: withdrawing a judgement must not look like
    # never having made one.
    history = store.history(BUNDLE_ID, identity.event_id)
    assert [h["payload"].get("deleted", False) for h in history] == [
        False, True, False,
    ]


def test_if_match_on_a_stale_revision_conflicts_and_writes_nothing(
    store: rs.ReviewStore,
) -> None:
    identity = _identity("fa-06074-20260612T1340Z")
    store.upsert(BUNDLE_ID, identity, _values())
    store.upsert(BUNDLE_ID, identity, _values(verdict="unclear"))  # revision 2

    with pytest.raises(rs.RevisionConflict) as excinfo:
        store.upsert(
            BUNDLE_ID, identity, _values(verdict="real_failure"), if_match="1",
        )
    assert excinfo.value.expected == "1"
    assert excinfo.value.actual == 2

    stored = store.get(BUNDLE_ID, identity.event_id)
    assert stored["verdict"] == "unclear"
    assert stored["revision"] == 2
    assert len(store.history(BUNDLE_ID, identity.event_id)) == 2

    # A matching revision goes through, and "0" means "must not exist yet".
    ok, _ = store.upsert(
        BUNDLE_ID, identity, _values(verdict="real_failure"), if_match="2",
    )
    assert ok["revision"] == 3
    with pytest.raises(rs.RevisionConflict):
        store.upsert(BUNDLE_ID, identity, _values(), if_match="0")


# ---------------------------------------------------------------------------
# 3. HTTP round trip against a real server
# ---------------------------------------------------------------------------


def test_http_round_trip_and_persistence(server, tmp_path: Path) -> None:
    port = server.server_address[1]

    status, health, _ = _request(port, "GET", "/review-api/health")
    assert status == 200
    assert health["ok"] is True
    assert health["bundle_id"] == BUNDLE_ID
    assert health["events"] == len(_EVENTS)
    assert health["annotated"] == 0
    assert health["schema_version"] == review_schema.REVIEW_SCHEMA_VERSION
    assert health["vocab_version"] == review_schema.REVIEW_VOCAB_VERSION

    status, vocab, _ = _request(port, "GET", "/review-api/vocabulary")
    assert status == 200
    assert vocab == review_schema.tags_document()

    event_id = "fa-06074-20260612T1340Z"
    body = {
        "verdict": "metric_artefact",
        "tags": ["fa_drizzle_below_gauge_floor", "fa_gauge_missed_it"],
        "confidence": 2,
        "needs_second_look": False,
        "note": "Radar 0.7 mm/h for two slots; gauge 0.1 mm, neighbour 1.2 mm.",
        "cursor_utc": "2026-06-12T13:50:00+00:00",
        "vocab_version": 1,
    }
    status, row, headers = _request(
        port, "PUT", f"/review-api/annotations/{event_id}", body,
    )
    assert status == 200
    assert row["revision"] == 1
    assert row["review_seq"] == 1
    assert row["tags"] == body["tags"]
    assert row["cursor_utc"] == "2026-06-12T13:50:00+00:00"
    assert row["station_id"] == "06074"
    assert headers["ETag"] == '"1"'

    status, fetched, _ = _request(
        port, "GET", f"/review-api/annotations/{event_id}",
    )
    assert status == 200
    assert fetched == row

    # A stale conditional write loses at the HTTP boundary too.
    status, conflict, _ = _request(
        port, "PUT", f"/review-api/annotations/{event_id}",
        {**body, "verdict": "unclear"}, {"If-Match": "99"},
    )
    assert status == 409
    assert conflict["error"] == "revision_conflict"
    assert conflict["stored"] == 1

    status, updated, _ = _request(
        port, "PUT", f"/review-api/annotations/{event_id}",
        {**body, "verdict": "real_failure"}, {"If-Match": '"1"'},
    )
    assert status == 200
    assert updated["revision"] == 2

    status, listing, _ = _request(port, "GET", "/review-api/annotations")
    assert status == 200
    assert listing["bundle_id"] == BUNDLE_ID
    assert [r["event_id"] for r in listing["annotations"]] == [event_id]

    # An event the bundle does not describe has no identity to denormalise.
    status, unknown, _ = _request(
        port, "PUT", "/review-api/annotations/fa-99999-20200101T0000Z", body,
    )
    assert status == 404
    assert unknown["error"] == "unknown_event"

    # Bundle documents and frames.
    status, events_doc, headers = _request(port, "GET", "/review-data/events.json")
    assert status == 200
    assert len(events_doc["events"]) == len(_EVENTS)
    assert headers["Cache-Control"] == "no-store"

    status, png, headers = _raw_get(
        port, "/review-data/frames/20260612T1340Z.overlay.png",
    )
    assert status == 200
    assert png == _PNG_BYTES
    assert headers["Content-Type"] == "image/png"
    assert "immutable" in headers["Cache-Control"]

    status, deleted, _ = _request(
        port, "DELETE", f"/review-api/annotations/{event_id}",
    )
    assert status == 200
    assert deleted["deleted"] is True
    status, _, _ = _request(port, "GET", f"/review-api/annotations/{event_id}")
    assert status == 404

    # A second judgement, so there is something to find after a restart.
    other = "miss-06080-20260612T1500Z"
    status, _, _ = _request(
        port, "PUT", f"/review-api/annotations/{other}",
        {"verdict": "real_failure", "tags": ["miss_disarmed_rearm"],
         "confidence": 3, "needs_second_look": True, "note": "re-arm"},
    )
    assert status == 200

    status, export_result, _ = _request(
        port, "POST", "/review-api/export?format=md", {},
    )
    assert status == 200
    assert export_result["rows"] == 1
    exported = Path(export_result["path"])
    assert exported.is_file()
    assert exported.parent == server.bundle.root / "exports"

    # Re-open the database as a separate store: the judgement survives the
    # process, and so does the history of the deleted one.
    reopened = rs.ReviewStore(tmp_path / "annotations.sqlite")
    try:
        rows = reopened.list(BUNDLE_ID)
        assert [r["event_id"] for r in rows] == [other]
        assert rows[0]["needs_second_look"] is True
        assert rows[0]["review_seq"] == 2
        assert [h["revision"] for h in reopened.history(BUNDLE_ID, event_id)] == [
            1, 2, 3,
        ]
    finally:
        reopened.close()


# ---------------------------------------------------------------------------
# 4. Validation: the vocabulary is controlled here or nowhere
# ---------------------------------------------------------------------------


def test_unknown_tag_code_is_422_naming_the_offender(server) -> None:
    port = server.server_address[1]
    status, payload, _ = _request(
        port, "PUT", "/review-api/annotations/fa-06074-20260612T1340Z",
        {"verdict": "real_failure", "tags": ["fa_cell_died", "fa_it_was_wrong"]},
    )
    assert status == 422
    assert payload["error"] == "validation_failed"
    problem = next(p for p in payload["problems"] if p["field"] == "tags")
    assert problem["offending"] == ["fa_it_was_wrong"]
    assert "fa_it_was_wrong" in problem["message"]
    # Nothing was stored: a rejected body must not half-save.
    status, _, _ = _request(
        port, "GET", "/review-api/annotations/fa-06074-20260612T1340Z",
    )
    assert status == 404


def test_unknown_verdict_is_422_naming_the_offender(server) -> None:
    port = server.server_address[1]
    status, payload, _ = _request(
        port, "PUT", "/review-api/annotations/fa-06074-20260612T1340Z",
        {"verdict": "probably_fine", "tags": []},
    )
    assert status == 422
    problem = next(p for p in payload["problems"] if p["field"] == "verdict")
    assert problem["offending"] == ["probably_fine"]
    assert "probably_fine" in problem["message"]


def test_validation_rejects_a_misspelled_field_and_a_naive_timestamp() -> None:
    with pytest.raises(rs.ValidationError) as excinfo:
        rs.validate_annotation({"verdict": "unclear", "tag": ["fa_cell_died"]})
    assert excinfo.value.problems[0]["offending"] == ["tag"]

    with pytest.raises(rs.ValidationError) as excinfo:
        rs.validate_annotation({"cursor_utc": "2026-06-12T13:50:00"})
    assert "offset" in excinfo.value.problems[0]["message"]

    # A verdict of null is legal — it is how "seen, not yet judged" is
    # stored — and duplicate tags are deduped rather than refused.
    values = rs.validate_annotation(
        {"verdict": None, "tags": ["fa_cell_died", "fa_cell_died"]},
    )
    assert values["verdict"] is None
    assert values["tags"] == ["fa_cell_died"]
    assert values["reviewer"] == "local"
    assert values["vocab_version"] == review_schema.REVIEW_VOCAB_VERSION


# ---------------------------------------------------------------------------
# 5. Path traversal
# ---------------------------------------------------------------------------


def test_bundle_paths_cannot_escape_the_bundle(server, tmp_path: Path) -> None:
    port = server.server_address[1]
    secret = tmp_path / "secret.json"
    secret.write_text('{"unreleased": true}', encoding="utf-8")
    (server.bundle.root / "escape.json").symlink_to(secret)

    for path in (
        "/review-data/../secret.json",
        "/review-data/%2e%2e/secret.json",
        "/review-data/frames/../../secret.json",
        "/review-data//etc/passwd",
        "/review-data/escape.json",          # symlink out of the tree
    ):
        status, body, _ = _raw_get(port, path)
        assert status == 403, f"{path} should be refused, got {status}"
        assert b"unreleased" not in body
        assert b"root:" not in body

    # A file inside the bundle still works, and an unexpected file type is
    # refused rather than guessed at.
    status, _, _ = _raw_get(port, "/review-data/manifest.json")
    assert status == 200
    (server.bundle.root / "notes.txt").write_text("hello", encoding="utf-8")
    status, _, _ = _raw_get(port, "/review-data/notes.txt")
    assert status == 403


def test_resolve_refuses_an_absolute_path(bundle_dir: Path) -> None:
    bundle = rs.Bundle(bundle_dir)
    assert bundle.resolve("/etc/passwd") is None
    assert bundle.resolve("../../etc/passwd") is None
    assert bundle.resolve("manifest.json") == bundle_dir.resolve() / "manifest.json"


# ---------------------------------------------------------------------------
# 6. Export
# ---------------------------------------------------------------------------


def _annotation(
    event_id: str,
    event_class: str,
    verdict: str | None,
    tags: list[str],
    review_seq: int,
    *,
    note: str = "",
    station_id: str = "06074",
) -> dict[str, Any]:
    """A stored row's shape, built by hand so the fixture is hand-workable."""
    return {
        "bundle_id": BUNDLE_ID,
        "event_id": event_id,
        "station_id": station_id,
        "anchor_utc": "2026-06-12T13:40:00+00:00",
        "event_class": event_class,
        "dual_truth": "both_dry",
        "season": "summer",
        "region": "fyn",
        "verdict": verdict,
        "tags": tags,
        "vocab_version": 1,
        "confidence": 2,
        "needs_second_look": False,
        "note": note,
        "cursor_utc": None,
        "review_seq": review_seq,
        "reviewer": "local",
        "created_utc": "2026-09-15T08:00:00+00:00",
        "updated_utc": "2026-09-15T08:05:00+00:00",
        "revision": 1,
    }


#: Hand-worked: 4 false alarms and 4 hits, all judged.
#:   fa_cell_died          2/4 false alarms = 50 %, 1/4 hits = 25 % → 2.0x
#:   fa_virga_or_aloft     1/4 false alarms = 25 %, 0/4 hits =  0 % → infinite
#:   fa_threshold_marginal 2/8 judged = 25 %, never ranked (non-mechanism)
_CROSSTAB_ROWS = [
    _annotation("fa-1", "false_alarm", "real_failure", ["fa_cell_died"], 1),
    _annotation(
        "fa-2", "false_alarm", "real_failure",
        ["fa_cell_died", "fa_threshold_marginal"], 2, note="marginal, 2 pts over",
    ),
    _annotation(
        "fa-3", "false_alarm", "metric_artefact", ["fa_virga_or_aloft"], 3,
        note="column-max over a dry gauge",
    ),
    _annotation("fa-4", "false_alarm", "unclear", ["fa_threshold_marginal"], 4),
    _annotation("hit-1", "hit", "real_failure", ["fa_cell_died"], 5),
    _annotation("hit-2", "hit", "real_failure", [], 6),
    _annotation("hit-3", "hit", "metric_artefact", [], 7),
    _annotation("hit-4", "hit", "metric_artefact", [], 8),
]


def test_markdown_crosstab_is_pinned_against_a_hand_worked_fixture() -> None:
    rows = rs.export_rows(_CROSSTAB_ROWS)
    identities = {
        r["event_id"]: rs.EventIdentity(
            event_id=r["event_id"], station_id=r["station_id"],
            anchor_utc=r["anchor_utc"], event_class=r["event_class"],
            dual_truth=r["dual_truth"], season=r["season"], region=r["region"],
        )
        for r in _CROSSTAB_ROWS
    }
    identities["fa-unjudged"] = rs.EventIdentity(
        event_id="fa-unjudged", station_id="06080",
        anchor_utc="2026-06-14T10:00:00+00:00", event_class="false_alarm",
        dual_truth="both_dry", season="summer", region="fyn",
    )
    text = rs.render_markdown(
        rows, bundle_id=BUNDLE_ID, identities=identities,
        generated_at_utc="2026-09-15T10:15:00+00:00",
    )
    lines = text.splitlines()

    # Verdicts by class: 4 false alarms (2 real_failure, 1 artefact, 1
    # unclear) against 9 events in the bundle, 4 hits (2/2/0).
    assert "| class | events | judged | real_failure | metric_artefact | unclear |" in lines
    assert "| false_alarm | 5 | 4 | 2 | 1 | 1 |" in lines
    assert "| hit | 4 | 4 | 2 | 2 | 0 |" in lines

    # The contrast column is the point of the report.
    assert "### false_alarm — 4 judged" in lines
    assert "| tag | n | share | control | lift |" in lines
    assert "| `fa_cell_died` | 2 | 50 % | 25 % | 2.0× |" in lines
    assert "| `fa_virga_or_aloft` | 1 | 25 % | 0 % | ∞ |" in lines

    # Non-mechanism tags are reported, but never inside a mechanism table:
    # "I could not decide" must not out-rank an explanation.
    mechanism_section = text.split("## Mechanisms by class")[1].split("## Non-")[0]
    assert "fa_threshold_marginal" not in mechanism_section
    assert "| `fa_threshold_marginal` | 2 | 25 % |" in lines
    # The hit control group is a base rate, not a class to explain.
    assert "### hit —" not in text

    assert "## Unreviewed (1)" in lines
    assert any(line.startswith("| `fa-unjudged` | false_alarm |") for line in lines)

    # Notes are filed under the first mechanism tag, never a "could not
    # decide" one.
    assert "### fa_cell_died" in lines
    assert any(
        line.startswith("- `fa-2` (false_alarm, real_failure) — marginal")
        for line in lines
    )
    assert "### fa_virga_or_aloft" in lines


def test_parquet_export_columns_are_pinned(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    rows = rs.export_rows(_CROSSTAB_ROWS)
    path = tmp_path / "annotations.parquet"
    written = rs.write_parquet(rows, path)
    assert written == len(_CROSSTAB_ROWS)

    table = pq.read_table(path)
    codes = sorted(review_schema.all_tag_codes())
    assert table.column_names == [
        *rs.EXPORT_SCALAR_COLUMNS, "tags", "mechanism_tags",
        *[f"tag_{code}" for code in codes],
    ]
    # Every vocabulary code gets a boolean column, so a DuckDB cross-tab is
    # one GROUP BY rather than a list unnest.
    import pyarrow as pa

    assert all(
        pa.types.is_boolean(table.schema.field(f"tag_{code}").type)
        for code in codes
    )
    tags_type = table.schema.field("tags").type
    assert pa.types.is_list(tags_type)
    assert pa.types.is_string(tags_type.value_type)

    frame = {name: table.column(name).to_pylist() for name in table.column_names}
    assert frame["event_id"] == [r["event_id"] for r in rows]
    assert frame["tags"][1] == ["fa_cell_died", "fa_threshold_marginal"]
    # mechanism_tags is how the SQL ranks mechanisms without restating the
    # vocabulary: the non-mechanism code is gone from it, not from tags.
    assert frame["mechanism_tags"][1] == ["fa_cell_died"]
    assert frame["tag_fa_cell_died"] == [True, True, False, False, True,
                                         False, False, False]
    # Eight judged rows, so decile_of(rank, 8) spreads them: no bucket
    # holds two, and the last is the tenth of the review, not the eighth.
    assert frame["review_decile"] == [1, 2, 3, 4, 6, 7, 8, 9]
    assert frame["anchor_utc"][0] == "2026-06-12T13:40:00+00:00"


def test_csv_export_is_readable(tmp_path: Path) -> None:
    import csv

    rows = rs.export_rows(_CROSSTAB_ROWS)
    path = tmp_path / "annotations.csv"
    assert rs.write_csv(rows, path) == len(rows)
    parsed = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    assert parsed[1]["tags"] == "fa_cell_died|fa_threshold_marginal"
    assert parsed[0]["event_class"] == "false_alarm"


def test_export_writes_into_the_bundle(bundle_dir: Path, tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    bundle = rs.Bundle(bundle_dir)
    store = rs.ReviewStore(tmp_path / "annotations.sqlite")
    try:
        store.upsert(BUNDLE_ID, _identity("fa-06074-20260612T1340Z"), _values())
        for fmt, suffix in (("parquet", ".parquet"), ("md", ".md"), ("csv", ".csv")):
            path, count = rs.export(bundle, store, fmt)
            assert count == 1
            assert path.suffix == suffix
            assert path.parent == bundle_dir.resolve() / "exports"
            assert path.is_file()
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 7. Progress and the drift check
# ---------------------------------------------------------------------------


def test_decile_of_is_monotone_and_spreads_small_samples() -> None:
    assert [rs.decile_of(r, 20) for r in range(1, 21)] == [
        1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10,
    ]
    assert [rs.decile_of(r, 5) for r in range(1, 6)] == [1, 3, 5, 7, 9]
    assert rs.decile_of(1, 1) == 1


def test_progress_ranks_by_review_order_not_by_seq_value() -> None:
    """``review_seq`` is monotone but not contiguous — gaps are normal.

    A withdrawn judgement leaves one, and a bundle re-drawn mid-review
    leaves several. A decile computed from the number rather than from the
    rank would put all twenty of these rows in the first bucket and the
    drift check would read as "no drift" for ever.
    """
    rows = [
        _annotation(
            f"fa-{i:02d}", "false_alarm",
            "real_failure" if i < 10 else "metric_artefact",
            ["fa_cell_died"], (i + 1) * 10,
        )
        for i in range(20)
    ]
    payload = rs.progress_payload(rows, {})
    deciles = payload["deciles"]
    assert [d["decile"] for d in deciles] == list(range(1, 11))
    assert [d["n"] for d in deciles] == [2] * 10
    assert [d["seq_from"] for d in deciles] == [
        10, 30, 50, 70, 90, 110, 130, 150, 170, 190,
    ]
    assert [d["seq_to"] for d in deciles] == [
        20, 40, 60, 80, 100, 120, 140, 160, 180, 200,
    ]
    assert [d["verdicts"]["real_failure"] for d in deciles] == [2] * 5 + [0] * 5


def test_progress_deciles_for_a_hand_worked_sequence(server) -> None:
    """The same, end to end: twenty judged rows and one deliberately not."""
    store = server.store
    identity = _identity("fa-06074-20260612T1340Z")
    for i in range(20):
        # review_seq 1..20 by construction (one upsert each), but written
        # against distinct ids so nothing collides.
        store.upsert(
            BUNDLE_ID,
            rs.EventIdentity(
                event_id=f"fa-x{i:02d}", station_id="06074",
                anchor_utc="2026-06-12T13:40:00+00:00",
                event_class="false_alarm", dual_truth="both_dry",
                season="summer", region="fyn",
            ),
            _values(
                verdict="real_failure" if i < 10 else "metric_artefact",
                tags=["fa_cell_died"] if i % 2 == 0 else ["fa_virga_or_aloft"],
            ),
        )
    # One row deliberately left unjudged: it must not enter the ranking.
    store.upsert(BUNDLE_ID, identity, _values(verdict=None, tags=[]))

    port = server.server_address[1]
    status, payload, _ = _request(port, "GET", "/review-api/progress")
    assert status == 200
    assert payload["bundle_id"] == BUNDLE_ID
    assert payload["stored"] == 21
    assert payload["annotated"] == 20

    deciles = payload["deciles"]
    assert [d["decile"] for d in deciles] == list(range(1, 11))
    assert [d["n"] for d in deciles] == [2] * 10
    assert [d["seq_from"] for d in deciles] == [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    assert [d["seq_to"] for d in deciles] == [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    # The verdict flipped exactly half way, which is what a drift check is
    # meant to make visible.
    assert [d["verdicts"]["real_failure"] for d in deciles] == [2] * 5 + [0] * 5
    assert [d["verdicts"]["metric_artefact"] for d in deciles] == [0] * 5 + [2] * 5
    assert deciles[0]["tags"] == {"fa_cell_died": 1, "fa_virga_or_aloft": 1}

    assert payload["by_tag"]["fa_cell_died"] == 10
    assert payload["by_class"]["false_alarm"]["annotated"] == 20
    assert payload["by_class"]["false_alarm"]["events"] == 1, (
        "the bundle index is the event denominator, not the annotation table"
    )
    assert payload["by_verdict"] == {
        "real_failure": 10, "metric_artefact": 10, "unclear": 0,
    }
