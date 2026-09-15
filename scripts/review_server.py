"""Serve one review bundle to a browser, and durably keep the judgements.

::

    python scripts/review_server.py --bundle review/bundle-2026-09-15
    # then open the dev-only /review route of the frontend (npm run dev),
    # which proxies /review-api and /review-data here (vite.config.ts).

The bundle is read-only input: a directory of JSON documents plus radar
PNGs built on the VM by ``scripts/build_review_bundle.py`` and copied to a
laptop (see :mod:`dmi_nowcast_core.review_schema` for its layout). This
server adds the only thing the bundle cannot carry — the human's verdict
on each event — and it is that annotation database, not the bundle, that
is the product of the whole exercise.

Three consequences shape everything below.

**The database lives outside the bundle.** A bundle is rebuilt whenever
the population is re-drawn or an event is deepened, and a rebuild is a
fresh directory. So the default database path is the bundle's *parent*
directory, and the table is keyed ``(bundle_id, event_id)`` — one file
accumulates every bundle a reviewer ever works through, and re-drawing a
bundle cannot take their judgements with it. ``review_schema.event_id``
is derived from (class, station, instant) precisely so that the same
event keeps the same key across rebuilds.

**An export must stand alone.** Each annotation row repeats the event's
identity — station, anchor instant, class, dual truth, season, region —
rather than pointing at the bundle for them. Denormalisation is normally
a smell; here it is the requirement. A year of judgements that can only
be read while a 250 MB directory of radar PNGs still exists is a year of
judgements one ``rm -rf`` from being a list of opaque ids.

**Reviewer drift is measurable or it is invisible.** Over 300 events and
many sessions a human's criteria move: the twentieth false alarm is
judged against a mental model the first one did not have. So every row
carries ``review_seq`` — the order in which it was *first* saved, which
never changes on an edit — and every write appends the whole row to
``annotation_history``. Tag rates by review-order decile are then a
query, and "what did I say before I changed my mind" is answerable.

The vocabulary is controlled at this boundary. An unknown verdict or tag
code is a 422 naming the offender, never a silent insert: a free-text tag
field over 300 events produces 300 unique tags and no pattern, which is
exactly the failure this tool exists to avoid.

Stdlib only (plus ``pyarrow`` for the Parquet export, already a dev
dependency), because this runs on a laptop that may be offline behind a
VPN that blocks the VM, and one file with no install step is the whole
point. It binds to 127.0.0.1 and nowhere else: the bundle contains
unreleased analysis of a production service's failures.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import mimetypes
import sqlite3
import sys
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final
from urllib.parse import parse_qs, unquote, urlsplit

# The reviewer's laptop need not have the package installed: this is a
# stdlib tool run out of a checkout, and the repo convention is to make
# ``src/`` importable rather than to require an editable install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from dmi_nowcast_core import review_schema  # noqa: E402

_log = logging.getLogger("review_server")

#: Loopback only. Never make this configurable — see the module docstring.
HOST: Final[str] = "127.0.0.1"

#: Matches the unconditional dev proxy in ``frontend/vite.config.ts``.
DEFAULT_PORT: Final[int] = 8770

#: The annotation database's own layout version, independent of the
#: bundle's ``REVIEW_SCHEMA_VERSION``: the two change for different
#: reasons and a bundle rebuild must not imply a database migration.
DB_VERSION: Final[int] = 1

#: Default filename, placed beside the bundle directory rather than in it.
DEFAULT_DB_NAME: Final[str] = "review-annotations.sqlite"

#: The control group every mechanism rate is read against. A tag that
#: describes 40 % of the hits as well as 40 % of the false alarms explains
#: nothing; only the contrast says so.
CONTROL_CLASS: Final[str] = "hit"


# ---------------------------------------------------------------------------
# Time helpers — ISO 8601 with an explicit offset, everywhere, always UTC
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(value: datetime) -> str:
    """Aware datetime → ``2026-06-12T13:40:00+00:00``. Naive assumed UTC."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 instant, requiring a real offset.

    A naive stamp is rejected rather than assumed UTC. The bundle states
    that every instant in it carries ``+00:00``; accepting a naive string
    here would let a browser in CEST write local time into a column every
    later aggregation reads as UTC, and nothing would ever complain.
    """
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("timestamp needs an explicit UTC offset")
    return parsed.astimezone(timezone.utc)


def _stamp(now: datetime) -> str:
    """Filename stamp for an export: ``20260612T134000Z``."""
    return now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# The annotation store
# ---------------------------------------------------------------------------

_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS annotations (
    bundle_id          TEXT    NOT NULL,
    event_id           TEXT    NOT NULL,
    station_id         TEXT    NOT NULL,
    anchor_utc         TEXT    NOT NULL,
    event_class        TEXT    NOT NULL,
    dual_truth         TEXT    NOT NULL,
    season             TEXT    NOT NULL,
    region             TEXT    NOT NULL,
    verdict            TEXT,
    tags               TEXT    NOT NULL DEFAULT '[]',
    vocab_version      INTEGER NOT NULL DEFAULT 1,
    confidence         INTEGER,
    needs_second_look  INTEGER NOT NULL DEFAULT 0,
    note               TEXT    NOT NULL DEFAULT '',
    cursor_utc         TEXT,
    review_seq         INTEGER NOT NULL,
    reviewer           TEXT    NOT NULL DEFAULT 'local',
    created_utc        TEXT    NOT NULL,
    updated_utc        TEXT    NOT NULL,
    revision           INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (bundle_id, event_id)
);

CREATE TABLE IF NOT EXISTS annotation_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    bundle_id   TEXT NOT NULL,
    event_id    TEXT NOT NULL,
    written_utc TEXT NOT NULL,
    revision    INTEGER NOT NULL,
    payload     TEXT NOT NULL
);

"""

#: The indexes, applied only **after** the migrations: an index names
#: columns, and a database written by an older shape may not have them yet.
#: A ``CREATE INDEX`` in the same script as the tables would abort the open
#: with "no such column" on exactly the file the migrations exist for.
_INDEX_SCHEMA: Final = """
CREATE INDEX IF NOT EXISTS annotations_class   ON annotations(bundle_id, event_class);
CREATE INDEX IF NOT EXISTS annotations_verdict ON annotations(bundle_id, verdict);
CREATE INDEX IF NOT EXISTS history_event       ON annotation_history(bundle_id, event_id);
"""

#: Every column of ``annotations``, in DDL order. Used for the SELECT list
#: and for the copy in a structural migration, so the two cannot drift.
_COLUMNS: Final[tuple[str, ...]] = (
    "bundle_id", "event_id",
    "station_id", "anchor_utc", "event_class", "dual_truth", "season", "region",
    "verdict", "tags", "vocab_version", "confidence", "needs_second_look",
    "note", "cursor_utc", "review_seq", "reviewer",
    "created_utc", "updated_utc", "revision",
)

#: Columns an older database may simply not have, with the definition used
#: to bolt them on. Every one is nullable or carries a literal default,
#: which is the only thing ``ALTER TABLE ADD COLUMN`` can do — anything
#: structural goes through :meth:`ReviewStore._migrate_composite_key`.
#:
#: ``review_seq`` gets ``DEFAULT 0`` here although the canonical DDL has no
#: default: rows that predate the column were reviewed before anything
#: else, and 0 sorts them exactly there.
_ADDITIVE_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    # Not in the first cut at all: it assumed a single bundle. Added empty
    # and filled by _migrate_composite_key, which is the only place that
    # knows which bundle those rows must have belonged to.
    ("bundle_id", "TEXT NOT NULL DEFAULT ''"),
    ("station_id", "TEXT NOT NULL DEFAULT ''"),
    ("anchor_utc", "TEXT NOT NULL DEFAULT ''"),
    ("event_class", "TEXT NOT NULL DEFAULT ''"),
    ("dual_truth", "TEXT NOT NULL DEFAULT ''"),
    ("season", "TEXT NOT NULL DEFAULT ''"),
    ("region", "TEXT NOT NULL DEFAULT ''"),
    ("verdict", "TEXT"),
    ("tags", "TEXT NOT NULL DEFAULT '[]'"),
    ("vocab_version", "INTEGER NOT NULL DEFAULT 1"),
    ("confidence", "INTEGER"),
    ("needs_second_look", "INTEGER NOT NULL DEFAULT 0"),
    ("note", "TEXT NOT NULL DEFAULT ''"),
    ("cursor_utc", "TEXT"),
    ("review_seq", "INTEGER NOT NULL DEFAULT 0"),
    ("reviewer", "TEXT NOT NULL DEFAULT 'local'"),
    ("created_utc", "TEXT NOT NULL DEFAULT ''"),
    ("updated_utc", "TEXT NOT NULL DEFAULT ''"),
    ("revision", "INTEGER NOT NULL DEFAULT 1"),
)


class RevisionConflict(Exception):
    """A conditional write lost: ``If-Match`` did not name the stored revision."""

    def __init__(self, expected: str, actual: int | None) -> None:
        super().__init__(f"expected revision {expected}, stored revision {actual}")
        self.expected = expected
        self.actual = actual


@dataclass(frozen=True)
class EventIdentity:
    """The denormalised half of an annotation row.

    Taken from the bundle's ``events.json`` index at write time and copied
    into the row, so the export survives the bundle's deletion. Note the
    rename: the index calls the field ``class`` (a Python keyword), the
    database calls it ``event_class``, and this is the single place the
    two are mapped.
    """

    event_id: str
    station_id: str
    anchor_utc: str
    event_class: str
    dual_truth: str
    season: str
    region: str


class ReviewStore:
    """Thread-safe SQLite annotation store: one connection behind a lock.

    ``ThreadingHTTPServer`` hands every request its own thread, so the
    idiom is the sidecar's :class:`~dmi_nowcast_sidecar.push.store.PushStore`
    one — a single connection with ``check_same_thread=False`` serialised
    by a :class:`threading.Lock`, WAL so a read never blocks behind a
    write, and ISO-8601 UTC strings in every timestamp column.
    """

    def __init__(self, db_path: Path, *, legacy_bundle_id: str | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            self._migrate_additive_columns()
            self._migrate_composite_key(legacy_bundle_id)
            self._migrate_backfill_review_seq()
            self._conn.executescript(_INDEX_SCHEMA)
            self._conn.commit()
            self._stamp_meta()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- migrations --------------------------------------------------------

    def _table_columns(self, table: str) -> list[sqlite3.Row]:
        return self._conn.execute(f"PRAGMA table_info({table})").fetchall()

    def _migrate_additive_columns(self) -> None:
        """Bolt on any column a previous version of this file did not write.

        The additive half of the migration story: a nullable column, or one
        with a literal default, is an ``ALTER TABLE ADD COLUMN`` and
        nothing more. Probing ``PRAGMA table_info`` rather than tracking a
        version number means a database written by any past shape converges
        on the current one. Called with ``self._lock`` held.
        """
        have = {row["name"] for row in self._table_columns("annotations")}
        added = [name for name, _ in _ADDITIVE_COLUMNS if name not in have]
        if not added:
            return
        for name, ddl in _ADDITIVE_COLUMNS:
            if name in have:
                continue
            self._conn.execute(f"ALTER TABLE annotations ADD COLUMN {name} {ddl}")
        rows = self._conn.execute(
            "SELECT COUNT(*) AS n FROM annotations",
        ).fetchone()["n"]
        self._conn.commit()
        _log.info(
            "review_store migrated: added column(s) %s to annotations "
            "(%d existing row(s) carried forward)",
            ", ".join(added), int(rows),
        )

    def _migrate_composite_key(self, legacy_bundle_id: str | None) -> None:
        """Re-key a single-bundle database on ``(bundle_id, event_id)``.

        The first cut of this tool assumed one bundle would ever exist and
        made ``event_id`` the whole primary key. The plan calls for a
        second bundle to be *additive*, and event ids are only unique
        within a bundle's draw, so that shape has to go. SQLite cannot
        change a primary key in place, so this is the documented
        create/copy/drop/rename, reduced to what two tables need and run
        inside one transaction: a crash halfway leaves the old table
        exactly as it was.

        A legacy row with no ``bundle_id`` is attributed to
        ``legacy_bundle_id`` — the bundle the server was started against.
        That is the correct fill and not a guess: a database written before
        the column existed could only ever have described one bundle, and
        the reviewer is by definition opening it now. Called with
        ``self._lock`` held.
        """
        columns = self._table_columns("annotations")
        primary = [c["name"] for c in sorted(columns, key=lambda c: c["pk"]) if c["pk"]]
        if primary == ["bundle_id", "event_id"]:
            return
        self._conn.execute("PRAGMA foreign_keys=OFF")
        columns_sql = ", ".join(_COLUMNS)
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            if legacy_bundle_id:
                # Two steps rather than a literal default in the ADD COLUMN,
                # so the bundle id never reaches SQL as interpolated text.
                self._conn.execute(
                    "UPDATE annotations SET bundle_id = ? "
                    "WHERE bundle_id = '' OR bundle_id IS NULL",
                    (legacy_bundle_id,),
                )
            self._conn.execute(
                _ANNOTATIONS_DDL.replace("annotations", "annotations_new", 1)
                .replace("IF NOT EXISTS ", ""),
            )
            self._conn.execute(
                f"INSERT INTO annotations_new ({columns_sql}) "
                f"SELECT {columns_sql} FROM annotations",
            )
            migrated = self._conn.execute(
                "SELECT COUNT(*) AS n FROM annotations_new",
            ).fetchone()["n"]
            self._conn.execute("DROP TABLE annotations")
            self._conn.execute("ALTER TABLE annotations_new RENAME TO annotations")
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._conn.execute("PRAGMA foreign_keys=ON")
        _log.info(
            "review_store migrated: primary key %s → (bundle_id, event_id), "
            "%d row(s) copied",
            tuple(primary) or "(rowid)", int(migrated),
        )

    def _migrate_backfill_review_seq(self) -> None:
        """Number rows that predate ``review_seq``, oldest judgement first.

        The additive migration can only give the column a literal default,
        so every pre-existing row arrives at 0 — and a drift check over a
        column where a third of the rows share one value is not a drift
        check. ``created_utc`` is the honest reconstruction of the order
        those events were first saved in, which is exactly what the column
        means. Called with ``self._lock`` held.
        """
        zeros = self._conn.execute(
            "SELECT bundle_id, event_id FROM annotations WHERE review_seq = 0 "
            "ORDER BY bundle_id ASC, created_utc ASC, event_id ASC",
        ).fetchall()
        if not zeros:
            return
        highs = {
            row["bundle_id"]: int(row["m"])
            for row in self._conn.execute(
                "SELECT bundle_id, COALESCE(MAX(review_seq), 0) AS m "
                "FROM annotations GROUP BY bundle_id",
            ).fetchall()
        }
        for row in zeros:
            bundle_id = row["bundle_id"]
            seq = highs.get(bundle_id, 0) + 1
            highs[bundle_id] = seq
            self._conn.execute(
                "UPDATE annotations SET review_seq = ? "
                "WHERE bundle_id = ? AND event_id = ?",
                (seq, bundle_id, row["event_id"]),
            )
        for bundle_id, high in highs.items():
            self._conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (f"review_seq_high:{bundle_id}", str(high)),
            )
        self._conn.commit()
        _log.info(
            "review_store migrated: review_seq backfilled by created_utc for "
            "%d row(s)", len(zeros),
        )

    def _stamp_meta(self) -> None:
        """Record what wrote this file. Called with ``self._lock`` held."""
        for key, value in (
            ("db_version", str(DB_VERSION)),
            ("review_schema_version", str(review_schema.REVIEW_SCHEMA_VERSION)),
            ("vocab_version", str(review_schema.REVIEW_VOCAB_VERSION)),
        ):
            self._conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
        self._conn.execute(
            "INSERT INTO schema_meta (key, value) VALUES ('created_utc', ?) "
            "ON CONFLICT(key) DO NOTHING",
            (_to_iso(_now_utc()),),
        )
        self._conn.commit()

    def meta(self) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute("SELECT key, value FROM schema_meta").fetchall()
        return {row["key"]: row["value"] for row in rows}

    # -- reads -------------------------------------------------------------

    def get(self, bundle_id: str, event_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM annotations "
                "WHERE bundle_id = ? AND event_id = ?",
                (bundle_id, event_id),
            ).fetchone()
        return _row_to_dict(row) if row is not None else None

    def list(self, bundle_id: str) -> list[dict[str, Any]]:
        """Every stored row for one bundle, in the order it was first saved."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM annotations "
                "WHERE bundle_id = ? ORDER BY review_seq ASC, event_id ASC",
                (bundle_id,),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def history(self, bundle_id: str, event_id: str) -> list[dict[str, Any]]:
        """Every version this event's annotation has ever had, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT written_utc, revision, payload FROM annotation_history "
                "WHERE bundle_id = ? AND event_id = ? ORDER BY id ASC",
                (bundle_id, event_id),
            ).fetchall()
        return [
            {
                "written_utc": r["written_utc"],
                "revision": int(r["revision"]),
                "payload": json.loads(r["payload"]),
            }
            for r in rows
        ]

    def counts(self, bundle_id: str) -> tuple[int, int]:
        """``(stored rows, rows carrying a verdict)`` for one bundle."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n, "
                "COALESCE(SUM(verdict IS NOT NULL), 0) AS judged "
                "FROM annotations WHERE bundle_id = ?",
                (bundle_id,),
            ).fetchone()
        return int(row["n"]), int(row["judged"])

    # -- writes ------------------------------------------------------------

    def upsert(
        self,
        bundle_id: str,
        identity: EventIdentity,
        values: Mapping[str, Any],
        *,
        if_match: str | None = None,
        now_utc: datetime | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Create or replace one annotation. Returns ``(row, created)``.

        ``values`` must already have been through :func:`validate_annotation`
        — this method trusts it and writes.

        Three things deliberately survive an edit: ``created_utc`` (when
        this event was first judged), ``review_seq`` (where it fell in the
        review order — the drift check reads it, so a re-judgement must not
        renumber it) and the row's place in ``annotation_history``, which
        gains one entry per write and never loses one.

        ``if_match`` is the caller's copy of ``revision``: ``"*"`` requires
        the row to exist, an integer requires it to still be at that
        revision, and ``"0"`` requires that it does not exist yet. A
        mismatch raises :class:`RevisionConflict` and writes nothing —
        two browser tabs open on the same event is a routine accident, and
        last-write-wins would silently discard the earlier judgement.
        """
        now = now_utc or _now_utc()
        now_iso = _to_iso(now)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                current = self._conn.execute(
                    f"SELECT {', '.join(_COLUMNS)} FROM annotations "
                    "WHERE bundle_id = ? AND event_id = ?",
                    (bundle_id, identity.event_id),
                ).fetchone()
                stored_revision = int(current["revision"]) if current else None
                if if_match is not None:
                    self._check_if_match(if_match, stored_revision)
                if current is None:
                    seq = self._next_review_seq(bundle_id)
                    revision = 1
                    created = now_iso
                else:
                    seq = int(current["review_seq"])
                    revision = int(current["revision"]) + 1
                    created = current["created_utc"]
                row = {
                    "bundle_id": bundle_id,
                    **asdict(identity),
                    "verdict": values["verdict"],
                    "tags": list(values["tags"]),
                    "vocab_version": int(values["vocab_version"]),
                    "confidence": values["confidence"],
                    "needs_second_look": bool(values["needs_second_look"]),
                    "note": values["note"],
                    "cursor_utc": values["cursor_utc"],
                    "review_seq": seq,
                    "reviewer": values["reviewer"],
                    "created_utc": created,
                    "updated_utc": now_iso,
                    "revision": revision,
                }
                self._conn.execute(
                    f"INSERT INTO annotations ({', '.join(_COLUMNS)}) VALUES "
                    f"({', '.join('?' for _ in _COLUMNS)}) "
                    "ON CONFLICT(bundle_id, event_id) DO UPDATE SET "
                    + ", ".join(
                        f"{c} = excluded.{c}"
                        for c in _COLUMNS
                        if c not in ("bundle_id", "event_id")
                    ),
                    _row_to_params(row),
                )
                self._append_history(row, now_iso)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return row, current is None

    def delete(
        self,
        bundle_id: str,
        event_id: str,
        *,
        if_match: str | None = None,
        now_utc: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Clear one annotation, keeping its history. Returns the row removed.

        The deletion itself is appended to ``annotation_history`` with the
        row as it stood plus ``"deleted": true``. Otherwise an accidental
        delete is indistinguishable from an event that was never judged,
        and the history would claim a judgement that no longer exists with
        nothing to say it was withdrawn.
        """
        now_iso = _to_iso(now_utc or _now_utc())
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                current = self._conn.execute(
                    f"SELECT {', '.join(_COLUMNS)} FROM annotations "
                    "WHERE bundle_id = ? AND event_id = ?",
                    (bundle_id, event_id),
                ).fetchone()
                stored_revision = int(current["revision"]) if current else None
                if if_match is not None:
                    self._check_if_match(if_match, stored_revision)
                if current is None:
                    self._conn.rollback()
                    return None
                row = _row_to_dict(current)
                self._conn.execute(
                    "DELETE FROM annotations WHERE bundle_id = ? AND event_id = ?",
                    (bundle_id, event_id),
                )
                self._append_history(
                    {**row, "revision": row["revision"] + 1, "deleted": True},
                    now_iso,
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return row

    # -- write helpers (all called with ``self._lock`` held) ---------------

    @staticmethod
    def _check_if_match(if_match: str, stored_revision: int | None) -> None:
        """Enforce a conditional write, accepting the ETag spellings.

        A browser echoes back the ``ETag`` it was given, quoted and
        possibly marked weak (``W/"3"``), while a ``curl -H 'If-Match: 3'``
        will not quote it at all. All three mean the same revision, and
        refusing two of them would only teach the reviewer to stop sending
        the header.
        """
        text = if_match.strip()
        if text.startswith(("W/", "w/")):
            text = text[2:].strip()
        text = text.strip('"')
        if text == "*":
            if stored_revision is None:
                raise RevisionConflict("*", None)
            return
        try:
            expected = int(text)
        except ValueError:
            raise RevisionConflict(if_match, stored_revision) from None
        if expected != (stored_revision or 0):
            raise RevisionConflict(text, stored_revision)

    def _next_review_seq(self, bundle_id: str) -> int:
        """The next review-order number for this bundle, monotone for ever.

        Taken as one past the high-water mark rather than one past the
        current maximum: an event that is judged, deleted and judged again
        must not reuse the number of the annotation it replaced, or the
        drift check reads two different judgements as one moment in the
        review order.
        """
        key = f"review_seq_high:{bundle_id}"
        stored = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key = ?", (key,),
        ).fetchone()
        in_table = self._conn.execute(
            "SELECT COALESCE(MAX(review_seq), 0) AS m FROM annotations "
            "WHERE bundle_id = ?",
            (bundle_id,),
        ).fetchone()["m"]
        high = max(int(stored["value"]) if stored else 0, int(in_table))
        nxt = high + 1
        self._conn.execute(
            "INSERT INTO schema_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(nxt)),
        )
        return nxt

    def _append_history(self, row: Mapping[str, Any], written_utc: str) -> None:
        self._conn.execute(
            "INSERT INTO annotation_history "
            "(bundle_id, event_id, written_utc, revision, payload) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                row["bundle_id"], row["event_id"], written_utc,
                int(row["revision"]),
                json.dumps(row, sort_keys=True, ensure_ascii=False),
            ),
        )


#: The ``annotations`` DDL alone, for the structural migration's rebuild.
_ANNOTATIONS_DDL: Final[str] = next(
    stmt.strip() + ";"
    for stmt in _SCHEMA.split(";")
    if "CREATE TABLE IF NOT EXISTS annotations (" in stmt
)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """One stored row as the API shape: JSON tags parsed, ints made bools."""
    return {
        "bundle_id": row["bundle_id"],
        "event_id": row["event_id"],
        "station_id": row["station_id"],
        "anchor_utc": row["anchor_utc"],
        "event_class": row["event_class"],
        "dual_truth": row["dual_truth"],
        "season": row["season"],
        "region": row["region"],
        "verdict": row["verdict"],
        "tags": json.loads(row["tags"] or "[]"),
        "vocab_version": int(row["vocab_version"]),
        "confidence": None if row["confidence"] is None else int(row["confidence"]),
        "needs_second_look": bool(row["needs_second_look"]),
        "note": row["note"],
        "cursor_utc": row["cursor_utc"],
        "review_seq": int(row["review_seq"]),
        "reviewer": row["reviewer"],
        "created_utc": row["created_utc"],
        "updated_utc": row["updated_utc"],
        "revision": int(row["revision"]),
    }


def _row_to_params(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """API shape → the positional parameters of an INSERT over ``_COLUMNS``."""
    out: list[Any] = []
    for column in _COLUMNS:
        value = row[column]
        if column == "tags":
            value = json.dumps(list(value), ensure_ascii=False)
        elif column == "needs_second_look":
            value = int(bool(value))
        out.append(value)
    return tuple(out)


# ---------------------------------------------------------------------------
# Validation — the controlled half of "controlled vocabulary"
# ---------------------------------------------------------------------------

#: Every key a PUT body may carry. An unknown one is a 422 rather than a
#: silent drop: ``{"tag": [...]}`` for ``{"tags": [...]}`` would otherwise
#: store an empty tag list and look like a successful save.
_PUT_FIELDS: Final[frozenset[str]] = frozenset({
    "verdict", "tags", "confidence", "needs_second_look",
    "note", "cursor_utc", "vocab_version", "reviewer",
})

#: Server-owned fields a client may echo back and we simply ignore. The
#: obvious browser implementation is "GET the row, edit it, PUT it back",
#: and rejecting the identity and bookkeeping columns it read from us would
#: punish the most natural client for our own response shape.
_READ_ONLY_FIELDS: Final[frozenset[str]] = frozenset({
    "bundle_id", "event_id", "station_id", "anchor_utc", "event_class",
    "dual_truth", "season", "region", "review_seq",
    "created_utc", "updated_utc", "revision",
})


class ValidationError(Exception):
    """A rejected PUT body, carrying the offending values for the 422."""

    def __init__(self, problems: list[dict[str, Any]]) -> None:
        super().__init__("; ".join(p["message"] for p in problems))
        self.problems = problems


def validate_annotation(body: Any) -> dict[str, Any]:
    """Validate a PUT body and fill its defaults, or raise.

    The tag check is against :func:`review_schema.all_tag_codes`, which is
    documented as "every code the server will accept" — deliberately the
    union over classes, not ``tags_for_class``. The hit control group is
    tagged from *both* cause lists on purpose (that is what makes it a base
    rate), and a reviewer who reaches for a false-alarm tag on an
    ``uncovered`` event is telling us something about the coverage rule,
    not making a mistake worth blocking.
    """
    problems: list[dict[str, Any]] = []
    if not isinstance(body, dict):
        raise ValidationError([{
            "field": "_body", "offending": [],
            "message": "body must be a JSON object",
        }])

    unknown = sorted(set(body) - _PUT_FIELDS - _READ_ONLY_FIELDS)
    if unknown:
        problems.append({
            "field": "_body", "offending": unknown,
            "message": f"unknown field(s): {', '.join(unknown)}",
        })

    verdict = body.get("verdict")
    if verdict is not None and verdict not in review_schema.VERDICTS:
        problems.append({
            "field": "verdict", "offending": [verdict],
            "message": (
                f"unknown verdict {verdict!r}; expected one of "
                f"{', '.join(sorted(review_schema.VERDICTS))}"
            ),
        })

    raw_tags = body.get("tags", [])
    tags: list[str] = []
    if not isinstance(raw_tags, list) or any(not isinstance(t, str) for t in raw_tags):
        problems.append({
            "field": "tags", "offending": [],
            "message": "tags must be a list of vocabulary codes",
        })
    else:
        known = review_schema.all_tag_codes()
        bad = [t for t in raw_tags if t not in known]
        if bad:
            problems.append({
                "field": "tags", "offending": sorted(set(bad)),
                "message": f"unknown tag code(s): {', '.join(sorted(set(bad)))}",
            })
        # Dedupe, keeping the reviewer's order: the UI can send a tag twice
        # by double-clicking and that is not an error worth a 422.
        tags = list(dict.fromkeys(raw_tags))

    confidence = body.get("confidence")
    if confidence is not None and (
        not isinstance(confidence, int)
        or isinstance(confidence, bool)
        or not 1 <= confidence <= 3
    ):
        problems.append({
            "field": "confidence", "offending": [confidence],
            "message": "confidence must be null or an integer 1..3",
        })

    needs_second_look = body.get("needs_second_look", False)
    if not isinstance(needs_second_look, bool):
        problems.append({
            "field": "needs_second_look", "offending": [needs_second_look],
            "message": "needs_second_look must be a boolean",
        })

    note = body.get("note", "")
    if not isinstance(note, str):
        problems.append({
            "field": "note", "offending": [], "message": "note must be a string",
        })

    cursor_utc = body.get("cursor_utc")
    if cursor_utc is not None:
        if not isinstance(cursor_utc, str):
            problems.append({
                "field": "cursor_utc", "offending": [cursor_utc],
                "message": "cursor_utc must be an ISO-8601 string or null",
            })
        else:
            try:
                cursor_utc = _to_iso(_parse_iso(cursor_utc))
            except ValueError as exc:
                problems.append({
                    "field": "cursor_utc", "offending": [cursor_utc],
                    "message": f"cursor_utc: {exc}",
                })

    vocab_version = body.get("vocab_version", review_schema.REVIEW_VOCAB_VERSION)
    if not isinstance(vocab_version, int) or isinstance(vocab_version, bool):
        problems.append({
            "field": "vocab_version", "offending": [vocab_version],
            "message": "vocab_version must be an integer",
        })

    reviewer = body.get("reviewer", "local")
    if not isinstance(reviewer, str) or not reviewer:
        problems.append({
            "field": "reviewer", "offending": [reviewer],
            "message": "reviewer must be a non-empty string",
        })

    if problems:
        raise ValidationError(problems)
    return {
        "verdict": verdict,
        "tags": tags,
        "confidence": confidence,
        "needs_second_look": needs_second_look,
        "note": note,
        "cursor_utc": cursor_utc,
        "vocab_version": vocab_version,
        "reviewer": reviewer,
    }


# ---------------------------------------------------------------------------
# The bundle: read-only input
# ---------------------------------------------------------------------------


class Bundle:
    """One review bundle on disk, read lazily and re-read when it changes.

    Only the index is parsed here. Detail documents and frames are served
    straight off disk by ``/review-data/`` — the browser needs three
    details at a time out of ~300, and parsing 10 MB of JSON to hand back
    35 KB would be work done 297 times for nothing.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"bundle directory not found: {self.root}")
        self._lock = threading.Lock()
        self._index_mtime: float | None = None
        self._events: list[dict[str, Any]] = []
        self._by_id: dict[str, EventIdentity] = {}
        manifest_path = self.root / "manifest.json"
        self.manifest: dict[str, Any] = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file() else {}
        )
        self.bundle_id: str = str(
            self.manifest.get("bundle_id") or self.root.name,
        )
        self.schema_version: int = int(
            self.manifest.get("schema_version", review_schema.REVIEW_SCHEMA_VERSION),
        )
        self._load_index()

    # -- index -------------------------------------------------------------

    def _load_index(self) -> None:
        """(Re-)parse ``events.json`` when its mtime moved.

        A bundle can be rebuilt under a running server (``--deepen`` is
        cheap and the reviewer will do it mid-session). Re-reading on mtime
        keeps the identity map honest without a restart; the annotations
        are untouched because they live in a different file entirely.
        """
        path = self.root / "events.json"
        if not path.is_file():
            self._events, self._by_id, self._index_mtime = [], {}, None
            return
        mtime = path.stat().st_mtime
        with self._lock:
            if self._index_mtime == mtime:
                return
            doc = json.loads(path.read_text(encoding="utf-8"))
            events = list(doc.get("events") or [])
            self._events = events
            self._by_id = {
                str(row["event_id"]): _identity_of(row)
                for row in events
                if row.get("event_id")
            }
            self._index_mtime = mtime

    @property
    def events(self) -> list[dict[str, Any]]:
        self._load_index()
        return self._events

    def identity(self, event_id: str) -> EventIdentity | None:
        self._load_index()
        return self._by_id.get(event_id)

    def identities(self) -> dict[str, EventIdentity]:
        self._load_index()
        return dict(self._by_id)

    # -- vocabulary --------------------------------------------------------

    def vocabulary(self) -> dict[str, Any]:
        """``tags.json`` as the bundle shipped it, else the frozen module's.

        Serving the bundle's own copy matters: a reviewer must see the
        vocabulary their events were drawn under, not whatever this file
        was last updated to. The fallback covers a bundle built before
        ``tags.json`` existed.
        """
        path = self.root / "tags.json"
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                _log.warning("tags.json is unreadable; serving the built-in copy")
        return review_schema.tags_document()

    # -- static files ------------------------------------------------------

    def resolve(self, relative: str) -> Path | None:
        """Bundle-relative path → an absolute path inside the bundle, or None.

        ``Path.resolve`` follows symlinks before the containment check, so
        an absolute path, a ``..`` climb and a symlink pointing out of the
        tree are all caught by the same test. Returning ``None`` means 403,
        never 404: "that file is not there" and "you may not ask" must not
        be distinguishable from outside.
        """
        if "\x00" in relative:
            return None
        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(self.root):
            return None
        return candidate


def _identity_of(row: Mapping[str, Any]) -> EventIdentity:
    """An ``events.json`` index row → the denormalised identity we store."""
    return EventIdentity(
        event_id=str(row["event_id"]),
        station_id=str(row.get("station_id", "")),
        anchor_utc=str(row.get("anchor_utc", "")),
        event_class=str(row.get("class", "")),
        dual_truth=str(row.get("dual_truth", "")),
        season=str(row.get("season", "")),
        region=str(row.get("region", "")),
    )


# ---------------------------------------------------------------------------
# Progress: the cross-tab, the tag tally and the drift check
# ---------------------------------------------------------------------------


def decile_of(rank: int, n: int) -> int:
    """1-based review-order decile for ``rank`` (1-based) out of ``n``.

    Deciles of the *review order*, not of ``review_seq`` itself: the
    numbers are monotone but not contiguous (deleted annotations leave
    gaps), and what the drift check asks is "did my criteria move between
    my first tenth of the work and my last", which is a question about
    rank.

    With fewer than ten judged rows the deciles spread out rather than
    crowding into the first bucket — with n = 5 the ranks land on 1, 3, 5,
    7, 9 — which keeps the mapping monotone and honest about how little
    data it is describing.
    """
    if n <= 0:
        raise ValueError("no judged rows to rank")
    return min(10, (rank - 1) * 10 // n + 1)


def progress_payload(
    rows: Sequence[Mapping[str, Any]],
    identities: Mapping[str, EventIdentity],
) -> dict[str, Any]:
    """Counts by class × verdict, by tag, and by review-order decile."""
    judged = [r for r in rows if r.get("verdict")]
    by_class: dict[str, dict[str, Any]] = {}
    class_totals: dict[str, int] = {}
    for identity in identities.values():
        class_totals[identity.event_class] = class_totals.get(
            identity.event_class, 0,
        ) + 1
    for event_class in _classes_present(class_totals, rows):
        entries = [r for r in rows if r.get("event_class") == event_class]
        judged_here = [r for r in entries if r.get("verdict")]
        by_class[event_class] = {
            "events": class_totals.get(event_class, 0),
            "stored": len(entries),
            "annotated": len(judged_here),
            "verdicts": {
                verdict: sum(1 for r in judged_here if r["verdict"] == verdict)
                for verdict in review_schema.VERDICTS
            },
        }

    by_tag: dict[str, int] = {}
    for row in judged:
        for tag in row.get("tags") or []:
            by_tag[tag] = by_tag.get(tag, 0) + 1

    deciles: list[dict[str, Any]] = []
    ordered = sorted(judged, key=lambda r: (r["review_seq"], r["event_id"]))
    n = len(ordered)
    if n:
        buckets: dict[int, list[Mapping[str, Any]]] = {}
        for rank, row in enumerate(ordered, start=1):
            buckets.setdefault(decile_of(rank, n), []).append(row)
        for decile in sorted(buckets):
            bucket = buckets[decile]
            tags: dict[str, int] = {}
            for row in bucket:
                for tag in row.get("tags") or []:
                    tags[tag] = tags.get(tag, 0) + 1
            deciles.append({
                "decile": decile,
                "n": len(bucket),
                "seq_from": bucket[0]["review_seq"],
                "seq_to": bucket[-1]["review_seq"],
                "verdicts": {
                    verdict: sum(1 for r in bucket if r["verdict"] == verdict)
                    for verdict in review_schema.VERDICTS
                },
                "tags": dict(sorted(tags.items())),
            })

    return {
        "events": len(identities),
        "stored": len(rows),
        "annotated": len(judged),
        "unreviewed": max(len(identities) - len(judged), 0),
        "needs_second_look": sum(1 for r in rows if r.get("needs_second_look")),
        "by_class": by_class,
        "by_verdict": {
            verdict: sum(1 for r in judged if r["verdict"] == verdict)
            for verdict in review_schema.VERDICTS
        },
        "by_tag": dict(sorted(by_tag.items(), key=lambda kv: (-kv[1], kv[0]))),
        "deciles": deciles,
    }


def _classes_present(
    class_totals: Mapping[str, int],
    rows: Iterable[Mapping[str, Any]],
) -> list[str]:
    """Classes in the canonical order, then anything unexpected, sorted."""
    seen = set(class_totals) | {r.get("event_class", "") for r in rows}
    seen.discard("")
    ordered = [c for c in review_schema.OUTCOME_CLASSES if c in seen]
    return ordered + sorted(seen - set(ordered))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

#: Scalar columns of an export, in order. ``tags`` / ``mechanism_tags``
#: (lists) and the ``tag_<code>`` booleans follow, built from the
#: vocabulary. Pinned by a test: a renamed column silently breaks every
#: saved DuckDB query over past exports.
EXPORT_SCALAR_COLUMNS: Final[tuple[str, ...]] = (
    "bundle_id", "event_id", "station_id", "anchor_utc",
    "event_class", "dual_truth", "season", "region",
    "verdict", "vocab_version", "confidence", "needs_second_look",
    "note", "cursor_utc", "review_seq", "review_decile", "reviewer",
    "created_utc", "updated_utc", "revision",
)


def export_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Stored rows → export rows: review decile added, mechanism tags split.

    ``review_decile`` is computed here rather than left to the query so
    that the Parquet, the markdown and ``sql/review_tags.sql`` cannot
    disagree about what a decile is. ``mechanism_tags`` carries the same
    list minus :data:`review_schema.NON_MECHANISM_TAGS`, which is how the
    SQL ranks mechanisms without restating the vocabulary.
    """
    judged = sorted(
        (r for r in rows if r.get("verdict")),
        key=lambda r: (r["review_seq"], r["event_id"]),
    )
    ranks = {
        r["event_id"]: decile_of(i, len(judged))
        for i, r in enumerate(judged, start=1)
    }
    out: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: (r["review_seq"], r["event_id"])):
        tags = list(row.get("tags") or [])
        out.append({
            **{k: row.get(k) for k in EXPORT_SCALAR_COLUMNS if k != "review_decile"},
            "review_decile": ranks.get(row["event_id"]),
            "tags": tags,
            "mechanism_tags": [
                t for t in tags if t not in review_schema.NON_MECHANISM_TAGS
            ],
        })
    return out


def write_parquet(rows: Sequence[Mapping[str, Any]], path: Path) -> int:
    """One row per annotation, tags in both shapes.

    ``tags`` stays a ``list<string>`` because it is the truth, and a
    ``tag_<code>`` boolean is added for every code in the vocabulary
    because a cross-tab over an unnested list is a join every time,
    whereas over booleans it is one ``GROUP BY``. Both, not either.

    Timestamps stay ISO-8601 strings rather than Arrow timestamps, as the
    calibration corpus does with ``event_time``: DuckDB's ``TIMESTAMPTZ``
    renders in the session timezone, and a reviewer in CEST reading a UTC
    analysis is exactly the confusion the bundle's time rules exist to
    prevent.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    codes = sorted(review_schema.all_tag_codes())
    columns: dict[str, pa.Array] = {}
    types: dict[str, pa.DataType] = {
        "vocab_version": pa.int32(), "confidence": pa.int32(),
        "review_seq": pa.int32(), "review_decile": pa.int32(),
        "revision": pa.int32(), "needs_second_look": pa.bool_(),
    }
    for name in EXPORT_SCALAR_COLUMNS:
        columns[name] = pa.array(
            [r.get(name) for r in rows], types.get(name, pa.string()),
        )
    columns["tags"] = pa.array(
        [list(r.get("tags") or []) for r in rows], pa.list_(pa.string()),
    )
    columns["mechanism_tags"] = pa.array(
        [list(r.get("mechanism_tags") or []) for r in rows], pa.list_(pa.string()),
    )
    for code in codes:
        columns[f"tag_{code}"] = pa.array(
            [code in (r.get("tags") or []) for r in rows], pa.bool_(),
        )
    table = pa.table(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return table.num_rows


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> int:
    """The same rows, for a quick eyeball. Tags joined with ``|``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [*EXPORT_SCALAR_COLUMNS, "tags"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{k: row.get(k) for k in EXPORT_SCALAR_COLUMNS},
                "tags": "|".join(row.get("tags") or []),
            })
    return len(rows)


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:.0f} %"


def render_markdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    bundle_id: str,
    identities: Mapping[str, EventIdentity] | None = None,
    generated_at_utc: str,
    control_class: str = CONTROL_CLASS,
) -> str:
    """The human-readable export: ``rows -> str``, pure and therefore testable.

    ``rows`` are :func:`export_rows` output. ``identities`` is the bundle's
    index, needed only to list the events nobody has judged yet — an export
    that does not say what is *missing* invites a conclusion drawn from the
    half of the sample that was easiest to judge.

    The mechanism tables carry the control group's rate beside each class's
    own. That column is the report's whole argument: ``fa_cell_died`` on
    40 % of the false alarms is a finding only if it is not also on 40 % of
    the hits, and a tally without the contrast cannot tell the difference.
    Tags in :data:`review_schema.NON_MECHANISM_TAGS` are reported, but in
    their own table and never ranked — "I could not decide" is not a
    mechanism and must not out-rank one.
    """
    identities = dict(identities or {})
    judged = [r for r in rows if r.get("verdict")]
    by_class = _classes_present(
        {i.event_class: 1 for i in identities.values()}, rows,
    )
    control = [r for r in judged if r.get("event_class") == control_class]

    lines: list[str] = []
    lines.append("# Event review annotations")
    lines.append("")
    lines.append(
        f"Bundle `{bundle_id}`, exported {generated_at_utc}. "
        f"{len(identities)} event(s) in the bundle, {len(rows)} with a stored "
        f"annotation, {len(judged)} judged, "
        f"{max(len(identities) - len(judged), 0)} still unreviewed."
    )
    lines.append("")
    lines.append(
        "A verdict of `metric_artefact` does NOT mean the event was fine — it "
        "means the scoring rule, not the forecast, produced the label. That "
        "split is the point of the exercise: the two need opposite fixes."
    )
    lines.append("")

    # -- verdicts by class ----------------------------------------------
    verdicts = list(review_schema.VERDICTS)
    lines.append("## Verdicts by outcome class")
    lines.append("")
    lines.append(
        "| class | events | judged | " + " | ".join(verdicts) + " |",
    )
    lines.append("| --- | ---: | ---: | " + " | ".join("---:" for _ in verdicts) + " |")
    for event_class in by_class:
        here = [r for r in judged if r.get("event_class") == event_class]
        total = sum(
            1 for i in identities.values() if i.event_class == event_class
        )
        counts = [
            str(sum(1 for r in here if r["verdict"] == v)) for v in verdicts
        ]
        lines.append(
            f"| {event_class} | {total} | {len(here)} | " + " | ".join(counts) + " |",
        )
    lines.append("")

    # -- mechanisms by class, against the control group -------------------
    lines.append(f"## Mechanisms by class (contrast: `{control_class}` control group)")
    lines.append("")
    if not control:
        lines.append(
            f"No judged `{control_class}` events yet, so every contrast column "
            "below is empty. Until the control group is reviewed a tag rate is "
            "a tally, not a finding."
        )
        lines.append("")
    for event_class in by_class:
        if event_class == control_class:
            continue
        here = [r for r in judged if r.get("event_class") == event_class]
        if not here:
            continue
        lines.append(f"### {event_class} — {len(here)} judged")
        lines.append("")
        lines.append("| tag | n | share | control | lift |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        ranked = _tag_rates(here, mechanisms_only=True)
        if not ranked:
            lines.append("| _(no mechanism tags)_ | 0 | – | – | – |")
        for tag, count in ranked:
            share = count / len(here)
            control_share = (
                sum(1 for r in control if tag in (r.get("tags") or [])) / len(control)
                if control else None
            )
            lift = (
                f"{share / control_share:.1f}×"
                if control_share else ("–" if control_share is None else "∞")
            )
            lines.append(
                f"| `{tag}` | {count} | {_pct(share)} | {_pct(control_share)} "
                f"| {lift} |",
            )
        lines.append("")

    # -- non-mechanism tags ------------------------------------------------
    lines.append("## Non-mechanism tags")
    lines.append("")
    lines.append(
        "Reported, never ranked: these say the reviewer could not decide or "
        "wants another look, so counting them as mechanisms would rank the "
        "hard cases above the explained ones."
    )
    lines.append("")
    lines.append("| tag | n | share of judged |")
    lines.append("| --- | ---: | ---: |")
    non_mechanism = _tag_rates(judged, mechanisms_only=False, non_mechanism_only=True)
    if not non_mechanism:
        lines.append("| _(none)_ | 0 | – |")
    for tag, count in non_mechanism:
        lines.append(
            f"| `{tag}` | {count} | {_pct(count / len(judged)) if judged else '–'} |",
        )
    lines.append("")

    # -- unreviewed --------------------------------------------------------
    stored_judged = {r["event_id"] for r in judged}
    unreviewed = [
        identity for event_id, identity in identities.items()
        if event_id not in stored_judged
    ]
    unreviewed.sort(key=lambda i: (i.event_class, i.anchor_utc, i.event_id))
    lines.append(f"## Unreviewed ({len(unreviewed)})")
    lines.append("")
    if not unreviewed:
        lines.append("None — every event in the bundle carries a verdict.")
        lines.append("")
    else:
        lines.append("| event | class | station | anchor |")
        lines.append("| --- | --- | --- | --- |")
        for identity in unreviewed[:50]:
            lines.append(
                f"| `{identity.event_id}` | {identity.event_class} "
                f"| {identity.station_id} | {identity.anchor_utc} |",
            )
        if len(unreviewed) > 50:
            lines.append("")
            lines.append(f"… and {len(unreviewed) - 50} more.")
        lines.append("")

    # -- notes -------------------------------------------------------------
    lines.append("## Notes, grouped by dominant tag")
    lines.append("")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in judged:
        if not (row.get("note") or "").strip():
            continue
        grouped.setdefault(_dominant_tag(row), []).append(row)
    if not grouped:
        lines.append("No notes.")
        lines.append("")
    for tag in sorted(grouped):
        lines.append(f"### {tag}")
        lines.append("")
        for row in sorted(grouped[tag], key=lambda r: r["review_seq"]):
            note = " ".join((row.get("note") or "").split())
            lines.append(
                f"- `{row['event_id']}` ({row.get('event_class')}, "
                f"{row.get('verdict')}) — {note}"
            )
        lines.append("")
    return "\n".join(lines)


def _tag_rates(
    rows: Sequence[Mapping[str, Any]],
    *,
    mechanisms_only: bool,
    non_mechanism_only: bool = False,
) -> list[tuple[str, int]]:
    """``[(tag, count), ...]`` most frequent first, ties broken by code."""
    counts: dict[str, int] = {}
    for row in rows:
        for tag in row.get("tags") or []:
            is_non_mechanism = tag in review_schema.NON_MECHANISM_TAGS
            if mechanisms_only and is_non_mechanism:
                continue
            if non_mechanism_only and not is_non_mechanism:
                continue
            counts[tag] = counts.get(tag, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def _dominant_tag(row: Mapping[str, Any]) -> str:
    """The tag a note is filed under: first mechanism tag, else first tag.

    Deterministic and shallow on purpose. Grouping notes is a reading aid,
    not an analysis — the analysis is the Parquet.
    """
    tags = list(row.get("tags") or [])
    for tag in tags:
        if tag not in review_schema.NON_MECHANISM_TAGS:
            return tag
    return tags[0] if tags else "(untagged)"


def export(
    bundle: Bundle,
    store: ReviewStore,
    fmt: str,
    *,
    now_utc: datetime | None = None,
) -> tuple[Path, int]:
    """Write one export into ``<bundle>/exports/``. Returns ``(path, rows)``.

    Exports land inside the bundle although the database does not: an
    export is a snapshot *of* a bundle's review and is meaningless beside a
    different one, while the database outlives every bundle. The bundle
    directory is gitignored, so neither reaches the repository.
    """
    now = now_utc or _now_utc()
    rows = export_rows(store.list(bundle.bundle_id))
    suffix = {"parquet": "parquet", "md": "md", "csv": "csv"}[fmt]
    path = bundle.root / "exports" / f"annotations-{_stamp(now)}.{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        written = write_parquet(rows, path)
    elif fmt == "csv":
        written = write_csv(rows, path)
    else:
        path.write_text(
            render_markdown(
                rows,
                bundle_id=bundle.bundle_id,
                identities=bundle.identities(),
                generated_at_utc=_to_iso(now.replace(microsecond=0)),
            ),
            encoding="utf-8",
        )
        written = len(rows)
    return path, written


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

_API_PREFIX: Final = "/review-api/"
_DATA_PREFIX: Final = "/review-data/"

#: Content types we serve from the bundle. Anything else is refused rather
#: than guessed: the bundle holds JSON and PNG, and a directory that grew a
#: ``.html`` file is a directory something unexpected happened to.
_ALLOWED_SUFFIXES: Final[frozenset[str]] = frozenset({".json", ".png"})


class ReviewHTTPServer(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` carrying the bundle and the store."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        *,
        bundle: Bundle,
        store: ReviewStore,
    ) -> None:
        super().__init__(address, handler)
        self.bundle = bundle
        self.store = store


class ReviewHandler(BaseHTTPRequestHandler):
    """The whole API. One method per verb, one dispatcher underneath."""

    server_version = "dmi-review/1"
    protocol_version = "HTTP/1.1"

    # -- plumbing ----------------------------------------------------------

    @property
    def bundle(self) -> Bundle:
        return self.server.bundle  # type: ignore[attr-defined]

    @property
    def store(self) -> ReviewStore:
        return self.server.store  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Silenced like the calibration dashboard's: the reviewer's terminal
        # is where the migration and export lines have to be visible, and a
        # per-frame access log buries them at 19 PNGs per event.
        return

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(
        self,
        status: int,
        payload: Any,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        merged = {"Cache-Control": "no-store", **(headers or {})}
        self._send(status, body, "application/json; charset=utf-8", headers=merged)

    def _error(self, status: int, error: str, **extra: Any) -> None:
        self._json(status, {"error": error, **extra})

    def _read_body(self) -> Any:
        """Read exactly ``Content-Length`` bytes and parse them as JSON.

        Exactly, because keep-alive is on: a body left unread desynchronises
        the connection and the next request reads this one's tail.
        """
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    # -- verbs -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_PUT(self) -> None:  # noqa: N802
        self._dispatch("PUT")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        parsed = urlsplit(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html") and method == "GET":
                self._send(
                    200,
                    (
                        f"dmi-nowcast review server\n"
                        f"bundle: {self.bundle.bundle_id} ({self.bundle.root})\n"
                        f"events: {len(self.bundle.events)}\n"
                        f"db:     {self.store.db_path}\n\n"
                        "The UI is the frontend's dev-only /review route "
                        "(npm run dev), which proxies here.\n"
                    ).encode("utf-8"),
                    "text/plain; charset=utf-8",
                )
                return
            if path.startswith(_DATA_PREFIX):
                if method != "GET":
                    self._error(405, "method_not_allowed")
                    return
                self._serve_bundle_file(path[len(_DATA_PREFIX):])
                return
            if path.startswith(_API_PREFIX):
                self._api(method, path[len(_API_PREFIX):], query)
                return
            self._error(404, "not_found", path=path)
        except json.JSONDecodeError as exc:
            self._error(400, "bad_json", detail=str(exc))
        except ValidationError as exc:
            self._error(422, "validation_failed", problems=exc.problems)
        except RevisionConflict as exc:
            self._error(
                409, "revision_conflict",
                expected=exc.expected, stored=exc.actual,
            )
        except BrokenPipeError:  # pragma: no cover - browser navigated away
            return
        except Exception as exc:  # noqa: BLE001
            _log.exception("unhandled error serving %s %s", method, self.path)
            self._error(500, "internal_error", detail=repr(exc))

    # -- API ---------------------------------------------------------------

    def _api(self, method: str, route: str, query: Mapping[str, list[str]]) -> None:
        bundle_id = self.bundle.bundle_id
        if route == "health" and method == "GET":
            stored, annotated = self.store.counts(bundle_id)
            self._json(200, {
                "ok": True,
                "bundle_id": bundle_id,
                "bundle_root": str(self.bundle.root),
                "events": len(self.bundle.events),
                "stored": stored,
                "annotated": annotated,
                "schema_version": self.bundle.schema_version,
                "vocab_version": review_schema.REVIEW_VOCAB_VERSION,
                "db_path": str(self.store.db_path),
            })
            return
        if route == "vocabulary" and method == "GET":
            self._json(200, self.bundle.vocabulary())
            return
        if route == "progress" and method == "GET":
            payload = progress_payload(
                self.store.list(bundle_id), self.bundle.identities(),
            )
            self._json(200, {"bundle_id": bundle_id, **payload})
            return
        if route == "export" and method == "POST":
            self._read_body()  # drain; the format is a query parameter
            fmt = (query.get("format") or ["parquet"])[0]
            if fmt not in ("parquet", "md", "csv"):
                self._error(400, "unknown_format", format=fmt,
                            expected=["parquet", "md", "csv"])
                return
            try:
                path, rows = export(self.bundle, self.store, fmt)
            except ImportError:
                self._error(
                    503, "pyarrow_missing",
                    detail="the parquet export needs pyarrow; md and csv do not",
                )
                return
            _log.info("exported %d annotation row(s) to %s", rows, path)
            self._json(200, {"path": str(path), "rows": rows, "format": fmt})
            return
        if route == "annotations" and method == "GET":
            self._json(200, {
                "bundle_id": bundle_id,
                "annotations": self.store.list(bundle_id),
            })
            return
        if route.startswith("annotations/"):
            event_id = route[len("annotations/"):]
            if not event_id or "/" in event_id:
                self._error(404, "not_found", path=self.path)
                return
            self._annotation(method, bundle_id, event_id)
            return
        self._error(404, "not_found", path=self.path)

    def _annotation(self, method: str, bundle_id: str, event_id: str) -> None:
        if method == "GET":
            row = self.store.get(bundle_id, event_id)
            if row is None:
                self._error(404, "not_annotated", event_id=event_id)
                return
            self._json(200, row, headers={"ETag": f'"{row["revision"]}"'})
            return
        if method == "PUT":
            body = self._read_body()
            values = validate_annotation(body)
            identity = self.bundle.identity(event_id)
            if identity is None:
                # The identity columns are copied from the bundle's index,
                # so an event the bundle does not know cannot be stored
                # without inventing the station, class and instant that the
                # export exists to preserve.
                self._error(404, "unknown_event", event_id=event_id)
                return
            row, _created = self.store.upsert(
                bundle_id, identity, values,
                if_match=self.headers.get("If-Match"),
            )
            self._json(200, row, headers={"ETag": f'"{row["revision"]}"'})
            return
        if method == "DELETE":
            row = self.store.delete(
                bundle_id, event_id, if_match=self.headers.get("If-Match"),
            )
            if row is None:
                self._error(404, "not_annotated", event_id=event_id)
                return
            self._json(200, {"event_id": event_id, "deleted": True, "was": row})
            return
        self._error(405, "method_not_allowed", method=method)

    # -- bundle files ------------------------------------------------------

    def _serve_bundle_file(self, relative: str) -> None:
        """Serve one file out of the bundle, or refuse.

        Caching is the opposite way round for the two kinds of file. Frame
        names are content-stamped (``<radar stamp>.overlay.png``) and a
        stamp's pixels never change, so they are immutable for a year and
        the scrubber never re-fetches. The JSON documents are rewritten by
        every rebuild under the same names, so they are ``no-store`` — a
        cached ``events.json`` after a ``--deepen`` run is a reviewer
        judging events the bundle no longer describes.
        """
        resolved = self.bundle.resolve(relative)
        if resolved is None:
            self._error(403, "forbidden", path=relative)
            return
        if resolved.suffix not in _ALLOWED_SUFFIXES:
            self._error(403, "forbidden_type", suffix=resolved.suffix)
            return
        if not resolved.is_file():
            self._error(404, "not_found", path=relative)
            return
        body = resolved.read_bytes()
        if resolved.suffix == ".png":
            content_type = "image/png"
            cache = "public, max-age=31536000, immutable"
        else:
            content_type = (
                mimetypes.guess_type(resolved.name)[0]
                or "application/octet-stream"
            )
            cache = "no-store"
        self._send(200, body, content_type, headers={"Cache-Control": cache})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_server(
    bundle_root: Path,
    db_path: Path | None = None,
    port: int = DEFAULT_PORT,
) -> ReviewHTTPServer:
    """Bundle + database + a socket. ``port=0`` picks a free one (tests)."""
    bundle = Bundle(bundle_root)
    store = ReviewStore(
        db_path or default_db_path(bundle.root),
        legacy_bundle_id=bundle.bundle_id,
    )
    return ReviewHTTPServer(
        (HOST, port), ReviewHandler, bundle=bundle, store=store,
    )


def default_db_path(bundle_root: Path) -> Path:
    """Beside the bundle, never inside it — see the module docstring."""
    return Path(bundle_root).resolve().parent / DEFAULT_DB_NAME


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Serve one review bundle to a browser on 127.0.0.1 and store the "
            "judgements it produces in SQLite beside the bundle. The UI is "
            "the frontend's dev-only /review route, which proxies here."
        ),
    )
    parser.add_argument(
        "--bundle", type=Path, required=True,
        help="the review bundle directory (manifest.json, events.json, frames/)",
    )
    parser.add_argument(
        "--db", type=Path, default=None,
        help=f"annotation database (default: <bundle>/../{DEFAULT_DB_NAME})",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--export", choices=("parquet", "md", "csv"), default=None,
        help="write one export and exit, without serving",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.export:
        bundle = Bundle(args.bundle)
        store = ReviewStore(
            args.db or default_db_path(bundle.root),
            legacy_bundle_id=bundle.bundle_id,
        )
        try:
            path, rows = export(bundle, store, args.export)
        finally:
            store.close()
        print(f"{rows} row(s) → {path}")
        return 0

    server = build_server(args.bundle, args.db, args.port)
    stored, annotated = server.store.counts(server.bundle.bundle_id)
    print(f"Review server: http://{HOST}:{server.server_address[1]}/")
    print(f"  bundle:      {server.bundle.root} ({server.bundle.bundle_id})")
    print(f"  events:      {len(server.bundle.events)}")
    print(f"  annotations: {annotated} judged of {stored} stored")
    print(f"  database:    {server.store.db_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
        server.store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
