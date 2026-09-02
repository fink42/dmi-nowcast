"""SQLite store for browser push subscriptions.

One table, one row per browser endpoint. The columns are exactly what the
service needs and nothing more — no email, no name, no IP, no user agent.
The only personal datum is the coordinate the subscriber asked to be
warned about, which is the whole point of the feature.

Two halves per row:

- **preferences** (``lat``/``lon``, ``threshold_pct``, ``lead_min``, quiet
  hours, ``tz``, ``lang``) — what the subscriber chose;
- **state machine** (``armed``, ``streak``, ``below_since_utc``,
  ``last_eval_radar_ts``, ``last_notified_utc``) — what
  ``push.engine.evaluate`` carries between radar observations.

Editing preferences restarts the state machine (see :meth:`PushStore.upsert`):
a subscriber who lowers their threshold expects the new setting to be
evaluated from scratch, not against a streak accumulated under the old one.

Datetimes are stored as ISO-8601 UTC strings and always come back
timezone-aware. Every access goes through one connection behind a lock, so
the event-loop thread (routes) and ``asyncio.to_thread`` workers (the
per-cycle evaluation) can use the same store safely.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS subscriptions (
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

_COLUMNS: Final = (
    "endpoint, p256dh, auth, lat, lon, threshold_pct, lead_min, "
    "quiet_enabled, quiet_start, quiet_end, tz, lang, created_utc, "
    "last_notified_utc, armed, streak, below_since_utc, last_eval_radar_ts"
)


class _Unchanged:
    """Sentinel for :meth:`PushStore.update_state`'s optional column.

    ``None`` is a meaningful value for ``last_notified_utc`` (never
    notified), so "leave it alone" needs its own marker.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unchanged>"


UNCHANGED: Final = _Unchanged()


@dataclass(frozen=True)
class NewSubscription:
    """A subscribe request, validated, ready to be written."""

    endpoint: str
    p256dh: str
    auth: str
    lat: float
    lon: float
    threshold_pct: int
    lead_min: int
    quiet_enabled: bool
    quiet_start: str
    quiet_end: str
    tz: str
    lang: str


@dataclass(frozen=True)
class Subscription:
    """One stored row: preferences plus the decision state machine."""

    endpoint: str
    p256dh: str
    auth: str
    lat: float
    lon: float
    threshold_pct: int
    lead_min: int
    quiet_enabled: bool
    quiet_start: str
    quiet_end: str
    tz: str
    lang: str
    created_utc: datetime
    last_notified_utc: datetime | None
    armed: bool
    streak: int
    below_since_utc: datetime | None
    last_eval_radar_ts: datetime | None


def sub_id(endpoint: str) -> str:
    """Short stable handle for logs. Never log the endpoint itself: it is
    a bearer capability to notify that browser."""
    return hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:10]


def _to_iso(value: datetime | None) -> str | None:
    """Timezone-aware datetime → ISO-8601 UTC string (naive assumed UTC)."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _from_iso(value: str | None) -> datetime | None:
    """Stored string → aware UTC datetime. Accepts ``Z`` and ``+00:00``."""
    if value is None:
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _row_to_subscription(row: sqlite3.Row) -> Subscription:
    return Subscription(
        endpoint=row["endpoint"],
        p256dh=row["p256dh"],
        auth=row["auth"],
        lat=float(row["lat"]),
        lon=float(row["lon"]),
        threshold_pct=int(row["threshold_pct"]),
        lead_min=int(row["lead_min"]),
        quiet_enabled=bool(row["quiet_enabled"]),
        quiet_start=row["quiet_start"],
        quiet_end=row["quiet_end"],
        tz=row["tz"],
        lang=row["lang"],
        created_utc=_from_iso(row["created_utc"]),  # type: ignore[arg-type]
        last_notified_utc=_from_iso(row["last_notified_utc"]),
        armed=bool(row["armed"]),
        streak=int(row["streak"]),
        below_since_utc=_from_iso(row["below_since_utc"]),
        last_eval_radar_ts=_from_iso(row["last_eval_radar_ts"]),
    )


class PushStore:
    """Thread-safe SQLite subscription store."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        # 0700: the directory holds the VAPID private key too (same parent
        # by default), and nothing else on the host has business reading it.
        self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            # WAL: a reader (a route) never blocks the per-cycle writer.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- writes ------------------------------------------------------------

    def upsert(self, sub: NewSubscription, *, now_utc: datetime | None = None) -> bool:
        """Create or update one subscription. True when it was created.

        An update rewrites the preferences and **restarts the state
        machine** (``armed=1, streak=0``, both timestamps cleared):
        continuing a streak accumulated under different preferences would
        fire against a rule the subscriber no longer has. ``created_utc``
        and ``last_notified_utc`` survive — the first is history, the
        second is the anti-spam floor and must not be resettable by
        re-subscribing.
        """
        now = now_utc or datetime.now(timezone.utc)
        with self._lock:
            cur = self._conn.execute(
                "SELECT endpoint FROM subscriptions WHERE endpoint = ?",
                (sub.endpoint,),
            )
            exists = cur.fetchone() is not None
            if exists:
                self._conn.execute(
                    """
                    UPDATE subscriptions SET
                        p256dh = ?, auth = ?, lat = ?, lon = ?,
                        threshold_pct = ?, lead_min = ?,
                        quiet_enabled = ?, quiet_start = ?, quiet_end = ?,
                        tz = ?, lang = ?,
                        armed = 1, streak = 0,
                        below_since_utc = NULL, last_eval_radar_ts = NULL
                    WHERE endpoint = ?
                    """,
                    (
                        sub.p256dh, sub.auth, sub.lat, sub.lon,
                        sub.threshold_pct, sub.lead_min,
                        int(sub.quiet_enabled), sub.quiet_start, sub.quiet_end,
                        sub.tz, sub.lang,
                        sub.endpoint,
                    ),
                )
            else:
                self._conn.execute(
                    f"INSERT INTO subscriptions ({_COLUMNS}) VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        sub.endpoint, sub.p256dh, sub.auth, sub.lat, sub.lon,
                        sub.threshold_pct, sub.lead_min,
                        int(sub.quiet_enabled), sub.quiet_start, sub.quiet_end,
                        sub.tz, sub.lang, _to_iso(now),
                        None, 1, 0, None, None,
                    ),
                )
            self._conn.commit()
        return not exists

    def delete(self, endpoint: str) -> bool:
        """Remove one subscription. False when there was nothing to remove."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM subscriptions WHERE endpoint = ?", (endpoint,),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def update_state(
        self,
        endpoint: str,
        *,
        armed: bool,
        streak: int,
        below_since_utc: datetime | None,
        last_eval_radar_ts: datetime | None,
        last_notified_utc: datetime | None | _Unchanged = UNCHANGED,
    ) -> None:
        """Persist the decision state machine for one subscription.

        ``last_notified_utc`` defaults to :data:`UNCHANGED` — only a cycle
        that actually notified passes it.
        """
        sets = [
            "armed = ?", "streak = ?",
            "below_since_utc = ?", "last_eval_radar_ts = ?",
        ]
        params: list[Any] = [
            int(bool(armed)), int(streak),
            _to_iso(below_since_utc), _to_iso(last_eval_radar_ts),
        ]
        if not isinstance(last_notified_utc, _Unchanged):
            sets.append("last_notified_utc = ?")
            params.append(_to_iso(last_notified_utc))
        params.append(endpoint)
        with self._lock:
            self._conn.execute(
                f"UPDATE subscriptions SET {', '.join(sets)} WHERE endpoint = ?",
                params,
            )
            self._conn.commit()

    # -- reads -------------------------------------------------------------

    def get(self, endpoint: str) -> Subscription | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_COLUMNS} FROM subscriptions WHERE endpoint = ?",
                (endpoint,),
            ).fetchone()
        return _row_to_subscription(row) if row is not None else None

    def list(self) -> list[Subscription]:
        """Every subscription, oldest first (stable fan-out order)."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_COLUMNS} FROM subscriptions "
                "ORDER BY created_utc ASC, endpoint ASC",
            ).fetchall()
        return [_row_to_subscription(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM subscriptions",
            ).fetchone()
        return int(row["n"])

    def stats(self) -> dict:
        """Counts for ``/api/push/stats`` — never endpoints, never points."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n, "
                "COALESCE(SUM(armed), 0) AS armed FROM subscriptions",
            ).fetchone()
        return {"subscriptions": int(row["n"]), "armed": int(row["armed"])}


__all__ = [
    "UNCHANGED",
    "NewSubscription",
    "PushStore",
    "Subscription",
    "sub_id",
]
