"""The live half of the gauge scoreboard (Phase F).

``scripts/replay_warnings.py`` puts a virtual subscriber at every DMI
rain gauge and replays the push decision rule over the frame archive.
This module does the same thing forward in time: once per cycle, after
the real push fan-out, it samples the same national grids at the same
station points, runs the same ``push.engine.evaluate``, and appends rows
of the same shape to the corpus. Replay output and live output therefore
concatenate into one table, and the day the replay ends is the day this
takes over.

Design constraints, in the order they bite:

- **Never cost a cycle.** Everything here is best-effort. A missing
  points file, an unwritable volume, a corrupt state file, a pyarrow that
  is not installed: each logs a warning and returns. The radar cycle has
  already written ``state.json`` by the time this runs, and nothing it
  does can be undone by a failure here.
- **All I/O off the loop.** ``after_cycle`` is awaited by the scheduler on
  the event loop; the sampling, the parquet rewrite and the state write
  all happen inside one ``asyncio.to_thread`` call.
- **One evaluation per radar observation.** The cycle fires every 5 min
  and fullRange composites land every ~10, so half the cycles re-emit the
  previous frame. Evaluating one twice would double-count a persistence
  streak — the same trap ``push.service`` documents, guarded the same way
  (last-evaluated radar timestamp, plus the engine's own idempotence on
  ``last_eval_radar_ts``).
- **Idempotent appends.** A month partition is rewritten atomically with
  the cycle's rows replacing any existing row for the same
  ``(radar_ts, station_id)``. Restarting the service, or replaying a
  frame, can add rows but can never duplicate one.
- **Private instance only.** ``server.public_mode`` refuses at config
  load (``Config._station_eval_is_private``); this module checks again
  before it does anything, because a guard that exists in one place is a
  guard that gets removed by a refactor.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import structlog

from dmi_nowcast_core.warning_score import (
    DECISION_COLUMNS,
    align_decision_table,
    decision_leads_in,
    decision_schema,
    decision_table,
    per_lead_columns,
)

from .config import Config
from .national_sample import sample_point
from .push.engine import INITIAL_STATE, Observation, Rules, SubState, evaluate

_log = structlog.get_logger(__name__)

#: Bump when the on-disk state file changes shape.
STATE_VERSION = 1


def stations_dir(config: Config) -> Path:
    """``<corpus_dir>/stations`` — shared with the gauge observation store."""
    corpus = config.storage.corpus_dir
    if corpus is None:
        raise ValueError("station_eval requires storage.corpus_dir")
    return Path(corpus) / "stations"


def state_path(config: Config) -> Path:
    return stations_dir(config) / "eval_state.json"


def partition_path(config: Config, instant: datetime) -> Path:
    return (
        stations_dir(config)
        / "eval"
        / f"{instant.year:04d}"
        / f"{instant.month:02d}.parquet"
    )


def load_points(path: Path) -> list[dict]:
    """Read the v2 station points file; raises on anything unexpected."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict) or int(raw.get("version", 0)) != 2:
        raise ValueError(f"{path}: expected a version-2 station points file")
    points = []
    for entry in raw.get("points", ()):
        points.append({
            "id": str(entry["id"]),
            "lat": float(entry["lat"]),
            "lon": float(entry["lon"]),
        })
    if not points:
        raise ValueError(f"{path}: no points")
    return points


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def state_to_json(state: SubState) -> dict:
    return {
        "armed": bool(state.armed),
        "streak": int(state.streak),
        "below_since_utc": (
            state.below_since_utc.isoformat() if state.below_since_utc else None
        ),
        "last_eval_radar_ts": (
            state.last_eval_radar_ts.isoformat()
            if state.last_eval_radar_ts else None
        ),
    }


def state_from_json(raw: Any) -> SubState:
    if not isinstance(raw, dict):
        return INITIAL_STATE
    return SubState(
        armed=bool(raw.get("armed", True)),
        streak=int(raw.get("streak", 0)),
        below_since_utc=_parse_iso(raw.get("below_since_utc")),
        last_eval_radar_ts=_parse_iso(raw.get("last_eval_radar_ts")),
    )


def _write_atomic(path: Path, write) -> None:
    """tmp + rename in the target directory, so a reader never sees a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def append_rows(path: Path, rows: Sequence[dict], leads_min=None) -> int:
    """Merge ``rows`` into a month partition, keyed on (radar_ts, station_id).

    Read-modify-write of one month rather than an append: parquet has no
    in-place append, the partition is small (a month of 10-min frames ×
    ~100 stations is ~430k rows) and a full rewrite is the only way to
    make the key idempotent. Existing rows for a key the cycle is writing
    are dropped, so re-running a frame corrects it instead of doubling it.

    The existing partition is aligned to the UNION of its own lead columns
    and this cycle's before the merge, so a month that was started before
    the ``p_rain_<lead>`` columns existed — or under a different
    ``national.leads_min`` — keeps every column it had and gains nulls for
    the rest, instead of failing the rewrite on a schema mismatch.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    new = decision_table(rows, leads_min)
    if path.is_file():
        try:
            existing = pq.read_table(path)
        except Exception as exc:  # noqa: BLE001 — a corrupt month is replaced
            _log.warning(
                "station_eval_partition_unreadable", path=str(path), error=str(exc),
            )
            existing = None
        if existing is not None and existing.num_rows:
            existing = align_decision_table(existing, leads_min)
            leads = decision_leads_in(existing)
            new = align_decision_table(new, leads)
            keys = {(r["radar_ts"], r["station_id"]) for r in new.to_pylist()}
            kept = [
                r for r in existing.to_pylist()
                if (r["radar_ts"], r["station_id"]) not in keys
            ]
            if kept:
                new = pa.concat_tables([decision_table(kept, leads), new])
    new = new.sort_by([("radar_ts", "ascending"), ("station_id", "ascending")])
    _write_atomic(path, lambda tmp: pq.write_table(new, tmp, compression="zstd"))
    return new.num_rows


class StationEvalService:
    """Owns the per-cycle gauge evaluation. One instance per process."""

    def __init__(self, config: Config, engine: Any) -> None:
        self.config = config
        self.engine = engine
        self._points: list[dict] | None = None
        self._states: dict[str, SubState] | None = None
        self._last_radar_ts: datetime | None = None
        self._last_summary: dict | None = None

    # -- introspection ------------------------------------------------------

    @property
    def last_summary(self) -> dict | None:
        return self._last_summary

    # -- the cycle hook -----------------------------------------------------

    async def after_cycle(self, result: Any) -> None:
        """Evaluate every station for one completed cycle. Never raises."""
        try:
            await self._after_cycle(result)
        except Exception as exc:  # noqa: BLE001 — the scoreboard is never
            # allowed to cost a radar cycle, a push, or the next poll.
            _log.warning("station_eval_failed", error=f"{type(exc).__name__}: {exc}")

    async def _after_cycle(self, result: Any) -> None:
        cfg = self.config.station_eval
        if not cfg.enabled:
            return
        if self.config.server.public_mode:
            # Config refuses this combination at load; checked again here so
            # a future wiring change cannot quietly start writing a corpus
            # on the internet-facing instance.
            _log.warning("station_eval_skipped", reason="public_mode")
            return
        state = getattr(result, "state", None)
        if state is None:
            return
        radar_ts = getattr(getattr(state, "radar", None), "latest_ts", None)
        if radar_ts is None:
            return
        if radar_ts.tzinfo is None:
            radar_ts = radar_ts.replace(tzinfo=timezone.utc)
        if self._last_radar_ts is not None and radar_ts <= self._last_radar_ts:
            return  # the no-new-frame fast path, or a re-emitted state

        latest = self.engine.national_latest
        geo = self.engine.geo
        if latest is None or geo is None:
            _log.info("station_eval_skipped", reason="no_national_products")
            return
        products, products_ts = latest
        if products_ts is not None and products_ts.tzinfo is None:
            products_ts = products_ts.replace(tzinfo=timezone.utc)
        if products_ts != radar_ts:
            # Same trap the push service guards: attributing one frame's
            # grids to another frame's timestamp, and then hiding the real
            # frame by advancing the marker.
            _log.info(
                "station_eval_skipped",
                reason="products_radar_ts_mismatch",
                products_ts=products_ts.isoformat() if products_ts else None,
                radar_ts=radar_ts.isoformat(),
            )
            return

        generated_at = (
            getattr(latest, "generated_at_utc", None)
            or datetime.now(timezone.utc)
        )
        summary = await asyncio.to_thread(
            self._evaluate_and_append,
            products,
            geo,
            radar_ts,
            generated_at,
            getattr(latest, "observed_mm_h", None),
            getattr(latest, "forecast_mm_h", None),
        )
        if summary is not None:
            self._last_radar_ts = radar_ts
            self._last_summary = summary

    # -- the work (runs in a worker thread) ---------------------------------

    def _rules(self) -> Rules:
        rules = self.config.station_eval.rules
        return Rules(
            persistence_obs=rules.persistence_obs,
            rearm_after_min=rules.rearm_after_min,
            # One detection threshold for the whole pipeline, exactly as
            # the push service does it.
            raining_now_mm_h=self.config.forecast.rain_threshold_mm_h,
        )

    def _ensure_loaded(self) -> None:
        if self._points is None:
            self._points = load_points(Path(self.config.station_eval.points_file))
        if self._states is None:
            self._states = self._read_state()

    def _read_state(self) -> dict[str, SubState]:
        path = state_path(self.config)
        if not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001 — a corrupt file restarts armed
            _log.warning("station_eval_state_unreadable", error=str(exc))
            return {}
        stations = raw.get("stations") if isinstance(raw, dict) else None
        if not isinstance(stations, dict):
            return {}
        return {str(k): state_from_json(v) for k, v in stations.items()}

    def _write_state(self, states: dict[str, SubState], generated_at: datetime) -> None:
        payload = {
            "version": STATE_VERSION,
            "updated_utc": generated_at.isoformat(),
            "stations": {sid: state_to_json(s) for sid, s in sorted(states.items())},
        }
        _write_atomic(
            state_path(self.config),
            lambda tmp: Path(tmp).write_text(json.dumps(payload, indent=2)),
        )

    def _evaluate_and_append(
        self,
        products: Any,
        geo: Any,
        radar_ts: datetime,
        generated_at: datetime,
        observed_mm_h: Any = None,
        forecast_mm_h: Any = None,
    ) -> dict | None:
        """Sample, decide, persist. Blocking; returns None when it did nothing."""
        self._ensure_loaded()
        assert self._points is not None and self._states is not None
        rules = self._rules()
        lead = int(self.config.station_eval.rules.lead_min)
        threshold_pct = int(self.config.station_eval.rules.threshold_pct)

        rows: list[dict] = []
        actions: dict[str, int] = {}
        errors = 0
        for point in self._points:
            station = point["id"]
            try:
                sample = sample_point(
                    products, geo, point["lat"], point["lon"],
                    observed_mm_h=observed_mm_h,
                    forecast_mm_h=forecast_mm_h,
                )
                series = sample.forecast_mm_h if sample else None
                obs = Observation(
                    radar_ts_utc=radar_ts,
                    p_rain=sample.p_rain.get(lead) if sample else None,
                    eta_min=sample.eta_min if sample else None,
                    intensity_mm_h=sample.intensity_mm_h if sample else None,
                    observed_mm_h=sample.observed_mm_h if sample else None,
                    forecast_now_mm_h=series.get(0) if series else None,
                )
                decision = evaluate(
                    self._states.get(station, INITIAL_STATE),
                    obs,
                    threshold_pct=threshold_pct,
                    quiet=None,
                    tz="UTC",
                    now_utc=generated_at,
                    rules=rules,
                )
            except Exception as exc:  # noqa: BLE001 — one bad station only
                errors += 1
                _log.warning(
                    "station_eval_error", station=station, error=str(exc),
                )
                continue
            self._states[station] = decision.state
            actions[decision.action] = actions.get(decision.action, 0) + 1
            rows.append({
                "radar_ts": radar_ts,
                "generated_at": generated_at,
                "station_id": station,
                # The rule's lead is what the decision was taken on; every
                # served lead rides along so the offline threshold sweep
                # never has to re-run STEPS.
                "p_rain": obs.p_rain,
                **per_lead_columns(sample.p_rain if sample else None),
                "eta_min": obs.eta_min,
                "intensity_mm_h": obs.intensity_mm_h,
                "observed_mm_h": obs.observed_mm_h,
                "forecast_now_mm_h": obs.forecast_now_mm_h,
                "action": decision.action,
                "armed_after": decision.state.armed,
                "streak_after": decision.state.streak,
            })
        if not rows:
            _log.info("station_eval_empty", radar_ts=radar_ts.isoformat())
            return None

        # State first, rows second: a crash between the two costs one
        # cycle's rows, never a double-counted streak.
        self._write_state(self._states, generated_at)
        n_rows = append_rows(
            partition_path(self.config, radar_ts), rows,
            getattr(products, "leads_min", None),
        )
        summary = {
            "radar_ts": radar_ts.isoformat(),
            "stations": len(rows),
            "eval_errors": errors,
            "actions": actions,
            "partition_rows": n_rows,
        }
        _log.info("station_eval", **summary)
        return summary


__all__ = [
    "DECISION_COLUMNS",
    "STATE_VERSION",
    "StationEvalService",
    "append_rows",
    "load_points",
    "partition_path",
    "state_from_json",
    "state_path",
    "state_to_json",
    "stations_dir",
]
