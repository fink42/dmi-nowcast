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
  all happen inside one worker call — on this step's own dedicated
  thread, not the shared executor, because a month-partition rewrite has
  a working set worth hundreds of megabytes and glibc charges that to
  every thread it has ever run on (see
  :mod:`dmi_nowcast_sidecar.workers`).
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

Since Phase H each row also carries the post-processing feature columns
and the model's own ``p_post_<lead>``, written through the same schema
the replay writes (``postprocess.feature_schema`` /
``postprocess.post_schema``). That is what puts a live row on the same
footing as a replay row for the nightly refit: the model is fitted on
both, and the threshold sweep replays the rule on the probability the
engine actually used. They are additive columns —
``align_decision_table`` conforms a file to the shared schema and drops
what is not in it — so every existing reader sees exactly what it always
saw.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import structlog

from dmi_nowcast_core import postprocess as core_postprocess
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
from .workers import release_arrow_pool, run_in_pool

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


#: The idempotency key of a decision row: one evaluation per station per
#: radar frame. Named once because the merge below joins on it and the
#: module docstring promises it.
MERGE_KEY: tuple[str, str] = ("radar_ts", "station_id")


def extra_schema(leads_min=None):
    """The post-processing columns a decision row carries beside the shared ones.

    The feature columns and ``p_post_<lead>``, in that order, from the
    core module the replay also writes through — so a live partition and a
    replay partition have the same columns with the same types and the
    nightly refit sees one table.

    Lead-bearing names (``raw_frac_<lead>``, ``p_post_<lead>``) follow the
    same lead set the shared ``p_rain_<lead>`` columns do, read back
    through ``decision_leads_in`` rather than restated, so a config that
    serves different leads cannot end up with the two disagreeing.
    """
    import pyarrow as pa

    leads = decision_leads_in(decision_schema(leads_min).names)
    return pa.schema(
        list(core_postprocess.feature_schema(leads))
        + list(core_postprocess.post_schema(leads))
    )


def _with_extras(table, rows: Sequence[dict] | None, schema) -> Any:
    """Append every field of ``schema``, from ``rows`` or from ``table``.

    Two callers, one rule about what "missing" means. Building this
    cycle's table the values come from the row dicts; conforming a month
    partition written before a column existed they come from the file, and
    a column the file does not have becomes nulls — never zeros, which
    would claim the feature was computed and came out dry.
    """
    import pyarrow as pa

    present = set(table.schema.names)
    for field_ in schema:
        if rows is not None:
            values = pa.array(
                [row.get(field_.name) for row in rows], type=field_.type,
            )
        elif field_.name in present:
            values = table.column(field_.name).cast(field_.type)
        else:
            values = pa.nulls(table.num_rows, field_.type)
        if field_.name in present:
            table = table.drop_columns([field_.name])
        table = table.append_column(field_, values)
    return table


def _conform(table, leads, schema) -> Any:
    """A table in the shared decision schema for ``leads``, plus ``schema``.

    ``align_decision_table`` deliberately drops everything it does not
    know, which is what keeps every other reader unaffected by these
    columns — and is exactly why a merge has to put them back, or the
    first rewrite of a month would silently delete the features in it.
    """
    return _with_extras(align_decision_table(table, leads), None, schema)


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

    The drop is an Arrow **left anti join**, not a Python set difference.
    The obvious spelling — ``existing.to_pylist()``, a set of key tuples,
    a list comprehension, ``decision_table`` to rebuild — materialises the
    whole month as row dicts, twelve datetimes and floats each, every ten
    minutes for the life of the month. Measured on the real partition
    shape at end-of-month size (430k rows): 3.6 s of blocking work and a
    449 MB peak of Python objects, against 0.2 s and 9 MB for the join.
    Same rows, same order, same file — see
    ``test_append_rows_merges_without_materialising_python_rows``.

    ``release_arrow_pool`` closes the loop the same way the gauge poller
    does: the pool otherwise keeps the largest month it ever built. It runs
    once the merge's frame is gone, so the tables it is asked about really
    are unreachable.
    """
    try:
        return _merge_and_write(path, rows, leads_min)
    finally:
        release_arrow_pool()


def _merge_and_write(path: Path, rows: Sequence[dict], leads_min) -> int:
    """The read-merge-sort-write half of :func:`append_rows`."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    merged = _with_extras(
        decision_table(rows, leads_min), rows, extra_schema(leads_min),
    )
    if path.is_file():
        try:
            existing = pq.read_table(path)
        except Exception as exc:  # noqa: BLE001 — a corrupt month is replaced
            _log.warning(
                "station_eval_partition_unreadable", path=str(path), error=str(exc),
            )
            existing = None
        if existing is not None and existing.num_rows:
            leads = decision_leads_in(
                decision_schema(leads_min).names,
            ) or ()
            leads = tuple(sorted(set(leads) | set(decision_leads_in(existing))))
            schema = extra_schema(leads)
            existing = _conform(existing, leads, schema)
            merged = _conform(merged, leads, schema)
            kept = existing.join(
                merged.select(list(MERGE_KEY)),
                keys=list(MERGE_KEY),
                join_type="left anti",
            )
            if kept.num_rows:
                # ``join`` is free to reorder columns; select back to the
                # aligned schema so ``concat_tables`` is never handed two
                # tables that merely happen to share a column set.
                merged = pa.concat_tables(
                    [kept.select(merged.schema.names), merged],
                )
    merged = merged.sort_by(
        [("radar_ts", "ascending"), ("station_id", "ascending")],
    )
    _write_atomic(path, lambda tmp: pq.write_table(merged, tmp, compression="zstd"))
    return merged.num_rows


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

    # -- what the cycle needs from us ---------------------------------------

    def decision_points(self) -> list[tuple[float, float]]:
        """Every gauge station's point, for the cycle's feature table (H-P).

        Registered with the engine by ``app.create_app`` and called once
        per full cycle, inside the cycle worker, so reading the points
        file here is allowed to block. An unreadable file raises and the
        engine's guard logs it: the scoreboard then writes rows without
        features rather than costing the cycle, which is this module's
        rule everywhere else too.
        """
        self._ensure_points()
        return [
            (float(point["lat"]), float(point["lon"]))
            for point in (self._points or ())
        ]

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
        summary = await run_in_pool(
            "station_eval",
            self._evaluate_and_append,
            products,
            geo,
            radar_ts,
            generated_at,
            getattr(latest, "observed_mm_h", None),
            getattr(latest, "forecast_mm_h", None),
            # The cycle's post-processing answer, checked against this
            # frame the same way the products are: one frame's features
            # must never be stamped with another frame's timestamp.
            self._postprocess_for(radar_ts),
        )
        if summary is not None:
            self._last_radar_ts = radar_ts
            self._last_summary = summary

    def _postprocess_for(self, radar_ts: datetime) -> Any:
        """The cycle's ``CyclePostprocess`` for this frame, or None."""
        latest = getattr(self.engine, "postprocess_latest", None)
        if latest is None:
            return None
        stamp = getattr(latest, "radar_ts_utc", None)
        if stamp is not None and stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return latest if stamp == radar_ts else None

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

    def _ensure_points(self) -> None:
        if self._points is None:
            self._points = load_points(Path(self.config.station_eval.points_file))

    def _ensure_loaded(self) -> None:
        self._ensure_points()
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
        postprocess: Any = None,
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
                # The feature columns and ``p_post_<lead>`` for this
                # station this cycle — computed once by the cycle for
                # every point it serves, read here by coordinate.
                extras = (
                    {} if postprocess is None
                    else postprocess.columns(point["lat"], point["lon"])
                )
                obs = Observation(
                    radar_ts_utc=radar_ts,
                    p_rain=sample.p_rain.get(lead) if sample else None,
                    eta_min=sample.eta_min if sample else None,
                    intensity_mm_h=sample.intensity_mm_h if sample else None,
                    observed_mm_h=sample.observed_mm_h if sample else None,
                    forecast_now_mm_h=series.get(0) if series else None,
                    # The scoreboard measures the rule the SERVICE runs,
                    # so it decides on the same probability the push
                    # engine does, under the same per-row fallback.
                    p_post=extras.get(core_postprocess.post_column(lead)),
                    p_source=self.config.push.probability_source,
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
                # Additive (Phase H): the features the nightly refit
                # trains on and the probability the decision above was
                # actually taken on. Unknown to every existing reader,
                # which is the point — ``align_decision_table`` drops
                # them and the sweep asks for them by name.
                **extras,
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
    "MERGE_KEY",
    "STATE_VERSION",
    "StationEvalService",
    "append_rows",
    "extra_schema",
    "load_points",
    "partition_path",
    "state_from_json",
    "state_path",
    "state_to_json",
    "stations_dir",
]
