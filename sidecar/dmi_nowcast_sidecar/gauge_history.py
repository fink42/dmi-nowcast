"""The gauge history one cycle reads, for the ``g_*`` features (v2, F1).

The post-processing model's radar features all come off grids the cycle
already holds. The gauge block does not: it is what the *station itself*
measured in the last hour, and the only place that exists is the corpus's
observation archive — the same parquet the gauge poller
(:mod:`dmi_nowcast_sidecar.station_obs`) writes every ten minutes and the
offline replay trains on.

Three things make this narrow on purpose.

**One query per cycle, never one per point.** ~110 points, every radar
frame, inside the cycle worker. The read is a single pushdown scan
(:meth:`~dmi_nowcast_core.station_store.StationObsStore.read_recent`) over
one six-hour window for every gauge at once — the time predicate prunes
row groups by statistics, so a month partition of ~1.4M rows costs the
few thousand that are in the window rather than a full decode.

**A point is a place, not a person.** The cycle asks for coordinates and
is told coordinates (``compute.CycleEngine.add_point_source``). This maps
a coordinate back to a **gauge station id** — and only to a gauge station
id, from the public station points file the scoreboard already reads. A
subscription's point resolves to nothing, gets a null gauge block and a
``g_known`` of 0, which is exactly what it is: a place with no gauge.

**The availability rule lives in the core.** This module reads slots; it
never decides which of them the model may see. That is
:func:`~dmi_nowcast_core.postprocess.station_gauge_features`, the same
function the replay calls, applied to the same
``(slot_end, wet, mm)`` triples
:func:`~dmi_nowcast_core.warning_score.gauge_slot_amounts` produces. If
the two ever diverged, a model fitted offline would be scored on rows
that mean something else — and nothing would fail.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import structlog

from dmi_nowcast_core.postprocess import (
    DEFAULT_GAUGE_LAG_MIN,
    GAUGE_SINCE_CAP_MIN,
)
from dmi_nowcast_core.station_store import StationObsStore
from dmi_nowcast_core.warning_score import (
    PRECIP_DUR_PARAM,
    PRECIP_PARAM,
    SLOT_MIN,
    gauge_slot_amounts,
)

from .push.postprocess import point_key

_log = structlog.get_logger(__name__)

__all__ = ["GaugeHistory", "build_gauge_history", "slots_by_station"]


def slots_by_station(
    table: Any,
    station_ids: Sequence[str],
    *,
    start_utc: datetime,
    end_utc: datetime,
) -> dict[str, list]:
    """``{station_id: [(slot_end, wet, mm), ...]}`` from one read.

    :func:`~dmi_nowcast_core.warning_score.gauge_slot_amounts` scans the
    rows it is given once per station, which is the right shape when a
    caller wants one station and the wrong one at a hundred: the table is
    converted and walked N times for an answer that needs one pass. This
    buckets the rows by station first and hands each call only its own,
    which is what lets the live cycle do this every radar frame and the
    replay do it for a whole day at once.

    The slot semantics are entirely ``gauge_slot_amounts``' — the wet
    rule, the contiguous grid, the trace sentinel. This only decides who
    sees which rows.
    """
    wanted = [str(sid) for sid in station_ids]
    buckets: dict[str, list] = {sid: [] for sid in wanted}
    rows = table.to_pylist() if hasattr(table, "to_pylist") else table
    for row in rows:
        bucket = buckets.get(str(row.get("station_id")))
        if bucket is not None:
            bucket.append(row)
    return {
        sid: gauge_slot_amounts(
            buckets[sid], sid, start_utc=start_utc, end_utc=end_utc,
        )
        for sid in wanted
    }


class GaugeHistory:
    """Gauge slots for the cycle's points, read once per cycle.

    ``points_file`` is the version-2 station points file
    (``station_eval.points_file``): the public catalogue of DMI gauges,
    the very list the scoreboard evaluates. It is read once, on first use,
    inside the cycle worker where blocking is allowed.
    """

    def __init__(
        self,
        corpus_dir: Path | str,
        points_file: Path | str,
        *,
        lag_min: float,
    ) -> None:
        self.store = StationObsStore(Path(corpus_dir))
        self.points_file = Path(points_file)
        self.lag_min = float(lag_min)
        self._stations: dict[tuple[float, float], str] | None = None

    # -- the coordinate → gauge map ----------------------------------------

    def stations(self) -> dict[tuple[float, float], str]:
        """``{rounded (lat, lon): station_id}``, read once. ``{}`` on failure.

        An unreadable points file costs the gauge block and nothing else —
        the same rule the rest of the feature path follows.
        """
        if self._stations is None:
            try:
                raw = json.loads(self.points_file.read_text())
                self._stations = {
                    point_key(float(entry["lat"]), float(entry["lon"])): str(
                        entry["id"],
                    )
                    for entry in (raw.get("points") or ())
                }
            except Exception as exc:  # noqa: BLE001 — every way a file is junk
                _log.warning(
                    "gauge_history_points_unreadable",
                    path=str(self.points_file),
                    error=f"{type(exc).__name__}: {exc}",
                )
                self._stations = {}
        return self._stations

    # -- the read ----------------------------------------------------------

    def window(self, now_utc: datetime) -> tuple[datetime, datetime]:
        """The ``[start, end]`` one cycle asks the store for.

        It ends at the visibility horizon rather than at *now*: rows the
        features may not look at are not worth decoding. It starts far
        enough back that ``g_min_since_wet`` can reach its six-hour cap
        plus one slot of slack.
        """
        end = now_utc - timedelta(minutes=self.lag_min)
        return end - timedelta(minutes=GAUGE_SINCE_CAP_MIN + float(SLOT_MIN)), end

    def slots_for(
        self, keys: Sequence[tuple[float, float]], *, now_utc: datetime,
    ) -> list[Any]:
        """One entry per point, in ``keys`` order: its slots, or ``None``.

        ``None`` means "no gauge here", which the core turns into a null
        gauge block with ``g_known = 0``. The store is read exactly once,
        for every gauge among the points together.
        """
        stations = self.stations()
        ids = [stations.get(point_key(lat, lon)) for lat, lon in keys]
        wanted = sorted({sid for sid in ids if sid})
        if not wanted:
            return [None] * len(keys)
        start, end = self.window(now_utc)
        table = self.store.read_recent(
            start, end, [PRECIP_PARAM, PRECIP_DUR_PARAM], wanted,
        )
        by_station = slots_by_station(
            table, wanted, start_utc=start, end_utc=end,
        )
        return [None if sid is None else by_station.get(sid) for sid in ids]


def build_gauge_history(config: Any) -> GaugeHistory | None:
    """A :class:`GaugeHistory` for this config, or None when there can be none.

    None — and therefore a null gauge block — whenever the deployment has
    no gauge archive (the public instance owns no corpus volume), no
    station catalogue to resolve a point against, or the feature turned
    off. All three are ordinary states, not errors.
    """
    settings = getattr(config, "postprocess", None)
    if settings is not None and not settings.gauge_features:
        return None
    corpus_dir = config.storage.corpus_dir
    points_file = config.station_eval.points_file
    if corpus_dir is None or points_file is None:
        return None
    return GaugeHistory(
        corpus_dir,
        points_file,
        lag_min=(
            DEFAULT_GAUGE_LAG_MIN if settings is None
            else float(settings.gauge_lag_min)
        ),
    )
