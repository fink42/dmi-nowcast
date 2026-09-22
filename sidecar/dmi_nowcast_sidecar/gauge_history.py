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

**The whole catalogue, not the cycle's points** (v2, S1). The ``g_*``
block only ever needed the gauge standing ON a point, so the first
version of this module read the stations among the cycle's points and
nothing else. The ``ng_*`` block —
:func:`~dmi_nowcast_core.postprocess.neighbour_gauge_features` — asks the
opposite question, what the gauges AROUND a point measured, and its
answer at a subscriber's address is made of stations that are not served
points at all. So one cycle now takes ONE read over every station in the
catalogue (:meth:`GaugeHistory.read`) and both blocks are served from it:
the per-point series for ``g_*``, the digested
:class:`~dmi_nowcast_core.postprocess.GaugeSlotTable` for ``ng_*``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import structlog

from dmi_nowcast_core.postprocess import (
    DEFAULT_GAUGE_LAG_MIN,
    GAUGE_SINCE_CAP_MIN,
    GaugeSlotTable,
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

__all__ = [
    "GaugeCycleRead",
    "GaugeHistory",
    "build_gauge_history",
    "slots_by_station",
]


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


@dataclass(frozen=True)
class GaugeCycleRead:
    """One cycle's gauge archive: read once, over every catalogue station.

    Two feature blocks eat this object and they want the same rows in two
    shapes. ``g_*`` wants one series per served point, in point order
    (:meth:`series_for`); ``ng_*`` wants every station at once, digested
    into numpy (:attr:`table`) together with the coordinates that turn a
    station id into a place (:attr:`coords`). Assembling both from ONE
    store read is the whole reason this type exists: the alternative was
    two reads per cycle for one window of one archive.

    ``by_station`` carries a key for every station in the catalogue,
    including the ones that reported nothing in the window — an empty
    series, which is a different statement from "not a gauge" and is what
    keeps a dark gauge out of ``ng_count_20km`` while leaving it in
    ``ng_near_km``'s geometry.
    """

    #: The decision instant the window was cut for.
    now_utc: datetime
    #: The window actually asked of the store, ``[start, end]``.
    start_utc: datetime
    end_utc: datetime
    #: ``{station_id: [(slot_end, wet, mm), ...]}`` for the whole catalogue.
    by_station: dict[str, list]
    #: ``{station_id: (lat, lon)}`` — the catalogue, unrounded.
    coords: dict[str, tuple[float, float]]
    #: ``{rounded (lat, lon): station_id}``, for the point → gauge lookup.
    stations: dict[tuple[float, float], str] = field(default_factory=dict)
    #: ``by_station`` digested once. Shared by every point of the cycle.
    table: GaugeSlotTable = field(
        default_factory=lambda: GaugeSlotTable.from_slots({}),
    )

    def station_ids(
        self, keys: Sequence[tuple[float, float]],
    ) -> list[str | None]:
        """Each point's OWN station id, or None — ``exclude_self``' input.

        One entry per point, in ``keys`` order, which is exactly what
        :func:`~dmi_nowcast_core.postprocess.neighbour_gauge_features`
        requires: leave-self-out is by name where the point is a gauge and
        by distance (``NG_SELF_KM``) where it is not.
        """
        return [self.stations.get(point_key(lat, lon)) for lat, lon in keys]

    def series_for(
        self, keys: Sequence[tuple[float, float]],
    ) -> list[Any]:
        """One entry per point, in ``keys`` order: its slots, or ``None``.

        ``None`` means "no gauge here", which the core turns into a null
        gauge block with ``g_known = 0``.
        """
        return [
            None if sid is None else self.by_station.get(sid)
            for sid in self.station_ids(keys)
        ]


class GaugeHistory:
    """The gauge archive one cycle reads, once, for both gauge blocks.

    ``points_file`` is the version-2 station points file
    (``station_eval.points_file``): the public catalogue of DMI gauges,
    the very list the scoreboard evaluates. It is read once, on first use,
    inside the cycle worker where blocking is allowed, and it answers two
    questions — which station stands on a given point (:meth:`stations`)
    and where a given station stands (:meth:`coords`).
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
        self._coords: dict[str, tuple[float, float]] | None = None

    # -- the coordinate → gauge map ----------------------------------------

    def _load_points(self) -> None:
        """Parse the catalogue once into BOTH directions of the map.

        One file, one read, one failure log line: the point → station
        lookup the ``g_*`` block needs and the station → coordinate map the
        ``ng_*`` geometry needs are the same twelve kilobytes of JSON, and
        parsing it twice would be two chances to disagree about which
        stations exist.
        """
        if self._stations is not None and self._coords is not None:
            return
        try:
            raw = json.loads(self.points_file.read_text())
            entries = list(raw.get("points") or ())
            self._stations = {
                point_key(float(entry["lat"]), float(entry["lon"])): str(
                    entry["id"],
                )
                for entry in entries
            }
            self._coords = {
                str(entry["id"]): (float(entry["lat"]), float(entry["lon"]))
                for entry in entries
            }
        except Exception as exc:  # noqa: BLE001 — every way a file is junk
            _log.warning(
                "gauge_history_points_unreadable",
                path=str(self.points_file),
                error=f"{type(exc).__name__}: {exc}",
            )
            self._stations = {}
            self._coords = {}

    def stations(self) -> dict[tuple[float, float], str]:
        """``{rounded (lat, lon): station_id}``, read once. ``{}`` on failure.

        An unreadable points file costs the gauge block and nothing else —
        the same rule the rest of the feature path follows.
        """
        self._load_points()
        return self._stations or {}

    def coords(self) -> dict[str, tuple[float, float]]:
        """``{station_id: (lat, lon)}``, read once. ``{}`` on failure.

        The other direction of :meth:`stations`, and UNrounded: these are
        the numbers the neighbour geometry measures kilometres with, and
        the offline builder reads the very same field of the very same
        file (``add_neighbour_gauge_features.load_station_coords``), so a
        served ``ng_*`` value and a trained one are computed from one
        coordinate.
        """
        self._load_points()
        return self._coords or {}

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

    def read(self, now_utc: datetime) -> GaugeCycleRead:
        """The cycle's ONE store read, over every station in the catalogue.

        Not only the stations among the cycle's points: ``ng_*`` asks what
        the gauges AROUND a point measured, and at an address that is not
        itself a gauge every one of those is a station the cycle would
        otherwise never have read. The catalogue is ~110 stations against
        the ~110 points a cycle serves, so this is the same order of rows
        the per-point read already cost — one window, one pushdown scan.

        Total, like everything else on this path: an unreadable catalogue
        yields an empty read (and one log line from :meth:`coords`), which
        gives a null gauge block and a null neighbour block rather than an
        exception inside the cycle worker.
        """
        stations = self.stations()
        coords = self.coords()
        wanted = sorted(set(coords) | {sid for sid in stations.values() if sid})
        start, end = self.window(now_utc)
        by_station: dict[str, list] = {}
        if wanted:
            table = self.store.read_recent(
                start, end, [PRECIP_PARAM, PRECIP_DUR_PARAM], wanted,
            )
            by_station = slots_by_station(
                table, wanted, start_utc=start, end_utc=end,
            )
        return GaugeCycleRead(
            now_utc=now_utc,
            start_utc=start,
            end_utc=end,
            by_station=by_station,
            coords=coords,
            stations=stations,
            table=GaugeSlotTable.from_slots(by_station),
        )

    def slots_for(
        self, keys: Sequence[tuple[float, float]], *, now_utc: datetime,
    ) -> list[Any]:
        """One entry per point, in ``keys`` order: its slots, or ``None``.

        The ``g_*`` half of :meth:`read`, kept for the callers that want
        nothing else (and for the tests that pin the two readers against
        each other). A cycle that computes both blocks calls :meth:`read`
        once and takes :meth:`GaugeCycleRead.series_for` off it, rather
        than reading the archive twice.
        """
        return self.read(now_utc).series_for(keys)


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
