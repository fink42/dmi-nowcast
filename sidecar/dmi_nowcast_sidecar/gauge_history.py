"""The gauge history one cycle reads, for the ``g_*`` features (v2, F1).

The post-processing model's radar features all come off grids the cycle
already holds. The gauge block does not: it is what the *station itself*
measured in the last hour, and the only place that exists is a
``StationObsStore`` on this host — the parquet the gauge poller
(:mod:`dmi_nowcast_sidecar.station_obs`) writes every ten minutes and the
offline replay trains on.

**Which store, and whose** (v2, S5). On the private instance that is the
corpus archive, kept for ever. On the public instance it is a bounded
store on the data volume, filled by that instance's own poller and pruned
to a week, with the station catalogue arriving through ``sync``. Two
deployments, two directories, one reader: the fallback chain lives in
:func:`resolved_gauge_store_dir` and :func:`resolved_gauge_points_path`,
and nothing below this line knows which instance it is running on.

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
    "resolved_gauge_points_path",
    "resolved_gauge_store_dir",
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
    (:func:`resolved_gauge_points_path`): the public catalogue of DMI
    gauges, the very list the scoreboard evaluates. It is read on first
    use — and on every use until it is there, which is what the public
    instance's synced copy needs — inside the cycle worker where blocking
    is allowed, and it answers two questions: which station stands on a
    given point (:meth:`stations`) and where a given station stands
    (:meth:`coords`).
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
        #: True once a catalogue with stations in it has been parsed. Until
        #: then every cycle re-reads the file (see :meth:`_load_points`).
        self._loaded = False
        #: The last failure, so the retry is quiet after the first line.
        self._points_error: str | None = None

    # -- the coordinate → gauge map ----------------------------------------

    def _load_points(self) -> None:
        """Parse the catalogue into BOTH directions of the map, once.

        One file, one read, one failure log line: the point → station
        lookup the ``g_*`` block needs and the station → coordinate map the
        ``ng_*`` geometry needs are the same twelve kilobytes of JSON, and
        parsing it twice would be two chances to disagree about which
        stations exist.

        **Once it exists.** On the private instance the catalogue is there
        before the process is, so "read on first use" was the whole story.
        On the public instance (S5) the file arrives by ``sync``, which
        means the first cycles run before it does — and a cache that
        remembered "unreadable" would hand out null gauge blocks until
        somebody noticed and restarted the container. So only a catalogue
        with stations in it is cached; anything else leaves ``_loaded``
        false and the next cycle looks again, at the cost of one failed
        ``read_text`` per cycle. The log line is emitted once per distinct
        failure rather than once per cycle, and the pickup says so.
        """
        if self._loaded:
            return
        try:
            raw = json.loads(self.points_file.read_text())
            entries = list(raw.get("points") or ())
            stations = {
                point_key(float(entry["lat"]), float(entry["lon"])): str(
                    entry["id"],
                )
                for entry in entries
            }
            coords = {
                str(entry["id"]): (float(entry["lat"]), float(entry["lon"]))
                for entry in entries
            }
        except Exception as exc:  # noqa: BLE001 — every way a file is junk
            self._note_points_problem(f"{type(exc).__name__}: {exc}")
            return
        if not coords:
            # A document that parsed and named nobody. Not a crash, not a
            # catalogue either — and worth retrying, because the file the
            # sync task is mid-way through replacing looks exactly like it.
            self._note_points_problem("no points in the document")
            return
        self._stations = stations
        self._coords = coords
        self._loaded = True
        self._points_error = None
        _log.info(
            "gauge_history_points_loaded",
            path=str(self.points_file),
            stations=len(coords),
        )

    def _note_points_problem(self, error: str) -> None:
        """Empty maps, and one log line per distinct reason — not per cycle."""
        self._stations = {}
        self._coords = {}
        if error != self._points_error:
            self._points_error = error
            _log.warning(
                "gauge_history_points_unreadable",
                path=str(self.points_file),
                error=error,
                note="null gauge blocks until the file is readable",
            )

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


def resolved_gauge_store_dir(config: Any) -> Path | None:
    """Root of the ``StationObsStore`` this instance reads gauges out of.

    One chain, three links, and the order is the point:

    1. ``postprocess.gauge_store_dir`` — an explicit override, for a store
       neither of the others names.
    2. ``station_obs.store_dir`` — the bounded store this instance's own
       poller fills (S5). The public stack sets that one key and the
       reader follows the writer, rather than the operator spelling the
       same directory twice and getting to spell it differently.
    3. ``storage.corpus_dir`` — the private instance's archive, which is
       what this function answered before any of the keys existed.

    ``None`` when none of them is set, which is a deployment with no gauge
    archive at all.
    """
    settings = getattr(config, "postprocess", None)
    override = None if settings is None else getattr(
        settings, "gauge_store_dir", None,
    )
    if override is not None:
        return Path(override)
    bounded = getattr(getattr(config, "station_obs", None), "store_dir", None)
    if bounded is not None:
        return Path(bounded)
    corpus_dir = config.storage.corpus_dir
    return None if corpus_dir is None else Path(corpus_dir)


def resolved_gauge_points_path(config: Any) -> Path | None:
    """The version-2 station catalogue this instance resolves gauges with.

    ``postprocess.gauge_points_file`` when set — the public instance's
    synced copy — else ``station_eval.points_file``, which is the private
    instance's own and the answer this had before the key existed. ``None``
    when neither is set: a deployment with nothing to map a coordinate to a
    station id with, which is a null gauge block and not an error.

    Also what ``GET /stations/station_points.json`` publishes and what
    ``sync``'s target path for that file is, so the catalogue an instance
    reads is by construction the one it serves.
    """
    settings = getattr(config, "postprocess", None)
    override = None if settings is None else getattr(
        settings, "gauge_points_file", None,
    )
    if override is not None:
        return Path(override)
    points_file = getattr(
        getattr(config, "station_eval", None), "points_file", None,
    )
    return None if points_file is None else Path(points_file)


def build_gauge_history(config: Any) -> GaugeHistory | None:
    """A :class:`GaugeHistory` for this config, or None when there can be none.

    None — and therefore a null gauge block — whenever the deployment has
    no gauge archive to read (:func:`resolved_gauge_store_dir`), no station
    catalogue to resolve a point against
    (:func:`resolved_gauge_points_path`), or the feature turned off. All
    three are ordinary states, not errors.

    The public instance used to be the first of those by construction: no
    corpus volume, no store, null ``ng_*`` columns, a tree model trained on
    21 features running on 21 nulls. It now polls its own bounded store and
    receives the catalogue through ``sync``, and both keys resolve — which
    is the whole of S5 as far as this module is concerned. A catalogue that
    has not arrived yet is NOT this branch: the object is built and every
    cycle re-reads the file until it exists (:meth:`_load_points`).
    """
    settings = getattr(config, "postprocess", None)
    if settings is not None and not settings.gauge_features:
        return None
    store_dir = resolved_gauge_store_dir(config)
    points_file = resolved_gauge_points_path(config)
    if store_dir is None or points_file is None:
        return None
    return GaugeHistory(
        store_dir,
        points_file,
        lag_min=(
            DEFAULT_GAUGE_LAG_MIN if settings is None
            else float(settings.gauge_lag_min)
        ),
    )
