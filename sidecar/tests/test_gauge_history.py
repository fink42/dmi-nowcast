"""One gauge read per cycle, for two feature blocks (v2, S1).

``g_*`` asks what the gauge ON a point measured, so the first version of
:mod:`dmi_nowcast_sidecar.gauge_history` read the stations among the
cycle's points and nothing else. ``ng_*`` asks what the gauges AROUND a
point measured, and at a subscriber's address every one of those is a
station the cycle would never otherwise have read — so the read is now
over the whole catalogue and both blocks are served from it.

The seams that would be invisible if they broke:

1. **The read really covers the catalogue.** A cycle serving one address
   that is not a gauge still has to see every gauge in the country, or the
   neighbour block would be null exactly where it is the only gauge signal
   there is.
2. **It still stops at the visibility horizon.** A row the features may
   not read is not worth decoding, and a read that quietly ran to *now*
   would hand the block rain the live service has not been told about.
3. **Leave-self-out by distance alone.** A point that IS a gauge excludes
   itself by id. A point that is not cannot be named — so the half
   kilometre radius has to catch the gauge standing on it, or an address
   next to a gauge would read that gauge's own rain as a neighbour's.

The fixture is four stations on one line of latitude, with a band already
raining over the westernmost. Offline and synthetic: no network, no radar.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import structlog

pytest.importorskip("pyarrow")

import numpy as np

from dmi_nowcast_core import postprocess as pp
from dmi_nowcast_core.metobs import Observation
from dmi_nowcast_core.station_store import StationObsStore
from dmi_nowcast_sidecar.gauge_history import GaugeHistory
from dmi_nowcast_sidecar.push.postprocess import point_key

RADAR_TS = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)
GENERATED_AT = RADAR_TS + timedelta(minutes=14)
LAG_MIN = pp.DEFAULT_GAUGE_LAG_MIN

BASE_LAT, BASE_LON = 56.0, 9.0

#: Kilometres east of ``BASE_LON``. 06180 is the wet one; 06120 is 10 km
#: downwind of it (inside the 60 km radius and inside the first travel-time
#: bin at 45 km/h), 06170 is 25 km out — inside 06120's 20 km vicinity and
#: outside the vicinity of a point beside 06180 — and 06070 at 90 km is
#: downstream of both and beyond the neighbour radius from 06180.
EAST_KM: dict[str, float] = {
    "06180": 0.0, "06120": 10.0, "06170": 25.0, "06070": 90.0,
}

#: The band's motion: eastward, so the gauges to the WEST are upstream.
BULK_KMH = 45.0
BULK_DIR_DEG = 90.0

#: A place that is not a gauge, 200 m east of 06180 — inside
#: ``NG_SELF_KM``, so 06180 must be invisible to it as a neighbour even
#: though nothing can tell this point its own station id.
NEXT_TO_A_GAUGE = 0.2


def _lon(east_km: float) -> float:
    return BASE_LON + east_km / pp.KM_PER_DEG_LON


def _coords() -> dict[str, tuple[float, float]]:
    return {sid: (BASE_LAT, _lon(east)) for sid, east in EAST_KM.items()}


def _points_file(tmp_path: Path) -> Path:
    path = tmp_path / "station_points.json"
    path.write_text(json.dumps({
        "version": 2,
        "points": [
            {"id": sid, "lat": lat, "lon": lon, "region": "Fixture"}
            for sid, (lat, lon) in _coords().items()
        ],
    }))
    return path


def _store(tmp_path: Path) -> StationObsStore:
    """Six ten-minute slots per station; only 06180 is wet in them.

    Every station reports, so every station has a KNOWN last half hour —
    which is what separates "dry there" from "nobody looked", the
    distinction the neighbour block's counts rest on.
    """
    store = StationObsStore(tmp_path / "corpus")
    rows: list[Observation] = []
    for sid in EAST_KM:
        for k in range(6):
            rows.append(Observation(
                station_id=sid,
                observed_utc=RADAR_TS - timedelta(minutes=10 * k),
                parameter_id="precip_past10min",
                value=1.2 if sid == "06180" else 0.0,
            ))
    store.append(rows)
    return store


def _history(tmp_path: Path, *, empty: bool = False) -> GaugeHistory:
    store = _store(tmp_path)
    return GaugeHistory(
        store.root,
        tmp_path / "missing.json" if empty else _points_file(tmp_path),
        lag_min=LAG_MIN,
    )


def _block(
    history: GaugeHistory, keys: list[tuple[float, float]],
) -> dict[str, np.ndarray]:
    """The ``ng_*`` block for ``keys``, exactly as the cycle computes it."""
    from dmi_nowcast_sidecar.compute import neighbour_features

    return neighbour_features(
        history.read(GENERATED_AT), keys,
        now_utc=GENERATED_AT,
        bulk_kmh=BULK_KMH, bulk_dir_deg=BULK_DIR_DEG,
        lag_min=LAG_MIN,
    )


# ---------------------------------------------------------------------------
# 1. The read
# ---------------------------------------------------------------------------


class TestTheCycleRead:
    def test_it_covers_every_catalogue_station_not_only_the_points(
        self, tmp_path: Path,
    ) -> None:
        """One address, no gauge on it — and the whole country is read.

        The ``g_*`` block would have been satisfied by reading nothing at
        all here. The neighbour block is only worth having because this
        read is wider than the cycle's point list.
        """
        history = _history(tmp_path)
        read = history.read(GENERATED_AT)
        assert set(read.by_station) == set(EAST_KM)
        assert set(read.table.stations) == set(EAST_KM)
        assert set(read.coords) == set(EAST_KM)
        # Every station really carries slots, not just a key.
        assert all(read.by_station[sid] for sid in EAST_KM)

    def test_it_ends_at_the_visibility_horizon(self, tmp_path: Path) -> None:
        """A row the features may not look at is not worth decoding."""
        read = _history(tmp_path).read(GENERATED_AT)
        assert read.end_utc == GENERATED_AT - timedelta(minutes=LAG_MIN)
        assert read.start_utc == read.end_utc - timedelta(
            minutes=pp.GAUGE_SINCE_CAP_MIN + 10.0,
        )
        assert read.end_utc - read.start_utc >= timedelta(
            minutes=pp.GAUGE_SINCE_CAP_MIN,
        )
        # And nothing inside the lag was REPORTED: ``gauge_slot_amounts``
        # lays a contiguous slot grid over the window, so the grid can run
        # one slot past the horizon — with no value in it, which is the
        # part that matters.
        beyond = read.table.ends > read.end_utc.timestamp()
        assert not np.isfinite(read.table.wet[:, beyond]).any()
        assert not np.isfinite(read.table.mm[:, beyond]).any()

    def test_the_slot_table_is_the_series_it_carries(
        self, tmp_path: Path,
    ) -> None:
        """One read, two shapes — and they are the same rows.

        ``series_for`` feeds ``g_*`` and ``table`` feeds ``ng_*``; if the
        digest were built off a second read the two blocks of one row would
        be describing different archives.
        """
        history = _history(tmp_path)
        read = history.read(GENERATED_AT)
        digest = pp.GaugeSlotTable.from_slots(read.by_station)
        assert read.table.stations == digest.stations
        np.testing.assert_array_equal(read.table.ends, digest.ends)
        np.testing.assert_allclose(read.table.mm, digest.mm, equal_nan=True)

    def test_series_for_is_the_old_per_point_answer(
        self, tmp_path: Path,
    ) -> None:
        history = _history(tmp_path)
        keys = [
            point_key(BASE_LAT, _lon(NEXT_TO_A_GAUGE)),
            point_key(*_coords()["06120"]),
        ]
        read = history.read(GENERATED_AT)
        assert read.series_for(keys) == history.slots_for(
            keys, now_utc=GENERATED_AT,
        )
        assert read.series_for(keys)[0] is None      # not a gauge
        assert read.series_for(keys)[1]              # a gauge, with slots
        assert read.station_ids(keys) == [None, "06120"]

    def test_the_catalogue_is_parsed_once_in_both_directions(
        self, tmp_path: Path,
    ) -> None:
        """``coords`` and ``stations`` are the same file, unrounded one way."""
        history = _history(tmp_path)
        assert history.coords() == _coords()
        assert history.stations() == {
            point_key(lat, lon): sid for sid, (lat, lon) in _coords().items()
        }

    def test_an_unreadable_catalogue_is_one_log_line_and_empty_maps(
        self, tmp_path: Path,
    ) -> None:
        history = _history(tmp_path, empty=True)
        with structlog.testing.capture_logs() as logs:
            read = history.read(GENERATED_AT)
            history.coords()
            history.stations()
        assert read.by_station == {} and read.coords == {}
        assert read.table.stations == ()
        assert [e["event"] for e in logs] == ["gauge_history_points_unreadable"]


# ---------------------------------------------------------------------------
# 2. Leave-self-out at a place with no gauge
# ---------------------------------------------------------------------------


class TestAPointThatIsNotAGauge:
    def test_it_gets_a_filled_block_excluded_by_distance_alone(
        self, tmp_path: Path,
    ) -> None:
        """200 m from 06180: the block is real, and 06180 is not in it.

        Nothing can tell this point its own station id — it has none — so
        the only thing standing between it and reading 06180's own rain
        back as a neighbour's is the ``NG_SELF_KM`` radius.
        """
        history = _history(tmp_path)
        key = point_key(BASE_LAT, _lon(NEXT_TO_A_GAUGE))
        read = history.read(GENERATED_AT)
        assert read.station_ids([key]) == [None]

        block = _block(history, [key])
        # Filled: a usable motion frame, a nearest neighbour, gauges around.
        assert block["ng_frame_ok"][0] == 1.0
        assert block["ng_count_20km"][0] == 1.0        # 06120, 9.8 km east
        # 06180 sits 0.2 km away and must NOT be the nearest gauge; 06120,
        # 9.8 km east, must be.
        assert float(block["ng_near_km"][0]) == pytest.approx(
            10.0 - NEXT_TO_A_GAUGE, abs=0.05,
        )
        # Its own neighbour's rain is dry (06120 is the dry one), and the
        # wet gauge 200 m away reached no column at all.
        assert float(block["ng_near_mm_60"][0]) == pytest.approx(0.0)
        assert math.isnan(float(block["ng_upwet_tau_min"][0]))
        assert float(block["ng_wet_share_20km"][0]) == pytest.approx(0.0)

    def test_a_gauge_point_excludes_itself_by_id(self, tmp_path: Path) -> None:
        """The same rule the training rows were built under.

        06120 is 10 km downwind of the wet 06180 at 45 km/h, so its
        neighbour block has to say "wet, 13 minutes upstream" — and say
        nothing about its own dry gauge.
        """
        history = _history(tmp_path)
        key = point_key(*_coords()["06120"])
        block = _block(history, [key])
        assert float(block["ng_upwet_tau_min"][0]) == pytest.approx(
            60.0 * 10.0 / BULK_KMH, abs=0.5,
        )
        assert float(block["ng_up_count_t30"][0]) == 1.0
        # Three visible wet slots of 1.2 mm — ``g_mm_30``'s own window,
        # read at the neighbour instead of at the point.
        assert float(block["ng_up_mm_max_t30"][0]) == pytest.approx(3.6)
        # 06170 is 15 km DOWNSTREAM: it is in the vicinity count and in no
        # upstream bin.
        assert float(block["ng_count_20km"][0]) == 2.0
        assert float(block["ng_up_count_t60"][0]) == 0.0

    def test_a_dead_motion_frame_keeps_the_vicinity_block(
        self, tmp_path: Path,
    ) -> None:
        """``ng_frame_ok`` = 0 nulls the corridor, never the count."""
        from dmi_nowcast_sidecar.compute import neighbour_features

        history = _history(tmp_path)
        key = point_key(*_coords()["06120"])
        block = neighbour_features(
            history.read(GENERATED_AT), [key],
            now_utc=GENERATED_AT,
            bulk_kmh=1.0, bulk_dir_deg=BULK_DIR_DEG, lag_min=LAG_MIN,
        )
        assert block["ng_frame_ok"][0] == 0.0
        assert block["ng_count_20km"][0] == 2.0
        assert math.isnan(float(block["ng_upwet_tau_min"][0]))
        assert math.isnan(float(block["ng_up_mm_max_t30"][0]))
        # Geometry survives a dead flow: it never needed a direction.
        assert math.isfinite(float(block["ng_near_km"][0]))
