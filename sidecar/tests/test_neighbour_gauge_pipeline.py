"""The neighbour-gauge builder and the random-point protocol, end to end.

Tested from the sidecar suite for the reason ``test_fit_postprocess`` is:
both halves lean on the core library AND on the sidecar (the gauge-slot
bucketing, the threshold sweep's row reader), and this environment is the
only one that has both.

The fixture is a band crossing Denmark west to east
-----------------------------------------------------
Six stations on one line of latitude, at 0, 5, 25, 40, 68 and 96 km east of
a base longitude — spaced unevenly on purpose, so their distances to the
nearest OTHER station land in three different bins (5, 5, 15, 15, 28,
28 km) and the by-distance table has more than one row. Every day, a band
crosses them at 45 km/h in the direction the motion columns claim, so each
station's gauge goes wet exactly ``east_km / 45 * 60`` minutes after the
first one does.

The radar features are deliberately useless: the ensemble fraction says
"something is coming within the next two hours" and every other column is a
constant. So the ONLY thing in the table that can say *when* is the gauge
network — the station's own gauge in the at-gauge protocol, and the
upstream neighbours' once the own gauge is masked away. That makes "the
``ng_*`` block carries information" a checkable claim rather than a hope.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
numpy = pytest.importorskip("numpy")
np = numpy

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import add_neighbour_gauge_features as ngb  # noqa: E402
import fit_postprocess as fit  # noqa: E402
import replay_warnings as rw  # noqa: E402

from dmi_nowcast_core import postprocess as pp  # noqa: E402
from dmi_nowcast_core.metobs import Observation  # noqa: E402
from dmi_nowcast_core.station_store import StationObsStore  # noqa: E402

#: Real DMI station ids, west to east; 06080 is left out on purpose (the
#: dead-gauge rule excludes it in production and a fixture should not
#: quietly depend on that).
STATIONS: tuple[str, ...] = (
    "06180", "06120", "06170", "06110", "06070", "06190",
)

#: Kilometres east of :data:`BASE_LON`, one per station. The gaps are 5,
#: 20, 15, 28, 28 km, so the nearest-OTHER-station distances are
#: 5, 5, 15, 15, 28, 28 — three distinct distance bins.
EAST_KM: tuple[float, ...] = (0.0, 5.0, 25.0, 40.0, 68.0, 96.0)

BASE_LAT, BASE_LON = 56.0, 9.0
LEADS = (20, 30)

#: The band's speed and the bearing it is heading toward (90 = east).
BAND_KMH = 45.0
BAND_DIR_DEG = 90.0

MONTHS: tuple[tuple[int, int], ...] = ((2026, 1), (2026, 4), (2026, 6))
DAYS_PER_MONTH = 7

FIRST_FRAME_MIN, LAST_FRAME_MIN = 6 * 60, 17 * 60
FIRST_SLOT_MIN, LAST_SLOT_MIN = 4 * 60, 22 * 60
EVENT_SLOTS = 6


def _lon(east_km: float) -> float:
    return BASE_LON + east_km / pp.KM_PER_DEG_LON


def _coords() -> dict[str, tuple[float, float]]:
    return {
        sid: (BASE_LAT, _lon(east))
        for sid, east in zip(STATIONS, EAST_KM)
    }


def _days() -> list[tuple[int, int, int]]:
    return [
        (year, month, 1 + offset)
        for year, month in MONTHS
        for offset in range(DAYS_PER_MONTH)
    ]


def _at(day: tuple[int, int, int], minute: float) -> datetime:
    return datetime(*day, tzinfo=timezone.utc) + timedelta(minutes=float(minute))


def _onset_min(day: tuple[int, int, int], station: str) -> int:
    """Minute of day at which this station's first wet slot ENDS.

    The band reaches the westernmost station at a time that walks through
    the week, then crosses the others at 45 km/h. Snapped to the slot grid,
    because a gauge slot ends on a ten-minute boundary.
    """
    start = 8 * 60 + 20 * (day[2] % 4)
    east = EAST_KM[STATIONS.index(station)]
    return int(round((start + east / BAND_KMH * 60.0) / 10.0)) * 10


def _gauge_slots(
    day: tuple[int, int, int], station: str,
) -> list[tuple[datetime, bool, float]]:
    """The station's own ``(slot_end, wet, mm)`` grid for one day."""
    first = _onset_min(day, station)
    wet = {first + 10 * k for k in range(EVENT_SLOTS)}
    return [
        (_at(day, minute), minute in wet, 0.6 if minute in wet else 0.0)
        for minute in range(FIRST_SLOT_MIN, LAST_SLOT_MIN + 1, 10)
    ]


def _decision_rows(day: tuple[int, int, int]) -> list[dict]:
    """One day of decision rows: a real motion, and radar features that lie.

    ``raw_frac_<lead>`` saturates at 1.0 for any row within two hours of
    this station's onset and sits at 0.05 otherwise, so it knows an event
    is coming and nothing about when. Every other radar column is a
    constant. The gauge network is the only source of timing in the table
    — the station's own ``g_*`` block, written here through the very
    function the replay calls, and the neighbours' once that is masked.
    """
    slots = [_gauge_slots(day, station) for station in STATIONS]
    rows: list[dict] = []
    for minute in range(FIRST_FRAME_MIN, LAST_FRAME_MIN + 1, 10):
        stamp = _at(day, minute)
        gauge = pp.station_gauge_features(
            slots, now_utc=stamp, lag_min=pp.DEFAULT_GAUGE_LAG_MIN,
        )
        for index, station in enumerate(STATIONS):
            ahead = _onset_min(day, station) - minute
            soon = -30.0 <= ahead <= 120.0
            rows.append({
                **{
                    name: pp.finite_or_none(values[index])
                    for name, values in gauge.items()
                },
                "radar_ts": stamp,
                "generated_at": stamp,
                "station_id": station,
                "p_rain": 0.99,
                "action": "none",
                **{f"p_rain_{lead}": (0.6 if soon else 0.02) for lead in LEADS},
                **{pp.raw_fraction_column(lead): (1.0 if soon else 0.05)
                   for lead in (10, 20, 30, 45, 60)},
                "eta_min": None,
                "intensity_mm_h": None,
                "observed_mm_h": 0.0,
                "forecast_now_mm_h": 0.0,
                "armed_after": True,
                "streak_after": 0,
                "season": pp.season_of_month(day[1]),
                "hour_utc": stamp.hour,
                # Constants: nothing here separates one row from another.
                "obs_max_5km_mm_h": 0.0,
                "up_max_20km_mm_h": 0.0,
                "up_max_40km_mm_h": 0.0,
                "up_dist_km": None,
                "up_wet_frac_40km": 0.0,
                "bulk_kmh": BAND_KMH,
                "bulk_dir_deg": BAND_DIR_DEG,
                "local_speed_kmh": BAND_KMH,
                "stalled_share": 0.0,
                "frame_age_min": 0.0,
                "station_radar_km": 30.0,
            })
    return rows


def _write_run(directory: Path) -> Path:
    for day in _days():
        rw.write_decisions(
            directory / "decisions"
            / f"{day[0]:04d}-{day[1]:02d}-{day[2]:02d}.parquet",
            _decision_rows(day), LEADS, features=True,
        )
    (directory / "summary.json").write_text(json.dumps({
        "run": {
            "n_days": len(_days()), "n_stations": len(STATIONS),
            "frame_age_min": 0,
            "steps": {
                "ensemble_size": 16, "n_cascade_levels": 6,
                "downsample_factor": 4, "horizon_min": 90,
                "leads_min": list(LEADS), "threshold_mm_h": 0.5,
            },
            "flow": {"completion": "confidence"},
            "rules": {"lead_min": 30, "threshold_pct": 40},
            "national_curves": "/x/national_curves.json",
            "features": {"enabled": True, "gauge_lag_min": 10.0},
        },
    }))
    return directory


def _write_gauge(corpus_dir: Path) -> None:
    """The archive behind the rows — the same slots :func:`_gauge_slots` has.

    One source of truth for "when does it rain at this station on this
    day": the decision rows' own ``g_*`` block, the builder's neighbour
    block and the benchmark's outcome all come off it.
    """
    store = StationObsStore(corpus_dir)
    observations: list[Observation] = []
    for day in _days():
        for station in STATIONS:
            for end, _wet, mm in _gauge_slots(day, station):
                observations.append(Observation(
                    station_id=station,
                    observed_utc=end,
                    parameter_id="precip_past10min",
                    value=float(mm),
                ))
    store.append(observations)


def _write_points(path: Path) -> Path:
    path.write_text(json.dumps({
        "version": 2,
        "points": [
            {"id": sid, "lat": lat, "lon": lon, "region": "Fixture"}
            for sid, (lat, lon) in _coords().items()
        ],
    }))
    return path


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("neighbour")
    _write_gauge(root / "corpus")
    _write_run(root / "replay")
    _write_points(root / "points.json")
    return root


@pytest.fixture(scope="module")
def built(corpus: Path) -> Path:
    """The ``_ng`` copy of the run, built by the script under test."""
    assert ngb.main([
        "--run", str(corpus / "replay"),
        "--corpus-dir", str(corpus / "corpus"),
        "--points", str(corpus / "points.json"),
        "--out-suffix", "_ng",
    ]) == 0
    return corpus / "replay_ng"


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------


class TestTheBuilder:
    def test_it_copies_the_run_and_adds_every_column(
        self, corpus: Path, built: Path,
    ) -> None:
        import pyarrow.parquet as pq

        source = sorted((corpus / "replay" / "decisions").glob("*.parquet"))
        copied = sorted((built / "decisions").glob("*.parquet"))
        assert [p.name for p in copied] == [p.name for p in source]
        before = pq.read_table(source[0])
        after = pq.read_table(copied[0])
        assert after.num_rows == before.num_rows
        assert set(before.schema.names) <= set(after.schema.names)
        for name, _definition in pp.SCALAR_FEATURE_COLUMNS_NG:
            assert name in after.schema.names, name

    def test_the_run_itself_is_untouched(self, corpus: Path, built: Path) -> None:
        """The source is the control arm of the comparison; it must survive."""
        import pyarrow.parquet as pq

        source = sorted((corpus / "replay" / "decisions").glob("*.parquet"))[0]
        table = pq.read_table(source)
        values = table.column("ng_near_km").to_pylist()
        assert all(value is None for value in values)

    def test_the_columns_are_filled_and_the_frame_is_usable(
        self, built: Path,
    ) -> None:
        import pyarrow.parquet as pq

        table = pq.read_table(sorted((built / "decisions").glob("*.parquet"))[0])
        near = np.asarray(
            table.column("ng_near_km").to_numpy(zero_copy_only=False),
        )
        assert np.all(np.isfinite(near))
        assert np.all(
            np.asarray(table.column("ng_frame_ok").to_numpy(zero_copy_only=False))
            == 1.0
        )

    def test_the_geometry_is_the_fixture_s_geometry(self, built: Path) -> None:
        """``ng_near_km`` is the distance to the nearest OTHER station."""
        import pyarrow.parquet as pq

        table = pq.read_table(sorted((built / "decisions").glob("*.parquet"))[0])
        stations = np.asarray(table.column("station_id").to_pylist())
        near = np.asarray(
            table.column("ng_near_km").to_numpy(zero_copy_only=False),
        )
        expected = {
            "06180": 5.0, "06120": 5.0, "06170": 15.0,
            "06110": 15.0, "06070": 28.0, "06190": 28.0,
        }
        for station, want in expected.items():
            got = near[stations == station]
            assert got == pytest.approx(want, abs=0.1), station

    def test_a_station_never_reads_its_own_rain(
        self, corpus: Path, tmp_path: Path,
    ) -> None:
        """Leave-self-out, on a fixture where a leak could not be missed.

        One station is rained on all day and every other gauge is deleted
        from the archive. If its own gauge could reach its own features,
        every ``ng_*`` column at that station would be soaked; because it
        cannot, the station has no neighbour with anything to say and the
        whole block is empty there.
        """
        import pyarrow.parquet as pq

        root = tmp_path / "solo"
        store = StationObsStore(root / "corpus")
        day = (2026, 6, 1)
        observations = [
            Observation(
                station_id="06170",
                observed_utc=_at(day, minute),
                parameter_id="precip_past10min",
                value=9.9,
            )
            for minute in range(FIRST_SLOT_MIN, LAST_SLOT_MIN + 1, 10)
        ]
        store.append(observations)
        (root / "replay" / "decisions").mkdir(parents=True)
        rw.write_decisions(
            root / "replay" / "decisions" / "2026-06-01.parquet",
            [r for r in _decision_rows(day)], LEADS, features=True,
        )
        _write_points(root / "points.json")
        assert ngb.main([
            "--run", str(root / "replay"),
            "--corpus-dir", str(root / "corpus"),
            "--points", str(root / "points.json"),
        ]) == 0
        table = pq.read_table(
            root / "replay_ng" / "decisions" / "2026-06-01.parquet",
        )
        stations = np.asarray(table.column("station_id").to_pylist())
        mine = stations == "06170"
        for name in ("ng_near_mm_60", "ng_upwet_mm_30", "ng_up_mm_max_t30",
                     "ng_up_mm_max_t60", "ng_up_mm_max_t120",
                     "ng_wet_share_20km"):
            values = np.asarray(
                table.column(name).to_numpy(zero_copy_only=False),
            )[mine]
            assert not np.any(np.nan_to_num(values, nan=0.0) > 0.0), name
        # ...while the stations DOWNSTREAM of it, which may read it, do.
        theirs = np.asarray(
            table.column("ng_near_mm_60").to_numpy(zero_copy_only=False),
        )[stations == "06110"]
        assert np.nanmax(theirs) > 0.0

    def test_the_decisions_directory_names_the_same_run(
        self, corpus: Path,
    ) -> None:
        """``--run x`` and ``--run x/decisions`` mean one run, one copy."""
        assert ngb.output_dir(corpus / "replay", "_ng") == (
            ngb.output_dir(corpus / "replay" / "decisions", "_ng")
        )
        assert [p.name for p in ngb.decision_files(corpus / "replay")] == [
            p.name for p in ngb.decision_files(corpus / "replay" / "decisions")
        ]

    def test_a_run_with_no_decisions_is_a_message_not_a_traceback(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / "empty" / "decisions").mkdir(parents=True)
        with pytest.raises(ngb.BuildError, match="no decisions"):
            ngb.decision_files(tmp_path / "empty")

    def test_the_summary_records_what_the_block_was_built_under(
        self, built: Path,
    ) -> None:
        summary = json.loads((built / "summary.json").read_text())
        # The original settings survive, so the benchmark's parity check
        # still has something to check.
        assert summary["run"]["steps"]["ensemble_size"] == 16
        block = summary["run"]["features"]["neighbour"]
        assert block["enabled"] is True
        assert block["radius_km"] == pp.NG_RADIUS_KM
        assert block["cross_km"] == pp.NG_CROSS_KM
        assert block["tau_edges_min"] == list(pp.NG_TAU_EDGES_MIN)
        assert block["gauge_lag_min"] == 10.0
        assert len(block["columns"]) == len(pp.SCALAR_FEATURE_COLUMNS_NG)

    def test_it_refuses_to_overwrite_and_then_repeats_itself(
        self, corpus: Path, built: Path,
    ) -> None:
        import pyarrow.parquet as pq

        argv = [
            "--run", str(corpus / "replay"),
            "--corpus-dir", str(corpus / "corpus"),
            "--points", str(corpus / "points.json"),
            "--out-suffix", "_ng",
        ]
        first = sorted((built / "decisions").glob("*.parquet"))[0]
        before = pq.read_table(first).column("ng_upwet_tau_min").to_numpy(
            zero_copy_only=False,
        )
        assert ngb.main(argv) == 2
        assert ngb.main(argv + ["--force"]) == 0
        after = pq.read_table(first).column("ng_upwet_tau_min").to_numpy(
            zero_copy_only=False,
        )
        assert np.array_equal(after, before, equal_nan=True)

    def test_workers_change_nothing_but_the_wall_clock(
        self, corpus: Path, tmp_path: Path,
    ) -> None:
        import pyarrow.parquet as pq

        assert ngb.main([
            "--run", str(corpus / "replay"),
            "--corpus-dir", str(corpus / "corpus"),
            "--points", str(corpus / "points.json"),
            "--out-suffix", "_par", "--workers", "2",
        ]) == 0
        name = "2026-06-01.parquet"
        one = pq.read_table(corpus / "replay_ng" / "decisions" / name)
        many = pq.read_table(corpus / "replay_par" / "decisions" / name)
        for column, _definition in pp.SCALAR_FEATURE_COLUMNS_NG:
            assert np.array_equal(
                one.column(column).to_numpy(zero_copy_only=False),
                many.column(column).to_numpy(zero_copy_only=False),
                equal_nan=True,
            ), column


class TestTheReplayWritesTheSameBlock:
    """The replay writes ``ng_*`` itself now (v2/S1); this is against whom.

    The builder above is the reference: the shipped model's ``_ng`` training
    columns came out of it. From 2026-09-22 ``replay_warnings.sample_frame``
    writes the block in the same pass as the ``g_*`` one, so every new run
    carries it without the copy — and the two writers have to agree, or the
    next refit would be trained on columns that mean something slightly
    different from the ones before it.

    Compared over a whole replayed day: ~72 decision instants x 6 stations,
    every one of the 21 columns, against the parquet the builder wrote.

    **Exact, except the millimetres.** The two read the same archive over
    slightly different windows — the replay reuses the day's read (from
    midnight minus the six-hour cap), the builder cuts its own from the
    file's first decision instant. Every slot the availability rule lets a
    row see is in both, so every count, share, distance, travel time and age
    is identical. The rainfall totals are a float32 sum whose pairwise
    grouping depends on how many slots the table holds, so they agree to
    float32 rounding and not to the bit — ~2e-7 relative on a 3.6 mm total,
    against columns the model reads in tenths of a millimetre.
    """

    #: The columns that are a SUM over slots, and therefore the only ones
    #: whose last bit may depend on the window the slots were read over.
    SUMMED = tuple(
        name for name, _d in pp.SCALAR_FEATURE_COLUMNS_NG if "_mm" in name
    )

    DAY = _days()[3]
    LAG = pp.DEFAULT_GAUGE_LAG_MIN

    def _built_rows(self, built: Path) -> list[dict]:
        import pyarrow.parquet as pq

        name = f"{self.DAY[0]:04d}-{self.DAY[1]:02d}-{self.DAY[2]:02d}.parquet"
        return pq.read_table(built / "decisions" / name).to_pylist()

    def test_every_row_of_a_whole_day_matches_the_builder(
        self, corpus: Path, built: Path,
    ) -> None:
        from datetime import date

        from dmi_nowcast_core.station_store import StationObsStore

        coords = _coords()
        points = [
            rw.StationPoint(id=sid, lat=lat, lon=lon, region=None)
            for sid, (lat, lon) in coords.items()
        ]
        order = {point.id: index for index, point in enumerate(points)}
        # The replay's own read — the day's slots, from the day worker —
        # rather than the builder's per-file window.
        slots = rw.day_feature_slots(
            StationObsStore(corpus / "corpus"),
            date(*self.DAY),
            list(coords),
            lag_min=self.LAG,
        )
        names = [name for name, _d in pp.SCALAR_FEATURE_COLUMNS_NG]
        by_instant: dict[object, list[dict]] = {}
        for row in self._built_rows(built):
            by_instant.setdefault(row["generated_at"], []).append(row)
        assert len(by_instant) > 50            # a whole day, not one frame
        checked = 0
        filled = 0
        for stamp, group in by_instant.items():
            block = rw.neighbour_gauge_columns(
                points, slots,
                bulk_kmh=BAND_KMH, bulk_dir_deg=BAND_DIR_DEG,
                now_utc=stamp, lag_min=self.LAG,
            )
            for row in group:
                index = order[row["station_id"]]
                for name in names:
                    mine = float(block[name][index])
                    theirs = row[name]
                    # The builder writes NaN where nothing is computable and
                    # the replay writes a parquet null for the same case, so
                    # both absences read as "not a number" here.
                    if theirs is None or not np.isfinite(theirs):
                        assert not np.isfinite(mine), (name, stamp, index)
                        continue
                    if name in self.SUMMED:
                        assert mine == pytest.approx(theirs, rel=1e-6), (
                            name, stamp, index,
                        )
                    else:
                        assert mine == theirs, (name, stamp, index)
                    filled += 1
                checked += 1
        assert checked == len(self._built_rows(built))
        # Parity on nulls is not parity: most of these are measurements.
        assert filled > 5 * checked

    def test_a_replayed_cycle_writes_the_block_into_its_own_row(
        self, corpus: Path,
    ) -> None:
        """The writer, not just the producer: through ``feature_row``.

        ``sample_frame`` is the only place the block is merged into the
        grid-feature dict a replay row is assembled from; this pins that the
        merge reaches the row under the schema's names, without running a
        radar cycle.
        """
        coords = _coords()
        points = [
            rw.StationPoint(id=sid, lat=lat, lon=lon, region=None)
            for sid, (lat, lon) in coords.items()
        ]
        stamp = _at(self.DAY, _onset_min(self.DAY, STATIONS[-1]))
        slots = {sid: _gauge_slots(self.DAY, sid) for sid in coords}
        grid_features = {
            "bulk_kmh": np.full(len(points), BAND_KMH, dtype=np.float32),
            "bulk_dir_deg": np.full(
                len(points), BAND_DIR_DEG, dtype=np.float32,
            ),
        }
        grid_features.update(rw.neighbour_gauge_columns(
            points, slots,
            bulk_kmh=float(np.float32(grid_features["bulk_kmh"][0])),
            bulk_dir_deg=float(np.float32(grid_features["bulk_dir_deg"][0])),
            now_utc=stamp, lag_min=self.LAG,
        ))
        row = rw._feature_row(
            grid_features, 0, points[0],
            raw_p_rain={}, pixel=None, leads_min=LEADS,
            season=pp.season_of_month(self.DAY[1]),
            hour_utc=stamp.hour, frame_age_min=0.0,
        )
        assert {name for name, _d in pp.SCALAR_FEATURE_COLUMNS_NG} <= set(row)
        assert row["ng_frame_ok"] == 1.0
        assert row["ng_near_km"] == pytest.approx(5.0, abs=0.05)
        # The westernmost station has nothing upstream of it, so the one
        # column that answers "when" is null there and the geometry is not.
        assert row["ng_upwet_tau_min"] is None

    def test_the_settings_block_is_the_builders(self) -> None:
        """One shape for ``run.features.neighbour``, whoever wrote the run."""
        mine = rw.neighbour_settings_block(rw.FrameSettings())
        theirs = ngb.neighbour_settings_block(pp.DEFAULT_GAUGE_LAG_MIN)
        assert set(mine) == set(theirs)
        assert {k: v for k, v in mine.items() if k != "gauge_lag_min"} == {
            k: v for k, v in theirs.items() if k != "gauge_lag_min"
        }


class TestTheBuilderDefersToTheReplay:
    """It must not quietly recompute a block the run already carries.

    The builder exists for the runs written before the replay wrote the
    block itself. Pointed at one that has it, a silent second opinion about
    the same rows — under possibly another lag or another catalogue — is
    worse than a message.
    """

    def test_it_refuses_a_run_that_already_carries_the_block(
        self, corpus: Path, built: Path, capsys: pytest.CaptureFixture,
    ) -> None:
        assert ngb.main([
            "--run", str(built),
            "--corpus-dir", str(corpus / "corpus"),
            "--points", str(corpus / "points.json"),
            "--out-suffix", "_twice",
        ]) == 2
        assert "already carries" in capsys.readouterr().err
        assert not (corpus / "replay_ng_twice").exists()

    def test_force_recomputes_it_and_says_what_it_replaced(
        self, corpus: Path, built: Path, tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        import shutil

        import pyarrow.parquet as pq

        source = sorted((built / "decisions").glob("*.parquet"))[0]
        run = tmp_path / "one_day"
        (run / "decisions").mkdir(parents=True)
        shutil.copy(source, run / "decisions" / source.name)
        shutil.copy(built / "summary.json", run / "summary.json")
        assert ngb.main([
            "--run", str(run),
            "--corpus-dir", str(corpus / "corpus"),
            "--points", str(corpus / "points.json"),
            "--out-suffix", "_again", "--force",
        ]) == 0
        err = capsys.readouterr().err
        assert "replaced 21 pre-existing ng_* column(s)" in err
        again = tmp_path / "one_day_again" / "decisions" / source.name
        before = pq.read_table(source)
        after = pq.read_table(again)
        for column, _definition in pp.SCALAR_FEATURE_COLUMNS_NG:
            assert np.array_equal(
                before.column(column).to_numpy(zero_copy_only=False),
                after.column(column).to_numpy(zero_copy_only=False),
                equal_nan=True,
            ), column
        # One column per name, not two: the drop is a replacement.
        assert len(after.schema.names) == len(before.schema.names)


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------


def _fit(root: Path, run: Path, out: Path, *extra: str) -> dict:
    argv = [
        "--run", str(run),
        "--corpus-dir", str(root / "corpus"),
        "--points", str(root / "points.json"),
        "--out-dir", str(out),
        "--leads", "20,30",
        "--design-leads", "10,20,30,45,60",
        "--design", "v2",
        "--resamples", "30",
        *extra,
    ]
    assert fit.main(argv) == 0
    return json.loads((out / "postprocess_report.json").read_text())


@pytest.fixture(scope="module")
def random_point(
    corpus: Path, built: Path, tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict, Path]:
    out = tmp_path_factory.mktemp("rp-out")
    report = _fit(
        corpus, built, out,
        "--protocol", "random-point",
        "--station-groups", "3",
        "--baseline", "refit-v1",
        "--compare", "logistic/drop=gauge+neighbour",
        "--random-point-samples", "20000",
    )
    return report, out


@pytest.fixture(scope="module")
def at_gauge(
    corpus: Path, built: Path, tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict, Path]:
    out = tmp_path_factory.mktemp("ag-out")
    return _fit(corpus, built, out, "--baseline", "refit-v1"), out


class TestTheProtocolRefusals:
    def test_station_offsets_are_refused(self, corpus: Path, built: Path,
                                         tmp_path: Path) -> None:
        assert fit.main([
            "--run", str(built), "--corpus-dir", str(corpus / "corpus"),
            "--points", str(corpus / "points.json"),
            "--out-dir", str(tmp_path / "out"),
            "--leads", "30", "--resamples", "0",
            "--protocol", "random-point", "--station-offsets",
        ]) == 2

    def test_it_needs_the_coordinates(self, corpus: Path, built: Path,
                                      tmp_path: Path) -> None:
        assert fit.main([
            "--run", str(built), "--corpus-dir", str(corpus / "corpus"),
            "--out-dir", str(tmp_path / "out"),
            "--leads", "30", "--resamples", "0",
            "--protocol", "random-point",
        ]) == 2

    def test_a_typo_in_a_family_name_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="unknown feature famil"):
            fit.parse_families("gauge,neigbour")

    def test_the_candidate_can_drop_families_too(self) -> None:
        """``--drop-families`` is the same knob, aimed at ``--model``."""
        args = fit.build_parser().parse_args([
            "--run", "x", "--corpus-dir", "y", "--out-dir", "z",
            "--drop-families", "gauge,neighbour",
        ])
        assert fit.settings_from_args(args).drop_families == (
            "gauge", "neighbour",
        )
        assert fit.arm_label("logistic", ("gauge", "neighbour")) == (
            "logistic −gauge,neighbour"
        )

    def test_an_arm_carries_its_own_ablation(self) -> None:
        args = fit.build_parser().parse_args([
            "--run", "x", "--corpus-dir", "y", "--out-dir", "z",
        ])
        arms = fit.parse_compare(
            "trees,logistic/drop=gauge+neighbour", args,
        )
        assert [label for label, _s in arms] == [
            "trees", "logistic −gauge,neighbour",
        ]
        assert arms[1][1].drop_families == ("gauge", "neighbour")


class TestTheMasking:
    def test_the_own_gauge_columns_reach_the_fit_at_a_gauge(
        self, at_gauge,
    ) -> None:
        """The control: without the protocol the model leans on them."""
        _report, out = at_gauge
        model = pp.PostprocessModel.loads((out / "postprocess.json").read_text())
        names = list(model.feature_names)
        gauge = [
            index for index, name in enumerate(names)
            if pp.feature_family(name) == "gauge"
        ]
        assert gauge
        weight = max(
            abs(model.models[30].coefficients[index]) for index in gauge
        )
        assert weight > 1e-3

    def test_they_are_dead_under_the_protocol(self, random_point) -> None:
        """Masked in training and in scoring, so they can carry nothing.

        The mask makes every own-gauge column constant across the table;
        a constant column standardises to zero, so its coefficient can
        only stay at the zero it was initialised to. A non-zero weight
        here would mean the mask never reached the design.
        """
        _report, out = random_point
        model = pp.PostprocessModel.loads((out / "postprocess.json").read_text())
        names = list(model.feature_names)
        gauge = [
            index for index, name in enumerate(names)
            if pp.feature_family(name) == "gauge"
        ]
        assert gauge
        for lead in LEADS:
            for index in gauge:
                assert abs(model.models[lead].coefficients[index]) < 1e-9

    def test_the_artefact_says_which_protocol_fitted_it(
        self, random_point,
    ) -> None:
        """A masked model must not be shippable by accident."""
        _report, out = random_point
        model = pp.PostprocessModel.loads((out / "postprocess.json").read_text())
        assert model.training["protocol"] == "random-point"
        assert model.training["folds"]["kind"] == "month x station group"

    def test_the_neighbour_block_is_not_masked(self, random_point) -> None:
        """It was never the point's own gauge, so it survives — and works."""
        _report, out = random_point
        model = pp.PostprocessModel.loads((out / "postprocess.json").read_text())
        names = list(model.feature_names)
        neighbour = [
            index for index, name in enumerate(names)
            if pp.feature_family(name) == "neighbour"
        ]
        assert neighbour
        weight = max(
            abs(model.models[30].coefficients[index]) for index in neighbour
        )
        assert weight > 1e-3

    def test_the_dry_subset_survives_the_mask(self, random_point) -> None:
        """It is truth about the point, derived before the mask went on."""
        report, _out = random_point
        assert report["dry_rows"] > 0
        assert report["evaluation"]["leads"]["30"][pp.DRY]["n"] > 0


class TestTheFolds:
    def test_every_month_is_crossed_with_every_group(self, random_point) -> None:
        report, _out = random_point
        block = report["settings"]["folds"]
        assert block["kind"] == "month x station group"
        assert block["station_groups"] == 3
        assert block["n_folds"] == len(MONTHS) * 3
        assert block["unmatched_rows"] == 0
        assert sorted(
            sid for members in block["members"].values() for sid in members
        ) == sorted(STATIONS)
        labels = [entry["fold"] for entry in report["evaluation"]["folds"]]
        assert len(labels) == len(MONTHS) * 3
        assert all(" x g" in label for label in labels)

    def test_a_fold_trains_on_neither_its_month_nor_its_group(
        self, random_point,
    ) -> None:
        """The rows sharing exactly one axis belong to neither side.

        Under a month-only plan every row is either test or train, so
        ``n_train + n_test`` is the whole table. Here it cannot be: a row
        from the held-out month at a different station, and a row from the
        held-out group in a different month, are both excluded.
        """
        report, _out = random_point
        total = report["rows"]
        for entry in report["evaluation"]["folds"]:
            assert entry["n_train"] > 0
            assert entry["n_test"] > 0
            assert entry["n_train"] + entry["n_test"] < total

    def test_the_default_protocol_still_holds_out_only_the_month(
        self, at_gauge,
    ) -> None:
        report, _out = at_gauge
        assert report["settings"]["protocol"] == "at-gauge"
        assert report["settings"]["folds"] == {"kind": "month"}
        labels = [entry["fold"] for entry in report["evaluation"]["folds"]]
        assert labels == ["2026-01", "2026-04", "2026-06"]
        for entry in report["evaluation"]["folds"]:
            assert entry["n_train"] + entry["n_test"] == report["rows"]


class TestTheDistanceTables:
    def test_the_rows_are_binned_by_distance_to_the_nearest_gauge(
        self, random_point,
    ) -> None:
        report, _out = random_point
        names = report["settings"]["distance_bins"]
        assert names == list(pp.distance_bin_labels())
        entry = report["evaluation"]["leads"]["30"]
        filled = {name: entry.get(name) for name in names}
        # The fixture's stations sit 5, 15 and 28 km from their nearest
        # neighbour, so exactly three bins carry rows.
        assert filled["0-10 km"]["n"] > 0
        assert filled["10-20 km"]["n"] > 0
        assert filled["20-30 km"]["n"] > 0
        assert filled["30+ km"] is None
        total = sum(
            block["n"] for block in filled.values() if block
        )
        assert total == entry[pp.POOLED]["n"]

    def test_both_distributions_are_reported(self, random_point) -> None:
        report, _out = random_point
        block = report["distance_weights"]
        assert block["column"] == "ng_near_km"
        assert block["stations"] == len(STATIONS)
        for side in ("random_point", "at_gauge"):
            weights = block[side]["weights"]
            assert len(weights) == 4
            assert sum(weights) == pytest.approx(1.0)
        assert block["random_point"]["n"] == 20_000
        assert block["random_point"]["seed"] == 0
        # The archive's own rows are exactly the fixture's geometry.
        assert block["at_gauge"]["counts"] == [2, 2, 2, 0]

    def test_the_expectation_reweights_the_bins(self, random_point) -> None:
        report, _out = random_point
        rows = report["random_point"]
        assert [row["lead"] for row in rows] == list(LEADS)
        names = report["settings"]["distance_bins"]
        weights = dict(zip(
            report["distance_weights"]["random_point"]["labels"],
            report["distance_weights"]["random_point"]["weights"],
        ))
        for row in rows:
            assert 0.0 <= row["covered"] <= 1.0
            entry = report["evaluation"]["leads"][str(row["lead"])]
            by_bin = {
                name: (
                    None if not entry.get(name)
                    else entry[name]["postprocess"]["bss"]
                )
                for name in names
            }
            expected = pp.random_point_expectation(by_bin, weights)["value"]
            assert row["bss_candidate"] == pytest.approx(expected, abs=1e-6)
            # The paired bootstrap returns (point, lo, hi); the
            # re-weighted difference must be built from the POINT
            # estimate, not from an end of the interval.
            deltas = {
                name: (
                    None if not entry.get(name)
                    else ((entry[name].get("difference") or {}).get("bss")
                          or [None])[0]
                )
                for name in names
            }
            assert row["bss_difference"] == pytest.approx(
                pp.random_point_expectation(deltas, weights)["value"], abs=1e-6,
            )
            # ...and it is not the lower bound by accident.
            lows = {
                name: (
                    None if not entry.get(name)
                    else ((entry[name].get("difference") or {}).get("bss")
                          or [None, None])[1]
                )
                for name in names
            }
            assert row["bss_difference"] != pytest.approx(
                pp.random_point_expectation(lows, weights)["value"], abs=1e-9,
            )

    def test_the_markdown_carries_both_new_sections(self, random_point) -> None:
        _report, out = random_point
        text = (out / "postprocess_report.md").read_text()
        assert "## Candidate vs baseline — protocol `random-point`" in text
        assert "addresses they stand in for" in text
        assert "## By distance to the nearest gauge" in text
        assert "## Expected at a random point" in text
        assert "| 0-10 km |" in text
        assert "weight covered" in text

    def test_the_at_gauge_report_grows_neither(self, at_gauge) -> None:
        report, out = at_gauge
        assert report["distance_weights"] is None
        assert report["random_point"] == []
        text = (out / "postprocess_report.md").read_text()
        assert "## Candidate vs baseline — protocol `at-gauge`" in text
        assert "## Expected at a random point" not in text
        # ...but the distance bins are still a stratum, because ng_near_km
        # is in the rows and a reader of an at-gauge run wants it too.
        assert "## By distance to the nearest gauge" in text


class TestTheArms:
    def test_the_no_gauge_arm_is_scored_beside_the_candidate(
        self, random_point,
    ) -> None:
        report, _out = random_point
        arms = report["arms"]
        assert [arm["model"] for arm in arms] == ["logistic −gauge,neighbour"]
        assert arms[0]["fit"]["drop_families"] == ["gauge", "neighbour"]
        for lead in LEADS:
            block = arms[0]["leads"][str(lead)][pp.POOLED]
            mine = report["evaluation"]["leads"][str(lead)][pp.POOLED]
            # Both arms are scored on exactly the same rows.
            assert block["n"] == mine["n"]

    def test_the_neighbours_are_what_the_arms_differ_by(
        self, random_point,
    ) -> None:
        """The fixture's only source of timing is the gauge network.

        With the point's own gauge masked away, the neighbour block is the
        only thing left that knows WHEN the band arrives — so the arm that
        keeps it must beat the arm that drops it. If this ever fails, the
        ``ng_*`` columns are not reaching the design.
        """
        report, _out = random_point
        without = report["arms"][0]["leads"]
        for lead in LEADS:
            with_ng = report["evaluation"]["leads"][str(lead)][pp.POOLED]
            assert (
                with_ng["postprocess"]["bss"]
                > without[str(lead)][pp.POOLED]["postprocess"]["bss"] + 0.01
            ), lead

    def test_the_headline_says_which_protocol_produced_it(
        self, random_point, at_gauge,
    ) -> None:
        report, _out = random_point
        assert report["settings"]["protocol"] == "random-point"
        assert report["settings"]["arm"] == "logistic"
        assert report["settings"]["compare"] == ["logistic −gauge,neighbour"]
        assert fit._headline(report)["protocol"] == "random-point"
        assert fit._headline(report)["folds"] == "month x station group"
        assert fit._headline(report)["random_point"]

    def test_the_write_back_carries_the_random_point_predictions(
        self, corpus: Path, built: Path, tmp_path: Path,
    ) -> None:
        import pyarrow.parquet as pq

        back = tmp_path / "replay_rp"
        _fit(
            corpus, built, tmp_path / "wb-out",
            "--protocol", "random-point", "--station-groups", "3",
            "--resamples", "0", "--write-back", str(back),
        )
        table = pq.read_table(
            sorted((back / "decisions").glob("*.parquet"))[0],
        )
        for lead in LEADS:
            values = table.column(pp.post_column(lead)).to_pylist()
            assert all(v is not None and 0.0 <= v <= 1.0 for v in values)
        summary = json.loads((back / "summary.json").read_text())
        assert summary["postprocess"]["column_template"] == "p_post_{lead}"
