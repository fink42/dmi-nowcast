"""F3 — the historical warning replay (``scripts/replay_warnings.py``).

The script lives at the repo root but its dependency set is core **plus**
sidecar (``national_sample`` for the point read-out, ``push.engine`` for
the decision), so it is tested from this suite: the sidecar environment is
the only one that has both, and the VM runs it out of exactly that
environment.

Fully offline and fully synthetic. One 256×256 ODIM composite triple is
written into an archive-shaped directory and put through the real
pipeline — real Farnebäck flow, real vendored STEPS, real
``national_products``, real ``sample_point``, real ``evaluate`` — with the
ensemble shrunk to 3 members / 4 cascade levels / ×2 downsample and a
30-minute horizon, the same shrink ``tests/test_probabilistic.py`` uses to
keep an end-to-end STEPS run inside a CI budget.

Covered:
- the per-frame worker on one archived frame triple: a sample per station,
  the decision row schema, and rows that actually reach parquet;
- the day worker's state chain, and that each day starts armed;
- resumability: a day recorded ``done`` in the progress file is not redone;
- the pure CLI helpers (points file, rule parsing, frame listing).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest
from pyproj import CRS, Transformer

pytest.importorskip("pyarrow")

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import replay_warnings as rw  # noqa: E402  (after the sys.path edit)

DMI_PROJ = (
    "+proj=stere +lat_0=56 +lon_0=10.5666 +lat_ts=56 +ellps=WGS84 "
    "+units=m +no_defs"
)
CENTRE_LON, CENTRE_LAT = 10.32, 55.33
GRID_PX = 256
PIXEL_M = 500.0
GAIN, OFFSET = 0.5, -32.0
NODATA, UNDETECT = 255, 0
#: The four dev stations are far outside a 128 km synthetic grid, so the
#: fixture uses points near the grid centre instead. Ids keep the DMI
#: shape (five digits) so nothing downstream can depend on their being
#: special.
POINTS = {
    "version": 2,
    "points": [
        {"id": "06180", "lat": 55.33, "lon": 10.32, "region": "fyn"},
        {"id": "06120", "lat": 55.40, "lon": 10.45, "region": "fyn"},
    ],
}
DAY = "2026-09-05"
T_ANCHOR = datetime(2026, 9, 5, 6, 20, tzinfo=timezone.utc)

#: The tiny-STEPS settings used everywhere below.
TINY = rw.FrameSettings(
    frame_age_min=14.0,
    ensemble_size=3,
    n_cascade_levels=4,
    downsample_factor=2,
    horizon_min=30,
    leads_min=(10, 20, 30),
    threshold_mm_h=rw.RAIN_THRESHOLD_MM_H,
)


# ---------------------------------------------------------------------------
# Synthetic archive
# ---------------------------------------------------------------------------


def _corners_lonlat() -> dict[str, tuple[float, float]]:
    crs = CRS.from_proj4(DMI_PROJ)
    to_proj = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x, y = to_proj.transform(CENTRE_LON, CENTRE_LAT)
    half = GRID_PX / 2 * PIXEL_M
    return {
        "UL": to_wgs.transform(x - half, y + half),
        "UR": to_wgs.transform(x + half, y + half),
        "LL": to_wgs.transform(x - half, y - half),
        "LR": to_wgs.transform(x + half, y - half),
    }


def _textured_dbz(shift: int) -> np.ndarray:
    """A textured rain field, rolled by ``shift`` rows.

    Uniform fields give STEPS nothing to decompose; the cascade wants
    structure. Same construction as ``tests/test_probabilistic.py``'s
    end-to-end smoke, at a fixed seed so the fixture is deterministic.
    """
    rng = np.random.default_rng(7)
    yy, xx = np.indices((GRID_PX, GRID_PX))
    rain = np.zeros((GRID_PX, GRID_PX), dtype=np.float32)
    for _ in range(18):
        cy, cx = rng.uniform(20, GRID_PX - 20), rng.uniform(20, GRID_PX - 20)
        sy, sx = rng.uniform(10, 30), rng.uniform(10, 30)
        rain += (rng.uniform(2.0, 12.0) * np.exp(
            -((yy - cy) ** 2 / (2 * sy ** 2) + (xx - cx) ** 2 / (2 * sx ** 2))
        )).astype(np.float32)
    rain = np.roll(rain, shift, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dbz = 10.0 * np.log10(200.0) + 16.0 * np.log10(np.maximum(rain, 1e-3))
    return np.clip(dbz, -30.0, 60.0).astype(np.float32)


def _write_composite(path: Path, ts: datetime, dbz: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.clip(
        np.round((dbz - OFFSET) / GAIN), UNDETECT + 1, NODATA - 1,
    ).astype(np.uint8)
    with h5py.File(path, "w") as h5:
        what = h5.create_group("what")
        what.attrs["gain"] = GAIN
        what.attrs["offset"] = OFFSET
        what.attrs["nodata"] = NODATA
        what.attrs["undetect"] = UNDETECT
        what.attrs["date"] = ts.strftime("%Y%m%d").encode()
        what.attrs["time"] = ts.strftime("%H%M%S").encode()
        what.attrs["product"] = b"DBZH"
        where = h5.create_group("where")
        where.attrs["projdef"] = DMI_PROJ.encode()
        where.attrs["xscale"] = PIXEL_M
        where.attrs["yscale"] = PIXEL_M
        for name, (lon, lat) in _corners_lonlat().items():
            where.attrs[f"{name}_lon"] = lon
            where.attrs[f"{name}_lat"] = lat
        how = h5.create_group("how")
        how.attrs["zr-a"] = 200.0
        how.attrs["zr-b"] = 1.6
        h5.create_group("dataset1").create_group("data1").create_dataset(
            "data", data=raw,
        )


@pytest.fixture(scope="module")
def archive_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Four consecutive fullRange frames — enough for two replayable ones."""
    root = tmp_path_factory.mktemp("archive")
    for i in range(4):
        ts = T_ANCHOR - timedelta(minutes=10 * (3 - i))
        _write_composite(
            rw.frame_path(root, ts), ts, _textured_dbz(shift=2 * (3 - i)),
        )
    return root


@pytest.fixture
def points_file(tmp_path: Path) -> Path:
    path = tmp_path / "station_points.json"
    path.write_text(json.dumps(POINTS))
    return path


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def test_load_points_reads_the_v2_schema(points_file: Path) -> None:
    points = rw.load_points(points_file)
    assert [p.id for p in points] == ["06180", "06120"]
    assert points[0].region == "fyn"


def test_load_points_rejects_another_version(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"version": 1, "points": POINTS["points"]}))
    with pytest.raises(ValueError, match="unsupported points version"):
        rw.load_points(path)


def test_load_points_rejects_a_duplicate_station(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps({
        "version": 2, "points": [POINTS["points"][0], POINTS["points"][0]],
    }))
    with pytest.raises(ValueError, match="duplicate station id"):
        rw.load_points(path)


def test_parse_rules_defaults_to_the_live_subscriber_row() -> None:
    assert rw.parse_rules(None) == rw.DEFAULT_RULES
    rules = rw.parse_rules("threshold_pct=60,persistence_obs=2")
    assert rules["threshold_pct"] == 60
    assert rules["persistence_obs"] == 2
    assert rules["rearm_after_min"] == 60      # untouched keys keep the default


def test_parse_rules_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="unknown rule"):
        rw.parse_rules("thresold_pct=40")
    with pytest.raises(ValueError, match="threshold_pct"):
        rw.parse_rules("threshold_pct=140")
    with pytest.raises(ValueError, match="served leads"):
        rw.parse_rules("lead_min=35")


def test_full_range_frames_lists_only_what_is_on_disk(archive_dir: Path) -> None:
    frames = rw.full_range_frames(archive_dir, datetime(2026, 9, 5).date())
    assert frames == [
        T_ANCHOR - timedelta(minutes=10 * i) for i in (3, 2, 1, 0)
    ]
    assert rw.full_range_frames(archive_dir, datetime(2026, 9, 4).date()) == []


# ---------------------------------------------------------------------------
# The per-frame worker — real STEPS, shrunk
# ---------------------------------------------------------------------------


def test_sample_frame_runs_the_pipeline_and_samples_every_station(
    archive_dir: Path,
) -> None:
    cache = rw.CompositeCache(archive_dir)
    samples = rw.sample_frame(
        cache, T_ANCHOR,
        [rw.StationPoint(**p) for p in POINTS["points"]],
        TINY,
    )
    assert [s["station_id"] for s in samples] == ["06180", "06120"]
    for sample in samples:
        assert sample["radar_ts"] == T_ANCHOR
        # generated_at is radar time plus the simulated compute latency —
        # the wall clock the decision would have run on.
        assert sample["generated_at"] == T_ANCHOR + timedelta(minutes=14)
        assert set(sample["p_rain"]) == set(TINY.leads_min)
        for value in sample["p_rain"].values():
            assert value is None or 0.0 <= value <= 1.0
        # The synthetic field is soaking wet, so the observation and the
        # lead-0 forecast must both read rain at these central points.
        assert sample["observed_mm_h"] is not None
        assert sample["forecast_now_mm_h"] is not None


def test_sample_frame_refuses_frames_off_the_ten_minute_grid(
    tmp_path: Path,
) -> None:
    """A file whose own timestamp is off the grid is not an input triple.

    The filename says 06:10; the composite inside says 06:05. STEPS' AR(2)
    model assumes the forecast timestep equals the input frame spacing, so
    a 5/15-minute pair must be refused rather than silently averaged.
    """
    root = tmp_path / "skewed"
    for offset, internal in ((20, 20), (10, 15), (0, 0)):
        path_ts = T_ANCHOR - timedelta(minutes=offset)
        _write_composite(
            rw.frame_path(root, path_ts),
            T_ANCHOR - timedelta(minutes=internal),
            _textured_dbz(0),
        )
    with pytest.raises(RuntimeError, match="10-min grid"):
        rw.sample_frame(
            rw.CompositeCache(root), T_ANCHOR,
            [rw.StationPoint("06180", 55.33, 10.32)], TINY,
        )


def test_sample_frame_needs_its_predecessors(
    archive_dir: Path,
) -> None:
    # The oldest archived frame has no T-10/T-20 behind it; the worker
    # raises and the day records the frame as an error instead of
    # inventing a dry station.
    with pytest.raises(OSError):
        rw.sample_frame(
            rw.CompositeCache(archive_dir), T_ANCHOR - timedelta(minutes=30),
            [rw.StationPoint("06180", 55.33, 10.32)], TINY,
        )


# ---------------------------------------------------------------------------
# The day worker
# ---------------------------------------------------------------------------


def _day_args(archive_dir: Path, out_dir: Path, day: str = DAY, **over) -> tuple:
    rules = dict(rw.DEFAULT_RULES)
    rules["lead_min"] = 30
    rules.update(over.pop("rules", {}))
    return (
        str(archive_dir), day,
        tuple(rw.StationPoint(**p) for p in POINTS["points"]),
        over.pop("settings", TINY), rules, str(out_dir),
        over.pop("start_min", 0), over.pop("end_min", 24 * 60),
    )


@pytest.fixture(scope="module")
def replayed_day(archive_dir: Path, tmp_path_factory: pytest.TempPathFactory):
    """One replayed day, shared by the assertions below (STEPS is not free)."""
    out_dir = tmp_path_factory.mktemp("out")
    result = rw.run_day(_day_args(archive_dir, out_dir))
    return result, out_dir


def test_run_day_writes_a_row_per_station_per_frame(replayed_day) -> None:
    result, out_dir = replayed_day
    assert result["failed"] is False
    # The archive holds 05:50-06:20. Only 06:10 and 06:20 have the two
    # predecessors the ensemble needs; the two oldest frames are recorded
    # as per-frame errors and the day carries on.
    assert result["frames"] == 2
    assert len(result["errors"]) == 2
    assert result["rows"] == result["frames"] * len(POINTS["points"])
    assert len(result["frame_ms"]) == result["frames"]

    rows = rw.read_decisions(out_dir / "decisions" / f"{DAY}.parquet")
    assert len(rows) == result["rows"]
    assert set(rows[0]) == set(rw.decision_schema().names)
    assert {r["station_id"] for r in rows} == {"06180", "06120"}
    assert all(r["radar_ts"].tzinfo is not None for r in rows)
    assert all(
        r["generated_at"] - r["radar_ts"] == timedelta(minutes=14) for r in rows
    )
    assert all(
        r["action"] in {"none", "notify", "already_raining", "deferred_quiet"}
        for r in rows
    )
    assert all(isinstance(r["armed_after"], bool) for r in rows)


def test_run_day_carries_state_across_frames(replayed_day) -> None:
    _, out_dir = replayed_day
    rows = sorted(
        rw.read_decisions(out_dir / "decisions" / f"{DAY}.parquet"),
        key=lambda r: (r["station_id"], r["radar_ts"]),
    )
    for station in ("06120", "06180"):
        seq = [r for r in rows if r["station_id"] == station]
        # The fixture is soaked, so the probability must be over the 40 %
        # threshold — if this fails the rest of the assertions are vacuous.
        assert seq[0]["p_rain"] is not None and seq[0]["p_rain"] >= 0.4
        # The field is wet everywhere, so with persistence_obs=1 the first
        # frame consumes the arm (a notify, or "already raining" because
        # the gauge point is under the shower) and the second cannot fire
        # again 10 minutes later — the 60-min re-arm has not elapsed.
        assert seq[0]["action"] in {"notify", "already_raining"}
        assert seq[0]["armed_after"] is False
        assert seq[1]["action"] == "none"
        assert seq[1]["armed_after"] is False


def test_run_day_reports_the_end_state_for_the_progress_file(replayed_day) -> None:
    result, _ = replayed_day
    assert set(result["state"]) == {"06180", "06120"}
    for state in result["state"].values():
        assert set(state) == {
            "armed", "streak", "below_since_utc", "last_eval_radar_ts",
        }
        assert state["last_eval_radar_ts"].startswith("2026-09-05T06:20")
    # Round-trip: the JSON state rebuilds into the engine's own dataclass.
    restored = rw.state_from_json(result["state"]["06180"])
    assert restored.armed is False
    assert restored.last_eval_radar_ts == T_ANCHOR


def test_each_day_starts_armed(archive_dir: Path, tmp_path: Path) -> None:
    """The day-parallel simplification, asserted rather than assumed."""
    one_frame = {"start_min": 6 * 60 + 20, "end_min": 6 * 60 + 20}
    first = rw.run_day(_day_args(archive_dir, tmp_path, **one_frame))
    assert first["state"]["06180"]["armed"] is False
    # A second run of the same day is independent of the first: no state is
    # threaded between day workers, so it reproduces the same sequence.
    again = rw.run_day(_day_args(archive_dir, tmp_path, **one_frame))
    assert again["state"] == first["state"]
    assert again["rows"] == first["rows"]


def test_run_day_clips_to_the_requested_window(
    archive_dir: Path, tmp_path: Path,
) -> None:
    result = rw.run_day(
        _day_args(archive_dir, tmp_path, start_min=6 * 60 + 20, end_min=6 * 60 + 20)
    )
    assert result["frames"] == 1
    rows = rw.read_decisions(tmp_path / "decisions" / f"{DAY}.parquet")
    assert {r["radar_ts"] for r in rows} == {T_ANCHOR}


def test_run_day_survives_a_missing_archive(tmp_path: Path) -> None:
    result = rw.run_day(_day_args(tmp_path / "nothing", tmp_path))
    assert result["frames"] == 0
    assert result["rows"] == 0
    assert result["failed"] is False        # an empty day is not a failure


# ---------------------------------------------------------------------------
# Parquet + resumability
# ---------------------------------------------------------------------------


def test_decision_parquet_round_trips_nulls(tmp_path: Path) -> None:
    path = tmp_path / "decisions" / "2026-01-01.parquet"
    rw.write_decisions(path, [{
        "radar_ts": T_ANCHOR,
        "generated_at": T_ANCHOR + timedelta(minutes=14),
        "station_id": "06180",
        "p_rain": None,                     # off coverage — never a dry 0.0
        "eta_min": None,
        "intensity_mm_h": None,
        "observed_mm_h": None,
        "forecast_now_mm_h": None,
        "action": "none",
        "armed_after": True,
        "streak_after": 0,
    }])
    rows = rw.read_decisions(path)
    assert rows[0]["p_rain"] is None
    assert rows[0]["radar_ts"] == T_ANCHOR
    assert rows[0]["armed_after"] is True


def test_progress_file_makes_a_finished_day_skippable(
    archive_dir: Path, points_file: Path, tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    progress = tmp_path / "progress.json"
    argv = [
        "--archive-dir", str(archive_dir),
        "--corpus-dir", str(tmp_path / "corpus"),
        "--points", str(points_file),
        "--days", DAY,
        "--out-dir", str(out_dir),
        "--progress", str(progress),
        "--ensemble-size", "3", "--cascade-levels", "4",
        "--downsample-factor", "2", "--horizon-min", "30",
        "--start-utc", "06:20", "--end-utc", "06:20",
        "--no-score",
    ]
    assert rw.main(argv) == 0
    saved = json.loads(progress.read_text())
    assert saved["days"][DAY]["status"] == "done"
    assert saved["days"][DAY]["frames"] == 1
    assert saved["days"][DAY]["state"]["06180"]["armed"] is False
    written = (out_dir / "decisions" / f"{DAY}.parquet").stat().st_mtime_ns

    # Second run: the day is already done, so nothing is recomputed and the
    # parquet is left exactly as it was.
    assert rw.main(argv) == 0
    assert (out_dir / "decisions" / f"{DAY}.parquet").stat().st_mtime_ns == written
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["run"]["n_frames"] == 1
    assert summary["run"]["n_decision_rows"] == 2
    assert summary["gauge"] == {"available": False, "reason": "--no-score"}


def test_summary_reports_the_missing_store_rather_than_crashing(
    archive_dir: Path, points_file: Path, tmp_path: Path,
) -> None:
    """An empty corpus is a normal state early in Phase F, not an error."""
    out_dir = tmp_path / "out"
    assert rw.main([
        "--archive-dir", str(archive_dir),
        "--corpus-dir", str(tmp_path / "empty-corpus"),
        "--points", str(points_file),
        "--days", DAY,
        "--out-dir", str(out_dir),
        "--ensemble-size", "3", "--cascade-levels", "4",
        "--downsample-factor", "2", "--horizon-min", "30",
        "--start-utc", "06:20", "--end-utc", "06:20",
    ]) == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["gauge"]["available"] is False
    assert "gauge observations" in summary["gauge"]["reason"] or (
        "station store" in summary["gauge"]["reason"]
    )


# ---------------------------------------------------------------------------
# Scoring against a stubbed store
# ---------------------------------------------------------------------------


class FakeStore:
    """Stands in for ``StationObsStore``: the one method the replay calls."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.calls: list[tuple] = []

    def read(self, start_utc, end_utc, parameter_ids, station_ids=None):
        self.calls.append((start_utc, end_utc, tuple(parameter_ids)))
        return [
            r for r in self.rows
            if start_utc <= r["observed_utc"] <= end_utc
            and (station_ids is None or r["station_id"] in station_ids)
        ]


def test_day_slots_pads_the_day_and_builds_a_contiguous_grid() -> None:
    rows = [{
        "station_id": "06180",
        "observed_utc": datetime(2026, 9, 5, 6, 20, tzinfo=timezone.utc),
        "parameter_id": rw.PRECIP_PARAM,
        "value": 1.0,
    }]
    store = FakeStore(rows)
    slots = rw.day_slots(store, datetime(2026, 9, 5).date(), ["06180"])
    start, end, params = store.calls[0]
    assert start == datetime(2026, 9, 4, 22, 0, tzinfo=timezone.utc)
    assert end == datetime(2026, 9, 6, 2, 0, tzinfo=timezone.utc)
    assert params == (rw.PRECIP_PARAM, rw.PRECIP_DUR_PARAM)
    grid = slots["06180"]
    assert len(grid) == 28 * 6 + 1          # 28 h of 10-min slots, inclusive
    wet = [ts for ts, w in grid if w]
    assert wet == [datetime(2026, 9, 5, 6, 20, tzinfo=timezone.utc)]


def test_score_matches_a_replayed_warning_to_a_gauge_onset() -> None:
    day = datetime(2026, 9, 5).date()
    base = datetime(2026, 9, 5, 6, 0, tzinfo=timezone.utc)
    # Dry from 05:30, onset at 06:40 — 20 min after a warning sent at 06:20.
    rows = []
    for i in range(12):
        ts = base - timedelta(minutes=30) + timedelta(minutes=10 * i)
        wet = ts >= base + timedelta(minutes=40)
        rows.append({
            "station_id": "06180",
            "observed_utc": ts,
            "parameter_id": rw.PRECIP_PARAM,
            "value": 1.2 if wet else 0.0,
        })
    windows = [rw.day_slots(FakeStore(rows), day, ["06180"])]
    decisions = [{
        "radar_ts": base + timedelta(minutes=6),
        "generated_at": base + timedelta(minutes=20),
        "station_id": "06180",
        "p_rain": 0.8, "eta_min": 25.0, "intensity_mm_h": 1.0,
        "observed_mm_h": 0.0, "forecast_now_mm_h": 0.0,
        "action": "notify", "armed_after": False, "streak_after": 1,
    }]
    results, slot_lists, agreement = rw.score(
        decisions, windows, [rw.StationPoint("06180", 55.33, 10.32)],
        lead_min=30, tolerance_min=10, dry_min=30, threshold_mm_h=0.5,
    )
    summary = results["06180"].summary
    assert summary["hits"] == 1
    assert summary["false_alarms"] == 0
    assert summary["misses"] == 0
    # onset 06:40 − sent 06:20 = 20 min of delivered lead against a 25 min
    # ETA: the rain came 5 min later than promised, i.e. the warning was
    # early → +5 under the eta − actual convention.
    assert summary["lead_error_min"]["p50"] == pytest.approx(5.0)
    # The decision's own cycle sits in the 06:20 slot, which was dry.
    assert agreement["n_scored"] == 1
    assert agreement["observed"]["correct_negatives"] == 1
    assert slot_lists["06180"]
