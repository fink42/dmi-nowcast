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


def _write_composite(
    path: Path,
    ts: datetime,
    dbz: np.ndarray,
    coverage: np.ndarray | None = None,
) -> None:
    """Write one ODIM composite; ``coverage=False`` pixels become ``nodata``.

    The coverage mask is how the doppler product is modelled: same grid,
    same scaling, but blank beyond its 120 km range.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.clip(
        np.round((dbz - OFFSET) / GAIN), UNDETECT + 1, NODATA - 1,
    ).astype(np.uint8)
    if coverage is not None:
        raw[~coverage] = NODATA
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

    rows = rw.read_decisions(
        out_dir / "decisions" / f"{DAY}.parquet", TINY.leads_min,
    )
    assert len(rows) == result["rows"]
    assert set(rows[0]) == set(rw.decision_schema(TINY.leads_min).names)
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
    wet = [(ts, mm) for ts, w, mm in grid if w]
    # The depth rides along with the flag: the onset rule needs it.
    assert wet == [(datetime(2026, 9, 5, 6, 20, tzinfo=timezone.utc), 1.0)]


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


# ---------------------------------------------------------------------------
# Per-lead probability columns
# ---------------------------------------------------------------------------


def test_the_replay_writes_one_probability_column_per_served_lead(
    replayed_day,
) -> None:
    """The offline threshold sweep must not need a second STEPS run."""
    from dmi_nowcast_core.warning_score import decision_columns

    _, out_dir = replayed_day
    rows = rw.read_decisions(
        out_dir / "decisions" / f"{DAY}.parquet", TINY.leads_min,
    )
    assert set(rows[0]) == set(decision_columns(TINY.leads_min))
    for row in rows:
        # The rule's lead is 30, so p_rain must equal the p_rain_30 column —
        # the sweep and the decision read the same number.
        assert row["p_rain"] == pytest.approx(row["p_rain_30"])
        for lead in TINY.leads_min:
            value = row[f"p_rain_{lead}"]
            assert value is None or 0.0 <= value <= 1.0
    # A soaked fixture: at least one lead carries a real probability.
    assert any(r["p_rain_10"] is not None for r in rows)


def test_read_decisions_tolerates_a_file_without_the_per_lead_columns(
    tmp_path: Path,
) -> None:
    """Days replayed before the columns existed must still score."""
    from dmi_nowcast_core.warning_score import DECISION_COLUMNS, decision_table

    path = tmp_path / "old.parquet"
    row = {
        "radar_ts": T_ANCHOR,
        "generated_at": T_ANCHOR + timedelta(minutes=14),
        "station_id": "06180",
        "p_rain": 0.8,
        "eta_min": 25.0,
        "intensity_mm_h": 1.0,
        "observed_mm_h": 0.0,
        "forecast_now_mm_h": 0.0,
        "action": "notify",
        "armed_after": False,
        "streak_after": 1,
    }
    old = decision_table([row], leads_min=())
    assert tuple(old.schema.names) == DECISION_COLUMNS
    rw._write_table_atomic(old, path)

    rows = rw.read_decisions(path)
    assert rows[0]["p_rain"] == pytest.approx(0.8)   # the old column survives
    assert rows[0]["p_rain_30"] is None              # the new one reads unknown


# ---------------------------------------------------------------------------
# The anchor policy (Phase H, H-L / L3)
# ---------------------------------------------------------------------------

#: doppler's 120 km edge in miniature: a disc around the grid centre.
DOPPLER_RADIUS_PX = 90.0
#: A day of both products around the module's anchor instant.
DUAL_START = T_ANCHOR - timedelta(minutes=40)


def _doppler_coverage() -> np.ndarray:
    rows, cols = np.indices((GRID_PX, GRID_PX))
    centre = (GRID_PX - 1) / 2.0
    return (
        (rows - centre) ** 2 + (cols - centre) ** 2
    ) <= DOPPLER_RADIUS_PX ** 2


def _harmonisation_payload(shift_db: float = 2.0) -> dict:
    """A constant ``+shift_db`` map, in every band and season.

    The real table is fitted per distance band and season (L2,
    2026-09-08); ``tests/test_doppler_harmonisation.py`` pins the fit. All
    this one has to do is be a real, applicable table.
    """
    from dmi_nowcast_core import product_pairs as pp

    edges = pp.mapped_edges()
    entry = {
        "mapped_dbz": [float(e + shift_db) for e in edges],
        "n_doppler_px": 1_000_000,
        "n_fullrange_px": 1_000_000,
    }
    return {
        "schema_version": pp.HARMONISATION_SCHEMA_VERSION,
        "meta": {"generated": "2026-09-08T18:44:53+00:00", "n_fit_triples": 1417},
        "bin_edges_dbz": [float(e) for e in edges],
        "tables": {
            pp.table_key(season, band): dict(entry)
            for season in pp.SEASON_ORDER
            for band in pp.DOPPLER_BANDS
        },
    }


@pytest.fixture(scope="module")
def dual_archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Both products on the 5-minute grid, 05:40 → 06:25.

    fullRange on ``:x0`` over the whole grid; doppler on ``:x5`` inside a
    disc and ``nodata`` outside it, so the fill really is doing something.
    """
    root = tmp_path_factory.mktemp("dual-archive")
    coverage = _doppler_coverage()
    for i in range(10):
        ts = DUAL_START + timedelta(minutes=5 * i)
        full = ts.minute % 10 == 0
        _write_composite(
            rw.frame_path(root, ts), ts,
            _textured_dbz(shift=2 * i),
            None if full else coverage,
        )
    return root


@pytest.fixture(scope="module")
def harmonisation_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("harm") / "doppler_harmonisation.json"
    path.write_text(json.dumps(_harmonisation_payload()))
    return path


def _anchor_settings(**over) -> "rw.AnchorSettings":
    base = {"frame_age_override_min": None}
    base.update(over)
    return rw.AnchorSettings(**base)


def test_plan_day_puts_both_policies_on_the_same_instants(
    dual_archive: Path, harmonisation_file: Path,
) -> None:
    """The candidate decides when the baseline does, five minutes fresher."""
    day = datetime(2026, 9, 5).date()
    base = rw.plan_day(
        dual_archive, day,
        rw.dc_replace(TINY, anchor=_anchor_settings()),
    )
    fresh = rw.plan_day(
        dual_archive, day,
        rw.dc_replace(TINY, anchor=_anchor_settings(
            policy="freshest", harmonisation_path=str(harmonisation_file),
        )),
    )
    assert [s.now for s, _h in base] == [s.now for s, _h in fresh]
    assert [s.timestamp.strftime("%H:%M") for s, _h in base] == [
        "05:40", "05:50", "06:00", "06:10", "06:20",
    ]
    assert [s.timestamp.strftime("%H:%M") for s, _h in fresh] == [
        "05:45", "05:55", "06:05", "06:15", "06:25",
    ]
    assert all(s.frame_age_min == pytest.approx(15.0) for s, _h in base)
    assert all(s.frame_age_min == pytest.approx(10.0) for s, _h in fresh)
    # Only the last two of each have a complete triple behind them.
    assert [h is not None for _s, h in base] == [False, False, True, True, True]


def test_plan_day_clips_on_the_anchor_frame(dual_archive: Path) -> None:
    day = datetime(2026, 9, 5).date()
    plan = rw.plan_day(
        dual_archive, day, rw.dc_replace(TINY, anchor=_anchor_settings()),
        start_min=6 * 60 + 20, end_min=6 * 60 + 20,
    )
    assert [s.timestamp for s, _h in plan] == [T_ANCHOR]
    # The decision itself runs 15 minutes after the frame, not at it.
    assert plan[0][0].now == T_ANCHOR + timedelta(minutes=15)


def _cli(archive: Path, out_dir: Path, points_file: Path, *extra: str) -> list[str]:
    return [
        "--archive-dir", str(archive),
        "--corpus-dir", str(out_dir / "corpus"),
        "--points", str(points_file),
        "--days", DAY,
        "--out-dir", str(out_dir),
        "--ensemble-size", "3", "--cascade-levels", "4",
        "--downsample-factor", "2", "--horizon-min", "30",
        "--no-score",
        *extra,
    ]


def test_the_default_run_records_a_fullrange_anchor(
    dual_archive: Path, points_file: Path, tmp_path: Path,
) -> None:
    out_dir = tmp_path / "base"
    assert rw.main(_cli(
        dual_archive, out_dir, points_file,
        "--start-utc", "06:20", "--end-utc", "06:20",
    )) == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    anchor = summary["run"]["anchor"]
    assert anchor["policy"] == "fullRange"
    assert anchor["harmonisation"] == {"path": None}
    assert anchor["counts"]["fullrange_anchored"] == 1
    assert anchor["counts"]["doppler_anchored"] == 0
    assert anchor["counts"]["degraded"] == 0
    assert anchor["frame_age_min"]["p50"] == pytest.approx(15.0)
    # The flat model is off, so the parity key is null rather than a lie.
    assert summary["run"]["frame_age_min"] is None

    rows = rw.read_decisions(out_dir / "decisions" / f"{DAY}.parquet")
    assert {r["radar_ts"] for r in rows} == {T_ANCHOR}
    assert all(
        r["generated_at"] - r["radar_ts"] == timedelta(minutes=15) for r in rows
    )


def test_the_freshest_anchor_stands_on_the_doppler_frame(
    dual_archive: Path, points_file: Path, harmonisation_file: Path,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "fresh"
    assert rw.main(_cli(
        dual_archive, out_dir, points_file,
        "--anchor", "freshest",
        "--harmonisation", str(harmonisation_file),
        "--start-utc", "06:15", "--end-utc", "06:15",
    )) == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    anchor = summary["run"]["anchor"]
    assert anchor["policy"] == "freshest"
    assert anchor["history_mode"] == "same-type"
    assert anchor["lag_min"] == {"fullRange": 13.1, "doppler": 8.1, "flat": None}
    assert anchor["counts"]["doppler_anchored"] == 1
    assert anchor["counts"]["fullrange_anchored"] == 0
    assert anchor["counts"]["history_fallback"] == 0
    assert anchor["frame_age_min"]["p50"] == pytest.approx(10.0)
    stamp = anchor["harmonisation"]
    assert stamp["schema_version"] == 1
    assert stamp["fitted_at"] == "2026-09-08T18:44:53+00:00"
    assert len(stamp["sha256"]) == 16

    rows = rw.read_decisions(out_dir / "decisions" / f"{DAY}.parquet")
    # The anchor is the :x5 frame — a minute the fullRange arm never
    # stamps, so a baseline and a candidate can share one directory
    # without their rows deduplicating each other away.
    assert {r["radar_ts"] for r in rows} == {
        T_ANCHOR - timedelta(minutes=5),
    }
    assert all(
        r["generated_at"] - r["radar_ts"] == timedelta(minutes=10) for r in rows
    )
    # The stations sit at the grid centre, inside doppler's disc, and the
    # fixture is soaked: the anchor field reads rain there.
    assert all(r["observed_mm_h"] is not None for r in rows)


def test_the_filled_history_keeps_the_grid_outside_doppler_coverage(
    dual_archive: Path, points_file: Path, harmonisation_file: Path,
    tmp_path: Path,
) -> None:
    """``--anchor-history filled`` is the variant that keeps national cover.

    Under ``same-type`` the cascade only ever sees doppler's disc, so
    ``p_rain`` is undefined outside it; under ``filled`` every history
    frame is itself a per-pixel freshest composite and the ensemble spans
    the fullRange domain again.
    """
    out_dir = tmp_path / "filled"
    assert rw.main(_cli(
        dual_archive, out_dir, points_file,
        "--anchor", "freshest",
        "--harmonisation", str(harmonisation_file),
        "--anchor-history", "filled",
        "--start-utc", "06:15", "--end-utc", "06:15",
    )) == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["run"]["anchor"]["history_mode"] == "filled"
    assert summary["run"]["anchor"]["counts"]["doppler_anchored"] == 1
    rows = rw.read_decisions(out_dir / "decisions" / f"{DAY}.parquet")
    assert rows and all(r["p_rain"] is not None for r in rows)


def test_september_has_no_doppler_and_the_candidate_degrades(
    archive_dir: Path, points_file: Path, harmonisation_file: Path,
    tmp_path: Path,
) -> None:
    """The 2026-09 archive is fullRange only; the run must cope and say so."""
    out_dir = tmp_path / "sept"
    assert rw.main(_cli(
        archive_dir, out_dir, points_file,
        "--anchor", "freshest",
        "--harmonisation", str(harmonisation_file),
        "--start-utc", "06:20", "--end-utc", "06:20",
    )) == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    counts = summary["run"]["anchor"]["counts"]
    assert counts["doppler_anchored"] == 0
    assert counts["fullrange_anchored"] == 1
    assert counts["degraded"] == 1          # the candidate ran as the baseline
    rows = rw.read_decisions(out_dir / "decisions" / f"{DAY}.parquet")
    assert {r["radar_ts"] for r in rows} == {T_ANCHOR}


def test_the_flat_frame_age_reproduces_a_pre_l3_run(
    dual_archive: Path, points_file: Path, tmp_path: Path,
) -> None:
    out_dir = tmp_path / "flat"
    assert rw.main(_cli(
        dual_archive, out_dir, points_file,
        "--frame-age-min", "14",
        "--start-utc", "06:20", "--end-utc", "06:20",
    )) == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["run"]["frame_age_min"] == 14.0
    assert summary["run"]["anchor"]["frame_age_override_min"] == 14.0
    assert summary["run"]["anchor"]["poll_interval_min"] == 0.0
    rows = rw.read_decisions(out_dir / "decisions" / f"{DAY}.parquet")
    assert all(
        r["generated_at"] - r["radar_ts"] == timedelta(minutes=14) for r in rows
    )


def test_freshest_without_a_map_is_refused(
    dual_archive: Path, points_file: Path, tmp_path: Path,
) -> None:
    """Raw doppler carries 20-30 %% less echo at 20 dBZ. Never unmapped."""
    with pytest.raises(SystemExit) as excinfo:
        rw.main(_cli(
            dual_archive, tmp_path / "nope", points_file, "--anchor", "freshest",
        ))
    assert excinfo.value.code == 2


def test_an_unreadable_map_fails_before_the_first_frame(
    dual_archive: Path, points_file: Path, tmp_path: Path,
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 99, "tables": {"a": {}}}))
    with pytest.raises(SystemExit) as excinfo:
        rw.main(_cli(
            dual_archive, tmp_path / "nope", points_file,
            "--anchor", "freshest", "--harmonisation", str(bad),
        ))
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# H-P feature columns (Phase H)
# ---------------------------------------------------------------------------


def _feature_table(out_dir: Path):
    import pyarrow.parquet as pq

    return pq.read_table(out_dir / "decisions" / f"{DAY}.parquet")


@pytest.fixture(scope="module")
def featured_day(archive_dir: Path, tmp_path_factory: pytest.TempPathFactory):
    """One replayed day with --features on (the default). STEPS is not free."""
    out_dir = tmp_path_factory.mktemp("features")
    result = rw.run_day(_day_args(archive_dir, out_dir))
    assert result["failed"] is False
    return _feature_table(out_dir), out_dir


class TestFeatureColumns:
    """The H-P predictors, written beside the decision they were taken on."""

    def test_every_documented_column_is_written(self, featured_day) -> None:
        table, _out_dir = featured_day
        expected = list(rw.feature_schema(TINY.leads_min).names)
        assert set(expected) <= set(table.schema.names)
        # The shared schema comes first and is untouched.
        shared = list(rw.decision_schema(TINY.leads_min).names)
        assert table.schema.names[:len(shared)] == shared

    def test_the_documentation_covers_exactly_those_columns(self) -> None:
        docs = rw.feature_documentation(TINY.leads_min)
        assert set(docs) == set(rw.feature_schema(TINY.leads_min).names)
        assert all(text.strip() for text in docs.values())

    def test_the_values_are_in_range(self, featured_day) -> None:
        table, _out_dir = featured_day
        for row in table.to_pylist():
            for lead in TINY.leads_min:
                raw = row[f"raw_frac_{lead}"]
                assert raw is None or 0.0 <= raw <= 1.0
            assert row["obs_max_5km_mm_h"] >= 0.0
            # The disc maximum cannot be below the block p90 at its centre.
            assert row["obs_max_5km_mm_h"] >= row["observed_mm_h"] - 1e-3
            assert row["stalled_share"] is None or 0.0 <= row["stalled_share"] <= 1.0
            assert row["bulk_kmh"] >= 0.0
            assert row["local_speed_kmh"] >= 0.0
            bearing = row["bulk_dir_deg"]
            assert bearing is None or 0.0 <= bearing < 360.0
            wet = row["up_wet_frac_40km"]
            assert wet is None or 0.0 <= wet <= 1.0
            distance = row["up_dist_km"]
            assert distance is None or 0.0 <= distance <= 40.0

    def test_the_calendar_columns_come_off_the_decision_instant(
        self, featured_day,
    ) -> None:
        table, _out_dir = featured_day
        for row in table.to_pylist():
            stamp = row["generated_at"]
            assert row["season"] == "summer"          # 2026-09-05
            assert row["hour_utc"] == stamp.hour
            assert row["frame_age_min"] == pytest.approx(14.0)

    def test_the_radar_distance_is_the_stations_own(self, featured_day) -> None:
        from dmi_nowcast_core.product_pairs import nearest_radar_km

        table, _out_dir = featured_day
        expected = {
            point["id"]: nearest_radar_km(point["lat"], point["lon"])
            for point in POINTS["points"]
        }
        for row in table.to_pylist():
            assert row["station_radar_km"] == pytest.approx(
                expected[row["station_id"]], abs=1e-3,
            )

    def test_the_raw_fraction_is_the_uncalibrated_one(
        self, archive_dir: Path,
    ) -> None:
        """With no curves the two agree; the curve is what separates them."""
        samples = rw.sample_frame(
            rw.CompositeCache(archive_dir), T_ANCHOR,
            [rw.StationPoint(**p) for p in POINTS["points"]], TINY,
        )
        for sample in samples:
            for lead, value in sample["p_rain"].items():
                raw = sample["features"][f"raw_frac_{lead}"]
                assert raw == pytest.approx(value)

    def test_no_features_reproduces_the_shared_schema(
        self, archive_dir: Path, tmp_path: Path,
    ) -> None:
        from dataclasses import replace

        settings = replace(TINY, features=False)
        result = rw.run_day(
            _day_args(archive_dir, tmp_path, settings=settings)
        )
        assert result["failed"] is False
        table = _feature_table(tmp_path)
        assert table.schema.names == list(
            rw.decision_schema(TINY.leads_min).names
        )


class TestFeaturesAreIgnoredDownstream:
    """The additive columns must be invisible to everything that scores."""

    def test_the_sweeps_reader_conforms_them_away(self, featured_day) -> None:
        from dmi_nowcast_sidecar.threshold_sweep import load_decisions

        _table, out_dir = featured_day
        rows, leads, _counts = load_decisions([out_dir], leads_min=TINY.leads_min)
        assert rows
        assert set(rows[0]) == set(rw.decision_schema(leads).names)

    def test_the_benchmark_loader_reads_the_probabilities_unchanged(
        self, featured_day,
    ) -> None:
        import sys as _sys

        _sys.path.insert(0, str(_REPO_ROOT / "scripts"))
        import benchmark_report as bench

        table, out_dir = featured_day
        loaded = bench.load_probabilities([out_dir], [30])
        assert loaded["rows"] == table.num_rows
        assert set(loaded["p"]) == {30}

    def test_the_replays_own_reader_drops_them(self, featured_day) -> None:
        _table, out_dir = featured_day
        rows = rw.read_decisions(
            out_dir / "decisions" / f"{DAY}.parquet", TINY.leads_min,
        )
        assert set(rows[0]) == set(rw.decision_schema(TINY.leads_min).names)


def test_the_summary_documents_the_feature_columns(
    archive_dir: Path, points_file: Path, tmp_path: Path,
) -> None:
    out_dir = tmp_path / "documented"
    assert rw.main(_cli(
        archive_dir, out_dir, points_file,
        "--start-utc", "06:20", "--end-utc", "06:20",
    )) == 0
    block = json.loads((out_dir / "summary.json").read_text())["run"]["features"]
    assert block["enabled"] is True
    assert "raw_frac_30" in block["columns"]
    assert block["upstream_corridor"]["far_km"] == 40.0
    assert block["wet_mm_h"] == 0.5


def test_no_features_is_recorded_as_such(
    archive_dir: Path, points_file: Path, tmp_path: Path,
) -> None:
    out_dir = tmp_path / "bare"
    assert rw.main(_cli(
        archive_dir, out_dir, points_file, "--no-features",
        "--start-utc", "06:20", "--end-utc", "06:20",
    )) == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["run"]["features"]["enabled"] is False
    # Documented anyway, so a reader of an old run can see what is missing.
    assert summary["run"]["features"]["columns"]
