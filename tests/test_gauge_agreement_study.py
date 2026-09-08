"""Tests for the L1 gauge-agreement study (Phase H, H-L).

Three things are worth pinning here:

1. The fast disc sampler is EXACTLY ``sample.sample_disc``. The study
   reads a hundred stations out of one frame instead of converting the
   whole grid to rain rate per station, and the whole comparison is
   worthless if that shortcut changes the statistic.
2. Coverage is a real distinction. A station outside doppler's 120 km
   range must be *unmeasured* there, never a doppler miss — the six
   NW-Jutland gauges would otherwise sink doppler's POD on their own.
3. The gauge wet rule is the shipped one, trace sentinel and duration arm
   included, and it is imported rather than re-implemented.

The synthetic-composite writer lives here and is re-used by
``test_doppler_harmonisation.py``: DMI's attribute layout is unusual
enough (scaling in the root ``/what``, ``product`` instead of
``quantity``, Z-R coefficients in ``/how``) that a second hand-rolled
copy would be a second chance to get it wrong.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import gauge_agreement_study as gas  # noqa: E402  (after sys.path edit)
from dmi_nowcast_core import product_pairs as pp  # noqa: E402
from dmi_nowcast_core.metobs import (  # noqa: E402
    PRECIP_DUR_PAST_10MIN,
    PRECIP_PAST_10MIN,
    Observation,
)
from dmi_nowcast_core.geo import CompositeGeo  # noqa: E402
from dmi_nowcast_core.parse import parse_composite  # noqa: E402
from dmi_nowcast_core.sample import sample_disc  # noqa: E402
from dmi_nowcast_core.station_store import StationObsStore  # noqa: E402
from dmi_nowcast_core.transform import dbz_to_rain_rate  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "composite_fullrange.h5"

#: DMI's own projection string, from the committed fixture.
PROJDEF = "+proj=stere +ellps=WGS84 +lat_0=56 +lon_0=10.5666 +lat_ts=56"

#: Upper-left corner of the synthetic grids: ~28 km north-west of the
#: Virring radar, so every pixel of a 48x48 grid at 500 m sits inside the
#: 0-60 km band and inside doppler's range.
SYNTHETIC_UL = (9.70, 56.20)

GAIN = 0.5
OFFSET = -32.0
NODATA = 255
UNDETECT = 0


# ---------------------------------------------------------------------------
# Synthetic composites
# ---------------------------------------------------------------------------


def write_composite(
    path: Path,
    dbz: np.ndarray,
    when: datetime,
    *,
    ul: tuple[float, float] = SYNTHETIC_UL,
    scale_m: float = 500.0,
    zr_a: float = 200.0,
    zr_b: float = 1.6,
) -> Path:
    """Write ``dbz`` as a DMI-shaped ODIM composite.

    ``NaN`` becomes ``nodata`` and ``-inf`` becomes ``undetect``, which is
    the inverse of what ``parse.parse_composite`` does, so a round trip
    through this writer is the identity on every value the parser can
    produce. Values are quantised to the 0.5 dB grid DMI's ``gain``
    imposes; a test that wants exact values should use multiples of 0.5.
    """
    import h5py
    from pyproj import CRS, Transformer

    dbz = np.asarray(dbz, dtype=np.float32)
    height, width = dbz.shape
    raw = np.full(dbz.shape, NODATA, dtype=np.uint8)
    finite = np.isfinite(dbz)
    raw[finite] = np.clip(
        np.rint((dbz[finite] - OFFSET) / GAIN), 1, 254,
    ).astype(np.uint8)
    raw[np.isneginf(dbz)] = UNDETECT

    crs = CRS.from_proj4(PROJDEF)
    to_proj = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x_min, y_max = to_proj.transform(*ul)
    x_max = x_min + (width - 1) * scale_m
    y_min = y_max - (height - 1) * scale_m
    corners = {
        "UL": (x_min, y_max), "UR": (x_max, y_max),
        "LL": (x_min, y_min), "LR": (x_max, y_min),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        what = h5.create_group("what")
        what.attrs["date"] = np.bytes_(when.strftime("%Y%m%d"))
        what.attrs["time"] = np.bytes_(when.strftime("%H%M%S"))
        what.attrs["object"] = np.bytes_("COMP")
        what.attrs["product"] = np.bytes_("DBZH")
        what.attrs["gain"] = np.float64(GAIN)
        what.attrs["offset"] = np.float64(OFFSET)
        what.attrs["nodata"] = np.int64(NODATA)
        what.attrs["undetect"] = np.float64(UNDETECT)
        where = h5.create_group("where")
        where.attrs["projdef"] = np.bytes_(PROJDEF)
        where.attrs["xscale"] = np.int64(scale_m)
        where.attrs["yscale"] = np.int64(scale_m)
        for name, (x, y) in corners.items():
            lon, lat = to_wgs.transform(x, y)
            where.attrs[f"{name}_lon"] = np.array([lon])
            where.attrs[f"{name}_lat"] = np.array([lat])
        how = h5.create_group("how")
        how.attrs["zr-a"] = np.array([zr_a])
        how.attrs["zr-b"] = np.array([zr_b])
        h5.create_dataset("dataset1/data1/data", data=raw)
    return path


def composite_name(when: datetime) -> str:
    return f"dk.com.{when:%Y%m%d%H%M}.500_max.h5"


def archive_frame(corpus: Path, dbz: np.ndarray, when: datetime, **kw) -> Path:
    """Write one composite into the corpus ``composites/YYYY/MM`` tree."""
    target = (
        corpus / "composites" / f"{when:%Y}" / f"{when:%m}" / composite_name(when)
    )
    return write_composite(target, dbz, when, **kw)


def station_coords(
    composite_path: Path, pixels: list[tuple[int, int]],
) -> list[tuple[float, float]]:
    """``(lat, lon)`` of the given ``(row, col)`` pixel centres."""
    geo = CompositeGeo(parse_composite(composite_path))
    out = []
    for row, col in pixels:
        lon, lat = geo.grid_to_lonlat(row, col)
        out.append((lat, lon))
    return out


# ---------------------------------------------------------------------------
# 1. The sampler is the production statistic
# ---------------------------------------------------------------------------


def test_radar_sites_match_the_calibration_point_set() -> None:
    """The two copies of DMI's radar list must never drift apart."""
    import build_calibration_points as bcp

    assert pp.RADAR_SITES == bcp.DMI_RADARS


def test_disc_sampler_matches_sample_disc_on_the_real_fixture() -> None:
    composite = parse_composite(FIXTURE)
    geo = CompositeGeo(composite)
    points = [
        (55.6726, 12.5645),   # Copenhagen
        (56.1629, 10.2039),   # Aarhus
        (57.4893, 10.1361),   # Sindal, on a radar
        (54.9000, 8.3000),    # far south-west, near the grid's rain edge
    ]
    rain = dbz_to_rain_rate(
        composite.reflectivity_dbz, zr_a=composite.zr_a, zr_b=composite.zr_b,
    )
    sampler = pp.DiscSampler(composite, points)
    got = sampler.sample(composite)
    for i, (lat, lon) in enumerate(points):
        want = sample_disc(rain, geo, lon, lat)
        assert got.p90[i] == pytest.approx(want.p90_mm_h, nan_ok=True, rel=1e-6)
        assert got.max_[i] == pytest.approx(want.max_mm_h, nan_ok=True, rel=1e-6)
        assert got.mean[i] == pytest.approx(want.mean_mm_h, nan_ok=True, rel=1e-6)
        assert got.valid_frac[i] == pytest.approx(
            want.n_valid / want.n_pixels_in_disc, rel=1e-6,
        )


def test_disc_sampler_is_all_nan_off_the_grid() -> None:
    composite = parse_composite(FIXTURE)
    sampler = pp.DiscSampler(composite, [(48.0, 2.0)])  # Paris: not on the grid
    got = sampler.sample(composite)
    assert np.isnan(got.p90[0])
    assert np.isnan(got.valid_frac[0])


def test_disc_sampler_valid_fraction_counts_nodata_out(tmp_path: Path) -> None:
    """Undetect is data (observed dry); nodata is not (unmeasured)."""
    dbz = np.full((48, 48), 20.0, dtype=np.float32)
    dbz[:, 20:] = np.nan                      # nodata over the right half
    dbz[10, 10] = -np.inf                     # one undetect inside the disc
    path = write_composite(tmp_path / "frame.h5", dbz,
                           datetime(2026, 6, 1, tzinfo=timezone.utc))
    composite = parse_composite(path)
    # (10, 10) is fully inside the data; (10, 20) straddles the nodata edge.
    coords = station_coords(path, [(10, 10), (10, 20)])
    got = pp.DiscSampler(composite, coords).sample(composite)
    assert got.valid_frac[0] == pytest.approx(1.0)
    assert 0.0 < got.valid_frac[1] < 1.0
    # The undetect pixel is 0 mm/h, so it drags the mean below the p90.
    assert got.mean[0] < got.p90[0]


# ---------------------------------------------------------------------------
# 2. Coverage: inside and outside doppler's range
# ---------------------------------------------------------------------------


def _day_frames(
    corpus: Path, day: datetime, dbz_full: np.ndarray, dbz_doppler: np.ndarray,
    slots: int = 2,
) -> dict:
    """fullRange at :x0 and doppler at :x5 for the first ``slots`` slots."""
    for i in range(slots + 1):
        archive_frame(corpus, dbz_full, day + timedelta(minutes=10 * i))
    for i in range(slots):
        archive_frame(corpus, dbz_doppler, day + timedelta(minutes=10 * i + 5))
    from dmi_nowcast_core.corpus import ArchiveIndex

    index = ArchiveIndex(corpus)
    return pp.frame_map(index.list_in_window(day, day + timedelta(days=1)))


def test_station_outside_doppler_coverage_is_not_scored(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    day = datetime(2026, 6, 1, tzinfo=timezone.utc)
    full = np.full((48, 48), 30.0, dtype=np.float32)
    doppler = np.full((48, 48), 30.0, dtype=np.float32)
    doppler[24:, :] = np.nan   # doppler does not reach the southern half
    frames = _day_frames(corpus, day, full, doppler)

    reference = next(iter(frames["fullRange"].values()))
    coords = station_coords(reference, [(10, 10), (40, 10)])
    _, columns, meta = gas.run_day(day.date().isoformat(), coords, frames, 1000.0)
    assert meta["frames_read"] == 5

    # Row layout is slot-major: rows 0..1 are the two stations of slot 0.
    assert columns["dop_valid"][0] == pytest.approx(1.0)
    assert columns["dop_valid"][1] == pytest.approx(0.0)
    assert np.isnan(columns["dop_p90"][1])
    assert columns["fr_end_valid"][1] == pytest.approx(1.0)

    table = {
        "gauge_known": np.array([True, True]),
        "gauge_wet": np.array([True, True]),
        "fr_end_p90": columns["fr_end_p90"][:2],
        "fr_end_valid": columns["fr_end_valid"][:2],
        "dop_p90": columns["dop_p90"][:2],
        "dop_valid": columns["dop_valid"][:2],
        "fr_start_p90": columns["fr_start_p90"][:2],
        "fr_start_valid": columns["fr_start_valid"][:2],
    }
    scored, radar_wet = gas.series_masks(table, "doppler_mid", 0.5, 0.5)
    assert scored.tolist() == [True, False]      # the far station is unmeasured
    assert radar_wet.tolist() == [True, False]
    scored_full, _ = gas.series_masks(table, "fullRange_end", 0.5, 0.5)
    assert scored_full.tolist() == [True, True]
    # Consensus needs both products, so it inherits doppler's coverage.
    scored_both, _ = gas.series_masks(table, "consensus", 0.5, 0.5)
    assert scored_both.tolist() == [True, False]


def test_run_day_puts_each_frame_in_its_own_slot_column(tmp_path: Path) -> None:
    """T-10 is the slot's start, T-5 its middle, T its end."""
    corpus = tmp_path / "corpus"
    day = datetime(2026, 6, 1, tzinfo=timezone.utc)
    # Three fullRange frames with different intensities, one doppler.
    archive_frame(corpus, np.full((48, 48), 20.0, np.float32), day)
    archive_frame(corpus, np.full((48, 48), 35.0, np.float32),
                  day + timedelta(minutes=5))
    archive_frame(corpus, np.full((48, 48), 40.0, np.float32),
                  day + timedelta(minutes=10))
    from dmi_nowcast_core.corpus import ArchiveIndex

    frames = pp.frame_map(
        ArchiveIndex(corpus).list_in_window(day, day + timedelta(days=1))
    )
    reference = frames["fullRange"][day]
    coords = station_coords(reference, [(10, 10)])
    _, columns, _ = gas.run_day(day.date().isoformat(), coords, frames, 1000.0)

    def rate(dbz: float) -> float:
        return float(dbz_to_rain_rate(np.array([dbz], np.float32))[0])

    # Slot 0 ends at 00:10.
    assert columns["fr_start_p90"][0] == pytest.approx(rate(20.0), rel=1e-5)
    assert columns["dop_p90"][0] == pytest.approx(rate(35.0), rel=1e-5)
    assert columns["fr_end_p90"][0] == pytest.approx(rate(40.0), rel=1e-5)
    # Slot 1 ends at 00:20 and has only its start frame (00:10).
    assert columns["fr_start_p90"][1] == pytest.approx(rate(40.0), rel=1e-5)
    assert np.isnan(columns["fr_end_p90"][1])


# ---------------------------------------------------------------------------
# 3. The gauge wet rule
# ---------------------------------------------------------------------------


def _write_gauge(corpus: Path, day: datetime, station: str) -> None:
    """One station's day: wet, trace, duration-only, silent, dry."""
    store = StationObsStore(corpus)
    obs = [
        # slot 0 (ends 00:10): a real 0.3 mm
        Observation(station, day + timedelta(minutes=10), PRECIP_PAST_10MIN, 0.3),
        Observation(station, day + timedelta(minutes=10), PRECIP_DUR_PAST_10MIN, 4.0),
        # slot 1 (00:20): DMI's trace sentinel, no duration -> known, dry
        Observation(station, day + timedelta(minutes=20), PRECIP_PAST_10MIN, -0.1),
        # slot 2 (00:30): no measurable depth but the gauge ran 2 minutes
        Observation(station, day + timedelta(minutes=30), PRECIP_PAST_10MIN, 0.0),
        Observation(station, day + timedelta(minutes=30), PRECIP_DUR_PAST_10MIN, 2.0),
        # slot 3 (00:40): nothing at all -> unknown
        # slot 4 (00:50): reported dry
        Observation(station, day + timedelta(minutes=50), PRECIP_PAST_10MIN, 0.0),
        Observation(station, day + timedelta(minutes=50), PRECIP_DUR_PAST_10MIN, 0.0),
    ]
    store.append(obs)


def test_gauge_slots_follow_the_shipped_wet_rule(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    day = datetime(2026, 6, 1, tzinfo=timezone.utc)
    _write_gauge(corpus, day, "06031")
    gauge = gas.load_gauge_days(corpus, [day.date()], ["06031"])
    truth = gauge[day.date()]

    known = truth["known"][0]
    wet = truth["wet"][0]
    mm = truth["mm"][0]
    dur = truth["dur"][0]

    assert known[:5].tolist() == [True, True, True, False, True]
    assert wet[:5].tolist() == [True, False, True, False, False]
    # The trace sentinel is folded to 0.0 mm, never a negative depth.
    assert mm[1] == pytest.approx(0.0)
    assert mm[0] == pytest.approx(0.3)
    assert dur[2] == pytest.approx(2.0)
    assert np.isnan(mm[3]) and np.isnan(dur[3])
    # A slot the station never reported is unknown, and unknown is not dry.
    assert not wet[3]


# ---------------------------------------------------------------------------
# 4. Scoring
# ---------------------------------------------------------------------------


def test_contingency_matches_hand_computed_scores() -> None:
    cell = gas.contingency(hits=6, misses=2, false_alarms=4, correct=88)
    assert cell["pod"] == pytest.approx(6 / 8)
    assert cell["far"] == pytest.approx(4 / 10)
    assert cell["csi"] == pytest.approx(6 / 12)
    assert cell["bias"] == pytest.approx(10 / 8)
    assert cell["base_rate"] == pytest.approx(8 / 100)
    assert cell["n"] == 100


def test_contingency_is_none_rather_than_zero_when_undefined() -> None:
    cell = gas.contingency(hits=0, misses=0, false_alarms=0, correct=10)
    assert cell["pod"] is None and cell["far"] is None and cell["csi"] is None


def test_consensus_is_the_intersection_and_either_the_union() -> None:
    table = {
        "gauge_known": np.array([True, True, True, True]),
        "fr_end_p90": np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "fr_end_valid": np.ones(4, dtype=np.float32),
        "dop_p90": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "dop_valid": np.ones(4, dtype=np.float32),
    }
    _, consensus = gas.series_masks(table, "consensus", 0.5, 0.5)
    _, either = gas.series_masks(table, "either", 0.5, 0.5)
    assert consensus.tolist() == [True, False, False, False]
    assert either.tolist() == [True, True, True, False]


def test_scores_of_a_perfect_and_a_blind_series() -> None:
    n = 20
    gauge_wet = np.zeros(n, dtype=bool)
    gauge_wet[:5] = True
    table = {
        "station_idx": np.zeros(n, dtype=np.int32),
        "gauge_known": np.ones(n, dtype=bool),
        "gauge_wet": gauge_wet,
        "fr_start_p90": np.where(gauge_wet, 2.0, 0.0).astype(np.float32),
        "fr_start_valid": np.ones(n, dtype=np.float32),
        "fr_end_p90": np.zeros(n, dtype=np.float32),
        "fr_end_valid": np.ones(n, dtype=np.float32),
        "dop_p90": np.full(n, 2.0, dtype=np.float32),
        "dop_valid": np.ones(n, dtype=np.float32),
    }
    scores = gas.score_groups(
        table, [("pooled", np.ones(n, dtype=bool))], [0.5], 0.5,
    )
    perfect = scores["fullRange_start"]["0.5"]["pooled"]
    assert perfect["pod"] == 1.0 and perfect["far"] == 0.0 and perfect["csi"] == 1.0
    blind = scores["fullRange_end"]["0.5"]["pooled"]
    assert blind["pod"] == 0.0 and blind["bias"] == 0.0
    crying_wolf = scores["doppler_mid"]["0.5"]["pooled"]
    assert crying_wolf["pod"] == 1.0
    assert crying_wolf["far"] == pytest.approx(15 / 20)
    assert crying_wolf["bias"] == pytest.approx(20 / 5)


# ---------------------------------------------------------------------------
# 5. End to end
# ---------------------------------------------------------------------------


def test_main_writes_parquet_json_and_markdown(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    corpus = tmp_path / "corpus"
    day = datetime(2026, 6, 1, tzinfo=timezone.utc)
    full = np.full((48, 48), 35.0, dtype=np.float32)
    doppler = np.full((48, 48), 35.0, dtype=np.float32)
    doppler[24:, :] = np.nan
    frames = _day_frames(corpus, day, full, doppler, slots=3)
    reference = next(iter(frames["fullRange"].values()))
    coords = station_coords(reference, [(10, 10), (40, 10)])

    points = {
        "version": 2,
        "points": [
            {"id": "near", "lat": coords[0][0], "lon": coords[0][1],
             "region": "Midtjylland"},
            {"id": "far", "lat": coords[1][0], "lon": coords[1][1],
             "region": "Midtjylland"},
        ],
    }
    points_path = tmp_path / "points.json"
    points_path.write_text(json.dumps(points))
    _write_gauge(corpus, day, "near")
    _write_gauge(corpus, day, "far")

    out_json = tmp_path / "out" / "study.json"
    out_md = tmp_path / "out" / "study.md"
    out_parquet = tmp_path / "out" / "slots.parquet"
    rc = gas.main([
        "--corpus-dir", str(corpus),
        "--points", str(points_path),
        "--days", day.date().isoformat(),
        "--workers", "1",
        "--out-json", str(out_json),
        "--out-md", str(out_md),
        "--out-parquet", str(out_parquet),
    ])
    assert rc == 0

    payload = json.loads(out_json.read_text())
    assert payload["meta"]["n_stations"] == 2
    assert payload["meta"]["n_slots"] == 2 * gas.SLOTS_PER_DAY
    # Only one of the two stations is inside doppler's synthetic coverage.
    assert payload["coverage"]["stations_doppler_covered"] == 0  # 3 of 144 slots
    assert payload["coverage"]["slots_doppler_covered"] == 3
    assert payload["coverage"]["slots_both_covered"] == 3
    pooled = payload["scores"]["fullRange_end"]["0.5"]["pooled"]
    assert pooled["hits"] >= 1
    assert "L1 — gauge agreement per radar product" in out_md.read_text()

    table = pq.read_table(out_parquet)
    assert table.num_rows == 2 * gas.SLOTS_PER_DAY
    assert set(gas.SAMPLE_COLUMNS) <= set(table.schema.names)
    assert "gauge_dur_min" in table.schema.names
    stations = set(table.column("station_id").to_pylist())
    assert stations == {"near", "far"}
