"""Tests for the L2 doppler → fullRange harmonisation map (Phase H, H-L).

The fit has one property worth a test above all others: on a synthetic
pair where doppler is exactly the fullRange field minus 3 dB, it must
recover a +3 dB mapping — and applying that mapping must give the
fullRange field back. Everything else here guards the edges around that:
the geometry the bands are cut with, the values the map must not touch
(nodata, undetect, sub-echo pixels, pixels beyond doppler's range), and
the refusal to apply a table of an unknown schema version.

The synthetic-composite writer is imported from
``test_gauge_agreement_study`` rather than copied — one description of
DMI's HDF5 layout, in one place.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import fit_doppler_harmonisation as fdh  # noqa: E402  (after sys.path edit)
from dmi_nowcast_core import product_pairs as pp  # noqa: E402
from dmi_nowcast_core.corpus import ArchiveIndex  # noqa: E402
from dmi_nowcast_core.geo import CompositeGeo  # noqa: E402
from dmi_nowcast_core.lightning import haversine_km  # noqa: E402
from dmi_nowcast_core.parse import parse_composite  # noqa: E402
from dmi_nowcast_core.transform import dbz_to_rain_rate  # noqa: E402

from .test_gauge_agreement_study import archive_frame, write_composite  # noqa: E402

DAY = datetime(2026, 6, 1, tzinfo=timezone.utc)
SHAPE = (48, 48)


def gradient_field(offset_db: float = 0.0) -> np.ndarray:
    """A field that populates every 0.5 dB bin from 5 to 49.5 dBZ.

    Values sit exactly on DMI's 0.5 dB quantisation grid, so a shift of a
    whole number of bins is recoverable exactly rather than to within
    rounding.
    """
    idx = np.arange(SHAPE[0] * SHAPE[1], dtype=np.float32)
    return (5.0 + 0.5 * (idx % 90.0)).reshape(SHAPE) + np.float32(offset_db)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_vectorised_haversine_matches_the_scalar_helper() -> None:
    lats = np.array([[55.0, 56.5], [57.2, 54.8]])
    lons = np.array([[9.0, 12.5], [10.1, 8.4]])
    got = pp._haversine_km_arrays(lats, lons, 56.024, 10.0246)
    for i in range(2):
        for j in range(2):
            want = haversine_km(lats[i, j], lons[i, j], 56.024, 10.0246)
            assert got[i, j] == pytest.approx(want, rel=1e-9)


def test_radar_distance_grid_agrees_with_the_point_distance(tmp_path: Path) -> None:
    path = write_composite(tmp_path / "f.h5", gradient_field(), DAY)
    composite = parse_composite(path)
    grid = pp.radar_distance_grid(composite)
    geo = CompositeGeo(composite)
    for row, col in ((0, 0), (12, 31), (47, 47)):
        lon, lat = geo.grid_to_lonlat(row, col)
        assert grid[row, col] == pytest.approx(pp.nearest_radar_km(lat, lon), rel=1e-4)
    # The Virring radar itself is inside this synthetic grid.
    assert grid.min() < 1.0


def test_band_edges_are_half_open_upwards() -> None:
    assert pp.band_of_km(0.0) == "0-60km"
    assert pp.band_of_km(59.9) == "0-60km"
    assert pp.band_of_km(60.0) == "60-90km"
    assert pp.band_of_km(120.0) == ">120km"
    assert pp.band_of_km(float("nan")) == ">120km"
    grid = np.array([[10.0, 75.0], [100.0, 200.0]], dtype=np.float32)
    assert pp.band_index_grid(grid).tolist() == [[0, 1], [2, 3]]


def test_rain_rate_to_dbz_inverts_the_z_r_conversion() -> None:
    for mm_h in (0.1, 0.5, 1.0, 5.0, 20.0):
        dbz = pp.rain_rate_to_dbz(mm_h)
        back = float(dbz_to_rain_rate(np.array([dbz], dtype=np.float32))[0])
        assert back == pytest.approx(mm_h, rel=1e-4)
    assert pp.rain_rate_to_dbz(0.5) == pytest.approx(18.19, abs=0.01)
    assert pp.rain_rate_to_dbz(0.0) == float("-inf")


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def _histograms_from(doppler: np.ndarray, full: np.ndarray) -> tuple:
    n_bins = pp.hist_edges().size
    hist_d = np.bincount(pp.digitize_dbz(doppler.ravel()), minlength=n_bins)
    hist_f = np.bincount(pp.digitize_dbz(full.ravel()), minlength=n_bins)
    return hist_d.astype(np.int64), hist_f.astype(np.int64)


def test_fit_recovers_a_three_db_offset_from_histograms() -> None:
    full = gradient_field()
    doppler = full - 3.0
    mapped = pp.fit_quantile_map(*_histograms_from(doppler, full))
    edges = pp.mapped_edges()
    # Only where the doppler sample actually has support: 7 .. 46.5 dBZ.
    inside = (edges >= 7.0) & (edges <= 46.5)
    assert np.allclose(mapped[inside] - edges[inside], 3.0, atol=1e-6)
    # Above the observed maximum the last offset is carried, not the top
    # of the histogram.
    assert mapped[-1] == pytest.approx(edges[-1] + 3.0, abs=1e-6)
    assert np.all(np.diff(mapped) >= 0.0)


def test_a_thin_tail_is_extrapolated_rather_than_saturated() -> None:
    """Above the last well-populated bin the map keeps its slope.

    Doppler's tail thins out faster than fullRange's, and matching two
    exhausted tails is what sent a 40 dBZ doppler echo to the top of the
    histogram (59.5 dBZ, ~1000 mm/h) before the tail guard existed. Above
    the anchor the map must be a straight offset instead — sloping at
    exactly one dB per dB, never flattening onto a ceiling.
    """
    edges = pp.hist_edges()
    hist_d = np.zeros(edges.size, dtype=np.int64)
    hist_f = np.zeros(edges.size, dtype=np.int64)
    hist_d[0] = hist_f[0] = 1_000_000                 # the dry majority
    body = (edges >= 7.0) & (edges < 30.0)
    hist_d[body] = 2_000
    hist_f[body] = 2_000
    hist_f[(edges >= 30.0) & (edges < 50.0)] = 300    # fullRange keeps going
    hist_d[(edges >= 30.0) & (edges < 32.0)] = 2      # doppler does not

    mapped = pp.fit_quantile_map(hist_d, hist_f)
    top = pp.mapped_edges() >= 32.0
    assert np.allclose(np.diff(mapped[top]), pp.HARM_BIN_WIDTH_DBZ)
    assert mapped[-1] > pp.HARM_HIST_HI_DBZ - pp.HARM_BIN_WIDTH_DBZ
    # In the body the map is still a real quantile match: fullRange
    # carries 12,000 pixels of tail that doppler does not, which is six
    # body bins of extra mass, so the body shifts up by about 3 dB —
    # a real correction, and a bounded one.
    fitted = (pp.mapped_edges() >= 7.0) & (pp.mapped_edges() < 25.0)
    shift = mapped[fitted] - pp.mapped_edges()[fitted]
    assert np.all(shift > 2.0) and np.all(shift < 4.0)


def test_fit_is_the_identity_without_any_sample() -> None:
    n_bins = pp.hist_edges().size
    mapped = pp.fit_quantile_map(np.zeros(n_bins, np.int64), np.zeros(n_bins, np.int64))
    assert np.allclose(mapped, pp.mapped_edges())


def test_triples_need_all_three_frames(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    for minutes in (0, 5, 10, 20, 30):   # 15 and 25 missing -> gaps
        archive_frame(corpus, gradient_field(), DAY + timedelta(minutes=minutes))
    frames = pp.frame_map(
        ArchiveIndex(corpus).list_in_window(DAY, DAY + timedelta(days=1))
    )
    triples = fdh.triples_for_day(frames)
    assert [t[0] for t in triples] == [DAY]      # only 00:00/00:05/00:10 is complete
    assert triples[0][1] == DAY + timedelta(minutes=5)
    assert triples[0][2] == DAY + timedelta(minutes=10)
    # --stride thins the list without changing what is eligible.
    for minutes in (15, 25):
        archive_frame(corpus, gradient_field(), DAY + timedelta(minutes=minutes))
    frames = pp.frame_map(
        ArchiveIndex(corpus).list_in_window(DAY, DAY + timedelta(days=1))
    )
    assert len(fdh.triples_for_day(frames)) == 3
    assert len(fdh.triples_for_day(frames, stride=2)) == 2


def _fit_one_day(corpus: Path, min_echo_pixels: int = 1) -> dict:
    frames = pp.frame_map(
        ArchiveIndex(corpus).list_in_window(DAY, DAY + timedelta(days=1))
    )
    _, result = fdh.run_day(DAY.date().isoformat(), frames, 1, "fit", None, 0.5)
    assert result["n_triples"] >= 1, result["errors"]
    tables, _ = fdh.build_tables(result["histograms"], min_echo_pixels)
    return tables


def test_run_day_fit_recovers_the_offset_end_to_end(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    full = gradient_field()
    archive_frame(corpus, full, DAY)
    archive_frame(corpus, full - 3.0, DAY + timedelta(minutes=5))
    archive_frame(corpus, full, DAY + timedelta(minutes=10))

    tables = _fit_one_day(corpus)
    # The synthetic grid sits inside 60 km of the Virring radar, in summer.
    assert set(tables) == {"summer|0-60km", "pooled|0-60km"}
    edges = pp.mapped_edges()
    for table in tables.values():
        mapped = np.asarray(table["mapped_dbz"])
        inside = (edges >= 7.0) & (edges <= 46.5)
        assert np.allclose(mapped[inside] - edges[inside], 3.0, atol=0.01)
        # Two bracketing fullRange frames against one doppler frame.
        assert table["n_fullrange_px"] == 2 * table["n_doppler_px"]


def test_cells_with_too_little_echo_fall_back_to_the_pooled_table(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    full = gradient_field()
    archive_frame(corpus, full, DAY)
    archive_frame(corpus, full - 3.0, DAY + timedelta(minutes=5))
    archive_frame(corpus, full, DAY + timedelta(minutes=10))
    tables = _fit_one_day(corpus, min_echo_pixels=10_000_000)
    assert set(tables) == {"pooled|0-60km"}     # the pooled table always fits
    payload = {
        "schema_version": pp.HARMONISATION_SCHEMA_VERSION,
        "bin_edges_dbz": [float(v) for v in pp.mapped_edges()],
        "tables": tables,
    }
    field = np.full((4, 4), 20.0, dtype=np.float32)
    distance = np.full((4, 4), 30.0, dtype=np.float32)
    out = pp.apply_harmonisation(field, distance, "summer", payload)
    assert np.allclose(out, 23.0, atol=0.01)    # served by pooled|0-60km


# ---------------------------------------------------------------------------
# Applying the map
# ---------------------------------------------------------------------------


@pytest.fixture
def three_db_payload(tmp_path: Path) -> dict:
    corpus = tmp_path / "corpus_fit"
    full = gradient_field()
    archive_frame(corpus, full, DAY)
    archive_frame(corpus, full - 3.0, DAY + timedelta(minutes=5))
    archive_frame(corpus, full, DAY + timedelta(minutes=10))
    return {
        "schema_version": pp.HARMONISATION_SCHEMA_VERSION,
        "bin_edges_dbz": [float(v) for v in pp.mapped_edges()],
        "tables": _fit_one_day(corpus),
    }


def test_apply_harmonisation_round_trips_the_offset(three_db_payload: dict) -> None:
    full = gradient_field()
    doppler = full - 3.0
    distance = np.full(SHAPE, 30.0, dtype=np.float32)
    out = pp.apply_harmonisation(doppler, distance, "summer", three_db_payload)
    echo = doppler >= pp.HARM_MIN_DBZ
    assert np.allclose(out[echo], full[echo], atol=0.01)
    # Below 7 dBZ the map does not reach: those pixels are untouched.
    assert np.array_equal(out[~echo], doppler[~echo])


def test_apply_harmonisation_leaves_sentinels_and_far_pixels_alone(
    three_db_payload: dict,
) -> None:
    field = np.full((4, 4), 20.0, dtype=np.float32)
    field[0, 0] = np.nan            # nodata
    field[0, 1] = -np.inf           # undetect
    field[0, 2] = 3.0               # below the echo floor
    distance = np.full((4, 4), 30.0, dtype=np.float32)
    distance[3, :] = 200.0          # beyond doppler's range: nothing fitted
    out = pp.apply_harmonisation(field, distance, "summer", three_db_payload)
    assert np.isnan(out[0, 0])
    assert np.isneginf(out[0, 1])
    assert out[0, 2] == pytest.approx(3.0)
    assert np.allclose(out[3, :], 20.0)          # unchanged beyond 120 km
    assert out[1, 1] == pytest.approx(23.0, abs=0.01)
    # The input is not modified in place.
    assert field[1, 1] == pytest.approx(20.0)


def test_apply_harmonisation_falls_back_to_pooled_for_an_unfitted_season(
    three_db_payload: dict,
) -> None:
    field = np.full((4, 4), 20.0, dtype=np.float32)
    distance = np.full((4, 4), 30.0, dtype=np.float32)
    winter = pp.apply_harmonisation(field, distance, "winter", three_db_payload)
    assert np.allclose(winter, 23.0, atol=0.01)


def test_apply_harmonisation_refuses_an_unknown_schema_version(
    three_db_payload: dict,
) -> None:
    payload = dict(three_db_payload, schema_version=99)
    with pytest.raises(ValueError, match="schema version"):
        pp.apply_harmonisation(
            np.zeros((2, 2), np.float32), np.zeros((2, 2), np.float32),
            "summer", payload,
        )


def test_apply_harmonisation_rejects_a_mismatched_distance_grid(
    three_db_payload: dict,
) -> None:
    with pytest.raises(ValueError, match="distance grid"):
        pp.apply_harmonisation(
            np.zeros((4, 4), np.float32), np.zeros((2, 2), np.float32),
            "summer", three_db_payload,
        )


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_main_writes_the_map_and_the_evaluation(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    full = gradient_field()
    for minutes in range(0, 40, 10):
        archive_frame(corpus, full, DAY + timedelta(minutes=minutes))
        archive_frame(corpus, full - 3.0, DAY + timedelta(minutes=minutes + 5))

    out_json = tmp_path / "out" / "doppler_harmonisation.json"
    out_md = tmp_path / "out" / "doppler_harmonisation.md"
    rc = fdh.main([
        "--corpus-dir", str(corpus),
        "--days", DAY.date().isoformat(),
        "--workers", "1",
        "--min-echo-pixels", "1",
        "--no-holdout",
        "--out-json", str(out_json),
        "--out-md", str(out_md),
    ])
    assert rc == 0

    payload = json.loads(out_json.read_text())
    assert payload["schema_version"] == pp.HARMONISATION_SCHEMA_VERSION
    assert "summer|0-60km" in payload["tables"]
    assert payload["meta"]["n_fit_triples"] == 3
    assert payload["meta"]["n_eval_triples"] == 3

    rows = {(r["season"], r["band"]): r for r in payload["evaluation"]}
    cell = rows[("summer", "0-60km")]
    # Before the map, doppler is 3 dB low: its wet area at 0.5 mm/h is
    # smaller than fullRange's. After it, the areas match exactly on this
    # synthetic field.
    assert cell["before"]["area_ratio"] < 0.99
    assert cell["after"]["area_ratio"] == pytest.approx(1.0, abs=0.02)
    assert cell["after"]["csi"] > cell["before"]["csi"]

    text = out_md.read_text()
    assert "harmonisation map" in text
    assert "area ratio before" in text
