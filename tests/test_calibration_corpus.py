"""Tests for the multi-point calibration corpus builder (Phase B: B0 + B2,
plus C1 fullRange-only scan-type filtering, 10-min spacing, and snapping —
and the simulated frame age that puts fitted and served leads on the same
instant).

Pure-function tests only: NO real STEPS, NO network, NO DMI calls. The
builder is a script, not a package — ``scripts/`` goes on ``sys.path``
exactly as ``tests/test_corpus_manifest.py`` does for the manifest builder.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_calibration_corpus as bcc  # noqa: E402  (after sys.path edit)
from dmi_nowcast_core.fetch import RadarFeature  # noqa: E402
from dmi_nowcast_core.geo import GridIndex  # noqa: E402
from dmi_nowcast_core.national import NationalProducts  # noqa: E402
from dmi_nowcast_core.sample import DiscStats  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers: synthetic points file, geo, products
# ---------------------------------------------------------------------------


POINTS_V2 = {
    "version": 2,
    "points": [
        # Extra keys per point must be tolerated and ignored.
        {"id": "fyn-centroid", "lat": 55.33, "lon": 10.32, "region": "fyn",
         "coast_km": 3.2, "note": "included by construction"},
        {"id": "cph", "lat": 55.6726, "lon": 12.5645, "region": "sjaelland"},
        {"id": "skagen", "lat": 57.72, "lon": 10.58, "region": "nordjylland"},
    ],
}


def _write_points(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "points.json"
    path.write_text(json.dumps(payload))
    return path


class FakeGeo:
    """Native-grid lookup keyed on (lon, lat) — only what sampling needs."""

    def __init__(self, mapping: dict[tuple[float, float], tuple[float, float]]):
        self._mapping = mapping

    def lonlat_to_grid(self, lon: float, lat: float) -> GridIndex:
        row, col = self._mapping[(lon, lat)]
        return GridIndex(row=row, col=col)


def _products(
    p_rain: dict[int, np.ndarray], downsample_factor: int = 4
) -> NationalProducts:
    shape = next(iter(p_rain.values())).shape
    leads = tuple(sorted(p_rain))
    return NationalProducts(
        p_rain=p_rain,
        eta_min=np.zeros(shape, dtype=np.float32),
        intensity_mm_h=np.zeros(shape, dtype=np.float32),
        leads_min=leads,
        threshold_mm_h=0.5,
        timestep_min=10.0,
        frame_age_min=0.0,
        downsample_factor=downsample_factor,
        n_members=16,
    )


def _settings(**overrides) -> bcc.CorpusSettings:
    """Runtime-parity settings. ``frame_age_range`` is deliberately left at
    the dataclass default (12-18 min — the live compute latency) so the
    tests exercise the convention the builder actually ships; the
    zero-age contrast is passed explicitly where it matters."""
    kwargs = dict(
        ensemble_size=16,
        n_cascade_levels=6,
        downsample_factor=4,
        threshold_mm_h=0.5,
        disc_radius_m=1000.0,
        detection_stat="p90",
        leads_min=(5, 10, 15, 20, 25, 30, 45, 60),
        scan_type="fullRange",
        timestep_min=10.0,
    )
    kwargs.update(overrides)
    return bcc.CorpusSettings(**kwargs)


# ---------------------------------------------------------------------------
# B0 — settings hash
# ---------------------------------------------------------------------------


def test_settings_hash_stable_across_construction_routes():
    a = _settings()
    b = bcc.CorpusSettings(
        leads_min=bcc.parse_leads("5,10,15,20,25,30,45,60"),
        detection_stat="p90",
        disc_radius_m=1000,  # int vs float must not change the hash
        threshold_mm_h=0.5,
        downsample_factor=4,
        n_cascade_levels=6,
        ensemble_size=16,
    )
    assert a.settings_hash == b.settings_hash
    assert len(a.settings_hash) == 16


@pytest.mark.parametrize("override", [
    {"ensemble_size": 24},
    {"n_cascade_levels": 7},
    {"downsample_factor": 2},
    {"threshold_mm_h": 0.1},
    {"disc_radius_m": 2000.0},
    {"detection_stat": "max"},
    {"leads_min": (10, 30, 60)},
    # C1: a corpus that differs ONLY in scan type or timestep must hash
    # differently — that is what makes the fitter refuse pre-fix corpora.
    {"scan_type": "doppler"},
    {"timestep_min": 5.0},
    # Frame age: a zero-age corpus verifies at different instants than a
    # latency-simulating one, so the two must never mix.
    {"frame_age_range": (0.0, 0.0)},
    {"frame_age_range": (10.0, 20.0)},
])
def test_settings_hash_changes_when_any_setting_changes(override):
    assert _settings().settings_hash != _settings(**override).settings_hash


def test_n_timesteps_spans_longest_effective_lead():
    """The ensemble must reach the longest lead the RUNTIME reads — nominal
    lead + the largest frame age the sampler can draw."""
    # 10-min timestep, 60-min horizon, 12-18 min age → ceil(78 / 10) = 8.
    assert _settings().n_timesteps == 8
    assert _settings(leads_min=(5, 10, 13)).n_timesteps == 4  # ceil(31 / 10)
    # Zero age reproduces the old horizon exactly.
    zero = {"frame_age_range": (0.0, 0.0)}
    assert _settings(**zero).n_timesteps == 6  # ceil(60 / 10)
    assert _settings(leads_min=(5, 10, 13), **zero).n_timesteps == 2  # ceil(13 / 10)
    # The derivation is timestep-agnostic (legacy 5-min step for contrast).
    assert _settings(timestep_min=5.0, **zero).n_timesteps == 12
    assert _settings(leads_min=(5, 10, 13), timestep_min=5.0, **zero).n_timesteps == 3
    # Only the UPPER bound of the range matters — it sets the horizon.
    assert _settings(frame_age_range=(0.0, 18.0)).n_timesteps == 8


def test_settings_columns_carry_hash_and_schema_version():
    s = _settings()
    cols = s.settings_columns()
    assert cols["settings_hash"] == s.settings_hash
    assert cols["schema_version"] == bcc.SCHEMA_VERSION
    assert cols["leads_min_csv"] == "5,10,15,20,25,30,45,60"
    assert cols["n_timesteps"] == 8
    assert cols["scan_type"] == "fullRange"
    assert cols["motion_method"] == "farneback_complete_v1"
    assert cols["timestep_min"] == pytest.approx(10.0)
    # The list-valued setting is flattened to a scalar Parquet column.
    assert cols["frame_age_range_csv"] == "12,18"
    assert "frame_age_range" not in cols
    assert _settings(frame_age_range=(0.0, 0.0)).settings_columns()[
        "frame_age_range_csv"
    ] == "0,0"
    assert _settings(frame_age_range=(12.5, 18.0)).settings_columns()[
        "frame_age_range_csv"
    ] == "12.5,18"


def test_parse_leads_rejects_bad_specs():
    assert bcc.parse_leads("5, 10 ,15") == (5, 10, 15)
    for bad in ("", "10,5", "10,10", "0,5", "a,b"):
        with pytest.raises(ValueError):
            bcc.parse_leads(bad)


# ---------------------------------------------------------------------------
# B0 — mixed-corpus refusal at the builder
# ---------------------------------------------------------------------------


def _write_corpus(tmp_path: Path, settings: bcc.CorpusSettings) -> Path:
    out = tmp_path / "corpus.parquet"
    rows = bcc.build_event_rows(
        "2026-08-01T12:00:00+00:00",
        bcc.load_points(_write_points(tmp_path, POINTS_V2)),
        settings.leads_min,
        {},
        {},
    )
    for row in rows:
        row["sample_weight"] = 1.0
        row.update(settings.settings_columns())
    bcc._append_rows(out, rows)
    return out


def test_check_existing_corpus_accepts_matching_hash(tmp_path: Path):
    s = _settings()
    out = _write_corpus(tmp_path, s)
    existing = bcc.check_existing_corpus(out, s.settings_hash)
    assert existing == {"2026-08-01T12:00:00+00:00"}


def test_check_existing_corpus_refuses_mismatched_hash(tmp_path: Path):
    out = _write_corpus(tmp_path, _settings())
    with pytest.raises(ValueError, match="mixed corpus"):
        bcc.check_existing_corpus(out, _settings(ensemble_size=24).settings_hash)


@pytest.mark.parametrize("override", [
    {"scan_type": "doppler"},
    {"timestep_min": 5.0},
    {"motion_method": "farneback_raw"},
])
def test_check_existing_corpus_refuses_scan_type_or_timestep_change(
    tmp_path: Path, override: dict
):
    """C1/R5: resuming a pre-fix corpus under the new scan-type, timestep
    or motion-method settings must be refused at the builder — no silent
    mixing."""
    out = _write_corpus(tmp_path, _settings(**override))
    with pytest.raises(ValueError, match="mixed corpus"):
        bcc.check_existing_corpus(out, _settings().settings_hash)


def test_check_existing_corpus_refuses_legacy_v1_file(tmp_path: Path):
    out = tmp_path / "legacy.parquet"
    table = pa.Table.from_pylist([
        {"event_time": "2026-01-01T00:00:00+00:00", "lead_min": 10,
         "raw_prob": 0.2, "outcome": 1, "error": ""},
    ])
    pq.write_table(table, out)
    with pytest.raises(ValueError, match="legacy"):
        bcc.check_existing_corpus(out, _settings().settings_hash)


def test_check_existing_corpus_missing_file_is_empty(tmp_path: Path):
    assert bcc.check_existing_corpus(tmp_path / "nope.parquet", "abc") == set()


# ---------------------------------------------------------------------------
# B2 — points file
# ---------------------------------------------------------------------------


def test_load_points_tolerates_extra_keys(tmp_path: Path):
    points = bcc.load_points(_write_points(tmp_path, POINTS_V2))
    assert [p.id for p in points] == ["fyn-centroid", "cph", "skagen"]
    assert points[0].lat == pytest.approx(55.33)
    assert points[0].region == "fyn"
    assert not hasattr(points[0], "coast_km")


@pytest.mark.parametrize("payload", [
    {"version": 3, "points": POINTS_V2["points"]},          # wrong version
    {"points": POINTS_V2["points"]},                        # no version
    {"version": 2, "points": []},                           # empty
    {"version": 2},                                         # missing points
    {"version": 2, "points": [{"id": "a", "lat": 55.0}]},   # missing keys
    {"version": 2, "points": [
        {"id": "a", "lat": 1.0, "lon": 2.0, "region": "x"},
        {"id": "a", "lat": 3.0, "lon": 4.0, "region": "y"},
    ]},                                                     # duplicate id
])
def test_load_points_rejects_bad_files(tmp_path: Path, payload: dict):
    with pytest.raises(ValueError):
        bcc.load_points(_write_points(tmp_path, payload))


# ---------------------------------------------------------------------------
# B2 — point sampling from the national grids (/forecast convention)
# ---------------------------------------------------------------------------


def test_sample_points_uses_forecast_pixel_convention(tmp_path: Path):
    points = bcc.load_points(_write_points(tmp_path, POINTS_V2))
    # Product grid 10x12 (downsampled by 4 from a fake 40x48 native grid);
    # each pixel encodes its own (row, col) so a mapping mistake is visible.
    h, w = 10, 12
    grid5 = (np.arange(h)[:, None] * 100 + np.arange(w)[None, :]).astype(np.float32)
    grid10 = grid5 + 0.5
    products = _products({5: grid5, 10: grid10}, downsample_factor=4)
    geo = FakeGeo({
        # native (10.4, 7.9) → round(10.4/4)=3, round(7.9/4)=2
        (10.32, 55.33): (10.4, 7.9),
        # native (0.0, 0.0) → (0, 0)
        (12.5645, 55.6726): (0.0, 0.0),
        # rounds to row 10 == h → out of grid → NaN
        (10.58, 57.72): (38.1, 4.0),
    })

    raw = bcc.sample_points_from_products(products, geo, points)

    assert raw["fyn-centroid"][5] == pytest.approx(302.0)   # grid5[3, 2]
    assert raw["fyn-centroid"][10] == pytest.approx(302.5)
    assert raw["cph"][5] == pytest.approx(0.0)      # grid5[0, 0]
    assert raw["cph"][10] == pytest.approx(0.5)
    assert np.isnan(raw["skagen"][5]) and np.isnan(raw["skagen"][10])


def test_sample_points_rounds_half_pixels_like_forecast_endpoint():
    # /forecast does int(round(idx.row / f)) — banker's rounding included.
    point = bcc.CalibrationPoint(id="p", lat=1.0, lon=2.0, region="r")
    grid = np.zeros((4, 4), dtype=np.float32)
    grid[2, 2] = 0.75
    products = _products({10: grid}, downsample_factor=4)
    geo = FakeGeo({(2.0, 1.0): (10.0, 10.0)})  # 10/4 = 2.5 → round → 2
    raw = bcc.sample_points_from_products(products, geo, (point,))
    assert raw["p"][10] == pytest.approx(0.75)


def test_build_event_rows_one_row_per_point_and_lead(tmp_path: Path):
    points = bcc.load_points(_write_points(tmp_path, POINTS_V2))
    leads = (5, 10)
    raw = {p.id: {5: 0.25, 10: 0.5} for p in points}
    outcomes = {p.id: {5: 1, 10: 0} for p in points}
    rows = bcc.build_event_rows("2026-08-01T12:00:00+00:00", points, leads, raw, outcomes)

    assert len(rows) == len(points) * len(leads)
    assert {(r["point_id"], r["lead_min"]) for r in rows} == {
        (p.id, lead) for p in points for lead in leads
    }
    home5 = next(
        r for r in rows if r["point_id"] == "fyn-centroid" and r["lead_min"] == 5
    )
    assert home5["raw_prob"] == pytest.approx(0.25)
    assert home5["outcome"] == 1
    assert home5["lat"] == pytest.approx(55.33)
    assert home5["region"] == "fyn"
    assert home5["error"] == ""


# ---------------------------------------------------------------------------
# Outcomes — detection stat vs threshold, missing frame → null
# ---------------------------------------------------------------------------


def _stats(max_=3.0, mean=0.4, p90=0.6, n_pixels=12, n_valid=12) -> DiscStats:
    return DiscStats(
        max_mm_h=max_, mean_mm_h=mean, p90_mm_h=p90,
        n_pixels_in_disc=n_pixels, n_valid=n_valid,
    )


def test_outcome_uses_configured_detection_stat():
    stats = _stats(max_=3.0, mean=0.4, p90=0.6)
    # p90 (runtime default) 0.6 >= 0.5 → rain
    assert bcc.outcome_from_stats(stats, "p90", 0.5) == 1
    # p90 below a higher threshold → dry, even though max is far above
    assert bcc.outcome_from_stats(stats, "p90", 0.7) == 0
    assert bcc.outcome_from_stats(stats, "max", 0.5) == 1
    assert bcc.outcome_from_stats(stats, "mean", 0.5) == 0
    with pytest.raises(ValueError):
        bcc.detection_stat_value(stats, "median")


def test_outcome_verifies_against_synthetic_rain_field():
    """End-to-end disc → stat → outcome on a synthetic verification frame."""
    from dmi_nowcast_core.sample import sample_disc

    class Composite:
        xscale_m = 500.0
        yscale_m = 500.0

    class Geo(FakeGeo):
        composite = Composite()

    rain = np.zeros((20, 20), dtype=np.float32)
    rain[9:12, 9:12] = 2.0  # a wet block covering the whole 1 km disc
    geo = Geo({(10.0, 55.0): (10.0, 10.0), (11.0, 56.0): (3.0, 3.0)})

    wet = sample_disc(rain, geo, 10.0, 55.0, radius_m=1000.0)
    dry = sample_disc(rain, geo, 11.0, 56.0, radius_m=1000.0)
    assert bcc.outcome_from_stats(wet, "p90", 0.5) == 1
    assert bcc.outcome_from_stats(dry, "p90", 0.5) == 0


def test_missing_verification_frame_yields_null_outcome(tmp_path: Path):
    points = bcc.load_points(_write_points(tmp_path, POINTS_V2))
    leads = (5, 10)
    raw = {p.id: {5: 0.25, 10: 0.5} for p in points}
    # Lead 10's verification frame is missing for everyone.
    outcomes = {p.id: {5: 0, 10: None} for p in points}
    rows = bcc.build_event_rows("2026-08-01T12:00:00+00:00", points, leads, raw, outcomes)
    for r in rows:
        if r["lead_min"] == 10:
            assert r["outcome"] is None       # kept as a row, null outcome
            assert np.isfinite(r["raw_prob"])  # forecast still recorded
        else:
            assert r["outcome"] == 0


def test_invalid_disc_stats_yield_null_outcome():
    empty = DiscStats(np.nan, np.nan, np.nan, 0, 0)
    assert bcc.outcome_from_stats(empty, "p90", 0.5) is None


def test_error_event_rows_keep_event_resumable(tmp_path: Path):
    points = bcc.load_points(_write_points(tmp_path, POINTS_V2))
    rows = bcc.error_event_rows("2026-08-01T12:00:00+00:00", points, (5, 10), "boom")
    assert len(rows) == len(points) * 2
    assert all(np.isnan(r["raw_prob"]) for r in rows)
    assert all(r["outcome"] is None for r in rows)
    assert all(r["error"] == "boom" for r in rows)


# ---------------------------------------------------------------------------
# sample_weight — inverse inclusion probability (Finding 3)
# ---------------------------------------------------------------------------


def test_stratum_weights_formula():
    wet_w, dry_w = bcc.stratum_weights(
        n_wet_available=40, n_dry_available=600, n_wet_drawn=30, n_dry_drawn=170,
    )
    assert wet_w == pytest.approx(40 / 30)
    assert dry_w == pytest.approx(600 / 170)
    # A stratum nothing was drawn from: weight NaN (never attached to a row).
    wet_w, dry_w = bcc.stratum_weights(
        n_wet_available=0, n_dry_available=10, n_wet_drawn=0, n_dry_drawn=5,
    )
    assert np.isnan(wet_w)
    assert dry_w == pytest.approx(2.0)


def test_sample_event_times_attaches_stratum_weights(tmp_path: Path, monkeypatch):
    """Full sampler path on a synthetic wet/dry index — no network.

    The candidate hours are pinned and the wet/dry index is pre-written to
    the cache under the name the default reference set keys, so
    ``_build_or_load_wet_dry_index`` has nothing to classify. (Runs with
    ``scan_type=""`` — the legacy unfiltered path; the CLI always passes a
    concrete scan type, covered by the C1 tests below.)
    """
    base = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    candidates = [base + timedelta(hours=h) for h in range(20)]
    monkeypatch.setattr(bcc, "_candidate_hours", lambda *, days_back: candidates)

    # Hours 0-7 wet (8), hours 8-19 dry (12).
    index = {bcc._ts_str(h): (i < 8) for i, h in enumerate(candidates)}
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / bcc.wet_dry_index_name("", bcc.DEFAULT_WET_REFS)).write_text(
        json.dumps(index)
    )

    sampled = bcc._sample_event_times(
        days_back=1, n_events=10, seed=7, wet_bias=0.3, cache_dir=cache,
    )
    assert len(sampled) == 10
    # n_wet = round(10 * 0.3) = 3 → weight 8/3; n_dry = 7 → weight 12/7.
    wet_times = {t for t, w in index.items() if w}
    n_wet = sum(1 for t, _ in sampled if bcc._ts_str(t) in wet_times)
    assert n_wet == 3
    for t, weight in sampled:
        expected = 8 / 3 if bcc._ts_str(t) in wet_times else 12 / 7
        assert weight == pytest.approx(expected)


def test_sample_event_times_uniform_weights_are_one(monkeypatch):
    base = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    candidates = [base + timedelta(hours=h) for h in range(10)]
    monkeypatch.setattr(bcc, "_candidate_hours", lambda *, days_back: candidates)
    sampled = bcc._sample_event_times(days_back=1, n_events=5, seed=3, wet_bias=0.0)
    assert len(sampled) == 5
    assert all(w == 1.0 for _, w in sampled)


# ---------------------------------------------------------------------------
# C1 — scan-type interleave filtering (fullRange :x0 / doppler :x5)
# ---------------------------------------------------------------------------


def _feature(dt: datetime, scan_type: str = "") -> RadarFeature:
    """A feature as resolved from the local archive: filename/timestamp
    only — no scanType label (DMI filenames don't encode it)."""
    name = f"dk.com.{dt:%Y%m%d%H%M}.500_max.h5"
    return RadarFeature(
        feature_id=name, datetime_utc=dt, scan_type=scan_type,
        download_url="", filename=name,
    )


def test_scan_grid_offset():
    assert bcc.scan_grid_offset_min("fullRange") == 0
    assert bcc.scan_grid_offset_min("doppler") == 5
    with pytest.raises(ValueError, match="unknown scan type"):
        bcc.scan_grid_offset_min("volume")


def test_filter_scan_type_by_filename_minute():
    """Mixed archive listing, unlabeled: the filename minute encodes the
    interleave, so only %10==0 frames survive a fullRange filter."""
    base = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    feats = [_feature(base + timedelta(minutes=5 * i)) for i in range(6)]  # :00–:25
    full = bcc.filter_scan_type(feats, "fullRange")
    assert [f.datetime_utc.minute for f in full] == [0, 10, 20]
    dopp = bcc.filter_scan_type(feats, "doppler")
    assert [f.datetime_utc.minute for f in dopp] == [5, 15, 25]
    # Empty scan type = no filtering (legacy mixed behaviour).
    assert bcc.filter_scan_type(feats, "") == feats


def test_filter_scan_type_prefers_api_label_over_minute():
    dt = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    labeled = _feature(dt, scan_type="doppler")  # label wins over the :00 minute
    assert not bcc.feature_matches_scan_type(labeled, "fullRange")
    assert bcc.feature_matches_scan_type(labeled, "doppler")
    assert bcc.feature_matches_scan_type(_feature(dt, "fullRange"), "fullRange")


# ---------------------------------------------------------------------------
# C1 — verification snapping rule (lead → 10-min frame grid)
# ---------------------------------------------------------------------------


def test_snap_lead_up_to_frame_grid():
    # ceil to the next whole timestep — the exact timestep the forecast
    # probability for that (effective) lead reads (national_products
    # convention). Zero-age arguments reproduce the original rule.
    assert bcc.snap_lead_min(5, 10.0) == 10
    assert bcc.snap_lead_min(10, 10.0) == 10
    assert bcc.snap_lead_min(15, 10.0) == 20
    assert bcc.snap_lead_min(25, 10.0) == 30
    assert bcc.snap_lead_min(45, 10.0) == 50
    assert bcc.snap_lead_min(60, 10.0) == 60
    # Degenerates to the identity on the legacy 5-min grid.
    assert all(bcc.snap_lead_min(x, 5.0) == x for x in (5, 10, 15, 45, 60))


def test_snap_lead_min_snaps_the_effective_lead():
    """Callers pass lead + frame age; the instant moves with the age.

    The headline case from the frame-age fix: lead 30 with a 15-min frame
    age on a 10-min grid verifies at T+50 — the timestep the live service
    actually reads — not at T+30.
    """
    assert bcc.snap_lead_min(30 + 15.0, 10.0) == 50
    assert bcc.snap_lead_min(60 + 18.0, 10.0) == 80   # the longest instant
    assert bcc.snap_lead_min(5 + 12.0, 10.0) == 20
    assert bcc.snap_lead_min(5 + 18.0, 10.0) == 30    # straddles a grid line
    # Float ages must not fall off the grid.
    assert bcc.snap_lead_min(30 + 12.3456, 10.0) == 50
    assert bcc.snap_lead_min(30 + 0.0, 10.0) == 30


def test_snapped_instant_is_the_timestep_national_products_reads():
    """Parity with the runtime, through the public API.

    ``national_products`` pools ensemble timesteps 0..k-1 for
    ``k = ceil((lead + frame_age) / timestep)``, and timestep ``t`` is
    valid at ``(t + 1) * timestep`` after radar time. So the last
    timestep the probability sees is exactly ``snap_lead_min(lead + age,
    timestep)`` minutes out — which is where the corpus verifies. Proven
    by walking a single-member ensemble's rain arrival across every
    timestep and checking the probability flips at that instant.
    """
    from dmi_nowcast_core.national import national_products

    timestep, age, lead, n_steps = 10.0, 15.0, 30, 8
    instant = bcc.snap_lead_min(lead + age, timestep)
    assert instant == 50
    flips = []
    for arrival in range(n_steps):
        ensemble = np.zeros((1, n_steps, 1, 1), dtype=np.float32)
        ensemble[0, arrival:, 0, 0] = 5.0  # rain from this timestep on
        products = national_products(
            ensemble, leads_min=(lead,), threshold_mm_h=0.5,
            timestep_min=timestep, frame_age_min=age, downsample_factor=1,
        )
        p = float(products.p_rain[lead][0, 0])
        assert p == pytest.approx(float((arrival + 1) * timestep <= instant))
        flips.append(p)
    # Sanity: the walk really did straddle the boundary (5 in, 3 out).
    assert flips == [1.0] * 5 + [0.0] * 3


def _mixed_listing(start: datetime, end: datetime) -> list[RadarFeature]:
    """A 5-min mixed fullRange/doppler interleave across [start, end],
    unlabeled — as if the API-side scanType filter were ignored."""
    feats = []
    t = start.replace(minute=(start.minute // 5) * 5, second=0, microsecond=0)
    while t <= end:
        if t >= start:
            feats.append(_feature(t))
        t += timedelta(minutes=5)
    return feats


def test_gather_event_frames_fullrange_inputs_and_snapped_truth(monkeypatch):
    event = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    captured = {}

    def fake_list(start, end, *, limit, scan_type=None):
        captured["scan_type"] = scan_type
        return _mixed_listing(start, end)

    monkeypatch.setattr(bcc, "list_in_window", fake_list)
    inputs, truth = bcc._gather_event_frames(
        event, _settings(leads_min=(5, 10, 15, 30))
    )

    assert captured["scan_type"] == "fullRange"  # API-side filter requested
    # Inputs at T-20, T-10, T — on the :x0 grid despite the mixed listing.
    assert [f.datetime_utc for f in inputs] == [
        event - timedelta(minutes=20),
        event - timedelta(minutes=10),
        event,
    ]
    # Verification snapped up: 5 → T+10, 15 → T+20; on-grid leads stay put.
    assert truth[5].datetime_utc == event + timedelta(minutes=10)
    assert truth[10].datetime_utc == event + timedelta(minutes=10)
    assert truth[15].datetime_utc == event + timedelta(minutes=20)
    assert truth[30].datetime_utc == event + timedelta(minutes=30)
    # Nothing off-grid leaked anywhere.
    for f in list(inputs) + list(truth.values()):
        assert f.datetime_utc.minute % 10 == 0


def test_gather_event_frames_missing_snapped_frame_leaves_outcome_null(monkeypatch):
    """No frame within tolerance of the SNAPPED target → the lead is absent
    from truth, and build_event_rows records a null outcome for it."""
    event = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)

    def fake_list(start, end, *, limit, scan_type=None):
        # Drop the T+10 fullRange frame. Lead 5 snaps to T+10; its nearest
        # surviving fullRange neighbours (T, T+20) are 10 min away — well
        # outside the ±4 min tolerance. The T+5/T+15 doppler frames would
        # be closer but must be filtered out.
        return [
            f for f in _mixed_listing(start, end)
            if f.datetime_utc != event + timedelta(minutes=10)
        ]

    monkeypatch.setattr(bcc, "list_in_window", fake_list)
    settings = _settings(leads_min=(5, 20))
    inputs, truth = bcc._gather_event_frames(event, settings)
    assert len(inputs) == 3
    assert 5 not in truth
    assert truth[20].datetime_utc == event + timedelta(minutes=20)

    # Downstream: the missing lead yields a null outcome row.
    outcomes = {"p": {5: None, 20: 1}}
    rows = bcc.build_event_rows(
        bcc._ts_str(event),
        (bcc.CalibrationPoint(id="p", lat=55.0, lon=10.0, region="r"),),
        settings.leads_min, {"p": {5: 0.3, 20: 0.3}}, outcomes,
    )
    assert next(r for r in rows if r["lead_min"] == 5)["outcome"] is None
    assert next(r for r in rows if r["lead_min"] == 20)["outcome"] == 1


def test_gather_event_frames_missing_input_raises(monkeypatch):
    """A fullRange corpus must not fall back to a doppler input frame."""
    event = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)

    def fake_list(start, end, *, limit, scan_type=None):
        # Only doppler-minute frames available around T-10.
        return [
            f for f in _mixed_listing(start, end)
            if f.datetime_utc != event - timedelta(minutes=10)
        ]

    monkeypatch.setattr(bcc, "list_in_window", fake_list)
    with pytest.raises(RuntimeError, match="no fullRange input frame"):
        bcc._gather_event_frames(event, _settings(leads_min=(10,)))


# ---------------------------------------------------------------------------
# Frame age — the corpus simulates the live compute latency
# ---------------------------------------------------------------------------


def test_parse_frame_age_range():
    assert bcc.parse_frame_age_range("12,18") == (12.0, 18.0)
    assert bcc.parse_frame_age_range(" 12.5 , 18 ") == (12.5, 18.0)
    assert bcc.parse_frame_age_range("0,0") == (0.0, 0.0)   # old convention
    assert bcc.parse_frame_age_range("15,15") == (15.0, 15.0)  # fixed age
    assert bcc.parse_frame_age_range("0,60") == (0.0, 60.0)  # bounds inclusive


@pytest.mark.parametrize("spec", [
    "", "12", "12,18,20", "a,b", "-1,18", "18,12", "0,61", "nan,18", "12,inf",
])
def test_parse_frame_age_range_rejects_bad_specs(spec: str):
    with pytest.raises(ValueError):
        bcc.parse_frame_age_range(spec)


def test_default_frame_age_range_matches_the_live_latency():
    # The live cycle finishes 12-18 min after its newest frame's radar
    # timestamp; the CLI default and the dataclass default agree.
    assert bcc.DEFAULT_FRAME_AGE_RANGE == (12.0, 18.0)
    assert _settings().frame_age_range == bcc.DEFAULT_FRAME_AGE_RANGE


EVENT_T = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)


def test_event_frame_age_is_inside_the_range():
    events = [EVENT_T + timedelta(hours=h) for h in range(200)]
    ages = [bcc.event_frame_age_min(t, 42, (12.0, 18.0)) for t in events]
    assert all(12.0 <= a <= 18.0 for a in ages)
    # It is a real draw, not a constant: the range is actually explored.
    assert len(set(ages)) > 150
    assert min(ages) < 13.0 and max(ages) > 17.0
    assert np.mean(ages) == pytest.approx(15.0, abs=0.5)


def test_zero_frame_age_range_gives_zero_everywhere():
    """``--frame-age-range 0,0`` reproduces the old zero-age convention."""
    for h in range(50):
        assert bcc.event_frame_age_min(
            EVENT_T + timedelta(hours=h), 42, (0.0, 0.0)
        ) == 0.0
    # A degenerate non-zero range is the same idea: one fixed age.
    assert bcc.event_frame_age_min(EVENT_T, 42, (15.0, 15.0)) == 15.0


def test_event_frame_age_is_reproducible_per_event_and_seed():
    """The resume guarantee: same (seed, event) → same age, whatever else
    the process has done to the global RNG or the order of calls."""
    import random as _random

    a = bcc.event_frame_age_min(EVENT_T, 42, (12.0, 18.0))
    _random.seed(0)
    [_random.random() for _ in range(17)]  # disturb the global RNG
    bcc.event_frame_age_min(EVENT_T + timedelta(hours=3), 42, (12.0, 18.0))
    assert bcc.event_frame_age_min(EVENT_T, 42, (12.0, 18.0)) == a
    # A different seed or a different event draws a different age.
    assert bcc.event_frame_age_min(EVENT_T, 43, (12.0, 18.0)) != a
    assert bcc.event_frame_age_min(
        EVENT_T + timedelta(minutes=10), 42, (12.0, 18.0)
    ) != a


def test_event_frame_age_is_reproducible_in_a_fresh_process():
    """A resumed build is a NEW process, possibly with a different hash
    seed — the draw must survive both (hence sha256, not ``hash()``)."""
    import os
    import subprocess

    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from datetime import datetime, timezone\n"
        "import build_calibration_corpus as bcc\n"
        "t = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)\n"
        "print(repr(bcc.event_frame_age_min(t, 42, (12.0, 18.0))))\n"
    ) % str(SCRIPTS)
    env = {**os.environ, "PYTHONHASHSEED": "1"}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        check=True, env=env,
    ).stdout.strip()
    assert float(out) == bcc.event_frame_age_min(EVENT_T, 42, (12.0, 18.0))


def test_settings_hash_covers_the_frame_age_key_itself():
    """Not just the VALUE: a hash computed over a settings dict without the
    key at all (i.e. any pre-frame-age corpus) must differ from both the
    zero-age and the default-age hash, so no old corpus can be appended to."""
    import hashlib
    import json as _json

    zero = _settings(frame_age_range=(0.0, 0.0))
    aged = _settings()
    assert zero.settings_hash != aged.settings_hash

    legacy_dict = zero.to_dict()
    legacy_dict.pop("frame_age_range")
    legacy_hash = hashlib.sha256(
        _json.dumps(legacy_dict, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    assert legacy_hash != zero.settings_hash
    assert legacy_hash != aged.settings_hash


def test_gather_event_frames_verifies_at_the_frame_aged_instant(monkeypatch):
    """L=30, age=15, timestep=10 → the verification frame is T+50, and the
    fetch window stretches to cover the longest aged instant (T+80)."""
    event = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    window: dict[str, datetime] = {}

    def fake_list(start, end, *, limit, scan_type=None):
        window["start"], window["end"] = start, end
        return _mixed_listing(start, end)

    monkeypatch.setattr(bcc, "list_in_window", fake_list)
    settings = _settings(leads_min=(5, 30, 60))
    inputs, truth = bcc._gather_event_frames(event, settings, 15.0)

    # Inputs are unaffected — they are anchored at radar-frame time.
    assert [f.datetime_utc for f in inputs] == [
        event - timedelta(minutes=20),
        event - timedelta(minutes=10),
        event,
    ]
    # ceil((30 + 15) / 10) * 10 = 50; ceil((60 + 15) / 10) * 10 = 80.
    assert truth[30].datetime_utc == event + timedelta(minutes=50)
    assert truth[60].datetime_utc == event + timedelta(minutes=80)
    assert truth[5].datetime_utc == event + timedelta(minutes=20)
    # The prefetch window must reach the latest instant (+ tolerance).
    assert window["end"] >= event + timedelta(minutes=80)
    assert window["start"] <= event - timedelta(minutes=20)

    # Zero age is the old behaviour, unchanged.
    _, truth0 = bcc._gather_event_frames(event, settings, 0.0)
    assert truth0[30].datetime_utc == event + timedelta(minutes=30)
    assert truth0[60].datetime_utc == event + timedelta(minutes=60)


def test_frame_age_is_recorded_on_every_row(tmp_path: Path):
    points = bcc.load_points(_write_points(tmp_path, POINTS_V2))
    rows = bcc.build_event_rows(
        "2026-08-01T12:00:00+00:00", points, (5, 10),
        {p.id: {5: 0.25, 10: 0.5} for p in points},
        {p.id: {5: 1, 10: 0} for p in points},
        frame_age_min=14.25,
    )
    assert all(r["frame_age_min"] == pytest.approx(14.25) for r in rows)
    # Failed events keep the age too — the row still says what was tried.
    err = bcc.error_event_rows(
        "2026-08-01T12:00:00+00:00", points, (5, 10), "boom", frame_age_min=17.5
    )
    assert all(r["frame_age_min"] == pytest.approx(17.5) for r in err)


def test_frame_age_column_survives_the_parquet_roundtrip(tmp_path: Path):
    s = _settings(leads_min=(5, 10))
    points = bcc.load_points(_write_points(tmp_path, POINTS_V2))
    rows = bcc.build_event_rows(
        "2026-08-01T12:00:00+00:00", points, (5, 10),
        {p.id: {5: 0.25, 10: 0.5} for p in points},
        {p.id: {5: 1, 10: 0} for p in points},
        frame_age_min=13.5,
    )
    for row in rows:
        row["sample_weight"] = 1.0
        row.update(s.settings_columns())

    out = tmp_path / "corpus.parquet"
    bcc._append_rows(out, rows)
    table = pq.read_table(out)

    assert "frame_age_min" in table.schema.names
    assert table.schema.field("frame_age_min").type == pa.float32()
    got = table.to_pylist()
    assert all(r["frame_age_min"] == pytest.approx(13.5) for r in got)
    assert all(r["frame_age_range_csv"] == "12,18" for r in got)


# ---------------------------------------------------------------------------
# Frame age — the whole per-event path (national_products + rows)
# ---------------------------------------------------------------------------


class _FakeComposite:
    """The handful of attributes ``_process_event`` touches on a composite."""

    reflectivity_dbz = np.zeros((4, 4), dtype=np.float32)
    zr_a, zr_b, xscale_m = 200.0, 1.6, 500.0


def test_process_event_uses_the_events_age_for_products_and_rows(
    tmp_path: Path, monkeypatch
):
    """Spy on the whole per-event path: the age the parent drew must reach
    ``national_products`` (the forecast), ``_gather_event_frames`` (the
    verification instants) and every row (the record)."""
    points = bcc.load_points(_write_points(tmp_path, POINTS_V2))
    settings = _settings(leads_min=(5, 10))
    event = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    age = 16.125
    seen: dict[str, float] = {}

    def fake_gather(event_time, s, frame_age_min=0.0):
        seen["gather"] = frame_age_min
        feats = [_feature(event + timedelta(minutes=m)) for m in (-20, -10, 0)]
        return feats, {}

    def fake_products(forecast, **kwargs):
        seen["products"] = kwargs["frame_age_min"]
        grid = np.full((2, 2), 0.4, dtype=np.float32)
        return _products({5: grid, 10: grid}, downsample_factor=4)

    monkeypatch.setattr(bcc, "_gather_event_frames", fake_gather)
    monkeypatch.setattr(bcc, "_resolve_frame", lambda *a, **k: Path("x.h5"))
    monkeypatch.setattr(bcc, "parse_composite", lambda p: _FakeComposite())
    monkeypatch.setattr(
        bcc, "CompositeGeo",
        lambda c: FakeGeo({(p.lon, p.lat): (0.0, 0.0) for p in points}),
    )
    monkeypatch.setattr(
        bcc, "dbz_to_rain_rate",
        lambda *a, **k: np.zeros((4, 4), dtype=np.float32),
    )
    monkeypatch.setattr(
        "dmi_nowcast_core.dense_flow.dense_flow",
        lambda a, b: (np.zeros((4, 4), np.float32), np.zeros((4, 4), np.float32)),
    )
    monkeypatch.setattr(
        "dmi_nowcast_core.dense_flow.complete_flow",
        lambda vy, vx, rain, **k: (vy, vx),
    )
    monkeypatch.setattr(bcc, "run_ensemble", lambda *a, **k: np.zeros((1, 8, 2, 2)))
    monkeypatch.setattr(bcc, "national_products", fake_products)
    monkeypatch.setattr(
        bcc, "_score_outcomes",
        lambda truth, pts, s, cache, corpus: {
            p.id: {5: 1, 10: 0} for p in pts
        },
    )

    result = bcc._process_event(
        (event, tmp_path, None, points, settings, age)
    )

    assert result.error is None
    assert seen["gather"] == pytest.approx(age)
    assert seen["products"] == pytest.approx(age)
    assert result.rows and all(
        r["frame_age_min"] == pytest.approx(age) for r in result.rows
    )


def test_process_event_records_the_age_on_a_failed_event(
    tmp_path: Path, monkeypatch
):
    points = bcc.load_points(_write_points(tmp_path, POINTS_V2))
    settings = _settings(leads_min=(5, 10))

    def boom(*a, **k):
        raise RuntimeError("no frames")

    monkeypatch.setattr(bcc, "_gather_event_frames", boom)
    result = bcc._process_event(
        (datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc),
         tmp_path, None, points, settings, 13.75)
    )
    assert "no frames" in (result.error or "")
    assert all(r["frame_age_min"] == pytest.approx(13.75) for r in result.rows)


# ---------------------------------------------------------------------------
# C1 — wet/dry index cache re-key + event grid
# ---------------------------------------------------------------------------


REFS_ONE = ((55.0, 10.0),)
REFS_TWO = ((55.0, 10.0), (57.0, 10.3))


def test_wet_dry_index_cache_is_keyed_by_scan_type(tmp_path: Path, monkeypatch):
    """A stale mixed-type wet_dry_index.json must NOT satisfy a fullRange
    build — the typed cache name forces reclassification."""
    hours = [datetime(2026, 8, 1, h, 0, tzinfo=timezone.utc) for h in range(4)]
    cache = tmp_path / "cache"
    cache.mkdir()
    # Stale mixed-type cache claiming everything is already classified wet.
    (cache / "wet_dry_index.json").write_text(
        json.dumps({bcc._ts_str(h): True for h in hours})
    )

    calls: list[datetime] = []

    def fake_classify(h, refs, cache_dir, corpus_dir=None, scan_type=""):
        calls.append(h)
        return h.hour < 2

    monkeypatch.setattr(bcc, "_classify_hour_wet", fake_classify)
    index = bcc._build_or_load_wet_dry_index(
        hours, REFS_ONE, cache_dir=cache, workers=2, scan_type="fullRange",
    )
    assert len(calls) == 4  # stale mixed cache ignored → everything reclassified
    assert (cache / bcc.wet_dry_index_name("fullRange", REFS_ONE)).exists()
    assert index == {bcc._ts_str(h): (h.hour < 2) for h in hours}

    # Second run hits the typed cache: nothing left to classify.
    calls.clear()
    again = bcc._build_or_load_wet_dry_index(
        hours, REFS_ONE, cache_dir=cache, workers=2, scan_type="fullRange",
    )
    assert calls == []
    assert again == index


# ---------------------------------------------------------------------------
# Multi-point wet reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec, expected", [
    ("55.5,11.8", ((55.5, 11.8),)),
    (" 55.5 , 11.8 ", ((55.5, 11.8),)),
    ("57.2,9.6;55.1,14.9", ((57.2, 9.6), (55.1, 14.9))),
    ("57.2,9.6;;55.1,14.9;", ((57.2, 9.6), (55.1, 14.9))),
])
def test_parse_wet_refs(spec: str, expected: tuple):
    assert bcc.parse_wet_refs(spec) == expected


@pytest.mark.parametrize("spec", ["", "55.5", "55.5,11.8,3", "a,b", "95.0,11.8"])
def test_parse_wet_refs_rejects_bad_specs(spec: str):
    with pytest.raises(ValueError):
        bcc.parse_wet_refs(spec)


def test_default_wet_refs_are_five_spread_danish_points():
    assert len(bcc.DEFAULT_WET_REFS) == 5
    lats = [la for la, _ in bcc.DEFAULT_WET_REFS]
    lons = [lo for _, lo in bcc.DEFAULT_WET_REFS]
    assert max(lats) - min(lats) > 1.5   # Jutland north ↔ south
    assert max(lons) - min(lons) > 5.0   # west coast ↔ Bornholm
    assert all(54.0 < la < 58.0 and 8.0 < lo < 15.5 for la, lo in bcc.DEFAULT_WET_REFS)


def test_resolve_wet_refs_precedence():
    # Default when nothing is given.
    assert bcc.resolve_wet_refs(None) == bcc.DEFAULT_WET_REFS
    # Repeatable flag, and ';'-separated within one flag.
    assert bcc.resolve_wet_refs(["57.2,9.6", "55.5,11.8"]) == (
        (57.2, 9.6), (55.5, 11.8),
    )
    assert bcc.resolve_wet_refs(["57.2,9.6;55.5,11.8"]) == (
        (57.2, 9.6), (55.5, 11.8),
    )
    # Deprecated single-point pair still works, and duplicates collapse.
    assert bcc.resolve_wet_refs(None, 55.5, 11.8) == ((55.5, 11.8),)
    assert bcc.resolve_wet_refs(["55.5,11.8"], 55.5, 11.8) == ((55.5, 11.8),)
    with pytest.raises(ValueError):
        bcc.resolve_wet_refs(None, 55.5, None)


def test_wet_refs_key_is_order_independent_and_set_specific():
    assert bcc.wet_refs_key(REFS_TWO) == bcc.wet_refs_key(tuple(reversed(REFS_TWO)))
    assert bcc.wet_refs_key(REFS_TWO) != bcc.wet_refs_key(REFS_ONE)
    assert bcc.wet_refs_key(REFS_ONE) != bcc.wet_refs_key(((55.001, 10.0),))
    assert len(bcc.wet_refs_key(REFS_TWO)) == 8


def test_wet_dry_index_name_carries_type_and_refs():
    name = bcc.wet_dry_index_name("fullRange", REFS_TWO)
    assert name == f"wet_dry_index_fullRange_{bcc.wet_refs_key(REFS_TWO)}.json"
    # Untyped (legacy) path still gets a reference-keyed name.
    assert bcc.wet_dry_index_name("", REFS_TWO).startswith("wet_dry_index_")
    # Different reference set → different file.
    assert bcc.wet_dry_index_name("fullRange", REFS_ONE) != name


def test_wet_dry_index_cache_is_keyed_by_reference_set(tmp_path: Path, monkeypatch):
    """An index built for one reference set must never satisfy another."""
    hours = [datetime(2026, 8, 1, h, 0, tzinfo=timezone.utc) for h in range(3)]
    cache = tmp_path / "cache"
    cache.mkdir()

    seen: list[tuple] = []

    def fake_classify(h, refs, cache_dir, corpus_dir=None, scan_type=""):
        seen.append(refs)
        return True

    monkeypatch.setattr(bcc, "_classify_hour_wet", fake_classify)
    bcc._build_or_load_wet_dry_index(
        hours, REFS_ONE, cache_dir=cache, workers=2, scan_type="fullRange",
    )
    assert len(seen) == 3 and all(r == REFS_ONE for r in seen)

    seen.clear()
    bcc._build_or_load_wet_dry_index(
        hours, REFS_TWO, cache_dir=cache, workers=2, scan_type="fullRange",
    )
    # The one-reference index is NOT reused: everything is reclassified.
    assert len(seen) == 3 and all(r == REFS_TWO for r in seen)
    assert (cache / bcc.wet_dry_index_name("fullRange", REFS_ONE)).exists()
    assert (cache / bcc.wet_dry_index_name("fullRange", REFS_TWO)).exists()


def test_classify_hour_wet_is_wet_when_any_reference_sees_rain(monkeypatch):
    """Rain 200 km from reference A but on top of reference B ⇒ wet."""
    rain = np.zeros((200, 200), dtype=np.float32)
    rain[150:155, 150:155] = 3.0  # only near the SECOND reference

    class _Geo:
        def lonlat_to_grid(self, lon, lat):
            return GridIndex(row=20.0, col=20.0) if lat == 55.0 \
                else GridIndex(row=152.0, col=152.0)

    class _Composite:
        reflectivity_dbz = np.zeros((200, 200), dtype=np.float32)
        zr_a, zr_b, xscale_m = 200.0, 1.6, 500.0

    feature = RadarFeature(
        feature_id="f", filename="dk.com.202608010000.500_max.h5",
        datetime_utc=datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc),
        scan_type="fullRange", download_url="",
    )
    monkeypatch.setattr(bcc, "list_in_window", lambda *a, **k: [feature])
    monkeypatch.setattr(bcc, "_resolve_frame", lambda *a, **k: Path("x.h5"))
    monkeypatch.setattr(bcc, "parse_composite", lambda p: _Composite())
    monkeypatch.setattr(bcc, "CompositeGeo", lambda c: _Geo())
    monkeypatch.setattr(bcc, "dbz_to_rain_rate", lambda *a, **k: rain)

    hour = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    cache = Path("/nonexistent")
    # Reference A alone: dry. A + B: wet — the union decides.
    assert bcc._classify_hour_wet(
        hour, ((55.0, 10.0),), cache, search_km=10.0, scan_type="fullRange"
    ) is False
    assert bcc._classify_hour_wet(
        hour, ((55.0, 10.0), (57.0, 10.3)), cache, search_km=10.0,
        scan_type="fullRange",
    ) is True


def test_sample_event_times_fullrange_uses_typed_index_and_grid(
    tmp_path: Path, monkeypatch
):
    base = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    candidates = [base + timedelta(hours=h) for h in range(20)]
    monkeypatch.setattr(bcc, "_candidate_hours", lambda *, days_back: candidates)
    monkeypatch.setattr(
        bcc, "_classify_hour_wet",
        lambda *a, **k: pytest.fail("classification must not run — cache present"),
    )

    # Hours 0-7 wet (8), hours 8-19 dry (12) — in the TYPED cache.
    index = {bcc._ts_str(h): (i < 8) for i, h in enumerate(candidates)}
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / bcc.wet_dry_index_name("fullRange", bcc.DEFAULT_WET_REFS)).write_text(
        json.dumps(index)
    )
    # Poisoned legacy mixed-type cache that must NOT be consulted.
    (cache / "wet_dry_index.json").write_text(
        json.dumps({bcc._ts_str(h): False for h in candidates})
    )

    sampled = bcc._sample_event_times(
        days_back=1, n_events=10, seed=7, wet_bias=0.3, cache_dir=cache,
        scan_type="fullRange",
    )
    assert len(sampled) == 10
    # Typed index used: 3 wet draws exist (the poisoned legacy index has none).
    wet_times = {t for t, w in index.items() if w}
    assert sum(1 for t, _ in sampled if bcc._ts_str(t) in wet_times) == 3
    # Every event anchor sits on the fullRange 10-min grid.
    assert all(t.minute % 10 == 0 for t, _ in sampled)


def test_sample_event_times_doppler_grid_offset(monkeypatch):
    base = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    candidates = [base + timedelta(hours=h) for h in range(10)]
    monkeypatch.setattr(bcc, "_candidate_hours", lambda *, days_back: candidates)
    sampled = bcc._sample_event_times(
        days_back=1, n_events=5, seed=3, wet_bias=0.0, scan_type="doppler",
    )
    # Doppler frames live at :x5 — event anchors shift onto that grid.
    assert all(t.minute == 5 for t, _ in sampled)


# ---------------------------------------------------------------------------
# Parquet schema v2
# ---------------------------------------------------------------------------


def test_parquet_schema_v2_roundtrip(tmp_path: Path):
    s = _settings(leads_min=(5, 10))
    points = bcc.load_points(_write_points(tmp_path, POINTS_V2))
    raw = {p.id: {5: 0.25, 10: float("nan")} for p in points}
    outcomes = {p.id: {5: 1, 10: None} for p in points}
    rows = bcc.build_event_rows("2026-08-01T12:00:00+00:00", points, (5, 10), raw, outcomes)
    for row in rows:
        row["sample_weight"] = 1.5
        row.update(s.settings_columns())

    out = tmp_path / "corpus.parquet"
    bcc._append_rows(out, rows)
    table = pq.read_table(out)

    assert table.schema.names == bcc._parquet_schema().names
    assert table.schema.field("outcome").type == pa.int8()
    assert table.schema.field("raw_prob").type == pa.float32()
    assert table.num_rows == len(points) * 2

    got = table.to_pylist()
    lead10 = [r for r in got if r["lead_min"] == 10]
    assert all(r["outcome"] is None for r in lead10)  # null preserved
    assert all(np.isnan(r["raw_prob"]) for r in lead10)
    assert all(r["sample_weight"] == pytest.approx(1.5) for r in got)
    assert all(r["settings_hash"] == s.settings_hash for r in got)
    assert all(r["schema_version"] == 2 for r in got)
    assert all(r["threshold_mm_h"] == pytest.approx(0.5) for r in got)
    assert all(r["ensemble_size"] == 16 for r in got)
    assert all(r["scan_type"] == "fullRange" for r in got)
    assert all(r["timestep_min"] == pytest.approx(10.0) for r in got)

    # Appending more rows keeps the file readable and resumable.
    rows2 = bcc.build_event_rows("2026-08-01T13:00:00+00:00", points, (5, 10), raw, outcomes)
    for row in rows2:
        row["sample_weight"] = 1.0
        row.update(s.settings_columns())
    bcc._append_rows(out, rows2)
    assert bcc.check_existing_corpus(out, s.settings_hash) == {
        "2026-08-01T12:00:00+00:00", "2026-08-01T13:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# C1 — hash coverage: the national fitter refuses pre-fix mixed corpora
# ---------------------------------------------------------------------------


def _event_rows(
    tmp_path: Path, settings: "bcc.CorpusSettings", event_iso: str
) -> list[dict]:
    points = bcc.load_points(_write_points(tmp_path, POINTS_V2))
    rows = bcc.build_event_rows(
        event_iso, points, settings.leads_min,
        {p.id: {lead: 0.5 for lead in settings.leads_min} for p in points},
        {p.id: {lead: 1 for lead in settings.leads_min} for p in points},
    )
    for row in rows:
        row["sample_weight"] = 1.0
        row.update(settings.settings_columns())
    return rows


@pytest.mark.parametrize("override", [
    {"scan_type": "doppler"},
    {"timestep_min": 5.0},
    {"motion_method": "farneback_raw"},
])
def test_fitter_refuses_corpus_mixing_scan_type_or_timestep(
    tmp_path: Path, override: dict
):
    """Rows that differ ONLY in scan_type / timestep_min / motion_method hash
    differently,
    so fit_national_calibration's mixed-hash refusal trips on any file
    that concatenates pre-fix and post-fix rows — automatically."""
    import fit_national_calibration as fnc

    out = tmp_path / "mixed.parquet"
    for i, settings in enumerate((_settings(), _settings(**override))):
        bcc._append_rows(
            out, _event_rows(tmp_path, settings, f"2026-08-01T1{i}:00:00+00:00")
        )

    with pytest.raises(fnc.CorpusError, match="mixes settings hashes"):
        fnc.load_v2_corpus(out)

    # The builder equally refuses to resume/append across the change.
    with pytest.raises(ValueError, match="mixed corpus"):
        bcc.check_existing_corpus(out, _settings().settings_hash)


def test_fitter_loads_single_scan_type_corpus(tmp_path: Path):
    """Positive control: a clean fullRange corpus (with the new scan_type
    column) still loads in the read-only national fitter."""
    import fit_national_calibration as fnc

    s = _settings()
    out = tmp_path / "clean.parquet"
    bcc._append_rows(out, _event_rows(tmp_path, s, "2026-08-01T12:00:00+00:00"))

    corpus = fnc.load_v2_corpus(out)
    assert corpus.settings_hash == s.settings_hash
    assert corpus.settings["timestep_min"] == pytest.approx(10.0)
    assert corpus.settings["n_timesteps"] == 8  # 60-min horizon + 18-min age
